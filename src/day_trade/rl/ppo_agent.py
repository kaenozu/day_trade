#!/usr/bin/env python3
"""
Next-Gen AI Trading PPO Agent
Proximal Policy Optimization (PPO) 強化学習エージェント実装

多資産ポートフォリオ管理・リスク調整・継続学習対応
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 深層学習フレームワーク
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# 強化学習ライブラリ（軽量代替実装）
try:
    import gym
    from gym import spaces

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

    # 軽量な代替実装
    class spaces:
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype


# オプショナルライブラリ
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from ..utils.logging_config import get_context_logger
from .trading_environment import (
    MultiAssetTradingEnvironment,
    create_trading_environment,
)

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class PPOConfig:
    """PPO エージェント設定"""

    # ネットワーク構造
    hidden_dim: int = 256
    num_layers: int = 3
    activation: str = "tanh"

    # PPO ハイパーパラメータ
    learning_rate: float = 3e-4
    gamma: float = 0.99  # 割引率
    lambda_gae: float = 0.95  # GAE パラメータ
    epsilon_clip: float = 0.2  # クリッピング閾値
    epochs_per_update: int = 10
    batch_size: int = 64

    # 価値関数とポリシーの重み
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # 学習設定
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    update_frequency: int = 2048  # ステップ数

    # 探索設定
    exploration_noise: float = 0.1
    noise_decay: float = 0.995
    min_noise: float = 0.01

    # GPU設定
    use_gpu: bool = True
    mixed_precision: bool = False

    # モニタリング
    log_frequency: int = 10
    save_frequency: int = 100
    eval_frequency: int = 50


@dataclass
class PPOExperience:
    """PPO経験データ"""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    dones: np.ndarray
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None


if PYTORCH_AVAILABLE:

    class ActorCriticNetwork(nn.Module):
        """Actor-Critic ネットワーク"""

        def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch が必要です")
else:

    class ActorCriticNetwork:
        """Actor-Critic ネットワーク（フォールバック）"""

        def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
            raise ImportError("PyTorch が必要です")

        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 共有特徴抽出レイヤー
        self.shared_layers = self._build_shared_network()

        # Actor ネットワーク (ポリシー)
        self.actor_mean = nn.Linear(config.hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic ネットワーク (価値関数)
        self.critic = nn.Linear(config.hidden_dim, 1)

        # 初期化
        self._initialize_weights()

    def _build_shared_network(self) -> nn.Module:
        """共有ネットワーク構築"""
        layers = []
        input_dim = self.state_dim

        for _ in range(self.config.num_layers):
            layers.append(nn.Linear(input_dim, self.config.hidden_dim))

            if self.config.activation == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.config.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))

            layers.append(nn.Dropout(0.1))
            input_dim = self.config.hidden_dim

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """順伝播"""
        # 共有特徴抽出
        features = self.shared_layers(state)

        # Actor (ポリシー)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std.clamp(-20, 2))

        # Critic (価値関数)
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ):
        """アクションと価値を取得"""
        action_mean, action_std, value = self.forward(state)

        # 正規分布ポリシー
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)

        return action, log_prob, entropy, value


class PPOAgent:
    """PPO強化学習エージェント"""

    def __init__(
        self, env: MultiAssetTradingEnvironment, config: Optional[PPOConfig] = None
    ):
        self.env = env
        self.config = config or PPOConfig()

        # 環境情報
        env_info = env.get_environment_info()
        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]

        # デバイス設定
        if PYTORCH_AVAILABLE and self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"GPU使用: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            logger.info("CPU使用")

        # ネットワーク初期化
        if PYTORCH_AVAILABLE:
            self.actor_critic = ActorCriticNetwork(
                self.state_dim, self.action_dim, self.config
            ).to(self.device)

            self.optimizer = torch.optim.Adam(
                self.actor_critic.parameters(), lr=self.config.learning_rate
            )
        else:
            raise ImportError("PyTorch が必要です")

        # 学習統計
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0
        self.best_reward = float("-inf")

        # 経験バッファ
        self.experience_buffer = []
        self.training_history = []

        # 探索ノイズ
        self.current_noise = self.config.exploration_noise

        # ログ設定
        self.setup_logging()

        logger.info("PPO Agent 初期化完了")
        logger.info(f"状態次元: {self.state_dim}, アクション次元: {self.action_dim}")

    def setup_logging(self):
        """ログ設定"""
        self.tensorboard_writer = None

        if TENSORBOARD_AVAILABLE:
            log_dir = Path("logs/ppo_training")
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir)

        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="next-gen-ai-trading",
                    name="ppo-agent",
                    config=self.config.__dict__,
                )
            except Exception as e:
                logger.warning(f"W&B初期化失敗: {e}")

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """アクション選択"""
        self.actor_critic.eval() if not training else self.actor_critic.train()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, entropy, value = self.actor_critic.get_action_and_value(
                state_tensor
            )

            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().item()
            value = value.cpu().item()

        # 探索ノイズ追加（学習時のみ）
        if training and self.current_noise > 0:
            noise = np.random.normal(0, self.current_noise, action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        return action, log_prob, value

    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """経験をバッファに格納"""
        self.experience_buffer.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "log_prob": log_prob,
                "value": value,
            }
        )

    def compute_advantages(self, experiences: List[Dict]) -> PPOExperience:
        """Advantage計算（GAE）"""
        states = np.array([exp["state"] for exp in experiences])
        actions = np.array([exp["action"] for exp in experiences])
        rewards = np.array([exp["reward"] for exp in experiences])
        values = np.array([exp["value"] for exp in experiences])
        log_probs = np.array([exp["log_prob"] for exp in experiences])
        dones = np.array([exp["done"] for exp in experiences])

        # 最終状態の価値推定
        with torch.no_grad():
            next_state = experiences[-1]["next_state"]
            if not experiences[-1]["done"]:
                next_state_tensor = (
                    torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                )
                _, _, _, next_value = self.actor_critic.get_action_and_value(
                    next_state_tensor
                )
                next_value = next_value.cpu().item()
            else:
                next_value = 0.0

        # GAE計算
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]

            delta = (
                rewards[t]
                + self.config.gamma * next_value_t * next_non_terminal
                - values[t]
            )
            gae = (
                delta
                + self.config.gamma * self.config.lambda_gae * next_non_terminal * gae
            )
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # 正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return PPOExperience(
            states=states,
            actions=actions,
            rewards=rewards,
            values=values,
            log_probs=log_probs,
            dones=dones,
            advantages=advantages,
            returns=returns,
        )

    def update_policy(self) -> Dict[str, float]:
        """ポリシー更新"""
        if len(self.experience_buffer) < self.config.update_frequency:
            return {}

        # 経験からAdvantage計算
        experience = self.compute_advantages(self.experience_buffer)

        # PyTorchテンソルに変換
        states = torch.FloatTensor(experience.states).to(self.device)
        actions = torch.FloatTensor(experience.actions).to(self.device)
        old_log_probs = torch.FloatTensor(experience.log_probs).to(self.device)
        advantages = torch.FloatTensor(experience.advantages).to(self.device)
        returns = torch.FloatTensor(experience.returns).to(self.device)
        old_values = torch.FloatTensor(experience.values).to(self.device)

        # 複数エポック更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for epoch in range(self.config.epochs_per_update):
            # ミニバッチ作成
            batch_size = self.config.batch_size
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 現在のポリシーでの評価
                (
                    _,
                    new_log_probs,
                    entropy,
                    new_values,
                ) = self.actor_critic.get_action_and_value(batch_states, batch_actions)

                # ポリシー損失（PPOクリッピング）
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.epsilon_clip,
                        1 + self.config.epsilon_clip,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 価値関数損失
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)

                # エントロピー損失（探索促進）
                entropy_loss = -entropy.mean()

                # 総損失
                total_loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # 勾配更新
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # 損失蓄積
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()

        # 経験バッファクリア
        self.experience_buffer.clear()

        # 探索ノイズ減衰
        self.current_noise = max(
            self.current_noise * self.config.noise_decay, self.config.min_noise
        )

        # 損失統計
        num_updates = self.config.epochs_per_update * (
            len(states) // self.config.batch_size
        )
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "exploration_noise": self.current_noise,
        }

    def train(self, max_episodes: Optional[int] = None) -> Dict[str, Any]:
        """エージェント訓練"""
        max_episodes = max_episodes or self.config.max_episodes

        logger.info(f"PPO訓練開始: {max_episodes} エピソード")
        start_time = time.time()

        episode_rewards = []
        episode_lengths = []

        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(self.config.max_steps_per_episode):
                # アクション選択
                action, log_prob, value = self.select_action(state, training=True)

                # 環境ステップ
                next_state, reward, done, info = self.env.step(action)

                # 経験格納
                self.store_experience(
                    state, action, reward, next_state, done, log_prob, value
                )

                # 更新
                state = next_state
                episode_reward += reward
                episode_length += 1
                self.step_count += 1

                # ポリシー更新
                if len(self.experience_buffer) >= self.config.update_frequency:
                    loss_info = self.update_policy()

                    # ログ記録
                    if loss_info and self.tensorboard_writer:
                        for key, value in loss_info.items():
                            self.tensorboard_writer.add_scalar(
                                f"Training/{key}", value, self.step_count
                            )

                if done:
                    break

            # エピソード統計
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            self.episode_count += 1
            self.total_reward += episode_reward

            # ベストスコア更新
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model(f"best_model_episode_{episode}.pth")

            # ログ出力
            if episode % self.config.log_frequency == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])

                logger.info(
                    f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Avg100={avg_reward:.2f}, Length={episode_length}, "
                    f"Noise={self.current_noise:.3f}"
                )

                # TensorBoard記録
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(
                        "Episode/Reward", episode_reward, episode
                    )
                    self.tensorboard_writer.add_scalar(
                        "Episode/Length", episode_length, episode
                    )
                    self.tensorboard_writer.add_scalar(
                        "Episode/AvgReward100", avg_reward, episode
                    )

                # W&B記録
                if WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "episode": episode,
                            "episode_reward": episode_reward,
                            "avg_reward_100": avg_reward,
                            "episode_length": episode_length,
                            "exploration_noise": self.current_noise,
                        }
                    )

            # モデル保存
            if episode > 0 and episode % self.config.save_frequency == 0:
                self.save_model(f"model_episode_{episode}.pth")

        training_time = time.time() - start_time

        # 訓練結果
        result = {
            "episodes_trained": max_episodes,
            "total_steps": self.step_count,
            "training_time": training_time,
            "final_avg_reward": np.mean(episode_rewards[-100:]),
            "best_reward": self.best_reward,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }

        self.training_history.append(result)
        logger.info(
            f"PPO訓練完了: {training_time:.2f}秒, ベストスコア: {self.best_reward:.2f}"
        )

        return result

    def evaluate(
        self, num_episodes: int = 10, render: bool = False
    ) -> Dict[str, float]:
        """エージェント評価"""
        logger.info(f"PPO評価開始: {num_episodes} エピソード")

        self.actor_critic.eval()
        episode_rewards = []
        episode_lengths = []
        portfolio_returns = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(self.config.max_steps_per_episode):
                # 決定論的アクション選択
                action, _, _ = self.select_action(state, training=False)

                # 環境ステップ
                next_state, reward, done, info = self.env.step(action)

                state = next_state
                episode_reward += reward
                episode_length += 1

                if render:
                    self.env.render()

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # ポートフォリオリターン記録
            if "portfolio_return" in info:
                portfolio_returns.append(info["portfolio_return"])

        # 評価結果
        evaluation_result = {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "avg_length": np.mean(episode_lengths),
            "avg_portfolio_return": np.mean(portfolio_returns)
            if portfolio_returns
            else 0,
            "win_rate": np.mean([r > 0 for r in episode_rewards]),
        }

        logger.info(
            f"評価完了: 平均報酬={evaluation_result['avg_reward']:.2f}, "
            f"勝率={evaluation_result['win_rate']*100:.1f}%"
        )

        return evaluation_result

    def save_model(self, filename: str):
        """モデル保存"""
        save_path = Path("models") / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "best_reward": self.best_reward,
            "training_history": self.training_history,
        }

        torch.save(checkpoint, save_path)
        logger.info(f"モデル保存完了: {save_path}")

    def load_model(self, filename: str):
        """モデル読み込み"""
        load_path = Path("models") / filename

        if not load_path.exists():
            raise FileNotFoundError(f"モデルファイル未発見: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint.get("episode_count", 0)
        self.step_count = checkpoint.get("step_count", 0)
        self.best_reward = checkpoint.get("best_reward", float("-inf"))
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"モデル読み込み完了: {load_path}")

    def get_agent_info(self) -> Dict[str, Any]:
        """エージェント情報取得"""
        return {
            "agent_type": "PPO",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "config": self.config.__dict__,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "best_reward": self.best_reward,
            "current_noise": self.current_noise,
            "device": str(self.device),
            "total_parameters": sum(p.numel() for p in self.actor_critic.parameters()),
        }


# 便利関数
def create_ppo_agent(
    symbols: List[str] = None, env_config: Dict = None, agent_config: Dict = None
) -> PPOAgent:
    """PPO エージェント作成"""

    # 環境作成
    env_config = env_config or {}
    env = create_trading_environment(symbols=symbols, **env_config)

    # エージェント設定
    if agent_config:
        config = PPOConfig(**agent_config)
    else:
        config = PPOConfig()

    # エージェント作成
    agent = PPOAgent(env, config)

    logger.info(
        f"PPO Agent 作成完了: {len(env.symbols)} 資産, パラメータ数: {agent.get_agent_info()['total_parameters']:,}"
    )

    return agent


def train_ppo_agent(
    symbols: List[str] = None, max_episodes: int = 1000, **kwargs
) -> Tuple[PPOAgent, Dict[str, Any]]:
    """PPO エージェント訓練"""

    agent = create_ppo_agent(symbols=symbols, **kwargs)
    training_result = agent.train(max_episodes=max_episodes)

    return agent, training_result


if __name__ == "__main__":
    # PPO エージェントテスト
    print("=== Next-Gen AI Trading PPO Agent テスト ===")

    test_symbols = ["STOCK_A", "STOCK_B", "STOCK_C"]

    # エージェント作成
    agent = create_ppo_agent(
        symbols=test_symbols,
        env_config={"initial_balance": 1000000, "max_steps": 200},
        agent_config={"max_episodes": 50, "log_frequency": 5, "learning_rate": 1e-3},
    )

    # 基本情報表示
    info = agent.get_agent_info()
    print("エージェント情報:")
    print(f"  状態次元: {info['state_dim']}")
    print(f"  アクション次元: {info['action_dim']}")
    print(f"  パラメータ数: {info['total_parameters']:,}")
    print(f"  デバイス: {info['device']}")

    # 短時間訓練
    print(f"\n訓練開始: {test_symbols}")
    training_result = agent.train(max_episodes=20)

    print("\n訓練結果:")
    print(f"  訓練時間: {training_result['training_time']:.2f}秒")
    print(f"  ベスト報酬: {training_result['best_reward']:.2f}")
    print(f"  最終平均報酬: {training_result['final_avg_reward']:.2f}")

    # 評価実行
    print("\n評価実行:")
    eval_result = agent.evaluate(num_episodes=5)

    print("評価結果:")
    print(f"  平均報酬: {eval_result['avg_reward']:.2f}")
    print(f"  勝率: {eval_result['win_rate']*100:.1f}%")
    print(
        f"  平均ポートフォリオリターン: {eval_result['avg_portfolio_return']*100:.2f}%"
    )

    print("\n=== テスト完了 ===")
