"""
AI駆動最適化システム

機械学習を活用した自動最適化・自己調整システム
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import time
import json
import pickle
from pathlib import Path
import multiprocessing as mp

warnings.filterwarnings('ignore')


@dataclass
class OptimizationTarget:
    """最適化対象"""
    name: str
    parameter_space: Dict[str, Union[Tuple[float, float], List[Any]]]
    objective_function: Callable
    maximize: bool = True
    constraints: Optional[List[Callable]] = None


@dataclass
class OptimizationResult:
    """最適化結果"""
    target_name: str
    best_params: Dict[str, Any]
    best_value: float
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_time_seconds: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemState:
    """システム状態"""
    timestamp: datetime
    performance_metrics: Dict[str, float]
    resource_utilization: Dict[str, float]
    configuration_parameters: Dict[str, Any]
    quality_score: float


class BayesianOptimizer:
    """ベイジアン最適化エンジン"""
    
    def __init__(self, n_trials: int = 100, n_jobs: int = 1):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_storage = {}
        self.optimization_history = []
        
    def optimize_target(self, target: OptimizationTarget) -> OptimizationResult:
        """ターゲット最適化実行"""
        start_time = time.time()
        
        try:
            # Optuna study作成
            direction = 'maximize' if target.maximize else 'minimize'
            study = optuna.create_study(
                direction=direction,
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            
            # 目的関数ラッパー
            def objective(trial):
                params = {}
                
                for param_name, param_config in target.parameter_space.items():
                    if isinstance(param_config, tuple) and len(param_config) == 2:
                        # 連続値パラメータ
                        low, high = param_config
                        if isinstance(low, float) or isinstance(high, float):
                            params[param_name] = trial.suggest_float(param_name, low, high)
                        else:
                            params[param_name] = trial.suggest_int(param_name, low, high)
                    elif isinstance(param_config, list):
                        # カテゴリパラメータ
                        params[param_name] = trial.suggest_categorical(param_name, param_config)
                
                # 制約チェック
                if target.constraints:
                    for constraint in target.constraints:
                        if not constraint(params):
                            raise optuna.exceptions.TrialPruned()
                
                # 目的関数評価
                try:
                    value = target.objective_function(params)
                    return value
                except Exception as e:
                    logging.warning(f"目的関数評価エラー: {e}")
                    raise optuna.exceptions.TrialPruned()
            
            # 最適化実行
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
            
            # 結果収集
            best_params = study.best_params
            best_value = study.best_value
            
            # 履歴収集
            history = []
            for trial in study.trials:
                if trial.value is not None:
                    history.append({
                        'trial_number': trial.number,
                        'params': trial.params,
                        'value': trial.value,
                        'state': trial.state.name
                    })
            
            # 収束情報
            convergence_info = {
                'total_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.value is not None]),
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'best_trial_number': study.best_trial.number
            }
            
            execution_time = time.time() - start_time
            
            # 最適化履歴保存
            self.study_storage[target.name] = study
            
            return OptimizationResult(
                target_name=target.name,
                best_params=best_params,
                best_value=best_value,
                optimization_history=history,
                convergence_info=convergence_info,
                execution_time_seconds=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return OptimizationResult(
                target_name=target.name,
                best_params={},
                best_value=0.0,
                optimization_history=[],
                convergence_info={},
                execution_time_seconds=execution_time,
                success=False,
                error_message=str(e)
            )


class ReinforcementLearningOptimizer:
    """強化学習ベース最適化"""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = np.random.uniform(-1, 1, (100, action_dim))  # 簡易Q-table
        self.learning_rate = 0.1
        self.epsilon = 0.1
        self.discount_factor = 0.95
        
        # 経験リプレイ
        self.experience_buffer = deque(maxlen=10000)
        self.state_scaler = MinMaxScaler()
        
    def get_action(self, state: np.ndarray) -> int:
        """行動選択（ε-greedy）"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        # 状態の離散化
        state_index = self._discretize_state(state)
        return np.argmax(self.q_table[state_index])
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Q-table更新"""
        state_index = self._discretize_state(state)
        next_state_index = self._discretize_state(next_state)
        
        # Q学習更新
        current_q = self.q_table[state_index, action]
        max_next_q = np.max(self.q_table[next_state_index])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_index, action] = new_q
        
        # 経験保存
        self.experience_buffer.append((state, action, reward, next_state))
    
    def _discretize_state(self, state: np.ndarray) -> int:
        """状態離散化"""
        # 簡易ハッシュベース離散化
        state_hash = hash(tuple(np.round(state, 2)))
        return abs(state_hash) % len(self.q_table)
    
    def train_from_experience(self, batch_size: int = 32):
        """経験からの学習"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # ランダムサンプリング
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        for idx in batch_indices:
            state, action, reward, next_state = self.experience_buffer[idx]
            self.update_q_table(state, action, reward, next_state)


class MetaLearningOptimizer:
    """メタ学習最適化"""
    
    def __init__(self):
        self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.parameter_history = []
        self.performance_history = []
        self.meta_features = []
        self.trained = False
        
    def add_optimization_experience(self, parameters: Dict[str, Any], 
                                   performance: float, 
                                   meta_features: Dict[str, float]):
        """最適化経験追加"""
        self.parameter_history.append(parameters)
        self.performance_history.append(performance)
        self.meta_features.append(meta_features)
        
        # 定期的にメタモデル更新
        if len(self.parameter_history) >= 20 and len(self.parameter_history) % 10 == 0:
            self._train_meta_model()
    
    def _train_meta_model(self):
        """メタモデル訓練"""
        if len(self.parameter_history) < 10:
            return
        
        try:
            # 特徴量構築
            X = []
            y = []
            
            for params, perf, meta_feat in zip(self.parameter_history, 
                                               self.performance_history, 
                                               self.meta_features):
                # パラメータと メタ特徴量を結合
                feature_vector = []
                
                # パラメータを数値化
                for value in params.values():
                    if isinstance(value, (int, float)):
                        feature_vector.append(value)
                    else:
                        feature_vector.append(hash(str(value)) % 1000)  # 簡易エンコーディング
                
                # メタ特徴量追加
                feature_vector.extend(meta_feat.values())
                
                X.append(feature_vector)
                y.append(perf)
            
            X = np.array(X)
            y = np.array(y)
            
            # メタモデル訓練
            self.meta_model.fit(X, y)
            self.trained = True
            
        except Exception as e:
            logging.error(f"メタモデル訓練エラー: {e}")
    
    def suggest_parameters(self, meta_features: Dict[str, float], 
                          parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータ推奨"""
        if not self.trained or len(self.parameter_history) < 5:
            # ランダム推奨
            return self._random_parameters(parameter_space)
        
        try:
            # 複数候補から最良を選択
            candidates = []
            
            for _ in range(50):  # 50個の候補生成
                candidate = self._random_parameters(parameter_space)
                
                # 特徴量ベクトル構築
                feature_vector = []
                for value in candidate.values():
                    if isinstance(value, (int, float)):
                        feature_vector.append(value)
                    else:
                        feature_vector.append(hash(str(value)) % 1000)
                
                feature_vector.extend(meta_features.values())
                
                # 性能予測
                if len(feature_vector) == len(self.meta_features[0]) + len(self.parameter_history[0]):
                    predicted_performance = self.meta_model.predict([feature_vector])[0]
                    candidates.append((candidate, predicted_performance))
            
            if candidates:
                # 最高予測性能の候補選択
                best_candidate = max(candidates, key=lambda x: x[1])
                return best_candidate[0]
        
        except Exception as e:
            logging.error(f"パラメータ推奨エラー: {e}")
        
        return self._random_parameters(parameter_space)
    
    def _random_parameters(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """ランダムパラメータ生成"""
        params = {}
        
        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, tuple) and len(param_config) == 2:
                low, high = param_config
                if isinstance(low, float) or isinstance(high, float):
                    params[param_name] = np.random.uniform(low, high)
                else:
                    params[param_name] = np.random.randint(low, high + 1)
            elif isinstance(param_config, list):
                params[param_name] = np.random.choice(param_config)
        
        return params


class AIOptimizationSystem:
    """AI駆動最適化システム"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # 最適化エンジン
        self.bayesian_optimizer = BayesianOptimizer(
            n_trials=self.config.get('n_trials', 100),
            n_jobs=self.config.get('n_jobs', 1)
        )
        
        self.rl_optimizer = ReinforcementLearningOptimizer()
        self.meta_optimizer = MetaLearningOptimizer()
        
        # システム状態追跡
        self.system_states: List[SystemState] = []
        self.optimization_targets: Dict[str, OptimizationTarget] = {}
        self.optimization_results: Dict[str, OptimizationResult] = {}
        
        # 自動最適化制御
        self.auto_optimization_active = False
        self.auto_optimization_thread: Optional[threading.Thread] = None
        
        # 統計
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'total_improvement': 0.0,
            'start_time': datetime.now()
        }
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def register_optimization_target(self, target: OptimizationTarget):
        """最適化対象登録"""
        self.optimization_targets[target.name] = target
        self.logger.info(f"最適化対象登録: {target.name}")
    
    def record_system_state(self, performance_metrics: Dict[str, float],
                           resource_utilization: Dict[str, float],
                           configuration_parameters: Dict[str, Any]):
        """システム状態記録"""
        # 品質スコア計算
        quality_score = self._calculate_quality_score(performance_metrics)
        
        state = SystemState(
            timestamp=datetime.now(),
            performance_metrics=performance_metrics,
            resource_utilization=resource_utilization,
            configuration_parameters=configuration_parameters,
            quality_score=quality_score
        )
        
        self.system_states.append(state)
        
        # 履歴サイズ制限
        if len(self.system_states) > 1000:
            self.system_states = self.system_states[-800:]  # 最新800件保持
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """品質スコア計算"""
        # 重み付き品質スコア
        weights = {
            'accuracy': 0.4,
            'processing_speed': 0.3,
            'memory_efficiency': 0.2,
            'stability': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                score += value * weights[metric]
                total_weight += weights[metric]
        
        # 正規化
        return score / total_weight if total_weight > 0 else 0.0
    
    async def run_bayesian_optimization(self, target_name: str) -> OptimizationResult:
        """ベイジアン最適化実行"""
        if target_name not in self.optimization_targets:
            raise ValueError(f"最適化対象が見つかりません: {target_name}")
        
        target = self.optimization_targets[target_name]
        self.logger.info(f"ベイジアン最適化開始: {target_name}")
        
        # 最適化実行
        result = self.bayesian_optimizer.optimize_target(target)
        
        # 結果保存
        self.optimization_results[target_name] = result
        
        # 統計更新
        self.optimization_stats['total_optimizations'] += 1
        if result.success:
            self.optimization_stats['successful_optimizations'] += 1
            
            # 改善度計算（簡易）
            if len(self.system_states) > 0:
                current_quality = self.system_states[-1].quality_score
                improvement = result.best_value - current_quality
                self.optimization_stats['total_improvement'] += max(0, improvement)
        
        self.logger.info(f"ベイジアン最適化完了: {target_name} (成功: {result.success})")
        
        return result
    
    def run_reinforcement_learning_optimization(self, episodes: int = 100):
        """強化学習最適化実行"""
        self.logger.info("強化学習最適化開始")
        
        if len(self.system_states) < 10:
            self.logger.warning("システム状態データが不足しています")
            return
        
        # 環境設定
        state_features = ['accuracy', 'processing_speed', 'memory_usage', 'cpu_usage']
        actions = ['increase_cache', 'decrease_cache', 'parallel_up', 'parallel_down', 'optimize_memory']
        
        for episode in range(episodes):
            # 初期状態
            if len(self.system_states) >= 2:
                current_state_data = self.system_states[-1]
                state_vector = self._extract_state_features(current_state_data, state_features)
                
                # 行動選択
                action = self.rl_optimizer.get_action(state_vector)
                
                # 行動実行（シミュレーション）
                reward = self._simulate_action_reward(action, current_state_data)
                
                # 次状態（簡易シミュレーション）
                next_state_vector = state_vector + np.random.normal(0, 0.1, len(state_vector))
                next_state_vector = np.clip(next_state_vector, 0, 1)
                
                # Q学習更新
                self.rl_optimizer.update_q_table(state_vector, action, reward, next_state_vector)
        
        # 経験からの学習
        self.rl_optimizer.train_from_experience()
        
        self.logger.info("強化学習最適化完了")
    
    def _extract_state_features(self, state: SystemState, feature_names: List[str]) -> np.ndarray:
        """状態特徴量抽出"""
        features = []
        
        for feature_name in feature_names:
            if feature_name in state.performance_metrics:
                features.append(state.performance_metrics[feature_name])
            elif feature_name in state.resource_utilization:
                features.append(state.resource_utilization[feature_name])
            else:
                features.append(0.0)  # デフォルト値
        
        return np.array(features)
    
    def _simulate_action_reward(self, action: int, state: SystemState) -> float:
        """行動報酬シミュレーション"""
        # 簡易報酬関数
        base_reward = state.quality_score
        
        # 行動に基づく報酬調整
        action_rewards = {
            0: 0.1,   # increase_cache
            1: -0.05, # decrease_cache
            2: 0.15,  # parallel_up
            3: -0.1,  # parallel_down
            4: 0.2    # optimize_memory
        }
        
        action_bonus = action_rewards.get(action, 0.0)
        
        return base_reward + action_bonus + np.random.normal(0, 0.05)
    
    async def run_meta_learning_optimization(self, target_name: str) -> Dict[str, Any]:
        """メタ学習最適化実行"""
        if target_name not in self.optimization_targets:
            raise ValueError(f"最適化対象が見つかりません: {target_name}")
        
        target = self.optimization_targets[target_name]
        self.logger.info(f"メタ学習最適化開始: {target_name}")
        
        # メタ特徴量抽出
        meta_features = self._extract_meta_features()
        
        # パラメータ推奨
        suggested_params = self.meta_optimizer.suggest_parameters(
            meta_features, target.parameter_space
        )
        
        # 推奨パラメータで評価
        try:
            performance = target.objective_function(suggested_params)
            
            # 経験をメタ学習器に追加
            self.meta_optimizer.add_optimization_experience(
                suggested_params, performance, meta_features
            )
            
            self.logger.info(f"メタ学習最適化完了: {target_name} (性能: {performance:.3f})")
            
            return {
                'suggested_params': suggested_params,
                'performance': performance,
                'meta_features': meta_features,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"メタ学習最適化エラー: {e}")
            return {
                'suggested_params': suggested_params,
                'performance': 0.0,
                'meta_features': meta_features,
                'success': False,
                'error': str(e)
            }
    
    def _extract_meta_features(self) -> Dict[str, float]:
        """メタ特徴量抽出"""
        if not self.system_states:
            return {'avg_quality': 0.5, 'stability': 0.5, 'trend': 0.0}
        
        recent_states = self.system_states[-50:]  # 最新50状態
        
        # 平均品質
        avg_quality = np.mean([s.quality_score for s in recent_states])
        
        # 安定性（品質スコアの標準偏差の逆数）
        quality_std = np.std([s.quality_score for s in recent_states])
        stability = 1.0 / (1.0 + quality_std)
        
        # トレンド（最初と最後の品質差）
        if len(recent_states) >= 2:
            trend = recent_states[-1].quality_score - recent_states[0].quality_score
        else:
            trend = 0.0
        
        return {
            'avg_quality': avg_quality,
            'stability': stability,
            'trend': trend
        }
    
    async def start_auto_optimization(self, interval_minutes: int = 60):
        """自動最適化開始"""
        if self.auto_optimization_active:
            return
        
        self.auto_optimization_active = True
        self.auto_optimization_thread = threading.Thread(
            target=self._auto_optimization_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.auto_optimization_thread.start()
        
        self.logger.info(f"自動最適化開始 (間隔: {interval_minutes}分)")
    
    def stop_auto_optimization(self):
        """自動最適化停止"""
        self.auto_optimization_active = False
        if self.auto_optimization_thread:
            self.auto_optimization_thread.join(timeout=10)
        
        self.logger.info("自動最適化停止")
    
    def _auto_optimization_loop(self, interval_minutes: int):
        """自動最適化ループ"""
        while self.auto_optimization_active:
            try:
                # 最適化実行
                for target_name in self.optimization_targets:
                    if not self.auto_optimization_active:
                        break
                    
                    # メタ学習最適化実行
                    asyncio.run(self.run_meta_learning_optimization(target_name))
                
                # 強化学習最適化（軽量）
                if self.auto_optimization_active:
                    self.run_reinforcement_learning_optimization(episodes=20)
                
                # 待機
                sleep_seconds = interval_minutes * 60
                start_time = time.time()
                
                while (time.time() - start_time < sleep_seconds and 
                       self.auto_optimization_active):
                    time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"自動最適化エラー: {e}")
                time.sleep(60)  # エラー時は1分待機
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化サマリー取得"""
        uptime = (datetime.now() - self.optimization_stats['start_time']).seconds
        
        # 最新の品質スコア
        current_quality = self.system_states[-1].quality_score if self.system_states else 0.0
        
        # 平均品質（最近の50状態）
        recent_states = self.system_states[-50:] if self.system_states else []
        avg_quality = np.mean([s.quality_score for s in recent_states]) if recent_states else 0.0
        
        # 成功率
        success_rate = (self.optimization_stats['successful_optimizations'] / 
                       max(1, self.optimization_stats['total_optimizations']) * 100)
        
        return {
            'uptime_seconds': uptime,
            'auto_optimization_active': self.auto_optimization_active,
            'total_optimizations': self.optimization_stats['total_optimizations'],
            'successful_optimizations': self.optimization_stats['successful_optimizations'],
            'success_rate_percent': success_rate,
            'total_improvement': self.optimization_stats['total_improvement'],
            'current_quality_score': current_quality,
            'average_quality_score': avg_quality,
            'registered_targets': len(self.optimization_targets),
            'system_states_count': len(self.system_states),
            'meta_learning_trained': self.meta_optimizer.trained
        }


# 使用例の目的関数
def sample_objective_function(params: Dict[str, Any]) -> float:
    """サンプル目的関数"""
    # パラメータに基づく性能シミュレーション
    cache_size = params.get('cache_size', 1000)
    max_workers = params.get('max_workers', 4)
    learning_rate = params.get('learning_rate', 0.01)
    
    # 複雑な性能関数をシミュレート
    performance = (
        np.log(cache_size) / 10.0 +
        np.sqrt(max_workers) / 5.0 +
        1.0 / (1.0 + abs(learning_rate - 0.1)) +
        np.random.normal(0, 0.1)
    )
    
    return max(0, performance)


async def demo_ai_optimization_system():
    """AI最適化システムデモ"""
    print("=== AI駆動最適化システム デモ ===")
    
    # システム初期化
    ai_optimizer = AIOptimizationSystem(config={'n_trials': 50, 'n_jobs': 1})
    
    try:
        # 最適化対象登録
        print("\n1. 最適化対象登録...")
        
        target = OptimizationTarget(
            name="system_performance",
            parameter_space={
                'cache_size': (100, 5000),
                'max_workers': (1, 16),
                'learning_rate': (0.001, 0.1),
                'batch_size': [32, 64, 128, 256]
            },
            objective_function=sample_objective_function,
            maximize=True
        )
        
        ai_optimizer.register_optimization_target(target)
        
        # システム状態シミュレーション
        print("2. システム状態シミュレーション...")
        
        for i in range(20):
            performance_metrics = {
                'accuracy': np.random.uniform(0.5, 0.9),
                'processing_speed': np.random.uniform(100, 1000),
                'memory_efficiency': np.random.uniform(0.3, 0.9)
            }
            
            resource_utilization = {
                'cpu_usage': np.random.uniform(20, 80),
                'memory_usage': np.random.uniform(30, 90)
            }
            
            config_params = {
                'current_cache_size': np.random.randint(500, 2000),
                'current_workers': np.random.randint(2, 8)
            }
            
            ai_optimizer.record_system_state(
                performance_metrics, resource_utilization, config_params
            )
        
        # ベイジアン最適化実行
        print("3. ベイジアン最適化実行...")
        bayesian_result = await ai_optimizer.run_bayesian_optimization("system_performance")
        
        print(f"=== ベイジアン最適化結果 ===")
        print(f"成功: {bayesian_result.success}")
        if bayesian_result.success:
            print(f"最適パラメータ: {bayesian_result.best_params}")
            print(f"最適値: {bayesian_result.best_value:.3f}")
            print(f"実行時間: {bayesian_result.execution_time_seconds:.2f}秒")
            print(f"試行回数: {bayesian_result.convergence_info['completed_trials']}")
        
        # メタ学習最適化実行
        print("\n4. メタ学習最適化実行...")
        meta_result = await ai_optimizer.run_meta_learning_optimization("system_performance")
        
        print(f"=== メタ学習最適化結果 ===")
        print(f"成功: {meta_result['success']}")
        if meta_result['success']:
            print(f"推奨パラメータ: {meta_result['suggested_params']}")
            print(f"予測性能: {meta_result['performance']:.3f}")
        
        # 強化学習最適化実行
        print("\n5. 強化学習最適化実行...")
        ai_optimizer.run_reinforcement_learning_optimization(episodes=50)
        
        # システムサマリー
        print("\n6. システムサマリー...")
        summary = ai_optimizer.get_optimization_summary()
        
        print(f"=== AI最適化システムサマリー ===")
        print(f"総最適化回数: {summary['total_optimizations']}")
        print(f"成功回数: {summary['successful_optimizations']}")
        print(f"成功率: {summary['success_rate_percent']:.1f}%")
        print(f"総改善度: {summary['total_improvement']:.3f}")
        print(f"現在の品質スコア: {summary['current_quality_score']:.3f}")
        print(f"平均品質スコア: {summary['average_quality_score']:.3f}")
        print(f"登録対象数: {summary['registered_targets']}")
        print(f"システム状態数: {summary['system_states_count']}")
        print(f"メタ学習訓練済み: {'はい' if summary['meta_learning_trained'] else 'いいえ'}")
        
        print(f"✅ AI駆動最適化システム完了")
        
        return summary
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    asyncio.run(demo_ai_optimization_system())