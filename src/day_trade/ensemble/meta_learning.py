#!/usr/bin/env python3
"""
メタ学習フレームワーク
Meta-Learning Framework

Issue #762: 高度なアンサンブル予測システムの強化 - Phase 2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import asyncio
import logging
import time
import copy
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import pickle
import warnings

# scikit-learn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ログ設定
logger = logging.getLogger(__name__)

@dataclass
class Task:
    """学習タスク"""
    task_id: str
    support_set: Tuple[np.ndarray, np.ndarray]  # (X, y)
    query_set: Tuple[np.ndarray, np.ndarray]    # (X, y)
    metadata: Dict[str, Any] = field(default_factory=dict)
    domain: str = "financial"
    difficulty: float = 1.0

@dataclass
class EpisodeResult:
    """エピソード結果"""
    task_id: str
    inner_loss: float
    outer_loss: float
    adaptation_steps: int
    performance_gain: float
    timestamp: float
    adapted_parameters: Optional[Dict[str, torch.Tensor]] = None

@dataclass
class KnowledgeUnit:
    """知識単位"""
    source_task: str
    target_task: str
    knowledge_type: str  # "features", "parameters", "gradients"
    knowledge_data: Any
    relevance_score: float
    transfer_success: bool
    timestamp: float

class MetaModelBase(nn.Module, ABC):
    """メタモデル基底クラス"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_fast_weights(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def set_fast_weights(self, fast_weights: Dict[str, torch.Tensor]) -> None:
        pass

class SimpleMetaModel(MetaModelBase):
    """シンプルなメタモデル"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__(input_dim, hidden_dim, output_dim)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Fast weights for adaptation
        self.fast_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fast_weights is not None:
            return self._forward_with_fast_weights(x)
        return self.layers(x)

    def _forward_with_fast_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Fast weights使用の順伝播"""
        h = x
        layer_idx = 0

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weight_key = f"layers.{layer_idx}.weight"
                bias_key = f"layers.{layer_idx}.bias"

                if weight_key in self.fast_weights:
                    weight = self.fast_weights[weight_key]
                    bias = self.fast_weights[bias_key]
                    h = F.linear(h, weight, bias)
                else:
                    h = layer(h)
                layer_idx += 2  # Linear + ReLU
            else:
                h = layer(h)

        return h

    def get_fast_weights(self) -> Dict[str, torch.Tensor]:
        """Fast weights取得"""
        if self.fast_weights is None:
            return {name: param.clone() for name, param in self.named_parameters()}
        return self.fast_weights.copy()

    def set_fast_weights(self, fast_weights: Dict[str, torch.Tensor]) -> None:
        """Fast weights設定"""
        self.fast_weights = fast_weights

class AttentionMetaModel(MetaModelBase):
    """Attention機構付きメタモデル"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, num_heads: int = 4):
        super().__init__(input_dim, hidden_dim, output_dim)

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.fast_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding
        h = self.embedding(x)

        if len(h.shape) == 2:
            h = h.unsqueeze(0)  # Add batch dimension

        # Self-attention
        attn_out, _ = self.attention(h, h, h)
        h = self.norm1(h + attn_out)

        # Feed-forward
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)

        # Output
        output = self.output_layer(h)

        if output.shape[0] == 1:
            output = output.squeeze(0)

        return output

    def get_fast_weights(self) -> Dict[str, torch.Tensor]:
        return {name: param.clone() for name, param in self.named_parameters()}

    def set_fast_weights(self, fast_weights: Dict[str, torch.Tensor]) -> None:
        self.fast_weights = fast_weights

class ModelKnowledgeTransfer:
    """モデル知識転移システム"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.knowledge_base: Dict[str, List[KnowledgeUnit]] = defaultdict(list)
        self.task_embeddings: Dict[str, np.ndarray] = {}
        self.scaler = StandardScaler()

        logger.info("ModelKnowledgeTransfer initialized")

    def _extract_task_features(self,
                             support_set: Tuple[np.ndarray, np.ndarray],
                             metadata: Dict[str, Any]) -> np.ndarray:
        """タスク特徴量抽出"""
        X, y = support_set

        features = []

        # データ統計
        features.extend([
            X.shape[0],  # サンプル数
            X.shape[1],  # 特徴数
            np.mean(X),  # 特徴平均
            np.std(X),   # 特徴標準偏差
            np.mean(y),  # ターゲット平均
            np.std(y),   # ターゲット標準偏差
        ])

        # 相関統計
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            features.extend([
                np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),  # 平均相関
                np.std(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])   # 相関標準偏差
            ])
        else:
            features.extend([0.0, 0.0])

        # X-y相関
        xy_corrs = [np.corrcoef(X[:, i], y.flatten())[0, 1] for i in range(X.shape[1])]
        xy_corrs = [c for c in xy_corrs if not np.isnan(c)]

        if xy_corrs:
            features.extend([
                np.mean(xy_corrs),
                np.std(xy_corrs),
                np.max(np.abs(xy_corrs))
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # メタデータ特徴
        features.append(metadata.get('difficulty', 1.0))
        features.append(hash(metadata.get('domain', 'default')) % 1000 / 1000.0)

        return np.array(features)

    def calculate_task_similarity(self, task1_id: str, task2_id: str) -> float:
        """タスク類似度計算"""
        if task1_id not in self.task_embeddings or task2_id not in self.task_embeddings:
            return 0.0

        emb1 = self.task_embeddings[task1_id]
        emb2 = self.task_embeddings[task2_id]

        # コサイン類似度
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)

    def store_knowledge(self,
                       source_task: str,
                       knowledge_type: str,
                       knowledge_data: Any,
                       target_task: Optional[str] = None) -> None:
        """知識保存"""
        knowledge_unit = KnowledgeUnit(
            source_task=source_task,
            target_task=target_task or "general",
            knowledge_type=knowledge_type,
            knowledge_data=knowledge_data,
            relevance_score=1.0,
            transfer_success=True,
            timestamp=time.time()
        )

        self.knowledge_base[source_task].append(knowledge_unit)

        # 知識ベース上限管理
        if len(self.knowledge_base[source_task]) > 50:
            self.knowledge_base[source_task] = self.knowledge_base[source_task][-25:]

    def retrieve_relevant_knowledge(self,
                                  target_task: str,
                                  knowledge_type: Optional[str] = None,
                                  top_k: int = 5) -> List[KnowledgeUnit]:
        """関連知識検索"""
        if target_task not in self.task_embeddings:
            return []

        relevant_knowledge = []

        for source_task, knowledge_list in self.knowledge_base.items():
            if source_task == target_task:
                continue

            similarity = self.calculate_task_similarity(source_task, target_task)

            if similarity >= self.similarity_threshold:
                for ku in knowledge_list:
                    if knowledge_type is None or ku.knowledge_type == knowledge_type:
                        ku_copy = copy.deepcopy(ku)
                        ku_copy.relevance_score = similarity
                        relevant_knowledge.append(ku_copy)

        # 関連度順ソート
        relevant_knowledge.sort(key=lambda x: x.relevance_score, reverse=True)

        return relevant_knowledge[:top_k]

    def register_task(self, task: Task) -> None:
        """タスク登録"""
        features = self._extract_task_features(task.support_set, task.metadata)
        self.task_embeddings[task.task_id] = features

        logger.debug(f"Task {task.task_id} registered with {len(features)} features")

class FewShotLearning:
    """少数ショット学習"""

    def __init__(self,
                 prototype_dim: int = 64,
                 distance_metric: str = "cosine"):
        self.prototype_dim = prototype_dim
        self.distance_metric = distance_metric
        self.prototypes: Dict[str, torch.Tensor] = {}
        self.prototype_network = self._build_prototype_network()

        logger.info("FewShotLearning initialized")

    def _build_prototype_network(self) -> nn.Module:
        """プロトタイプネットワーク構築"""
        return nn.Sequential(
            nn.Linear(1, 32),  # 入力次元は後で調整
            nn.ReLU(),
            nn.Linear(32, self.prototype_dim),
            nn.ReLU(),
            nn.Linear(self.prototype_dim, self.prototype_dim)
        )

    def _compute_prototypes(self,
                          support_set: Tuple[np.ndarray, np.ndarray]) -> Dict[str, torch.Tensor]:
        """プロトタイプ計算"""
        X_support, y_support = support_set

        # データをテンソルに変換
        X_tensor = torch.FloatTensor(X_support)
        y_tensor = torch.FloatTensor(y_support).flatten()

        # 入力次元調整
        if X_tensor.shape[1] != self.prototype_network[0].in_features:
            self.prototype_network[0] = nn.Linear(X_tensor.shape[1], 32)

        # 特徴抽出
        features = self.prototype_network(X_tensor)

        # 回帰問題の場合、連続値を離散化してプロトタイプ作成
        # y値を3つの範囲に分割
        y_min, y_max = torch.min(y_tensor), torch.max(y_tensor)
        if y_max - y_min > 1e-6:
            # 低・中・高の3クラス
            thresholds = [
                y_min + (y_max - y_min) * 0.33,
                y_min + (y_max - y_min) * 0.67
            ]

            class_assignments = torch.zeros_like(y_tensor, dtype=torch.long)
            class_assignments[y_tensor >= thresholds[1]] = 2  # High
            class_assignments[(y_tensor >= thresholds[0]) & (y_tensor < thresholds[1])] = 1  # Mid
            # Low = 0 (default)
        else:
            # 全て同じ値の場合
            class_assignments = torch.zeros_like(y_tensor, dtype=torch.long)

        # クラス別プロトタイプ計算
        prototypes = {}
        for class_id in torch.unique(class_assignments):
            class_mask = class_assignments == class_id
            if torch.sum(class_mask) > 0:
                class_features = features[class_mask]
                prototype = torch.mean(class_features, dim=0)
                prototypes[f"class_{class_id.item()}"] = prototype

        return prototypes

    def _compute_distance(self, query_features: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        """距離計算"""
        if self.distance_metric == "euclidean":
            return torch.norm(query_features - prototype, dim=1)
        elif self.distance_metric == "cosine":
            query_norm = F.normalize(query_features, p=2, dim=1)
            prototype_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1)
            return 1 - torch.mm(query_norm, prototype_norm.t()).squeeze()
        else:
            # Default to euclidean
            return torch.norm(query_features - prototype, dim=1)

    async def few_shot_predict(self,
                             support_set: Tuple[np.ndarray, np.ndarray],
                             query_set: np.ndarray) -> np.ndarray:
        """少数ショット予測"""
        try:
            # プロトタイプ計算
            prototypes = self._compute_prototypes(support_set)

            if not prototypes:
                # フォールバック: 平均値予測
                _, y_support = support_set
                mean_value = np.mean(y_support)
                return np.full((query_set.shape[0], 1), mean_value)

            # クエリ特徴抽出
            X_query = torch.FloatTensor(query_set)
            query_features = self.prototype_network(X_query)

            # 最近傍プロトタイプ特定と予測
            predictions = []

            for i in range(query_features.shape[0]):
                query_feature = query_features[i].unsqueeze(0)

                min_distance = float('inf')
                best_class = None

                for class_name, prototype in prototypes.items():
                    distance = self._compute_distance(query_feature, prototype)
                    if distance < min_distance:
                        min_distance = distance
                        best_class = class_name

                # クラスから予測値への変換
                if best_class:
                    class_id = int(best_class.split('_')[1])
                    # クラスIDを連続値に変換（簡単な線形マッピング）
                    pred_value = class_id * 0.5  # 調整可能
                else:
                    pred_value = 0.0

                predictions.append(pred_value)

            return np.array(predictions).reshape(-1, 1)

        except Exception as e:
            logger.error(f"Error in few-shot prediction: {e}")
            # フォールバック
            return np.zeros((query_set.shape[0], 1))

class ContinualLearning:
    """継続学習フレームワーク"""

    def __init__(self,
                 memory_size: int = 1000,
                 consolidation_strength: float = 0.1):
        self.memory_size = memory_size
        self.consolidation_strength = consolidation_strength

        # Experience replay memory
        self.memory_buffer: deque = deque(maxlen=memory_size)

        # Elastic Weight Consolidation (EWC)
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_parameters: Dict[str, torch.Tensor] = {}

        logger.info("ContinualLearning initialized")

    def store_experience(self,
                        task_id: str,
                        experience: Tuple[np.ndarray, np.ndarray]) -> None:
        """経験保存"""
        self.memory_buffer.append({
            'task_id': task_id,
            'experience': experience,
            'timestamp': time.time()
        })

    def sample_memory(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """メモリサンプリング"""
        if len(self.memory_buffer) == 0:
            return []

        sample_size = min(batch_size, len(self.memory_buffer))
        indices = np.random.choice(len(self.memory_buffer), sample_size, replace=False)

        return [self.memory_buffer[i] for i in indices]

    def compute_fisher_information(self,
                                 model: nn.Module,
                                 dataset: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Fisher情報行列計算"""
        fisher_info = {}

        model.eval()

        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)

        for X, y in dataset:
            model.zero_grad()
            output = model(X)
            loss = F.mse_loss(output, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2

        # 正規化
        for name in fisher_info:
            fisher_info[name] /= len(dataset)

        return fisher_info

    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Elastic Weight Consolidation損失"""
        ewc_loss = 0

        for name, param in model.named_parameters():
            if name in self.fisher_information and name in self.optimal_parameters:
                fisher = self.fisher_information[name]
                optimal = self.optimal_parameters[name]
                ewc_loss += torch.sum(fisher * (param - optimal) ** 2)

        return self.consolidation_strength * ewc_loss

    def consolidate_knowledge(self, model: nn.Module, task_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """知識固定化"""
        # Fisher情報行列計算
        self.fisher_information = self.compute_fisher_information(model, task_data)

        # 最適パラメータ保存
        self.optimal_parameters = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        logger.info("Knowledge consolidation completed")

class MetaLearnerEngine:
    """メタ学習エンジン"""

    def __init__(self,
                 input_dim: int,
                 model_type: str = "simple",
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 adaptation_steps: int = 5):

        self.input_dim = input_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.adaptation_steps = adaptation_steps

        # メタモデル初期化
        if model_type == "attention":
            self.meta_model = AttentionMetaModel(input_dim)
        else:
            self.meta_model = SimpleMetaModel(input_dim)

        # オプティマイザ
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=outer_lr)

        # コンポーネント
        self.knowledge_transfer = ModelKnowledgeTransfer()
        self.few_shot_learner = FewShotLearning()
        self.continual_learner = ContinualLearning()

        # 履歴
        self.episode_history: List[EpisodeResult] = []
        self.task_registry: Dict[str, Task] = {}

        logger.info(f"MetaLearnerEngine initialized with {model_type} model")

    def _convert_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """NumPy配列をテンソルに変換"""
        return torch.FloatTensor(data)

    async def _inner_loop_adaptation(self,
                                   support_set: Tuple[np.ndarray, np.ndarray],
                                   task_id: str) -> Dict[str, torch.Tensor]:
        """内側ループ適応"""
        X_support, y_support = support_set
        X_tensor = self._convert_to_tensor(X_support)
        y_tensor = self._convert_to_tensor(y_support).view(-1, 1)

        # 現在のパラメータをfast weightsとして設定
        fast_weights = self.meta_model.get_fast_weights()

        # 内側ループ最適化
        for step in range(self.adaptation_steps):
            # Forward pass with fast weights
            self.meta_model.set_fast_weights(fast_weights)
            predictions = self.meta_model(X_tensor)

            # Loss計算
            loss = F.mse_loss(predictions, y_tensor)

            # 勾配計算
            grads = torch.autograd.grad(
                loss,
                fast_weights.values(),
                create_graph=True,
                allow_unused=True
            )

            # Fast weights更新
            new_fast_weights = {}
            for (name, param), grad in zip(fast_weights.items(), grads):
                if grad is not None:
                    new_fast_weights[name] = param - self.inner_lr * grad
                else:
                    new_fast_weights[name] = param

            fast_weights = new_fast_weights

        return fast_weights

    async def _outer_loop_update(self,
                               query_set: Tuple[np.ndarray, np.ndarray],
                               adapted_weights: Dict[str, torch.Tensor]) -> float:
        """外側ループ更新"""
        X_query, y_query = query_set
        X_tensor = self._convert_to_tensor(X_query)
        y_tensor = self._convert_to_tensor(y_query).view(-1, 1)

        # 適応後の重みで予測
        self.meta_model.set_fast_weights(adapted_weights)
        predictions = self.meta_model(X_tensor)

        # 外側ループ損失
        outer_loss = F.mse_loss(predictions, y_tensor)

        # 継続学習損失追加
        if hasattr(self.continual_learner, 'ewc_loss'):
            ewc_loss = self.continual_learner.ewc_loss(self.meta_model)
            outer_loss += ewc_loss

        return outer_loss.item()

    async def meta_learn_episode(self, task: Task) -> EpisodeResult:
        """メタ学習エピソード"""
        try:
            # タスク登録
            self.task_registry[task.task_id] = task
            self.knowledge_transfer.register_task(task)

            # 関連知識検索
            relevant_knowledge = self.knowledge_transfer.retrieve_relevant_knowledge(
                task.task_id, knowledge_type="parameters"
            )

            # 内側ループ適応前の損失
            X_support, y_support = task.support_set
            X_tensor = self._convert_to_tensor(X_support)
            y_tensor = self._convert_to_tensor(y_support).view(-1, 1)

            initial_predictions = self.meta_model(X_tensor)
            inner_loss_before = F.mse_loss(initial_predictions, y_tensor).item()

            # 内側ループ適応
            adapted_weights = await self._inner_loop_adaptation(task.support_set, task.task_id)

            # 適応後の損失
            self.meta_model.set_fast_weights(adapted_weights)
            adapted_predictions = self.meta_model(X_tensor)
            inner_loss_after = F.mse_loss(adapted_predictions, y_tensor).item()

            # 外側ループ更新
            self.meta_optimizer.zero_grad()
            outer_loss = await self._outer_loop_update(task.query_set, adapted_weights)

            # パラメータ更新
            loss_tensor = torch.tensor(outer_loss, requires_grad=True)
            if loss_tensor.requires_grad:
                loss_tensor.backward()
                self.meta_optimizer.step()

            # Fast weights reset
            self.meta_model.set_fast_weights(None)

            # 結果記録
            performance_gain = max(0, inner_loss_before - inner_loss_after)

            result = EpisodeResult(
                task_id=task.task_id,
                inner_loss=inner_loss_after,
                outer_loss=outer_loss,
                adaptation_steps=self.adaptation_steps,
                performance_gain=performance_gain,
                timestamp=time.time(),
                adapted_parameters=adapted_weights
            )

            self.episode_history.append(result)

            # 知識保存
            self.knowledge_transfer.store_knowledge(
                source_task=task.task_id,
                knowledge_type="parameters",
                knowledge_data=adapted_weights
            )

            # 経験保存（継続学習）
            self.continual_learner.store_experience(task.task_id, task.support_set)

            logger.debug(f"Meta-learning episode completed for {task.task_id}")
            return result

        except Exception as e:
            logger.error(f"Error in meta-learning episode: {e}")
            return EpisodeResult(
                task_id=task.task_id,
                inner_loss=1.0,
                outer_loss=1.0,
                adaptation_steps=0,
                performance_gain=0.0,
                timestamp=time.time()
            )

    async def meta_learn_batch(self, tasks: List[Task]) -> List[EpisodeResult]:
        """バッチメタ学習"""
        results = []

        for task in tasks:
            result = await self.meta_learn_episode(task)
            results.append(result)

        # 知識固定化（継続学習）
        if len(results) > 0:
            task_data = []
            for task in tasks:
                X, y = task.support_set
                X_tensor = self._convert_to_tensor(X)
                y_tensor = self._convert_to_tensor(y).view(-1, 1)
                task_data.extend([(X_tensor[i:i+1], y_tensor[i:i+1]) for i in range(X_tensor.shape[0])])

            if task_data:
                self.continual_learner.consolidate_knowledge(self.meta_model, task_data)

        return results

    async def predict_with_adaptation(self,
                                    support_set: Tuple[np.ndarray, np.ndarray],
                                    query_data: np.ndarray,
                                    task_id: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """適応予測"""
        try:
            # タスクID生成（未指定の場合）
            if task_id is None:
                task_id = f"predict_task_{int(time.time())}"

            # 少数ショット学習も試行
            few_shot_pred = await self.few_shot_learner.few_shot_predict(
                support_set, query_data
            )

            # メタ学習適応
            adapted_weights = await self._inner_loop_adaptation(support_set, task_id)

            # 予測実行
            X_query = self._convert_to_tensor(query_data)
            self.meta_model.set_fast_weights(adapted_weights)

            with torch.no_grad():
                meta_predictions = self.meta_model(X_query)
                meta_pred_np = meta_predictions.numpy()

            # Fast weights reset
            self.meta_model.set_fast_weights(None)

            # アンサンブル予測（メタ学習 + 少数ショット）
            ensemble_pred = 0.7 * meta_pred_np + 0.3 * few_shot_pred

            # メタデータ
            metadata = {
                'meta_prediction': meta_pred_np.tolist(),
                'few_shot_prediction': few_shot_pred.tolist(),
                'ensemble_weights': [0.7, 0.3],
                'adaptation_steps': self.adaptation_steps,
                'task_id': task_id,
                'timestamp': time.time()
            }

            return ensemble_pred, metadata

        except Exception as e:
            logger.error(f"Error in adaptive prediction: {e}")
            # フォールバック予測
            fallback_pred = np.zeros((query_data.shape[0], 1))
            return fallback_pred, {'error': str(e)}

    def get_learning_statistics(self) -> Dict[str, Any]:
        """学習統計取得"""
        if not self.episode_history:
            return {}

        recent_episodes = self.episode_history[-50:]  # 直近50エピソード

        return {
            'total_episodes': len(self.episode_history),
            'avg_inner_loss': np.mean([ep.inner_loss for ep in recent_episodes]),
            'avg_outer_loss': np.mean([ep.outer_loss for ep in recent_episodes]),
            'avg_performance_gain': np.mean([ep.performance_gain for ep in recent_episodes]),
            'knowledge_base_size': sum(len(knowledge_list) for knowledge_list in self.knowledge_transfer.knowledge_base.values()),
            'memory_buffer_size': len(self.continual_learner.memory_buffer),
            'registered_tasks': len(self.task_registry)
        }

# 便利関数
def create_meta_learner(input_dim: int, config: Optional[Dict[str, Any]] = None) -> MetaLearnerEngine:
    """メタ学習エンジン作成"""
    if config is None:
        config = {}

    return MetaLearnerEngine(
        input_dim=input_dim,
        model_type=config.get('model_type', 'simple'),
        inner_lr=config.get('inner_lr', 0.01),
        outer_lr=config.get('outer_lr', 0.001),
        adaptation_steps=config.get('adaptation_steps', 5)
    )

async def demo_meta_learning():
    """メタ学習デモ"""
    # サンプルタスク生成
    np.random.seed(42)
    torch.manual_seed(42)

    def generate_task(task_id: str, input_dim: int = 5) -> Task:
        # ランダムな線形関係のタスク
        n_support = 20
        n_query = 10

        true_weights = np.random.randn(input_dim, 1)
        noise_level = 0.1

        # サポートセット
        X_support = np.random.randn(n_support, input_dim)
        y_support = X_support @ true_weights + np.random.randn(n_support, 1) * noise_level

        # クエリセット
        X_query = np.random.randn(n_query, input_dim)
        y_query = X_query @ true_weights + np.random.randn(n_query, 1) * noise_level

        return Task(
            task_id=task_id,
            support_set=(X_support, y_support),
            query_set=(X_query, y_query),
            metadata={'domain': 'synthetic', 'difficulty': 1.0}
        )

    # メタ学習エンジン作成
    engine = create_meta_learner(input_dim=5)

    # トレーニングタスク
    train_tasks = [generate_task(f"train_task_{i}") for i in range(10)]

    print("Starting meta-learning training...")
    results = await engine.meta_learn_batch(train_tasks)

    print(f"Training completed. {len(results)} episodes.")

    # 統計表示
    stats = engine.get_learning_statistics()
    print("Learning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # テストタスクで予測
    test_task = generate_task("test_task")
    prediction, metadata = await engine.predict_with_adaptation(
        test_task.support_set,
        test_task.query_set[0],
        "test_task"
    )

    print(f"Test prediction shape: {prediction.shape}")
    print(f"Test metadata keys: {list(metadata.keys())}")

if __name__ == "__main__":
    asyncio.run(demo_meta_learning())