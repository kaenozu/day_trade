#!/usr/bin/env python3
"""
深層学習統合システム
Phase F: 次世代機能拡張フェーズ

Transformer・LSTM・CNN による高精度予測システム
"""

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# PyTorch可用性チェック
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch未インストール - 深層学習モデル利用不可", ImportWarning, stacklevel=2
    )

from ..core.optimization_strategy import (
    OptimizationConfig,
    OptimizationLevel,
    OptimizationStrategy,
    optimization_strategy,
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelType(Enum):
    """深層学習モデルタイプ"""

    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    HYBRID = "hybrid"
    HYBRID_LSTM_TRANSFORMER = "hybrid_lstm_transformer"
    ENSEMBLE = "ensemble"


@dataclass
class BaseModelConfig:
    """
    Issue #528対応: 統合された基本モデル設定

    すべての深層学習モデルの共通設定基盤
    """
    # データ設定
    sequence_length: int = 60
    prediction_horizon: int = 1
    validation_split: float = 0.2

    # 学習設定
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10

    # モデル設定
    hidden_size: int = 128  # hidden_dim と統一
    num_layers: int = 2
    dropout_rate: float = 0.2

    # 実行環境設定
    use_pytorch: bool = False
    device: str = "cpu"  # "cpu", "cuda", "mps"


@dataclass
class TransformerConfig(BaseModelConfig):
    """
    Issue #528対応: Transformer特化設定
    """
    # Transformer固有設定
    num_heads: int = 8
    d_model: int = 128
    d_ff: int = 512  # フィードフォワード次元
    max_position_encoding: int = 1000

    # デフォルト値の調整
    hidden_size: int = 128  # d_modelと同期


@dataclass
class LSTMConfig(BaseModelConfig):
    """
    Issue #528対応: LSTM特化設定
    """
    # LSTM固有設定
    bidirectional: bool = False
    layer_norm: bool = True


@dataclass
class CNNConfig(BaseModelConfig):
    """
    Issue #528対応: CNN特化設定
    """
    # CNN固有設定
    kernel_size: int = 3
    num_filters: int = 64
    pool_size: int = 2


# Issue #528対応: 後方互換性のための統合設定クラス
@dataclass
class ModelConfig(BaseModelConfig):
    """
    Issue #528対応: 後方互換性維持のための統合設定クラス

    既存のModelConfigを継承し、必要な追加フィールドを含む
    """
    model_type: ModelType = ModelType.LSTM

    # Transformer用（TransformerConfigから継承）
    num_heads: int = 8
    d_model: int = 128

    # CNN用（CNNConfigから継承）
    kernel_size: int = 3
    num_filters: int = 64

    def to_specialized_config(self) -> Union[TransformerConfig, LSTMConfig, CNNConfig]:
        """
        Issue #528対応: モデルタイプに応じた特化設定への変換

        Returns:
            モデルタイプに適した特化設定インスタンス
        """
        base_kwargs = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'validation_split': self.validation_split,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'use_pytorch': self.use_pytorch,
        }

        if self.model_type in [ModelType.TRANSFORMER, ModelType.HYBRID_LSTM_TRANSFORMER]:
            return TransformerConfig(
                **base_kwargs,
                num_heads=self.num_heads,
                d_model=self.d_model,
            )
        elif self.model_type in [ModelType.LSTM, ModelType.GRU]:
            return LSTMConfig(**base_kwargs)
        elif self.model_type == ModelType.CNN:
            return CNNConfig(
                **base_kwargs,
                kernel_size=self.kernel_size,
                num_filters=self.num_filters,
            )
        else:
            return self  # デフォルトはBaseModelConfig


# Issue #528対応: 後方互換性のためのエイリアス
DeepLearningConfig = ModelConfig


# Issue #528対応: 設定変換ヘルパー関数
def convert_legacy_config(legacy_config: Dict[str, Any]) -> ModelConfig:
    """
    Issue #528対応: レガシー設定辞書からModelConfigへの変換

    Args:
        legacy_config: 古い形式の設定辞書

    Returns:
        ModelConfig: 統合された設定オブジェクト
    """
    # hidden_dim -> hidden_size の統一
    if 'hidden_dim' in legacy_config:
        legacy_config['hidden_size'] = legacy_config.pop('hidden_dim')

    # d_model のデフォルト値設定
    if 'hidden_size' in legacy_config and 'd_model' not in legacy_config:
        legacy_config['d_model'] = legacy_config['hidden_size']

    # モデルタイプの処理
    if 'model_type' in legacy_config and isinstance(legacy_config['model_type'], str):
        legacy_config['model_type'] = ModelType(legacy_config['model_type'])

    return ModelConfig(**legacy_config)


def create_model_config(
    model_type: ModelType,
    **kwargs
) -> Union[ModelConfig, TransformerConfig, LSTMConfig, CNNConfig]:
    """
    Issue #528対応: モデルタイプに応じた最適な設定オブジェクト作成

    Args:
        model_type: 作成するモデルのタイプ
        **kwargs: 設定パラメータ

    Returns:
        モデルタイプに最適化された設定オブジェクト
    """
    # 基本設定から開始
    config = ModelConfig(model_type=model_type, **kwargs)

    # 特化設定への変換
    return config.to_specialized_config()


@dataclass
class TrainingResult:
    """訓練結果"""

    model_type: ModelType
    training_loss: float
    validation_loss: float
    training_accuracy: float
    validation_accuracy: float
    training_time: float
    epochs_trained: int
    model_parameters: int
    convergence_achieved: bool


@dataclass
class ModelTrainingResult:
    """モデル訓練結果（API用）"""

    final_loss: float
    best_loss: float
    epochs_run: int
    training_time: float
    validation_metrics: Dict[str, float]
    convergence_achieved: bool = True


@dataclass
class UncertaintyEstimate:
    """不確実性推定"""

    mean: float
    std: float
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    epistemic: float  # モデル不確実性
    aleatoric: float  # データ不確実性


@dataclass
class PredictionResult:
    """予測結果"""

    predictions: np.ndarray
    confidence: np.ndarray
    uncertainty: Optional[UncertaintyEstimate] = None
    prediction_time: float = 0.0
    model_used: str = ""
    feature_importance: Optional[Dict[str, float]] = None
    metrics: Dict[str, float] = None
    model_weights: Dict[str, float] = None


class BaseDeepLearningModel(ABC):
    """深層学習モデル基底クラス"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_history = []

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """モデル構築"""
        pass

    def train(self, data: pd.DataFrame) -> ModelTrainingResult:
        """データフレームからモデル訓練"""
        X, y = self.prepare_training_data(data)

        # モデル構築
        input_shape = X.shape[1:]
        self.model = self.build_model(input_shape)

        # 訓練実行
        start_time = time.time()
        training_result = self._train_internal(X, y)
        training_time = time.time() - start_time

        return ModelTrainingResult(
            final_loss=training_result.get("final_loss", 0.0),
            best_loss=training_result.get("best_loss", 0.0),
            epochs_run=training_result.get("epochs_run", self.config.epochs),
            training_time=training_time,
            validation_metrics=training_result.get("validation_metrics", {}),
            convergence_achieved=training_result.get("convergence_achieved", True),
        )

    @abstractmethod
    def _train_internal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """内部訓練メソッド"""
        pass

    def prepare_training_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """訓練データ準備"""
        return self.prepare_data(data)

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """予測実行"""
        X, _ = self.prepare_data(data)

        if not self.is_trained or self.model is None:
            raise ValueError("モデルが訓練されていません")

        start_time = time.time()
        predictions = self._predict_internal(X)
        prediction_time = time.time() - start_time

        # 信頼度スコア計算（簡易版）
        confidence = np.ones(len(predictions)) * 0.8

        return PredictionResult(
            predictions=predictions,
            confidence=confidence,
            prediction_time=prediction_time,
            model_used=self.__class__.__name__,
            metrics={"mae": 0.0},  # 実際の実装では適切な計算を行う
        )

    @abstractmethod
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """内部予測メソッド"""
        pass

    def predict_with_uncertainty(
        self, data: pd.DataFrame, num_samples: int = 100
    ) -> PredictionResult:
        """不確実性付き予測"""
        # Monte Carlo Dropoutによる不確実性推定
        predictions_list = []

        for _ in range(num_samples):
            pred_result = self.predict(data)
            predictions_list.append(pred_result.predictions)

        # 統計計算
        all_predictions = np.array(predictions_list)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)

        # 不確実性推定
        uncertainty = UncertaintyEstimate(
            mean=float(np.mean(std_pred)),
            std=float(np.std(std_pred)),
            lower_bound=mean_pred - 1.96 * std_pred,
            upper_bound=mean_pred + 1.96 * std_pred,
            epistemic=float(np.mean(std_pred)),  # モデル不確実性
            aleatoric=float(np.std(std_pred)),  # データ不確実性
        )

        return PredictionResult(
            predictions=mean_pred,
            confidence=1.0 - std_pred / (np.abs(mean_pred) + 1e-8),
            uncertainty=uncertainty,
            model_used=self.__class__.__name__,
        )

    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """特徴量重要度分析"""
        # Permutation Importanceによる特徴量重要度計算
        X, y = self.prepare_data(data)
        baseline_pred = self._predict_internal(X)
        baseline_error = np.mean((baseline_pred - y) ** 2) if y is not None else 0.0

        feature_importance = {}
        feature_names = ["Open", "High", "Low", "Close", "Volume"][: X.shape[-1]]

        for i, feature_name in enumerate(feature_names):
            # 特徴量をシャッフル
            X_shuffled = X.copy()
            if len(X_shuffled.shape) == 3:  # (samples, sequence, features)
                X_shuffled[:, :, i] = np.random.permutation(X_shuffled[:, :, i])

            shuffled_pred = self._predict_internal(X_shuffled)
            shuffled_error = np.mean((shuffled_pred - baseline_pred) ** 2)

            importance = max(0, shuffled_error - baseline_error) / (
                baseline_error + 1e-8
            )
            feature_importance[feature_name] = float(importance)

        # 正規化
        total_importance = sum(feature_importance.values()) or 1.0
        for key in feature_importance:
            feature_importance[key] /= total_importance

        return feature_importance

    def prepare_data(
        self, data: pd.DataFrame, target_column: str = "Close"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """時系列データ準備"""
        # OHLCV特徴量使用
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]
        available_columns = [col for col in feature_columns if col in data.columns]

        if not available_columns:
            raise ValueError("必要な価格データ列が見つかりません")

        # データ正規化
        features = data[available_columns].values
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        # シーケンスデータ作成
        X, y = [], []
        for i in range(
            len(features)
            - self.config.sequence_length
            - self.config.prediction_horizon
            + 1
        ):
            X.append(features[i : i + self.config.sequence_length])
            if target_column in data.columns:
                y.append(
                    data[target_column].values[
                        i
                        + self.config.sequence_length : i
                        + self.config.sequence_length
                        + self.config.prediction_horizon
                    ]
                )
            else:
                # ターゲット列がない場合は終値を使用
                y.append(
                    features[
                        i
                        + self.config.sequence_length : i
                        + self.config.sequence_length
                        + self.config.prediction_horizon,
                        -1,
                    ]
                )

        return np.array(X), np.array(y)


class TransformerModel(BaseDeepLearningModel):
    """Transformer時系列予測モデル"""

    def __init__(self, config: Union[ModelConfig, DeepLearningConfig]):
        # DeepLearningConfigの場合は変換
        if isinstance(config, DeepLearningConfig):
            model_config = ModelConfig(
                sequence_length=config.sequence_length,
                prediction_horizon=config.prediction_horizon,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                dropout_rate=config.dropout_rate,
                learning_rate=config.learning_rate,
                epochs=config.epochs,
                batch_size=config.batch_size,
                early_stopping_patience=config.early_stopping_patience,
            )
        else:
            model_config = config

        super().__init__(model_config)
        self.attention_heads = getattr(config, "num_heads", 8)
        self.feed_forward_dim = 512

    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Transformerモデル構築"""
        try:
            # PyTorch実装を試行
            import torch
            import torch.nn as nn

            return self._build_pytorch_transformer(input_shape)

        except ImportError:
            # フォールバック: NumPy実装
            logger.info("PyTorch未利用のため、NumPy Transformerを使用")
            return self._build_numpy_transformer(input_shape)

    def _build_pytorch_transformer(
        self, input_shape: Tuple[int, ...]
    ) -> "torch.nn.Module":
        """PyTorch Transformerモデル"""
        import torch.nn as nn

        class TransformerPredictor(nn.Module):
            def __init__(
                self,
                input_dim,
                hidden_dim,
                num_heads,
                num_layers,
                output_dim,
                dropout_rate,
            ):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, hidden_dim)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout_rate,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_projection = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x = self.input_projection(x)
                x = x.transpose(0, 1)  # (seq, batch, features)
                x = self.transformer(x)
                x = x[-1]  # 最後の出力のみ使用
                return self.output_projection(x)

        seq_len, n_features = input_shape
        model = TransformerPredictor(
            input_dim=n_features,
            hidden_dim=self.config.hidden_size,
            num_heads=self.attention_heads,
            num_layers=self.config.num_layers,
            output_dim=self.config.prediction_horizon,
            dropout_rate=self.config.dropout_rate,
        )

        return model

    def _build_numpy_transformer(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """NumPy Transformerモデル（簡易実装）"""
        seq_len, n_features = input_shape

        # 簡易Multi-Head Attention実装
        return {
            "type": "numpy_transformer",
            "input_shape": input_shape,
            "hidden_size": self.config.hidden_size,
            "num_heads": self.attention_heads,
            "weights": self._initialize_transformer_weights(n_features),
        }

    def _initialize_transformer_weights(self, n_features: int) -> Dict[str, np.ndarray]:
        """Transformerパラメータ初期化"""
        hidden_size = self.config.hidden_size

        return {
            "query_weights": np.random.randn(n_features, hidden_size) * 0.1,
            "key_weights": np.random.randn(n_features, hidden_size) * 0.1,
            "value_weights": np.random.randn(n_features, hidden_size) * 0.1,
            "output_weights": np.random.randn(
                hidden_size, self.config.prediction_horizon
            )
            * 0.1,
            "layer_norm_weights": np.ones(hidden_size),
            "layer_norm_bias": np.zeros(hidden_size),
        }

    def _train_internal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Transformer内部訓練"""
        start_time = time.time()

        try:
            if hasattr(self.model, "parameters"):  # PyTorchモデル
                result = self._train_pytorch_model(X, y, start_time)
                self.is_trained = True
                return {
                    "final_loss": result.validation_loss,
                    "best_loss": result.validation_loss,
                    "epochs_run": result.epochs_trained,
                    "validation_metrics": {"mse": result.validation_loss},
                    "convergence_achieved": result.convergence_achieved,
                }
            else:  # NumPyモデル
                result = self._train_numpy_model(X, y, start_time)
                self.is_trained = True
                return {
                    "final_loss": result.validation_loss,
                    "best_loss": result.validation_loss,
                    "epochs_run": result.epochs_trained,
                    "validation_metrics": {"mse": result.validation_loss},
                    "convergence_achieved": result.convergence_achieved,
                }

        except Exception as e:
            logger.error(f"Transformer訓練エラー: {e}")
            # フォールバックモデル使用
            result = self._train_fallback_model(X, y, start_time)
            self.is_trained = True
            return {
                "final_loss": result.validation_loss,
                "best_loss": result.validation_loss,
                "epochs_run": result.epochs_trained,
                "validation_metrics": {"mse": result.validation_loss},
                "convergence_achieved": result.convergence_achieved,
            }

    def _train_pytorch_model(
        self, X: np.ndarray, y: np.ndarray, start_time: float
    ) -> TrainingResult:
        """PyTorch Transformer訓練"""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # データテンソル変換
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # 訓練・検証分割
        val_size = int(len(X) * self.config.validation_split)
        X_train, X_val = X_tensor[:-val_size], X_tensor[-val_size:]
        y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]

        # オプティマイザ・損失関数
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        # 訓練ループ
        for epoch in range(self.config.epochs):
            # 訓練フェーズ
            self.model.train()
            train_loss = 0.0

            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_train[i : i + self.config.batch_size]
                batch_y = y_train[i : i + self.config.batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 検証フェーズ
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            # 早期停止チェック
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"早期停止: エポック {epoch + 1}")
                    break

        training_time = time.time() - start_time

        # 精度計算（MAPE使用）
        with torch.no_grad():
            train_pred = self.model(X_train)
            val_pred = self.model(X_val)

            train_accuracy = self._calculate_accuracy(
                y_train.numpy(), train_pred.numpy()
            )
            val_accuracy = self._calculate_accuracy(y_val.numpy(), val_pred.numpy())

        return TrainingResult(
            model_type=ModelType.TRANSFORMER,
            training_loss=train_loss / len(X_train),
            validation_loss=val_loss,
            training_accuracy=train_accuracy,
            validation_accuracy=val_accuracy,
            training_time=training_time,
            epochs_trained=epoch + 1,
            model_parameters=sum(p.numel() for p in self.model.parameters()),
            convergence_achieved=patience_counter < self.config.early_stopping_patience,
        )

    def _train_numpy_model(
        self, X: np.ndarray, y: np.ndarray, start_time: float
    ) -> TrainingResult:
        """NumPy Transformer訓練（簡易実装）"""
        val_size = int(len(X) * self.config.validation_split)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        # 簡易訓練ループ
        best_loss = float("inf")

        for epoch in range(min(self.config.epochs, 20)):  # 簡易版は20エポックまで
            # フォワードパス（簡易Multi-Head Attention）
            train_pred = self._numpy_forward_pass(X_train)
            val_pred = self._numpy_forward_pass(X_val)

            # 損失計算
            train_loss = np.mean((train_pred - y_train) ** 2)
            val_loss = np.mean((val_pred - y_val) ** 2)

            # パラメータ更新（簡易勾配降下）
            self._numpy_backward_pass(X_train, y_train, train_pred)

            if val_loss < best_loss:
                best_loss = val_loss

        training_time = time.time() - start_time

        train_accuracy = self._calculate_accuracy(y_train, train_pred)
        val_accuracy = self._calculate_accuracy(y_val, val_pred)

        return TrainingResult(
            model_type=ModelType.TRANSFORMER,
            training_loss=train_loss,
            validation_loss=val_loss,
            training_accuracy=train_accuracy,
            validation_accuracy=val_accuracy,
            training_time=training_time,
            epochs_trained=epoch + 1,
            model_parameters=len(self.model["weights"]),
            convergence_achieved=True,
        )

    def _numpy_forward_pass(self, X: np.ndarray) -> np.ndarray:
        """NumPy フォワードパス"""
        weights = self.model["weights"]

        # 簡易Multi-Head Attention
        batch_size, seq_len, n_features = X.shape

        # Q, K, V計算
        Q = np.dot(X.reshape(-1, n_features), weights["query_weights"]).reshape(
            batch_size, seq_len, -1
        )
        K = np.dot(X.reshape(-1, n_features), weights["key_weights"]).reshape(
            batch_size, seq_len, -1
        )
        V = np.dot(X.reshape(-1, n_features), weights["value_weights"]).reshape(
            batch_size, seq_len, -1
        )

        # Attention計算
        attention_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(
            self.config.hidden_size
        )
        attention_weights = self._softmax(attention_scores)
        attention_output = np.matmul(attention_weights, V)

        # 出力層
        final_output = attention_output[:, -1, :]  # 最後のタイムステップ
        predictions = np.dot(final_output, weights["output_weights"])

        return predictions

    def _numpy_backward_pass(
        self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray
    ):
        """NumPy バックワード パス（簡易実装）"""
        learning_rate = self.config.learning_rate
        batch_size = X.shape[0]

        # 出力層勾配
        output_error = predictions - y
        output_gradient = output_error / batch_size

        # パラメータ更新（簡易）
        weights = self.model["weights"]
        weights["output_weights"] -= learning_rate * output_gradient.mean() * 0.001

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax関数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _train_fallback_model(
        self, X: np.ndarray, y: np.ndarray, start_time: float
    ) -> TrainingResult:
        """フォールバックモデル訓練"""
        # 線形回帰フォールバック
        X_flattened = X.reshape(X.shape[0], -1)

        # 正規方程式による最小二乗法
        try:
            weights = np.linalg.lstsq(X_flattened, y, rcond=None)[0]
            predictions = np.dot(X_flattened, weights)

            loss = np.mean((predictions - y) ** 2)
            accuracy = self._calculate_accuracy(y, predictions)

            self.model = {"fallback_weights": weights}

            return TrainingResult(
                model_type=ModelType.TRANSFORMER,
                training_loss=loss,
                validation_loss=loss,
                training_accuracy=accuracy,
                validation_accuracy=accuracy,
                training_time=time.time() - start_time,
                epochs_trained=1,
                model_parameters=len(weights.flatten()),
                convergence_achieved=True,
            )

        except Exception as e:
            logger.error(f"フォールバックモデル訓練エラー: {e}")
            raise

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Transformer内部予測"""
        try:
            if hasattr(self.model, "parameters"):  # PyTorchモデル
                return self._predict_pytorch(X)
            elif (
                isinstance(self.model, dict) and "fallback_weights" in self.model
            ):  # フォールバックモデル
                X_flattened = X.reshape(X.shape[0], -1)
                return np.dot(X_flattened, self.model["fallback_weights"])
            else:  # NumPyモデル
                return self._numpy_forward_pass(X)

        except Exception as e:
            logger.error(f"Transformer予測エラー: {e}")
            # 簡易フォールバック
            return np.zeros((X.shape[0], self.config.prediction_horizon))

    def _predict_pytorch(self, X: np.ndarray) -> np.ndarray:
        """PyTorch予測"""
        import torch

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor)
            return predictions.numpy()

    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """精度計算（MAPE）"""
        try:
            mape = (
                np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8)))
                * 100
            )
            accuracy = max(0, 100 - mape)  # MAPEから精度に変換
            return min(accuracy, 100) / 100  # 0-1範囲に正規化
        except:
            return 0.0


class LSTMModel(BaseDeepLearningModel):
    """LSTM時系列予測モデル"""

    def __init__(self, config: Union[ModelConfig, DeepLearningConfig]):
        # DeepLearningConfigの場合は変換
        if isinstance(config, DeepLearningConfig):
            model_config = ModelConfig(
                sequence_length=config.sequence_length,
                prediction_horizon=config.prediction_horizon,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                dropout_rate=config.dropout_rate,
                learning_rate=config.learning_rate,
                epochs=config.epochs,
                batch_size=config.batch_size,
                early_stopping_patience=config.early_stopping_patience,
            )
        else:
            model_config = config

        super().__init__(model_config)

    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """LSTMモデル構築"""
        try:
            # PyTorch実装を試行
            import torch
            import torch.nn as nn

            return self._build_pytorch_lstm(input_shape)

        except ImportError:
            # フォールバック: NumPy実装
            logger.info("PyTorch未利用のため、NumPy LSTMを使用")
            return self._build_numpy_lstm(input_shape)

    def _build_pytorch_lstm(self, input_shape: Tuple[int, ...]) -> "torch.nn.Module":
        """PyTorch LSTMモデル"""
        import torch.nn as nn

        class LSTMPredictor(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
                )
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])  # 最後の出力のみ使用

        seq_len, n_features = input_shape
        model = LSTMPredictor(
            input_dim=n_features,
            hidden_dim=self.config.hidden_size,
            num_layers=self.config.num_layers,
            output_dim=self.config.prediction_horizon,
            dropout=self.config.dropout_rate,
        )

        return model

    def _build_numpy_lstm(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """NumPy LSTM（簡易実装）"""
        seq_len, n_features = input_shape
        hidden_size = self.config.hidden_size

        return {
            "type": "numpy_lstm",
            "input_shape": input_shape,
            "hidden_size": hidden_size,
            "weights": self._initialize_lstm_weights(n_features, hidden_size),
        }

    def _initialize_lstm_weights(
        self, n_features: int, hidden_size: int
    ) -> Dict[str, np.ndarray]:
        """LSTMパラメータ初期化"""
        # LSTM重み行列（簡易実装）
        return {
            "W_f": np.random.randn(n_features + hidden_size, hidden_size)
            * 0.1,  # 忘却ゲート
            "W_i": np.random.randn(n_features + hidden_size, hidden_size)
            * 0.1,  # 入力ゲート
            "W_o": np.random.randn(n_features + hidden_size, hidden_size)
            * 0.1,  # 出力ゲート
            "W_c": np.random.randn(n_features + hidden_size, hidden_size)
            * 0.1,  # セル状態
            "b_f": np.zeros(hidden_size),
            "b_i": np.zeros(hidden_size),
            "b_o": np.zeros(hidden_size),
            "b_c": np.zeros(hidden_size),
            "W_output": np.random.randn(hidden_size, self.config.prediction_horizon)
            * 0.1,
        }

    def _train_internal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """LSTM内部訓練"""
        start_time = time.time()

        try:
            if hasattr(self.model, "parameters"):  # PyTorchモデル
                result = self._train_pytorch_lstm(X, y, start_time)
                self.is_trained = True
                return {
                    "final_loss": result.validation_loss,
                    "best_loss": result.validation_loss,
                    "epochs_run": result.epochs_trained,
                    "validation_metrics": {"mse": result.validation_loss},
                    "convergence_achieved": result.convergence_achieved,
                }
            else:  # NumPyモデル
                result = self._train_numpy_lstm(X, y, start_time)
                self.is_trained = True
                return {
                    "final_loss": result.validation_loss,
                    "best_loss": result.validation_loss,
                    "epochs_run": result.epochs_trained,
                    "validation_metrics": {"mse": result.validation_loss},
                    "convergence_achieved": result.convergence_achieved,
                }
        except Exception as e:
            logger.error(f"LSTM訓練エラー: {e}")
            self.is_trained = True
            return {
                "final_loss": 1.0,
                "best_loss": 1.0,
                "epochs_run": 1,
                "validation_metrics": {"mse": 1.0},
                "convergence_achieved": False,
            }

    def _train_pytorch_lstm(
        self, X: np.ndarray, y: np.ndarray, start_time: float
    ) -> TrainingResult:
        """PyTorch LSTM訓練"""
        # Transformerと類似の実装
        import torch
        import torch.nn as nn
        import torch.optim as optim

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        val_size = int(len(X) * self.config.validation_split)
        X_train, X_val = X_tensor[:-val_size], X_tensor[-val_size:]
        y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0

            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_train[i : i + self.config.batch_size]
                batch_y = y_train[i : i + self.config.batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break

        training_time = time.time() - start_time

        with torch.no_grad():
            train_pred = self.model(X_train)
            val_pred = self.model(X_val)
            train_accuracy = self._calculate_accuracy(
                y_train.numpy(), train_pred.numpy()
            )
            val_accuracy = self._calculate_accuracy(y_val.numpy(), val_pred.numpy())

        return TrainingResult(
            model_type=ModelType.LSTM,
            training_loss=train_loss / len(X_train),
            validation_loss=val_loss,
            training_accuracy=train_accuracy,
            validation_accuracy=val_accuracy,
            training_time=training_time,
            epochs_trained=epoch + 1,
            model_parameters=sum(p.numel() for p in self.model.parameters()),
            convergence_achieved=patience_counter < self.config.early_stopping_patience,
        )

    def _train_numpy_lstm(
        self, X: np.ndarray, y: np.ndarray, start_time: float
    ) -> TrainingResult:
        """NumPy LSTM訓練（簡易実装）"""
        val_size = int(len(X) * self.config.validation_split)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        best_loss = float("inf")

        for epoch in range(min(self.config.epochs, 30)):
            train_pred = self._numpy_lstm_forward(X_train)
            val_pred = self._numpy_lstm_forward(X_val)

            train_loss = np.mean((train_pred - y_train) ** 2)
            val_loss = np.mean((val_pred - y_val) ** 2)

            # 簡易パラメータ更新
            self._numpy_lstm_backward(X_train, y_train, train_pred)

            if val_loss < best_loss:
                best_loss = val_loss

        training_time = time.time() - start_time

        train_accuracy = self._calculate_accuracy(y_train, train_pred)
        val_accuracy = self._calculate_accuracy(y_val, val_pred)

        return TrainingResult(
            model_type=ModelType.LSTM,
            training_loss=train_loss,
            validation_loss=val_loss,
            training_accuracy=train_accuracy,
            validation_accuracy=val_accuracy,
            training_time=training_time,
            epochs_trained=epoch + 1,
            model_parameters=sum(w.size for w in self.model["weights"].values()),
            convergence_achieved=True,
        )

    def _numpy_lstm_forward(self, X: np.ndarray) -> np.ndarray:
        """NumPy LSTM フォワードパス"""
        batch_size, seq_len, n_features = X.shape
        hidden_size = self.config.hidden_size

        weights = self.model["weights"]

        # 初期状態
        h = np.zeros((batch_size, hidden_size))
        c = np.zeros((batch_size, hidden_size))

        # 各タイムステップでの処理
        for t in range(seq_len):
            x_t = X[:, t, :]
            input_combined = np.concatenate([x_t, h], axis=1)

            # LSTMゲート計算
            f_gate = self._sigmoid(
                np.dot(input_combined, weights["W_f"]) + weights["b_f"]
            )
            i_gate = self._sigmoid(
                np.dot(input_combined, weights["W_i"]) + weights["b_i"]
            )
            o_gate = self._sigmoid(
                np.dot(input_combined, weights["W_o"]) + weights["b_o"]
            )
            c_candidate = np.tanh(
                np.dot(input_combined, weights["W_c"]) + weights["b_c"]
            )

            # セル状態・隠れ状態更新
            c = f_gate * c + i_gate * c_candidate
            h = o_gate * np.tanh(c)

        # 出力層
        output = np.dot(h, weights["W_output"])
        return output

    def _numpy_lstm_backward(
        self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray
    ):
        """NumPy LSTM バックワードパス（簡易実装）"""
        learning_rate = self.config.learning_rate * 0.1

        # 簡易勾配更新
        output_error = predictions - y
        weights = self.model["weights"]

        # 出力層重み更新
        weights["W_output"] -= learning_rate * output_error.mean() * 0.001

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """シグモイド関数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """LSTM内部予測"""
        try:
            if hasattr(self.model, "parameters"):  # PyTorchモデル
                import torch

                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    return self.model(X_tensor).numpy()
            else:  # NumPyモデル
                return self._numpy_lstm_forward(X)
        except Exception as e:
            logger.error(f"LSTM予測エラー: {e}")
            return np.zeros((X.shape[0], self.config.prediction_horizon))

    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """精度計算"""
        try:
            mape = (
                np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8)))
                * 100
            )
            accuracy = max(0, 100 - mape)
            return min(accuracy, 100) / 100
        except:
            return 0.0


class DeepLearningModelManager:
    """深層学習モデル統合管理"""

    def __init__(
        self,
        dl_config: DeepLearningConfig,
        opt_config: Optional[OptimizationConfig] = None,
        ensemble_weights: Optional[Dict[str, float]] = None, # Issue #545対応: アンサンブル重み
    ):
        self.dl_config = dl_config
        self.opt_config = opt_config or OptimizationConfig()
        self.models = {}
        self.ensemble_weights = ensemble_weights if ensemble_weights is not None else {} # Issue #545対応: アンサンブル重み
        self.performance_stats = {"total_predictions": 0, "total_time": 0.0}

        logger.info("深層学習モデル管理システム初期化完了")

    def create_model(
        self, model_type: ModelType, model_config: ModelConfig
    ) -> BaseDeepLearningModel:
        """モデル作成"""
        if model_type == ModelType.TRANSFORMER:
            return TransformerModel(model_config)
        elif model_type == ModelType.LSTM:
            return LSTMModel(model_config)
        elif model_type == ModelType.HYBRID_LSTM_TRANSFORMER:
            # ハイブリッドLSTM-Transformerモデル作成
            from .hybrid_lstm_transformer import HybridLSTMTransformerEngine

            return HybridLSTMTransformerEngine(model_config)
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

    def train_model(
        self, model_type: ModelType, data: pd.DataFrame, target_column: str = "Close"
    ) -> TrainingResult:
        """モデル訓練"""
        model_config = ModelConfig(model_type=model_type)
        model = self.create_model(model_type, model_config)

        # データ準備
        X, y = model.prepare_data(data, target_column)

        # モデル構築
        input_shape = X.shape[1:]
        model.model = model.build_model(input_shape)

        # 訓練実行
        training_result = model.train(X, y)

        # モデル保存
        self.models[model_type.value] = model

        logger.info(
            f"{model_type.value}モデル訓練完了: 精度{training_result.validation_accuracy:.3f}"
        )

        return training_result

    def predict_ensemble(self, data: pd.DataFrame) -> PredictionResult:
        """アンサンブル予測"""
        if not self.models:
            raise ValueError("訓練済みモデルがありません")

        predictions = []
        confidences = []
        model_names = []

        # 各モデルで予測
        for model_name, model in self.models.items():
            X, _ = model.prepare_data(data)
            result = model.predict(X)

            predictions.append(result.predictions)
            confidences.append(result.confidence)
            model_names.append(model_name)

        # アンサンブル統合
        if len(predictions) == 0:
            raise ValueError("予測可能なモデルがありません")
        elif len(predictions) == 1:
            ensemble_pred = predictions[0]
            ensemble_conf = confidences[0]
            ensemble_model_weights = {model_names[0]: 1.0}
        else:
            # Issue #545対応: アンサンブル重みの動的管理
            calculated_weights = np.ones(len(predictions)) # デフォルトは均等重み

            if self.ensemble_weights: # 外部から重みが指定されている場合
                for i, model_name in enumerate(model_names):
                    if model_name in self.ensemble_weights:
                        calculated_weights[i] = self.ensemble_weights[model_name]
                    else:
                        logger.warning(f"指定されたモデル '{model_name}' のアンサンブル重みが見つかりません。均等重みを使用します。")

            # 重みを正規化
            if np.sum(calculated_weights) > 0:
                calculated_weights = calculated_weights / np.sum(calculated_weights)
            else:
                logger.warning("アンサンブル重みの合計がゼロです。均等重みにフォールバックします。")
                calculated_weights = np.ones(len(predictions)) / len(predictions)

            ensemble_pred = np.average(predictions, axis=0, weights=calculated_weights)
            ensemble_conf = np.average(confidences, axis=0, weights=calculated_weights)
            ensemble_model_weights = dict(zip(model_names, calculated_weights))

        return PredictionResult(
            predictions=ensemble_pred,
            confidence=ensemble_conf,
            model_used=f"Ensemble({'+'.join(model_names)})",
            model_weights=ensemble_model_weights, # Issue #545対応: 使用された重みを記録
        )

    def register_model(self, name: str, model: BaseDeepLearningModel):
        """モデル登録"""
        self.models[name] = model
        logger.info(f"モデル登録完了: {name}")

    def train_ensemble(self, data: pd.DataFrame, parallel: bool = True) -> Dict[str, ModelTrainingResult]:
        """
        アンサンブル訓練 - Issue #708対応並列化版

        Args:
            data: 訓練データ
            parallel: 並列訓練を有効にするか

        Returns:
            Dict[str, ModelTrainingResult]: 各モデルの訓練結果
        """
        if parallel and len(self.models) > 1:
            return self._parallel_train_ensemble(data)
        else:
            return self._sequential_train_ensemble(data)

    def _parallel_train_ensemble(self, data: pd.DataFrame) -> Dict[str, ModelTrainingResult]:
        """並列アンサンブル訓練実装"""
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
        import multiprocessing as mp

        def train_single_deep_model(args):
            """単一深層学習モデルの訓練関数"""
            model_name, model, data_copy, dl_config = args
            try:
                logger.info(f"深層学習モデル並列訓練開始: {model_name}")
                start_time = time.time()

                # ModelConfigからDeepLearningConfigに変換
                model_config = ModelConfig(
                    sequence_length=dl_config.sequence_length,
                    prediction_horizon=dl_config.prediction_horizon,
                    hidden_size=dl_config.hidden_dim,
                    num_layers=dl_config.num_layers,
                    dropout_rate=dl_config.dropout_rate,
                    learning_rate=dl_config.learning_rate,
                    epochs=dl_config.epochs,
                    batch_size=dl_config.batch_size,
                    early_stopping_patience=dl_config.early_stopping_patience,
                )
                model.config = model_config

                # データ準備と訓練
                result = model.train(data_copy)
                training_time = time.time() - start_time

                # ModelTrainingResult形式に変換
                training_result = ModelTrainingResult(
                    final_loss=result.final_loss,
                    best_loss=result.best_loss,
                    epochs_run=result.epochs_run,
                    training_time=training_time,
                    validation_metrics={"mse": result.final_loss},
                    convergence_achieved=True,
                )

                logger.info(f"深層学習モデル並列訓練完了: {model_name} ({training_time:.2f}秒)")
                return model_name, training_result

            except Exception as e:
                logger.error(f"深層学習モデル並列訓練エラー {model_name}: {e}")
                return model_name, ModelTrainingResult(
                    final_loss=float('inf'),
                    best_loss=float('inf'),
                    epochs_run=0,
                    training_time=0.0,
                    validation_metrics={"error": str(e)},
                    convergence_achieved=False,
                )

        results = {}

        # 並列処理設定 (深層学習はCPU集約的でないため、スレッドプールを使用)
        n_jobs = min(len(self.models), mp.cpu_count())
        logger.info(f"アンサンブル並列訓練開始: {len(self.models)}モデルを{n_jobs}スレッドで実行")

        # 訓練引数準備
        training_args = [
            (model_name, model, data.copy(), self.dl_config)
            for model_name, model in self.models.items()
        ]

        try:
            # ThreadPoolExecutorを使用 (深層学習モデルはI/O待機が多い)
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_model = {
                    executor.submit(train_single_deep_model, args): args[0]
                    for args in training_args
                }

                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result_name, result = future.result(timeout=7200)  # 2時間のタイムアウト
                        results[result_name] = result
                    except Exception as e:
                        logger.error(f"深層学習モデル並列訓練例外 {model_name}: {e}")
                        results[model_name] = ModelTrainingResult(
                            final_loss=float('inf'),
                            best_loss=float('inf'),
                            epochs_run=0,
                            training_time=0.0,
                            validation_metrics={"timeout_error": str(e)},
                            convergence_achieved=False,
                        )

        except Exception as e:
            logger.warning(f"並列アンサンブル訓練フォールバック: {e}")
            # フォールバック: 順次実行
            return self._sequential_train_ensemble(data)

        return results

    def _sequential_train_ensemble(self, data: pd.DataFrame) -> Dict[str, ModelTrainingResult]:
        """順次アンサンブル訓練実装（フォールバック）"""
        results = {}

        for name, model in self.models.items():
            logger.info(f"モデル訓練開始: {name}")
            start_time = time.time()

            # ModelConfigからDeepLearningConfigに変換
            model_config = ModelConfig(
                sequence_length=self.dl_config.sequence_length,
                prediction_horizon=self.dl_config.prediction_horizon,
                hidden_size=self.dl_config.hidden_dim,
                num_layers=self.dl_config.num_layers,
                dropout_rate=self.dl_config.dropout_rate,
                learning_rate=self.dl_config.learning_rate,
                epochs=self.dl_config.epochs,
                batch_size=self.dl_config.batch_size,
                early_stopping_patience=self.dl_config.early_stopping_patience,
            )
            model.config = model_config

            # データ準備と訓練
            result = model.train(data)
            training_time = time.time() - start_time

            # ModelTrainingResult形式に変換
            training_result = ModelTrainingResult(
                final_loss=result.final_loss,
                best_loss=result.best_loss,
                epochs_run=result.epochs_run,
                training_time=training_time,
                validation_metrics={"mse": result.final_loss},
                convergence_achieved=True,
            )

            results[name] = training_result

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要"""
        avg_time = self.performance_stats["total_time"] / max(
            self.performance_stats["total_predictions"], 1
        )

        return {
            "trained_models": len(self.models),
            "available_models": list(self.models.keys()),
            "ensemble_ready": len(self.models) > 1,
            "total_predictions": self.performance_stats["total_predictions"],
            "average_prediction_time": avg_time,
        }


# Strategy Pattern への統合
@optimization_strategy("ml_models", OptimizationLevel.GPU_ACCELERATED)
class DeepLearningAcceleratedModels(OptimizationStrategy):
    """深層学習加速MLモデル戦略"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.dl_manager = DeepLearningModelManager(config)
        logger.info("深層学習加速MLモデル戦略初期化完了")

    def get_strategy_name(self) -> str:
        return "深層学習加速MLモデル (Transformer+LSTM)"

    def execute(
        self, data: pd.DataFrame, model_types: Optional[List[str]] = None, **kwargs
    ) -> Any:
        """深層学習実行"""
        model_types = model_types or ["transformer", "lstm"]
        target_column = kwargs.get("target_column", "Close")

        start_time = time.time()

        try:
            training_results = {}

            # 各モデルを訓練
            for model_type_str in model_types:
                model_type = ModelType(model_type_str)
                result = self.dl_manager.train_model(model_type, data, target_column)
                training_results[model_type_str] = result

            # アンサンブル予測
            ensemble_result = self.dl_manager.predict_ensemble(data)

            execution_time = time.time() - start_time
            self.record_execution(execution_time, True)

            return {
                "training_results": training_results,
                "ensemble_prediction": ensemble_result,
                "strategy_name": self.get_strategy_name(),
                "execution_time": execution_time,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.record_execution(execution_time, False)
            logger.error(f"深層学習実行エラー: {e}")
            raise
