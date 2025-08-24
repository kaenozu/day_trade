#!/usr/bin/env python3
"""
深層学習統合システム - Transformerモデル
Phase F: 次世代機能拡張フェーズ

Transformer時系列予測モデルの実装
"""

import time
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np

# Issue #696対応: PyTorch可用性チェック
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_model import BaseDeepLearningModel
from .transformer_components import TransformerBuilderMixin
from .model_types import ModelConfig, DeepLearningConfig, ModelType, TrainingResult
from .utils import calculate_accuracy

try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class TransformerModel(BaseDeepLearningModel, TransformerBuilderMixin):
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
            if TORCH_AVAILABLE:
                return self.build_pytorch_transformer(input_shape)
        except Exception as e:
            logger.warning(f"PyTorch Transformerモデル構築エラー: {e}")

        # フォールバック: NumPy実装
        logger.info("PyTorch未利用のため、NumPy Transformerを使用")
        return self.build_numpy_transformer(input_shape)

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

        # 精度計算
        with torch.no_grad():
            train_pred = self.model(X_train)
            val_pred = self.model(X_val)

            train_accuracy = calculate_accuracy(
                y_train.numpy(), train_pred.numpy()
            )
            val_accuracy = calculate_accuracy(y_val.numpy(), val_pred.numpy())

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
            # フォワードパス
            train_pred = self.numpy_forward_pass(X_train)
            val_pred = self.numpy_forward_pass(X_val)

            # 損失計算
            train_loss = np.mean((train_pred - y_train) ** 2)
            val_loss = np.mean((val_pred - y_val) ** 2)

            # パラメータ更新
            self.numpy_backward_pass(X_train, y_train, train_pred)

            if val_loss < best_loss:
                best_loss = val_loss

        training_time = time.time() - start_time

        train_accuracy = calculate_accuracy(y_train, train_pred)
        val_accuracy = calculate_accuracy(y_val, val_pred)

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

    def _train_fallback_model(
        self, X: np.ndarray, y: np.ndarray, start_time: float
    ) -> TrainingResult:
        """フォールバックモデル訓練"""
        try:
            self.model = self.build_fallback_model(X)
            
            # 簡易線形予測
            X_flattened = X.reshape(X.shape[0], -1)
            predictions = np.dot(X_flattened, self.model["fallback_weights"])

            loss = np.mean((predictions - y) ** 2)
            accuracy = calculate_accuracy(y, predictions)

            logger.warning(f"フォールバックモデル訓練完了 - 精度: {accuracy:.3f}（参考値）")

            return TrainingResult(
                model_type=ModelType.TRANSFORMER,
                training_loss=loss,
                validation_loss=loss,
                training_accuracy=accuracy,
                validation_accuracy=accuracy,
                training_time=time.time() - start_time,
                epochs_trained=1,
                model_parameters=len(self.model["fallback_weights"].flatten()),
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
                return self.numpy_forward_pass(X)

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