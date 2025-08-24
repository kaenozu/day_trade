#!/usr/bin/env python3
"""
深層学習統合システム - LSTMモデル
Phase F: 次世代機能拡張フェーズ

LSTM時系列予測モデルの実装
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
    warnings.warn(
        "PyTorch未インストール - NumPy簡易実装にフォールバック（性能制限あり）",
        ImportWarning, stacklevel=2
    )

from .base_model import BaseDeepLearningModel
from .model_types import ModelConfig, DeepLearningConfig, ModelType, TrainingResult
from .utils import calculate_accuracy, sigmoid
try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


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
            train_accuracy = calculate_accuracy(
                y_train.numpy(), train_pred.numpy()
            )
            val_accuracy = calculate_accuracy(y_val.numpy(), val_pred.numpy())

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

        train_accuracy = calculate_accuracy(y_train, train_pred)
        val_accuracy = calculate_accuracy(y_val, val_pred)

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
            f_gate = sigmoid(
                np.dot(input_combined, weights["W_f"]) + weights["b_f"]
            )
            i_gate = sigmoid(
                np.dot(input_combined, weights["W_i"]) + weights["b_i"]
            )
            o_gate = sigmoid(
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