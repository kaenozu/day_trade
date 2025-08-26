#!/usr/bin/env python3
"""
深層学習統合システム - Transformerコンポーネント
Phase F: 次世代機能拡張フェーズ

Transformerモデルの構築とNumPy実装
"""

import warnings
from typing import Any, Dict, Tuple

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

from .utils import softmax

try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class TransformerBuilderMixin:
    """Transformerモデル構築用Mixinクラス"""

    def build_pytorch_transformer(
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

    def build_numpy_transformer(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
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

    def numpy_forward_pass(self, X: np.ndarray) -> np.ndarray:
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
        attention_weights = softmax(attention_scores)
        attention_output = np.matmul(attention_weights, V)

        # 出力層
        final_output = attention_output[:, -1, :]  # 最後のタイムステップ
        predictions = np.dot(final_output, weights["output_weights"])

        return predictions

    def numpy_backward_pass(
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

    def build_fallback_model(self, X: np.ndarray) -> Dict[str, Any]:
        """
        NumPyフォールバックモデル構築

        Issue #696対応: 重要警告 - これは簡易線形回帰実装です
        深層学習モデルとしての性能は期待できません。
        本番環境では必ずPyTorchをインストールしてください。
        """
        logger.warning(
            "=== NumPy フォールバック実行中 ===\n"
            "簡易線形回帰モデルを使用中です。\n"
            "・深層学習の性能は期待できません\n"
            "・複雑なパターン学習は不可能です\n"
            "・本番環境での使用は推奨されません\n"
            "PyTorchインストール推奨: pip install torch\n"
            "================================"
        )

        # 線形回帰フォールバック（最小二乗法）
        X_flattened = X.reshape(X.shape[0], -1)

        try:
            weights = np.linalg.lstsq(X_flattened, np.zeros((X.shape[0], 1)), rcond=None)[0]

            # Issue #696対応: フォールバック実装であることを明記
            model = {
                "fallback_weights": weights,
                "model_type": "numpy_linear_regression",
                "performance_warning": "深層学習性能は期待できません",
                "recommendation": "PyTorchインストールを強く推奨"
            }

            return model

        except Exception as e:
            logger.error(f"フォールバックモデル構築エラー: {e}")
            raise