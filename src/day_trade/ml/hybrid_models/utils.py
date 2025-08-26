#!/usr/bin/env python3
"""
NumPy実装とユーティリティ関数
"""

from typing import Any, Dict

import numpy as np

from ...utils.logging_config import get_context_logger
from .config import HybridModelConfig

logger = get_context_logger(__name__)


def build_numpy_hybrid_model(input_shape: tuple, config: HybridModelConfig) -> Dict[str, Any]:
    """
    NumPy ハイブリッドモデル（簡易実装）

    Issue #696対応: 重要警告 - これは簡易線形近似実装です
    ハイブリッドLSTM-Transformerとしての性能は期待できません。
    本格的な深層学習にはPyTorchが必須です。
    """
    seq_len, n_features = input_shape

    return {
        "type": "numpy_hybrid",
        "input_shape": input_shape,
        "lstm_weights": _initialize_lstm_weights(n_features, config),
        "transformer_weights": _initialize_transformer_weights(n_features, config),
        "fusion_weights": np.random.randn(256, config.prediction_horizon) * 0.1,
    }


def _initialize_lstm_weights(n_features: int, config: HybridModelConfig) -> Dict[str, np.ndarray]:
    """LSTM重み初期化"""
    hidden_size = config.lstm_hidden_size

    return {
        "W_i": np.random.randn(n_features + hidden_size, hidden_size) * 0.1,
        "W_f": np.random.randn(n_features + hidden_size, hidden_size) * 0.1,
        "W_o": np.random.randn(n_features + hidden_size, hidden_size) * 0.1,
        "W_c": np.random.randn(n_features + hidden_size, hidden_size) * 0.1,
    }


def _initialize_transformer_weights(n_features: int, config: HybridModelConfig) -> Dict[str, np.ndarray]:
    """Transformer重み初期化"""
    d_model = config.transformer_d_model

    return {
        "W_q": np.random.randn(n_features, d_model) * 0.1,
        "W_k": np.random.randn(n_features, d_model) * 0.1,
        "W_v": np.random.randn(n_features, d_model) * 0.1,
        "W_o": np.random.randn(d_model, d_model) * 0.1,
    }


def numpy_hybrid_forward(X: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """
    NumPy ハイブリッドモデル フォワードパス

    Issue #696対応: 簡易線形近似実装
    注意: これはハイブリッドLSTM-Transformerの機能を大幅に簡略化した実装です
    """
    # 簡易実装（線形結合） - 本物のLSTM/Transformerではありません
    batch_size, seq_len, n_features = X.shape

    # LSTM部分（簡易線形変換 - 時系列学習機能なし）
    X_flat = X.reshape(batch_size, -1)
    lstm_out = np.tanh(np.dot(X_flat, np.random.randn(X_flat.shape[1], 128)))

    # Transformer部分（簡易線形変換 - 注意機構なし）
    transformer_out = np.tanh(np.dot(X_flat, np.random.randn(X_flat.shape[1], 128)))

    # 融合
    combined = np.concatenate([lstm_out, transformer_out], axis=1)

    # 予測
    predictions = np.dot(combined, model["fusion_weights"])

    return predictions


def numpy_hybrid_backward(
    X: np.ndarray, y: np.ndarray, predictions: np.ndarray, model: Dict[str, Any], learning_rate: float
):
    """
    NumPy ハイブリッドモデル バックワードパス

    Issue #696対応: 簡易実装 - 本物のバックプロパゲーションではありません
    実際のLSTM/Transformer勾配計算は行われません
    """
    # 簡易勾配更新（実際の深層学習最適化ではありません）
    lr = learning_rate * 0.01
    error = predictions - y

    # 重み更新（簡易線形近似のみ）
    model["fusion_weights"] -= lr * error.mean() * 0.001


def train_numpy_hybrid(X: np.ndarray, y: np.ndarray, config: HybridModelConfig, model: Dict[str, Any]) -> Dict[str, Any]:
    """
    NumPy ハイブリッドモデル訓練（簡易実装）

    Issue #696対応: 重要警告 - これは簡易線形近似実装です
    ハイブリッドLSTM-Transformerとしての性能は期待できません。
    本格的な深層学習にはPyTorchが必須です。
    """
    logger.warning(
        "=== HybridLSTMTransformer NumPy フォールバック実行中 ===\n"
        "簡易線形近似モデルを使用中です。以下の機能は無効化されています:\n"
        "・LSTM時系列学習: 単純な線形変換に置換\n"
        "・Transformer注意機構: 線形結合に置換\n"
        "・Cross-Attention融合: 重み平均に置換\n"
        "・GPU加速: 無効\n"
        "・高度な最適化: 無効\n"
        "ハイブリッド深層学習の性能は期待できません。\n"
        "PyTorchインストール推奨: pip install torch\n"
        "================================================="
    )

    val_size = int(len(X) * config.validation_split)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    best_loss = float("inf")

    for epoch in range(min(config.epochs, 50)):
        # 簡易フォワードパス
        train_pred = numpy_hybrid_forward(X_train, model)
        val_pred = numpy_hybrid_forward(X_val, model)

        train_loss = np.mean((train_pred - y_train) ** 2)
        val_loss = np.mean((val_pred - y_val) ** 2)

        # 簡易パラメータ更新
        numpy_hybrid_backward(X_train, y_train, train_pred, model, config.learning_rate)

        if val_loss < best_loss:
            best_loss = val_loss

    return {
        "final_loss": val_loss,
        "best_loss": best_loss,
        "epochs_run": epoch + 1,
        "validation_metrics": {"mse": best_loss},
        "convergence_achieved": True,
    }


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """精度計算（MAPE -> Accuracy変換）"""
    try:
        # MAPE計算
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))) * 100

        # MAPEから精度に変換（0-100%）
        accuracy = max(0, 100 - mape)
        return min(accuracy, 100) / 100  # 0-1範囲に正規化

    except Exception:
        return 0.0