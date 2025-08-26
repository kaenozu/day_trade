#!/usr/bin/env python3
"""
Advanced ML Engine Models Module

PyTorchベースのハイブリッドモデル定義
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd

# PyTorch遅延インポート - メモリ効率化
PYTORCH_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, TensorDataset

        # PyTorch決定論的実行設定
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        PYTORCH_AVAILABLE = True
except ImportError:
    pass

from ..utils.logging_config import get_context_logger
from .config import ModelConfig

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


if PYTORCH_AVAILABLE:

    class LSTMTransformerHybrid(nn.Module):
        """ハイブリッド LSTM-Transformer 予測モデル"""

        def __init__(self, config: ModelConfig):
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch が必要です")

            super().__init__()
            self.config = config

            # LSTM分岐
            self.lstm = nn.LSTM(
                input_size=config.num_features,
                hidden_size=config.lstm_hidden_size,
                num_layers=config.lstm_num_layers,
                dropout=config.lstm_dropout,
                batch_first=True,
                bidirectional=True,
            )

            # Transformer分岐
            self.positional_encoding = PositionalEncoding(config.transformer_d_model)
            self.input_projection = nn.Linear(
                config.num_features, config.transformer_d_model
            )

            transformer_layer = nn.TransformerEncoderLayer(
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                dim_feedforward=config.transformer_dim_feedforward,
                dropout=config.transformer_dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                transformer_layer, config.transformer_num_layers
            )

            # Feature Fusion Layer
            lstm_output_size = config.lstm_hidden_size * 2  # 双方向
            fusion_input_size = lstm_output_size + config.transformer_d_model

            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # 予測ヘッド
            self.prediction_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, config.prediction_horizon),
            )

            # 信頼度推定ヘッド
            self.confidence_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, config.prediction_horizon),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """順伝播"""
            batch_size, seq_len, features = x.shape

            # LSTM分岐
            lstm_out, _ = self.lstm(x)
            lstm_features = lstm_out[:, -1, :]  # 最後の出力を使用

            # Transformer分岐
            transformer_input = self.input_projection(x)
            transformer_input = self.positional_encoding(transformer_input)
            transformer_out = self.transformer(transformer_input)
            transformer_features = transformer_out.mean(dim=1)  # グローバル平均プーリング

            # Feature Fusion
            fused_features = torch.cat([lstm_features, transformer_features], dim=1)
            fused_output = self.fusion_layer(fused_features)

            # 予測と信頼度
            predictions = self.prediction_head(fused_output)
            confidence = self.confidence_head(fused_output)

            return predictions, confidence


if PYTORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """位置エンコーディング"""

        def __init__(self, d_model: int, max_len: int = 2000):
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch が必要です")

            super().__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """位置エンコーディング追加"""
            return x + self.pe[: x.size(0), :].transpose(0, 1)


# PyTorchが利用できない場合のフォールバック
if not PYTORCH_AVAILABLE:
    logger.warning("PyTorch 未インストール - モデルクラスは利用できません")

    class LSTMTransformerHybrid:
        """フォールバック版 - 基本機能のみ"""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")

    class PositionalEncoding:
        """フォールバック版 - 基本機能のみ"""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")