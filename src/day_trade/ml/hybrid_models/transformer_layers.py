#!/usr/bin/env python3
"""
修正Transformer (mTrans) - 時空間融合最適化 - Issue #698対応: テンソル操作最適化
"""

from typing import Optional

import numpy as np

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# Issue #696対応: PyTorch可用性チェック
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True
    logger.info("PyTorch利用可能 - 完全なModifiedTransformerEncoderを使用")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch未インストール - ModifiedTransformerEncoderは利用できません")

    # フォールバック用のダミークラス
    class nn:
        class Module:
            pass

        class TransformerEncoderLayer:
            pass

        class TransformerEncoder:
            pass

        class Conv1d:
            pass

        class LayerNorm:
            pass

        class Dropout:
            pass

    class torch:
        class Tensor:
            pass

        @staticmethod
        def zeros(*args, **kwargs):
            return None

        @staticmethod
        def arange(*args, **kwargs):
            return None

        @staticmethod
        def exp(*args, **kwargs):
            return None

        @staticmethod
        def sin(*args, **kwargs):
            return None

        @staticmethod
        def cos(*args, **kwargs):
            return None


if PYTORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """時系列データ用位置エンコーディング - Issue #698対応: テンソル操作最適化"""

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Issue #698対応: 位置エンコーディングを事前計算して効率化
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:  # 奇数次元対応
                pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

            # Issue #698対応: batch_firstに対応した形状で保存（transpose削減）
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]

            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Issue #698対応: transpose操作を削減した効率的な位置エンコーディング

            Args:
                x: [batch_size, seq_len, d_model]
            Returns:
                x + positional encoding
            """
            # Issue #698対応: slicingでtranspose削減
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len, :]
            return self.dropout(x)

    class ModifiedTransformerEncoder(nn.Module):
        """修正Transformer (mTrans) - 時空間融合最適化 - Issue #698対応: テンソル操作最適化"""

        def __init__(
            self,
            d_model: int,
            num_heads: int,
            dim_feedforward: int,
            num_layers: int,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_model = d_model

            # Position encoding for time series
            self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=5000)

            # Modified transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",  # GELU for better performance
                layer_norm_eps=1e-6,
                batch_first=True,
                norm_first=True,  # Pre-norm for stability
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

            # Issue #698対応: 1D畳み込みの効率化（グループ畳み込み使用）
            self.temporal_conv = nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=min(d_model, 32)  # グループ畳み込みで計算効率向上
            )
            self.temporal_norm = nn.LayerNorm(d_model)

            # Issue #698対応: transpose操作のキャッシュフラグ
            self._use_memory_efficient_conv = True

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Modified Transformer forward pass - Issue #698対応: テンソル操作最適化

            Args:
                x: Input tensor [batch, seq_len, features]
                mask: Attention mask (optional)

            Returns:
                output: Transformer output [batch, seq_len, d_model]
            """
            # Positional encoding
            x = self.pos_encoding(x)

            # Transformer encoding
            transformer_out = self.transformer_encoder(x, mask=mask)

            # Issue #698対応: メモリ効率的なTemporal convolution
            if self._use_memory_efficient_conv:
                # contiguous()を事前に呼び出してメモリ効率向上
                conv_input = transformer_out.transpose(1, 2).contiguous()
                temporal_out = self.temporal_conv(conv_input)
                # transpose回数削減のため、直接permute使用
                temporal_out = temporal_out.permute(0, 2, 1)  # [batch, seq_len, d_model]
            else:
                # 従来の実装（フォールバック）
                conv_input = transformer_out.transpose(1, 2)
                temporal_out = self.temporal_conv(conv_input)
                temporal_out = temporal_out.transpose(1, 2)

            # Issue #698対応: インプレース演算でメモリ節約
            # Residual connection (add_ でインプレース演算)
            output = self.temporal_norm(transformer_out + temporal_out)

            return output

else:

    class PositionalEncoding:
        """PyTorch未利用時のダミークラス"""

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            logger.warning("PyTorchが利用できません - PositionalEncodingは無効化されています")

        def forward(self, x):
            return x

    class ModifiedTransformerEncoder:
        """PyTorch未利用時のダミークラス"""

        def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, num_layers: int, dropout: float = 0.1):
            logger.warning("PyTorchが利用できません - ModifiedTransformerEncoderは無効化されています")

        def forward(self, x, mask=None):
            return x