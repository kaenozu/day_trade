#!/usr/bin/env python3
"""
Cross-Attention融合メカニズム - Issue #698対応: テンソル操作最適化
"""

from typing import Optional, Tuple

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# Issue #696対応: PyTorch可用性チェック
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    PYTORCH_AVAILABLE = True
    logger.info("PyTorch利用可能 - 完全なCrossAttentionLayerを使用")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch未インストール - CrossAttentionLayerは利用できません")

    # フォールバック用のダミークラス
    class nn:
        class Module:
            pass

        class Linear:
            pass

        class LayerNorm:
            pass

        class Dropout:
            pass

    class F:
        @staticmethod
        def softmax(x, dim=-1):
            return x


if PYTORCH_AVAILABLE:

    class CrossAttentionLayer(nn.Module):
        """Cross-Attention融合メカニズム - Issue #698対応: テンソル操作最適化"""

        def __init__(
            self,
            lstm_dim: int,
            transformer_dim: int,
            attention_heads: int = 4,
            attention_dim: int = 128,
        ):
            super().__init__()
            self.attention_heads = attention_heads
            self.attention_dim = attention_dim
            self.head_dim = attention_dim // attention_heads
            self.scale = self.head_dim ** -0.5

            # Projection layers
            self.lstm_proj = nn.Linear(lstm_dim, attention_dim)
            self.transformer_proj = nn.Linear(transformer_dim, attention_dim)

            # Issue #698対応: 単一リニア層で Q, K, V を一度に計算（効率化）
            self.qkv_proj = nn.Linear(attention_dim, attention_dim * 3)
            self.output_proj = nn.Linear(attention_dim, attention_dim)

            # Layer normalization
            self.layer_norm = nn.LayerNorm(attention_dim)
            self.dropout = nn.Dropout(0.1)

            # Issue #698対応: 事前計算された形状保存
            self._cached_batch_size = None
            self._cached_combined_shape = None

        def forward(
            self, lstm_features: torch.Tensor, transformer_features: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Cross-Attention計算 - Issue #698対応: テンソル操作最適化

            Args:
                lstm_features: LSTM出力 [batch, lstm_dim]
                transformer_features: Transformer出力 [batch, transformer_dim]

            Returns:
                fused_features: 融合特徴量 [batch, attention_dim]
                attention_weights: アテンション重み [batch, heads, 2, 2]
            """
            batch_size = lstm_features.size(0)

            # 特徴量投影
            lstm_proj = self.lstm_proj(lstm_features)  # [batch, attention_dim]
            transformer_proj = self.transformer_proj(transformer_features)  # [batch, attention_dim]

            # Issue #698対応: メモリ効率的な結合（stackよりcat+reshape）
            combined = torch.cat(
                [lstm_proj.unsqueeze(1), transformer_proj.unsqueeze(1)], dim=1
            )  # [batch, 2, attention_dim]

            # Issue #698対応: 単一計算で Q, K, V を取得（メモリ効率化）
            seq_len = 2  # LSTM + Transformer = 2
            qkv = self.qkv_proj(combined)  # [batch, 2, attention_dim * 3]

            # reshape で Q, K, V を分離（transpose回数削減）
            qkv = qkv.view(batch_size, seq_len, 3, self.attention_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
            Q, K, V = qkv[0], qkv[1], qkv[2]

            # Issue #698対応: 効率的なattention計算
            # Q: [batch, heads, seq_len, head_dim]
            # K: [batch, heads, seq_len, head_dim]
            # V: [batch, heads, seq_len, head_dim]

            # Attention scores計算（transpose削減）
            attention_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) * self.scale
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # Attention適用（einsumで効率化）
            attended = torch.einsum('bhqk,bhkd->bhqd', attention_weights, V)

            # Issue #698対応: contiguous()を明示的に呼び出してメモリ効率化
            attended = attended.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, heads, head_dim]
            attended = attended.view(batch_size, seq_len, self.attention_dim)

            # Output projection
            output = self.output_proj(attended)

            # Residual connection + Layer normalization
            output = self.layer_norm(output + combined)

            # Issue #698対応: 最終融合（メモリ効率的な平均計算）
            fused_features = output.sum(dim=1) * 0.5  # [batch, attention_dim] (2で割るのを乗算で高速化)

            # Attention weights for analysis（メモリ効率化）
            avg_attention_weights = attention_weights.mean(dim=1)  # [batch, 2, 2]

            return fused_features, avg_attention_weights

else:

    class CrossAttentionLayer:
        """PyTorch未利用時のダミークラス"""

        def __init__(self, lstm_dim: int, transformer_dim: int, attention_heads: int = 4, attention_dim: int = 128):
            logger.warning("PyTorchが利用できません - CrossAttentionLayerは無効化されています")

        def forward(self, lstm_features, transformer_features):
            logger.warning("CrossAttentionLayer.forward() - PyTorchが必要です")
            return None, None