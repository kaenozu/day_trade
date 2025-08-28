#!/usr/bin/env python3
"""
ハイブリッドLSTM-Transformerモデル設定
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..deep_learning_models import ModelConfig, TrainingResult


@dataclass
class HybridModelConfig(ModelConfig):
    """ハイブリッドLSTM-Transformerモデル設定"""

    # LSTM設定
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True

    # Transformer設定
    transformer_d_model: int = 128
    transformer_num_heads: int = 8
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 512

    # Cross-Attention設定
    cross_attention_heads: int = 4
    cross_attention_dim: int = 128

    # 融合設定
    fusion_dropout: float = 0.3
    fusion_hidden_dims: List[int] = None

    # 最適化設定
    weight_decay: float = 1e-4
    gradient_clip_value: float = 1.0
    lr_scheduler_factor: float = 0.8
    lr_scheduler_patience: int = 5

    # 予測設定
    use_uncertainty_estimation: bool = True
    monte_carlo_samples: int = 50

    def __post_init__(self):
        if self.fusion_hidden_dims is None:
            self.fusion_hidden_dims = [512, 256, 128]


@dataclass
class HybridTrainingResult(TrainingResult):
    """ハイブリッドモデル訓練結果"""

    lstm_loss: float = 0.0
    transformer_loss: float = 0.0
    fusion_loss: float = 0.0
    attention_weights: Optional[Dict[str, float]] = None
    computational_efficiency: float = 0.0  # FLOPS/second