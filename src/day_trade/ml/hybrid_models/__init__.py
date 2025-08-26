#!/usr/bin/env python3
"""
ハイブリッドLSTM-Transformerモデルパッケージ

このパッケージは、高精度時系列予測のためのハイブリッドLSTM-Transformerモデルを提供します。
大きなファイルをモジュール単位に分割し、保守性と可読性を向上させています。

技術仕様:
- LSTM Branch: 長期依存関係学習
- Transformer Branch: Multi-Head Attention
- mTrans: 修正Transformer時空間融合
- Cross-Attention: ブランチ間情報統合
- MLP Prediction Head: 最終予測レイヤー

目標性能:
- 予測精度: 95%+ (現在89%から向上)
- MAE: 0.6以下
- RMSE: 0.8以下
- 推論レイテンシ: <100ms
"""

# 後方互換性を保つため、元のクラス・関数を再エクスポート
from .attention_layers import CrossAttentionLayer
from .config import HybridModelConfig, HybridTrainingResult
from .dataset import TimeSeriesDataset
from .engine import HybridLSTMTransformerEngine
from .factory import create_hybrid_model
from .pytorch_model import HybridLSTMTransformerModel
from .transformer_layers import ModifiedTransformerEncoder, PositionalEncoding
from .utils import build_numpy_hybrid_model, calculate_accuracy, numpy_hybrid_forward, train_numpy_hybrid

# モジュール公開リスト
__all__ = [
    # 設定クラス
    "HybridModelConfig",
    "HybridTrainingResult",
    # データセット
    "TimeSeriesDataset",
    # アテンション機構
    "CrossAttentionLayer",
    # Transformerコンポーネント
    "ModifiedTransformerEncoder",
    "PositionalEncoding",
    # PyTorchモデル
    "HybridLSTMTransformerModel",
    # エンジンクラス
    "HybridLSTMTransformerEngine",
    # ファクトリ関数
    "create_hybrid_model",
    # ユーティリティ関数
    "build_numpy_hybrid_model",
    "numpy_hybrid_forward",
    "train_numpy_hybrid",
    "calculate_accuracy",
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Day Trade System"
__description__ = "Hybrid LSTM-Transformer Model for Time Series Prediction"