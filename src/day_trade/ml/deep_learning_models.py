#!/usr/bin/env python3
"""
深層学習統合システム - バックワード互換性インポート
Phase F: 次世代機能拡張フェーズ

Transformer・LSTM・CNN による高精度予測システム

このファイルは deep_learning パッケージへのバックワード互換性を提供します。
実際の実装は deep_learning/ サブパッケージに分割されています。
"""

import warnings

# 新しいパッケージからすべてをインポート
from .deep_learning import *

# バックワード互換性の警告
warnings.warn(
    "deep_learning_models.py からの直接インポートは非推奨です。"
    "新しいコードでは day_trade.ml.deep_learning パッケージを使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# 使用方法の情報をログ出力
from .utils.logging_config import get_context_logger
logger = get_context_logger(__name__)

logger.info(
    "深層学習統合システム初期化完了 (バックワード互換性モード)\n"
    "推奨される新しいインポート方法:\n"
    "  from day_trade.ml.deep_learning import ModelConfig, TransformerModel, LSTMModel\n"
    "  from day_trade.ml.deep_learning import DeepLearningModelManager\n"
    "機能別分割により以下が改善されました:\n"
    "  - コードの保守性向上\n"
    "  - モジュール間の依存関係明確化\n"
    "  - テスト容易性向上\n"
    "  - 個別機能のインポート可能"
)