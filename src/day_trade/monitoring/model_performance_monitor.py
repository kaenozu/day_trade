#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - 後方互換性インターフェース
元のmodel_performance_monitor.pyファイルの後方互換性を保つためのインターフェース

このファイルはmodularizedされた新しい構造への移行期間中に使用されます。
新しいコードでは model_performance パッケージから直接インポートしてください。
"""

import warnings

# 新しいモジュラー構造からすべてをインポート
from .model_performance import *

# 非推奨警告を発行
warnings.warn(
    "model_performance_monitor.pyは非推奨です。"
    "新しいコードでは 'from .model_performance import ModelPerformanceMonitor' "
    "を使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# ロギング設定（後方互換性）
import logging
logger = logging.getLogger(__name__)