"""
シグナル生成モジュール

売買シグナル生成システムの統合インターフェース
バックワード互換性を維持しながら分割されたモジュールを提供
"""

# 基本型の公開
from .types import SignalStrength, SignalType, TradingSignal

# 設定管理の公開
from .config import SignalRulesConfig, get_shared_config

# ルール基底クラスの公開
from .base import SignalRule

# テクニカル指標ルールの公開
from .technical_rules import (
    BollingerBandRule,
    MACDCrossoverRule,
    MACDDeathCrossRule,
    RSIOverboughtRule,
    RSIOversoldRule,
    VolumeSpikeBuyRule,
)

# パターンベースルールの公開
from .pattern_rules import DeadCrossRule, GoldenCrossRule, PatternBreakoutRule

# 検証機能の公開
from .validators import DataValidator, SignalValidator

# メインクラスの公開
from .generator import TradingSignalGenerator

# 元のファイルと同じ名前でアクセスできるように
__all__ = [
    # 基本型
    "SignalType",
    "SignalStrength", 
    "TradingSignal",
    # 設定管理
    "SignalRulesConfig",
    "get_shared_config",
    # ルール基底クラス
    "SignalRule",
    # テクニカル指標ルール
    "RSIOversoldRule",
    "RSIOverboughtRule",
    "MACDCrossoverRule",
    "MACDDeathCrossRule",
    "BollingerBandRule",
    "VolumeSpikeBuyRule",
    # パターンベースルール
    "PatternBreakoutRule",
    "GoldenCrossRule",
    "DeadCrossRule",
    # 検証機能
    "SignalValidator",
    "DataValidator",
    # メインクラス
    "TradingSignalGenerator",
]

# バックワード互換性のための追加エクスポート
# 元のファイルで使用されていた内部関数も公開
_get_shared_config = get_shared_config  # 内部関数としてのアクセス

# バージョン情報
__version__ = "2.0.0"

# モジュール情報
__author__ = "Day Trade System"
__description__ = "売買シグナル生成システム - 機能別分割版"