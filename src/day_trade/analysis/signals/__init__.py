"""
Signals モジュール - 売買シグナル生成システム

元の signals.py ファイルから分割されたモジュールです。
後方互換性を保つため、すべてのクラス・関数を再エクスポートします。
"""

# 基本型とデータクラス
from .types import (
    SignalStrength,
    SignalType,
    TradingSignal,
)

# 設定管理
from .config import (
    SignalRulesConfig,
    _get_shared_config,
)

# 基底ルールクラス
from .base_rules import (
    SignalRule,
)

# 各種ルールクラス
from .band_rules import (
    BollingerBandRule,
)
from .macd_rules import (
    MACDCrossoverRule,
    MACDDeathCrossRule,
)
from .pattern_rules import (
    DeadCrossRule,
    GoldenCrossRule,
    PatternBreakoutRule,
)
from .rsi_rules import (
    RSIOversoldRule,
    RSIOverboughtRule,
)
from .volume_rules import (
    VolumeSpikeBuyRule,
)

# メインジェネレータクラス
from .generator import (
    TradingSignalGenerator,
)

# 後方互換性のため、すべてをエクスポート
__all__ = [
    # 基本型
    "SignalStrength",
    "SignalType", 
    "TradingSignal",
    # 設定
    "SignalRulesConfig",
    "_get_shared_config",
    # ルールクラス
    "SignalRule",
    "BollingerBandRule",
    "MACDCrossoverRule",
    "MACDDeathCrossRule",
    "DeadCrossRule",
    "GoldenCrossRule",
    "PatternBreakoutRule",
    "RSIOversoldRule",
    "RSIOverboughtRule",
    "VolumeSpikeBuyRule",
    # メインクラス
    "TradingSignalGenerator",
]