#!/usr/bin/env python3
"""
デイトレード推奨データクラス

DayTradingRecommendationデータクラスと関連ユーティリティ関数を提供します。
"""

from dataclasses import dataclass
from .enums import DayTradingSignal


@dataclass
class DayTradingRecommendation:
    """デイトレード推奨データ"""
    symbol: str
    name: str
    signal: DayTradingSignal
    entry_timing: str          # エントリータイミング
    target_profit: float       # 目標利益率(%)
    stop_loss: float          # 損切りライン(%)
    holding_time: str         # 推奨保有時間
    confidence: float         # 信頼度
    risk_level: str          # リスクレベル
    volume_trend: str        # 出来高動向
    price_momentum: str      # 価格モメンタム
    intraday_volatility: float  # 日中ボラティリティ
    market_timing_score: float  # 市場タイミングスコア

    def __post_init__(self):
        """データクラス初期化後の検証"""
        if self.target_profit < 0:
            raise ValueError("目標利益率は0以上である必要があります")
        if self.stop_loss < 0:
            raise ValueError("損切りラインは0以上である必要があります")
        if not (0 <= self.confidence <= 100):
            raise ValueError("信頼度は0-100の範囲である必要があります")
        if not (0 <= self.market_timing_score <= 100):
            raise ValueError("市場タイミングスコアは0-100の範囲である必要があります")

    def is_buy_signal(self) -> bool:
        """買いシグナルかどうか判定"""
        return self.signal in [DayTradingSignal.STRONG_BUY, DayTradingSignal.BUY]

    def is_sell_signal(self) -> bool:
        """売りシグナルかどうか判定"""
        return self.signal in [DayTradingSignal.STRONG_SELL, DayTradingSignal.SELL]

    def is_high_confidence(self) -> bool:
        """高信頼度かどうか判定（80%以上）"""
        return self.confidence >= 80.0

    def get_risk_color(self) -> str:
        """リスクレベルに応じた色コード"""
        risk_colors = {
            "低": "green",
            "[低リスク]": "green",
            "中": "orange",
            "[中リスク]": "orange",
            "高": "red",
            "[高リスク]": "red"
        }
        return risk_colors.get(self.risk_level, "gray")

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "signal": self.signal.value,
            "entry_timing": self.entry_timing,
            "target_profit": self.target_profit,
            "stop_loss": self.stop_loss,
            "holding_time": self.holding_time,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "volume_trend": self.volume_trend,
            "price_momentum": self.price_momentum,
            "intraday_volatility": self.intraday_volatility,
            "market_timing_score": self.market_timing_score
        }

    def __str__(self) -> str:
        """文字列表現"""
        return (
            f"{self.symbol} ({self.name}): {self.signal.value} "
            f"[信頼度: {self.confidence:.1f}%] "
            f"[スコア: {self.market_timing_score:.1f}]"
        )