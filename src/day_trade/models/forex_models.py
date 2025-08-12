#!/usr/bin/env python3
"""
Forex Market Database Models
外国為替市場データのデータベースモデル
"""

from decimal import Decimal
from typing import Optional

from sqlalchemy import Column, DateTime, Index, Integer, Numeric, String
from sqlalchemy.sql import func

from .base import Base


class ForexPrice(Base):
    """外国為替価格データモデル"""

    __tablename__ = "forex_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(10), nullable=False, index=True)  # EUR/USD, GBP/JPY, etc.
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # 価格データ
    bid_price = Column(Numeric(precision=12, scale=6), nullable=False)
    ask_price = Column(Numeric(precision=12, scale=6), nullable=False)
    spread = Column(Numeric(precision=12, scale=6), nullable=True)

    # ボリューム・流動性データ
    volume = Column(Numeric(precision=20, scale=2), nullable=True)

    # メタデータ
    source = Column(String(50), nullable=False, default="unknown")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_forex_pair_timestamp", "pair", "timestamp"),
        Index("idx_forex_timestamp_desc", "timestamp", postgresql_using="btree"),
        Index("idx_forex_pair_created", "pair", "created_at"),
    )

    def __repr__(self):
        return f"<ForexPrice(pair={self.pair}, bid={self.bid_price}, ask={self.ask_price}, time={self.timestamp})>"

    @property
    def mid_price(self) -> Decimal:
        """中間価格計算"""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread_pips(self) -> Optional[Decimal]:
        """スプレッド（pips）"""
        if not self.spread:
            return None

        # JPY通貨ペアは小数点2位、その他は4位
        if "JPY" in self.pair:
            return self.spread * 100  # JPYは2桁
        else:
            return self.spread * 10000  # その他は4桁


class ForexDailyStats(Base):
    """外国為替日次統計データ"""

    __tablename__ = "forex_daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(10), nullable=False, index=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)

    # OHLC データ
    open_price = Column(Numeric(precision=12, scale=6), nullable=False)
    high_price = Column(Numeric(precision=12, scale=6), nullable=False)
    low_price = Column(Numeric(precision=12, scale=6), nullable=False)
    close_price = Column(Numeric(precision=12, scale=6), nullable=False)

    # 統計データ
    avg_spread = Column(Numeric(precision=12, scale=6), nullable=True)
    total_volume = Column(Numeric(precision=20, scale=2), nullable=True)
    tick_count = Column(Integer, nullable=True)
    volatility = Column(Numeric(precision=8, scale=6), nullable=True)  # 日次ボラティリティ

    # 価格変動データ
    price_change = Column(Numeric(precision=12, scale=6), nullable=True)
    price_change_percent = Column(Numeric(precision=8, scale=4), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_forex_daily_pair_date", "pair", "date"),
        Index("idx_forex_daily_date_desc", "date", postgresql_using="btree"),
    )

    def __repr__(self):
        return f"<ForexDailyStats(pair={self.pair}, date={self.date}, close={self.close_price})>"


class ForexCorrelation(Base):
    """通貨ペア相関データ"""

    __tablename__ = "forex_correlations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair1 = Column(String(10), nullable=False, index=True)
    pair2 = Column(String(10), nullable=False, index=True)

    # 相関係数データ
    correlation_1h = Column(Numeric(precision=8, scale=6), nullable=True)
    correlation_1d = Column(Numeric(precision=8, scale=6), nullable=True)
    correlation_1w = Column(Numeric(precision=8, scale=6), nullable=True)
    correlation_1m = Column(Numeric(precision=8, scale=6), nullable=True)

    # メタデータ
    last_updated = Column(DateTime(timezone=True), nullable=False)
    calculation_method = Column(String(50), default="pearson")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_forex_corr_pairs", "pair1", "pair2"),
        Index("idx_forex_corr_updated", "last_updated"),
    )

    def __repr__(self):
        return f"<ForexCorrelation(pair1={self.pair1}, pair2={self.pair2}, corr_1d={self.correlation_1d})>"
