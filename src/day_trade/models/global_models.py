#!/usr/bin/env python3
"""
Global Trading Engine Database Models
グローバル市場統合データモデル
"""

import enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.sql import func

from .base import Base


class MarketType(enum.Enum):
    """市場タイプ列挙"""

    STOCK = "stock"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    BOND = "bond"
    INDEX = "index"


class GlobalMarketData(Base):
    """グローバル市場統合データ"""

    __tablename__ = "global_market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 基本情報
    symbol = Column(String(50), nullable=False, index=True)
    market_type = Column(Enum(MarketType), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # 価格データ
    price = Column(Numeric(precision=20, scale=8), nullable=False)
    bid_price = Column(Numeric(precision=20, scale=8), nullable=True)
    ask_price = Column(Numeric(precision=20, scale=8), nullable=True)
    spread = Column(Numeric(precision=20, scale=8), nullable=True)

    # ボリューム・流動性
    volume = Column(Numeric(precision=30, scale=8), nullable=True)
    volume_24h = Column(Numeric(precision=30, scale=8), nullable=True)

    # 価格変動
    price_change_1h = Column(Numeric(precision=8, scale=4), nullable=True)
    price_change_24h = Column(Numeric(precision=8, scale=4), nullable=True)
    price_change_7d = Column(Numeric(precision=8, scale=4), nullable=True)

    # 市場固有データ（JSON）
    market_specific_data = Column(JSON, nullable=True)

    # メタデータ
    source = Column(String(50), nullable=False, default="api")
    exchange = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_global_symbol_type_time", "symbol", "market_type", "timestamp"),
        Index("idx_global_market_time", "market_type", "timestamp"),
        Index("idx_global_timestamp_desc", "timestamp", postgresql_using="btree"),
    )

    def __repr__(self):
        return f"<GlobalMarketData(symbol={self.symbol}, type={self.market_type.value}, price={self.price})>"


class CrossMarketCorrelation(Base):
    """クロスマーケット相関データ"""

    __tablename__ = "cross_market_correlations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 相関対象
    asset1_symbol = Column(String(50), nullable=False, index=True)
    asset1_market = Column(Enum(MarketType), nullable=False)
    asset2_symbol = Column(String(50), nullable=False, index=True)
    asset2_market = Column(Enum(MarketType), nullable=False)

    # 相関係数データ
    correlation_1h = Column(Numeric(precision=8, scale=6), nullable=True)
    correlation_4h = Column(Numeric(precision=8, scale=6), nullable=True)
    correlation_1d = Column(Numeric(precision=8, scale=6), nullable=True)
    correlation_1w = Column(Numeric(precision=8, scale=6), nullable=True)
    correlation_1m = Column(Numeric(precision=8, scale=6), nullable=True)

    # 統計データ
    sample_size = Column(Integer, nullable=True)
    confidence_level = Column(Numeric(precision=6, scale=4), nullable=True)
    p_value = Column(Numeric(precision=10, scale=8), nullable=True)

    # メタデータ
    calculation_method = Column(String(50), default="pearson")
    last_updated = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_cross_corr_assets", "asset1_symbol", "asset2_symbol"),
        Index("idx_cross_corr_markets", "asset1_market", "asset2_market"),
        Index("idx_cross_corr_updated", "last_updated"),
    )

    def __repr__(self):
        return (
            f"<CrossMarketCorrelation({self.asset1_symbol}[{self.asset1_market.value}] "
            f"vs {self.asset2_symbol}[{self.asset2_market.value}], corr_1d={self.correlation_1d})>"
        )


class GlobalMarketEvent(Base):
    """グローバル市場イベント"""

    __tablename__ = "global_market_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # イベント基本情報
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    event_type = Column(
        String(100), nullable=False, index=True
    )  # earnings, fed_meeting, news, etc.

    # 影響範囲
    affected_markets = Column(JSON, nullable=True)  # ["stock", "forex", "crypto"]
    affected_symbols = Column(JSON, nullable=True)  # ["BTCUSDT", "EUR/USD", "AAPL"]

    # 重要度・インパクト
    importance_level = Column(Integer, nullable=False, default=1, index=True)  # 1-5
    expected_volatility = Column(Numeric(precision=8, scale=4), nullable=True)
    actual_impact_score = Column(Numeric(precision=8, scale=4), nullable=True)

    # タイムスタンプ
    event_datetime = Column(DateTime(timezone=True), nullable=False, index=True)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    # メタデータ
    source = Column(String(100), nullable=True)
    source_url = Column(String(500), nullable=True)
    tags = Column(JSON, nullable=True)  # ["federal_reserve", "interest_rate", ...]

    # 処理フラグ
    is_processed = Column(Boolean, default=False)
    is_predicted = Column(Boolean, default=False)

    # インデックス設定
    __table_args__ = (
        Index("idx_global_event_datetime", "event_datetime"),
        Index("idx_global_event_importance", "importance_level", "event_datetime"),
        Index("idx_global_event_type", "event_type", "event_datetime"),
    )

    def __repr__(self):
        return f"<GlobalMarketEvent(title={self.title[:50]}..., importance={self.importance_level})>"


class TradingSession(Base):
    """取引セッション情報"""

    __tablename__ = "trading_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # セッション情報
    session_name = Column(
        String(100), nullable=False
    )  # "Tokyo", "London", "New York", "Sydney"
    market_type = Column(Enum(MarketType), nullable=False)

    # 時間情報
    session_date = Column(DateTime(timezone=True), nullable=False, index=True)
    session_start = Column(DateTime(timezone=True), nullable=False)
    session_end = Column(DateTime(timezone=True), nullable=False)

    # セッション統計
    total_volume = Column(Numeric(precision=30, scale=2), nullable=True)
    avg_volatility = Column(Numeric(precision=8, scale=6), nullable=True)
    active_symbols_count = Column(Integer, nullable=True)

    # 主要指標
    session_high = Column(Numeric(precision=20, scale=8), nullable=True)
    session_low = Column(Numeric(precision=20, scale=8), nullable=True)
    session_change = Column(Numeric(precision=8, scale=4), nullable=True)

    # メタデータ
    timezone = Column(String(50), nullable=False)
    is_holiday = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_session_name_date", "session_name", "session_date"),
        Index("idx_session_market_date", "market_type", "session_date"),
    )

    def __repr__(self):
        return f"<TradingSession(name={self.session_name}, date={self.session_date}, market={self.market_type.value})>"


class SystemPerformanceMetrics(Base):
    """システムパフォーマンス指標"""

    __tablename__ = "system_performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 時系列データ
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    metric_type = Column(
        String(100), nullable=False, index=True
    )  # "prediction_accuracy", "latency", etc.

    # メトリクス値
    value = Column(Numeric(precision=20, scale=8), nullable=False)
    unit = Column(String(20), nullable=True)  # "ms", "%", "count", etc.

    # 詳細データ
    details = Column(JSON, nullable=True)
    component = Column(
        String(100), nullable=True, index=True
    )  # "ml_engine", "data_collector", etc.

    # メタデータ
    environment = Column(
        String(50), default="production"
    )  # production, development, test
    version = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_perf_type_time", "metric_type", "timestamp"),
        Index("idx_perf_component_time", "component", "timestamp"),
        Index("idx_perf_timestamp_desc", "timestamp", postgresql_using="btree"),
    )

    def __repr__(self):
        return f"<SystemPerformanceMetrics(type={self.metric_type}, value={self.value}, component={self.component})>"
