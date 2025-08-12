#!/usr/bin/env python3
"""
Cryptocurrency Market Database Models
暗号通貨市場データのデータベースモデル
"""

from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.sql import func

from .base import Base


class CryptoPrice(Base):
    """暗号通貨価格データモデル"""

    __tablename__ = "crypto_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)  # BTCUSDT, ETHUSDT, etc.
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # 価格データ
    price = Column(Numeric(precision=20, scale=8), nullable=False)  # 暗号通貨は8桁精度
    bid_price = Column(Numeric(precision=20, scale=8), nullable=True)
    ask_price = Column(Numeric(precision=20, scale=8), nullable=True)
    spread = Column(Numeric(precision=20, scale=8), nullable=True)

    # ボリューム・市場データ
    volume_24h = Column(Numeric(precision=30, scale=8), nullable=True)
    market_cap = Column(Numeric(precision=30, scale=2), nullable=True)

    # 価格変動データ
    price_change_24h = Column(Numeric(precision=8, scale=4), nullable=True)  # パーセント
    price_change_7d = Column(Numeric(precision=8, scale=4), nullable=True)
    price_change_30d = Column(Numeric(precision=8, scale=4), nullable=True)

    # 取引所・メタデータ
    exchange = Column(String(50), nullable=False, default="binance")
    source = Column(String(50), nullable=False, default="api")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_crypto_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_crypto_timestamp_desc", "timestamp", postgresql_using="btree"),
        Index("idx_crypto_exchange_symbol", "exchange", "symbol"),
    )

    def __repr__(self):
        return f"<CryptoPrice(symbol={self.symbol}, price={self.price}, exchange={self.exchange}, time={self.timestamp})>"

    @property
    def mid_price(self) -> Optional[Decimal]:
        """中間価格計算"""
        if self.bid_price and self.ask_price:
            return (self.bid_price + self.ask_price) / 2
        return self.price

    @property
    def spread_percent(self) -> Optional[Decimal]:
        """スプレッド率（％）"""
        if not self.spread or not self.price:
            return None
        return (self.spread / self.price) * 100


class CryptoDailyStats(Base):
    """暗号通貨日次統計データ"""

    __tablename__ = "crypto_daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)

    # OHLCV データ
    open_price = Column(Numeric(precision=20, scale=8), nullable=False)
    high_price = Column(Numeric(precision=20, scale=8), nullable=False)
    low_price = Column(Numeric(precision=20, scale=8), nullable=False)
    close_price = Column(Numeric(precision=20, scale=8), nullable=False)
    volume = Column(Numeric(precision=30, scale=8), nullable=False)

    # 市場統計
    market_cap_start = Column(Numeric(precision=30, scale=2), nullable=True)
    market_cap_end = Column(Numeric(precision=30, scale=2), nullable=True)
    avg_price = Column(Numeric(precision=20, scale=8), nullable=True)

    # ボラティリティ・統計
    volatility = Column(Numeric(precision=8, scale=6), nullable=True)
    trade_count = Column(Integer, nullable=True)
    active_addresses = Column(Integer, nullable=True)  # オンチェーンデータ

    # 価格変動データ
    price_change = Column(Numeric(precision=20, scale=8), nullable=True)
    price_change_percent = Column(Numeric(precision=8, scale=4), nullable=True)
    volume_change_percent = Column(Numeric(precision=8, scale=4), nullable=True)

    # 取引所データ
    exchange = Column(String(50), nullable=False, default="binance")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_crypto_daily_symbol_date", "symbol", "date"),
        Index("idx_crypto_daily_date_desc", "date", postgresql_using="btree"),
        Index("idx_crypto_daily_exchange", "exchange", "symbol"),
    )

    def __repr__(self):
        return (
            f"<CryptoDailyStats(symbol={self.symbol}, date={self.date}, close={self.close_price})>"
        )


class CryptoOrderBook(Base):
    """暗号通貨オーダーブックデータ"""

    __tablename__ = "crypto_orderbooks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # オーダーブックデータ（JSON格式）
    bids = Column(JSON, nullable=False)  # [[price, quantity], ...]
    asks = Column(JSON, nullable=False)  # [[price, quantity], ...]

    # 統計データ
    bid_depth = Column(Numeric(precision=30, scale=8), nullable=True)  # 買い板の厚み
    ask_depth = Column(Numeric(precision=30, scale=8), nullable=True)  # 売り板の厚み
    spread = Column(Numeric(precision=20, scale=8), nullable=True)
    mid_price = Column(Numeric(precision=20, scale=8), nullable=True)

    # 流動性指標
    liquidity_score = Column(Numeric(precision=8, scale=4), nullable=True)

    # 取引所・メタデータ
    exchange = Column(String(50), nullable=False, default="binance")
    depth_level = Column(Integer, default=5)  # 取得した板の深さ
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # インデックス設定
    __table_args__ = (
        Index("idx_crypto_ob_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_crypto_ob_exchange", "exchange", "symbol"),
    )

    def __repr__(self):
        return f"<CryptoOrderBook(symbol={self.symbol}, exchange={self.exchange}, time={self.timestamp})>"


class CryptoNews(Base):
    """暗号通貨関連ニュース"""

    __tablename__ = "crypto_news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)

    # 関連シンボル
    related_symbols = Column(JSON, nullable=True)  # ["BTC", "ETH", ...]

    # センチメント分析結果
    sentiment_score = Column(Numeric(precision=6, scale=4), nullable=True)  # -1.0 to 1.0
    sentiment_label = Column(String(20), nullable=True)  # positive/negative/neutral

    # メタデータ
    source_url = Column(String(500), nullable=True)
    source_name = Column(String(100), nullable=True)
    author = Column(String(100), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # フラグ
    is_processed = Column(Boolean, default=False)
    is_important = Column(Boolean, default=False)

    # インデックス設定
    __table_args__ = (
        Index("idx_crypto_news_published", "published_at", postgresql_using="btree"),
        Index("idx_crypto_news_sentiment", "sentiment_score"),
        Index("idx_crypto_news_processed", "is_processed", "published_at"),
    )

    def __repr__(self):
        return f"<CryptoNews(title={self.title[:50]}..., sentiment={self.sentiment_score})>"


class CryptoWhaleTransaction(Base):
    """大口取引（Whale Transaction）追跡"""

    __tablename__ = "crypto_whale_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_hash = Column(String(100), nullable=False, unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    # 取引データ
    amount = Column(Numeric(precision=30, scale=8), nullable=False)
    usd_value = Column(Numeric(precision=30, scale=2), nullable=True)

    # アドレス情報
    from_address = Column(String(100), nullable=False)
    to_address = Column(String(100), nullable=False)
    from_label = Column(String(100), nullable=True)  # Exchange, Whale, etc.
    to_label = Column(String(100), nullable=True)

    # タイムスタンプ
    block_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    # 分類・重要度
    transaction_type = Column(
        String(50), nullable=True
    )  # exchange_inflow, exchange_outflow, wallet_to_wallet
    importance_score = Column(Numeric(precision=6, scale=4), nullable=True)  # 0.0 to 1.0

    # インデックス設定
    __table_args__ = (
        Index("idx_whale_symbol_timestamp", "symbol", "block_timestamp"),
        Index("idx_whale_amount_desc", "usd_value", postgresql_using="btree"),
        Index("idx_whale_importance", "importance_score"),
    )

    def __repr__(self):
        return f"<CryptoWhaleTransaction(symbol={self.symbol}, amount={self.amount}, usd_value={self.usd_value})>"
