"""
銘柄関連のデータベースモデル
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from .base import BaseModel


class Stock(BaseModel):
    """銘柄マスタ"""

    __tablename__ = "stocks"

    code = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    market = Column(String(20))  # 東証プライム、スタンダード、グロース
    sector = Column(String(50))
    industry = Column(String(50))

    # リレーション
    price_data = relationship(
        "PriceData", back_populates="stock", cascade="all, delete-orphan"
    )
    trades = relationship("Trade", back_populates="stock")
    watchlist_items = relationship(
        "WatchlistItem", back_populates="stock", cascade="all, delete-orphan"
    )
    alerts = relationship("Alert", back_populates="stock", cascade="all, delete-orphan")

    # インデックス
    __table_args__ = (
        Index("idx_stock_sector", "sector"),
        Index("idx_stock_market", "market"),
    )


class PriceData(BaseModel):
    """価格データ"""

    __tablename__ = "price_data"

    stock_code = Column(String(10), ForeignKey("stocks.code"), nullable=False)
    datetime = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Integer)

    # リレーション
    stock = relationship("Stock", back_populates="price_data")

    # インデックス
    __table_args__ = (
        Index("idx_price_stock_datetime", "stock_code", "datetime", unique=True),
        Index("idx_price_datetime", "datetime"),
    )


class Trade(BaseModel):
    """取引履歴"""

    __tablename__ = "trades"

    stock_code = Column(String(10), ForeignKey("stocks.code"), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0)
    trade_datetime = Column(DateTime, nullable=False)
    memo = Column(String(500))

    # リレーション
    stock = relationship("Stock", back_populates="trades")

    # インデックス
    __table_args__ = (
        Index("idx_trade_stock_datetime", "stock_code", "trade_datetime"),
        Index("idx_trade_type", "trade_type"),
    )

    @property
    def total_amount(self):
        """取引総額（手数料込み）"""
        if self.trade_type == "buy":
            return self.price * self.quantity + self.commission
        else:  # sell
            return self.price * self.quantity - self.commission


class WatchlistItem(BaseModel):
    """ウォッチリストアイテム"""

    __tablename__ = "watchlist_items"

    stock_code = Column(String(10), ForeignKey("stocks.code"), nullable=False)
    group_name = Column(String(50), default="default")
    memo = Column(String(200))

    # リレーション
    stock = relationship("Stock", back_populates="watchlist_items")

    # インデックス
    __table_args__ = (
        Index("idx_watchlist_stock_group", "stock_code", "group_name", unique=True),
        Index("idx_watchlist_group", "group_name"),
    )


class Alert(BaseModel):
    """アラート設定"""

    __tablename__ = "alerts"

    stock_code = Column(String(10), ForeignKey("stocks.code"), nullable=False)
    alert_type = Column(
        String(20), nullable=False
    )  # 'price_above', 'price_below', 'change_percent', etc.
    threshold = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    last_triggered = Column(DateTime)
    memo = Column(String(200))

    # リレーション
    stock = relationship("Stock", back_populates="alerts")

    # インデックス
    __table_args__ = (
        Index("idx_alert_stock_active", "stock_code", "is_active"),
        Index("idx_alert_type", "alert_type"),
    )
