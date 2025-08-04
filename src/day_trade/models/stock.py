"""
銘柄関連のデータベースモデル
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    desc,
    func,
)
from sqlalchemy.orm import Session, relationship

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
        Index("idx_stock_code_name", "code", "name"),  # コード・名前検索用
    )

    @classmethod
    def get_by_sector(cls, session: Session, sector: str) -> List["Stock"]:
        """セクター別銘柄取得（最適化済み）"""
        return session.query(cls).filter(cls.sector == sector).all()

    @classmethod
    def search_by_name_or_code(
        cls, session: Session, query: str, limit: int = 50
    ) -> List["Stock"]:
        """銘柄名またはコードで検索（最適化済み）"""
        return (
            session.query(cls)
            .filter((cls.name.like(f"%{query}%")) | (cls.code.like(f"%{query}%")))
            .limit(limit)
            .all()
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

    # インデックス（パフォーマンス最適化）
    __table_args__ = (
        Index("idx_price_stock_datetime", "stock_code", "datetime", unique=True),
        Index("idx_price_datetime", "datetime"),
        Index("idx_price_stock_close", "stock_code", "close"),  # 価格検索用
        Index("idx_price_volume", "volume"),  # 出来高検索用
        Index(
            "idx_price_datetime_desc", "datetime", postgresql_using="btree"
        ),  # 時系列ソート用
    )

    @classmethod
    def get_latest_prices(
        cls, session: Session, stock_codes: List[str]
    ) -> Dict[str, "PriceData"]:
        """複数銘柄の最新価格を効率的に取得"""
        # サブクエリで各銘柄の最新日時を取得
        latest_dates = (
            session.query(cls.stock_code, func.max(cls.datetime).label("max_datetime"))
            .filter(cls.stock_code.in_(stock_codes))
            .group_by(cls.stock_code)
            .subquery()
        )

        # 最新価格データを取得
        result = (
            session.query(cls)
            .join(
                latest_dates,
                (cls.stock_code == latest_dates.c.stock_code)
                & (cls.datetime == latest_dates.c.max_datetime),
            )
            .all()
        )

        return {price.stock_code: price for price in result}

    @classmethod
    def get_price_range(
        cls, session: Session, stock_code: str, start_date: datetime, end_date: datetime
    ) -> List["PriceData"]:
        """指定期間の価格データを効率的に取得（時系列順）"""
        return (
            session.query(cls)
            .filter(
                cls.stock_code == stock_code,
                cls.datetime >= start_date,
                cls.datetime <= end_date,
            )
            .order_by(cls.datetime)
            .all()
        )

    @classmethod
    def get_volume_spike_candidates(
        cls,
        session: Session,
        volume_threshold: float = 2.0,
        days_back: int = 20,
        limit: int = 100,
    ) -> List["PriceData"]:
        """出来高急増銘柄を効率的に検出"""
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # 平均出来高を計算するサブクエリ
        avg_volume_sq = (
            session.query(cls.stock_code, func.avg(cls.volume).label("avg_volume"))
            .filter(cls.datetime >= cutoff_date)
            .group_by(cls.stock_code)
            .subquery()
        )

        # 最新の出来高データと比較
        return (
            session.query(cls)
            .join(avg_volume_sq, cls.stock_code == avg_volume_sq.c.stock_code)
            .filter(
                cls.volume > avg_volume_sq.c.avg_volume * volume_threshold,
                cls.datetime >= datetime.now() - timedelta(days=1),
            )
            .order_by(desc(cls.volume))
            .limit(limit)
            .all()
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

    # インデックス（パフォーマンス最適化）
    __table_args__ = (
        Index("idx_trade_stock_datetime", "stock_code", "trade_datetime"),
        Index("idx_trade_type", "trade_type"),
        Index(
            "idx_trade_datetime_desc", "trade_datetime", postgresql_using="btree"
        ),  # 時系列ソート用
        Index("idx_trade_stock_type", "stock_code", "trade_type"),  # 複合検索用
        Index("idx_trade_price", "price"),  # 価格範囲検索用
    )

    @property
    def total_amount(self):
        """取引総額（手数料込み）"""
        if self.trade_type == "buy":
            return self.price * self.quantity + self.commission
        else:  # sell
            return self.price * self.quantity - self.commission

    @classmethod
    def get_portfolio_summary(
        cls, session: Session, start_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """ポートフォリオサマリーを効率的に計算"""
        query = session.query(cls)
        if start_date:
            query = query.filter(cls.trade_datetime >= start_date)

        trades = query.all()

        portfolio = {}
        total_cost = 0
        total_proceeds = 0

        for trade in trades:
            code = trade.stock_code
            if code not in portfolio:
                portfolio[code] = {
                    "quantity": 0,
                    "total_cost": 0,
                    "avg_price": 0,
                    "trades": [],
                }

            if trade.trade_type == "buy":
                portfolio[code]["quantity"] += trade.quantity
                portfolio[code]["total_cost"] += trade.total_amount
                total_cost += trade.total_amount
            else:  # sell
                portfolio[code]["quantity"] -= trade.quantity
                total_proceeds += trade.total_amount

            portfolio[code]["trades"].append(trade)

        # 平均価格を計算
        for _code, data in portfolio.items():
            if data["quantity"] > 0:
                data["avg_price"] = data["total_cost"] / data["quantity"]

        return {
            "portfolio": portfolio,
            "total_cost": total_cost,
            "total_proceeds": total_proceeds,
            "net_position": total_proceeds - total_cost,
        }

    @classmethod
    def get_recent_trades(
        cls, session: Session, days: int = 30, limit: int = 100
    ) -> List["Trade"]:
        """最近の取引履歴を効率的に取得"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return (
            session.query(cls)
            .filter(cls.trade_datetime >= cutoff_date)
            .order_by(desc(cls.trade_datetime))
            .limit(limit)
            .all()
        )

    @classmethod
    def create_buy_trade(
        cls,
        session: Session,
        stock_code: str,
        quantity: int,
        price: float,
        commission: float = 0,
        memo: str = "",
    ) -> "Trade":
        """買い取引を作成"""
        trade = cls(
            stock_code=stock_code,
            trade_type="buy",
            quantity=quantity,
            price=price,
            commission=commission,
            trade_datetime=datetime.now(),
            memo=memo,
        )
        session.add(trade)
        session.flush()  # IDを取得
        return trade

    @classmethod
    def create_sell_trade(
        cls,
        session: Session,
        stock_code: str,
        quantity: int,
        price: float,
        commission: float = 0,
        memo: str = "",
    ) -> "Trade":
        """売り取引を作成"""
        trade = cls(
            stock_code=stock_code,
            trade_type="sell",
            quantity=quantity,
            price=price,
            commission=commission,
            trade_datetime=datetime.now(),
            memo=memo,
        )
        session.add(trade)
        session.flush()  # IDを取得
        return trade


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
