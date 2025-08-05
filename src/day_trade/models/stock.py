"""
銘柄関連のデータベースモデル（改善版）

改善点:
- 金融データのFloat → DECIMAL型への変更（計算精度向上）
- データベース固有オプションの削除（クロスプラットフォーム対応）
- 責務分離の改善（モデル定義に特化）
"""
from datetime import datetime as dt
from datetime import timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    desc,
    func,
)
from sqlalchemy.orm import Session, relationship
from sqlalchemy.types import DECIMAL

from .base import BaseModel
from .enums import AlertType, TradeType


class Stock(BaseModel):
    """銘柄マスタ"""

    __tablename__ = "stocks"
    __table_args__ = {'extend_existing': True}

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
    open = Column(DECIMAL(precision=10, scale=2))
    high = Column(DECIMAL(precision=10, scale=2))
    low = Column(DECIMAL(precision=10, scale=2))
    close = Column(DECIMAL(precision=10, scale=2), nullable=False)
    volume = Column(Integer)

    # リレーション
    stock = relationship("Stock", back_populates="price_data")

    # インデックス（パフォーマンス最適化）
    __table_args__ = (
        Index("idx_price_stock_datetime", "stock_code", "datetime", unique=True),
        Index("idx_price_datetime", "datetime"),
        Index("idx_price_stock_close", "stock_code", "close"),  # 価格検索用
        Index("idx_price_volume", "volume"),  # 出来高検索用
        Index("idx_price_datetime_desc", "datetime"),  # 時系列ソート用
        {'extend_existing': True}
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
        cls, session: Session, stock_code: str, start_date: dt, end_date: dt
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
        reference_date: Optional[dt] = None,
    ) -> List["PriceData"]:
        """
        出来高急増銘柄を効率的に検出（設定可能化対応）

        Args:
            session: データベースセッション
            volume_threshold: 出来高倍率の閾値
            days_back: 遡る日数
            limit: 取得件数制限
            reference_date: 基準日（Noneの場合は現在日時を使用）
        """
        base_date = reference_date or dt.now()
        cutoff_date = base_date - timedelta(days=days_back)

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
                cls.datetime >= base_date - timedelta(days=1),
            )
            .order_by(desc(cls.volume))
            .limit(limit)
            .all()
        )


class Trade(BaseModel):
    """取引履歴"""

    __tablename__ = "trades"

    stock_code = Column(String(10), ForeignKey("stocks.code"), nullable=False)
    trade_type = Column(Enum(TradeType), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(DECIMAL(precision=10, scale=2), nullable=False)
    commission = Column(DECIMAL(precision=8, scale=2), default=0)
    trade_datetime = Column(DateTime, nullable=False)
    memo = Column(String(500))

    # リレーション
    stock = relationship("Stock", back_populates="trades")

    # インデックス（パフォーマンス最適化）
    __table_args__ = (
        Index("idx_trade_stock_datetime", "stock_code", "trade_datetime"),
        Index("idx_trade_type", "trade_type"),
        Index("idx_trade_datetime_desc", "trade_datetime"),  # 時系列ソート用
        Index("idx_trade_stock_type", "stock_code", "trade_type"),  # 複合検索用
        Index("idx_trade_price", "price"),  # 価格範囲検索用
        {'extend_existing': True}
    )

    @property
    def total_amount(self) -> Decimal:
        """
        取引総額（手数料込み）

        Returns:
            Decimal: 取引総額（DECIMAL型で精度を保持）
        """
        if not self.price or not self.quantity:
            return Decimal('0')

        base_amount = Decimal(str(self.price)) * Decimal(str(self.quantity))
        commission = Decimal(str(self.commission)) if self.commission else Decimal('0')

        if self.trade_type == TradeType.BUY:
            return base_amount + commission
        else:  # SELL
            return base_amount - commission

    @classmethod
    def get_portfolio_summary(
        cls, session: Session, start_date: Optional[dt] = None
    ) -> Dict[str, Any]:
        """
        ポートフォリオサマリーを効率的に計算

        注意: 複雑なビジネスロジックを含むため、将来的には
        PortfolioManager または TradeAnalyzer クラスへの移行を推奨
        """
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

            if trade.trade_type == TradeType.BUY:
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
        cutoff_date = dt.now() - timedelta(days=days)
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
        price: Decimal,
        commission: Decimal = Decimal('0'),
        memo: str = "",
    ) -> "Trade":
        """買い取引を作成"""
        trade = cls(
            stock_code=stock_code,
            trade_type=TradeType.BUY,
            quantity=quantity,
            price=price,
            commission=commission,
            trade_datetime=dt.now(),
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
        price: Decimal,
        commission: Decimal = Decimal('0'),
        memo: str = "",
    ) -> "Trade":
        """売り取引を作成"""
        trade = cls(
            stock_code=stock_code,
            trade_type=TradeType.SELL,
            quantity=quantity,
            price=price,
            commission=commission,
            trade_datetime=dt.now(),
            memo=memo,
        )
        session.add(trade)
        session.flush()  # IDを取得
        return trade


class WatchlistItem(BaseModel):
    """ウォッチリストアイテム"""

    __tablename__ = "watchlist_items"
    __table_args__ = {'extend_existing': True}

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
    __table_args__ = {'extend_existing': True}

    stock_code = Column(String(10), ForeignKey("stocks.code"), nullable=False)
    alert_type = Column(Enum(AlertType), nullable=False)
    threshold = Column(DECIMAL(precision=10, scale=3), nullable=False)
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
