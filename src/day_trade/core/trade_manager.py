"""
取引記録管理機能
売買履歴を記録し、損益計算を行う
データベース永続化対応版
"""

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from ..models.database import db_manager
from ..models.stock import Stock
from ..models.stock import Trade as DBTrade
from ..utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)

logger = get_context_logger(__name__)


class TradeType(Enum):
    """取引タイプ"""

    BUY = "buy"
    SELL = "sell"


class TradeStatus(Enum):
    """取引ステータス"""

    PENDING = "pending"  # 注文中
    EXECUTED = "executed"  # 約定済み
    CANCELLED = "cancelled"  # キャンセル
    PARTIAL = "partial"  # 一部約定


@dataclass
class Trade:
    """取引記録"""

    id: str
    symbol: str
    trade_type: TradeType
    quantity: int
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal("0")
    status: TradeStatus = TradeStatus.EXECUTED
    notes: str = ""

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        data = asdict(self)
        data["trade_type"] = self.trade_type.value
        data["status"] = self.status.value
        data["price"] = str(self.price)
        data["commission"] = str(self.commission)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "Trade":
        """辞書から復元"""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            trade_type=TradeType(data["trade_type"]),
            quantity=data["quantity"],
            price=Decimal(data["price"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            commission=Decimal(data["commission"]),
            status=TradeStatus(data["status"]),
            notes=data.get("notes", ""),
        )


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int
    average_price: Decimal
    total_cost: Decimal
    current_price: Decimal = Decimal("0")

    @property
    def market_value(self) -> Decimal:
        """時価総額"""
        return self.current_price * Decimal(self.quantity)

    @property
    def unrealized_pnl(self) -> Decimal:
        """含み損益"""
        return self.market_value - self.total_cost

    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """含み損益率"""
        if self.total_cost == 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.total_cost) * 100

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": str(self.average_price),
            "total_cost": str(self.total_cost),
            "current_price": str(self.current_price),
            "market_value": str(self.market_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "unrealized_pnl_percent": str(
                self.unrealized_pnl_percent.quantize(Decimal("0.01"))
            ),
        }


@dataclass
class RealizedPnL:
    """実現損益"""

    symbol: str
    quantity: int
    buy_price: Decimal
    sell_price: Decimal
    buy_commission: Decimal
    sell_commission: Decimal
    pnl: Decimal
    pnl_percent: Decimal
    buy_date: datetime
    sell_date: datetime

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "buy_price": str(self.buy_price),
            "sell_price": str(self.sell_price),
            "buy_commission": str(self.buy_commission),
            "sell_commission": str(self.sell_commission),
            "pnl": str(self.pnl),
            "pnl_percent": str(self.pnl_percent.quantize(Decimal("0.01"))),
            "buy_date": self.buy_date.isoformat(),
            "sell_date": self.sell_date.isoformat(),
        }


class TradeManager:
    """取引記録管理クラス"""

    def __init__(
        self,
        commission_rate: Decimal = Decimal("0.001"),
        tax_rate: Decimal = Decimal("0.2"),
        load_from_db: bool = False,
    ):
        """
        初期化

        Args:
            commission_rate: 手数料率（デフォルト0.1%）
            tax_rate: 税率（デフォルト20%）
            load_from_db: データベースから取引履歴を読み込むかどうか
        """
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: List[RealizedPnL] = []
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self._trade_counter = 0

        # ロガーを初期化
        self.logger = get_context_logger(__name__)

        if load_from_db:
            self._load_trades_from_db()

    def _generate_trade_id(self) -> str:
        """取引IDを生成"""
        self._trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d')}{self._trade_counter:04d}"

    def _calculate_commission(self, price: Decimal, quantity: int) -> Decimal:
        """手数料を計算"""
        total_value = price * Decimal(quantity)
        commission = total_value * self.commission_rate
        # 最低100円の手数料
        return max(commission, Decimal("100"))

    def add_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: int,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        commission: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> str:
        """
        取引を追加（データベース永続化対応）

        Args:
            symbol: 銘柄コード
            trade_type: 取引タイプ
            quantity: 数量
            price: 価格
            timestamp: 取引日時
            commission: 手数料（Noneの場合は自動計算）
            notes: メモ
            persist_to_db: データベースに永続化するかどうか

        Returns:
            取引ID
        """
        operation_logger = logger.bind(
            operation="add_trade",
            symbol=symbol,
            trade_type=trade_type.value,
            quantity=quantity,
            price=str(price),
            persist_to_db=persist_to_db,
        )

        operation_logger.info("取引追加処理開始")

        try:
            if timestamp is None:
                timestamp = datetime.now()

            if commission is None:
                commission = self._calculate_commission(price, quantity)

            trade_id = self._generate_trade_id()

            # メモリ内データ構造のトレード
            memory_trade = Trade(
                id=trade_id,
                symbol=symbol,
                trade_type=trade_type,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                commission=commission,
                notes=notes,
            )

            if persist_to_db:
                # データベース永続化（トランザクション保護）
                with db_manager.transaction_scope() as session:
                    # 1. 銘柄マスタの存在確認・作成
                    stock = session.query(Stock).filter(Stock.code == symbol).first()
                    if not stock:
                        operation_logger.info("銘柄マスタに未登録、新規作成")
                        stock = Stock(
                            code=symbol,
                            name=symbol,  # 名前が不明な場合はコードを使用
                            market="未定",
                            sector="未定",
                            industry="未定",
                        )
                        session.add(stock)
                        session.flush()  # IDを確定

                    # 2. データベース取引記録を作成
                    db_trade = (
                        DBTrade.create_buy_trade(
                            session=session,
                            stock_code=symbol,
                            quantity=quantity,
                            price=price,
                            commission=commission,
                            memo=notes,
                        )
                        if trade_type == TradeType.BUY
                        else DBTrade.create_sell_trade(
                            session=session,
                            stock_code=symbol,
                            quantity=quantity,
                            price=price,
                            commission=commission,
                            memo=notes,
                        )
                    )

                    # 3. メモリ内データ構造を更新
                    self.trades.append(memory_trade)
                    self._update_position(memory_trade)

                    # 中間状態をflushして整合性を確認
                    session.flush()

                    # ビジネスイベントログ
                    log_business_event(
                        "trade_added",
                        trade_id=trade_id,
                        symbol=symbol,
                        trade_type=trade_type.value,
                        quantity=quantity,
                        price=str(price),
                        commission=str(commission),
                        persisted=True,
                    )

                    operation_logger.info(
                        "取引追加完了（DB永続化）",
                        trade_id=trade_id,
                        db_trade_id=db_trade.id,
                    )
            else:
                # メモリ内のみの処理（後方互換性）
                self.trades.append(memory_trade)
                self._update_position(memory_trade)

                log_business_event(
                    "trade_added",
                    trade_id=trade_id,
                    symbol=symbol,
                    trade_type=trade_type.value,
                    quantity=quantity,
                    price=str(price),
                    commission=str(commission),
                    persisted=False,
                )

                operation_logger.info("取引追加完了（メモリのみ）", trade_id=trade_id)

            return trade_id

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "add_trade",
                    "symbol": symbol,
                    "trade_type": trade_type.value,
                    "quantity": quantity,
                    "price": str(price),
                    "persist_to_db": persist_to_db,
                },
            )
            operation_logger.error("取引追加失敗", error=str(e))
            raise

    def _update_position(self, trade: Trade) -> None:
        """ポジションを更新"""
        symbol = trade.symbol

        if trade.trade_type == TradeType.BUY:
            if symbol in self.positions:
                # 既存ポジションに追加
                position = self.positions[symbol]
                total_cost = (
                    position.total_cost
                    + (trade.price * Decimal(trade.quantity))
                    + trade.commission
                )
                total_quantity = position.quantity + trade.quantity
                average_price = total_cost / Decimal(total_quantity)

                position.quantity = total_quantity
                position.average_price = average_price
                position.total_cost = total_cost
            else:
                # 新規ポジション
                total_cost = (trade.price * Decimal(trade.quantity)) + trade.commission
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    average_price=total_cost / Decimal(trade.quantity),
                    total_cost=total_cost,
                )

        elif trade.trade_type == TradeType.SELL:
            if symbol in self.positions:
                position = self.positions[symbol]

                if position.quantity >= trade.quantity:
                    # 実現損益を計算
                    buy_price = position.average_price
                    sell_price = trade.price

                    # 手数料を按分
                    buy_commission_per_share = (
                        position.total_cost / Decimal(position.quantity)
                        - position.average_price
                    )
                    buy_commission = buy_commission_per_share * Decimal(trade.quantity)

                    pnl_before_tax = (
                        (sell_price - buy_price) * Decimal(trade.quantity)
                        - buy_commission
                        - trade.commission
                    )

                    # 税金計算（利益が出た場合のみ）
                    tax = Decimal("0")
                    if pnl_before_tax > 0:
                        tax = pnl_before_tax * self.tax_rate

                    pnl = pnl_before_tax - tax
                    pnl_percent = (pnl / (buy_price * Decimal(trade.quantity))) * 100

                    # 実現損益を記録
                    realized_pnl = RealizedPnL(
                        symbol=symbol,
                        quantity=trade.quantity,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        buy_commission=buy_commission,
                        sell_commission=trade.commission,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        buy_date=self._get_earliest_buy_date(symbol),
                        sell_date=trade.timestamp,
                    )

                    self.realized_pnl.append(realized_pnl)

                    # ポジション更新
                    remaining_quantity = position.quantity - trade.quantity
                    if remaining_quantity > 0:
                        # 按分してコストを調整
                        remaining_ratio = Decimal(remaining_quantity) / Decimal(
                            position.quantity
                        )
                        remaining_cost = position.total_cost * remaining_ratio
                        position.quantity = remaining_quantity
                        position.total_cost = remaining_cost
                        position.average_price = remaining_cost / Decimal(
                            remaining_quantity
                        )
                    else:
                        # ポジション完全クローズ
                        del self.positions[symbol]

                else:
                    logger.warning(
                        f"銘柄 '{symbol}' の売却数量が保有数量 ({position.quantity}) を超過しています。売却数量: {trade.quantity}。取引は処理されません。"
                    )
            else:
                logger.warning(
                    f"ポジションを保有していない銘柄 '{symbol}' の売却を試みました。取引は無視されます。"
                )

    def _get_earliest_buy_date(self, symbol: str) -> datetime:
        """最も古い買い取引の日付を取得"""
        buy_trades = [
            t
            for t in self.trades
            if t.symbol == symbol and t.trade_type == TradeType.BUY
        ]
        if buy_trades:
            return min(trade.timestamp for trade in buy_trades)
        return datetime.now()

    def _load_trades_from_db(self) -> None:
        """データベースから取引履歴を読み込み（トランザクション保護版）"""
        load_logger = self.logger.bind(operation="load_trades_from_db")
        load_logger.info("データベースから取引履歴読み込み開始")

        try:
            # トランザクション内で一括処理
            with db_manager.transaction_scope() as session:
                # データベースから全取引を取得
                db_trades = (
                    session.query(DBTrade).order_by(DBTrade.trade_datetime).all()
                )

                load_logger.info("DB取引データ取得", count=len(db_trades))

                # メモリ内データ構造を一旦クリア（原子性保証）
                trades_backup = self.trades.copy()
                positions_backup = self.positions.copy()
                realized_pnl_backup = self.realized_pnl.copy()
                counter_backup = self._trade_counter

                try:
                    # メモリ内データクリア
                    self.trades.clear()
                    self.positions.clear()
                    self.realized_pnl.clear()
                    self._trade_counter = 0

                    for db_trade in db_trades:
                        # セッションから切り離す前に必要な属性を読み込み
                        trade_id = db_trade.id
                        stock_code = db_trade.stock_code
                        trade_type_str = db_trade.trade_type
                        quantity = db_trade.quantity
                        price = db_trade.price
                        trade_datetime = db_trade.trade_datetime
                        commission = db_trade.commission or Decimal("0")
                        memo = db_trade.memo or ""

                        # メモリ内形式に変換
                        if isinstance(trade_type_str, TradeType):
                            trade_type = trade_type_str
                        else:
                            trade_type = (
                                TradeType.BUY
                                if str(trade_type_str).lower() == "buy"
                                else TradeType.SELL
                            )

                        memory_trade = Trade(
                            id=f"DB_{trade_id}",  # DBから読み込んだことを示すプレフィックス
                            symbol=stock_code,
                            trade_type=trade_type,
                            quantity=quantity,
                            price=Decimal(str(price)),
                            timestamp=trade_datetime,
                            commission=Decimal(str(commission)),
                            status=TradeStatus.EXECUTED,
                            notes=memo,
                        )

                        self.trades.append(memory_trade)
                        self._update_position(memory_trade)

                    # 取引カウンターを最大値+1に設定
                    if db_trades:
                        max_id = max(db_trade.id for db_trade in db_trades)
                        self._trade_counter = max_id + 1

                    load_logger.info(
                        "データベース読み込み完了",
                        loaded_trades=len(db_trades),
                        trade_counter=self._trade_counter,
                    )

                except Exception as restore_error:
                    # メモリ内データの復元
                    self.trades = trades_backup
                    self.positions = positions_backup
                    self.realized_pnl = realized_pnl_backup
                    self._trade_counter = counter_backup
                    load_logger.error(
                        "読み込み処理失敗、メモリ内データを復元",
                        error=str(restore_error),
                    )
                    raise restore_error

        except Exception as e:
            log_error_with_context(e, {"operation": "load_trades_from_db"})
            load_logger.error("データベース読み込み失敗", error=str(e))
            raise

    def sync_with_db(self) -> None:
        """データベースとの同期を実行（原子性保証版）"""
        sync_logger = self.logger.bind(operation="sync_with_db")
        sync_logger.info("データベース同期開始")

        # 現在のメモリ内データをバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        realized_pnl_backup = self.realized_pnl.copy()
        counter_backup = self._trade_counter

        try:
            # 現在のメモリ内データをクリア
            self.trades.clear()
            self.positions.clear()
            self.realized_pnl.clear()
            self._trade_counter = 0

            # データベースから再読み込み（トランザクション保護済み）
            self._load_trades_from_db()

            sync_logger.info("データベース同期完了")

        except Exception as e:
            # エラー時にはバックアップデータを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(e, {"operation": "sync_with_db"})
            sync_logger.error(
                "データベース同期失敗、メモリ内データを復元", error=str(e)
            )
            raise

    def add_trades_batch(
        self, trades_data: List[Dict], persist_to_db: bool = True
    ) -> List[str]:
        """
        複数の取引を一括追加（トランザクション保護）

        Args:
            trades_data: 取引データのリスト
                [{"symbol": "7203", "trade_type": TradeType.BUY, "quantity": 100, "price": Decimal("2500"), ...}, ...]
            persist_to_db: データベースに永続化するかどうか

        Returns:
            作成された取引IDのリスト

        Raises:
            Exception: いずれかの取引処理でエラーが発生した場合、すべての処理がロールバック
        """
        batch_logger = logger.bind(
            operation="add_trades_batch",
            batch_size=len(trades_data),
            persist_to_db=persist_to_db,
        )
        batch_logger.info("一括取引追加処理開始")

        if not trades_data:
            batch_logger.warning("空の取引データが渡されました")
            return []

        trade_ids = []

        # メモリ内データのバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        realized_pnl_backup = self.realized_pnl.copy()
        counter_backup = self._trade_counter

        try:
            if persist_to_db:
                # データベース永続化の場合は全処理をトランザクション内で実行
                with db_manager.transaction_scope() as session:
                    for i, trade_data in enumerate(trades_data):
                        try:
                            # 取引データの検証と補完
                            symbol = trade_data["symbol"]
                            trade_type = trade_data["trade_type"]
                            quantity = trade_data["quantity"]
                            price = trade_data["price"]
                            timestamp = trade_data.get("timestamp", datetime.now())
                            commission = trade_data.get("commission")
                            notes = trade_data.get("notes", "")

                            if commission is None:
                                commission = self._calculate_commission(price, quantity)

                            trade_id = self._generate_trade_id()

                            # 1. 銘柄マスタの存在確認・作成
                            stock = (
                                session.query(Stock)
                                .filter(Stock.code == symbol)
                                .first()
                            )
                            if not stock:
                                stock = Stock(
                                    code=symbol,
                                    name=symbol,
                                    market="未定",
                                    sector="未定",
                                    industry="未定",
                                )
                                session.add(stock)
                                session.flush()

                            # 2. データベース取引記録を作成
                            DBTrade.create_buy_trade(
                                session=session,
                                stock_code=symbol,
                                quantity=quantity,
                                price=price,
                                commission=commission,
                                memo=notes,
                            ) if trade_type == TradeType.BUY else DBTrade.create_sell_trade(
                                session=session,
                                stock_code=symbol,
                                quantity=quantity,
                                price=price,
                                commission=commission,
                                memo=notes,
                            )

                            # 3. メモリ内データ構造を更新
                            memory_trade = Trade(
                                id=trade_id,
                                symbol=symbol,
                                trade_type=trade_type,
                                quantity=quantity,
                                price=price,
                                timestamp=timestamp,
                                commission=commission,
                                notes=notes,
                            )

                            self.trades.append(memory_trade)
                            self._update_position(memory_trade)
                            trade_ids.append(trade_id)

                            # バッチ内の中間状態をflush
                            if (i + 1) % 10 == 0:  # 10件ごとにflush
                                session.flush()

                        except Exception as trade_error:
                            batch_logger.error(
                                "個別取引処理失敗",
                                trade_index=i,
                                symbol=trade_data.get("symbol", "unknown"),
                                error=str(trade_error),
                            )
                            raise trade_error

                    # 最終的なビジネスイベントログ
                    log_business_event(
                        "trades_batch_added",
                        batch_size=len(trades_data),
                        trade_ids=trade_ids,
                        persisted=True,
                    )

                    batch_logger.info(
                        "一括取引追加完了（DB永続化）", trade_count=len(trade_ids)
                    )

            else:
                # メモリ内のみの処理
                for trade_data in trades_data:
                    symbol = trade_data["symbol"]
                    trade_type = trade_data["trade_type"]
                    quantity = trade_data["quantity"]
                    price = trade_data["price"]
                    timestamp = trade_data.get("timestamp", datetime.now())
                    commission = trade_data.get("commission")
                    notes = trade_data.get("notes", "")

                    if commission is None:
                        commission = self._calculate_commission(price, quantity)

                    trade_id = self._generate_trade_id()

                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=trade_type,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        commission=commission,
                        notes=notes,
                    )

                    self.trades.append(memory_trade)
                    self._update_position(memory_trade)
                    trade_ids.append(trade_id)

                log_business_event(
                    "trades_batch_added",
                    batch_size=len(trades_data),
                    trade_ids=trade_ids,
                    persisted=False,
                )

                batch_logger.info(
                    "一括取引追加完了（メモリのみ）", trade_count=len(trade_ids)
                )

            return trade_ids

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e,
                {
                    "operation": "add_trades_batch",
                    "batch_size": len(trades_data),
                    "persist_to_db": persist_to_db,
                    "completed_trades": len(trade_ids),
                },
            )
            batch_logger.error(
                "一括取引追加失敗、すべての変更をロールバック", error=str(e)
            )
            raise

    def clear_all_data(self, persist_to_db: bool = True) -> None:
        """
        すべての取引データを削除（トランザクション保護）

        Args:
            persist_to_db: データベースからも削除するかどうか

        Warning:
            この操作は取引履歴、ポジション、実現損益をすべて削除します
        """
        clear_logger = logger.bind(
            operation="clear_all_data", persist_to_db=persist_to_db
        )
        clear_logger.warning("全データ削除処理開始")

        # メモリ内データのバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        realized_pnl_backup = self.realized_pnl.copy()
        counter_backup = self._trade_counter

        try:
            if persist_to_db:
                # データベースとメモリ両方をクリア
                with db_manager.transaction_scope() as session:
                    # データベースの取引データを削除
                    deleted_count = session.query(DBTrade).delete()
                    clear_logger.info(
                        "データベース取引データ削除", deleted_count=deleted_count
                    )

                    # メモリ内データクリア
                    self.trades.clear()
                    self.positions.clear()
                    self.realized_pnl.clear()
                    self._trade_counter = 0

                    log_business_event(
                        "all_data_cleared",
                        deleted_db_records=deleted_count,
                        persisted=True,
                    )

                    clear_logger.warning("全データ削除完了（DB + メモリ）")
            else:
                # メモリ内のみクリア
                self.trades.clear()
                self.positions.clear()
                self.realized_pnl.clear()
                self._trade_counter = 0

                log_business_event(
                    "all_data_cleared", deleted_db_records=0, persisted=False
                )

                clear_logger.warning("全データ削除完了（メモリのみ）")

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e, {"operation": "clear_all_data", "persist_to_db": persist_to_db}
            )
            clear_logger.error("全データ削除失敗、メモリ内データを復元", error=str(e))
            raise

    def get_position(self, symbol: str) -> Optional[Position]:
        """ポジション情報を取得"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """全ポジション情報を取得"""
        return self.positions.copy()

    def update_current_prices(self, prices: Dict[str, Decimal]) -> None:
        """現在価格を更新"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """取引履歴を取得"""
        if symbol:
            return [trade for trade in self.trades if trade.symbol == symbol]
        return self.trades.copy()

    def get_realized_pnl_history(
        self, symbol: Optional[str] = None
    ) -> List[RealizedPnL]:
        """実現損益履歴を取得"""
        if symbol:
            return [pnl for pnl in self.realized_pnl if pnl.symbol == symbol]
        return self.realized_pnl.copy()

    def get_portfolio_summary(self) -> Dict:
        """ポートフォリオサマリーを取得"""
        total_cost = sum(pos.total_cost for pos in self.positions.values())
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = total_market_value - total_cost
        total_realized_pnl = sum(pnl.pnl for pnl in self.realized_pnl)

        return {
            "total_positions": len(self.positions),
            "total_cost": str(total_cost),
            "total_market_value": str(total_market_value),
            "total_unrealized_pnl": str(total_unrealized_pnl),
            "total_realized_pnl": str(total_realized_pnl),
            "total_pnl": str(total_unrealized_pnl + total_realized_pnl),
            "total_trades": len(self.trades),
            "winning_trades": len([pnl for pnl in self.realized_pnl if pnl.pnl > 0]),
            "losing_trades": len([pnl for pnl in self.realized_pnl if pnl.pnl < 0]),
            "win_rate": (
                f"{(len([pnl for pnl in self.realized_pnl if pnl.pnl > 0]) / max(len(self.realized_pnl), 1) * 100):.1f}%"
                if self.realized_pnl
                else "0.0%"
            ),
        }

    def export_to_csv(self, filepath: str, data_type: str = "trades") -> None:
        """
        CSVファイルにエクスポート

        Args:
            filepath: 出力ファイルパス
            data_type: データタイプ ('trades', 'positions', 'realized_pnl')
        """
        try:
            if data_type == "trades":
                data = [trade.to_dict() for trade in self.trades]
                fieldnames = [
                    "id",
                    "symbol",
                    "trade_type",
                    "quantity",
                    "price",
                    "timestamp",
                    "commission",
                    "status",
                    "notes",
                ]

            elif data_type == "positions":
                data = [pos.to_dict() for pos in self.positions.values()]
                fieldnames = [
                    "symbol",
                    "quantity",
                    "average_price",
                    "total_cost",
                    "current_price",
                    "market_value",
                    "unrealized_pnl",
                    "unrealized_pnl_percent",
                ]

            elif data_type == "realized_pnl":
                data = [pnl.to_dict() for pnl in self.realized_pnl]
                fieldnames = [
                    "symbol",
                    "quantity",
                    "buy_price",
                    "sell_price",
                    "buy_commission",
                    "sell_commission",
                    "pnl",
                    "pnl_percent",
                    "buy_date",
                    "sell_date",
                ]

            else:
                raise ValueError(f"Invalid data_type: {data_type}")

            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"CSV出力完了: {filepath} ({len(data)}件)")

        except Exception as e:
            logger.error(
                f"データのエクスポート中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {e}"
            )
            raise

    def save_to_json(self, filepath: str) -> None:
        """JSON形式で保存"""
        try:
            data = {
                "trades": [trade.to_dict() for trade in self.trades],
                "positions": {
                    symbol: pos.to_dict() for symbol, pos in self.positions.items()
                },
                "realized_pnl": [pnl.to_dict() for pnl in self.realized_pnl],
                "settings": {
                    "commission_rate": str(self.commission_rate),
                    "tax_rate": str(self.tax_rate),
                    "trade_counter": self._trade_counter,
                },
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"JSON保存完了: {filepath}")

        except Exception as e:
            logger.error(
                f"データの保存中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {e}"
            )
            raise

    def load_from_json(self, filepath: str) -> None:
        """JSON形式から読み込み"""
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            # 取引履歴復元
            self.trades = [
                Trade.from_dict(trade_data) for trade_data in data.get("trades", [])
            ]

            # ポジション復元
            self.positions = {}
            for symbol, pos_data in data.get("positions", {}).items():
                self.positions[symbol] = Position(
                    symbol=pos_data["symbol"],
                    quantity=pos_data["quantity"],
                    average_price=Decimal(pos_data["average_price"]),
                    total_cost=Decimal(pos_data["total_cost"]),
                    current_price=Decimal(pos_data["current_price"]),
                )

            # 実現損益復元
            self.realized_pnl = []
            for pnl_data in data.get("realized_pnl", []):
                self.realized_pnl.append(
                    RealizedPnL(
                        symbol=pnl_data["symbol"],
                        quantity=pnl_data["quantity"],
                        buy_price=Decimal(pnl_data["buy_price"]),
                        sell_price=Decimal(pnl_data["sell_price"]),
                        buy_commission=Decimal(pnl_data["buy_commission"]),
                        sell_commission=Decimal(pnl_data["sell_commission"]),
                        pnl=Decimal(pnl_data["pnl"]),
                        pnl_percent=Decimal(pnl_data["pnl_percent"]),
                        buy_date=datetime.fromisoformat(pnl_data["buy_date"]),
                        sell_date=datetime.fromisoformat(pnl_data["sell_date"]),
                    )
                )

            # 設定復元
            settings = data.get("settings", {})
            if "commission_rate" in settings:
                self.commission_rate = Decimal(settings["commission_rate"])
            if "tax_rate" in settings:
                self.tax_rate = Decimal(settings["tax_rate"])
            if "trade_counter" in settings:
                self._trade_counter = settings["trade_counter"]

            logger.info(f"JSON読み込み完了: {filepath}")

        except Exception as e:
            logger.error(
                f"データの読み込み中にエラーが発生しました。ファイル形式が正しいか、破損していないか確認してください。詳細: {e}"
            )
            raise

    def buy_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        株式買い注文を実行（完全なトランザクション保護）

        ポートフォリオ更新と取引履歴追加を単一のトランザクションで処理し、
        データの整合性を保証する。

        Args:
            symbol: 銘柄コード
            quantity: 購入数量
            price: 購入価格
            current_market_price: 現在の市場価格（ポートフォリオ評価用）
            notes: 取引メモ
            persist_to_db: データベースに永続化するかどうか

        Returns:
            取引結果辞書（取引ID、更新後ポジション、手数料等）

        Raises:
            ValueError: 無効な購入パラメータ
            Exception: データベース処理エラー
        """
        buy_logger = logger.bind(
            operation="buy_stock",
            symbol=symbol,
            quantity=quantity,
            price=str(price),
            persist_to_db=persist_to_db,
        )

        buy_logger.info("株式買い注文処理開始")

        # パラメータ検証
        if quantity <= 0:
            raise ValueError(f"購入数量は正数である必要があります: {quantity}")
        if price <= 0:
            raise ValueError(f"購入価格は正数である必要があります: {price}")

        # メモリ内データのバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        counter_backup = self._trade_counter

        try:
            # 手数料計算
            commission = self._calculate_commission(price, quantity)
            timestamp = datetime.now()

            if persist_to_db:
                # データベース永続化の場合は全処理をトランザクション内で実行
                with db_manager.transaction_scope() as session:
                    # 1. 銘柄マスタの存在確認・作成
                    stock = session.query(Stock).filter(Stock.code == symbol).first()
                    if not stock:
                        buy_logger.info("銘柄マスタに未登録、新規作成")
                        stock = Stock(
                            code=symbol,
                            name=symbol,
                            market="未定",
                            sector="未定",
                            industry="未定",
                        )
                        session.add(stock)
                        session.flush()

                    # 2. 取引ID生成
                    trade_id = self._generate_trade_id()

                    # 3. データベース取引記録を作成
                    db_trade = DBTrade.create_buy_trade(
                        session=session,
                        stock_code=symbol,
                        quantity=quantity,
                        price=price,
                        commission=commission,
                        memo=notes,
                    )

                    # 4. メモリ内取引記録作成
                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=TradeType.BUY,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        commission=commission,
                        notes=notes,
                    )

                    # 5. ポジション更新（原子的実行）
                    old_position = self.positions.get(symbol)
                    self.trades.append(memory_trade)
                    self._update_position(memory_trade)
                    new_position = self.positions.get(symbol)

                    # 6. 現在価格更新（指定されている場合）
                    if current_market_price and symbol in self.positions:
                        self.positions[symbol].current_price = current_market_price

                    # 中間状態をflushして整合性を確認
                    session.flush()

                    # ビジネスイベントログ
                    log_business_event(
                        "stock_purchased",
                        trade_id=trade_id,
                        symbol=symbol,
                        quantity=quantity,
                        price=str(price),
                        commission=str(commission),
                        old_position=old_position.to_dict() if old_position else None,
                        new_position=new_position.to_dict() if new_position else None,
                        persisted=True,
                    )

                    buy_logger.info(
                        "株式買い注文完了（DB永続化）",
                        trade_id=trade_id,
                        db_trade_id=db_trade.id,
                        commission=str(commission),
                    )
            else:
                # メモリ内のみの処理
                trade_id = self._generate_trade_id()

                memory_trade = Trade(
                    id=trade_id,
                    symbol=symbol,
                    trade_type=TradeType.BUY,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    commission=commission,
                    notes=notes,
                )

                old_position = self.positions.get(symbol)
                self.trades.append(memory_trade)
                self._update_position(memory_trade)
                new_position = self.positions.get(symbol)

                if current_market_price and symbol in self.positions:
                    self.positions[symbol].current_price = current_market_price

                log_business_event(
                    "stock_purchased",
                    trade_id=trade_id,
                    symbol=symbol,
                    quantity=quantity,
                    price=str(price),
                    commission=str(commission),
                    old_position=old_position.to_dict() if old_position else None,
                    new_position=new_position.to_dict() if new_position else None,
                    persisted=False,
                )

                buy_logger.info(
                    "株式買い注文完了（メモリのみ）",
                    trade_id=trade_id,
                    commission=str(commission),
                )

            # 結果データ作成
            result = {
                "success": True,
                "trade_id": trade_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": str(price),
                "commission": str(commission),
                "timestamp": timestamp.isoformat(),
                "position": self.positions[symbol].to_dict()
                if symbol in self.positions
                else None,
                "total_cost": str(price * quantity + commission),
            }

            return result

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e,
                {
                    "operation": "buy_stock",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": str(price),
                    "persist_to_db": persist_to_db,
                },
            )
            buy_logger.error("株式買い注文失敗、変更をロールバック", error=str(e))
            raise

    def sell_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        株式売り注文を実行（完全なトランザクション保護）

        ポートフォリオ更新、取引履歴追加、実現損益計算を
        単一のトランザクションで処理し、データの整合性を保証する。

        Args:
            symbol: 銘柄コード
            quantity: 売却数量
            price: 売却価格
            current_market_price: 現在の市場価格（ポートフォリオ評価用）
            notes: 取引メモ
            persist_to_db: データベースに永続化するかどうか

        Returns:
            取引結果辞書（取引ID、更新後ポジション、実現損益等）

        Raises:
            ValueError: 無効な売却パラメータ（保有数量不足等）
            Exception: データベース処理エラー
        """
        sell_logger = logger.bind(
            operation="sell_stock",
            symbol=symbol,
            quantity=quantity,
            price=str(price),
            persist_to_db=persist_to_db,
        )

        sell_logger.info("株式売り注文処理開始")

        # パラメータ検証
        if quantity <= 0:
            raise ValueError(f"売却数量は正数である必要があります: {quantity}")
        if price <= 0:
            raise ValueError(f"売却価格は正数である必要があります: {price}")

        # ポジション存在確認
        if symbol not in self.positions:
            raise ValueError(f"銘柄 '{symbol}' のポジションが存在しません")

        current_position = self.positions[symbol]
        if current_position.quantity < quantity:
            raise ValueError(
                f"売却数量 ({quantity}) が保有数量 ({current_position.quantity}) を超過しています"
            )

        # メモリ内データのバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        realized_pnl_backup = self.realized_pnl.copy()
        counter_backup = self._trade_counter

        try:
            # 手数料計算
            commission = self._calculate_commission(price, quantity)
            timestamp = datetime.now()

            if persist_to_db:
                # データベース永続化の場合は全処理をトランザクション内で実行
                with db_manager.transaction_scope() as session:
                    # 1. 取引ID生成
                    trade_id = self._generate_trade_id()

                    # 2. データベース取引記録を作成
                    db_trade = DBTrade.create_sell_trade(
                        session=session,
                        stock_code=symbol,
                        quantity=quantity,
                        price=price,
                        commission=commission,
                        memo=notes,
                    )

                    # 3. メモリ内取引記録作成
                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=TradeType.SELL,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        commission=commission,
                        notes=notes,
                    )

                    # 4. ポジション更新と実現損益計算（原子的実行）
                    old_position = self.positions[symbol]
                    old_realized_pnl_count = len(self.realized_pnl)

                    self.trades.append(memory_trade)
                    self._update_position(memory_trade)

                    new_position = self.positions.get(symbol)
                    new_realized_pnl = None
                    if len(self.realized_pnl) > old_realized_pnl_count:
                        new_realized_pnl = self.realized_pnl[-1]

                    # 5. 現在価格更新（指定されている場合）
                    if current_market_price and symbol in self.positions:
                        self.positions[symbol].current_price = current_market_price

                    # 中間状態をflushして整合性を確認
                    session.flush()

                    # ビジネスイベントログ
                    log_business_event(
                        "stock_sold",
                        trade_id=trade_id,
                        symbol=symbol,
                        quantity=quantity,
                        price=str(price),
                        commission=str(commission),
                        old_position=old_position.to_dict(),
                        new_position=new_position.to_dict() if new_position else None,
                        realized_pnl=new_realized_pnl.to_dict()
                        if new_realized_pnl
                        else None,
                        persisted=True,
                    )

                    sell_logger.info(
                        "株式売り注文完了（DB永続化）",
                        trade_id=trade_id,
                        db_trade_id=db_trade.id,
                        commission=str(commission),
                        realized_pnl=str(new_realized_pnl.pnl)
                        if new_realized_pnl
                        else None,
                    )
            else:
                # メモリ内のみの処理
                trade_id = self._generate_trade_id()

                memory_trade = Trade(
                    id=trade_id,
                    symbol=symbol,
                    trade_type=TradeType.SELL,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    commission=commission,
                    notes=notes,
                )

                old_position = self.positions[symbol]
                old_realized_pnl_count = len(self.realized_pnl)

                self.trades.append(memory_trade)
                self._update_position(memory_trade)

                new_position = self.positions.get(symbol)
                new_realized_pnl = None
                if len(self.realized_pnl) > old_realized_pnl_count:
                    new_realized_pnl = self.realized_pnl[-1]

                if current_market_price and symbol in self.positions:
                    self.positions[symbol].current_price = current_market_price

                log_business_event(
                    "stock_sold",
                    trade_id=trade_id,
                    symbol=symbol,
                    quantity=quantity,
                    price=str(price),
                    commission=str(commission),
                    old_position=old_position.to_dict(),
                    new_position=new_position.to_dict() if new_position else None,
                    realized_pnl=new_realized_pnl.to_dict()
                    if new_realized_pnl
                    else None,
                    persisted=False,
                )

                sell_logger.info(
                    "株式売り注文完了（メモリのみ）",
                    trade_id=trade_id,
                    commission=str(commission),
                    realized_pnl=str(new_realized_pnl.pnl)
                    if new_realized_pnl
                    else None,
                )

            # 結果データ作成
            result = {
                "success": True,
                "trade_id": trade_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": str(price),
                "commission": str(commission),
                "timestamp": timestamp.isoformat(),
                "position": new_position.to_dict() if new_position else None,
                "position_closed": new_position is None,
                "realized_pnl": new_realized_pnl.to_dict()
                if new_realized_pnl
                else None,
                "gross_proceeds": str(price * quantity - commission),
            }

            return result

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e,
                {
                    "operation": "sell_stock",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": str(price),
                    "persist_to_db": persist_to_db,
                },
            )
            sell_logger.error("株式売り注文失敗、変更をロールバック", error=str(e))
            raise

    def execute_trade_order(
        self, trade_order: Dict[str, Any], persist_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        取引注文を実行（買い/売りを統一インターフェースで処理）

        Args:
            trade_order: 取引注文辞書
                {
                    "action": "buy" | "sell",
                    "symbol": str,
                    "quantity": int,
                    "price": Decimal,
                    "current_market_price": Optional[Decimal],
                    "notes": str
                }
            persist_to_db: データベースに永続化するかどうか

        Returns:
            取引結果辞書
        """
        action = str(trade_order.get("action", "")).lower()

        if action == "buy":
            return self.buy_stock(
                symbol=trade_order["symbol"],
                quantity=trade_order["quantity"],
                price=trade_order["price"],
                current_market_price=trade_order.get("current_market_price"),
                notes=trade_order.get("notes", ""),
                persist_to_db=persist_to_db,
            )
        elif action == "sell":
            return self.sell_stock(
                symbol=trade_order["symbol"],
                quantity=trade_order["quantity"],
                price=trade_order["price"],
                current_market_price=trade_order.get("current_market_price"),
                notes=trade_order.get("notes", ""),
                persist_to_db=persist_to_db,
            )
        else:
            raise ValueError(
                f"無効な取引アクション: {action}. 'buy' または 'sell' を指定してください"
            )

    def calculate_tax_implications(self, year: int) -> Dict:
        """税務計算"""
        try:
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31, 23, 59, 59)

            # 年内の実現損益
            year_pnl = [
                pnl
                for pnl in self.realized_pnl
                if year_start <= pnl.sell_date <= year_end
            ]

            total_gain = (
                sum(pnl.pnl for pnl in year_pnl if pnl.pnl > 0)
                if year_pnl
                else Decimal("0")
            )
            total_loss = (
                sum(abs(pnl.pnl) for pnl in year_pnl if pnl.pnl < 0)
                if year_pnl
                else Decimal("0")
            )
            net_gain = total_gain - total_loss

            # 税額計算
            tax_due = Decimal("0")
            if net_gain > 0:
                tax_due = net_gain * self.tax_rate

            return {
                "year": year,
                "total_trades": len(year_pnl),
                "total_gain": str(total_gain),
                "total_loss": str(total_loss),
                "net_gain": str(net_gain),
                "tax_due": str(tax_due),
                "winning_trades": len([pnl for pnl in year_pnl if pnl.pnl > 0]),
                "losing_trades": len([pnl for pnl in year_pnl if pnl.pnl < 0]),
            }

        except Exception as e:
            logger.error(
                f"税務計算中に予期せぬエラーが発生しました。入力データまたは計算ロジックを確認してください。詳細: {e}"
            )
            raise


# 使用例
if __name__ == "__main__":
    from datetime import datetime, timedelta
    from decimal import Decimal

    # 取引管理システムを初期化
    tm = TradeManager(commission_rate=Decimal("0.001"), tax_rate=Decimal("0.2"))

    # サンプル取引を追加
    base_date = datetime.now() - timedelta(days=30)

    # トヨタ株の取引例
    tm.add_trade("7203", TradeType.BUY, 100, Decimal("2500"), base_date)
    tm.add_trade(
        "7203", TradeType.BUY, 200, Decimal("2450"), base_date + timedelta(days=1)
    )

    # 現在価格を更新
    tm.update_current_prices({"7203": Decimal("2600")})

    # ポジション情報表示
    position = tm.get_position("7203")
    if position:
        logger.info(
            "ポジション情報表示",
            section="position_info",
            symbol=position.symbol,
            quantity=position.quantity,
            average_price=str(position.average_price),
            total_cost=str(position.total_cost),
            current_price=str(position.current_price),
            market_value=str(position.market_value),
            unrealized_pnl=str(position.unrealized_pnl),
            unrealized_pnl_percent=str(position.unrealized_pnl_percent),
        )

    # 一部売却
    tm.add_trade(
        "7203", TradeType.SELL, 100, Decimal("2650"), base_date + timedelta(days=5)
    )

    # 実現損益表示
    realized_pnl = tm.get_realized_pnl_history("7203")
    if realized_pnl:
        logger.info("実現損益表示開始", section="realized_pnl")
        for pnl in realized_pnl:
            logger.info(
                "実現損益詳細",
                section="realized_pnl_detail",
                symbol=pnl.symbol,
                quantity=pnl.quantity,
                buy_price=str(pnl.buy_price),
                sell_price=str(pnl.sell_price),
                pnl=str(pnl.pnl),
                pnl_percent=str(pnl.pnl_percent),
            )

    # ポートフォリオサマリー
    summary = tm.get_portfolio_summary()
    logger.info("ポートフォリオサマリー", section="portfolio_summary", **summary)

    # CSV出力例
    logger.info("CSV出力開始", section="csv_export")
    try:
        tm.export_to_csv("trades.csv", "trades")
        tm.export_to_csv("positions.csv", "positions")
        tm.export_to_csv("realized_pnl.csv", "realized_pnl")
        logger.info("CSV出力完了", section="csv_export", status="success")
    except Exception as e:
        log_error_with_context(e, {"section": "csv_export", "operation": "export_csv"})
