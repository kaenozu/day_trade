from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..models.database import db_manager
from ..models.enums import TradeType
from ..models.stock import Stock
from ..models.stock import Trade as DBTrade
from ..utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)
from .trade_models import Trade, TradeStatus
from .trade_utils import (
    mask_sensitive_info,
    safe_decimal_conversion,
    validate_positive_decimal,
)


class OrderManager:
    def __init__(self, trade_manager):
        self.tm = trade_manager
        self.logger = get_context_logger(__name__)

    def _generate_trade_id(self) -> str:
        """取引IDを生成"""
        self.tm._trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d')}{self.tm._trade_counter:04d}"

    def _calculate_commission(self, price: Decimal, quantity: int) -> Decimal:
        """手数料を計算"""
        total_value = price * Decimal(quantity)
        commission = total_value * self.tm.commission_rate
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
        """
        try:
            safe_price = safe_decimal_conversion(price, "取引価格")
            validate_positive_decimal(safe_price, "取引価格")

            if not isinstance(quantity, int) or quantity <= 0:
                raise ValueError(f"数量は正の整数である必要があります: {quantity}")

            context_info = {
                "operation": "add_trade",
                "symbol": symbol,
                "trade_type": trade_type.value,
                "quantity": quantity,
                "price_masked": mask_sensitive_info(str(safe_price)),
                "persist_to_db": persist_to_db,
            }

            self.logger.info("取引追加処理開始", extra=context_info)

            if timestamp is None:
                timestamp = datetime.now()

            if commission is None:
                safe_commission = self._calculate_commission(safe_price, quantity)
            else:
                safe_commission = safe_decimal_conversion(commission, "手数料")
                validate_positive_decimal(safe_commission, "手数料", allow_zero=True)

            trade_id = self._generate_trade_id()
            memory_trade = Trade(
                id=trade_id,
                symbol=symbol,
                trade_type=trade_type,
                quantity=quantity,
                price=safe_price,
                timestamp=timestamp,
                commission=safe_commission,
                notes=notes,
            )

            if persist_to_db:
                with db_manager.transaction_scope() as session:
                    stock = session.query(Stock).filter(Stock.code == symbol).first()
                    if not stock:
                        self.logger.info("銘柄マスタに未登録、新規作成", extra=context_info)
                        stock = Stock(
                            code=symbol,
                            name=symbol,  # 名前が不明な場合はコードを使用
                            market="未定",
                            sector="未定",
                            industry="未定",
                        )
                        session.add(stock)
                        session.flush()

                    db_trade = (
                        DBTrade.create_buy_trade(
                            session=session,
                            stock_code=symbol,
                            quantity=quantity,
                            price=safe_price,
                            commission=safe_commission,
                            memo=notes,
                        )
                        if trade_type == TradeType.BUY
                        else DBTrade.create_sell_trade(
                            session=session,
                            stock_code=symbol,
                            quantity=quantity,
                            price=safe_price,
                            commission=safe_commission,
                            memo=notes,
                        )
                    )

                    self.tm.trades.append(memory_trade)
                    self.tm._update_position(memory_trade)

                    session.flush()

                    log_business_event(
                        "trade_added",
                        trade_id=mask_sensitive_info(str(trade_id)),
                        symbol=symbol,
                        trade_type=trade_type.value,
                        quantity=mask_sensitive_info(f"quantity: {quantity}"),
                        price=mask_sensitive_info(f"price: {str(price)}"),
                        commission=mask_sensitive_info(
                            f"commission: {str(commission)}"
                        ),
                        persisted=True,
                    )

                    self.logger.info(
                        "取引追加完了（DB永続化）",
                        extra={
                            **context_info,
                            "trade_id": mask_sensitive_info(str(trade_id)),
                            "db_trade_id": mask_sensitive_info(str(db_trade.id)),
                        },
                    )
            else:
                self.tm.trades.append(memory_trade)
                self.tm._update_position(memory_trade)

                log_business_event(
                    "trade_added",
                    trade_id=mask_sensitive_info(str(trade_id)),
                    symbol=symbol,
                    trade_type=trade_type.value,
                    quantity=mask_sensitive_info(f"quantity: {quantity}"),
                    price=mask_sensitive_info(f"price: {str(price)}"),
                    commission=mask_sensitive_info(f"commission: {str(commission)}"),
                    persisted=False,
                )

                self.logger.info(
                    "取引追加完了（メモリのみ）",
                    extra={
                        **context_info,
                        "trade_id": mask_sensitive_info(str(trade_id)),
                    },
                )

            return trade_id

        except Exception as e:
            self.logger.error(f"取引追加エラー: {mask_sensitive_info(str(e))}")
            log_error_with_context(
                e,
                {
                    "operation": "add_trade",
                    "symbol": symbol,
                    "trade_type": trade_type.value,
                    "quantity": quantity,
                    "price_masked": mask_sensitive_info(str(price)),
                    "persist_to_db": persist_to_db,
                },
            )
            raise

    def add_trades_batch(
        self, trades_data: List[Dict], persist_to_db: bool = True
    ) -> List[str]:
        trade_ids = []
        with db_manager.transaction_scope() as session:
            for trade_data in trades_data:
                trade_id = self.add_trade(
                    session=session,
                    persist_to_db=persist_to_db,
                    **trade_data
                )
                trade_ids.append(trade_id)
        return trade_ids

    def buy_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, Any]:
        return self.add_trade(
            symbol=symbol,
            trade_type=TradeType.BUY,
            quantity=quantity,
            price=price,
            notes=notes,
            persist_to_db=persist_to_db,
        )

    def sell_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, Any]:
        return self.add_trade(
            symbol=symbol,
            trade_type=TradeType.SELL,
            quantity=quantity,
            price=price,
            notes=notes,
            persist_to_db=persist_to_db,
        )

    def execute_trade_order(
        self, trade_order: Dict[str, Any], persist_to_db: bool = True
    ) -> Dict[str, Any]:
        trade_type = TradeType(trade_order["trade_type"])
        if trade_type == TradeType.BUY:
            return self.buy_stock(persist_to_db=persist_to_db, **trade_order)
        else:
            return self.sell_stock(persist_to_db=persist_to_db, **trade_order)
