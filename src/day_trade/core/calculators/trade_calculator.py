"""
取引計算エンジン

売買損益、手数料、税金計算など、取引に関する数値計算を担当
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import List, Tuple, Union
from datetime import datetime

from ...utils.logging_config import get_context_logger
from ..models.trade_models import BuyLot, RealizedPnL, Trade

logger = get_context_logger(__name__)


class DecimalCalculator:
    """高精度小数計算クラス"""

    @staticmethod
    def safe_decimal_conversion(
        value: Union[str, int, float, Decimal],
        context: str = "値"
    ) -> Decimal:
        """
        安全なDecimal変換（浮動小数点誤差回避）

        Args:
            value: 変換する値
            context: エラー時のコンテキスト情報

        Returns:
            Decimal: 変換された値

        Raises:
            ValueError: 変換できない値の場合
        """
        try:
            if isinstance(value, Decimal):
                return value
            elif isinstance(value, (int, float)):
                return Decimal(str(value))
            elif isinstance(value, str):
                # 空文字列やNoneの処理
                if not value or value.lower() in ('none', 'null', 'nan'):
                    return Decimal('0')
                return Decimal(value)
            else:
                logger.warning(f"未対応の型からDecimal変換: {type(value)}")
                return Decimal(str(value))
        except (InvalidOperation, ValueError) as e:
            error_msg = f"{context}のDecimal変換エラー: {value} ({type(value)})"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    @staticmethod
    def round_currency(amount: Decimal, places: int = 0) -> Decimal:
        """通貨として適切な丸め処理"""
        return amount.quantize(Decimal('0.1') ** places, rounding=ROUND_HALF_UP)


class CommissionCalculator:
    """手数料計算クラス"""

    DEFAULT_COMMISSION_RATE = Decimal('0.001')  # 0.1%
    MIN_COMMISSION = Decimal('100')  # 最低手数料100円
    MAX_COMMISSION = Decimal('5000')  # 最大手数料5000円

    @classmethod
    def calculate_commission(
        cls,
        trade_amount: Decimal,
        commission_rate: Decimal = None
    ) -> Decimal:
        """
        手数料計算

        Args:
            trade_amount: 取引金額
            commission_rate: 手数料率（デフォルト: 0.1%）

        Returns:
            Decimal: 計算された手数料
        """
        if commission_rate is None:
            commission_rate = cls.DEFAULT_COMMISSION_RATE

        commission = trade_amount * commission_rate

        # 最低・最大手数料の適用
        if commission < cls.MIN_COMMISSION:
            commission = cls.MIN_COMMISSION
        elif commission > cls.MAX_COMMISSION:
            commission = cls.MAX_COMMISSION

        return DecimalCalculator.round_currency(commission)


class PnLCalculator:
    """損益計算クラス"""

    @staticmethod
    def calculate_fifo_pnl(
        sell_quantity: int,
        sell_price: Decimal,
        sell_commission: Decimal,
        buy_lots: List[BuyLot]
    ) -> Tuple[List[RealizedPnL], List[BuyLot]]:
        """
        FIFO法による損益計算

        Args:
            sell_quantity: 売却数量
            sell_price: 売却価格
            sell_commission: 売却手数料
            buy_lots: 買い建て玉リスト

        Returns:
            Tuple[List[RealizedPnL], List[BuyLot]]: 実現損益リストと更新された建て玉リスト
        """
        realized_pnl_list = []
        remaining_sell_quantity = sell_quantity
        updated_buy_lots = []

        for buy_lot in buy_lots:
            if remaining_sell_quantity <= 0:
                updated_buy_lots.append(buy_lot)
                continue

            if buy_lot.remaining_quantity <= 0:
                updated_buy_lots.append(buy_lot)
                continue

            # 売却数量の決定
            sell_from_this_lot = min(remaining_sell_quantity, buy_lot.remaining_quantity)

            # 手数料の按分計算
            sell_commission_portion = (
                sell_commission * sell_from_this_lot / sell_quantity
                if sell_quantity > 0 else Decimal('0')
            )

            buy_commission_portion = (
                buy_lot.commission * sell_from_this_lot / buy_lot.quantity
                if buy_lot.quantity > 0 else Decimal('0')
            )

            # 損益計算
            pnl_before_commission = (sell_price - buy_lot.price) * sell_from_this_lot
            net_pnl = pnl_before_commission - sell_commission_portion - buy_commission_portion

            # 実現損益レコード作成
            realized_pnl = RealizedPnL(
                symbol=buy_lot.symbol,
                quantity=sell_from_this_lot,
                buy_price=buy_lot.price,
                sell_price=sell_price,
                buy_timestamp=buy_lot.timestamp,
                sell_timestamp=datetime.now(),
                pnl_before_commission=pnl_before_commission,
                buy_commission=buy_commission_portion,
                sell_commission=sell_commission_portion,
                net_pnl=net_pnl
            )

            realized_pnl_list.append(realized_pnl)

            # 建て玉の残数量更新
            updated_lot = BuyLot(
                symbol=buy_lot.symbol,
                quantity=buy_lot.quantity,
                price=buy_lot.price,
                timestamp=buy_lot.timestamp,
                remaining_quantity=buy_lot.remaining_quantity - sell_from_this_lot,
                commission=buy_lot.commission
            )

            if updated_lot.remaining_quantity > 0:
                updated_buy_lots.append(updated_lot)

            remaining_sell_quantity -= sell_from_this_lot

        # 残りの建て玉を追加
        if remaining_sell_quantity > 0:
            logger.warning(f"売却数量が建て玉を超過: {remaining_sell_quantity}")

        return realized_pnl_list, updated_buy_lots

    @staticmethod
    def calculate_unrealized_pnl(
        buy_lots: List[BuyLot],
        current_price: Decimal
    ) -> Decimal:
        """
        未実現損益計算

        Args:
            buy_lots: 買い建て玉リスト
            current_price: 現在価格

        Returns:
            Decimal: 未実現損益
        """
        total_unrealized = Decimal('0')

        for lot in buy_lots:
            if lot.remaining_quantity > 0:
                market_value = current_price * lot.remaining_quantity
                cost_basis = lot.average_price_with_commission * lot.remaining_quantity
                unrealized = market_value - cost_basis
                total_unrealized += unrealized

        return total_unrealized


class PositionCalculator:
    """ポジション計算クラス"""

    @staticmethod
    def calculate_average_price(buy_lots: List[BuyLot]) -> Decimal:
        """
        平均取得価格計算

        Args:
            buy_lots: 買い建て玉リスト

        Returns:
            Decimal: 平均取得価格
        """
        total_quantity = sum(lot.remaining_quantity for lot in buy_lots)
        if total_quantity == 0:
            return Decimal('0')

        total_cost = sum(
            lot.price * lot.remaining_quantity +
            lot.commission * lot.remaining_quantity / lot.quantity
            for lot in buy_lots
        )

        return DecimalCalculator.round_currency(
            total_cost / total_quantity, places=2
        )

    @staticmethod
    def calculate_total_commission(buy_lots: List[BuyLot]) -> Decimal:
        """
        総手数料計算

        Args:
            buy_lots: 買い建て玉リスト

        Returns:
            Decimal: 総手数料
        """
        return sum(
            lot.commission * lot.remaining_quantity / lot.quantity
            for lot in buy_lots
            if lot.quantity > 0
        )