"""
リスク・手数料計算

取引リスク評価・手数料計算・ポジションサイジング機能
"""

from decimal import Decimal
from typing import Dict, Optional

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class RiskCalculator:
    """
    リスク・手数料計算クラス

    取引手数料・リスク指標・ポジションサイズの計算機能を提供
    """

    def __init__(self, commission_rate: Decimal = Decimal("0.001")):
        """
        初期化

        Args:
            commission_rate: 手数料率（デフォルト0.1%）
        """
        self.commission_rate = commission_rate
        logger.info(f"リスク計算機初期化完了 - 手数料率: {commission_rate:.3%}")

    def calculate_commission(self, quantity: int, price: Decimal) -> Decimal:
        """
        取引手数料計算

        Args:
            quantity: 取引数量
            price: 取引価格

        Returns:
            計算された手数料
        """
        trade_value = Decimal(quantity) * price
        commission = trade_value * self.commission_rate

        # 最小手数料（例: 100円）
        min_commission = Decimal("100")
        commission = max(commission, min_commission)

        logger.debug(f"手数料計算: {quantity}株 × {price}円 = 手数料{commission}円")
        return commission

    def calculate_position_risk(
        self,
        quantity: int,
        entry_price: Decimal,
        current_price: Decimal,
        stop_loss_price: Optional[Decimal] = None,
    ) -> Dict[str, Decimal]:
        """
        ポジションリスク計算

        Args:
            quantity: 保有数量
            entry_price: 平均取得価格
            current_price: 現在価格
            stop_loss_price: ストップロス価格

        Returns:
            リスク指標辞書
        """
        position_value = Decimal(quantity) * current_price
        entry_value = Decimal(quantity) * entry_price

        # 含み損益
        unrealized_pnl = position_value - entry_value
        unrealized_pnl_pct = (
            (unrealized_pnl / entry_value * 100) if entry_value > 0 else Decimal("0")
        )

        # 最大リスク（ストップロス設定時）
        max_risk = Decimal("0")
        max_risk_pct = Decimal("0")
        if stop_loss_price:
            max_risk_value = Decimal(quantity) * stop_loss_price
            max_risk = max_risk_value - entry_value
            max_risk_pct = (
                (max_risk / entry_value * 100) if entry_value > 0 else Decimal("0")
            )

        risk_metrics = {
            "position_value": position_value,
            "entry_value": entry_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_percentage": unrealized_pnl_pct,
            "max_risk": max_risk,
            "max_risk_percentage": max_risk_pct,
        }

        logger.debug(f"ポジションリスク計算完了: {quantity}株, 評価額{position_value}円")
        return risk_metrics

    def calculate_position_size(
        self,
        available_capital: Decimal,
        price: Decimal,
        risk_percentage: Decimal = Decimal("2.0"),
        stop_loss_price: Optional[Decimal] = None,
    ) -> Dict[str, int]:
        """
        適正ポジションサイズ計算

        Args:
            available_capital: 利用可能資本
            price: 取引価格
            risk_percentage: リスク許容率（%）
            stop_loss_price: ストップロス価格

        Returns:
            推奨ポジションサイズ情報
        """
        # リスク許容額
        risk_amount = available_capital * (risk_percentage / 100)

        # 基本ポジションサイズ（リスク考慮なし）
        basic_position_size = int(available_capital / price)

        # リスクベースポジションサイズ
        risk_based_size = basic_position_size
        if stop_loss_price and stop_loss_price < price:
            risk_per_share = price - stop_loss_price
            risk_based_size = int(risk_amount / risk_per_share)

        # 最終推奨サイズ（より保守的な方を選択）
        recommended_size = min(basic_position_size, risk_based_size)

        # 100株単位に調整
        recommended_size = (recommended_size // 100) * 100

        position_info = {
            "basic_size": basic_position_size,
            "risk_based_size": risk_based_size,
            "recommended_size": recommended_size,
            "required_capital": recommended_size * price,
        }

        logger.info(
            f"ポジションサイズ計算: 推奨{recommended_size}株 "
            f"(必要資本{position_info['required_capital']}円)"
        )
        return position_info

    def calculate_portfolio_risk(self, positions: Dict[str, Dict]) -> Dict[str, Decimal]:
        """
        ポートフォリオ全体リスク計算

        Args:
            positions: ポジション辞書（銘柄別）

        Returns:
            ポートフォリオリスク指標
        """
        total_value = Decimal("0")
        total_pnl = Decimal("0")
        position_count = 0

        concentration_risks = {}

        for symbol, position_data in positions.items():
            position_value = Decimal(str(position_data.get("market_value", "0")))
            unrealized_pnl = Decimal(str(position_data.get("unrealized_pnl", "0")))

            total_value += position_value
            total_pnl += unrealized_pnl
            position_count += 1

            # 集中リスク計算
            concentration_risks[symbol] = position_value

        # ポートフォリオ指標計算
        portfolio_pnl_pct = (
            (total_pnl / total_value * 100) if total_value > 0 else Decimal("0")
        )

        # 最大集中度計算
        max_concentration = Decimal("0")
        max_concentration_symbol = ""
        if total_value > 0:
            for symbol, value in concentration_risks.items():
                concentration_pct = value / total_value * 100
                if concentration_pct > max_concentration:
                    max_concentration = concentration_pct
                    max_concentration_symbol = symbol

        portfolio_risk = {
            "total_value": total_value,
            "total_unrealized_pnl": total_pnl,
            "portfolio_pnl_percentage": portfolio_pnl_pct,
            "position_count": Decimal(position_count),
            "max_concentration_percentage": max_concentration,
            "max_concentration_symbol": max_concentration_symbol,
        }

        logger.info(
            f"ポートフォリオリスク計算完了: "
            f"評価額{total_value}円, {position_count}銘柄, "
            f"最大集中度{max_concentration:.1f}%({max_concentration_symbol})"
        )

        return portfolio_risk

    def validate_trade_risk(
        self,
        quantity: int,
        price: Decimal,
        available_capital: Decimal,
        max_position_size_pct: Decimal = Decimal("10.0"),
    ) -> Dict[str, bool]:
        """
        取引リスク検証

        Args:
            quantity: 取引数量
            price: 取引価格
            available_capital: 利用可能資本
            max_position_size_pct: 最大ポジションサイズ率（%）

        Returns:
            検証結果辞書
        """
        trade_value = Decimal(quantity) * price
        position_size_pct = (
            (trade_value / available_capital * 100) if available_capital > 0 else Decimal("100")
        )

        validations = {
            "sufficient_capital": trade_value <= available_capital,
            "position_size_acceptable": position_size_pct <= max_position_size_pct,
            "minimum_quantity": quantity >= 100,  # 最小単元
            "price_reasonable": price > Decimal("0"),
        }

        # 全体検証結果
        validations["overall_valid"] = all(validations.values())

        if not validations["overall_valid"]:
            logger.warning(f"取引リスク検証失敗: {quantity}株 × {price}円")
        else:
            logger.info(f"取引リスク検証成功: {quantity}株 × {price}円")

        return validations
