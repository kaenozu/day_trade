"""
ポジション管理

ポジション追跡・更新・計算・サマリー機能
"""

from decimal import Decimal
from typing import Dict, List, Optional

from ...utils.logging_config import get_context_logger
from .types import Position, Trade, TradeType

logger = get_context_logger(__name__)


class PositionManager:
    """
    ポジション管理クラス

    取引に基づくポジション追跡・更新・計算機能を提供
    """

    def __init__(self):
        """初期化"""
        self.positions: Dict[str, Position] = {}
        logger.info("ポジションマネージャー初期化完了")

    def update_position_from_trade(self, trade: Trade) -> None:
        """
        取引からポジション更新

        Args:
            trade: 取引記録
        """
        symbol = trade.symbol
        current_position = self.positions.get(symbol)

        if trade.trade_type == TradeType.BUY:
            self._handle_buy_trade(trade, current_position)
        elif trade.trade_type == TradeType.SELL:
            self._handle_sell_trade(trade, current_position)

        logger.debug(f"ポジション更新完了: {symbol}")

    def _handle_buy_trade(self, trade: Trade, current_position: Optional[Position]) -> None:
        """
        買い取引処理

        Args:
            trade: 買い取引記録
            current_position: 現在のポジション
        """
        symbol = trade.symbol
        trade_cost = trade.price * Decimal(trade.quantity) + trade.commission

        if current_position is None:
            # 新規ポジション
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=trade.quantity,
                average_price=trade.price,
                total_cost=trade_cost,
            )
            logger.info(f"新規ポジション作成: {symbol} {trade.quantity}株")

        else:
            # 既存ポジション追加
            new_quantity = current_position.quantity + trade.quantity
            new_total_cost = current_position.total_cost + trade_cost

            # 平均取得価格再計算
            new_average_price = new_total_cost / Decimal(new_quantity)

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=new_quantity,
                average_price=new_average_price,
                total_cost=new_total_cost,
                current_price=current_position.current_price,
            )
            logger.info(f"ポジション追加: {symbol} {trade.quantity}株追加, 計{new_quantity}株")

    def _handle_sell_trade(self, trade: Trade, current_position: Optional[Position]) -> None:
        """
        売り取引処理

        Args:
            trade: 売り取引記録
            current_position: 現在のポジション
        """
        symbol = trade.symbol

        if current_position is None:
            logger.error(f"売却エラー: {symbol}のポジションが存在しません")
            return

        if current_position.quantity < trade.quantity:
            logger.error(
                f"売却エラー: {symbol}の保有数量{current_position.quantity}株 < "
                f"売却数量{trade.quantity}株"
            )
            return

        # 新しい保有数量計算
        remaining_quantity = current_position.quantity - trade.quantity

        if remaining_quantity == 0:
            # ポジション完全クローズ
            del self.positions[symbol]
            logger.info(f"ポジション完全売却: {symbol}")

        else:
            # 部分売却 - 平均取得価格は変更しない
            # 売却分に対応するコスト減算
            cost_per_share = current_position.total_cost / Decimal(current_position.quantity)
            remaining_cost = cost_per_share * Decimal(remaining_quantity)

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining_quantity,
                average_price=current_position.average_price,
                total_cost=remaining_cost,
                current_price=current_position.current_price,
            )
            logger.info(
                f"ポジション部分売却: {symbol} {trade.quantity}株売却, "
                f"残り{remaining_quantity}株"
            )

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        特定銘柄のポジション取得

        Args:
            symbol: 銘柄コード

        Returns:
            ポジション（存在しない場合はNone）
        """
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """
        全ポジション取得

        Returns:
            全ポジション辞書
        """
        return self.positions.copy()

    def update_current_prices(self, price_data: Dict[str, Decimal]) -> None:
        """
        現在価格一括更新

        Args:
            price_data: 価格データ辞書 {銘柄コード: 価格}
        """
        updated_count = 0
        for symbol, position in self.positions.items():
            if symbol in price_data:
                position.current_price = price_data[symbol]
                updated_count += 1

        logger.info(f"現在価格更新完了: {updated_count}銘柄")

    def get_portfolio_summary(self) -> Dict[str, Decimal]:
        """
        ポートフォリオサマリー計算

        Returns:
            ポートフォリオサマリー辞書
        """
        total_cost = Decimal("0")
        total_market_value = Decimal("0")
        total_unrealized_pnl = Decimal("0")

        for position in self.positions.values():
            total_cost += position.total_cost
            total_market_value += position.market_value
            total_unrealized_pnl += position.unrealized_pnl

        # ポートフォリオ指標計算
        portfolio_return_pct = (
            (total_unrealized_pnl / total_cost * 100) if total_cost > 0 else Decimal("0")
        )

        portfolio_summary = {
            "total_positions": Decimal(len(self.positions)),
            "total_cost": total_cost,
            "total_market_value": total_market_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "portfolio_return_percentage": portfolio_return_pct,
        }

        logger.debug(
            f"ポートフォリオサマリー: 評価額{total_market_value}円, {len(self.positions)}銘柄"
        )
        return portfolio_summary

    def get_positions_by_value(self, descending: bool = True) -> List[Position]:
        """
        評価額順でポジションリスト取得

        Args:
            descending: 降順（デフォルトTrue）

        Returns:
            評価額順ポジションリスト
        """
        positions_list = list(self.positions.values())
        positions_list.sort(key=lambda p: p.market_value, reverse=descending)

        return positions_list

    def get_positions_by_pnl(self, descending: bool = True) -> List[Position]:
        """
        含み損益順でポジションリスト取得

        Args:
            descending: 降順（デフォルトTrue）

        Returns:
            含み損益順ポジションリスト
        """
        positions_list = list(self.positions.values())
        positions_list.sort(key=lambda p: p.unrealized_pnl, reverse=descending)

        return positions_list

    def get_concentration_analysis(self) -> Dict[str, Dict]:
        """
        集中度分析

        Returns:
            集中度分析結果
        """
        if not self.positions:
            return {}

        portfolio_summary = self.get_portfolio_summary()
        total_value = portfolio_summary["total_market_value"]

        concentration_data = {}
        for symbol, position in self.positions.items():
            concentration_pct = (
                (position.market_value / total_value * 100) if total_value > 0 else Decimal("0")
            )

            concentration_data[symbol] = {
                "market_value": position.market_value,
                "concentration_percentage": concentration_pct,
                "quantity": position.quantity,
                "unrealized_pnl": position.unrealized_pnl,
            }

        # 集中度順にソート
        concentration_data = dict(
            sorted(
                concentration_data.items(),
                key=lambda x: x[1]["concentration_percentage"],
                reverse=True,
            )
        )

        logger.debug(f"集中度分析完了: {len(concentration_data)}銘柄")
        return concentration_data

    def validate_sell_quantity(self, symbol: str, quantity: int) -> Dict[str, bool]:
        """
        売却可能数量検証

        Args:
            symbol: 銘柄コード
            quantity: 売却予定数量

        Returns:
            検証結果辞書
        """
        position = self.positions.get(symbol)

        validations = {
            "position_exists": position is not None,
            "sufficient_quantity": False,
            "valid_quantity": quantity > 0,
        }

        if position:
            validations["sufficient_quantity"] = position.quantity >= quantity

        validations["overall_valid"] = all(validations.values())

        if not validations["overall_valid"]:
            logger.warning(f"売却検証失敗: {symbol} {quantity}株")

        return validations

    def clear_all_positions(self) -> None:
        """
        全ポジションクリア
        """
        cleared_count = len(self.positions)
        self.positions.clear()
        logger.info(f"全ポジションクリア完了: {cleared_count}銘柄")

    def export_positions_summary(self) -> Dict:
        """
        ポジションサマリーエクスポート

        Returns:
            エクスポート用ポジションデータ
        """
        export_data = {
            "portfolio_summary": self.get_portfolio_summary(),
            "positions": {},
            "concentration_analysis": self.get_concentration_analysis(),
        }

        # 各ポジション詳細
        for symbol, position in self.positions.items():
            export_data["positions"][symbol] = position.to_dict()

        return export_data
