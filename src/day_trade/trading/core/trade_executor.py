"""
取引実行エンジン

取引記録・検証・実行の中核機能
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from ...models.enums import TradeType
from ...utils.logging_config import get_context_logger, log_business_event
from .position_manager import PositionManager
from .risk_calculator import RiskCalculator
from .types import Trade, TradeStatus

logger = get_context_logger(__name__)


class TradeExecutor:
    """
    取引実行エンジンクラス

    取引の検証・記録・実行・管理の中核機能を提供
    """

    def __init__(
        self,
        position_manager: PositionManager,
        risk_calculator: RiskCalculator,
    ):
        """
        初期化

        Args:
            position_manager: ポジション管理インスタンス
            risk_calculator: リスク計算インスタンス
        """
        self.position_manager = position_manager
        self.risk_calculator = risk_calculator
        self.trades: List[Trade] = []
        self.trade_history: Dict[str, List[Trade]] = {}  # 銘柄別取引履歴

        logger.info("取引実行エンジン初期化完了")

    def add_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: int,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        notes: str = "",
    ) -> Optional[Trade]:
        """
        取引記録追加

        Args:
            symbol: 銘柄コード
            trade_type: 取引タイプ
            quantity: 数量
            price: 価格
            timestamp: 取引時刻（指定しない場合は現在時刻）
            notes: メモ

        Returns:
            作成された取引記録（失敗時はNone）
        """
        try:
            # 取引時刻設定
            if timestamp is None:
                timestamp = datetime.now()

            # 手数料計算
            commission = self.risk_calculator.calculate_commission(quantity, price)

            # 取引ID生成
            trade_id = self._generate_trade_id()

            # 取引記録作成
            trade = Trade(
                id=trade_id,
                symbol=symbol,
                trade_type=trade_type,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                commission=commission,
                status=TradeStatus.EXECUTED,
                notes=notes,
            )

            # 売却時の数量検証
            if trade_type == TradeType.SELL:
                validation = self.position_manager.validate_sell_quantity(symbol, quantity)
                if not validation["overall_valid"]:
                    logger.error(f"売却検証失敗: {symbol} {quantity}株")
                    return None

            # 取引記録を追加
            self.trades.append(trade)

            # 銘柄別履歴に追加
            if symbol not in self.trade_history:
                self.trade_history[symbol] = []
            self.trade_history[symbol].append(trade)

            # ポジション更新
            self.position_manager.update_position_from_trade(trade)

            # ビジネスイベントログ
            log_business_event(
                f"取引実行: {symbol} {trade_type.value} {quantity}株 @{price}円",
                {"trade_id": trade_id, "symbol": symbol, "type": trade_type.value}
            )

            logger.info(
                f"取引記録追加完了: {trade_id} {symbol} {trade_type.value} "
                f"{quantity}株 @{price}円 (手数料{commission}円)"
            )

            return trade

        except Exception as e:
            logger.error(f"取引記録追加エラー: {e}")
            return None

    def buy_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        validate_risk: bool = True,
        available_capital: Optional[Decimal] = None,
        notes: str = "",
    ) -> Optional[Trade]:
        """
        株式購入

        Args:
            symbol: 銘柄コード
            quantity: 購入数量
            price: 購入価格
            validate_risk: リスク検証実行（デフォルトTrue）
            available_capital: 利用可能資本（リスク検証用）
            notes: メモ

        Returns:
            取引記録（失敗時はNone）
        """
        # リスク検証
        if validate_risk and available_capital:
            risk_validation = self.risk_calculator.validate_trade_risk(
                quantity, price, available_capital
            )
            if not risk_validation["overall_valid"]:
                logger.warning(f"購入リスク検証失敗: {symbol} {quantity}株 @{price}円")
                return None

        # 取引実行
        trade = self.add_trade(
            symbol=symbol,
            trade_type=TradeType.BUY,
            quantity=quantity,
            price=price,
            notes=notes,
        )

        if trade:
            logger.info(f"株式購入成功: {symbol} {quantity}株 @{price}円")
        else:
            logger.error(f"株式購入失敗: {symbol} {quantity}株 @{price}円")

        return trade

    def sell_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        notes: str = "",
    ) -> Optional[Trade]:
        """
        株式売却

        Args:
            symbol: 銘柄コード
            quantity: 売却数量
            price: 売却価格
            notes: メモ

        Returns:
            取引記録（失敗時はNone）
        """
        # 売却可能性検証（add_trade内でも実行されるが事前チェック）
        position = self.position_manager.get_position(symbol)
        if not position:
            logger.error(f"売却失敗: {symbol}のポジションが存在しません")
            return None

        if position.quantity < quantity:
            logger.error(
                f"売却失敗: {symbol}の保有数量{position.quantity}株 < "
                f"売却数量{quantity}株"
            )
            return None

        # 取引実行
        trade = self.add_trade(
            symbol=symbol,
            trade_type=TradeType.SELL,
            quantity=quantity,
            price=price,
            notes=notes,
        )

        if trade:
            logger.info(f"株式売却成功: {symbol} {quantity}株 @{price}円")
        else:
            logger.error(f"株式売却失敗: {symbol} {quantity}株 @{price}円")

        return trade

    def execute_trade_order(
        self,
        order_data: Dict,
        available_capital: Optional[Decimal] = None,
    ) -> Optional[Trade]:
        """
        統一取引実行インターフェース

        Args:
            order_data: 注文データ辞書
            available_capital: 利用可能資本

        Returns:
            取引記録（失敗時はNone）
        """
        required_fields = ["symbol", "trade_type", "quantity", "price"]

        # 必須フィールド検証
        for field in required_fields:
            if field not in order_data:
                logger.error(f"注文データ不正: 必須フィールド'{field}'が不足")
                return None

        symbol = order_data["symbol"]
        trade_type_str = order_data["trade_type"]
        quantity = int(order_data["quantity"])
        price = Decimal(str(order_data["price"]))
        notes = order_data.get("notes", "")

        # 取引タイプ変換
        try:
            trade_type = TradeType(trade_type_str)
        except ValueError:
            logger.error(f"無効な取引タイプ: {trade_type_str}")
            return None

        # 取引実行
        if trade_type == TradeType.BUY:
            return self.buy_stock(
                symbol=symbol,
                quantity=quantity,
                price=price,
                available_capital=available_capital,
                notes=notes,
            )
        elif trade_type == TradeType.SELL:
            return self.sell_stock(
                symbol=symbol,
                quantity=quantity,
                price=price,
                notes=notes,
            )
        else:
            logger.error(f"未サポートの取引タイプ: {trade_type}")
            return None

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """
        取引履歴取得

        Args:
            symbol: 特定銘柄のみ（指定しない場合は全銘柄）
            limit: 取得件数制限

        Returns:
            取引履歴リスト
        """
        if symbol:
            trades = self.trade_history.get(symbol, [])
        else:
            trades = self.trades.copy()

        # 時刻順でソート（新しい順）
        trades.sort(key=lambda t: t.timestamp, reverse=True)

        # 件数制限
        if limit:
            trades = trades[:limit]

        return trades

    def get_trade_statistics(self) -> Dict:
        """
        取引統計取得

        Returns:
            取引統計辞書
        """
        total_trades = len(self.trades)
        buy_trades = sum(1 for t in self.trades if t.trade_type == TradeType.BUY)
        sell_trades = sum(1 for t in self.trades if t.trade_type == TradeType.SELL)

        total_buy_value = sum(
            t.price * Decimal(t.quantity) + t.commission
            for t in self.trades if t.trade_type == TradeType.BUY
        )
        total_sell_value = sum(
            t.price * Decimal(t.quantity) - t.commission
            for t in self.trades if t.trade_type == TradeType.SELL
        )
        total_commission = sum(t.commission for t in self.trades)

        symbols_traded = len(set(t.symbol for t in self.trades))

        statistics = {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "total_buy_value": total_buy_value,
            "total_sell_value": total_sell_value,
            "total_commission": total_commission,
            "symbols_traded": symbols_traded,
        }

        return statistics

    def clear_trade_history(self) -> None:
        """
        取引履歴クリア
        """
        cleared_count = len(self.trades)
        self.trades.clear()
        self.trade_history.clear()

        logger.info(f"取引履歴クリア完了: {cleared_count}件")

    def _generate_trade_id(self) -> str:
        """
        取引ID生成

        Returns:
            一意の取引ID
        """
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"TRD_{timestamp_str}_{unique_id}"

    def export_trades_summary(self) -> Dict:
        """
        取引サマリーエクスポート

        Returns:
            エクスポート用取引データ
        """
        export_data = {
            "trade_statistics": self.get_trade_statistics(),
            "trades": [trade.to_dict() for trade in self.trades],
            "trade_history_by_symbol": {}
        }

        # 銘柄別履歴
        for symbol, trades in self.trade_history.items():
            export_data["trade_history_by_symbol"][symbol] = [
                trade.to_dict() for trade in trades
            ]

        return export_data
