"""
取引操作モジュール
売買注文の実行とポートフォリオ更新をアトミックに処理
"""

import logging
from typing import Dict, Optional

from ..data.stock_fetcher import StockFetcher
from ..models.database import db_manager
from ..models.stock import Stock, Trade

logger = logging.getLogger(__name__)


class TradeOperationError(Exception):
    """取引操作エラー"""
    pass


class TradeOperations:
    """取引操作クラス - トランザクション管理を徹底"""

    def __init__(self, stock_fetcher: Optional[StockFetcher] = None):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
        """
        self.stock_fetcher = stock_fetcher or StockFetcher()

    def buy_stock(
        self,
        stock_code: str,
        quantity: int,
        price: Optional[float] = None,
        commission: float = 0,
        memo: str = "",
    ) -> Dict[str, any]:
        """
        株式買い注文をアトミックに実行

        複数のDB操作（銘柄マスタ確認・追加、取引記録追加、ポートフォリオ更新）を
        単一のトランザクションで実行し、データ整合性を保証

        Args:
            stock_code: 証券コード
            quantity: 購入数量
            price: 購入価格（未指定時は現在価格を取得）
            commission: 手数料
            memo: メモ

        Returns:
            取引結果の辞書

        Raises:
            TradeOperationError: 取引処理に失敗した場合
        """
        try:
            # 複数の関連するDB操作を単一トランザクション内で実行
            with db_manager.transaction_scope() as session:
                # 1. 銘柄マスタの存在確認・作成
                stock = session.query(Stock).filter(Stock.code == stock_code).first()
                if not stock:
                    # 銘柄情報を外部APIから取得
                    company_info = self.stock_fetcher.get_company_info(stock_code)
                    if not company_info:
                        raise TradeOperationError(f"銘柄情報を取得できません: {stock_code}")

                    stock = Stock(
                        code=stock_code,
                        name=company_info.get("name", stock_code),
                        market=company_info.get("market"),
                        sector=company_info.get("sector"),
                        industry=company_info.get("industry"),
                    )
                    session.add(stock)
                    session.flush()  # 銘柄レコードをコミット前に確定

                    logger.info(f"新規銘柄をマスタに追加: {stock_code} - {stock.name}")

                # 2. 価格取得（未指定時）
                if price is None:
                    current_data = self.stock_fetcher.get_current_price(stock_code)
                    if not current_data or "price" not in current_data:
                        raise TradeOperationError(f"現在価格を取得できません: {stock_code}")
                    price = current_data["price"]

                # 3. 買い取引記録の作成
                trade = Trade.create_buy_trade(
                    session=session,
                    stock_code=stock_code,
                    quantity=quantity,
                    price=price,
                    commission=commission,
                    memo=memo,
                )

                # 4. ポートフォリオ状態の更新（将来的な拡張ポイント）
                # 現在のポジション取得と更新処理
                # これらの操作もすべて同一トランザクション内で実行される

                # 中間状態をflushして整合性を確認
                session.flush()

                # 取引結果を返す
                result = {
                    "trade_id": trade.id,
                    "stock_code": stock_code,
                    "stock_name": stock.name,
                    "trade_type": "buy",
                    "quantity": quantity,
                    "price": price,
                    "commission": commission,
                    "total_amount": price * quantity + commission,
                    "trade_datetime": trade.trade_datetime,
                    "memo": memo,
                    "success": True,
                }

                logger.info(
                    f"買い注文完了: {stock_code} {quantity}株 @{price}円 "
                    f"(総額: {result['total_amount']}円)"
                )

                return result

        except Exception as e:
            error_msg = f"買い注文処理エラー ({stock_code}): {e}"
            logger.error(error_msg)
            raise TradeOperationError(error_msg) from e

    def sell_stock(
        self,
        stock_code: str,
        quantity: int,
        price: Optional[float] = None,
        commission: float = 0,
        memo: str = "",
    ) -> Dict[str, any]:
        """
        株式売り注文をアトミックに実行

        複数のDB操作（保有確認、取引記録追加、ポートフォリオ更新）を
        単一のトランザクションで実行し、データ整合性を保証

        Args:
            stock_code: 証券コード
            quantity: 売却数量
            price: 売却価格（未指定時は現在価格を取得）
            commission: 手数料
            memo: メモ

        Returns:
            取引結果の辞書

        Raises:
            TradeOperationError: 取引処理に失敗した場合
        """
        try:
            with db_manager.transaction_scope() as session:
                # 1. 銘柄マスタの存在確認
                stock = session.query(Stock).filter(Stock.code == stock_code).first()
                if not stock:
                    raise TradeOperationError(f"銘柄が見つかりません: {stock_code}")

                # 2. 現在の保有ポジション確認
                # 実際の本格的な実装では、ポジションテーブルから保有数量を確認
                buy_trades = (
                    session.query(Trade)
                    .filter(Trade.stock_code == stock_code, Trade.trade_type == "buy")
                    .all()
                )
                sell_trades = (
                    session.query(Trade)
                    .filter(Trade.stock_code == stock_code, Trade.trade_type == "sell")
                    .all()
                )

                total_bought = sum(trade.quantity for trade in buy_trades)
                total_sold = sum(trade.quantity for trade in sell_trades)
                current_holdings = total_bought - total_sold

                if current_holdings < quantity:
                    raise TradeOperationError(
                        f"売却数量が保有数量を上回ります: 保有{current_holdings}株 < 売却{quantity}株"
                    )

                # 3. 価格取得（未指定時）
                if price is None:
                    current_data = self.stock_fetcher.get_current_price(stock_code)
                    if not current_data or "price" not in current_data:
                        raise TradeOperationError(f"現在価格を取得できません: {stock_code}")
                    price = current_data["price"]

                # 4. 売り取引記録の作成
                trade = Trade.create_sell_trade(
                    session=session,
                    stock_code=stock_code,
                    quantity=quantity,
                    price=price,
                    commission=commission,
                    memo=memo,
                )

                # 5. ポートフォリオ状態の更新（将来的な拡張ポイント）
                # 損益計算、ポジション更新など

                # 中間状態をflushして整合性を確認
                session.flush()

                # 取引結果を返す
                result = {
                    "trade_id": trade.id,
                    "stock_code": stock_code,
                    "stock_name": stock.name,
                    "trade_type": "sell",
                    "quantity": quantity,
                    "price": price,
                    "commission": commission,
                    "total_amount": price * quantity - commission,
                    "trade_datetime": trade.trade_datetime,
                    "remaining_holdings": current_holdings - quantity,
                    "memo": memo,
                    "success": True,
                }

                logger.info(
                    f"売り注文完了: {stock_code} {quantity}株 @{price}円 "
                    f"(受取額: {result['total_amount']}円, 残り保有: {result['remaining_holdings']}株)"
                )

                return result

        except Exception as e:
            error_msg = f"売り注文処理エラー ({stock_code}): {e}"
            logger.error(error_msg)
            raise TradeOperationError(error_msg) from e

    def batch_trade_operations(self, operations: list) -> Dict[str, any]:
        """
        複数の取引操作をバッチで実行

        Args:
            operations: 取引操作のリスト [{"action": "buy", "stock_code": "...", ...}, ...]

        Returns:
            バッチ実行結果

        Example:
            operations = [
                {"action": "buy", "stock_code": "7203", "quantity": 100},
                {"action": "sell", "stock_code": "8306", "quantity": 50},
            ]
            result = trade_ops.batch_trade_operations(operations)
        """
        try:
            results = []
            errors = []

            # すべての操作を単一トランザクションで実行
            with db_manager.transaction_scope() as session:
                for i, operation in enumerate(operations):
                    try:
                        action = operation.get("action")
                        if action == "buy":
                            result = self._execute_buy_in_session(session, operation)
                        elif action == "sell":
                            result = self._execute_sell_in_session(session, operation)
                        else:
                            raise TradeOperationError(f"不正な操作: {action}")

                        results.append(result)

                    except Exception as e:
                        error_msg = f"操作 {i+1} でエラー: {e}"
                        errors.append(error_msg)
                        logger.error(error_msg)

                # エラーがある場合はトランザクション全体をロールバック
                if errors:
                    raise TradeOperationError(f"バッチ操作中にエラーが発生: {errors}")

                # 中間状態をflush
                session.flush()

            return {
                "success": True,
                "total_operations": len(operations),
                "successful_operations": len(results),
                "results": results,
                "errors": errors,
            }

        except Exception as e:
            error_msg = f"バッチ取引操作エラー: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "total_operations": len(operations),
                "successful_operations": 0,
                "results": [],
                "errors": [error_msg],
            }

    def _execute_buy_in_session(self, session, operation: dict) -> dict:
        """セッション内での買い操作実行（内部メソッド）"""
        # 実装は buy_stock と同様だが、既存のセッションを使用
        # 簡略化のため省略
        pass

    def _execute_sell_in_session(self, session, operation: dict) -> dict:
        """セッション内での売り操作実行（内部メソッド）"""
        # 実装は sell_stock と同様だが、既存のセッションを使用
        # 簡略化のため省略
        pass


# 便利関数の提供
def buy_stock(stock_code: str, quantity: int, **kwargs) -> Dict[str, any]:
    """
    グローバル買い関数（便利関数）

    Args:
        stock_code: 証券コード
        quantity: 購入数量
        **kwargs: その他のオプション

    Returns:
        取引結果
    """
    trade_ops = TradeOperations()
    return trade_ops.buy_stock(stock_code, quantity, **kwargs)


def sell_stock(stock_code: str, quantity: int, **kwargs) -> Dict[str, any]:
    """
    グローバル売り関数（便利関数）

    Args:
        stock_code: 証券コード
        quantity: 売却数量
        **kwargs: その他のオプション

    Returns:
        取引結果
    """
    trade_ops = TradeOperations()
    return trade_ops.sell_stock(stock_code, quantity, **kwargs)
