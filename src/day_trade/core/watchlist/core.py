"""
ウォッチリスト管理 - コアCRUD機能
基本的なウォッチリスト操作（追加、削除、取得、更新）を提供
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import and_

from ...data.stock_fetcher import StockFetcher
from ...models import Stock, WatchlistItem, db_manager
from ...utils.logging_config import get_context_logger, log_error_with_context

logger = get_context_logger(__name__)


class WatchlistCore:
    """ウォッチリストの基本CRUD操作を提供するクラス"""

    def __init__(self, stock_fetcher: Optional[StockFetcher] = None):
        self.fetcher = stock_fetcher or StockFetcher()

    def add_stock(
        self, stock_code: str, group_name: str = "default", memo: str = ""
    ) -> bool:
        """
        銘柄をウォッチリストに追加

        Args:
            stock_code: 証券コード
            group_name: グループ名
            memo: メモ

        Returns:
            追加に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                # 重複チェック
                existing = (
                    session.query(WatchlistItem)
                    .filter(
                        and_(
                            WatchlistItem.stock_code == stock_code,
                            WatchlistItem.group_name == group_name,
                        )
                    )
                    .first()
                )

                if existing:
                    return False  # 既に存在

                # 銘柄マスタにない場合は作成
                stock = session.query(Stock).filter(Stock.code == stock_code).first()
                if not stock:
                    # 企業情報を取得して銘柄マスタに追加
                    company_info = self.fetcher.get_company_info(stock_code)
                    if company_info:
                        stock = Stock(
                            code=stock_code,
                            name=company_info.get("name", stock_code),
                            sector=company_info.get("sector"),
                            industry=company_info.get("industry"),
                        )
                        session.add(stock)
                        session.flush()  # IDを取得するため

                # ウォッチリストに追加
                watchlist_item = WatchlistItem(
                    stock_code=stock_code, group_name=group_name, memo=memo
                )
                session.add(watchlist_item)

                return True

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "add_stock_to_watchlist",
                    "stock_code": stock_code,
                    "group_name": group_name,
                },
            )
            return False

    def remove_stock(self, stock_code: str, group_name: str = "default") -> bool:
        """
        銘柄をウォッチリストから削除

        Args:
            stock_code: 証券コード
            group_name: グループ名

        Returns:
            削除に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                item = (
                    session.query(WatchlistItem)
                    .filter(
                        and_(
                            WatchlistItem.stock_code == stock_code,
                            WatchlistItem.group_name == group_name,
                        )
                    )
                    .first()
                )

                if item:
                    session.delete(item)
                    return True
                else:
                    return False

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "remove_stock_from_watchlist",
                    "stock_code": stock_code,
                    "group_name": group_name,
                },
            )
            return False

    def get_watchlist(self, group_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        ウォッチリストを取得

        Args:
            group_name: グループ名（指定しない場合は全て）

        Returns:
            ウォッチリストアイテムのリスト
        """
        try:
            with db_manager.session_scope() as session:
                query = session.query(WatchlistItem).join(Stock)

                if group_name:
                    query = query.filter(WatchlistItem.group_name == group_name)

                items = query.all()

                result = []
                for item in items:
                    result.append(
                        {
                            "stock_code": item.stock_code,
                            "stock_name": (
                                item.stock.name if item.stock else item.stock_code
                            ),
                            "group_name": item.group_name,
                            "memo": item.memo,
                            "added_date": item.created_at,
                        }
                    )

                return result

        except Exception as e:
            log_error_with_context(
                e, {"operation": "get_watchlist", "group_name": group_name}
            )
            return []

    def get_watchlist_with_prices(
        self, group_name: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        価格情報付きのウォッチリストを取得

        Args:
            group_name: グループ名（指定しない場合は全て）

        Returns:
            銘柄コードをキーとした価格情報付きデータ
        """
        watchlist = self.get_watchlist(group_name)
        if not watchlist:
            return {}

        # 銘柄コードを抽出
        stock_codes = [item["stock_code"] for item in watchlist]

        # 価格情報を取得
        price_data = self.fetcher.get_realtime_data(stock_codes)

        # ウォッチリスト情報と価格情報をマージ
        result = {}
        for item in watchlist:
            code = item["stock_code"]
            result[code] = {**item, **price_data.get(code, {})}

        return result

    def update_memo(self, stock_code: str, group_name: str, memo: str) -> bool:
        """
        メモを更新

        Args:
            stock_code: 証券コード
            group_name: グループ名
            memo: 新しいメモ

        Returns:
            更新に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                item = (
                    session.query(WatchlistItem)
                    .filter(
                        and_(
                            WatchlistItem.stock_code == stock_code,
                            WatchlistItem.group_name == group_name,
                        )
                    )
                    .first()
                )

                if item:
                    item.memo = memo
                    return True
                else:
                    return False

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "update_memo",
                    "stock_code": stock_code,
                    "group_name": group_name,
                },
            )
            return False