"""
ウォッチリスト管理 - 一括処理機能
複数銘柄の一括追加、一括削除、ウォッチリストクリア等の機能を提供
"""

from typing import Dict, List, Optional

from sqlalchemy import and_

from ...data.stock_fetcher import StockFetcher
from ...models import Stock, WatchlistItem, db_manager
from ...utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)

logger = get_context_logger(__name__)


class WatchlistBulkOperations:
    """ウォッチリストの一括処理機能を提供するクラス"""

    def __init__(self, stock_fetcher: Optional[StockFetcher] = None):
        self.fetcher = stock_fetcher or StockFetcher()

    def bulk_add_stocks(self, stock_data: List[Dict[str, str]]) -> Dict[str, bool]:
        """
        複数銘柄を一括でウォッチリストに追加（最適化版）

        Args:
            stock_data: [{"code": "7203", "group": "default", "memo": ""}...]

        Returns:
            銘柄コードをキーとした成功/失敗の辞書
        """
        results = {}

        try:
            with db_manager.session_scope() as session:
                for data in stock_data:
                    code = data.get("code", "")
                    group = data.get("group", "default")
                    memo = data.get("memo", "")

                    try:
                        # 重複チェック
                        existing = (
                            session.query(WatchlistItem)
                            .filter(
                                and_(
                                    WatchlistItem.stock_code == code,
                                    WatchlistItem.group_name == group,
                                )
                            )
                            .first()
                        )

                        if existing:
                            results[code] = False  # 既に存在
                            continue

                        # 銘柄マスタにない場合は作成
                        stock = session.query(Stock).filter(Stock.code == code).first()
                        if not stock:
                            # 企業情報を取得
                            company_info = self.fetcher.get_company_info(code)
                            if company_info:
                                stock = Stock(
                                    code=code,
                                    name=company_info.get("name", code),
                                    sector=company_info.get("sector"),
                                    industry=company_info.get("industry"),
                                )
                                session.add(stock)

                        # ウォッチリストに追加
                        watchlist_item = WatchlistItem(
                            stock_code=code, group_name=group, memo=memo
                        )
                        session.add(watchlist_item)
                        results[code] = True

                    except Exception as e:
                        log_error_with_context(
                            e,
                            {
                                "operation": "bulk_add_individual_stock",
                                "stock_code": code,
                                "group_name": group,
                            },
                        )
                        results[code] = False

                # 成功した銘柄数をログ出力
                success_count = sum(1 for success in results.values() if success)
                log_business_event(
                    "bulk_add_completed",
                    total_stocks=len(stock_data),
                    success_count=success_count,
                )

        except Exception as e:
            log_error_with_context(
                e, {"operation": "bulk_add_stocks", "total_stocks": len(stock_data)}
            )
            # 失敗した銘柄を記録
            for data in stock_data:
                code = data.get("code", "")
                if code not in results:
                    results[code] = False

        return results

    def bulk_remove_stocks(
        self, stock_codes: List[str], group_name: str = "default"
    ) -> Dict[str, bool]:
        """
        複数銘柄を一括でウォッチリストから削除

        Args:
            stock_codes: 証券コードのリスト
            group_name: グループ名

        Returns:
            銘柄コードをキーとした成功/失敗の辞書
        """
        results = {}

        try:
            with db_manager.session_scope() as session:
                for code in stock_codes:
                    try:
                        item = (
                            session.query(WatchlistItem)
                            .filter(
                                and_(
                                    WatchlistItem.stock_code == code,
                                    WatchlistItem.group_name == group_name,
                                )
                            )
                            .first()
                        )

                        if item:
                            session.delete(item)
                            results[code] = True
                        else:
                            results[code] = False

                    except Exception as e:
                        log_error_with_context(
                            e,
                            {
                                "operation": "bulk_remove_individual_stock",
                                "stock_code": code,
                                "group_name": group_name,
                            },
                        )
                        results[code] = False

                # 成功した銘柄数をログ出力
                success_count = sum(1 for success in results.values() if success)
                log_business_event(
                    "bulk_remove_completed",
                    total_stocks=len(stock_codes),
                    success_count=success_count,
                    group_name=group_name,
                )

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "bulk_remove_stocks",
                    "total_stocks": len(stock_codes),
                    "group_name": group_name,
                },
            )
            # 失敗した銘柄を記録
            for code in stock_codes:
                if code not in results:
                    results[code] = False

        return results

    def clear_watchlist(self, group_name: Optional[str] = None) -> bool:
        """
        ウォッチリストをクリア（最適化版）

        Args:
            group_name: グループ名（指定しない場合は全て）

        Returns:
            クリアに成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                query = session.query(WatchlistItem)

                if group_name:
                    query = query.filter(WatchlistItem.group_name == group_name)

                # 一括削除
                deleted_count = query.delete()

                log_business_event(
                    "watchlist_cleared",
                    group_name=group_name,
                    deleted_count=deleted_count,
                )
                return True

        except Exception as e:
            log_error_with_context(
                e, {"operation": "clear_watchlist", "group_name": group_name}
            )
            return False

    def bulk_move_to_group(
        self, stock_codes: List[str], from_group: str, to_group: str
    ) -> Dict[str, bool]:
        """
        複数銘柄を別のグループに一括移動

        Args:
            stock_codes: 証券コードのリスト
            from_group: 移動元グループ
            to_group: 移動先グループ

        Returns:
            銘柄コードをキーとした成功/失敗の辞書
        """
        results = {}

        try:
            with db_manager.session_scope() as session:
                for code in stock_codes:
                    try:
                        item = (
                            session.query(WatchlistItem)
                            .filter(
                                and_(
                                    WatchlistItem.stock_code == code,
                                    WatchlistItem.group_name == from_group,
                                )
                            )
                            .first()
                        )

                        if item:
                            # 移動先に同じ銘柄が既に存在するかチェック
                            existing = (
                                session.query(WatchlistItem)
                                .filter(
                                    and_(
                                        WatchlistItem.stock_code == code,
                                        WatchlistItem.group_name == to_group,
                                    )
                                )
                                .first()
                            )

                            if existing:
                                results[code] = False  # 移動先に既に存在
                            else:
                                item.group_name = to_group
                                results[code] = True
                        else:
                            results[code] = False

                    except Exception as e:
                        log_error_with_context(
                            e,
                            {
                                "operation": "bulk_move_individual_stock",
                                "stock_code": code,
                                "from_group": from_group,
                                "to_group": to_group,
                            },
                        )
                        results[code] = False

                # 成功した銘柄数をログ出力
                success_count = sum(1 for success in results.values() if success)
                log_business_event(
                    "bulk_move_completed",
                    total_stocks=len(stock_codes),
                    success_count=success_count,
                    from_group=from_group,
                    to_group=to_group,
                )

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "bulk_move_to_group",
                    "total_stocks": len(stock_codes),
                    "from_group": from_group,
                    "to_group": to_group,
                },
            )
            # 失敗した銘柄を記録
            for code in stock_codes:
                if code not in results:
                    results[code] = False

        return results

    def bulk_update_memo(
        self, updates: List[Dict[str, str]], group_name: str = "default"
    ) -> Dict[str, bool]:
        """
        複数銘柄のメモを一括更新

        Args:
            updates: [{"code": "7203", "memo": "新しいメモ"}...]
            group_name: グループ名

        Returns:
            銘柄コードをキーとした成功/失敗の辞書
        """
        results = {}

        try:
            with db_manager.session_scope() as session:
                for update in updates:
                    code = update.get("code", "")
                    memo = update.get("memo", "")

                    try:
                        item = (
                            session.query(WatchlistItem)
                            .filter(
                                and_(
                                    WatchlistItem.stock_code == code,
                                    WatchlistItem.group_name == group_name,
                                )
                            )
                            .first()
                        )

                        if item:
                            item.memo = memo
                            results[code] = True
                        else:
                            results[code] = False

                    except Exception as e:
                        log_error_with_context(
                            e,
                            {
                                "operation": "bulk_update_memo_individual",
                                "stock_code": code,
                                "group_name": group_name,
                            },
                        )
                        results[code] = False

                # 成功した銘柄数をログ出力
                success_count = sum(1 for success in results.values() if success)
                log_business_event(
                    "bulk_memo_update_completed",
                    total_stocks=len(updates),
                    success_count=success_count,
                    group_name=group_name,
                )

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "bulk_update_memo",
                    "total_stocks": len(updates),
                    "group_name": group_name,
                },
            )
            # 失敗した銘柄を記録
            for update in updates:
                code = update.get("code", "")
                if code not in results:
                    results[code] = False

        return results

    def get_bulk_operation_summary(self) -> Dict[str, int]:
        """
        一括処理の実行可能な操作のサマリーを取得

        Returns:
            操作可能な銘柄数等の情報
        """
        try:
            with db_manager.session_scope() as session:
                total_items = session.query(WatchlistItem).count()
                
                # グループ別の銘柄数
                from .groups import WatchlistGroups
                groups_manager = WatchlistGroups()
                group_summary = groups_manager.get_group_summary()

                return {
                    "total_watchlist_items": total_items,
                    "total_groups": len(group_summary),
                    "group_breakdown": group_summary,
                    "available_operations": [
                        "bulk_add_stocks",
                        "bulk_remove_stocks", 
                        "bulk_move_to_group",
                        "bulk_update_memo",
                        "clear_watchlist"
                    ]
                }

        except Exception as e:
            log_error_with_context(e, {"operation": "get_bulk_operation_summary"})
            return {
                "total_watchlist_items": 0,
                "total_groups": 0,
                "group_breakdown": {},
                "available_operations": []
            }