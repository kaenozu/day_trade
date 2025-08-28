"""
ウォッチリスト管理 - グループ機能
ウォッチリストのグループ管理（グループ一覧、グループ移動、サマリー）を提供
"""

from typing import Any, Dict, List

from sqlalchemy import and_

from ...models import WatchlistItem, db_manager
from ...utils.logging_config import get_context_logger, log_error_with_context

logger = get_context_logger(__name__)


class WatchlistGroups:
    """ウォッチリストのグループ管理機能を提供するクラス"""

    def get_groups(self) -> List[str]:
        """
        ウォッチリストのグループ一覧を取得

        Returns:
            グループ名のリスト
        """
        try:
            with db_manager.session_scope() as session:
                groups = session.query(WatchlistItem.group_name).distinct().all()
                return [group[0] for group in groups]

        except Exception as e:
            log_error_with_context(e, {"operation": "get_groups"})
            return []

    def move_to_group(self, stock_code: str, from_group: str, to_group: str) -> bool:
        """
        銘柄を別のグループに移動

        Args:
            stock_code: 証券コード
            from_group: 移動元グループ
            to_group: 移動先グループ

        Returns:
            移動に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                item = (
                    session.query(WatchlistItem)
                    .filter(
                        and_(
                            WatchlistItem.stock_code == stock_code,
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
                                WatchlistItem.stock_code == stock_code,
                                WatchlistItem.group_name == to_group,
                            )
                        )
                        .first()
                    )

                    if existing:
                        return False  # 移動先に既に存在

                    item.group_name = to_group
                    return True
                else:
                    return False

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "move_to_group",
                    "stock_code": stock_code,
                    "from_group": from_group,
                    "to_group": to_group,
                },
            )
            return False

    def get_watchlist_summary(self) -> Dict[str, Any]:
        """
        ウォッチリストのサマリー情報を取得

        Returns:
            サマリー情報
        """
        try:
            from ..alerts import AlertManager
            from .core import WatchlistCore

            # 依存関係を注入して使用
            core = WatchlistCore()
            groups = self.get_groups()
            total_stocks = 0

            for group in groups:
                watchlist = core.get_watchlist(group)
                total_stocks += len(watchlist)

            # アラート数を取得（AlertManagerから）
            alert_manager = AlertManager()
            alerts = alert_manager.get_alerts(active_only=True)
            total_alerts = len(alerts)

            return {
                "total_groups": len(groups),
                "total_stocks": total_stocks,
                "total_alerts": total_alerts,
                "groups": groups,
            }

        except Exception as e:
            log_error_with_context(e, {"operation": "get_watchlist_summary"})
            return {
                "total_groups": 0,
                "total_stocks": 0,
                "total_alerts": 0,
                "groups": [],
            }

    def get_group_stock_count(self, group_name: str) -> int:
        """
        指定されたグループの銘柄数を取得

        Args:
            group_name: グループ名

        Returns:
            銘柄数
        """
        try:
            with db_manager.session_scope() as session:
                count = (
                    session.query(WatchlistItem)
                    .filter(WatchlistItem.group_name == group_name)
                    .count()
                )
                return count

        except Exception as e:
            log_error_with_context(
                e, {"operation": "get_group_stock_count", "group_name": group_name}
            )
            return 0

    def get_group_summary(self) -> Dict[str, int]:
        """
        全グループの銘柄数サマリーを取得

        Returns:
            グループ名をキーとした銘柄数の辞書
        """
        try:
            groups = self.get_groups()
            summary = {}
            
            for group in groups:
                summary[group] = self.get_group_stock_count(group)
            
            return summary

        except Exception as e:
            log_error_with_context(e, {"operation": "get_group_summary"})
            return {}

    def rename_group(self, old_name: str, new_name: str) -> bool:
        """
        グループ名を変更

        Args:
            old_name: 現在のグループ名
            new_name: 新しいグループ名

        Returns:
            変更に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                # 新しい名前のグループが既に存在するかチェック
                existing = (
                    session.query(WatchlistItem)
                    .filter(WatchlistItem.group_name == new_name)
                    .first()
                )
                
                if existing:
                    return False  # 新しい名前のグループが既に存在
                
                # グループ名を一括更新
                updated_count = (
                    session.query(WatchlistItem)
                    .filter(WatchlistItem.group_name == old_name)
                    .update({WatchlistItem.group_name: new_name})
                )
                
                return updated_count > 0

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "rename_group",
                    "old_name": old_name,
                    "new_name": new_name,
                },
            )
            return False