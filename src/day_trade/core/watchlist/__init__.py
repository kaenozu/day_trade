"""
ウォッチリスト管理モジュール
分離されたウォッチリスト機能をまとめて提供
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...data.stock_fetcher import StockFetcher
from ...models.enums import AlertType
from .bulk_operations import WatchlistBulkOperations
from .core import WatchlistCore
from .data_export import WatchlistDataExport
from .groups import WatchlistGroups
from .integration import WatchlistIntegration
from .legacy_alerts import AlertNotification, WatchlistLegacyAlerts
from .optimized import WatchlistOptimized

# 後方互換性のためのエクスポート
__all__ = [
    "WatchlistManager",
    "AlertNotification",
    "WatchlistCore",
    "WatchlistGroups", 
    "WatchlistLegacyAlerts",
    "WatchlistDataExport",
    "WatchlistBulkOperations",
    "WatchlistOptimized",
    "WatchlistIntegration",
]


class WatchlistManager:
    """
    統合ウォッチリスト管理クラス（後方互換性維持）
    
    分離された機能モジュールを統合して、元のインターフェースを維持
    """

    def __init__(self):
        self.fetcher = StockFetcher()
        
        # 各モジュールのインスタンスを作成
        self._core = WatchlistCore(self.fetcher)
        self._groups = WatchlistGroups()
        self._legacy_alerts = WatchlistLegacyAlerts(self.fetcher)
        self._data_export = WatchlistDataExport(self._core)
        self._bulk_operations = WatchlistBulkOperations(self.fetcher)
        self._optimized = WatchlistOptimized()
        self._integration = WatchlistIntegration(self.fetcher)

    # === Core CRUD機能 ===
    def add_stock(
        self, stock_code: str, group_name: str = "default", memo: str = ""
    ) -> bool:
        """銘柄をウォッチリストに追加"""
        return self._core.add_stock(stock_code, group_name, memo)

    def remove_stock(self, stock_code: str, group_name: str = "default") -> bool:
        """銘柄をウォッチリストから削除"""
        return self._core.remove_stock(stock_code, group_name)

    def get_watchlist(self, group_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """ウォッチリストを取得"""
        return self._core.get_watchlist(group_name)

    def get_watchlist_with_prices(
        self, group_name: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """価格情報付きのウォッチリストを取得"""
        return self._core.get_watchlist_with_prices(group_name)

    def update_memo(self, stock_code: str, group_name: str, memo: str) -> bool:
        """メモを更新"""
        return self._core.update_memo(stock_code, group_name, memo)

    # === グループ管理機能 ===
    def get_groups(self) -> List[str]:
        """ウォッチリストのグループ一覧を取得"""
        return self._groups.get_groups()

    def move_to_group(self, stock_code: str, from_group: str, to_group: str) -> bool:
        """銘柄を別のグループに移動"""
        return self._groups.move_to_group(stock_code, from_group, to_group)

    def get_watchlist_summary(self) -> Dict[str, Any]:
        """ウォッチリストのサマリー情報を取得"""
        return self._groups.get_watchlist_summary()

    # === レガシーアラート機能（非推奨） ===
    def add_alert(
        self, stock_code: str, alert_type: AlertType, threshold: float, memo: str = ""
    ) -> bool:
        """アラート条件を追加（レガシーメソッド）"""
        return self._legacy_alerts.add_alert(stock_code, alert_type, threshold, memo)

    def remove_alert(
        self, stock_code: str, alert_type: AlertType, threshold: float
    ) -> bool:
        """アラート条件を削除"""
        return self._legacy_alerts.remove_alert(stock_code, alert_type, threshold)

    def get_alerts(
        self, stock_code: Optional[str] = None, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """アラート一覧を取得"""
        return self._legacy_alerts.get_alerts(stock_code, active_only)

    def toggle_alert(self, alert_id: int) -> bool:
        """アラートのアクティブ状態を切り替え"""
        return self._legacy_alerts.toggle_alert(alert_id)

    def check_alerts(self) -> List[AlertNotification]:
        """アラート条件をチェックして通知リストを生成（非推奨）"""
        return self._legacy_alerts.check_alerts()

    # === データエクスポート機能 ===
    def export_watchlist_to_csv(
        self, filename: str, group_name: Optional[str] = None
    ) -> bool:
        """ウォッチリストをCSVファイルにエクスポート"""
        return self._data_export.export_watchlist_to_csv(filename, group_name)

    # === 一括処理機能 ===
    def bulk_add_stocks(self, stock_data: List[Dict[str, str]]) -> Dict[str, bool]:
        """複数銘柄を一括でウォッチリストに追加（最適化版）"""
        return self._bulk_operations.bulk_add_stocks(stock_data)

    def clear_watchlist(self, group_name: Optional[str] = None) -> bool:
        """ウォッチリストをクリア（最適化版）"""
        return self._bulk_operations.clear_watchlist(group_name)

    # === 最適化機能 ===
    def get_watchlist_optimized(
        self, group_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """最適化されたウォッチリスト取得（一括JOIN）"""
        return self._optimized.get_watchlist_optimized(group_name)

    # === 統合機能 ===
    def get_alert_manager(self):
        """AlertManagerインスタンスを取得（統合ヘルパー、DI対応）"""
        return self._integration.get_alert_manager()

    def migrate_alerts_to_alert_manager(self) -> bool:
        """既存のアラートをAlertManager形式に移行（ユーティリティ）"""
        return self._integration.migrate_alerts_to_alert_manager()

    def get_recommended_alert_manager_usage(self) -> Dict[str, str]:
        """AlertManagerの推奨使用方法を返す（ドキュメントヘルパー）"""
        return self._integration.get_recommended_alert_manager_usage()

    # === 拡張機能のアクセス ===
    @property
    def core(self) -> WatchlistCore:
        """コア機能へのアクセス"""
        return self._core

    @property
    def groups(self) -> WatchlistGroups:
        """グループ管理機能へのアクセス"""
        return self._groups

    @property
    def legacy_alerts(self) -> WatchlistLegacyAlerts:
        """レガシーアラート機能へのアクセス"""
        return self._legacy_alerts

    @property
    def data_export(self) -> WatchlistDataExport:
        """データエクスポート機能へのアクセス"""
        return self._data_export

    @property
    def bulk_operations(self) -> WatchlistBulkOperations:
        """一括処理機能へのアクセス"""
        return self._bulk_operations

    @property
    def optimized(self) -> WatchlistOptimized:
        """最適化機能へのアクセス"""
        return self._optimized

    @property
    def integration(self) -> WatchlistIntegration:
        """統合機能へのアクセス"""
        return self._integration