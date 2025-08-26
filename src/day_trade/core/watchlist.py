"""
ウォッチリスト管理モジュール - 統合インターフェース
お気に入り銘柄の管理とアラート機能を提供

このファイルは後方互換性のため、分離されたwatchlistパッケージから
主要クラスをインポートして提供します。

新しいコードでは、個別モジュールを直接使用することを推奨します：
- from day_trade.core.watchlist import WatchlistCore
- from day_trade.core.watchlist import WatchlistGroups
- from day_trade.core.watchlist import WatchlistOptimized
等
"""

# 後方互換性のため、分離されたモジュールから統合クラスをインポート
from .watchlist import (
    AlertNotification,
    WatchlistBulkOperations,
    WatchlistCore,
    WatchlistDataExport,
    WatchlistGroups,
    WatchlistIntegration,
    WatchlistLegacyAlerts,
    WatchlistManager,
    WatchlistOptimized,
)

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