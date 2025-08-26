"""
ウォッチリスト管理 - 統合機能
AlertManagerとの統合、データ移行、推奨使用方法の提供
"""

from typing import Any, Dict, Optional

from ...data.stock_fetcher import StockFetcher
from ...utils.logging_config import get_context_logger
from .legacy_alerts import WatchlistLegacyAlerts

logger = get_context_logger(__name__)


class WatchlistIntegration:
    """ウォッチリストの統合機能を提供するクラス"""

    def __init__(self, stock_fetcher: Optional[StockFetcher] = None):
        self.fetcher = stock_fetcher or StockFetcher()
        self.legacy_alerts = WatchlistLegacyAlerts(stock_fetcher)

    def get_alert_manager(self):
        """
        AlertManagerインスタンスを取得（統合ヘルパー、DI対応）

        Note:
            StockFetcherインスタンスを共有し、循環参照を防ぐため
            WatchlistManagerを参照させないようにしています。

        Returns:
            AlertManagerインスタンス
        """
        from ..alerts import AlertManager

        # StockFetcherインスタンスを共有し、watchlist_managerはNoneにして循環参照を避ける
        return AlertManager(
            stock_fetcher=self.fetcher,
            watchlist_manager=None,  # 循環参照を避ける
        )

    def migrate_alerts_to_alert_manager(self) -> bool:
        """
        既存のアラートをAlertManager形式に移行（ユーティリティ）

        Returns:
            移行成功フラグ
        """
        try:
            from decimal import Decimal

            from ..alerts import AlertCondition, AlertPriority

            alert_manager = self.get_alert_manager()
            legacy_alerts = self.legacy_alerts.get_alerts(active_only=False)

            migrated_count = 0
            for alert in legacy_alerts:
                try:
                    # レガシーアラートをAlertManager形式に変換
                    alert_condition = AlertCondition(
                        alert_id=f"migrated_{alert['id']}",
                        symbol=alert["stock_code"],
                        alert_type=alert["alert_type"],
                        condition_value=Decimal(str(alert["threshold"])),
                        comparison_operator=(
                            ">"
                            if "_up" in alert["alert_type"].value
                            or "above" in alert["alert_type"].value
                            else "<"
                        ),
                        is_active=alert["is_active"],
                        priority=AlertPriority.MEDIUM,
                        description=f"Migrated from WatchlistManager: {alert['memo']}",
                    )

                    if alert_manager.add_alert(alert_condition):
                        migrated_count += 1

                except Exception as e:
                    logger.error(f"アラート移行エラー (ID: {alert.get('id')}): {e}")

            logger.info(f"アラート移行完了: {migrated_count}/{len(legacy_alerts)}件")
            return migrated_count > 0

        except Exception as e:
            logger.error(f"アラート移行エラー: {e}")
            return False

    def get_recommended_alert_manager_usage(self) -> Dict[str, str]:
        """
        AlertManagerの推奨使用方法を返す（ドキュメントヘルパー）

        Returns:
            推奨使用方法の説明
        """
        return {
            "deprecated_methods": {
                "add_alert()": "Use AlertManager.add_alert() with AlertCondition",
                "check_alerts()": "Use AlertManager.check_all_alerts() or start_monitoring()",
                "remove_alert()": "Use AlertManager.remove_alert()",
                "toggle_alert()": "Use AlertManager's alert management",
            },
            "migration_helper": "Use migrate_alerts_to_alert_manager() to migrate existing alerts",
            "recommended_pattern": """
# Recommended usage:
from day_trade.core.alerts import AlertManager, AlertCondition, AlertPriority
from day_trade.models.enums import AlertType
from decimal import Decimal

# Create AlertManager
alert_manager = watchlist_manager.get_alert_manager()

# Add advanced alert
alert_condition = AlertCondition(
    alert_id="unique_alert_id",
    symbol="7203",
    alert_type=AlertType.PRICE_ABOVE,
    condition_value=Decimal("3000"),
    priority=AlertPriority.HIGH,
    cooldown_minutes=30
)
alert_manager.add_alert(alert_condition)

# Start monitoring
alert_manager.start_monitoring(interval_seconds=60)
            """,
        }

    def get_integration_status(self) -> Dict[str, Any]:
        """
        統合機能の状態を取得

        Returns:
            統合状態の情報
        """
        try:
            # レガシーアラートの数
            legacy_alerts_count = len(self.legacy_alerts.get_alerts(active_only=False))
            
            # AlertManagerの利用可能性チェック
            alert_manager_available = True
            try:
                alert_manager = self.get_alert_manager()
                modern_alerts_count = len(alert_manager.get_alerts(active_only=False))
            except Exception:
                alert_manager_available = False
                modern_alerts_count = 0

            return {
                "alert_manager_available": alert_manager_available,
                "legacy_alerts_count": legacy_alerts_count,
                "modern_alerts_count": modern_alerts_count,
                "migration_recommended": legacy_alerts_count > 0 and alert_manager_available,
                "integration_features": [
                    "alert_manager_integration",
                    "legacy_alert_migration",
                    "cross_system_compatibility"
                ]
            }

        except Exception as e:
            logger.error(f"統合状態取得エラー: {e}")
            return {
                "alert_manager_available": False,
                "legacy_alerts_count": 0,
                "modern_alerts_count": 0,
                "migration_recommended": False,
                "integration_features": []
            }