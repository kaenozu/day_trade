"""
ウォッチリスト管理 - レガシーアラート機能
後方互換性のためのアラート機能（非推奨 - AlertManagerを使用することを推奨）
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_

from ...data.stock_fetcher import StockFetcher
from ...models import Alert, Stock, db_manager
from ...models.enums import AlertType
from ...utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)

logger = get_context_logger(__name__)


@dataclass
class AlertNotification:
    """アラート通知"""

    stock_code: str
    stock_name: str
    alert_type: AlertType
    threshold: float
    current_value: float
    triggered_at: datetime
    memo: str = ""


class WatchlistLegacyAlerts:
    """レガシーアラート機能を提供するクラス（非推奨）"""

    def __init__(self, stock_fetcher: Optional[StockFetcher] = None):
        self.fetcher = stock_fetcher or StockFetcher()

    def add_alert(
        self, stock_code: str, alert_type: AlertType, threshold: float, memo: str = ""
    ) -> bool:
        """
        アラート条件を追加（レガシーメソッド）

        Note:
            このメソッドは下位互換性のために残されています。
            新しいコードではAlertManagerのadd_alert()を使用してください。

        Args:
            stock_code: 証券コード
            alert_type: アラートタイプ
            threshold: 閾値
            memo: メモ

        Returns:
            追加に成功した場合True
        """
        logger.warning(
            "WatchlistLegacyAlerts.add_alert() is deprecated. "
            "Use AlertManager.add_alert() instead.",
            operation="deprecated_method_usage",
        )
        
        try:
            with db_manager.session_scope() as session:
                # 既存のアラートをチェック
                existing = (
                    session.query(Alert)
                    .filter(
                        and_(
                            Alert.stock_code == stock_code,
                            Alert.alert_type == alert_type,
                            Alert.threshold == threshold,
                        )
                    )
                    .first()
                )

                if existing:
                    logger.warning(
                        "Alert condition already exists",
                        stock_code=stock_code,
                        alert_type=alert_type.value,
                        threshold=threshold,
                    )
                    return False

                # 新しいアラートを追加
                alert = Alert(
                    stock_code=stock_code,
                    alert_type=alert_type,
                    threshold=threshold,
                    memo=memo,
                    is_active=True,
                )
                session.add(alert)

                log_business_event(
                    "alert_added",
                    stock_code=stock_code,
                    alert_type=alert_type.value,
                    threshold=threshold,
                )
                return True

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "add_alert",
                    "stock_code": stock_code,
                    "alert_type": alert_type.value,
                },
            )
            return False

    def remove_alert(
        self, stock_code: str, alert_type: AlertType, threshold: float
    ) -> bool:
        """
        アラート条件を削除

        Args:
            stock_code: 証券コード
            alert_type: アラートタイプ
            threshold: 閾値

        Returns:
            削除に成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                alert = (
                    session.query(Alert)
                    .filter(
                        and_(
                            Alert.stock_code == stock_code,
                            Alert.alert_type == alert_type,
                            Alert.threshold == threshold,
                        )
                    )
                    .first()
                )

                if alert:
                    session.delete(alert)
                    log_business_event(
                        "alert_removed",
                        stock_code=stock_code,
                        alert_type=alert_type.value,
                        threshold=threshold,
                    )
                    return True
                else:
                    logger.warning(
                        "Alert not found for removal",
                        stock_code=stock_code,
                        alert_type=alert_type.value,
                        threshold=threshold,
                    )
                    return False

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "remove_alert",
                    "stock_code": stock_code,
                    "alert_type": alert_type.value,
                    "threshold": threshold,
                },
            )
            return False

    def get_alerts(
        self, stock_code: Optional[str] = None, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        アラート一覧を取得

        Args:
            stock_code: 証券コード（指定しない場合は全て）
            active_only: アクティブなアラートのみ取得

        Returns:
            アラート一覧
        """
        try:
            with db_manager.session_scope() as session:
                query = session.query(Alert)

                if stock_code:
                    query = query.filter(Alert.stock_code == stock_code)

                if active_only:
                    query = query.filter(Alert.is_active)

                alerts = query.all()

                result = []
                for alert in alerts:
                    # 銘柄名を別途取得
                    try:
                        stock = (
                            session.query(Stock)
                            .filter(Stock.code == alert.stock_code)
                            .first()
                        )
                        stock_name = stock.name if stock else alert.stock_code
                    except Exception:
                        stock_name = alert.stock_code

                    result.append(
                        {
                            "id": alert.id,
                            "stock_code": alert.stock_code,
                            "stock_name": stock_name,
                            "alert_type": alert.alert_type,
                            "threshold": alert.threshold,
                            "is_active": alert.is_active,
                            "last_triggered": alert.last_triggered,
                            "memo": alert.memo,
                            "created_at": alert.created_at,
                        }
                    )

                return result

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "get_alerts",
                    "stock_code": stock_code,
                    "active_only": active_only,
                },
            )
            return []

    def toggle_alert(self, alert_id: int) -> bool:
        """
        アラートのアクティブ状態を切り替え

        Args:
            alert_id: アラートID

        Returns:
            切り替えに成功した場合True
        """
        try:
            with db_manager.session_scope() as session:
                alert = session.query(Alert).filter(Alert.id == alert_id).first()

                if alert:
                    alert.is_active = not alert.is_active
                    log_business_event(
                        "alert_toggled",
                        alert_id=alert_id,
                        stock_code=alert.stock_code,
                        new_status=alert.is_active,
                    )
                    return True
                else:
                    logger.warning(
                        "Alert not found for toggle", extra={"alert_id": alert_id}
                    )
                    return False

        except Exception as e:
            log_error_with_context(
                e, {"operation": "toggle_alert", "alert_id": alert_id}
            )
            return False

    def check_alerts(self) -> List[AlertNotification]:
        """
        アラート条件をチェックして通知リストを生成（非推奨）

        Note:
            このメソッドは下位互換性のために残されています。
            新しいコードではAlertManager.check_all_alerts()を使用してください。
            AlertManagerはより高度な機能（バルク処理、クールダウン、カスタム条件等）を提供します。

        Returns:
            トリガーされたアラートの通知リスト
        """
        logger.warning(
            "WatchlistLegacyAlerts.check_alerts() is deprecated. "
            "Use AlertManager.check_all_alerts() instead.",
            operation="deprecated_method_usage",
        )
        
        notifications = []

        try:
            # アクティブなアラートを取得
            alerts = self.get_alerts(active_only=True)
            if not alerts:
                return notifications

            # 銘柄コードを抽出して価格データを取得
            stock_codes = list(set(alert["stock_code"] for alert in alerts))
            price_data = self.fetcher.get_realtime_data(stock_codes)

            with db_manager.session_scope() as session:
                for alert in alerts:
                    try:
                        code = alert["stock_code"]
                        current_data = price_data.get(code)

                        if not current_data:
                            logger.warning(
                                "Price data not available for alert check",
                                stock_code=code,
                            )
                            continue

                        # アラート条件をチェック
                        is_triggered = self._check_alert_condition(alert, current_data)

                        if is_triggered:
                            # 通知を作成
                            notification = self._create_notification(
                                alert, current_data
                            )
                            notifications.append(notification)

                            # データベースのトリガー日時を更新
                            db_alert = (
                                session.query(Alert)
                                .filter(Alert.id == alert["id"])
                                .first()
                            )
                            if db_alert:
                                db_alert.last_triggered = datetime.now()

                    except Exception as e:
                        log_error_with_context(
                            e,
                            {
                                "operation": "individual_alert_check",
                                "stock_code": alert["stock_code"],
                                "alert_id": alert.get("id"),
                            },
                        )
                        continue

        except Exception as e:
            log_error_with_context(e, {"operation": "bulk_alert_check"})

        return notifications

    def _check_alert_condition(
        self, alert: Dict[str, Any], price_data: Dict[str, Any]
    ) -> bool:
        """
        個別のアラート条件をチェック

        Args:
            alert: アラート情報
            price_data: 価格データ

        Returns:
            条件に合致した場合True
        """
        alert_type = alert["alert_type"]
        threshold = alert["threshold"]

        try:
            if alert_type == AlertType.PRICE_ABOVE:
                return price_data["current_price"] > threshold

            elif alert_type == AlertType.PRICE_BELOW:
                return price_data["current_price"] < threshold

            elif alert_type == AlertType.CHANGE_PERCENT_UP:
                return price_data.get("change_percent", 0) > threshold

            elif alert_type == AlertType.CHANGE_PERCENT_DOWN:
                return price_data.get("change_percent", 0) < threshold

            elif alert_type == AlertType.VOLUME_SPIKE:
                # 出来高が平均の何倍かをチェック（簡易版）
                volume = price_data.get("volume", 0)
                # 実際の実装では過去平均出来高と比較すべき
                return volume > threshold

            else:
                logger.warning(
                    "Unknown alert type",
                    alert_type=alert_type,
                    stock_code=alert.get("stock_code", "unknown"),
                )
                return False

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "check_alert_condition",
                    "alert_type": alert.get("alert_type"),
                    "stock_code": alert.get("stock_code"),
                },
            )
            return False

    def _create_notification(
        self, alert: Dict[str, Any], price_data: Dict[str, Any]
    ) -> AlertNotification:
        """
        アラート通知を作成

        Args:
            alert: アラート情報
            price_data: 価格データ

        Returns:
            アラート通知
        """
        alert_type = AlertType(alert["alert_type"])

        # 現在値を取得
        if alert_type in [AlertType.PRICE_ABOVE, AlertType.PRICE_BELOW]:
            current_value = price_data["current_price"]
        elif alert_type in [AlertType.CHANGE_PERCENT_UP, AlertType.CHANGE_PERCENT_DOWN]:
            current_value = price_data.get("change_percent", 0)
        elif alert_type == AlertType.VOLUME_SPIKE:
            current_value = price_data.get("volume", 0)
        else:
            current_value = 0

        return AlertNotification(
            stock_code=alert["stock_code"],
            stock_name=alert["stock_name"],
            alert_type=alert_type,
            threshold=alert["threshold"],
            current_value=current_value,
            triggered_at=datetime.now(),
            memo=alert["memo"],
        )