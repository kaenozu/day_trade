"""
ウォッチリスト管理モジュール
お気に入り銘柄の管理とアラート機能を提供
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_

from ..data.stock_fetcher import StockFetcher
from ..models import Alert, PriceData, Stock, WatchlistItem, db_manager
from ..models.enums import AlertType
from ..utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)

# AlertManager関連のインポートを遅延読み込みで避けて循環参照を防ぐ
# from .alerts import AlertCondition

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


class WatchlistManager:
    """ウォッチリスト管理クラス"""

    def __init__(self):
        self.fetcher = StockFetcher()

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

    # アラート機能（非推奨 - AlertManagerを使用することを推奨）
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
            "WatchlistManager.check_alerts() is deprecated. Use AlertManager.check_all_alerts() instead.",
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

    def get_watchlist_summary(self) -> Dict[str, Any]:
        """
        ウォッチリストのサマリー情報を取得

        Returns:
            サマリー情報
        """
        try:
            groups = self.get_groups()
            total_stocks = 0
            total_alerts = 0

            for group in groups:
                watchlist = self.get_watchlist(group)
                total_stocks += len(watchlist)

            alerts = self.get_alerts(active_only=True)
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

    def export_watchlist_to_csv(
        self, filename: str, group_name: Optional[str] = None
    ) -> bool:
        """
        ウォッチリストをCSVファイルにエクスポート

        Args:
            filename: 出力ファイル名
            group_name: グループ名（指定しない場合は全て）

        Returns:
            エクスポートに成功した場合True
        """
        try:
            import pandas as pd

            # 価格情報付きウォッチリストを取得
            watchlist_data = self.get_watchlist_with_prices(group_name)

            if not watchlist_data:
                logger.warning(
                    "No data available for export", extra={"group_name": group_name}
                )
                return False

            # DataFrameに変換
            df_data = []
            for code, data in watchlist_data.items():
                df_data.append(
                    {
                        "証券コード": code,
                        "銘柄名": data.get("stock_name", ""),
                        "グループ": data.get("group_name", ""),
                        "現在価格": data.get("current_price", 0),
                        "前日比": data.get("change", 0),
                        "変化率(%)": data.get("change_percent", 0),
                        "出来高": data.get("volume", 0),
                        "メモ": data.get("memo", ""),
                        "追加日": data.get("added_date", ""),
                    }
                )

            df = pd.DataFrame(df_data)
            df.to_csv(filename, index=False, encoding="utf-8-sig")

            logger.info(
                "Watchlist exported to CSV",
                filename=filename,
                group_name=group_name,
                item_count=len(watchlist_data),
            )
            return True

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "export_watchlist_to_csv",
                    "filename": filename,
                    "group_name": group_name,
                },
            )
            return False

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

    def get_watchlist_optimized(
        self, group_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        最適化されたウォッチリスト取得（一括JOIN）

        Args:
            group_name: グループ名（指定しない場合は全て）

        Returns:
            価格情報付きウォッチリストアイテムのリスト
        """
        try:
            with db_manager.session_scope() as session:
                # WatchlistItemとStockを一括JOINで取得
                query = session.query(WatchlistItem, Stock).join(
                    Stock, WatchlistItem.stock_code == Stock.code
                )

                if group_name:
                    query = query.filter(WatchlistItem.group_name == group_name)

                items = query.all()

                # 銘柄コードを抽出して一括で価格データを取得
                stock_codes = [item.WatchlistItem.stock_code for item in items]

                if stock_codes:
                    # 最適化されたメソッドで一括取得
                    latest_prices = PriceData.get_latest_prices(session, stock_codes)
                else:
                    latest_prices = {}

                # 結果を構築
                result = []
                for item in items:
                    watchlist_item = item.WatchlistItem
                    stock = item.Stock
                    code = watchlist_item.stock_code
                    price_data = latest_prices.get(code)

                    result.append(
                        {
                            "stock_code": code,
                            "stock_name": stock.name,
                            "group_name": watchlist_item.group_name,
                            "memo": watchlist_item.memo,
                            "added_date": watchlist_item.created_at,
                            "sector": stock.sector,
                            "industry": stock.industry,
                            "current_price": price_data.close if price_data else None,
                            "volume": price_data.volume if price_data else None,
                            "last_updated": price_data.datetime if price_data else None,
                        }
                    )

                return result

        except Exception as e:
            log_error_with_context(
                e, {"operation": "get_watchlist_optimized", "group_name": group_name}
            )
            return []

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

    def get_alert_manager(self):
        """
        AlertManagerインスタンスを取得（統合ヘルパー、DI対応）

        Note:
            StockFetcherインスタンスを共有し、循環参照を防ぐため
            WatchlistManagerを参照させないようにしています。

        Returns:
            AlertManagerインスタンス
        """
        from .alerts import AlertManager

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

            from .alerts import AlertCondition, AlertPriority

            alert_manager = self.get_alert_manager()
            legacy_alerts = self.get_alerts(active_only=False)

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
