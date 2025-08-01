"""
ウォッチリスト管理モジュール
お気に入り銘柄の管理とアラート機能を提供
"""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy import and_
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from ..models import db_manager, WatchlistItem, Stock, Alert
from ..data.stock_fetcher import StockFetcher

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """アラートタイプ"""

    PRICE_ABOVE = "price_above"  # 価格が閾値を上回る
    PRICE_BELOW = "price_below"  # 価格が閾値を下回る
    CHANGE_PERCENT_UP = "change_percent_up"  # 変化率が閾値を上回る
    CHANGE_PERCENT_DOWN = "change_percent_down"  # 変化率が閾値を下回る
    VOLUME_SPIKE = "volume_spike"  # 出来高急増


@dataclass
class AlertCondition:
    """アラート条件"""

    stock_code: str
    alert_type: AlertType
    threshold: float
    memo: str = ""


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
            logger.error(f"ウォッチリスト追加エラー: {e}")
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
            logger.error(f"ウォッチリスト削除エラー: {e}")
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
            logger.error(f"ウォッチリスト取得エラー: {e}")
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
            logger.error(f"グループ取得エラー: {e}")
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
            logger.error(f"メモ更新エラー: {e}")
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
            logger.error(f"グループ移動エラー: {e}")
            return False

    # アラート機能
    def add_alert(self, condition: AlertCondition) -> bool:
        """
        アラート条件を追加

        Args:
            condition: アラート条件

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
                            Alert.stock_code == condition.stock_code,
                            Alert.alert_type == condition.alert_type.value,
                            Alert.threshold == condition.threshold,
                        )
                    )
                    .first()
                )

                if existing:
                    logger.warning(
                        f"同じアラート条件が既に存在します: {condition.stock_code}"
                    )
                    return False

                # 新しいアラートを追加
                alert = Alert(
                    stock_code=condition.stock_code,
                    alert_type=condition.alert_type.value,
                    threshold=condition.threshold,
                    memo=condition.memo,
                    is_active=True,
                )
                session.add(alert)

                logger.info(
                    f"アラートを追加しました: {condition.stock_code} {condition.alert_type.value}"
                )
                return True

        except Exception as e:
            logger.error(f"アラート追加エラー: {e}")
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
                            Alert.alert_type == alert_type.value,
                            Alert.threshold == threshold,
                        )
                    )
                    .first()
                )

                if alert:
                    session.delete(alert)
                    logger.info(
                        f"アラートを削除しました: {stock_code} {alert_type.value}"
                    )
                    return True
                else:
                    logger.warning(f"削除対象のアラートが見つかりません: {stock_code}")
                    return False

        except Exception as e:
            logger.error(f"アラート削除エラー: {e}")
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
                query = session.query(Alert).join(Stock)

                if stock_code:
                    query = query.filter(Alert.stock_code == stock_code)

                if active_only:
                    query = query.filter(Alert.is_active == True)

                alerts = query.all()

                result = []
                for alert in alerts:
                    result.append(
                        {
                            "id": alert.id,
                            "stock_code": alert.stock_code,
                            "stock_name": (
                                alert.stock.name if alert.stock else alert.stock_code
                            ),
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
            logger.error(f"アラート取得エラー: {e}")
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
                    logger.info(
                        f"アラート状態を変更: {alert.stock_code} -> {alert.is_active}"
                    )
                    return True
                else:
                    logger.warning(f"対象のアラートが見つかりません: {alert_id}")
                    return False

        except Exception as e:
            logger.error(f"アラート切り替えエラー: {e}")
            return False

    def check_alerts(self) -> List[AlertNotification]:
        """
        アラート条件をチェックして通知リストを生成

        Returns:
            トリガーされたアラートの通知リスト
        """
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
                            logger.warning(f"価格データが取得できません: {code}")
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
                        logger.error(
                            f"アラートチェックエラー ({alert['stock_code']}): {e}"
                        )
                        continue

        except Exception as e:
            logger.error(f"アラート一括チェックエラー: {e}")

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
            if alert_type == AlertType.PRICE_ABOVE.value:
                return price_data["current_price"] > threshold

            elif alert_type == AlertType.PRICE_BELOW.value:
                return price_data["current_price"] < threshold

            elif alert_type == AlertType.CHANGE_PERCENT_UP.value:
                return price_data.get("change_percent", 0) > threshold

            elif alert_type == AlertType.CHANGE_PERCENT_DOWN.value:
                return price_data.get("change_percent", 0) < threshold

            elif alert_type == AlertType.VOLUME_SPIKE.value:
                # 出来高が平均の何倍かをチェック（簡易版）
                volume = price_data.get("volume", 0)
                # 実際の実装では過去平均出来高と比較すべき
                return volume > threshold

            else:
                logger.warning(f"未知のアラートタイプ: {alert_type}")
                return False

        except Exception as e:
            logger.error(f"アラート条件チェックエラー: {e}")
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
            logger.error(f"サマリー取得エラー: {e}")
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
                logger.warning("エクスポートするデータがありません")
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

            logger.info(f"ウォッチリストをエクスポートしました: {filename}")
            return True

        except Exception as e:
            logger.error(f"CSVエクスポートエラー: {e}")
            return False
