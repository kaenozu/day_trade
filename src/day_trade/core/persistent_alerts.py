"""
永続化対応のアラートマネージャー
データベースを使用してアラート条件と履歴を永続化
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from decimal import Decimal

from sqlalchemy.orm import Session

from .alerts import (
    AlertCondition, AlertTrigger, AlertManager, NotificationHandler,
    NotificationMethod, AlertPriority
)
from ..models.alerts import AlertConditionModel, AlertTriggerModel, AlertConfigModel
from ..models.database import db_manager
from ..data.stock_fetcher import StockFetcher
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class PersistentAlertManager(AlertManager):
    """永続化対応アラートマネージャー"""

    def __init__(
        self,
        stock_fetcher: Optional[StockFetcher] = None,
        db_manager_instance=None
    ):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
            db_manager_instance: データベースマネージャー（テスト用）
        """
        # 親クラスの初期化（メモリベースの初期化）
        super().__init__(stock_fetcher)

        # データベースマネージャー
        self.db_manager = db_manager_instance or db_manager

        # データベースの初期化
        self._ensure_tables()

        # データベースからアラート条件をロード
        self._load_alert_conditions()

        # 設定の読み込み
        self._load_settings()

    def _ensure_tables(self):
        """必要なテーブルの作成"""
        try:
            # AlertConditionModel, AlertTriggerModel, AlertConfigModelのテーブルを作成
            self.db_manager.create_tables()
            logger.info("アラート関連テーブルを確認・作成しました")
        except Exception as e:
            logger.error(f"テーブル作成中にエラーが発生しました: {e}")

    def _load_alert_conditions(self):
        """データベースからアラート条件をロード"""
        try:
            with self.db_manager.session_scope() as session:
                conditions = session.query(AlertConditionModel).all()

                # メモリ内の辞書をクリア
                self.alert_conditions.clear()

                # データベースから読み込み
                for condition_model in conditions:
                    condition = condition_model.to_alert_condition()
                    self.alert_conditions[condition.alert_id] = condition

                logger.info(f"データベースから {len(conditions)} 件のアラート条件をロードしました")

        except Exception as e:
            logger.error(f"アラート条件のロード中にエラーが発生: {e}")

    def _load_settings(self):
        """設定をデータベースから読み込み"""
        try:
            with self.db_manager.session_scope() as session:
                # 監視間隔の設定
                monitoring_interval = AlertConfigModel.get_config(
                    session, "monitoring_interval", 60
                )
                self.monitoring_interval = monitoring_interval

                # デフォルト通知方法
                default_methods = AlertConfigModel.get_config(
                    session, "default_notification_methods",
                    [method.value for method in self.default_notification_methods]
                )
                self.default_notification_methods = [
                    NotificationMethod(method) for method in default_methods
                ]

                logger.debug("設定をデータベースから読み込みました")

        except Exception as e:
            logger.error(f"設定の読み込み中にエラーが発生: {e}")

    def add_alert(self, condition: AlertCondition) -> bool:
        """アラート条件を追加（データベースに永続化）"""
        try:
            # 親クラスのバリデーション
            if not self._validate_condition(condition):
                logger.error(f"アラート条件の検証に失敗: {condition.alert_id}")
                return False

            # データベースに保存
            with self.db_manager.session_scope() as session:
                # 既存の条件があるかチェック
                existing = session.query(AlertConditionModel).filter_by(
                    alert_id=condition.alert_id
                ).first()

                if existing:
                    # 更新
                    existing.symbol = condition.symbol
                    existing.alert_type = condition.alert_type.value
                    existing.condition_value = str(condition.condition_value)
                    existing.comparison_operator = condition.comparison_operator
                    existing.is_active = condition.is_active
                    existing.priority = condition.priority.value
                    existing.cooldown_minutes = condition.cooldown_minutes
                    existing.expiry_date = condition.expiry_date
                    existing.description = condition.description
                    existing.custom_parameters = condition.custom_parameters
                    logger.info(f"アラート条件を更新: {condition.alert_id}")
                else:
                    # 新規作成
                    condition_model = AlertConditionModel.from_alert_condition(condition)
                    session.add(condition_model)
                    logger.info(f"アラート条件を追加: {condition.alert_id}")

            # メモリ内の辞書も更新
            self.alert_conditions[condition.alert_id] = condition
            return True

        except Exception as e:
            logger.error(f"アラート条件の保存中にエラーが発生: {e}")
            return False

    def remove_alert(self, alert_id: str) -> bool:
        """アラート条件を削除（データベースからも削除）"""
        try:
            with self.db_manager.session_scope() as session:
                condition = session.query(AlertConditionModel).filter_by(
                    alert_id=alert_id
                ).first()

                if condition:
                    session.delete(condition)
                    # メモリからも削除
                    if alert_id in self.alert_conditions:
                        del self.alert_conditions[alert_id]
                    logger.info(f"アラート条件を削除: {alert_id}")
                    return True
                else:
                    logger.warning(f"削除対象のアラート条件が見つかりません: {alert_id}")
                    return False

        except Exception as e:
            logger.error(f"アラート条件の削除中にエラーが発生: {e}")
            return False

    def _handle_alert_trigger(self, trigger: AlertTrigger):
        """アラート発火の処理（データベースに永続化）"""
        try:
            # データベースに保存
            with self.db_manager.session_scope() as session:
                trigger_model = AlertTriggerModel.from_alert_trigger(trigger)
                session.add(trigger_model)

            # 親クラスの処理（通知送信など）
            super()._handle_alert_trigger(trigger)

            # 履歴の制限（データベース上でも古い履歴を削除）
            self._cleanup_old_triggers()

        except Exception as e:
            logger.error(f"アラート発火の処理中にエラーが発生: {e}")

    def _cleanup_old_triggers(self, keep_days: int = 30):
        """古いアラート履歴を削除"""
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)

            with self.db_manager.session_scope() as session:
                deleted_count = session.query(AlertTriggerModel).filter(
                    AlertTriggerModel.trigger_time < cutoff_date
                ).delete()

                if deleted_count > 0:
                    logger.info(f"古いアラート履歴 {deleted_count} 件を削除しました")

        except Exception as e:
            logger.error(f"古いアラート履歴の削除中にエラーが発生: {e}")

    def get_alert_history(
        self, symbol: Optional[str] = None, hours: int = 24
    ) -> List[AlertTrigger]:
        """アラート履歴を取得（データベースから）"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            with self.db_manager.session_scope() as session:
                query = session.query(AlertTriggerModel).filter(
                    AlertTriggerModel.trigger_time >= cutoff_time
                )

                if symbol:
                    query = query.filter(AlertTriggerModel.symbol == symbol)

                trigger_models = query.order_by(
                    AlertTriggerModel.trigger_time.desc()
                ).all()

                # DataclassのAlertTriggerに変換
                return [model.to_alert_trigger() for model in trigger_models]

        except Exception as e:
            logger.error(f"アラート履歴の取得中にエラーが発生: {e}")
            return []

    def get_alert_statistics(self, days: int = 7) -> Dict:
        """アラート統計を取得"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)

            with self.db_manager.session_scope() as session:
                triggers = session.query(AlertTriggerModel).filter(
                    AlertTriggerModel.trigger_time >= cutoff_time
                ).all()

                stats = {
                    "total_triggers": len(triggers),
                    "by_symbol": {},
                    "by_type": {},
                    "by_priority": {},
                    "by_day": {}
                }

                for trigger in triggers:
                    # 銘柄別統計
                    symbol = trigger.symbol
                    if symbol not in stats["by_symbol"]:
                        stats["by_symbol"][symbol] = 0
                    stats["by_symbol"][symbol] += 1

                    # タイプ別統計
                    alert_type = trigger.alert_type
                    if alert_type not in stats["by_type"]:
                        stats["by_type"][alert_type] = 0
                    stats["by_type"][alert_type] += 1

                    # 優先度別統計
                    priority = trigger.priority
                    if priority not in stats["by_priority"]:
                        stats["by_priority"][priority] = 0
                    stats["by_priority"][priority] += 1

                    # 日別統計
                    day = trigger.trigger_time.date().isoformat()
                    if day not in stats["by_day"]:
                        stats["by_day"][day] = 0
                    stats["by_day"][day] += 1

                return stats

        except Exception as e:
            logger.error(f"アラート統計の取得中にエラーが発生: {e}")
            return {}

    def configure_notifications(self, methods: List[NotificationMethod]):
        """通知方法の設定（データベースに永続化）"""
        try:
            super().configure_notifications(methods)

            # データベースに設定を保存
            with self.db_manager.session_scope() as session:
                AlertConfigModel.set_config(
                    session,
                    "default_notification_methods",
                    [method.value for method in methods],
                    "デフォルトの通知方法"
                )

            logger.info(f"通知方法を設定: {[m.value for m in methods]}")

        except Exception as e:
            logger.error(f"通知方法の設定中にエラーが発生: {e}")

    def set_monitoring_interval(self, seconds: int):
        """監視間隔の設定（データベースに永続化）"""
        try:
            self.monitoring_interval = seconds

            # データベースに設定を保存
            with self.db_manager.session_scope() as session:
                AlertConfigModel.set_config(
                    session,
                    "monitoring_interval",
                    seconds,
                    "アラート監視間隔（秒）"
                )

            logger.info(f"監視間隔を設定: {seconds}秒")

        except Exception as e:
            logger.error(f"監視間隔の設定中にエラーが発生: {e}")

    def export_all_data(self, filename: str):
        """すべてのアラートデータをエクスポート"""
        try:
            export_data = {
                "alert_conditions": [],
                "alert_history": [],
                "config": {},
                "export_time": datetime.now().isoformat()
            }

            with self.db_manager.session_scope() as session:
                # アラート条件
                conditions = session.query(AlertConditionModel).all()
                for condition in conditions:
                    condition_data = condition.to_dict()
                    export_data["alert_conditions"].append(condition_data)

                # アラート履歴（最近30日分）
                cutoff_time = datetime.now() - timedelta(days=30)
                triggers = session.query(AlertTriggerModel).filter(
                    AlertTriggerModel.trigger_time >= cutoff_time
                ).all()
                for trigger in triggers:
                    trigger_data = trigger.to_dict()
                    export_data["alert_history"].append(trigger_data)

                # 設定
                configs = session.query(AlertConfigModel).all()
                for config in configs:
                    export_data["config"][config.config_key] = config.config_value

            # ファイルに保存
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"アラートデータをエクスポート: {filename}")

        except Exception as e:
            logger.error(f"アラートデータのエクスポート中にエラーが発生: {e}")


# ファクトリー関数
def create_persistent_alert_manager(
    stock_fetcher: Optional[StockFetcher] = None,
    db_manager_instance=None
) -> PersistentAlertManager:
    """永続化対応アラートマネージャーを作成"""
    return PersistentAlertManager(
        stock_fetcher=stock_fetcher,
        db_manager_instance=db_manager_instance
    )
