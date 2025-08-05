"""
アラート永続化管理
データベースとメモリ間でのアラート条件・履歴の同期
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ..models.database import db_manager
from ..models.alerts import PersistentAlertCondition, PersistentAlertTrigger
from ..utils.logging_config import get_context_logger
from .alerts import AlertCondition, AlertTrigger

logger = get_context_logger(__name__)


class PersistentAlertsManager:
    """アラート永続化管理クラス"""

    def __init__(self, db_manager_instance=None):
        """
        初期化

        Args:
            db_manager_instance: データベースマネージャーインスタンス
        """
        self.db_manager = db_manager_instance or db_manager

    def save_alert_condition(self, condition: AlertCondition) -> bool:
        """
        アラート条件をデータベースに保存

        Args:
            condition: 保存するアラート条件

        Returns:
            bool: 保存成功フラグ
        """
        try:
            with self.db_manager.session_scope() as session:
                # 既存の条件をチェック
                existing = session.query(PersistentAlertCondition).filter(
                    PersistentAlertCondition.alert_id == condition.alert_id
                ).first()

                if existing:
                    # 既存の条件を更新
                    self._update_persistent_condition(existing, condition)
                    logger.info(f"アラート条件更新: {condition.alert_id}")
                else:
                    # 新規条件を作成
                    persistent_condition = PersistentAlertCondition.from_alert_condition(condition)
                    session.add(persistent_condition)
                    logger.info(f"アラート条件新規作成: {condition.alert_id}")

                session.commit()
                return True

        except Exception as e:
            logger.error(f"アラート条件保存エラー ({condition.alert_id}): {e}")
            return False

    def load_alert_conditions(self, active_only: bool = True) -> List[AlertCondition]:
        """
        データベースからアラート条件を読み込み

        Args:
            active_only: アクティブな条件のみ取得するか

        Returns:
            List[AlertCondition]: アラート条件のリスト
        """
        try:
            with self.db_manager.session_scope() as session:
                query = session.query(PersistentAlertCondition)

                if active_only:
                    query = query.filter(
                        PersistentAlertCondition.is_active == True,
                        PersistentAlertCondition.is_expired == False
                    )

                persistent_conditions = query.all()

                # メモリ上のオブジェクトに変換
                conditions = []
                for pc in persistent_conditions:
                    # 期限切れチェック
                    if pc.expires_at and pc.expires_at <= datetime.now():
                        pc.is_expired = True
                        continue

                    condition = pc.to_alert_condition()
                    conditions.append(condition)

                session.commit()  # 期限切れフラグの更新をコミット
                logger.info(f"アラート条件読み込み完了: {len(conditions)}件")
                return conditions

        except Exception as e:
            logger.error(f"アラート条件読み込みエラー: {e}")
            return []

    def save_alert_trigger(self, trigger: AlertTrigger) -> bool:
        """
        アラートトリガー（履歴）をデータベースに保存

        Args:
            trigger: 保存するアラートトリガー

        Returns:
            bool: 保存成功フラグ
        """
        try:
            with self.db_manager.session_scope() as session:
                persistent_trigger = PersistentAlertTrigger.from_alert_trigger(trigger)
                session.add(persistent_trigger)
                session.commit()

                logger.info(f"アラートトリガー保存: {trigger.alert_id} - {trigger.symbol}")
                return True

        except Exception as e:
            logger.error(f"アラートトリガー保存エラー ({trigger.alert_id}): {e}")
            return False

    def load_alert_history(
        self,
        symbol: Optional[str] = None,
        days_back: int = 30,
        limit: int = 100
    ) -> List[AlertTrigger]:
        """
        アラート履歴をデータベースから読み込み

        Args:
            symbol: 特定銘柄の履歴のみ取得（Noneで全銘柄）
            days_back: 遡る日数
            limit: 取得件数制限

        Returns:
            List[AlertTrigger]: アラート履歴のリスト
        """
        try:
            with self.db_manager.session_scope() as session:
                cutoff_date = datetime.now() - timedelta(days=days_back)

                query = session.query(PersistentAlertTrigger).filter(
                    PersistentAlertTrigger.trigger_time >= cutoff_date
                )

                if symbol:
                    query = query.filter(PersistentAlertTrigger.symbol == symbol)

                persistent_triggers = (
                    query.order_by(PersistentAlertTrigger.trigger_time.desc())
                    .limit(limit)
                    .all()
                )

                # メモリ上のオブジェクトに変換
                triggers = [pt.to_alert_trigger() for pt in persistent_triggers]

                logger.info(f"アラート履歴読み込み完了: {len(triggers)}件")
                return triggers

        except Exception as e:
            logger.error(f"アラート履歴読み込みエラー: {e}")
            return []

    def delete_alert_condition(self, alert_id: str) -> bool:
        """
        アラート条件を削除

        Args:
            alert_id: 削除するアラートID

        Returns:
            bool: 削除成功フラグ
        """
        try:
            with self.db_manager.session_scope() as session:
                condition = session.query(PersistentAlertCondition).filter(
                    PersistentAlertCondition.alert_id == alert_id
                ).first()

                if condition:
                    session.delete(condition)
                    session.commit()
                    logger.info(f"アラート条件削除: {alert_id}")
                    return True
                else:
                    logger.warning(f"削除対象のアラート条件が見つかりません: {alert_id}")
                    return False

        except Exception as e:
            logger.error(f"アラート条件削除エラー ({alert_id}): {e}")
            return False

    def deactivate_alert_condition(self, alert_id: str) -> bool:
        """
        アラート条件を非アクティブ化

        Args:
            alert_id: 非アクティブ化するアラートID

        Returns:
            bool: 非アクティブ化成功フラグ
        """
        try:
            with self.db_manager.session_scope() as session:
                condition = session.query(PersistentAlertCondition).filter(
                    PersistentAlertCondition.alert_id == alert_id
                ).first()

                if condition:
                    condition.is_active = False
                    session.commit()
                    logger.info(f"アラート条件非アクティブ化: {alert_id}")
                    return True
                else:
                    logger.warning(f"対象のアラート条件が見つかりません: {alert_id}")
                    return False

        except Exception as e:
            logger.error(f"アラート条件非アクティブ化エラー ({alert_id}): {e}")
            return False

    def cleanup_expired_conditions(self) -> int:
        """
        期限切れのアラート条件をクリーンアップ

        Returns:
            int: クリーンアップした件数
        """
        try:
            with self.db_manager.session_scope() as session:
                now = datetime.now()

                # 期限切れフラグを更新
                expired_count = session.query(PersistentAlertCondition).filter(
                    PersistentAlertCondition.expires_at <= now,
                    PersistentAlertCondition.is_expired == False
                ).update({
                    PersistentAlertCondition.is_expired: True,
                    PersistentAlertCondition.is_active: False
                })

                session.commit()
                logger.info(f"期限切れアラート条件クリーンアップ: {expired_count}件")
                return expired_count

        except Exception as e:
            logger.error(f"期限切れアラート条件クリーンアップエラー: {e}")
            return 0

    def get_alert_statistics(self) -> Dict[str, int]:
        """
        アラート統計情報を取得

        Returns:
            Dict[str, int]: 統計情報
        """
        try:
            with self.db_manager.session_scope() as session:
                stats = {}

                # アクティブなアラート条件数
                stats['active_conditions'] = session.query(PersistentAlertCondition).filter(
                    PersistentAlertCondition.is_active == True,
                    PersistentAlertCondition.is_expired == False
                ).count()

                # 期限切れアラート条件数
                stats['expired_conditions'] = session.query(PersistentAlertCondition).filter(
                    PersistentAlertCondition.is_expired == True
                ).count()

                # 今日のトリガー数
                from datetime import date
                today = date.today()
                stats['todays_triggers'] = session.query(PersistentAlertTrigger).filter(
                    PersistentAlertTrigger.trigger_time >= datetime.combine(today, datetime.min.time())
                ).count()

                # 総トリガー数
                stats['total_triggers'] = session.query(PersistentAlertTrigger).count()

                return stats

        except Exception as e:
            logger.error(f"アラート統計情報取得エラー: {e}")
            return {}

    def _update_persistent_condition(self, existing: PersistentAlertCondition, condition: AlertCondition):
        """既存の永続化条件を更新"""
        existing.symbol = condition.symbol
        existing.alert_type = condition.alert_type
        existing.condition_value = condition.condition_value
        existing.comparison_operator = condition.comparison_operator
        existing.priority = condition.priority
        existing.description = condition.description
        existing.is_active = condition.is_active
        existing.expires_at = condition.expires_at
        existing.custom_parameters = condition.custom_parameters
        existing.notification_methods = condition.notification_methods

    def sync_conditions_to_memory(self, alert_manager) -> int:
        """
        データベースからメモリにアラート条件を同期

        Args:
            alert_manager: AlertManagerインスタンス

        Returns:
            int: 同期した条件数
        """
        try:
            conditions = self.load_alert_conditions(active_only=True)

            # メモリ上の条件をクリア
            alert_manager.alert_conditions.clear()

            # データベースから読み込んだ条件を設定
            for condition in conditions:
                alert_manager.alert_conditions[condition.alert_id] = condition

            logger.info(f"アラート条件をメモリに同期: {len(conditions)}件")
            return len(conditions)

        except Exception as e:
            logger.error(f"アラート条件メモリ同期エラー: {e}")
            return 0

    def sync_conditions_to_database(self, alert_manager) -> int:
        """
        メモリからデータベースにアラート条件を同期

        Args:
            alert_manager: AlertManagerインスタンス

        Returns:
            int: 同期した条件数
        """
        try:
            success_count = 0

            for condition in alert_manager.alert_conditions.values():
                if self.save_alert_condition(condition):
                    success_count += 1

            logger.info(f"アラート条件をデータベースに同期: {success_count}件")
            return success_count

        except Exception as e:
            logger.error(f"アラート条件データベース同期エラー: {e}")
            return 0
