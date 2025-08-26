#!/usr/bin/env python3
"""
監視システムコアモジュール
高度データ鮮度・整合性監視システムのメインクラスとデータソース管理
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from ...monitoring.advanced_anomaly_detection_alerts import (
        AdvancedAnomalyAlertSystem,
    )
    from ...monitoring.structured_logging_enhancement import (
        StructuredLoggingEnhancementSystem,
    )
    from ...utils.unified_cache_manager import UnifiedCacheManager
    from ..comprehensive_data_quality_system import ComprehensiveDataQualitySystem

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from .alert_manager import AlertManager
from .dashboard import DashboardManager
from .database import DatabaseManager
from .freshness_checker import FreshnessCheckManager
from .integrity_checker import IntegrityCheckManager
from .models import DataAlert, DataSourceConfig, MonitoringLevel
from .recovery_manager import RecoveryManager
from .sla_metrics import SLAMetricsManager


class AdvancedDataFreshnessMonitor:
    """高度データ鮮度・整合性監視システム"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # 設定管理
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.monitoring_active = False
        self.monitor_thread = None

        # コンポーネント初期化
        self.database_manager = DatabaseManager()
        self.freshness_checker = FreshnessCheckManager()
        self.integrity_checker = IntegrityCheckManager(self.database_manager)
        self.alert_manager = AlertManager(self.database_manager)
        self.recovery_manager = RecoveryManager(self.database_manager)
        self.sla_metrics_manager = SLAMetricsManager(self.database_manager)
        self.dashboard_manager = DashboardManager(self.database_manager)

        # 外部コンポーネント
        self.quality_system = None
        self.anomaly_detector = None
        self.cache_manager = None

        # メトリクス・統計
        self.monitoring_stats = {
            "total_checks_performed": 0,
            "freshness_violations": 0,
            "integrity_violations": 0,
            "recovery_actions_taken": 0,
            "alerts_generated": 0,
            "sla_violations": 0,
        }

        # 監視データ (インメモリ)
        self.recent_checks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, List[DataAlert]] = defaultdict(list)

        if config_path:
            self.load_config(config_path)

        self._initialize_components()

    def _initialize_components(self):
        """コンポーネント初期化"""
        if DEPENDENCIES_AVAILABLE:
            try:
                self.quality_system = ComprehensiveDataQualitySystem()
                self.anomaly_detector = AdvancedAnomalyAlertSystem()
                self.cache_manager = UnifiedCacheManager()
                self.logger.info("監視システムコンポーネント初期化完了")
            except Exception as e:
                self.logger.warning(f"コンポーネント初期化エラー: {e}")

    def add_data_source(self, config: DataSourceConfig):
        """データソース追加"""
        self.data_sources[config.source_id] = config
        self.database_manager.initialize_data_source_state(config)
        self.logger.info(f"データソース追加: {config.source_id}")

    def remove_data_source(self, source_id: str):
        """データソース削除"""
        if source_id in self.data_sources:
            del self.data_sources[source_id]

            # アクティブアラートをクローズ
            if source_id in self.active_alerts:
                for alert in self.active_alerts[source_id]:
                    alert.resolved = True
                    alert.resolved_at = datetime.now(timezone.utc)
                    alert.resolution_notes = "データソース削除により自動解決"
                del self.active_alerts[source_id]

            self.logger.info(f"データソース削除: {source_id}")

    def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            self.logger.warning("監視は既に実行中です")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()

        self.logger.info("データ鮮度・整合性監視開始")

    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        self.logger.info("データ鮮度・整合性監視停止")

    def _monitoring_loop(self):
        """監視ループ（スレッド実行）"""
        while self.monitoring_active:
            try:
                # 各データソースをチェック
                for source_config in self.data_sources.values():
                    asyncio.run(self._check_data_source(source_config))

                # SLAメトリクス計算（毎時）
                current_time = datetime.now(timezone.utc)
                if current_time.minute == 0:  # 毎時0分
                    asyncio.run(
                        self.sla_metrics_manager.calculate_hourly_sla_metrics(
                            self.data_sources
                        )
                    )

                # 10秒待機
                time.sleep(10)

            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(30)  # エラー時は長めに待機

    async def _check_data_source(self, config: DataSourceConfig):
        """データソースチェック"""
        try:
            source_id = config.source_id

            # 鮮度チェック
            freshness_result = await self.freshness_checker.perform_freshness_check(
                config
            )

            # 整合性チェック（レベルに応じて）
            integrity_results = []
            if config.monitoring_level in [
                MonitoringLevel.COMPREHENSIVE,
                MonitoringLevel.CRITICAL,
            ]:
                integrity_results = await self.integrity_checker.perform_integrity_checks(
                    config
                )

            # 結果保存
            await self.database_manager.save_freshness_check(freshness_result)
            for integrity_result in integrity_results:
                await self.database_manager.save_integrity_check(integrity_result)

            # インメモリデータ更新
            self.recent_checks[source_id].append(freshness_result)
            self.monitoring_stats["total_checks_performed"] += 1

            # アラート判定
            alerts = await self.alert_manager.evaluate_alerts(
                config, freshness_result, integrity_results
            )

            # 統計更新
            if any(alert.alert_type == "data_freshness" for alert in alerts):
                self.monitoring_stats["freshness_violations"] += 1
            if any(alert.alert_type == "data_integrity" for alert in alerts):
                self.monitoring_stats["integrity_violations"] += 1
            self.monitoring_stats["alerts_generated"] += len(alerts)

            # アクティブアラートリストに追加
            self.active_alerts[source_id].extend(alerts)

            # データソース状態更新
            await self._update_source_state(config, freshness_result, integrity_results)

            # 回復アクション判定
            await self.recovery_manager.evaluate_recovery_actions(
                config, freshness_result, integrity_results
            )
            if alerts:  # 何らかのアラートがある場合は回復アクションを実行したと仮定
                self.monitoring_stats["recovery_actions_taken"] += 1

        except Exception as e:
            self.logger.error(f"データソースチェックエラー ({config.source_id}): {e}")

    async def _update_source_state(
        self,
        config: DataSourceConfig,
        freshness_result,
        integrity_results,
    ):
        """データソース状態更新"""
        try:
            current_time = datetime.now(timezone.utc)

            # 総合ステータス判定
            is_healthy = (
                self.freshness_checker.is_data_fresh(freshness_result)
                and not self.integrity_checker.has_integrity_issues(integrity_results)
            )

            current_status = "healthy" if is_healthy else "unhealthy"

            # 現在の状態取得
            source_state = self.database_manager.get_source_state(config.source_id)
            consecutive_failures = source_state[0] if source_state else 0

            # 失敗カウント更新
            if is_healthy:
                consecutive_failures = 0
                last_success = current_time
            else:
                consecutive_failures += 1
                last_success = None

            # データベース更新
            self.database_manager.update_source_state(
                config.source_id, current_status, consecutive_failures, last_success
            )

        except Exception as e:
            self.logger.error(f"データソース状態更新エラー ({config.source_id}): {e}")

    # コールバック管理
    def add_alert_callback(self, callback):
        """アラートコールバック追加"""
        self.alert_manager.add_alert_callback(callback)

    def add_recovery_callback(self, action, callback):
        """回復アクションコールバック追加"""
        self.recovery_manager.add_recovery_callback(action, callback)

    # ダッシュボード・レポート
    async def get_monitoring_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """監視ダッシュボードデータ取得"""
        return await self.dashboard_manager.get_monitoring_dashboard(
            self.data_sources, self.monitoring_stats, self.monitoring_active, hours
        )

    async def get_sla_summary(self, hours: int = 24) -> Dict[str, Any]:
        """SLAサマリー取得"""
        return await self.sla_metrics_manager.get_sla_summary(hours)

    async def generate_summary_report(self, hours: int = 24) -> str:
        """サマリーレポート生成"""
        dashboard_data = await self.get_monitoring_dashboard(hours)
        return await self.dashboard_manager.generate_summary_report(dashboard_data)

    # アラート管理
    async def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """アラート解決"""
        await self.alert_manager.resolve_alert(alert_id, resolution_notes)

        # アクティブアラートから削除
        for source_alerts in self.active_alerts.values():
            for alert in source_alerts[:]:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now(timezone.utc)
                    alert.resolution_notes = resolution_notes
                    source_alerts.remove(alert)
                    break

    # 設定管理
    def load_config(self, config_path: str):
        """設定ファイル読み込み"""
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)

            for source_data in config_data.get("data_sources", []):
                config = DataSourceConfig(
                    source_id=source_data["source_id"],
                    source_type=source_data["source_type"],
                    endpoint_url=source_data.get("endpoint_url"),
                    connection_params=source_data.get("connection_params", {}),
                    expected_frequency=source_data.get("expected_frequency", 60),
                    freshness_threshold=source_data.get("freshness_threshold", 300),
                    quality_threshold=source_data.get("quality_threshold", 80.0),
                    monitoring_level=source_data.get("monitoring_level", "standard"),
                    enable_recovery=source_data.get("enable_recovery", True),
                    recovery_strategy=source_data.get("recovery_strategy", "retry"),
                    max_retry_attempts=source_data.get("max_retry_attempts", 3),
                    sla_target=source_data.get("sla_target", 99.9),
                )
                self.add_data_source(config)

            self.logger.info(
                f"設定読み込み完了: {len(self.data_sources)}個のデータソース"
            )

        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")

    # 統計・メトリクス取得
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """監視統計取得"""
        return {
            "monitoring_stats": self.monitoring_stats.copy(),
            "active_sources": len(self.data_sources),
            "monitoring_active": self.monitoring_active,
            "total_active_alerts": sum(len(alerts) for alerts in self.active_alerts.values()),
        }

    def get_source_health(self, source_id: str) -> Optional[Dict[str, Any]]:
        """データソースヘルス取得"""
        if source_id not in self.data_sources:
            return None

        recent_checks_list = list(self.recent_checks.get(source_id, []))
        if not recent_checks_list:
            return {"status": "no_data", "message": "チェックデータがありません"}

        latest_check = recent_checks_list[-1]
        active_alerts = self.active_alerts.get(source_id, [])

        return {
            "source_id": source_id,
            "latest_status": latest_check.status.value,
            "latest_age_seconds": latest_check.age_seconds,
            "latest_quality_score": latest_check.quality_score,
            "active_alerts_count": len(active_alerts),
            "recent_checks_count": len(recent_checks_list),
            "last_check_time": latest_check.timestamp.isoformat(),
        }


# Factory function
def create_advanced_freshness_monitor(
    config_path: Optional[str] = None,
) -> AdvancedDataFreshnessMonitor:
    """高度データ鮮度監視システム作成"""
    return AdvancedDataFreshnessMonitor(config_path)