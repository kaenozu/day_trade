#!/usr/bin/env python3
"""
高度データ鮮度・整合性監視システム - メインモニタークラス
Issue #420: データ管理とデータ品質保証メカニズムの強化

リアルタイム品質監視と自動回復機能の統合管理を提供します。
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

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
from .database_operations import DatabaseOperations
from .enums import (
    AlertSeverity,
    DataSourceType,
    FreshnessStatus,
    MonitoringLevel,
    RecoveryAction,
)
from .freshness_checker import FreshnessChecker
from .integrity_checker import IntegrityChecker
from .models import (
    DataAlert,
    DataSourceConfig,
    FreshnessCheck,
    IntegrityCheck,
    MonitoringStats,
    SLAMetrics,
)
from .recovery_manager import RecoveryManager
from .sla_metrics import SLAMetricsCalculator


class AdvancedDataFreshnessMonitor:
    """高度データ鮮度・整合性監視システム

    マルチソースデータの鮮度監視、整合性チェック、SLA管理、
    自動回復機能を統合した包括的な監視システムです。

    主要機能:
    - リアルタイム鮮度監視
    - 多層整合性チェック
    - インテリジェントアラート管理
    - 自動回復アクション
    - SLA追跡・レポート
    - 包括的ダッシュボード
    """

    def __init__(self, config_path: Optional[str] = None):
        """監視システム初期化

        Args:
            config_path: 設定ファイルパス（オプション）
        """
        self.logger = logging.getLogger(__name__)

        # 設定管理
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.monitoring_active = False
        self.monitor_thread = None

        # データベース設定
        self.db_path = "data_freshness_monitor.db"

        # コンポーネント初期化
        self.db_ops = DatabaseOperations(self.db_path)
        self.freshness_checker = FreshnessChecker()
        self.integrity_checker = IntegrityChecker(self.db_path)
        self.alert_manager = AlertManager(self.db_path)
        self.recovery_manager = RecoveryManager(self.db_path)
        self.sla_calculator = SLAMetricsCalculator(self.db_path)
        self.dashboard_manager = DashboardManager(self.db_path)

        # 外部コンポーネント
        self.quality_system = None
        self.anomaly_detector = None
        self.cache_manager = (
            UnifiedCacheManager() if DEPENDENCIES_AVAILABLE else None
        )

        # 監視統計
        self.monitoring_stats = MonitoringStats()

        # 監視データ (インメモリ)
        self.recent_checks: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.active_alerts: Dict[str, List[DataAlert]] = defaultdict(list)
        self.sla_metrics: Dict[str, List[SLAMetrics]] = defaultdict(list)

        # イベントコールバック
        self.alert_callbacks: List[Callable] = []
        self.recovery_callbacks: Dict[RecoveryAction, List[Callable]] = (
            defaultdict(list)
        )

        # データベース初期化
        self.db_ops.initialize_database()

        # 設定ファイル読み込み
        if config_path:
            self.load_config(config_path)

        # コンポーネント初期化
        self._initialize_components()

    def _initialize_components(self):
        """外部コンポーネント初期化"""
        if DEPENDENCIES_AVAILABLE:
            try:
                self.quality_system = ComprehensiveDataQualitySystem()
                self.anomaly_detector = AdvancedAnomalyAlertSystem()
                self.logger.info("監視システムコンポーネント初期化完了")
            except Exception as e:
                self.logger.warning(f"コンポーネント初期化エラー: {e}")

    def add_data_source(self, config: DataSourceConfig):
        """データソース追加

        Args:
            config: データソース設定
        """
        self.data_sources[config.source_id] = config
        self.db_ops.initialize_data_source(config)
        self.logger.info(f"データソース追加: {config.source_id}")

    def remove_data_source(self, source_id: str):
        """データソース削除

        Args:
            source_id: データソースID
        """
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
                    asyncio.run(self._calculate_hourly_sla_metrics())

                # 10秒待機
                time.sleep(10)

            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(30)  # エラー時は長めに待機

    async def _check_data_source(self, config: DataSourceConfig):
        """データソースチェック

        Args:
            config: データソース設定
        """
        try:
            source_id = config.source_id

            # 鮮度チェック
            freshness_result = await self.freshness_checker.perform_freshness_check(
                config, self.quality_system
            )

            # 整合性チェック（レベルに応じて）
            integrity_results = []
            if config.monitoring_level in [
                MonitoringLevel.COMPREHENSIVE,
                MonitoringLevel.CRITICAL,
            ]:
                integrity_results = (
                    await self.integrity_checker.perform_integrity_checks(config)
                )

            # 結果保存
            await self.db_ops.save_freshness_check(freshness_result)
            for integrity_result in integrity_results:
                await self.db_ops.save_integrity_check(integrity_result)

            # インメモリデータ更新
            self.recent_checks[source_id].append(freshness_result)
            self.monitoring_stats.total_checks_performed += 1

            # アラート判定
            alerts = await self.alert_manager.evaluate_alerts(
                config, freshness_result, integrity_results
            )

            # 統計更新
            self.monitoring_stats.freshness_violations += (
                1 if freshness_result.status != FreshnessStatus.FRESH else 0
            )
            self.monitoring_stats.integrity_violations += sum(
                1 for check in integrity_results if not check.passed
            )
            self.monitoring_stats.alerts_generated += len(alerts)

            # アクティブアラート更新
            for alert in alerts:
                self.active_alerts[source_id].append(alert)
                # コールバック実行
                for callback in self.alert_callbacks:
                    try:
                        await callback(alert)
                    except Exception as e:
                        self.logger.error(f"アラートコールバックエラー: {e}")

            # データソース状態更新
            await self.db_ops.update_source_state(
                config, freshness_result, integrity_results
            )

            # 回復アクション判定
            if freshness_result.status != FreshnessStatus.FRESH or any(
                not r.passed for r in integrity_results
            ):
                recovery_executed = (
                    await self.recovery_manager.evaluate_and_execute_recovery(
                        config, freshness_result, integrity_results
                    )
                )
                if recovery_executed:
                    self.monitoring_stats.recovery_actions_taken += 1

        except Exception as e:
            self.logger.error(f"データソースチェックエラー ({config.source_id}): {e}")

    async def _calculate_hourly_sla_metrics(self):
        """時間別SLAメトリクス計算"""
        try:
            for source_id in self.data_sources.keys():
                sla_metrics = await self.sla_calculator.calculate_hourly_sla(
                    source_id, self.data_sources[source_id].sla_target
                )
                if sla_metrics:
                    self.sla_metrics[source_id].append(sla_metrics)
                    if sla_metrics.sla_violations > 0:
                        self.monitoring_stats.sla_violations += 1

        except Exception as e:
            self.logger.error(f"SLAメトリクス計算エラー: {e}")

    def add_alert_callback(self, callback: Callable):
        """アラートコールバック追加

        Args:
            callback: アラート発生時のコールバック関数
        """
        self.alert_callbacks.append(callback)

    def add_recovery_callback(self, action: RecoveryAction, callback: Callable):
        """回復アクションコールバック追加

        Args:
            action: 回復アクション種別
            callback: 回復アクション実行時のコールバック関数
        """
        self.recovery_callbacks[action].append(callback)

    async def get_monitoring_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """監視ダッシュボードデータ取得

        Args:
            hours: 表示時間範囲（時間）

        Returns:
            ダッシュボードデータ辞書
        """
        return await self.dashboard_manager.get_dashboard_data(
            self.data_sources, self.monitoring_stats, hours
        )

    async def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """アラート解決

        Args:
            alert_id: アラートID
            resolution_notes: 解決メモ
        """
        try:
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

            self.logger.info(f"アラート解決: {alert_id}")

        except Exception as e:
            self.logger.error(f"アラート解決エラー: {e}")

    def load_config(self, config_path: str):
        """設定ファイル読み込み

        Args:
            config_path: 設定ファイルパス
        """
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)

            for source_data in config_data.get("data_sources", []):
                config = DataSourceConfig(
                    source_id=source_data["source_id"],
                    source_type=DataSourceType(source_data["source_type"]),
                    endpoint_url=source_data.get("endpoint_url"),
                    connection_params=source_data.get("connection_params", {}),
                    expected_frequency=source_data.get("expected_frequency", 60),
                    freshness_threshold=source_data.get("freshness_threshold", 300),
                    quality_threshold=source_data.get("quality_threshold", 80.0),
                    monitoring_level=MonitoringLevel(
                        source_data.get("monitoring_level", "standard")
                    ),
                    enable_recovery=source_data.get("enable_recovery", True),
                    recovery_strategy=RecoveryAction(
                        source_data.get("recovery_strategy", "retry")
                    ),
                    max_retry_attempts=source_data.get("max_retry_attempts", 3),
                    sla_target=source_data.get("sla_target", 99.9),
                )
                self.add_data_source(config)

            self.logger.info(
                f"設定読み込み完了: {len(self.data_sources)}個のデータソース"
            )

        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")

    @property
    def stats(self) -> Dict[str, int]:
        """監視統計取得

        Returns:
            監視統計の辞書
        """
        return self.monitoring_stats.to_dict()


# Factory function
def create_advanced_freshness_monitor(
    config_path: Optional[str] = None,
) -> AdvancedDataFreshnessMonitor:
    """高度データ鮮度監視システム作成

    Args:
        config_path: 設定ファイルパス（オプション）

    Returns:
        AdvancedDataFreshnessMonitor: 監視システムインスタンス
    """
    return AdvancedDataFreshnessMonitor(config_path)


# バックワード互換性のためのエイリアス
AdvancedFreshnessMonitor = AdvancedDataFreshnessMonitor