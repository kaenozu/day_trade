#!/usr/bin/env python3
"""
Basic Monitor Core
基本監視システムのコア機能

メインの監視システムクラスの実装
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .alert_handler import AlertHandler
from .consistency_checker import ConsistencyCheck
from .freshness_checker import FreshnessCheck
from .models import (
    AlertSeverity,
    AlertType,
    DataSourceHealth,
    MonitorAlert,
    MonitorRule,
    MonitorStatus,
    RecoveryAction,
    SLAMetrics,
)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)

try:
    from ...utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    class UnifiedCacheManager:
        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"monitor_key_{hash(str(args))}"

# データ品質レベル定義（依存関係を削除）
class DataQualityLevel:
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


logger = get_context_logger(__name__)


class DataFreshnessMonitor:
    """データ鮮度・整合性監視システム"""

    def __init__(
        self,
        storage_path: str = "data/monitoring",
        enable_cache: bool = True,
        alert_retention_days: int = 30,
        check_interval_seconds: int = 300,
    ):
        self.storage_path = Path(storage_path)
        self.enable_cache = enable_cache
        self.alert_retention_days = alert_retention_days
        self.check_interval_seconds = check_interval_seconds

        # ディレクトリ初期化
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # キャッシュマネージャー初期化
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=32, l2_memory_mb=128, l3_disk_mb=256
                )
                logger.info("監視システムキャッシュ初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None

        # 監視状態管理
        self.monitor_status = MonitorStatus.INACTIVE
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # 監視ルール管理
        self.monitor_rules: Dict[str, MonitorRule] = {}

        # チェック実装
        self.checks: Dict[str, Any] = {
            "freshness": FreshnessCheck(threshold_minutes=60),
            "consistency": ConsistencyCheck(),
        }

        # データソースヘルス管理
        self.data_source_health: Dict[str, DataSourceHealth] = {}

        # アラート管理
        self.active_alerts: Dict[str, MonitorAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)

        # SLA管理
        self.sla_metrics: Dict[str, SLAMetrics] = {}

        # メトリクス管理
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))

        # アラートハンドラー
        self.alert_handler = AlertHandler(self.storage_path, alert_retention_days)

        # デフォルト設定
        self._setup_default_rules()
        self._setup_default_sla()

        logger.info("データ鮮度・整合性監視システム初期化完了")
        logger.info(f"  - ストレージパス: {self.storage_path}")
        logger.info(f"  - チェック間隔: {check_interval_seconds}秒")
        logger.info(f"  - アラート保持期間: {alert_retention_days}日")

    def _setup_default_rules(self):
        """デフォルト監視ルール設定"""
        # 価格データ鮮度ルール
        price_freshness_rule = MonitorRule(
            rule_id="price_freshness",
            name="価格データ鮮度監視",
            description="価格データが1時間以内に更新されているかチェック",
            data_source="price_data",
            rule_type="freshness",
            threshold_value=60,
            threshold_unit="minutes",
            severity=AlertSeverity.HIGH,
            check_interval_seconds=300,
            recovery_actions=[RecoveryAction.RETRY_FETCH, RecoveryAction.USE_FALLBACK],
        )
        self.monitor_rules[price_freshness_rule.rule_id] = price_freshness_rule

        # ニュースデータ鮮度ルール
        news_freshness_rule = MonitorRule(
            rule_id="news_freshness",
            name="ニュースデータ鮮度監視",
            description="ニュースデータが4時間以内に更新されているかチェック",
            data_source="news_data",
            rule_type="freshness",
            threshold_value=240,
            threshold_unit="minutes",
            severity=AlertSeverity.MEDIUM,
            check_interval_seconds=600,
        )
        self.monitor_rules[news_freshness_rule.rule_id] = news_freshness_rule

        # データ整合性ルール
        consistency_rule = MonitorRule(
            rule_id="data_consistency",
            name="データ整合性監視",
            description="データの論理的整合性をチェック",
            data_source="all",
            rule_type="consistency",
            threshold_value=0,
            threshold_unit="violations",
            severity=AlertSeverity.HIGH,
            check_interval_seconds=300,
            recovery_actions=[RecoveryAction.AUTO_FIX, RecoveryAction.NOTIFY_ADMIN],
        )
        self.monitor_rules[consistency_rule.rule_id] = consistency_rule

    def _setup_default_sla(self):
        """デフォルトSLA設定"""
        # 価格データSLA
        price_sla = SLAMetrics(
            sla_id="price_data_sla",
            name="価格データSLA",
            target_availability=0.999,  # 99.9%
            target_freshness_minutes=60,
            target_quality_score=0.95,
            current_availability=1.0,
            current_freshness_minutes=0.0,
            current_quality_score=1.0,
            violations_count=0,
            measurement_period="daily",
        )
        self.sla_metrics[price_sla.sla_id] = price_sla

    async def start_monitoring(self):
        """監視開始"""
        if self.monitor_status == MonitorStatus.ACTIVE:
            logger.warning("監視は既にアクティブです")
            return

        logger.info("データ鮮度・整合性監視開始")

        self.monitor_status = MonitorStatus.ACTIVE
        self.stop_event.clear()

        # 監視スレッド開始
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, name="DataFreshnessMonitor", daemon=True
        )
        self.monitor_thread.start()

        logger.info("監視システム開始完了")

    async def stop_monitoring(self):
        """監視停止"""
        if self.monitor_status != MonitorStatus.ACTIVE:
            logger.warning("監視は既に停止しています")
            return

        logger.info("データ鮮度・整合性監視停止中...")

        self.monitor_status = MonitorStatus.INACTIVE
        self.stop_event.set()

        # 監視スレッド終了待機
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)

        logger.info("監視システム停止完了")

    def _monitor_loop(self):
        """監視ループ（バックグラウンド実行）"""
        logger.info("監視ループ開始")

        while not self.stop_event.is_set():
            try:
                # 各監視ルールの実行時間をチェック
                current_time = time.time()

                for rule_id, rule in self.monitor_rules.items():
                    if not rule.enabled:
                        continue

                    # 前回チェックからの経過時間確認
                    last_check_key = f"last_check_{rule_id}"
                    last_check_time = getattr(self, last_check_key, 0)

                    if current_time - last_check_time >= rule.check_interval_seconds:
                        # 非同期でチェック実行
                        asyncio.run_coroutine_threadsafe(
                            self._execute_rule_check(rule), asyncio.new_event_loop()
                        )
                        setattr(self, last_check_key, current_time)

                # メトリクス更新
                self._update_system_metrics()

                # アラートクリーンアップ
                self.alert_handler.cleanup_expired_alerts(self.active_alerts)

                # 短い間隔で再チェック
                self.stop_event.wait(30)  # 30秒間隔

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                self.stop_event.wait(60)  # エラー時は1分待機

        logger.info("監視ループ終了")

    async def _execute_rule_check(self, rule: MonitorRule):
        """監視ルールチェック実行"""
        try:
            logger.debug(f"監視ルールチェック実行: {rule.rule_id}")

            # データソース固有のチェック実行
            if rule.data_source != "all":
                await self._check_data_source(rule, rule.data_source)
            else:
                # 全データソースをチェック
                for source_id in self.data_source_health.keys():
                    await self._check_data_source(rule, source_id)

        except Exception as e:
            logger.error(f"ルールチェック実行エラー {rule.rule_id}: {e}")

    async def _check_data_source(self, rule: MonitorRule, data_source: str):
        """データソース個別チェック"""
        try:
            # データソースからデータ取得（模擬）
            data, context = await self._fetch_data_for_monitoring(data_source)

            if data is None:
                # データ取得失敗
                alert = MonitorAlert(
                    alert_id=f"data_missing_{data_source}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    alert_type=AlertType.DATA_MISSING,
                    severity=AlertSeverity.HIGH,
                    title="データ取得失敗",
                    message=f"データソース {data_source} からデータを取得できません",
                    data_source=data_source,
                    triggered_at=datetime.utcnow(),
                )
                await self._handle_alert(alert)
                return

            # 適切なチェック実行
            check_passed = True
            generated_alert = None

            if rule.rule_type == "freshness" and "freshness" in self.checks:
                check_passed, generated_alert = await self.checks[
                    "freshness"
                ].execute_check(data_source, data, context)
            elif rule.rule_type == "consistency" and "consistency" in self.checks:
                check_passed, generated_alert = await self.checks[
                    "consistency"
                ].execute_check(data_source, data, context)

            # アラート処理
            if not check_passed and generated_alert:
                await self._handle_alert(generated_alert)

            # データソースヘルス更新
            await self.alert_handler.update_data_source_health(
                data_source, check_passed, context, self.data_source_health
            )

        except Exception as e:
            logger.error(f"データソースチェックエラー {data_source}: {e}")

    async def _fetch_data_for_monitoring(
        self, data_source: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """監視用データ取得（模擬実装）"""
        try:
            # 実際の実装では、各データソースから最新データを取得
            # ここでは模擬データを返す

            context = {
                "data_timestamp": datetime.utcnow() - timedelta(minutes=30),
                "source_type": "api",
                "response_time_ms": 150,
            }

            if data_source == "price_data":
                data = pd.DataFrame(
                    {
                        "Open": [2500, 2520],
                        "High": [2550, 2540],
                        "Low": [2480, 2500],
                        "Close": [2530, 2485],
                        "Volume": [1500000, 1200000],
                    },
                    index=pd.date_range(
                        datetime.utcnow() - timedelta(hours=2), periods=2, freq="H"
                    ),
                )

            elif data_source == "news_data":
                data = [
                    {
                        "title": "Test News 1",
                        "timestamp": datetime.utcnow() - timedelta(hours=1),
                        "summary": "Test summary 1",
                    },
                    {
                        "title": "Test News 2",
                        "timestamp": datetime.utcnow() - timedelta(minutes=30),
                        "summary": "Test summary 2",
                    },
                ]

            else:
                data = {"test_value": 123, "timestamp": datetime.utcnow()}

            return data, context

        except Exception as e:
            logger.error(f"監視用データ取得エラー {data_source}: {e}")
            return None, {}

    async def _handle_alert(self, alert: MonitorAlert):
        """アラート処理"""
        # アクティブアラート管理
        self.active_alerts[alert.alert_id] = alert

        # アラート履歴保存
        self.alert_history.append(alert)

        # アラートハンドラーに処理を委譲
        await self.alert_handler.handle_alert(
            alert, self.monitor_rules, self.sla_metrics
        )

    def _update_system_metrics(self):
        """システムメトリクス更新"""
        try:
            current_time = datetime.utcnow()

            # アクティブアラート数
            self.metrics_history["active_alerts_count"].append(
                {"timestamp": current_time, "value": len(self.active_alerts)}
            )

            # データソースヘルス統計
            healthy_sources = sum(
                1
                for health in self.data_source_health.values()
                if health.health_status == "healthy"
            )
            total_sources = len(self.data_source_health)

            if total_sources > 0:
                health_percentage = healthy_sources / total_sources * 100
                self.metrics_history["system_health_percentage"].append(
                    {"timestamp": current_time, "value": health_percentage}
                )

            # 平均品質スコア
            if self.data_source_health:
                avg_quality = np.mean(
                    [
                        health.quality_score
                        for health in self.data_source_health.values()
                    ]
                )
                self.metrics_history["avg_quality_score"].append(
                    {"timestamp": current_time, "value": avg_quality}
                )

        except Exception as e:
            logger.error(f"システムメトリクス更新エラー: {e}")

    def add_alert_callback(self, callback):
        """アラートコールバック追加"""
        self.alert_handler.add_alert_callback(callback)

    def add_monitor_rule(self, rule: MonitorRule):
        """監視ルール追加"""
        self.monitor_rules[rule.rule_id] = rule
        logger.info(f"監視ルール追加: {rule.rule_id}")

    def remove_monitor_rule(self, rule_id: str):
        """監視ルール削除"""
        if rule_id in self.monitor_rules:
            del self.monitor_rules[rule_id]
            logger.info(f"監視ルール削除: {rule_id}")

    def get_system_dashboard(self) -> Dict[str, Any]:
        """システムダッシュボード情報取得"""
        try:
            current_time = datetime.utcnow()

            # アラート統計
            alert_stats = {
                "total_active": len(self.active_alerts),
                "by_severity": defaultdict(int),
                "by_type": defaultdict(int),
            }

            for alert in self.active_alerts.values():
                alert_stats["by_severity"][alert.severity.value] += 1
                alert_stats["by_type"][alert.alert_type.value] += 1

            # データソースヘルス統計
            health_stats = {
                "total_sources": len(self.data_source_health),
                "by_status": defaultdict(int),
                "avg_quality_score": 0.0,
                "avg_availability": 0.0,
            }

            if self.data_source_health:
                for health in self.data_source_health.values():
                    health_stats["by_status"][health.health_status] += 1

                health_stats["avg_quality_score"] = np.mean(
                    [
                        health.quality_score
                        for health in self.data_source_health.values()
                    ]
                )
                health_stats["avg_availability"] = np.mean(
                    [health.availability for health in self.data_source_health.values()]
                )

            # SLA統計
            sla_stats = {}
            for sla_id, sla in self.sla_metrics.items():
                sla_stats[sla_id] = {
                    "name": sla.name,
                    "availability": sla.current_availability,
                    "freshness_minutes": sla.current_freshness_minutes,
                    "quality_score": sla.current_quality_score,
                    "violations_count": sla.violations_count,
                    "target_availability": sla.target_availability,
                    "target_freshness": sla.target_freshness_minutes,
                    "target_quality": sla.target_quality_score,
                }

            # 最近のメトリクス
            recent_metrics = {}
            for metric_name, metric_history in self.metrics_history.items():
                if metric_history:
                    latest_metric = metric_history[-1]
                    recent_metrics[metric_name] = latest_metric["value"]

            return {
                "generated_at": current_time.isoformat(),
                "monitor_status": self.monitor_status.value,
                "uptime_minutes": (
                    current_time - getattr(self, "_start_time", current_time)
                ).total_seconds()
                / 60,
                "alert_statistics": dict(alert_stats),
                "health_statistics": dict(health_stats),
                "sla_metrics": sla_stats,
                "recent_metrics": recent_metrics,
                "system_configuration": {
                    "check_interval_seconds": self.check_interval_seconds,
                    "alert_retention_days": self.alert_retention_days,
                    "total_rules": len(self.monitor_rules),
                    "active_rules": sum(
                        1 for rule in self.monitor_rules.values() if rule.enabled
                    ),
                },
            }

        except Exception as e:
            logger.error(f"システムダッシュボード情報取得エラー: {e}")
            return {"error": str(e)}

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """アラート確認"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by

            logger.info(f"アラート確認: {alert_id} by {acknowledged_by}")
        else:
            logger.warning(f"アラートが見つかりません: {alert_id}")

    async def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """アラート解決"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow()

            # アクティブアラートから削除
            del self.active_alerts[alert_id]

            logger.info(f"アラート解決: {alert_id} by {resolved_by}")
        else:
            logger.warning(f"アラートが見つかりません: {alert_id}")

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("データ鮮度・整合性監視システム クリーンアップ開始")

        # 監視停止
        await self.stop_monitoring()

        # メトリクス履歴クリア
        self.metrics_history.clear()

        # キャッシュクリア
        if self.cache_manager:
            # 具体的なクリア処理は実装に依存
            pass

        logger.info("データ鮮度・整合性監視システム クリーンアップ完了")