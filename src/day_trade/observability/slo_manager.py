"""
APM・オブザーバビリティ統合基盤 - Issue #442 Phase 4
SLO/SLI自動監視・エラーバジェット管理・品質ゲート連携

このモジュールは以下を提供します:
- SLO/SLI定義・計算・監視
- エラーバジェット管理
- 自動アラート生成
- CI/CD品質ゲート連携
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .metrics_collector import get_metrics_collector
from .structured_logger import get_structured_logger


class SLOStatus(Enum):
    """SLO状態"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SLODefinition:
    """SLO定義"""

    name: str
    description: str
    service: str

    # SLI定義
    sli_query: str  # Prometheus query
    sli_description: str

    # SLO目標
    target_percentage: float  # 99.9, 99.99 etc.
    time_window_hours: int = 24  # 評価期間

    # アラート設定
    warning_threshold: float = 0.5  # 警告しきい値（エラーバジェット消費率）
    critical_threshold: float = 0.8  # クリティカルしきい値

    # メタデータ
    tags: Dict[str, str] = field(default_factory=dict)
    runbook_url: str = ""
    owner: str = ""

    # 計算設定
    evaluation_interval_seconds: int = 60
    min_sample_size: int = 100


@dataclass
class SLIDataPoint:
    """SLIデータポイント"""

    timestamp: float
    value: float
    success: bool
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLOReport:
    """SLO評価レポート"""

    slo_name: str
    timestamp: float
    time_window_start: float
    time_window_end: float

    # SLI統計
    sli_current: float
    sli_target: float
    total_samples: int
    successful_samples: int

    # エラーバジェット
    error_budget_total: float
    error_budget_consumed: float
    error_budget_remaining: float
    error_budget_consumption_rate: float

    # 状態
    status: SLOStatus
    alert_severity: Optional[AlertSeverity] = None

    # 予測
    projected_exhaustion_time: Optional[float] = None
    recommended_actions: List[str] = field(default_factory=list)


class SLOManager:
    """
    SLO/SLI管理クラス

    Features:
    - リアルタイムSLO/SLI計算
    - エラーバジェット自動管理
    - 動的アラート生成
    - 品質ゲート自動判定
    - 予測分析
    """

    def __init__(self):
        self.slo_definitions: Dict[str, SLODefinition] = {}
        self.sli_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100000))
        self.slo_reports: Dict[str, SLOReport] = {}

        self.metrics_collector = get_metrics_collector()
        self.logger = get_structured_logger()

        # 評価ループ制御
        self._evaluation_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

        # アラートコールバック
        self._alert_callbacks: List[Callable[[SLOReport], None]] = []

        # データ保持期間（秒）
        self.data_retention_seconds = 7 * 24 * 3600  # 7日間

        # 標準SLO定義を登録
        self._register_standard_slos()

    def _register_standard_slos(self) -> None:
        """標準SLO定義の登録"""

        # API レスポンス時間 SLO
        self.register_slo(
            SLODefinition(
                name="api_latency_slo",
                description="API response time should be under 50ms for 99.9% of requests",
                service="api",
                sli_query="histogram_quantile(0.999, rate(day_trade_request_duration_seconds_bucket[5m])) * 1000",
                sli_description="99.9th percentile API response time in milliseconds",
                target_percentage=99.9,
                time_window_hours=1,
                warning_threshold=0.5,
                critical_threshold=0.8,
                tags={"category": "performance", "priority": "high"},
                runbook_url="https://docs.daytrade.local/runbooks/api-latency",
                owner="platform-team",
                evaluation_interval_seconds=30,
            )
        )

        # 取引実行レイテンシ SLO
        self.register_slo(
            SLODefinition(
                name="trade_latency_slo",
                description="Trade execution latency should be under 50μs for 99.9% of trades",
                service="trading",
                sli_query="histogram_quantile(0.999, rate(day_trade_trade_latency_microseconds_bucket[30s]))",
                sli_description="99.9th percentile trade execution latency in microseconds",
                target_percentage=99.9,
                time_window_hours=1,
                warning_threshold=0.3,
                critical_threshold=0.5,
                tags={"category": "hft", "priority": "critical"},
                runbook_url="https://docs.daytrade.local/runbooks/trade-latency",
                owner="trading-team",
                evaluation_interval_seconds=10,
            )
        )

        # システム可用性 SLO
        self.register_slo(
            SLODefinition(
                name="system_availability_slo",
                description="System should be available 99.99% of the time",
                service="system",
                sli_query='avg_over_time(up{job="day-trade-app"}[5m]) * 100',
                sli_description="System availability percentage",
                target_percentage=99.99,
                time_window_hours=24,
                warning_threshold=0.5,
                critical_threshold=0.8,
                tags={"category": "availability", "priority": "critical"},
                runbook_url="https://docs.daytrade.local/runbooks/availability",
                owner="sre-team",
                evaluation_interval_seconds=60,
            )
        )

        # 取引成功率 SLO
        self.register_slo(
            SLODefinition(
                name="trade_success_slo",
                description="Trade success rate should be above 99.95%",
                service="trading",
                sli_query='(rate(day_trade_trades_total{status="success"}[5m]) / rate(day_trade_trades_total[5m])) * 100',
                sli_description="Trade success rate percentage",
                target_percentage=99.95,
                time_window_hours=4,
                warning_threshold=0.4,
                critical_threshold=0.7,
                tags={"category": "business", "priority": "high"},
                runbook_url="https://docs.daytrade.local/runbooks/trade-success",
                owner="trading-team",
                evaluation_interval_seconds=30,
            )
        )

        # エラー率 SLO
        self.register_slo(
            SLODefinition(
                name="error_rate_slo",
                description="Error rate should be below 0.1%",
                service="application",
                sli_query="(rate(day_trade_errors_total[5m]) / rate(day_trade_requests_total[5m])) * 100",
                sli_description="Error rate percentage",
                target_percentage=99.9,  # 0.1%エラー率 = 99.9%成功率
                time_window_hours=2,
                warning_threshold=0.6,
                critical_threshold=0.8,
                tags={"category": "reliability", "priority": "medium"},
                runbook_url="https://docs.daytrade.local/runbooks/error-rate",
                owner="platform-team",
                evaluation_interval_seconds=60,
            )
        )

    # === SLO管理 ===

    def register_slo(self, slo_definition: SLODefinition) -> None:
        """SLO定義登録"""
        self.slo_definitions[slo_definition.name] = slo_definition

        self.logger.info(
            "SLO registered",
            slo_name=slo_definition.name,
            service=slo_definition.service,
            target=slo_definition.target_percentage,
        )

    def unregister_slo(self, slo_name: str) -> None:
        """SLO定義削除"""
        if slo_name in self.slo_definitions:
            del self.slo_definitions[slo_name]

            # 評価タスクも停止
            if slo_name in self._evaluation_tasks:
                self._evaluation_tasks[slo_name].cancel()
                del self._evaluation_tasks[slo_name]

            self.logger.info("SLO unregistered", slo_name=slo_name)

    # === SLI データ記録 ===

    def record_sli_data(
        self, slo_name: str, value: float, success: bool, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """SLI データポイント記録"""
        if slo_name not in self.slo_definitions:
            return

        datapoint = SLIDataPoint(
            timestamp=time.time(), value=value, success=success, tags=tags or {}
        )

        self.sli_data[slo_name].append(datapoint)

        # メトリクス記録
        self.metrics_collector.record_sli_event(slo_name, success, value)

    # === SLO評価・計算 ===

    def calculate_slo(self, slo_name: str) -> Optional[SLOReport]:
        """SLO評価・計算"""
        if slo_name not in self.slo_definitions:
            return None

        slo_def = self.slo_definitions[slo_name]
        current_time = time.time()
        window_start = current_time - (slo_def.time_window_hours * 3600)

        # 時間窓内のデータ取得
        recent_data = [dp for dp in self.sli_data[slo_name] if dp.timestamp >= window_start]

        if len(recent_data) < slo_def.min_sample_size:
            self.logger.debug(
                "Insufficient data for SLO calculation",
                slo_name=slo_name,
                sample_size=len(recent_data),
                required=slo_def.min_sample_size,
            )
            return None

        # SLI統計計算
        total_samples = len(recent_data)
        successful_samples = sum(1 for dp in recent_data if dp.success)
        sli_current = (successful_samples / total_samples) * 100 if total_samples > 0 else 0

        # エラーバジェット計算
        error_budget_total = (100 - slo_def.target_percentage) * total_samples / 100
        error_failures = total_samples - successful_samples
        error_budget_consumed = error_failures
        error_budget_remaining = max(0, error_budget_total - error_budget_consumed)
        error_budget_consumption_rate = (
            error_budget_consumed / error_budget_total if error_budget_total > 0 else 0
        )

        # ステータス判定
        status = self._determine_slo_status(
            sli_current, slo_def.target_percentage, error_budget_consumption_rate, slo_def
        )
        alert_severity = self._determine_alert_severity(
            status, error_budget_consumption_rate, slo_def
        )

        # 将来予測
        projected_exhaustion_time = self._predict_budget_exhaustion(
            recent_data, error_budget_remaining, slo_def
        )

        # 推奨アクション
        recommended_actions = self._generate_recommendations(
            status, error_budget_consumption_rate, sli_current, slo_def
        )

        # レポート作成
        report = SLOReport(
            slo_name=slo_name,
            timestamp=current_time,
            time_window_start=window_start,
            time_window_end=current_time,
            sli_current=sli_current,
            sli_target=slo_def.target_percentage,
            total_samples=total_samples,
            successful_samples=successful_samples,
            error_budget_total=error_budget_total,
            error_budget_consumed=error_budget_consumed,
            error_budget_remaining=error_budget_remaining,
            error_budget_consumption_rate=error_budget_consumption_rate,
            status=status,
            alert_severity=alert_severity,
            projected_exhaustion_time=projected_exhaustion_time,
            recommended_actions=recommended_actions,
        )

        # レポート保存
        self.slo_reports[slo_name] = report

        # アラート処理
        if alert_severity:
            self._trigger_alert(report)

        return report

    def _determine_slo_status(
        self,
        sli_current: float,
        sli_target: float,
        budget_consumption_rate: float,
        slo_def: SLODefinition,
    ) -> SLOStatus:
        """SLO状態判定"""

        if sli_current < sli_target:
            return SLOStatus.BREACH

        if budget_consumption_rate >= slo_def.critical_threshold:
            return SLOStatus.CRITICAL
        elif budget_consumption_rate >= slo_def.warning_threshold:
            return SLOStatus.WARNING
        else:
            return SLOStatus.HEALTHY

    def _determine_alert_severity(
        self, status: SLOStatus, budget_consumption_rate: float, slo_def: SLODefinition
    ) -> Optional[AlertSeverity]:
        """アラート重要度判定"""

        if status == SLOStatus.BREACH:
            return AlertSeverity.CRITICAL
        elif status == SLOStatus.CRITICAL:
            return AlertSeverity.HIGH
        elif status == SLOStatus.WARNING:
            return AlertSeverity.MEDIUM

        return None

    def _predict_budget_exhaustion(
        self, recent_data: List[SLIDataPoint], remaining_budget: float, slo_def: SLODefinition
    ) -> Optional[float]:
        """エラーバジェット枯渇時刻予測"""

        if remaining_budget <= 0:
            return time.time()  # 既に枯渇

        if len(recent_data) < 10:
            return None  # 予測に十分なデータがない

        # 最近1時間のエラー率トレンド
        hour_ago = time.time() - 3600
        recent_hour_data = [dp for dp in recent_data if dp.timestamp >= hour_ago]

        if len(recent_hour_data) < 5:
            return None

        error_rate = (
            len(recent_hour_data) - sum(1 for dp in recent_hour_data if dp.success)
        ) / len(recent_hour_data)

        if error_rate <= 0:
            return None  # エラーが発生していない

        # 現在のエラー率で継続した場合の枯渇予測
        expected_errors_per_hour = error_rate * len(recent_hour_data)
        hours_to_exhaustion = (
            remaining_budget / expected_errors_per_hour if expected_errors_per_hour > 0 else None
        )

        if hours_to_exhaustion and hours_to_exhaustion > 0:
            return time.time() + (hours_to_exhaustion * 3600)

        return None

    def _generate_recommendations(
        self,
        status: SLOStatus,
        budget_consumption_rate: float,
        sli_current: float,
        slo_def: SLODefinition,
    ) -> List[str]:
        """推奨アクション生成"""

        recommendations = []

        if status == SLOStatus.BREACH:
            recommendations.extend(
                [
                    "🚨 IMMEDIATE ACTION REQUIRED: SLO is currently breached",
                    "Consider rolling back recent changes",
                    "Investigate root cause immediately",
                    "Implement incident response procedures",
                ]
            )
        elif status == SLOStatus.CRITICAL:
            recommendations.extend(
                [
                    "⚠️ ERROR BUDGET CRITICAL: Immediate attention needed",
                    "Review recent deployments and changes",
                    "Consider pausing new deployments",
                    "Investigate performance degradation",
                ]
            )
        elif status == SLOStatus.WARNING:
            recommendations.extend(
                [
                    "⚠️ ERROR BUDGET WARNING: Monitor closely",
                    "Review system performance trends",
                    "Plan capacity scaling if needed",
                ]
            )

        # サービス固有の推奨事項
        if slo_def.service == "trading" and "latency" in slo_def.name:
            recommendations.extend(
                [
                    "Check HFT system performance",
                    "Verify network latency to exchanges",
                    "Review algorithm efficiency",
                ]
            )
        elif slo_def.service == "api" and "latency" in slo_def.name:
            recommendations.extend(
                [
                    "Check database query performance",
                    "Review cache hit rates",
                    "Consider API rate limiting",
                ]
            )

        return recommendations

    # === アラート管理 ===

    def register_alert_callback(self, callback: Callable[[SLOReport], None]) -> None:
        """アラートコールバック登録"""
        self._alert_callbacks.append(callback)

    def _trigger_alert(self, report: SLOReport) -> None:
        """アラートトリガー"""

        # ログ記録
        self.logger.error(
            "SLO alert triggered",
            slo_name=report.slo_name,
            status=report.status.value,
            severity=report.alert_severity.value if report.alert_severity else None,
            sli_current=report.sli_current,
            sli_target=report.sli_target,
            budget_consumption_rate=report.error_budget_consumption_rate,
            recommended_actions=report.recommended_actions,
        )

        # メトリクス記録
        self.metrics_collector.increment_counter(
            "slo_alerts_total",
            attributes={
                "slo_name": report.slo_name,
                "status": report.status.value,
                "severity": report.alert_severity.value if report.alert_severity else "none",
            },
        )

        # コールバック実行
        for callback in self._alert_callbacks:
            try:
                callback(report)
            except Exception as e:
                self.logger.error("Alert callback failed", error=e, slo_name=report.slo_name)

    # === 品質ゲート ===

    def evaluate_quality_gate(
        self, deployment_context: Dict[str, Any], slo_names: Optional[List[str]] = None
    ) -> Tuple[bool, List[str], Dict[str, SLOReport]]:
        """
        品質ゲート評価

        Returns:
            (is_passing, failure_reasons, slo_reports)
        """

        target_slos = slo_names or list(self.slo_definitions.keys())
        failure_reasons = []
        reports = {}

        for slo_name in target_slos:
            report = self.calculate_slo(slo_name)
            if not report:
                failure_reasons.append(f"SLO {slo_name}: Insufficient data for evaluation")
                continue

            reports[slo_name] = report

            # 品質ゲート判定
            if report.status in [SLOStatus.BREACH, SLOStatus.CRITICAL]:
                failure_reasons.append(
                    f"SLO {slo_name}: {report.status.value} "
                    f"(SLI: {report.sli_current:.2f}%, Target: {report.sli_target:.2f}%)"
                )

            # エラーバジェット消費率での判定
            if report.error_budget_consumption_rate > 0.9:
                failure_reasons.append(
                    f"SLO {slo_name}: Error budget nearly exhausted "
                    f"({report.error_budget_consumption_rate*100:.1f}%)"
                )

        is_passing = len(failure_reasons) == 0

        # 品質ゲートログ
        self.logger.info(
            "Quality gate evaluation",
            deployment_context=deployment_context,
            is_passing=is_passing,
            evaluated_slos=len(reports),
            failure_reasons=failure_reasons,
        )

        return is_passing, failure_reasons, reports

    # === 自動評価ループ ===

    async def start_automatic_evaluation(self) -> None:
        """自動評価ループ開始"""

        for slo_name, slo_def in self.slo_definitions.items():
            task = asyncio.create_task(
                self._evaluation_loop(slo_name, slo_def.evaluation_interval_seconds)
            )
            self._evaluation_tasks[slo_name] = task

        self.logger.info("Automatic SLO evaluation started", slo_count=len(self._evaluation_tasks))

    async def stop_automatic_evaluation(self) -> None:
        """自動評価ループ停止"""

        self._shutdown_event.set()

        # 全タスクの完了を待機
        if self._evaluation_tasks:
            await asyncio.gather(*self._evaluation_tasks.values(), return_exceptions=True)

        self._evaluation_tasks.clear()

        self.logger.info("Automatic SLO evaluation stopped")

    async def _evaluation_loop(self, slo_name: str, interval_seconds: int) -> None:
        """SLO評価ループ"""

        while not self._shutdown_event.is_set():
            try:
                # SLO計算実行
                report = self.calculate_slo(slo_name)

                if report:
                    # メトリクス更新
                    self.metrics_collector.set_gauge(
                        "slo_current_sli", report.sli_current, attributes={"slo_name": slo_name}
                    )

                    self.metrics_collector.set_gauge(
                        "slo_error_budget_remaining",
                        report.error_budget_remaining,
                        attributes={"slo_name": slo_name},
                    )

                # インターバル待機
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("SLO evaluation loop error", error=e, slo_name=slo_name)
                await asyncio.sleep(min(interval_seconds, 60))  # エラー時は最大1分待機

    # === レポート・エクスポート ===

    def get_slo_summary(self) -> Dict[str, Any]:
        """SLO概要取得"""

        summary = {"timestamp": time.time(), "total_slos": len(self.slo_definitions), "slos": {}}

        status_counts = defaultdict(int)

        for slo_name, report in self.slo_reports.items():
            summary["slos"][slo_name] = {
                "status": report.status.value,
                "sli_current": report.sli_current,
                "sli_target": report.sli_target,
                "error_budget_consumption_rate": report.error_budget_consumption_rate,
                "alert_severity": report.alert_severity.value if report.alert_severity else None,
            }

            status_counts[report.status.value] += 1

        summary["status_distribution"] = dict(status_counts)

        return summary

    def export_slo_config(self, filepath: str) -> None:
        """SLO設定エクスポート"""

        config = {"version": "1.0", "timestamp": datetime.now(timezone.utc).isoformat(), "slos": []}

        for slo_name, slo_def in self.slo_definitions.items():
            config["slos"].append(
                {
                    "name": slo_def.name,
                    "description": slo_def.description,
                    "service": slo_def.service,
                    "sli_query": slo_def.sli_query,
                    "sli_description": slo_def.sli_description,
                    "target_percentage": slo_def.target_percentage,
                    "time_window_hours": slo_def.time_window_hours,
                    "warning_threshold": slo_def.warning_threshold,
                    "critical_threshold": slo_def.critical_threshold,
                    "tags": slo_def.tags,
                    "runbook_url": slo_def.runbook_url,
                    "owner": slo_def.owner,
                }
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


# グローバルSLOマネージャー
_global_slo_manager: Optional[SLOManager] = None


def get_slo_manager() -> SLOManager:
    """グローバルSLOマネージャー取得"""
    global _global_slo_manager

    if _global_slo_manager is None:
        _global_slo_manager = SLOManager()

    return _global_slo_manager


# 便利関数
def record_sli(slo_name: str, value: float, success: bool, **tags):
    """SLI記録便利関数"""
    manager = get_slo_manager()
    manager.record_sli_data(slo_name, value, success, tags)


def check_quality_gate(deployment_context: Dict[str, Any]) -> bool:
    """品質ゲートチェック便利関数"""
    manager = get_slo_manager()
    is_passing, _, _ = manager.evaluate_quality_gate(deployment_context)
    return is_passing
