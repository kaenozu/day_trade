#!/usr/bin/env python3
"""
パフォーマンスアラートシステム
Issue #318: 監視・アラートシステム - Phase 2

投資パフォーマンス監視・アラート生成・しきい値管理
- 予測精度低下アラート
- 収益率異常検知
- リスク指標監視
- パフォーマンス自動レポート
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PerformanceMetricType(Enum):
    """パフォーマンス指標タイプ"""

    PREDICTION_ACCURACY = "prediction_accuracy"
    RETURN_RATE = "return_rate"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_LOSS_RATIO = "profit_loss_ratio"
    VOLATILITY = "volatility"
    BETA = "beta"
    ALPHA = "alpha"
    VALUE_AT_RISK = "value_at_risk"


class AlertConditionType(Enum):
    """アラート条件タイプ"""

    THRESHOLD_BELOW = "threshold_below"
    THRESHOLD_ABOVE = "threshold_above"
    PERCENTAGE_CHANGE = "percentage_change"
    MOVING_AVERAGE_DEVIATION = "moving_average_deviation"
    TREND_REVERSAL = "trend_reversal"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class PerformanceThreshold:
    """パフォーマンスしきい値設定"""

    metric_type: PerformanceMetricType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    condition_type: AlertConditionType
    evaluation_period_minutes: int = 60
    min_data_points: int = 10
    enabled: bool = True


@dataclass
class PerformanceAlert:
    """パフォーマンスアラート"""

    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_type: PerformanceMetricType
    current_value: float
    threshold_value: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceMetric:
    """パフォーマンス指標データ"""

    timestamp: datetime
    metric_type: PerformanceMetricType
    value: float
    symbol: Optional[str] = None
    period: Optional[str] = None
    confidence: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """アラート設定"""

    # 基本アラート設定
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    enable_database_logging: bool = True
    alert_history_retention_days: int = 30

    # パフォーマンス評価設定
    evaluation_interval_minutes: int = 15
    anomaly_detection_sensitivity: float = 2.0  # 標準偏差倍数
    trend_analysis_window: int = 20
    moving_average_window: int = 10

    # アラート抑制設定
    duplicate_alert_cooldown_minutes: int = 30
    escalation_threshold_minutes: int = 60
    max_alerts_per_hour: int = 20


class PerformanceAlertSystem:
    """パフォーマンスアラートシステム"""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.performance_history: List[PerformanceMetric] = []
        self.alerts: List[PerformanceAlert] = []
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            severity: [] for severity in AlertSeverity
        }

        # 統計分析用
        self.metric_statistics: Dict[PerformanceMetricType, Dict[str, float]] = {}
        self.anomaly_baselines: Dict[PerformanceMetricType, Dict[str, float]] = {}

        # アラート抑制管理
        self.recent_alerts: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, int] = {}

        # 監視タスク
        self.monitor_task: Optional[asyncio.Task] = None
        self._is_running = False

        # デフォルトしきい値設定
        self._setup_default_thresholds()

    def _setup_default_thresholds(self) -> None:
        """デフォルトしきい値設定"""
        default_thresholds = [
            # 予測精度アラート
            PerformanceThreshold(
                PerformanceMetricType.PREDICTION_ACCURACY,
                warning_threshold=0.6,
                critical_threshold=0.5,
                emergency_threshold=0.4,
                condition_type=AlertConditionType.THRESHOLD_BELOW,
                evaluation_period_minutes=30,
            ),
            # 収益率アラート
            PerformanceThreshold(
                PerformanceMetricType.RETURN_RATE,
                warning_threshold=-5.0,  # -5%
                critical_threshold=-10.0,  # -10%
                emergency_threshold=-20.0,  # -20%
                condition_type=AlertConditionType.THRESHOLD_BELOW,
                evaluation_period_minutes=60,
            ),
            # シャープレシオアラート
            PerformanceThreshold(
                PerformanceMetricType.SHARPE_RATIO,
                warning_threshold=0.5,
                critical_threshold=0.2,
                emergency_threshold=0.0,
                condition_type=AlertConditionType.THRESHOLD_BELOW,
                evaluation_period_minutes=120,
            ),
            # 最大ドローダウンアラート
            PerformanceThreshold(
                PerformanceMetricType.MAX_DRAWDOWN,
                warning_threshold=-15.0,  # -15%
                critical_threshold=-25.0,  # -25%
                emergency_threshold=-40.0,  # -40%
                condition_type=AlertConditionType.THRESHOLD_BELOW,
                evaluation_period_minutes=60,
            ),
            # ボラティリティアラート
            PerformanceThreshold(
                PerformanceMetricType.VOLATILITY,
                warning_threshold=0.3,  # 30%
                critical_threshold=0.5,  # 50%
                emergency_threshold=0.8,  # 80%
                condition_type=AlertConditionType.THRESHOLD_ABOVE,
                evaluation_period_minutes=60,
            ),
            # VaRアラート
            PerformanceThreshold(
                PerformanceMetricType.VALUE_AT_RISK,
                warning_threshold=-5.0,  # -5%
                critical_threshold=-10.0,  # -10%
                emergency_threshold=-15.0,  # -15%
                condition_type=AlertConditionType.THRESHOLD_BELOW,
                evaluation_period_minutes=30,
            ),
        ]

        for threshold in default_thresholds:
            self.add_threshold(threshold)

    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """しきい値追加"""
        threshold_key = (
            f"{threshold.metric_type.value}_{threshold.condition_type.value}"
        )
        self.thresholds[threshold_key] = threshold
        logger.info(f"パフォーマンスしきい値追加: {threshold_key}")

    def remove_threshold(
        self, metric_type: PerformanceMetricType, condition_type: AlertConditionType
    ) -> None:
        """しきい値削除"""
        threshold_key = f"{metric_type.value}_{condition_type.value}"
        if threshold_key in self.thresholds:
            del self.thresholds[threshold_key]
            logger.info(f"パフォーマンスしきい値削除: {threshold_key}")

    def register_alert_handler(
        self, severity: AlertSeverity, handler: Callable[[PerformanceAlert], None]
    ) -> None:
        """アラートハンドラー登録"""
        self.alert_handlers[severity].append(handler)
        logger.info(f"アラートハンドラー登録: {severity.value}")

    async def start_monitoring(self) -> None:
        """監視開始"""
        if self._is_running:
            logger.warning("パフォーマンス監視は既に実行中です")
            return

        self._is_running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("パフォーマンスアラート監視開始")

    async def stop_monitoring(self) -> None:
        """監視停止"""
        self._is_running = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("パフォーマンスアラート監視停止")

    async def _monitoring_loop(self) -> None:
        """監視ループ"""
        while self._is_running:
            try:
                # パフォーマンス評価実行
                await self._evaluate_performance_metrics()

                # アラート履歴クリーンアップ
                await self._cleanup_alert_history()

                # 統計情報更新
                await self._update_metric_statistics()

                # 異常検知ベースライン更新
                await self._update_anomaly_baselines()

                # 評価間隔待機
                await asyncio.sleep(self.config.evaluation_interval_minutes * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"パフォーマンス監視ループエラー: {e}")
                await asyncio.sleep(60)

    async def _evaluate_performance_metrics(self) -> None:
        """パフォーマンス指標評価"""
        current_time = datetime.now()

        for threshold_key, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue

            try:
                # 評価期間のデータ取得
                evaluation_period_start = current_time - timedelta(
                    minutes=threshold.evaluation_period_minutes
                )
                relevant_metrics = [
                    m
                    for m in self.performance_history
                    if (
                        m.metric_type == threshold.metric_type
                        and m.timestamp >= evaluation_period_start
                    )
                ]

                if len(relevant_metrics) < threshold.min_data_points:
                    continue

                # 指標値計算
                current_value = await self._calculate_current_metric_value(
                    threshold.metric_type, relevant_metrics, threshold.condition_type
                )

                if current_value is None:
                    continue

                # しきい値判定
                alert_severity = await self._evaluate_threshold(
                    threshold, current_value
                )

                if alert_severity:
                    await self._generate_alert(threshold, current_value, alert_severity)

            except Exception as e:
                logger.error(f"パフォーマンス指標評価エラー {threshold_key}: {e}")

    async def _calculate_current_metric_value(
        self,
        metric_type: PerformanceMetricType,
        metrics: List[PerformanceMetric],
        condition_type: AlertConditionType,
    ) -> Optional[float]:
        """現在の指標値計算"""
        if not metrics:
            return None

        values = [m.value for m in metrics]

        if condition_type == AlertConditionType.MOVING_AVERAGE_DEVIATION:
            # 移動平均からの偏差
            if len(values) < self.config.moving_average_window:
                return None

            moving_avg = mean(values[-self.config.moving_average_window :])
            current_value = values[-1]
            deviation = (current_value - moving_avg) / moving_avg * 100
            return deviation

        elif condition_type == AlertConditionType.PERCENTAGE_CHANGE:
            # パーセント変化
            if len(values) < 2:
                return None

            previous_value = values[-2]
            current_value = values[-1]
            if previous_value == 0:
                return 0

            change = (current_value - previous_value) / previous_value * 100
            return change

        elif condition_type == AlertConditionType.TREND_REVERSAL:
            # トレンド反転検出
            if len(values) < self.config.trend_analysis_window:
                return None

            recent_values = values[-self.config.trend_analysis_window :]

            # 線形回帰による傾き計算
            x = np.arange(len(recent_values))
            y = np.array(recent_values)
            slope = np.corrcoef(x, y)[0, 1]

            return slope

        elif condition_type == AlertConditionType.ANOMALY_DETECTION:
            # 異常検知
            if len(values) < 10:
                return None

            # Z-スコア計算
            mean_val = mean(values)
            std_val = stdev(values) if len(values) > 1 else 0

            if std_val == 0:
                return 0

            current_value = values[-1]
            z_score = abs(current_value - mean_val) / std_val

            return z_score

        else:
            # 通常のしきい値判定（最新値）
            return values[-1]

    async def _evaluate_threshold(
        self, threshold: PerformanceThreshold, current_value: float
    ) -> Optional[AlertSeverity]:
        """しきい値評価"""

        if threshold.condition_type in [AlertConditionType.THRESHOLD_BELOW]:
            # 下限しきい値
            if current_value <= threshold.emergency_threshold:
                return AlertSeverity.EMERGENCY
            elif current_value <= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif current_value <= threshold.warning_threshold:
                return AlertSeverity.WARNING

        elif threshold.condition_type in [AlertConditionType.THRESHOLD_ABOVE]:
            # 上限しきい値
            if current_value >= threshold.emergency_threshold:
                return AlertSeverity.EMERGENCY
            elif current_value >= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif current_value >= threshold.warning_threshold:
                return AlertSeverity.WARNING

        elif threshold.condition_type == AlertConditionType.ANOMALY_DETECTION:
            # 異常検知（Z-スコア基準）
            if current_value >= self.config.anomaly_detection_sensitivity * 2:
                return AlertSeverity.CRITICAL
            elif current_value >= self.config.anomaly_detection_sensitivity:
                return AlertSeverity.WARNING

        elif threshold.condition_type in [
            AlertConditionType.PERCENTAGE_CHANGE,
            AlertConditionType.MOVING_AVERAGE_DEVIATION,
        ]:
            # 変化率・偏差判定
            abs_value = abs(current_value)
            if abs_value >= abs(threshold.emergency_threshold):
                return AlertSeverity.EMERGENCY
            elif abs_value >= abs(threshold.critical_threshold):
                return AlertSeverity.CRITICAL
            elif abs_value >= abs(threshold.warning_threshold):
                return AlertSeverity.WARNING

        return None

    async def _generate_alert(
        self,
        threshold: PerformanceThreshold,
        current_value: float,
        severity: AlertSeverity,
    ) -> None:
        """アラート生成"""
        alert_key = f"{threshold.metric_type.value}_{severity.value}"

        # アラート抑制チェック
        if not await self._should_generate_alert(alert_key):
            return

        # アラート生成
        alert_id = f"perf_alert_{int(time.time() * 1000)}"
        threshold_value = self._get_threshold_value(threshold, severity)

        message = await self._generate_alert_message(
            threshold.metric_type,
            current_value,
            threshold_value,
            severity,
            threshold.condition_type,
        )

        alert = PerformanceAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            metric_type=threshold.metric_type,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            details={
                "condition_type": threshold.condition_type.value,
                "evaluation_period_minutes": threshold.evaluation_period_minutes,
                "threshold_config": {
                    "warning": threshold.warning_threshold,
                    "critical": threshold.critical_threshold,
                    "emergency": threshold.emergency_threshold,
                },
            },
        )

        self.alerts.append(alert)

        # アラートハンドラー実行
        await self._execute_alert_handlers(alert)

        # アラート抑制記録
        self.recent_alerts[alert_key] = datetime.now()
        self.alert_counts[alert_key] = self.alert_counts.get(alert_key, 0) + 1

        logger.warning(f"パフォーマンスアラート生成: {alert.message}")

    def _get_threshold_value(
        self, threshold: PerformanceThreshold, severity: AlertSeverity
    ) -> float:
        """しきい値取得"""
        if severity == AlertSeverity.WARNING:
            return threshold.warning_threshold
        elif severity == AlertSeverity.CRITICAL:
            return threshold.critical_threshold
        elif severity == AlertSeverity.EMERGENCY:
            return threshold.emergency_threshold
        return 0.0

    async def _generate_alert_message(
        self,
        metric_type: PerformanceMetricType,
        current_value: float,
        threshold_value: float,
        severity: AlertSeverity,
        condition_type: AlertConditionType,
    ) -> str:
        """アラートメッセージ生成"""

        metric_names = {
            PerformanceMetricType.PREDICTION_ACCURACY: "予測精度",
            PerformanceMetricType.RETURN_RATE: "収益率",
            PerformanceMetricType.SHARPE_RATIO: "シャープレシオ",
            PerformanceMetricType.MAX_DRAWDOWN: "最大ドローダウン",
            PerformanceMetricType.WIN_RATE: "勝率",
            PerformanceMetricType.VOLATILITY: "ボラティリティ",
            PerformanceMetricType.VALUE_AT_RISK: "VaR",
        }

        condition_descriptions = {
            AlertConditionType.THRESHOLD_BELOW: "が下限値を下回りました",
            AlertConditionType.THRESHOLD_ABOVE: "が上限値を上回りました",
            AlertConditionType.PERCENTAGE_CHANGE: "の変化率が異常です",
            AlertConditionType.MOVING_AVERAGE_DEVIATION: "が移動平均から大きく乖離しました",
            AlertConditionType.ANOMALY_DETECTION: "で異常が検出されました",
        }

        severity_labels = {
            AlertSeverity.WARNING: "警告",
            AlertSeverity.CRITICAL: "重要",
            AlertSeverity.EMERGENCY: "緊急",
        }

        metric_name = metric_names.get(metric_type, metric_type.value)
        condition_desc = condition_descriptions.get(
            condition_type, "で異常が発生しました"
        )
        severity_label = severity_labels.get(severity, severity.value.upper())

        if condition_type in [
            AlertConditionType.PERCENTAGE_CHANGE,
            AlertConditionType.MOVING_AVERAGE_DEVIATION,
        ]:
            return f"[{severity_label}] {metric_name}{condition_desc} (現在値: {current_value:.2f}%, しきい値: {threshold_value:.2f}%)"
        else:
            return f"[{severity_label}] {metric_name}{condition_desc} (現在値: {current_value:.4f}, しきい値: {threshold_value:.4f})"

    async def _should_generate_alert(self, alert_key: str) -> bool:
        """アラート生成判定"""
        current_time = datetime.now()

        # 重複アラート抑制
        if alert_key in self.recent_alerts:
            last_alert_time = self.recent_alerts[alert_key]
            if current_time - last_alert_time < timedelta(
                minutes=self.config.duplicate_alert_cooldown_minutes
            ):
                return False

        # 時間あたりアラート数制限
        hour_key = f"{alert_key}_{current_time.hour}"
        hourly_count = self.alert_counts.get(hour_key, 0)
        return not (hourly_count >= self.config.max_alerts_per_hour)

    async def _execute_alert_handlers(self, alert: PerformanceAlert) -> None:
        """アラートハンドラー実行"""
        handlers = self.alert_handlers.get(alert.severity, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"アラートハンドラー実行エラー: {e}")

    async def _cleanup_alert_history(self) -> None:
        """アラート履歴クリーンアップ"""
        cutoff_time = datetime.now() - timedelta(
            days=self.config.alert_history_retention_days
        )

        # アラート履歴クリーンアップ
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]

        # 最近のアラート記録クリーンアップ
        cutoff_recent = datetime.now() - timedelta(
            minutes=self.config.duplicate_alert_cooldown_minutes * 2
        )
        self.recent_alerts = {
            key: timestamp
            for key, timestamp in self.recent_alerts.items()
            if timestamp > cutoff_recent
        }

    async def _update_metric_statistics(self) -> None:
        """指標統計情報更新"""
        for metric_type in PerformanceMetricType:
            recent_metrics = [
                m
                for m in self.performance_history
                if (
                    m.metric_type == metric_type
                    and m.timestamp > datetime.now() - timedelta(hours=24)
                )
            ]

            if len(recent_metrics) >= 5:
                values = [m.value for m in recent_metrics]

                self.metric_statistics[metric_type] = {
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "latest": values[-1],
                }

    async def _update_anomaly_baselines(self) -> None:
        """異常検知ベースライン更新"""
        for metric_type in PerformanceMetricType:
            # 過去1週間のデータでベースライン計算
            baseline_period = datetime.now() - timedelta(days=7)
            baseline_metrics = [
                m
                for m in self.performance_history
                if (m.metric_type == metric_type and m.timestamp > baseline_period)
            ]

            if len(baseline_metrics) >= 20:
                values = [m.value for m in baseline_metrics]

                self.anomaly_baselines[metric_type] = {
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0,
                    "percentile_95": np.percentile(values, 95),
                    "percentile_5": np.percentile(values, 5),
                    "updated_at": datetime.now(),
                }

    async def add_performance_metric(self, metric: PerformanceMetric) -> None:
        """パフォーマンス指標追加"""
        self.performance_history.append(metric)

        # 履歴サイズ制限（メモリ管理）
        max_history_size = 10000
        if len(self.performance_history) > max_history_size:
            self.performance_history = self.performance_history[-max_history_size:]

        logger.debug(
            f"パフォーマンス指標追加: {metric.metric_type.value}={metric.value}"
        )

    async def resolve_alert(
        self, alert_id: str, resolution_note: Optional[str] = None
    ) -> bool:
        """アラート解決"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                if resolution_note:
                    alert.details["resolution_note"] = resolution_note

                logger.info(f"アラート解決: {alert_id}")
                return True

        return False

    async def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[PerformanceAlert]:
        """アクティブアラート取得"""
        active_alerts = [alert for alert in self.alerts if not alert.resolved]

        if severity:
            active_alerts = [
                alert for alert in active_alerts if alert.severity == severity
            ]

        return sorted(active_alerts, key=lambda a: a.timestamp, reverse=True)

    async def get_alert_summary(self) -> Dict[str, Any]:
        """アラート概要取得"""
        active_alerts = await self.get_active_alerts()

        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(
                [alert for alert in active_alerts if alert.severity == severity]
            )

        metric_alerts = {}
        for metric_type in PerformanceMetricType:
            metric_alerts[metric_type.value] = len(
                [alert for alert in active_alerts if alert.metric_type == metric_type]
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_active_alerts": len(active_alerts),
            "severity_breakdown": severity_counts,
            "metric_breakdown": metric_alerts,
            "total_alerts_generated": len(self.alerts),
            "resolved_alerts": len([alert for alert in self.alerts if alert.resolved]),
            "monitoring_status": {
                "is_running": self._is_running,
                "active_thresholds": len(
                    [t for t in self.thresholds.values() if t.enabled]
                ),
                "total_thresholds": len(self.thresholds),
            },
        }

    async def get_performance_trends(
        self, metric_type: PerformanceMetricType, hours: int = 24
    ) -> Dict[str, Any]:
        """パフォーマンストレンド分析"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        relevant_metrics = [
            m
            for m in self.performance_history
            if m.metric_type == metric_type and m.timestamp > cutoff_time
        ]

        if not relevant_metrics:
            return {
                "metric_type": metric_type.value,
                "period_hours": hours,
                "data_points": 0,
                "trend": "insufficient_data",
            }

        values = [m.value for m in relevant_metrics]
        timestamps = [m.timestamp for m in relevant_metrics]

        # トレンド分析
        x = np.arange(len(values))
        y = np.array(values)

        if len(values) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
            trend = (
                "increasing"
                if correlation > 0.1
                else "decreasing"
                if correlation < -0.1
                else "stable"
            )
        else:
            trend = "stable"

        return {
            "metric_type": metric_type.value,
            "period_hours": hours,
            "data_points": len(values),
            "trend": trend,
            "current_value": values[-1] if values else None,
            "min_value": min(values) if values else None,
            "max_value": max(values) if values else None,
            "mean_value": mean(values) if values else None,
            "trend_correlation": correlation if len(values) > 1 else 0,
            "first_timestamp": timestamps[0].isoformat() if timestamps else None,
            "last_timestamp": timestamps[-1].isoformat() if timestamps else None,
        }


# 標準アラートハンドラー


async def log_alert_handler(alert: PerformanceAlert) -> None:
    """ログアラートハンドラー"""
    logger.warning(f"パフォーマンスアラート: {alert.message}")


def console_alert_handler(alert: PerformanceAlert) -> None:
    """コンソールアラートハンドラー"""
    print(
        f"[{alert.timestamp.strftime('%H:%M:%S')}] {alert.severity.value.upper()}: {alert.message}"
    )


async def file_alert_handler(alert: PerformanceAlert) -> None:
    """ファイルアラートハンドラー"""
    alert_file = Path("alerts") / "performance_alerts.log"
    alert_file.parent.mkdir(exist_ok=True)

    alert_data = {
        "timestamp": alert.timestamp.isoformat(),
        "severity": alert.severity.value,
        "metric_type": alert.metric_type.value,
        "current_value": alert.current_value,
        "threshold_value": alert.threshold_value,
        "message": alert.message,
        "details": alert.details,
    }

    with open(alert_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert_data, ensure_ascii=False) + "\n")


# 使用例・テスト関数


async def setup_performance_monitoring() -> PerformanceAlertSystem:
    """パフォーマンス監視セットアップ"""
    config = AlertConfig(
        evaluation_interval_minutes=10,
        anomaly_detection_sensitivity=2.5,
        duplicate_alert_cooldown_minutes=30,
    )

    alert_system = PerformanceAlertSystem(config)

    # アラートハンドラー登録
    alert_system.register_alert_handler(AlertSeverity.WARNING, console_alert_handler)
    alert_system.register_alert_handler(AlertSeverity.CRITICAL, log_alert_handler)
    alert_system.register_alert_handler(AlertSeverity.EMERGENCY, file_alert_handler)

    return alert_system


async def simulate_performance_data(alert_system: PerformanceAlertSystem) -> None:
    """パフォーマンスデータシミュレーション"""
    import random

    # 正常データ
    for i in range(20):
        await alert_system.add_performance_metric(
            PerformanceMetric(
                timestamp=datetime.now() - timedelta(minutes=i * 5),
                metric_type=PerformanceMetricType.PREDICTION_ACCURACY,
                value=0.75 + random.uniform(-0.05, 0.05),
                symbol="TOPIX",
                confidence=0.8,
            )
        )

    # 異常データ（アラート発生）
    await alert_system.add_performance_metric(
        PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=PerformanceMetricType.PREDICTION_ACCURACY,
            value=0.45,  # 警告しきい値以下
            symbol="TOPIX",
            confidence=0.6,
        )
    )


if __name__ == "__main__":

    async def main():
        alert_system = await setup_performance_monitoring()

        # 監視開始
        await alert_system.start_monitoring()

        # テストデータ追加
        await simulate_performance_data(alert_system)

        # 10秒間監視実行
        await asyncio.sleep(10)

        # アラート概要取得
        summary = await alert_system.get_alert_summary()
        print(f"アクティブアラート数: {summary['total_active_alerts']}")

        # アクティブアラート表示
        active_alerts = await alert_system.get_active_alerts()
        for alert in active_alerts:
            print(f"アラート: {alert.message}")

        # 監視停止
        await alert_system.stop_monitoring()

    asyncio.run(main())
