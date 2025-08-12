"""
高度化異常検知アラートシステム
Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

機械学習ベースの異常検知、リアルタイムアラート、
適応型しきい値、インテリジェント通知システム。
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class AnomalyType(Enum):
    """異常タイプ"""

    STATISTICAL_OUTLIER = "statistical_outlier"  # 統計的外れ値
    TREND_CHANGE = "trend_change"  # トレンド変化
    PATTERN_BREAK = "pattern_break"  # パターン破綻
    SPIKE_DETECTION = "spike_detection"  # スパイク検知
    THRESHOLD_BREACH = "threshold_breach"  # 閾値違反
    RATE_CHANGE = "rate_change"  # 変化率異常
    SEASONAL_ANOMALY = "seasonal_anomaly"  # 季節性異常
    CORRELATION_BREAK = "correlation_break"  # 相関異常


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """アラート状態"""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


@dataclass
class AnomalyScore:
    """異常スコア"""

    metric_name: str
    timestamp: datetime
    value: float
    anomaly_score: float  # 0-1, 1が最も異常
    anomaly_type: AnomalyType
    confidence: float  # 信頼度 0-1
    expected_range: Tuple[float, float]
    historical_mean: float
    historical_std: float
    z_score: float
    percentile: float


@dataclass
class Alert:
    """アラート"""

    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    anomaly_score: AnomalyScore
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    escalation_count: int = 0
    notification_count: int = 0


@dataclass
class MetricTimeSeries:
    """メトリクス時系列データ"""

    metric_name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def add_point(self, value: float, timestamp: Optional[datetime] = None):
        """データポイント追加"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        self.values.append(value)
        self.timestamps.append(timestamp)
        self.last_updated = timestamp

    def get_recent_values(
        self, minutes: int = 60
    ) -> Tuple[List[float], List[datetime]]:
        """最近のデータ取得"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        recent_values = []
        recent_timestamps = []

        for value, timestamp in zip(self.values, self.timestamps):
            if timestamp >= cutoff_time:
                recent_values.append(value)
                recent_timestamps.append(timestamp)

        return recent_values, recent_timestamps

    def get_statistical_summary(self, minutes: int = 60) -> Dict[str, float]:
        """統計サマリー取得"""
        values, _ = self.get_recent_values(minutes)

        if not values:
            return {}

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }


class StatisticalAnomalyDetector:
    """統計的異常検知"""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold

    def detect_anomalies(self, time_series: MetricTimeSeries) -> List[AnomalyScore]:
        """異常検知実行"""
        anomalies = []

        if len(time_series.values) < 10:
            return anomalies

        values = list(time_series.values)[-self.window_size :]
        timestamps = list(time_series.timestamps)[-self.window_size :]

        # 統計的外れ値検知
        anomalies.extend(
            self._detect_statistical_outliers(
                time_series.metric_name, values, timestamps
            )
        )

        # スパイク検知
        anomalies.extend(
            self._detect_spikes(time_series.metric_name, values, timestamps)
        )

        # トレンド変化検知
        anomalies.extend(
            self._detect_trend_changes(time_series.metric_name, values, timestamps)
        )

        return anomalies

    def _detect_statistical_outliers(
        self, metric_name: str, values: List[float], timestamps: List[datetime]
    ) -> List[AnomalyScore]:
        """統計的外れ値検知"""
        anomalies = []

        if len(values) < 10:
            return anomalies

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0

        if std == 0:
            return anomalies

        for i, (value, timestamp) in enumerate(zip(values[-10:], timestamps[-10:])):
            z_score = abs(value - mean) / std

            if z_score > self.z_threshold:
                # パーセンタイル計算
                percentile = stats.percentileofscore(values, value)

                anomaly_score = AnomalyScore(
                    metric_name=metric_name,
                    timestamp=timestamp,
                    value=value,
                    anomaly_score=min(z_score / (self.z_threshold * 2), 1.0),
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    confidence=min(z_score / self.z_threshold, 1.0),
                    expected_range=(mean - 2 * std, mean + 2 * std),
                    historical_mean=mean,
                    historical_std=std,
                    z_score=z_score,
                    percentile=percentile,
                )
                anomalies.append(anomaly_score)

        return anomalies

    def _detect_spikes(
        self, metric_name: str, values: List[float], timestamps: List[datetime]
    ) -> List[AnomalyScore]:
        """スパイク検知"""
        anomalies = []

        if len(values) < 5:
            return anomalies

        # 移動平均との差でスパイクを検知
        window = 5
        for i in range(window, len(values)):
            current_value = values[i]
            window_values = values[i - window : i]
            window_mean = statistics.mean(window_values)

            if len(window_values) > 1:
                window_std = statistics.stdev(window_values)

                if window_std > 0:
                    spike_ratio = abs(current_value - window_mean) / window_std

                    if spike_ratio > 4.0:  # 4σを超えるスパイク
                        anomaly_score = AnomalyScore(
                            metric_name=metric_name,
                            timestamp=timestamps[i],
                            value=current_value,
                            anomaly_score=min(spike_ratio / 8.0, 1.0),
                            anomaly_type=AnomalyType.SPIKE_DETECTION,
                            confidence=min(spike_ratio / 4.0, 1.0),
                            expected_range=(
                                window_mean - 2 * window_std,
                                window_mean + 2 * window_std,
                            ),
                            historical_mean=window_mean,
                            historical_std=window_std,
                            z_score=spike_ratio,
                            percentile=90 if current_value > window_mean else 10,
                        )
                        anomalies.append(anomaly_score)

        return anomalies

    def _detect_trend_changes(
        self, metric_name: str, values: List[float], timestamps: List[datetime]
    ) -> List[AnomalyScore]:
        """トレンド変化検知"""
        anomalies = []

        if len(values) < 20:
            return anomalies

        # 前半と後半の平均を比較
        mid_point = len(values) // 2
        first_half = values[:mid_point]
        second_half = values[mid_point:]

        first_mean = statistics.mean(first_half)
        second_mean = statistics.mean(second_half)

        # 全体の標準偏差
        overall_std = statistics.stdev(values)

        if overall_std > 0:
            trend_change_ratio = abs(second_mean - first_mean) / overall_std

            if trend_change_ratio > 2.0:  # 大きなトレンド変化
                anomaly_score = AnomalyScore(
                    metric_name=metric_name,
                    timestamp=timestamps[-1],
                    value=second_mean,
                    anomaly_score=min(trend_change_ratio / 4.0, 1.0),
                    anomaly_type=AnomalyType.TREND_CHANGE,
                    confidence=min(trend_change_ratio / 2.0, 1.0),
                    expected_range=(first_mean - overall_std, first_mean + overall_std),
                    historical_mean=first_mean,
                    historical_std=overall_std,
                    z_score=trend_change_ratio,
                    percentile=75 if second_mean > first_mean else 25,
                )
                anomalies.append(anomaly_score)

        return anomalies


class MLAnomalyDetector:
    """機械学習ベース異常検知"""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.is_fitted: Dict[str, bool] = {}

    def detect_anomalies(self, time_series: MetricTimeSeries) -> List[AnomalyScore]:
        """ML異常検知実行"""
        if not SKLEARN_AVAILABLE:
            return []

        anomalies = []
        metric_name = time_series.metric_name

        if len(time_series.values) < 50:
            return anomalies

        # データ準備
        values = list(time_series.values)[-200:]  # 最新200ポイント
        timestamps = list(time_series.timestamps)[-200:]

        # 特徴量作成
        features = self._create_features(values)

        if len(features) < 20:
            return anomalies

        # モデル訓練/更新
        self._fit_model(metric_name, features)

        if not self.is_fitted.get(metric_name, False):
            return anomalies

        # 異常検知実行
        model = self.models[metric_name]
        scaler = self.scalers[metric_name]

        # 最新データの異常スコア計算
        recent_features = features[-10:]  # 最新10ポイント
        scaled_features = scaler.transform(recent_features)

        anomaly_scores = model.decision_function(scaled_features)
        anomaly_predictions = model.predict(scaled_features)

        # 異常ポイントを識別
        for i, (score, prediction, timestamp) in enumerate(
            zip(anomaly_scores, anomaly_predictions, timestamps[-10:])
        ):
            if prediction == -1:  # 異常
                # スコア正規化（-1〜1を0〜1に）
                normalized_score = max(0, (1 - score) / 2)

                anomaly_score = AnomalyScore(
                    metric_name=metric_name,
                    timestamp=timestamp,
                    value=values[len(values) - 10 + i],
                    anomaly_score=normalized_score,
                    anomaly_type=AnomalyType.PATTERN_BREAK,
                    confidence=normalized_score,
                    expected_range=(
                        np.mean(values) - np.std(values),
                        np.mean(values) + np.std(values),
                    ),
                    historical_mean=np.mean(values),
                    historical_std=np.std(values),
                    z_score=abs(score),
                    percentile=90 if score < 0 else 10,
                )
                anomalies.append(anomaly_score)

        return anomalies

    def _create_features(self, values: List[float]) -> np.ndarray:
        """特徴量作成"""
        if not SKLEARN_AVAILABLE:
            return np.array([])

        features = []
        window_sizes = [5, 10, 20]

        for i in range(max(window_sizes), len(values)):
            feature_vector = []

            # 現在の値
            feature_vector.append(values[i])

            # 移動平均
            for window in window_sizes:
                if i >= window:
                    moving_avg = np.mean(values[i - window : i])
                    feature_vector.append(moving_avg)

                    # 現在値と移動平均の差
                    feature_vector.append(values[i] - moving_avg)

            # 変化率
            if i > 0:
                feature_vector.append(values[i] - values[i - 1])
            else:
                feature_vector.append(0)

            # 変動性
            if i >= 5:
                feature_vector.append(np.std(values[i - 5 : i + 1]))
            else:
                feature_vector.append(0)

            features.append(feature_vector)

        return np.array(features)

    def _fit_model(self, metric_name: str, features: np.ndarray):
        """モデル訓練"""
        if not SKLEARN_AVAILABLE:
            return

        try:
            # Isolation Forest モデル
            if metric_name not in self.models:
                self.models[metric_name] = IsolationForest(
                    contamination=self.contamination, random_state=42, n_estimators=100
                )
                self.scalers[metric_name] = StandardScaler()

            # データスケーリング
            scaler = self.scalers[metric_name]
            scaled_features = scaler.fit_transform(features)

            # モデル訓練
            model = self.models[metric_name]
            model.fit(scaled_features)

            self.is_fitted[metric_name] = True

        except Exception as e:
            logging.error(f"MLモデル訓練エラー {metric_name}: {e}")
            self.is_fitted[metric_name] = False


class AdaptiveThresholdDetector:
    """適応型閾値検知"""

    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.baseline_history: Dict[str, deque] = {}

    def detect_anomalies(self, time_series: MetricTimeSeries) -> List[AnomalyScore]:
        """適応型閾値異常検知"""
        anomalies = []
        metric_name = time_series.metric_name

        if len(time_series.values) < 10:
            return anomalies

        # 閾値初期化/更新
        self._update_thresholds(metric_name, time_series.values)

        if metric_name not in self.thresholds:
            return anomalies

        thresholds = self.thresholds[metric_name]
        current_value = time_series.values[-1]
        current_timestamp = time_series.timestamps[-1]

        # 閾値違反チェック
        if current_value > thresholds["upper"] or current_value < thresholds["lower"]:
            # 異常スコア計算
            if current_value > thresholds["upper"]:
                anomaly_score_val = min(
                    (current_value - thresholds["upper"])
                    / (thresholds["critical_upper"] - thresholds["upper"]),
                    1.0,
                )
            else:
                anomaly_score_val = min(
                    (thresholds["lower"] - current_value)
                    / (thresholds["lower"] - thresholds["critical_lower"]),
                    1.0,
                )

            anomaly_score = AnomalyScore(
                metric_name=metric_name,
                timestamp=current_timestamp,
                value=current_value,
                anomaly_score=anomaly_score_val,
                anomaly_type=AnomalyType.THRESHOLD_BREACH,
                confidence=anomaly_score_val,
                expected_range=(thresholds["lower"], thresholds["upper"]),
                historical_mean=thresholds["mean"],
                historical_std=thresholds["std"],
                z_score=abs((current_value - thresholds["mean"]) / thresholds["std"]),
                percentile=95 if current_value > thresholds["upper"] else 5,
            )
            anomalies.append(anomaly_score)

        return anomalies

    def _update_thresholds(self, metric_name: str, values: deque):
        """閾値更新"""
        if metric_name not in self.baseline_history:
            self.baseline_history[metric_name] = deque(maxlen=100)

        # ベースライン履歴更新
        baseline = self.baseline_history[metric_name]
        baseline.extend(list(values)[-10:])  # 最新10値を追加

        if len(baseline) < 10:
            return

        # 統計計算
        mean = statistics.mean(baseline)
        std = statistics.stdev(baseline) if len(baseline) > 1 else 0

        # 適応型閾値計算
        if metric_name in self.thresholds:
            # 既存閾値を段階的に調整
            old_thresholds = self.thresholds[metric_name]

            new_upper = mean + 2 * std
            new_lower = mean - 2 * std

            # 適応率を適用した段階的更新
            adapted_upper = (
                old_thresholds["upper"] * (1 - self.adaptation_rate)
                + new_upper * self.adaptation_rate
            )
            adapted_lower = (
                old_thresholds["lower"] * (1 - self.adaptation_rate)
                + new_lower * self.adaptation_rate
            )
        else:
            # 初回設定
            adapted_upper = mean + 2 * std
            adapted_lower = mean - 2 * std

        self.thresholds[metric_name] = {
            "mean": mean,
            "std": std,
            "upper": adapted_upper,
            "lower": adapted_lower,
            "critical_upper": mean + 3 * std,
            "critical_lower": mean - 3 * std,
        }


class AdvancedAnomalyAlertSystem:
    """高度化異常検知アラートシステム"""

    def __init__(self):
        self.time_series_data: Dict[str, MetricTimeSeries] = {}
        self.detectors = {
            "statistical": StatisticalAnomalyDetector(),
            "ml": MLAnomalyDetector(),
            "adaptive": AdaptiveThresholdDetector(),
        }

        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)

        # アラート設定
        self.suppression_rules: Dict[str, timedelta] = defaultdict(
            lambda: timedelta(minutes=5)
        )
        self.escalation_thresholds: Dict[AlertSeverity, int] = {
            AlertSeverity.INFO: 10,
            AlertSeverity.LOW: 5,
            AlertSeverity.MEDIUM: 3,
            AlertSeverity.HIGH: 2,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.EMERGENCY: 1,
        }

        self.logger = logging.getLogger(__name__)

    def ingest_metric(
        self, metric_name: str, value: float, timestamp: Optional[datetime] = None
    ):
        """メトリクス取り込み"""
        if metric_name not in self.time_series_data:
            self.time_series_data[metric_name] = MetricTimeSeries(metric_name)

        self.time_series_data[metric_name].add_point(value, timestamp)

    async def process_anomaly_detection(self, metric_name: str) -> List[Alert]:
        """異常検知処理実行"""
        if metric_name not in self.time_series_data:
            return []

        time_series = self.time_series_data[metric_name]
        all_anomalies = []

        # 各検知器で異常検知実行
        for detector_name, detector in self.detectors.items():
            try:
                anomalies = detector.detect_anomalies(time_series)
                all_anomalies.extend(anomalies)
            except Exception as e:
                self.logger.error(f"異常検知エラー {detector_name}: {e}")

        # 異常をアラートに変換
        new_alerts = []
        for anomaly in all_anomalies:
            alert = self._create_alert_from_anomaly(anomaly)
            if alert and self._should_create_alert(alert):
                new_alerts.append(alert)
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)

        return new_alerts

    def _create_alert_from_anomaly(self, anomaly: AnomalyScore) -> Alert:
        """異常からアラート作成"""
        # 重要度決定
        severity = self._determine_severity(anomaly)

        # アラートID生成
        alert_id = (
            f"alert_{anomaly.metric_name}_{int(time.time())}_{hash(str(anomaly.value))}"
        )

        # タイトル・説明生成
        title = f"{anomaly.anomaly_type.value.replace('_', ' ').title()} - {anomaly.metric_name}"

        description = (
            f"メトリクス '{anomaly.metric_name}' で{anomaly.anomaly_type.value}を検知しました。"
            f"現在値: {anomaly.value:.2f}, "
            f"期待範囲: {anomaly.expected_range[0]:.2f} - {anomaly.expected_range[1]:.2f}, "
            f"異常スコア: {anomaly.anomaly_score:.2f}, "
            f"信頼度: {anomaly.confidence:.2f}"
        )

        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            status=AlertStatus.ACTIVE,
            anomaly_score=anomaly,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[anomaly.anomaly_type.value, anomaly.metric_name],
            metadata={
                "detector_type": anomaly.anomaly_type.value,
                "metric_name": anomaly.metric_name,
                "anomaly_score": anomaly.anomaly_score,
                "confidence": anomaly.confidence,
            },
        )

        return alert

    def _determine_severity(self, anomaly: AnomalyScore) -> AlertSeverity:
        """重要度決定"""
        score = anomaly.anomaly_score
        confidence = anomaly.confidence

        # 重要度マッピング
        if score >= 0.9 and confidence >= 0.9:
            return AlertSeverity.EMERGENCY
        elif score >= 0.8 and confidence >= 0.8:
            return AlertSeverity.CRITICAL
        elif score >= 0.6 and confidence >= 0.7:
            return AlertSeverity.HIGH
        elif score >= 0.4 and confidence >= 0.5:
            return AlertSeverity.MEDIUM
        elif score >= 0.2:
            return AlertSeverity.LOW
        else:
            return AlertSeverity.INFO

    def _should_create_alert(self, alert: Alert) -> bool:
        """アラート作成判定"""
        # 抑制ルールチェック
        metric_name = alert.anomaly_score.metric_name
        suppression_period = self.suppression_rules.get(
            metric_name, timedelta(minutes=5)
        )

        # 同じメトリクスの最近のアラートをチェック
        cutoff_time = datetime.utcnow() - suppression_period

        for existing_alert in self.active_alerts.values():
            if (
                existing_alert.anomaly_score.metric_name == metric_name
                and existing_alert.created_at > cutoff_time
                and existing_alert.status == AlertStatus.ACTIVE
            ):
                return False  # 抑制

        return True

    async def process_all_metrics(self) -> Dict[str, List[Alert]]:
        """全メトリクスの異常検知処理"""
        all_new_alerts = {}

        for metric_name in self.time_series_data.keys():
            try:
                new_alerts = await self.process_anomaly_detection(metric_name)
                if new_alerts:
                    all_new_alerts[metric_name] = new_alerts
            except Exception as e:
                self.logger.error(f"メトリクス処理エラー {metric_name}: {e}")

        return all_new_alerts

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """アラート確認"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.utcnow()
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """アラート解決"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            return True
        return False

    def suppress_alert(self, alert_id: str, duration: timedelta) -> bool:
        """アラート抑制"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = datetime.utcnow() + duration
            alert.updated_at = datetime.utcnow()
            return True
        return False

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """アクティブアラート取得"""
        alerts = []

        for alert in self.active_alerts.values():
            if alert.status == AlertStatus.ACTIVE:
                if severity is None or alert.severity == severity:
                    alerts.append(alert)

        # 重要度順にソート
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.HIGH: 2,
            AlertSeverity.MEDIUM: 3,
            AlertSeverity.LOW: 4,
            AlertSeverity.INFO: 5,
        }

        alerts.sort(
            key=lambda x: (severity_order.get(x.severity, 6), x.created_at),
            reverse=True,
        )
        return alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """アラート統計取得"""
        active_count_by_severity = defaultdict(int)
        total_count_by_severity = defaultdict(int)

        # アクティブアラート統計
        for alert in self.active_alerts.values():
            if alert.status == AlertStatus.ACTIVE:
                active_count_by_severity[alert.severity.value] += 1

        # 履歴アラート統計
        for alert in self.alert_history:
            total_count_by_severity[alert.severity.value] += 1

        # メトリクス別統計
        metrics_stats = defaultdict(int)
        for alert in self.active_alerts.values():
            if alert.status == AlertStatus.ACTIVE:
                metrics_stats[alert.anomaly_score.metric_name] += 1

        return {
            "active_alerts": {
                "total": len(
                    [
                        a
                        for a in self.active_alerts.values()
                        if a.status == AlertStatus.ACTIVE
                    ]
                ),
                "by_severity": dict(active_count_by_severity),
            },
            "total_alerts": {
                "total": len(self.alert_history),
                "by_severity": dict(total_count_by_severity),
            },
            "by_metric": dict(metrics_stats),
            "detection_rates": {
                detector: len(
                    [
                        a
                        for a in self.alert_history
                        if a.anomaly_score.anomaly_type.value in detector
                    ]
                )
                for detector in ["statistical", "ml", "adaptive"]
            },
        }

    async def cleanup_expired_alerts(self):
        """期限切れアラートクリーンアップ"""
        current_time = datetime.utcnow()
        expired_alert_ids = []

        for alert_id, alert in self.active_alerts.items():
            # 抑制期限チェック
            if (
                alert.status == AlertStatus.SUPPRESSED
                and alert.suppressed_until
                and current_time > alert.suppressed_until
            ):
                alert.status = AlertStatus.ACTIVE
                alert.updated_at = current_time

            # 古いアラート自動解決（24時間経過）
            elif (
                alert.status == AlertStatus.ACTIVE
                and (current_time - alert.created_at).total_seconds() > 86400
            ):
                alert.status = AlertStatus.EXPIRED
                alert.updated_at = current_time
                expired_alert_ids.append(alert_id)

        # 期限切れアラートを削除
        for alert_id in expired_alert_ids:
            del self.active_alerts[alert_id]


# Factory function
def create_advanced_anomaly_alert_system() -> AdvancedAnomalyAlertSystem:
    """高度化異常検知アラートシステム作成"""
    return AdvancedAnomalyAlertSystem()


if __name__ == "__main__":
    # テスト実行
    async def test_advanced_anomaly_detection():
        print("=== 高度化異常検知アラートシステムテスト ===")

        try:
            # システム初期化
            alert_system = create_advanced_anomaly_alert_system()

            print("\n1. 高度化異常検知アラートシステム初期化完了")
            print(f"   検知器数: {len(alert_system.detectors)}")

            # 正常データの取り込み
            print("\n2. 正常データ取り込み...")

            # 正常範囲のデータ（平均100、標準偏差10）
            np.random.seed(42)

            for i in range(100):
                # 正常データ
                normal_value = np.random.normal(100, 10)
                timestamp = datetime.utcnow() - timedelta(minutes=100 - i)
                alert_system.ingest_metric("cpu_usage", normal_value, timestamp)

                # API応答時間データ
                api_time = np.random.exponential(0.2)  # 指数分布
                alert_system.ingest_metric("api_response_time", api_time, timestamp)

                await asyncio.sleep(0.01)  # わずかな間隔

            print("   正常データ取り込み完了: 100件")

            # 異常データの注入
            print("\n3. 異常データ注入テスト...")

            # スパイク異常
            alert_system.ingest_metric(
                "cpu_usage", 200, datetime.utcnow()
            )  # 正常値の2倍
            await asyncio.sleep(0.1)

            # 統計的外れ値
            alert_system.ingest_metric(
                "api_response_time", 5.0, datetime.utcnow()
            )  # 通常0.2秒のところ5秒
            await asyncio.sleep(0.1)

            # 閾値違反データ
            alert_system.ingest_metric("cpu_usage", 150, datetime.utcnow())
            await asyncio.sleep(0.1)

            print("   異常データ注入完了")

            # 異常検知実行
            print("\n4. 異常検知処理実行...")

            all_alerts = await alert_system.process_all_metrics()
            total_alerts = sum(len(alerts) for alerts in all_alerts.values())

            print(f"   検知されたアラート数: {total_alerts}")

            for metric_name, alerts in all_alerts.items():
                print(f"   {metric_name}: {len(alerts)}件")
                for alert in alerts:
                    print(f"     - {alert.severity.value}: {alert.title}")
                    print(f"       異常スコア: {alert.anomaly_score.anomaly_score:.3f}")

            # アクティブアラート確認
            print("\n5. アクティブアラート確認...")

            active_alerts = alert_system.get_active_alerts()
            print(f"   アクティブアラート数: {len(active_alerts)}")

            for alert in active_alerts[:5]:  # 上位5件表示
                print(f"   - [{alert.severity.value}] {alert.title}")
                print(f"     作成: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     値: {alert.anomaly_score.value:.2f}")

            # アラート統計
            print("\n6. アラート統計...")

            stats = alert_system.get_alert_statistics()
            print(f"   総アクティブアラート: {stats['active_alerts']['total']}")
            print("   重要度別:")
            for severity, count in stats["active_alerts"]["by_severity"].items():
                print(f"     {severity}: {count}")

            print("   メトリクス別:")
            for metric, count in stats["by_metric"].items():
                print(f"     {metric}: {count}")

            # アラート操作テスト
            print("\n7. アラート操作テスト...")

            if active_alerts:
                first_alert = active_alerts[0]

                # アラート確認
                success = alert_system.acknowledge_alert(
                    first_alert.alert_id, "test_user"
                )
                print(f"   アラート確認: {'成功' if success else '失敗'}")

                # アラート抑制
                success = alert_system.suppress_alert(
                    first_alert.alert_id, timedelta(minutes=10)
                )
                print(f"   アラート抑制: {'成功' if success else '失敗'}")

            # 継続監視テスト
            print("\n8. 継続監視テスト...")

            # さらに異常データを投入
            for i in range(5):
                # 徐々に悪化するデータ
                deteriorating_value = 100 + (i * 20)
                alert_system.ingest_metric("system_load", deteriorating_value)
                await asyncio.sleep(0.1)

            # 再度異常検知
            new_alerts = await alert_system.process_all_metrics()
            new_total = sum(len(alerts) for alerts in new_alerts.values())
            print(f"   新規検知アラート: {new_total}件")

            # クリーンアップテスト
            print("\n9. アラートクリーンアップテスト...")

            await alert_system.cleanup_expired_alerts()

            final_active = len(alert_system.get_active_alerts())
            print(f"   クリーンアップ後アクティブアラート: {final_active}件")

            # パフォーマンステスト
            print("\n10. パフォーマンステスト...")

            start_time = time.time()

            # 大量データ処理
            for i in range(1000):
                value = np.random.normal(100, 10)
                alert_system.ingest_metric("performance_test", value)

            # 異常検知実行
            perf_alerts = await alert_system.process_anomaly_detection(
                "performance_test"
            )

            duration = time.time() - start_time
            print(f"   1000件処理時間: {duration:.3f}秒")
            print(f"   スループット: {1000 / duration:.1f}件/秒")
            print(f"   検知アラート: {len(perf_alerts)}件")

            print("\n✅ 高度化異常検知アラートシステムテスト完了")

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_advanced_anomaly_detection())
