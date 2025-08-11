#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - リアルタイムパフォーマンス監視システム
システム・AI・取引パフォーマンスの統合監視

CPU、メモリ、AI推論性能、取引成果のリアルタイム監視
"""

import asyncio
import json
import time
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from .alert_system import (
    Alert,
    AlertLevel,
    AlertManager,
    AlertType,
)
from .live_prediction_engine import LivePrediction

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class PerformanceConfig:
    """パフォーマンス監視設定"""

    # 監視間隔
    monitoring_interval: float = 1.0  # 秒
    metrics_retention_hours: int = 24
    alert_check_interval: float = 30.0  # 30秒間隔でアラートチェック

    # システムメトリクス閾値
    cpu_warning_threshold: float = 70.0  # 70%
    cpu_critical_threshold: float = 90.0  # 90%
    memory_warning_threshold: float = 80.0  # 80%
    memory_critical_threshold: float = 95.0  # 95%

    # AI性能閾値
    prediction_latency_warning: float = 1000.0  # 1秒
    prediction_latency_critical: float = 5000.0  # 5秒
    prediction_accuracy_warning: float = 0.6  # 60%
    model_confidence_warning: float = 0.5  # 50%

    # データ品質閾値
    data_quality_warning: float = 0.8  # 80%
    data_freshness_warning: float = 60.0  # 60秒

    # 取引パフォーマンス
    max_drawdown_warning: float = 0.05  # 5%
    max_drawdown_critical: float = 0.10  # 10%

    # ストレージ設定
    enable_metrics_export: bool = True
    metrics_export_interval: int = 300  # 5分


@dataclass
class SystemMetrics:
    """システムメトリクス"""

    timestamp: datetime

    # CPU・メモリ
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float

    # ネットワーク
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0

    # ディスク
    disk_usage_percent: float = 0.0
    disk_free_gb: float = 0.0

    # プロセス固有
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    thread_count: int = 0
    open_files: int = 0


@dataclass
class AIPerformanceMetrics:
    """AI性能メトリクス"""

    timestamp: datetime

    # 予測性能
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    average_prediction_latency: float = 0.0  # ミリ秒

    # モデル別性能
    ml_predictions: int = 0
    ml_avg_confidence: float = 0.0
    ml_avg_latency: float = 0.0

    rl_decisions: int = 0
    rl_avg_confidence: float = 0.0
    rl_avg_latency: float = 0.0

    sentiment_analyses: int = 0
    sentiment_avg_confidence: float = 0.0
    sentiment_avg_latency: float = 0.0

    # データ品質
    data_quality_score: float = 1.0
    data_freshness_seconds: float = 0.0

    # エラー率
    error_rate: float = 0.0
    timeout_rate: float = 0.0


@dataclass
class TradingMetrics:
    """取引パフォーマンスメトリクス"""

    timestamp: datetime

    # 基本統計
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0

    # 信頼度統計
    avg_signal_confidence: float = 0.0
    high_confidence_signals: int = 0  # >80%

    # 仮想パフォーマンス（実際の取引結果の代替）
    virtual_portfolio_value: float = 1000000.0
    virtual_return: float = 0.0
    virtual_drawdown: float = 0.0
    virtual_sharpe_ratio: float = 0.0

    # リスク指標
    position_concentration: float = 0.0
    volatility: float = 0.0


class SystemPerformanceMonitor:
    """システムパフォーマンス監視"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.process = psutil.Process()

        # メトリクス履歴
        max_points = int(
            config.metrics_retention_hours * 3600 / config.monitoring_interval
        )
        self.metrics_history: deque = deque(maxlen=max_points)

        # 前回のネットワーク統計
        self.prev_network_stats = psutil.net_io_counters()

        logger.info("System Performance Monitor initialized")

    def collect_metrics(self) -> SystemMetrics:
        """システムメトリクス収集"""

        try:
            # CPU・メモリ
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()

            # ネットワーク
            network_stats = psutil.net_io_counters()

            # ディスク
            disk_usage = psutil.disk_usage("/")

            # プロセス情報
            process_info = self.process.as_dict(
                ["cpu_percent", "memory_info", "num_threads", "num_fds"]
            )

            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_info.percent,
                memory_used_gb=memory_info.used / 1024**3,
                memory_available_gb=memory_info.available / 1024**3,
                network_bytes_sent=network_stats.bytes_sent
                - self.prev_network_stats.bytes_sent,
                network_bytes_recv=network_stats.bytes_recv
                - self.prev_network_stats.bytes_recv,
                disk_usage_percent=disk_usage.percent,
                disk_free_gb=disk_usage.free / 1024**3,
                process_cpu_percent=process_info.get("cpu_percent", 0),
                process_memory_mb=process_info.get("memory_info", {}).get("rss", 0)
                / 1024**2,
                thread_count=process_info.get("num_threads", 0),
                open_files=process_info.get("num_fds", 0)
                if hasattr(self.process, "num_fds")
                else 0,
            )

            self.prev_network_stats = network_stats
            self.metrics_history.append(metrics)

            return metrics

        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_available_gb=0.0,
            )

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """メトリクス要約取得"""

        if not self.metrics_history:
            return {}

        # 指定時間内のメトリクス
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {}

        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]

        return {
            "period_hours": hours,
            "sample_count": len(recent_metrics),
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
            },
            "memory": {
                "current": memory_values[-1] if memory_values else 0,
                "average": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
            },
            "latest_metrics": recent_metrics[-1] if recent_metrics else None,
        }


class AIPerformanceMonitor:
    """AI性能監視"""

    def __init__(self, config: PerformanceConfig):
        self.config = config

        # メトリクス履歴
        max_points = int(
            config.metrics_retention_hours * 3600 / config.monitoring_interval
        )
        self.metrics_history: deque = deque(maxlen=max_points)

        # 性能統計
        self.prediction_latencies: deque = deque(maxlen=1000)
        self.prediction_confidences: deque = deque(maxlen=1000)
        self.error_count = 0
        self.timeout_count = 0
        self.total_predictions = 0

        logger.info("AI Performance Monitor initialized")

    def record_prediction(self, prediction: LivePrediction):
        """予測結果記録"""

        try:
            self.total_predictions += 1

            # レイテンシー記録
            if prediction.processing_time_ms > 0:
                self.prediction_latencies.append(prediction.processing_time_ms)

            # 信頼度記録
            self.prediction_confidences.append(prediction.confidence)

        except Exception as e:
            logger.error(f"Prediction recording error: {e}")

    def record_error(self, error_type: str = "general"):
        """エラー記録"""
        self.error_count += 1

        if error_type == "timeout":
            self.timeout_count += 1

    def collect_metrics(self) -> AIPerformanceMetrics:
        """AI性能メトリクス収集"""

        try:
            # 基本統計
            successful_predictions = max(0, self.total_predictions - self.error_count)

            # レイテンシー統計
            avg_latency = (
                np.mean(self.prediction_latencies) if self.prediction_latencies else 0.0
            )

            # 信頼度統計
            avg_confidence = (
                np.mean(self.prediction_confidences)
                if self.prediction_confidences
                else 0.0
            )

            # エラー率
            error_rate = self.error_count / max(self.total_predictions, 1)
            timeout_rate = self.timeout_count / max(self.total_predictions, 1)

            metrics = AIPerformanceMetrics(
                timestamp=datetime.now(),
                total_predictions=self.total_predictions,
                successful_predictions=successful_predictions,
                failed_predictions=self.error_count,
                average_prediction_latency=avg_latency,
                ml_avg_confidence=avg_confidence,  # 簡略化
                ml_avg_latency=avg_latency,
                rl_avg_confidence=avg_confidence * 0.9,
                rl_avg_latency=avg_latency * 1.2,
                sentiment_avg_confidence=avg_confidence * 0.8,
                sentiment_avg_latency=avg_latency * 2.0,
                data_quality_score=1.0 - error_rate,
                error_rate=error_rate,
                timeout_rate=timeout_rate,
            )

            self.metrics_history.append(metrics)
            return metrics

        except Exception as e:
            logger.error(f"AI metrics collection error: {e}")
            return AIPerformanceMetrics(timestamp=datetime.now())

    def get_performance_summary(self) -> Dict[str, Any]:
        """AI性能要約"""

        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]

        # レイテンシー分布
        latency_percentiles = {}
        if self.prediction_latencies:
            latency_percentiles = {
                "p50": np.percentile(self.prediction_latencies, 50),
                "p95": np.percentile(self.prediction_latencies, 95),
                "p99": np.percentile(self.prediction_latencies, 99),
            }

        return {
            "total_predictions": latest.total_predictions,
            "success_rate": latest.successful_predictions
            / max(latest.total_predictions, 1),
            "error_rate": latest.error_rate,
            "average_latency_ms": latest.average_prediction_latency,
            "latency_percentiles": latency_percentiles,
            "average_confidence": latest.ml_avg_confidence,
            "data_quality_score": latest.data_quality_score,
        }


class TradingPerformanceMonitor:
    """取引パフォーマンス監視"""

    def __init__(self, config: PerformanceConfig):
        self.config = config

        # メトリクス履歴
        max_points = int(
            config.metrics_retention_hours * 3600 / config.monitoring_interval
        )
        self.metrics_history: deque = deque(maxlen=max_points)

        # 取引統計
        self.signal_history: List[Dict] = []
        self.virtual_trades: List[Dict] = []
        self.virtual_portfolio_value = 1000000.0  # 初期100万

        logger.info("Trading Performance Monitor initialized")

    def record_trading_signal(self, prediction: LivePrediction):
        """取引シグナル記録"""

        try:
            signal_data = {
                "timestamp": prediction.timestamp,
                "symbol": prediction.symbol,
                "action": prediction.final_action,
                "confidence": prediction.action_confidence,
                "predicted_return": prediction.predicted_return,
                "position_size": prediction.position_size_recommendation,
            }

            self.signal_history.append(signal_data)

            # 履歴サイズ制限
            if len(self.signal_history) > 10000:
                self.signal_history = self.signal_history[-5000:]

            # 仮想取引実行
            self._execute_virtual_trade(signal_data)

        except Exception as e:
            logger.error(f"Trading signal recording error: {e}")

    def _execute_virtual_trade(self, signal: Dict):
        """仮想取引実行"""

        try:
            if signal["action"] in ["BUY", "SELL"] and signal["confidence"] > 0.6:
                # 仮想取引記録
                virtual_trade = {
                    "timestamp": signal["timestamp"],
                    "symbol": signal["symbol"],
                    "action": signal["action"],
                    "confidence": signal["confidence"],
                    "position_size": signal["position_size"],
                    "expected_return": signal["predicted_return"],
                }

                self.virtual_trades.append(virtual_trade)

                # 簡単な仮想P&L計算
                expected_return = signal["predicted_return"] * signal["confidence"]
                position_value = self.virtual_portfolio_value * signal["position_size"]

                if signal["action"] == "BUY":
                    pnl = position_value * expected_return
                elif signal["action"] == "SELL":
                    pnl = position_value * (-expected_return)  # ショート利益
                else:
                    pnl = 0

                # ポートフォリオ価値更新
                self.virtual_portfolio_value += pnl
                virtual_trade["pnl"] = pnl

        except Exception as e:
            logger.error(f"Virtual trade execution error: {e}")

    def collect_metrics(self) -> TradingMetrics:
        """取引メトリクス収集"""

        try:
            # 最近1時間のシグナル統計
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_signals = [
                s for s in self.signal_history if s["timestamp"] > cutoff_time
            ]

            # シグナル統計
            total_signals = len(recent_signals)
            buy_signals = len([s for s in recent_signals if s["action"] == "BUY"])
            sell_signals = len([s for s in recent_signals if s["action"] == "SELL"])
            hold_signals = total_signals - buy_signals - sell_signals

            # 信頼度統計
            confidences = [s["confidence"] for s in recent_signals]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            high_confidence = len([c for c in confidences if c > 0.8])

            # 仮想パフォーマンス
            virtual_return = (self.virtual_portfolio_value - 1000000.0) / 1000000.0

            # 仮想ドローダウン計算
            virtual_drawdown = self._calculate_virtual_drawdown()

            metrics = TradingMetrics(
                timestamp=datetime.now(),
                total_signals=total_signals,
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
                avg_signal_confidence=avg_confidence,
                high_confidence_signals=high_confidence,
                virtual_portfolio_value=self.virtual_portfolio_value,
                virtual_return=virtual_return,
                virtual_drawdown=virtual_drawdown,
                virtual_sharpe_ratio=self._calculate_virtual_sharpe(),
            )

            self.metrics_history.append(metrics)
            return metrics

        except Exception as e:
            logger.error(f"Trading metrics collection error: {e}")
            return TradingMetrics(timestamp=datetime.now())

    def _calculate_virtual_drawdown(self) -> float:
        """仮想ドローダウン計算"""

        if len(self.virtual_trades) < 10:
            return 0.0

        # 過去の仮想取引からポートフォリオ価値の履歴を再構築
        portfolio_values = [1000000.0]  # 初期値

        for trade in self.virtual_trades[-50:]:  # 最近50取引
            portfolio_values.append(portfolio_values[-1] + trade.get("pnl", 0))

        # 最大ドローダウン計算
        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_virtual_sharpe(self) -> float:
        """仮想シャープレシオ計算"""

        if len(self.virtual_trades) < 30:
            return 0.0

        # 取引リターン計算
        returns = []
        for trade in self.virtual_trades[-100:]:  # 最近100取引
            if "pnl" in trade:
                trade_return = trade["pnl"] / 1000000.0  # 初期資本で正規化
                returns.append(trade_return)

        if not returns:
            return 0.0

        # シャープレシオ計算
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # 年率換算（簡略化）
        sharpe = (avg_return / std_return) * np.sqrt(252)
        return sharpe

    def get_trading_summary(self) -> Dict[str, Any]:
        """取引要約"""

        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]

        return {
            "total_signals": latest.total_signals,
            "signal_breakdown": {
                "buy": latest.buy_signals,
                "sell": latest.sell_signals,
                "hold": latest.hold_signals,
            },
            "avg_confidence": latest.avg_signal_confidence,
            "high_confidence_ratio": latest.high_confidence_signals
            / max(latest.total_signals, 1),
            "virtual_portfolio_value": latest.virtual_portfolio_value,
            "virtual_return": latest.virtual_return,
            "virtual_drawdown": latest.virtual_drawdown,
            "virtual_sharpe_ratio": latest.virtual_sharpe_ratio,
        }


class RealTimePerformanceMonitor:
    """統合リアルタイムパフォーマンス監視システム"""

    def __init__(
        self, config: PerformanceConfig, alert_manager: Optional[AlertManager] = None
    ):
        self.config = config
        self.alert_manager = alert_manager

        # 個別モニター
        self.system_monitor = SystemPerformanceMonitor(config)
        self.ai_monitor = AIPerformanceMonitor(config)
        self.trading_monitor = TradingPerformanceMonitor(config)

        # 監視状態
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_alert_check = time.time()

        # 統計
        self.stats = {
            "monitoring_start_time": None,
            "total_monitoring_cycles": 0,
            "alerts_generated": 0,
            "last_metrics_export": None,
        }

        logger.info("Real-Time Performance Monitor initialized")

    async def start_monitoring(self):
        """監視開始"""

        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self.monitoring_active = True
        self.stats["monitoring_start_time"] = datetime.now()

        # 監視タスク開始
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Real-time performance monitoring started")

    async def stop_monitoring(self):
        """監視停止"""

        self.monitoring_active = False

        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Real-time performance monitoring stopped")

    async def _monitoring_loop(self):
        """監視メインループ"""

        logger.info("Performance monitoring loop started")

        while self.monitoring_active:
            try:
                cycle_start = time.time()

                # メトリクス収集
                system_metrics = self.system_monitor.collect_metrics()
                ai_metrics = self.ai_monitor.collect_metrics()
                trading_metrics = self.trading_monitor.collect_metrics()

                # アラートチェック
                current_time = time.time()
                if (
                    current_time - self.last_alert_check
                    >= self.config.alert_check_interval
                ):
                    await self._check_alerts(
                        system_metrics, ai_metrics, trading_metrics
                    )
                    self.last_alert_check = current_time

                # メトリクスエクスポート
                if self.config.enable_metrics_export:
                    await self._export_metrics(
                        system_metrics, ai_metrics, trading_metrics
                    )

                # 統計更新
                self.stats["total_monitoring_cycles"] += 1

                # 次のサイクルまで待機
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.config.monitoring_interval - cycle_time)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Performance monitoring cycle error: {e}")
                await asyncio.sleep(1)  # エラー時短時間待機

    async def _check_alerts(
        self,
        system_metrics: SystemMetrics,
        ai_metrics: AIPerformanceMetrics,
        trading_metrics: TradingMetrics,
    ):
        """パフォーマンスアラートチェック"""

        if not self.alert_manager:
            return

        alerts_to_send = []

        # システムアラート
        if system_metrics.cpu_percent > self.config.cpu_critical_threshold:
            alerts_to_send.append(
                self._create_system_alert(
                    "High CPU Usage",
                    f"CPU usage is at {system_metrics.cpu_percent:.1f}%",
                    AlertLevel.CRITICAL,
                    {"cpu_percent": system_metrics.cpu_percent},
                )
            )
        elif system_metrics.cpu_percent > self.config.cpu_warning_threshold:
            alerts_to_send.append(
                self._create_system_alert(
                    "CPU Usage Warning",
                    f"CPU usage is at {system_metrics.cpu_percent:.1f}%",
                    AlertLevel.WARNING,
                    {"cpu_percent": system_metrics.cpu_percent},
                )
            )

        # メモリアラート
        if system_metrics.memory_percent > self.config.memory_critical_threshold:
            alerts_to_send.append(
                self._create_system_alert(
                    "High Memory Usage",
                    f"Memory usage is at {system_metrics.memory_percent:.1f}%",
                    AlertLevel.CRITICAL,
                    {"memory_percent": system_metrics.memory_percent},
                )
            )

        # AI性能アラート
        if (
            ai_metrics.average_prediction_latency
            > self.config.prediction_latency_critical
        ):
            alerts_to_send.append(
                self._create_ai_alert(
                    "High AI Latency",
                    f"AI prediction latency is {ai_metrics.average_prediction_latency:.0f}ms",
                    AlertLevel.CRITICAL,
                    {"latency_ms": ai_metrics.average_prediction_latency},
                )
            )

        if ai_metrics.error_rate > 0.1:  # 10%エラー率
            alerts_to_send.append(
                self._create_ai_alert(
                    "High AI Error Rate",
                    f"AI error rate is {ai_metrics.error_rate:.1%}",
                    AlertLevel.WARNING,
                    {"error_rate": ai_metrics.error_rate},
                )
            )

        # 取引パフォーマンスアラート
        if trading_metrics.virtual_drawdown > self.config.max_drawdown_critical:
            alerts_to_send.append(
                self._create_trading_alert(
                    "High Drawdown",
                    f"Virtual portfolio drawdown is {trading_metrics.virtual_drawdown:.1%}",
                    AlertLevel.CRITICAL,
                    {"drawdown": trading_metrics.virtual_drawdown},
                )
            )

        # アラート送信
        for alert in alerts_to_send:
            await self.alert_manager.send_alert(alert)
            self.stats["alerts_generated"] += 1

    def _create_system_alert(
        self, title: str, message: str, level: AlertLevel, data: Dict
    ) -> Alert:
        """システムアラート作成"""
        return Alert(
            id=f"system_{int(time.time())}",
            timestamp=datetime.now(),
            level=level,
            alert_type=AlertType.PERFORMANCE_ALERT,
            title=title,
            message=message,
            confidence=1.0,
            data=data,
        )

    def _create_ai_alert(
        self, title: str, message: str, level: AlertLevel, data: Dict
    ) -> Alert:
        """AIアラート作成"""
        return Alert(
            id=f"ai_{int(time.time())}",
            timestamp=datetime.now(),
            level=level,
            alert_type=AlertType.PERFORMANCE_ALERT,
            title=title,
            message=message,
            confidence=1.0,
            data=data,
        )

    def _create_trading_alert(
        self, title: str, message: str, level: AlertLevel, data: Dict
    ) -> Alert:
        """取引アラート作成"""
        return Alert(
            id=f"trading_{int(time.time())}",
            timestamp=datetime.now(),
            level=level,
            alert_type=AlertType.RISK_ALERT,
            title=title,
            message=message,
            confidence=1.0,
            data=data,
        )

    async def _export_metrics(
        self,
        system_metrics: SystemMetrics,
        ai_metrics: AIPerformanceMetrics,
        trading_metrics: TradingMetrics,
    ):
        """メトリクスエクスポート"""

        # 簡略化実装 - 実際にはPrometheus、InfluxDB等に送信
        current_time = time.time()
        last_export = self.stats.get("last_metrics_export", 0)

        if current_time - last_export >= self.config.metrics_export_interval:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "process_memory_mb": system_metrics.process_memory_mb,
                },
                "ai": {
                    "total_predictions": ai_metrics.total_predictions,
                    "success_rate": ai_metrics.successful_predictions
                    / max(ai_metrics.total_predictions, 1),
                    "average_latency": ai_metrics.average_prediction_latency,
                    "error_rate": ai_metrics.error_rate,
                },
                "trading": {
                    "total_signals": trading_metrics.total_signals,
                    "virtual_return": trading_metrics.virtual_return,
                    "virtual_drawdown": trading_metrics.virtual_drawdown,
                    "sharpe_ratio": trading_metrics.virtual_sharpe_ratio,
                },
            }

            # ログ出力（実際の実装では外部システムに送信）
            logger.info(f"Metrics export: {json.dumps(export_data, indent=2)}")

            self.stats["last_metrics_export"] = current_time

    def record_prediction(self, prediction: LivePrediction):
        """予測記録"""
        self.ai_monitor.record_prediction(prediction)
        self.trading_monitor.record_trading_signal(prediction)

    def record_error(self, error_type: str = "general"):
        """エラー記録"""
        self.ai_monitor.record_error(error_type)

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """包括的ステータス取得"""

        return {
            "monitoring_active": self.monitoring_active,
            "monitoring_stats": self.stats,
            "system_summary": self.system_monitor.get_metrics_summary(hours=1),
            "ai_summary": self.ai_monitor.get_performance_summary(),
            "trading_summary": self.trading_monitor.get_trading_summary(),
            "last_update": datetime.now().isoformat(),
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """ダッシュボード用データ"""

        system_summary = self.system_monitor.get_metrics_summary(hours=1)
        ai_summary = self.ai_monitor.get_performance_summary()
        trading_summary = self.trading_monitor.get_trading_summary()

        return {
            "system": {
                "cpu_percent": system_summary.get("cpu", {}).get("current", 0),
                "memory_percent": system_summary.get("memory", {}).get("current", 0),
                "status": "healthy"
                if system_summary.get("cpu", {}).get("current", 0) < 70
                else "warning",
            },
            "ai": {
                "total_predictions": ai_summary.get("total_predictions", 0),
                "success_rate": ai_summary.get("success_rate", 0),
                "average_latency": ai_summary.get("average_latency_ms", 0),
                "status": "healthy"
                if ai_summary.get("error_rate", 0) < 0.1
                else "warning",
            },
            "trading": {
                "total_signals": trading_summary.get("total_signals", 0),
                "virtual_return": trading_summary.get("virtual_return", 0),
                "virtual_drawdown": trading_summary.get("virtual_drawdown", 0),
                "status": "healthy"
                if trading_summary.get("virtual_drawdown", 0) < 0.05
                else "warning",
            },
            "timestamp": datetime.now().isoformat(),
        }


# 便利関数
def create_performance_monitor(
    alert_manager: Optional[AlertManager] = None,
) -> RealTimePerformanceMonitor:
    """パフォーマンス監視システム作成"""

    config = PerformanceConfig(
        monitoring_interval=2.0,  # 2秒間隔
        cpu_warning_threshold=70.0,
        memory_warning_threshold=80.0,
        enable_metrics_export=True,
    )

    return RealTimePerformanceMonitor(config, alert_manager)


if __name__ == "__main__":
    # パフォーマンス監視テスト
    async def test_performance_monitor():
        print("=== Performance Monitor Test ===")

        try:
            # パフォーマンス監視システム作成
            monitor = create_performance_monitor()

            print("Starting performance monitoring...")

            # 監視開始
            await monitor.start_monitoring()

            # 10秒間監視
            for i in range(5):
                await asyncio.sleep(2)

                # ダミー予測記録
                from .live_prediction_engine import LivePrediction

                dummy_prediction = LivePrediction(
                    symbol="AAPL",
                    timestamp=datetime.now(),
                    predicted_price=150.0,
                    predicted_return=0.02,
                    confidence=0.8,
                    final_action="BUY",
                    action_confidence=0.75,
                    processing_time_ms=500,
                )

                monitor.record_prediction(dummy_prediction)

                # ステータス取得
                if i % 2 == 0:
                    status = monitor.get_comprehensive_status()
                    print(
                        f"Monitoring cycle {i+1}: {status['monitoring_stats']['total_monitoring_cycles']} cycles"
                    )

            # ダッシュボードデータ取得
            dashboard_data = monitor.get_dashboard_data()
            print(f"Dashboard data: {json.dumps(dashboard_data, indent=2)}")

            # 監視停止
            await monitor.stop_monitoring()

            print("Performance monitor test completed")

        except Exception as e:
            print(f"Test error: {e}")
            import traceback

            traceback.print_exc()

    # テスト実行
    asyncio.run(test_performance_monitor())
