"""
APM・オブザーバビリティ統合基盤 - Issue #442
カスタムメトリクス収集とSLO/SLI監視

このモジュールは以下を提供します:
- ビジネスメトリクス収集
- HFT対応の低レイテンシメトリクス
- SLO/SLI自動計算
- アラート条件評価
"""

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import metrics
from opentelemetry.metrics import (
    Counter,
    Gauge,
    Histogram,
    ObservableCounter,
    ObservableGauge,
)

from .telemetry_config import get_logger, get_meter


@dataclass
class SLOConfig:
    """SLO設定"""

    name: str
    description: str
    target_percentage: float  # 99.9, 99.99 etc.
    time_window_seconds: int  # 集計期間
    threshold_value: float  # 閾値
    comparison_operator: str  # "<", ">", "==" etc.


class MetricsCollector:
    """
    メトリクス収集・SLO監視クラス

    Features:
    - HFT対応の高性能メトリクス収集
    - リアルタイムSLO/SLI計算
    - 自動アラート評価
    - ビジネスメトリクス追跡
    """

    def __init__(self, service_name: str = "day-trade"):
        self.service_name = service_name
        self.meter = get_meter()
        self.logger = get_logger()

        # メトリクス保持用
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._observable_counters: Dict[str, ObservableCounter] = {}
        self._observable_gauges: Dict[str, ObservableGauge] = {}

        # SLO設定
        self._slo_configs: Dict[str, SLOConfig] = {}

        # HFT用高速メトリクス（メモリ内）
        self._hft_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._metric_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)

        # SLI計算用データ
        self._sli_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100000))

        # 初期メトリクス作成
        self._create_standard_metrics()

    def _create_standard_metrics(self) -> None:
        """標準メトリクスの作成"""

        # === アプリケーションパフォーマンスメトリクス ===

        # リクエスト数
        self._counters["requests_total"] = self.meter.create_counter(
            name="day_trade_requests_total",
            description="Total number of requests",
            unit="1",
        )

        # レスポンス時間
        self._histograms["request_duration"] = self.meter.create_histogram(
            name="day_trade_request_duration_seconds",
            description="Request duration in seconds",
            unit="s",
        )

        # エラー数
        self._counters["errors_total"] = self.meter.create_counter(
            name="day_trade_errors_total",
            description="Total number of errors",
            unit="1",
        )

        # === HFTビジネスメトリクス ===

        # 取引数
        self._counters["trades_total"] = self.meter.create_counter(
            name="day_trade_trades_total",
            description="Total number of trades executed",
            unit="1",
        )

        # 取引レイテンシ（マイクロ秒）
        self._histograms["trade_latency"] = self.meter.create_histogram(
            name="day_trade_trade_latency_microseconds",
            description="Trade execution latency in microseconds",
            unit="us",
        )

        # 利益・損失
        self._histograms["pnl"] = self.meter.create_histogram(
            name="day_trade_pnl_dollars",
            description="Profit and Loss in dollars",
            unit="USD",
        )

        # ポジション数
        self._gauges["positions_count"] = self.meter.create_gauge(
            name="day_trade_positions_count",
            description="Current number of positions",
            unit="1",
        )

        # === システムメトリクス ===

        # CPU使用率
        self._observable_gauges["cpu_usage"] = self.meter.create_observable_gauge(
            name="day_trade_cpu_usage_percent",
            callbacks=[self._get_cpu_usage],
            description="CPU usage percentage",
            unit="%",
        )

        # メモリ使用率
        self._observable_gauges["memory_usage"] = self.meter.create_observable_gauge(
            name="day_trade_memory_usage_bytes",
            callbacks=[self._get_memory_usage],
            description="Memory usage in bytes",
            unit="bytes",
        )

    def _get_cpu_usage(self, options) -> List:
        """CPU使用率取得"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)
            return [metrics.Observation(cpu_percent)]
        except ImportError:
            return [metrics.Observation(0.0)]

    def _get_memory_usage(self, options) -> List:
        """メモリ使用率取得"""
        try:
            import psutil

            memory = psutil.Process().memory_info()
            return [metrics.Observation(memory.rss)]
        except ImportError:
            return [metrics.Observation(0)]

    # === メトリクス記録メソッド ===

    def increment_counter(
        self, name: str, value: float = 1.0, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """カウンター増加"""
        if name in self._counters:
            self._counters[name].add(value, attributes or {})

    def record_histogram(
        self, name: str, value: float, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """ヒストグラム記録"""
        if name in self._histograms:
            self._histograms[name].record(value, attributes or {})

    def set_gauge(
        self, name: str, value: float, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """ゲージ設定"""
        if name in self._gauges:
            self._gauges[name].set(value, attributes or {})

    # === HFT対応高速メトリクス ===

    def record_hft_latency(self, operation: str, latency_microseconds: float) -> None:
        """HFT用超低レイテンシメトリクス記録"""
        timestamp = time.time()

        with self._metric_locks[f"hft_latency_{operation}"]:
            self._hft_metrics[f"hft_latency_{operation}"].append(
                {"timestamp": timestamp, "latency_us": latency_microseconds}
            )

        # OpenTelemetryメトリクスにも記録
        self.record_histogram(
            "trade_latency", latency_microseconds, {"operation": operation}
        )

    def get_hft_latency_percentiles(
        self, operation: str, percentiles: List[float] = None
    ) -> Dict[str, float]:
        """HFT用レイテンシパーセンタイル計算"""
        if percentiles is None:
            percentiles = [50.0, 95.0, 99.0, 99.9]

        with self._metric_locks[f"hft_latency_{operation}"]:
            latencies = [
                item["latency_us"]
                for item in self._hft_metrics[f"hft_latency_{operation}"]
            ]

        if not latencies:
            return {f"p{p}": 0.0 for p in percentiles}

        import numpy as np

        latencies_array = np.array(latencies)

        return {f"p{p}": np.percentile(latencies_array, p) for p in percentiles}

    # === SLO/SLI管理 ===

    def register_slo(self, slo_config: SLOConfig) -> None:
        """SLO設定登録"""
        self._slo_configs[slo_config.name] = slo_config
        self.logger.info(
            "SLO registered",
            slo_name=slo_config.name,
            target=slo_config.target_percentage,
        )

    def record_sli_event(
        self, slo_name: str, success: bool, value: Optional[float] = None
    ) -> None:
        """SLIイベント記録"""
        timestamp = time.time()

        sli_event = {"timestamp": timestamp, "success": success, "value": value}

        with self._metric_locks[f"sli_{slo_name}"]:
            self._sli_data[slo_name].append(sli_event)

    def calculate_sli(
        self, slo_name: str, time_window_seconds: Optional[int] = None
    ) -> float:
        """SLI計算"""
        if slo_name not in self._slo_configs:
            raise ValueError(f"SLO '{slo_name}' not registered")

        slo_config = self._slo_configs[slo_name]
        window = time_window_seconds or slo_config.time_window_seconds
        cutoff_time = time.time() - window

        with self._metric_locks[f"sli_{slo_name}"]:
            recent_events = [
                event
                for event in self._sli_data[slo_name]
                if event["timestamp"] >= cutoff_time
            ]

        if not recent_events:
            return 100.0  # データがない場合は100%とみなす

        if slo_config.comparison_operator == "<":
            # レイテンシ系SLO
            success_count = sum(
                1
                for event in recent_events
                if event["value"] is not None
                and event["value"] < slo_config.threshold_value
            )
        elif slo_config.comparison_operator == ">":
            # スループット系SLO
            success_count = sum(
                1
                for event in recent_events
                if event["value"] is not None
                and event["value"] > slo_config.threshold_value
            )
        else:
            # ブール系SLO
            success_count = sum(1 for event in recent_events if event["success"])

        return (success_count / len(recent_events)) * 100.0

    def calculate_error_budget(self, slo_name: str) -> float:
        """エラーバジェット計算"""
        sli = self.calculate_sli(slo_name)
        slo_target = self._slo_configs[slo_name].target_percentage

        if sli >= slo_target:
            # SLO達成時: 残りバジェット
            return slo_target - sli + (100.0 - slo_target)
        else:
            # SLO未達成時: 消費バジェット
            return slo_target - sli

    def check_slo_violation(self, slo_name: str) -> bool:
        """SLO違反チェック"""
        sli = self.calculate_sli(slo_name)
        slo_target = self._slo_configs[slo_name].target_percentage

        return sli < slo_target

    # === コンテキストマネージャー ===

    @contextmanager
    def measure_operation(
        self, operation_name: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """操作時間測定"""
        start_time = time.perf_counter()
        start_time_us = time.time_ns() / 1000  # マイクロ秒

        attributes = attributes or {}
        attributes["operation"] = operation_name

        try:
            yield

            # 成功時の処理
            self.increment_counter(
                "requests_total", attributes={**attributes, "status": "success"}
            )

        except Exception as e:
            # エラー時の処理
            self.increment_counter(
                "requests_total", attributes={**attributes, "status": "error"}
            )
            self.increment_counter(
                "errors_total",
                attributes={**attributes, "error_type": type(e).__name__},
            )

            # SLIエラーイベント記録
            for slo_name in self._slo_configs:
                self.record_sli_event(slo_name, success=False)

            raise

        finally:
            # 実行時間記録
            end_time = time.perf_counter()
            end_time_us = time.time_ns() / 1000

            duration_seconds = end_time - start_time
            duration_microseconds = end_time_us - start_time_us

            # メトリクス記録
            self.record_histogram("request_duration", duration_seconds, attributes)

            # HFTレイテンシ記録
            self.record_hft_latency(operation_name, duration_microseconds)

            # SLIイベント記録（レスポンス時間）
            for slo_name, slo_config in self._slo_configs.items():
                if "latency" in slo_name.lower():
                    self.record_sli_event(
                        slo_name, success=True, value=duration_microseconds
                    )

    # === 便利メソッド ===

    def record_trade_execution(
        self,
        symbol: str,
        quantity: float,
        price: float,
        latency_us: float,
        success: bool,
        pnl: Optional[float] = None,
    ) -> None:
        """取引実行メトリクス記録"""
        attributes = {"symbol": symbol, "status": "success" if success else "error"}

        # 取引数
        self.increment_counter("trades_total", attributes=attributes)

        # 取引レイテンシ
        self.record_histogram("trade_latency", latency_us, attributes)
        self.record_hft_latency("trade_execution", latency_us)

        # P&L
        if pnl is not None:
            self.record_histogram("pnl", pnl, attributes)

        # SLIイベント記録
        self.record_sli_event("trade_latency_slo", success, latency_us)
        self.record_sli_event("trade_success_slo", success)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """メトリクスサマリー取得"""
        summary = {}

        # HFTレイテンシサマリー
        for operation in ["trade_execution", "market_data_fetch", "order_placement"]:
            percentiles = self.get_hft_latency_percentiles(operation)
            summary[f"{operation}_latency"] = percentiles

        # SLO状況
        slo_status = {}
        for slo_name in self._slo_configs:
            sli = self.calculate_sli(slo_name)
            error_budget = self.calculate_error_budget(slo_name)
            violation = self.check_slo_violation(slo_name)

            slo_status[slo_name] = {
                "sli": sli,
                "error_budget": error_budget,
                "violation": violation,
                "target": self._slo_configs[slo_name].target_percentage,
            }

        summary["slo_status"] = slo_status

        return summary


# グローバルメトリクスコレクター
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(service_name: str = "day-trade") -> MetricsCollector:
    """グローバルメトリクスコレクター取得"""
    global _global_metrics_collector

    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector(service_name)

        # 標準SLO設定
        _global_metrics_collector.register_slo(
            SLOConfig(
                name="api_latency_slo",
                description="API response time SLO",
                target_percentage=99.9,
                time_window_seconds=300,  # 5分
                threshold_value=50000,  # 50ms in microseconds
                comparison_operator="<",
            )
        )

        _global_metrics_collector.register_slo(
            SLOConfig(
                name="trade_latency_slo",
                description="Trade execution latency SLO",
                target_percentage=99.9,
                time_window_seconds=300,
                threshold_value=50,  # 50μs
                comparison_operator="<",
            )
        )

        _global_metrics_collector.register_slo(
            SLOConfig(
                name="system_availability_slo",
                description="System availability SLO",
                target_percentage=99.99,
                time_window_seconds=3600,  # 1時間
                threshold_value=0,
                comparison_operator="==",
            )
        )

        _global_metrics_collector.register_slo(
            SLOConfig(
                name="trade_success_slo",
                description="Trade success rate SLO",
                target_percentage=99.95,
                time_window_seconds=3600,
                threshold_value=0,
                comparison_operator="==",
            )
        )

    return _global_metrics_collector
