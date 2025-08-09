#!/usr/bin/env python3
"""
Prometheus メトリクス統合
Phase G: 本番運用最適化フェーズ

Prometheus形式でのメトリクス出力とサービス監視
"""

import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import json


@dataclass
class PrometheusMetric:
    """Prometheusメトリクス"""
    name: str
    metric_type: str  # counter, gauge, histogram, summary
    help_text: str
    labels: Dict[str, str]
    value: float
    timestamp: Optional[float] = None


class PrometheusRegistry:
    """Prometheusメトリクスレジストリ"""

    def __init__(self):
        self.metrics: Dict[str, PrometheusMetric] = {}
        self.lock = threading.Lock()

    def register_metric(self, metric: PrometheusMetric):
        """メトリクス登録"""
        with self.lock:
            key = f"{metric.name}_{self._labels_to_key(metric.labels)}"
            self.metrics[key] = metric

    def increment_counter(self, name: str, labels: Dict[str, str] = None,
                         amount: float = 1.0):
        """カウンター増加"""
        if labels is None:
            labels = {}

        key = f"{name}_{self._labels_to_key(labels)}"

        with self.lock:
            if key in self.metrics:
                self.metrics[key].value += amount
            else:
                self.metrics[key] = PrometheusMetric(
                    name=name,
                    metric_type="counter",
                    help_text=f"Counter metric {name}",
                    labels=labels,
                    value=amount
                )

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """ゲージ設定"""
        if labels is None:
            labels = {}

        key = f"{name}_{self._labels_to_key(labels)}"

        with self.lock:
            if key in self.metrics:
                self.metrics[key].value = value
            else:
                self.metrics[key] = PrometheusMetric(
                    name=name,
                    metric_type="gauge",
                    help_text=f"Gauge metric {name}",
                    labels=labels,
                    value=value
                )

    def observe_histogram(self, name: str, value: float,
                         labels: Dict[str, str] = None):
        """ヒストグラム観測"""
        if labels is None:
            labels = {}

        # 簡易ヒストグラム実装（本格的には bucket 管理が必要）
        buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

        for bucket in buckets:
            if value <= bucket:
                bucket_labels = {**labels, "le": str(bucket)}
                self.increment_counter(f"{name}_bucket", bucket_labels)

        # +Inf bucket
        inf_labels = {**labels, "le": "+Inf"}
        self.increment_counter(f"{name}_bucket", inf_labels)

        # Count and sum
        self.increment_counter(f"{name}_count", labels)
        self.increment_counter(f"{name}_sum", labels, value)

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """ラベル辞書をキー文字列に変換"""
        if not labels:
            return ""
        return "_".join(f"{k}_{v}" for k, v in sorted(labels.items()))

    def generate_metrics_text(self) -> str:
        """Prometheus形式のメトリクステキスト生成"""
        with self.lock:
            lines = []
            metrics_by_name = defaultdict(list)

            # メトリクス名でグループ化
            for metric in self.metrics.values():
                metrics_by_name[metric.name].append(metric)

            for name, metric_list in metrics_by_name.items():
                # ヘルプとタイプ情報
                if metric_list:
                    first_metric = metric_list[0]
                    lines.append(f"# HELP {name} {first_metric.help_text}")
                    lines.append(f"# TYPE {name} {first_metric.metric_type}")

                # 各メトリクス値
                for metric in metric_list:
                    if metric.labels:
                        label_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
                        lines.append(f"{metric.name}{{{label_str}}} {metric.value}")
                    else:
                        lines.append(f"{metric.name} {metric.value}")

                lines.append("")  # 空行でセパレート

            return "\n".join(lines)


class MetricsHandler(BaseHTTPRequestHandler):
    """メトリクスHTTPハンドラー"""

    def __init__(self, registry: PrometheusRegistry):
        self.registry = registry
        super().__init__()

    def do_GET(self):
        """GETリクエスト処理"""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()

            metrics_text = self.registry.generate_metrics_text()
            self.wfile.write(metrics_text.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


class PrometheusExporter:
    """Prometheusメトリクスエクスポーター"""

    def __init__(self, port: int = 9090):
        self.port = port
        self.registry = PrometheusRegistry()
        self.server = None
        self.server_thread = None

        # システムメトリクス初期化
        self._initialize_system_metrics()

        print(f"[PROMETHEUS] Prometheus エクスポーター初期化 (ポート: {port})")

    def _initialize_system_metrics(self):
        """システムメトリクス初期化"""
        # アプリケーション起動時間
        self.registry.set_gauge("daytrade_start_time_seconds", time.time())

        # 基本カウンター
        self.registry.register_metric(PrometheusMetric(
            name="daytrade_requests_total",
            metric_type="counter",
            help_text="Total number of requests",
            labels={},
            value=0
        ))

        self.registry.register_metric(PrometheusMetric(
            name="daytrade_errors_total",
            metric_type="counter",
            help_text="Total number of errors",
            labels={},
            value=0
        ))

    def start_server(self):
        """メトリクスサーバー開始"""
        if self.server:
            return

        def handler_factory(registry):
            def create_handler(*args, **kwargs):
                return MetricsHandler(registry, *args, **kwargs)
            return create_handler

        try:
            self.server = HTTPServer(('0.0.0.0', self.port), handler_factory(self.registry))
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()

            print(f"[PROMETHEUS] メトリクスサーバー開始: http://localhost:{self.port}/metrics")
        except Exception as e:
            print(f"[ERROR] メトリクスサーバー開始エラー: {e}")

    def stop_server(self):
        """メトリクスサーバー停止"""
        if self.server:
            self.server.shutdown()
            self.server = None
            print("[PROMETHEUS] メトリクスサーバー停止")

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """リクエストメトリクス記録"""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code)
        }

        self.registry.increment_counter("daytrade_requests_total", labels)
        self.registry.observe_histogram("daytrade_request_duration_seconds", duration, labels)

        if status_code >= 400:
            self.registry.increment_counter("daytrade_errors_total", labels)

    def record_trading_operation(self, operation_type: str, symbol: str,
                               success: bool, duration: float):
        """取引操作メトリクス記録"""
        labels = {
            "operation": operation_type,
            "symbol": symbol,
            "status": "success" if success else "error"
        }

        self.registry.increment_counter("daytrade_operations_total", labels)
        self.registry.observe_histogram("daytrade_operation_duration_seconds", duration, labels)

    def update_system_metrics(self, cpu_percent: float, memory_usage_mb: float,
                            active_connections: int):
        """システムメトリクス更新"""
        self.registry.set_gauge("daytrade_cpu_usage_percent", cpu_percent)
        self.registry.set_gauge("daytrade_memory_usage_mb", memory_usage_mb)
        self.registry.set_gauge("daytrade_active_connections", active_connections)

    def record_ml_prediction(self, model_name: str, prediction_time: float,
                           confidence: float):
        """ML予測メトリクス記録"""
        labels = {"model": model_name}

        self.registry.increment_counter("daytrade_predictions_total", labels)
        self.registry.observe_histogram("daytrade_prediction_duration_seconds",
                                      prediction_time, labels)
        self.registry.set_gauge("daytrade_prediction_confidence", confidence, labels)

    def record_data_fetch(self, source: str, symbol: str, success: bool, duration: float):
        """データ取得メトリクス記録"""
        labels = {
            "source": source,
            "symbol": symbol,
            "status": "success" if success else "error"
        }

        self.registry.increment_counter("daytrade_data_fetches_total", labels)
        self.registry.observe_histogram("daytrade_data_fetch_duration_seconds",
                                      duration, labels)


class ApplicationMetricsCollector:
    """アプリケーションメトリクス収集器"""

    def __init__(self, prometheus_exporter: PrometheusExporter):
        self.prometheus = prometheus_exporter
        self.operation_start_times = {}

    def start_operation(self, operation_id: str) -> str:
        """操作開始"""
        self.operation_start_times[operation_id] = time.time()
        return operation_id

    def end_operation(self, operation_id: str, operation_type: str,
                     symbol: str = "", success: bool = True):
        """操作終了"""
        if operation_id in self.operation_start_times:
            duration = time.time() - self.operation_start_times[operation_id]
            del self.operation_start_times[operation_id]

            self.prometheus.record_trading_operation(
                operation_type, symbol, success, duration
            )

    def record_api_call(self, method: str, endpoint: str, status_code: int, duration: float):
        """API呼び出し記録"""
        self.prometheus.record_request(method, endpoint, status_code, duration)

    def record_error(self, error_type: str, component: str):
        """エラー記録"""
        labels = {"error_type": error_type, "component": component}
        self.prometheus.registry.increment_counter("daytrade_errors_by_type_total", labels)


def main():
    """テスト実行"""
    exporter = PrometheusExporter(port=9090)
    collector = ApplicationMetricsCollector(exporter)

    try:
        # サーバー開始
        exporter.start_server()

        print("[TEST] Prometheusメトリクス テスト実行中...")

        # テストメトリクス生成
        for i in range(10):
            # リクエスト記録
            collector.record_api_call("GET", "/api/predict", 200, 0.1 + i * 0.01)

            # 取引操作記録
            op_id = collector.start_operation(f"trade_{i}")
            time.sleep(0.01)  # 模擬処理時間
            collector.end_operation(op_id, "buy", "AAPL", success=(i % 7 != 0))

            # ML予測記録
            exporter.record_ml_prediction("LSTM", 0.05 + i * 0.001, 0.8 + i * 0.01)

            # システムメトリクス更新
            exporter.update_system_metrics(50.0 + i, 1024 + i * 10, 100 + i)

            time.sleep(0.1)

        print(f"[INFO] メトリクス確認: http://localhost:9090/metrics")
        print("テストメトリクス生成完了")

        # テスト用に短時間待機
        print("30秒後に停止...")
        time.sleep(30)

    except KeyboardInterrupt:
        pass
    finally:
        exporter.stop_server()
        print("[COMPLETE] Prometheusエクスポーター テスト完了")


if __name__ == "__main__":
    main()
