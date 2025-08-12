"""
リアルタイムメトリクス収集システム

Prometheus風のメトリクス収集とエクスポート機能を提供。
システムリソース、アプリケーション性能、MLモデルの性能指標を監視。
"""

import asyncio
import json
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock, RLock
from typing import Dict, List, Optional, Tuple

import psutil


class MetricType(Enum):
    """メトリクス種別"""

    COUNTER = "counter"  # 単調増加する値（リクエスト数など）
    GAUGE = "gauge"  # 上下する値（CPU使用率など）
    HISTOGRAM = "histogram"  # 分布を記録（レスポンス時間など）
    SUMMARY = "summary"  # 統計サマリー


class MetricSource(Enum):
    """メトリクス収集元"""

    SYSTEM = "system"  # システムリソース
    APPLICATION = "application"  # アプリケーション性能
    ML_MODEL = "ml_model"  # 機械学習モデル
    DATABASE = "database"  # データベース関連
    NETWORK = "network"  # ネットワーク関連
    BUSINESS = "business"  # ビジネスメトリクス


@dataclass
class MetricSample:
    """メトリクスサンプル"""

    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """メトリクス定義"""

    name: str
    metric_type: MetricType
    help_text: str
    source: MetricSource
    labels: Dict[str, str] = field(default_factory=dict)
    samples: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add_sample(self, value: float, labels: Dict[str, str] = None):
        """サンプルを追加"""
        combined_labels = {**self.labels}
        if labels:
            combined_labels.update(labels)

        sample = MetricSample(timestamp=datetime.utcnow(), value=value, labels=combined_labels)
        self.samples.append(sample)


class MetricCollector(ABC):
    """メトリクス収集ベースクラス"""

    def __init__(self, name: str, source: MetricSource):
        self.name = name
        self.source = source
        self.metrics: Dict[str, Metric] = {}
        self._lock = Lock()

    @abstractmethod
    async def collect_metrics(self) -> Dict[str, Metric]:
        """メトリクス収集（サブクラスで実装）"""
        pass

    def register_metric(self, metric: Metric):
        """メトリクスを登録"""
        with self._lock:
            self.metrics[metric.name] = metric

    def get_metric(self, name: str) -> Optional[Metric]:
        """メトリクスを取得"""
        with self._lock:
            return self.metrics.get(name)


class SystemMetricsCollector(MetricCollector):
    """システムメトリクス収集器"""

    def __init__(self):
        super().__init__("system", MetricSource.SYSTEM)
        self._initialize_metrics()

    def _initialize_metrics(self):
        """システムメトリクスを初期化"""
        metrics_config = [
            ("cpu_usage_percent", MetricType.GAUGE, "CPU使用率（パーセント）"),
            ("memory_usage_percent", MetricType.GAUGE, "メモリ使用率（パーセント）"),
            ("memory_usage_bytes", MetricType.GAUGE, "メモリ使用量（バイト）"),
            ("disk_usage_percent", MetricType.GAUGE, "ディスク使用率（パーセント）"),
            ("disk_io_read_bytes", MetricType.COUNTER, "ディスクI/O読み込みバイト数"),
            ("disk_io_write_bytes", MetricType.COUNTER, "ディスクI/O書き込みバイト数"),
            ("network_io_sent_bytes", MetricType.COUNTER, "ネットワーク送信バイト数"),
            ("network_io_recv_bytes", MetricType.COUNTER, "ネットワーク受信バイト数"),
            ("process_count", MetricType.GAUGE, "プロセス数"),
        ]

        for name, metric_type, help_text in metrics_config:
            metric = Metric(name, metric_type, help_text, self.source)
            self.register_metric(metric)

    async def collect_metrics(self) -> Dict[str, Metric]:
        """システムメトリクス収集"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.get_metric("cpu_usage_percent").add_sample(cpu_percent)

            # メモリ使用率
            memory = psutil.virtual_memory()
            self.get_metric("memory_usage_percent").add_sample(memory.percent)
            self.get_metric("memory_usage_bytes").add_sample(memory.used)

            # ディスク使用率
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            self.get_metric("disk_usage_percent").add_sample(disk_percent)

            # ディスクI/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.get_metric("disk_io_read_bytes").add_sample(disk_io.read_bytes)
                self.get_metric("disk_io_write_bytes").add_sample(disk_io.write_bytes)

            # ネットワークI/O
            net_io = psutil.net_io_counters()
            if net_io:
                self.get_metric("network_io_sent_bytes").add_sample(net_io.bytes_sent)
                self.get_metric("network_io_recv_bytes").add_sample(net_io.bytes_recv)

            # プロセス数
            process_count = len(psutil.pids())
            self.get_metric("process_count").add_sample(process_count)

        except Exception as e:
            # エラーログを出力（実際の実装では適切なロガーを使用）
            print(f"システムメトリクス収集エラー: {e}")

        return self.metrics


class ApplicationMetricsCollector(MetricCollector):
    """アプリケーション性能メトリクス収集器"""

    def __init__(self):
        super().__init__("application", MetricSource.APPLICATION)
        self._initialize_metrics()
        self._request_times = defaultdict(list)
        self._error_counts = defaultdict(int)

    def _initialize_metrics(self):
        """アプリケーションメトリクスを初期化"""
        metrics_config = [
            ("http_requests_total", MetricType.COUNTER, "HTTP リクエスト総数"),
            (
                "http_request_duration_seconds",
                MetricType.HISTOGRAM,
                "HTTP リクエスト処理時間（秒）",
            ),
            ("http_errors_total", MetricType.COUNTER, "HTTP エラー総数"),
            ("active_connections", MetricType.GAUGE, "アクティブ接続数"),
            ("cache_hit_rate", MetricType.GAUGE, "キャッシュヒット率"),
            ("queue_length", MetricType.GAUGE, "キュー長"),
        ]

        for name, metric_type, help_text in metrics_config:
            metric = Metric(name, metric_type, help_text, self.source)
            self.register_metric(metric)

    async def collect_metrics(self) -> Dict[str, Metric]:
        """アプリケーション性能メトリクス収集"""
        # 実際のアプリケーション状態から収集
        # ここでは例として基本的な実装を提供
        return self.metrics

    def record_request(self, endpoint: str, method: str, duration: float, status_code: int):
        """HTTPリクエストを記録"""
        labels = {"endpoint": endpoint, "method": method, "status": str(status_code)}

        # リクエスト総数
        self.get_metric("http_requests_total").add_sample(1, labels)

        # 処理時間
        self.get_metric("http_request_duration_seconds").add_sample(duration, labels)

        # エラー数（4xx, 5xx）
        if status_code >= 400:
            self.get_metric("http_errors_total").add_sample(1, labels)


class MLModelMetricsCollector(MetricCollector):
    """機械学習モデル性能メトリクス収集器"""

    def __init__(self):
        super().__init__("ml_model", MetricSource.ML_MODEL)
        self._initialize_metrics()

    def _initialize_metrics(self):
        """MLモデルメトリクスを初期化"""
        metrics_config = [
            (
                "model_prediction_duration_seconds",
                MetricType.HISTOGRAM,
                "モデル予測処理時間（秒）",
            ),
            ("model_accuracy", MetricType.GAUGE, "モデル精度"),
            ("model_predictions_total", MetricType.COUNTER, "予測実行総数"),
            ("model_errors_total", MetricType.COUNTER, "モデルエラー総数"),
            ("feature_importance_score", MetricType.GAUGE, "特徴量重要度スコア"),
            ("training_loss", MetricType.GAUGE, "学習時損失"),
        ]

        for name, metric_type, help_text in metrics_config:
            metric = Metric(name, metric_type, help_text, self.source)
            self.register_metric(metric)

    async def collect_metrics(self) -> Dict[str, Metric]:
        """MLモデルメトリクス収集"""
        return self.metrics

    def record_prediction(self, model_name: str, duration: float, accuracy: float = None):
        """予測実行を記録"""
        labels = {"model": model_name}

        self.get_metric("model_prediction_duration_seconds").add_sample(duration, labels)
        self.get_metric("model_predictions_total").add_sample(1, labels)

        if accuracy is not None:
            self.get_metric("model_accuracy").add_sample(accuracy, labels)


class MetricsStorage:
    """メトリクスストレージ"""

    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._lock = RLock()
        self._initialize_database()

    def _initialize_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT,
                    help_text TEXT
                )
            """
            )

            # インデックス作成
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp
                ON metrics(metric_name, timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_source
                ON metrics(source)
            """
            )

            conn.commit()

    def store_metrics(self, metrics: Dict[str, Metric]):
        """メトリクスを保存"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for metric in metrics.values():
                    for sample in metric.samples:
                        conn.execute(
                            """
                            INSERT INTO metrics
                            (metric_name, metric_type, source, timestamp, value, labels, help_text)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                metric.name,
                                metric.metric_type.value,
                                metric.source.value,
                                sample.timestamp.isoformat(),
                                sample.value,
                                json.dumps(sample.labels),
                                metric.help_text,
                            ),
                        )

                    # 古いサンプルをクリア（メモリ使用量を抑制）
                    metric.samples.clear()

                conn.commit()

    def query_metrics(
        self,
        metric_name: str,
        start_time: datetime = None,
        end_time: datetime = None,
        labels: Dict[str, str] = None,
    ) -> List[Tuple[datetime, float, Dict[str, str]]]:
        """メトリクスをクエリ"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT timestamp, value, labels FROM metrics WHERE metric_name = ?"
                params = [metric_name]

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                query += " ORDER BY timestamp"

                cursor = conn.execute(query, params)
                results = []

                for row in cursor.fetchall():
                    timestamp = datetime.fromisoformat(row[0])
                    value = row[1]
                    sample_labels = json.loads(row[2]) if row[2] else {}

                    # ラベルフィルタリング
                    if labels and not all(sample_labels.get(k) == v for k, v in labels.items()):
                        continue

                    results.append((timestamp, value, sample_labels))

                return results


class MetricsExporter:
    """メトリクスエクスポーター（Prometheus形式）"""

    def __init__(self, storage: MetricsStorage):
        self.storage = storage

    def export_prometheus_format(self, metrics: Dict[str, Metric]) -> str:
        """Prometheus形式でメトリクスをエクスポート"""
        lines = []

        for metric in metrics.values():
            # ヘルプテキスト
            lines.append(f"# HELP {metric.name} {metric.help_text}")
            lines.append(f"# TYPE {metric.name} {metric.metric_type.value}")

            # サンプル
            for sample in metric.samples:
                labels_str = ""
                if sample.labels:
                    labels_list = [f'{k}="{v}"' for k, v in sample.labels.items()]
                    labels_str = f"{{{','.join(labels_list)}}}"

                lines.append(f"{metric.name}{labels_str} {sample.value}")

        return "\n".join(lines)

    def export_json_format(self, metrics: Dict[str, Metric]) -> str:
        """JSON形式でメトリクスをエクスポート"""
        data = {}

        for metric in metrics.values():
            data[metric.name] = {
                "type": metric.metric_type.value,
                "help": metric.help_text,
                "source": metric.source.value,
                "samples": [
                    {
                        "timestamp": sample.timestamp.isoformat(),
                        "value": sample.value,
                        "labels": sample.labels,
                    }
                    for sample in metric.samples
                ],
            }

        return json.dumps(data, indent=2, ensure_ascii=False)


class MetricsCollectionSystem:
    """メトリクス収集システム"""

    def __init__(self, storage_path: str = "metrics.db", collection_interval: float = 30.0):
        self.collectors: List[MetricCollector] = []
        self.storage = MetricsStorage(storage_path)
        self.exporter = MetricsExporter(self.storage)
        self.collection_interval = collection_interval
        self._running = False
        self._collection_task = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    def add_collector(self, collector: MetricCollector):
        """コレクターを追加"""
        self.collectors.append(collector)

    def start(self):
        """メトリクス収集開始"""
        if self._running:
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())

    def stop(self):
        """メトリクス収集停止"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
        self._executor.shutdown(wait=True)

    async def _collection_loop(self):
        """メトリクス収集ループ"""
        while self._running:
            try:
                # 全コレクターから並列でメトリクス収集
                tasks = [collector.collect_metrics() for collector in self.collectors]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 結果をマージして保存
                all_metrics = {}
                for result in results:
                    if isinstance(result, dict):
                        all_metrics.update(result)

                # 非同期でストレージに保存
                await asyncio.get_event_loop().run_in_executor(
                    self._executor, self.storage.store_metrics, all_metrics
                )

                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"メトリクス収集エラー: {e}")
                await asyncio.sleep(5)  # エラー時は短い間隔で再試行

    def get_metrics_prometheus(self) -> str:
        """Prometheus形式でメトリクスを取得"""
        all_metrics = {}
        for collector in self.collectors:
            all_metrics.update(collector.metrics)
        return self.exporter.export_prometheus_format(all_metrics)

    def get_metrics_json(self) -> str:
        """JSON形式でメトリクスを取得"""
        all_metrics = {}
        for collector in self.collectors:
            all_metrics.update(collector.metrics)
        return self.exporter.export_json_format(all_metrics)

    def query_metrics(
        self,
        metric_name: str,
        start_time: datetime = None,
        end_time: datetime = None,
        labels: Dict[str, str] = None,
    ) -> List[Tuple[datetime, float, Dict[str, str]]]:
        """メトリクスをクエリ"""
        return self.storage.query_metrics(metric_name, start_time, end_time, labels)


def create_default_metrics_system() -> MetricsCollectionSystem:
    """デフォルトメトリクス収集システムを作成"""
    system = MetricsCollectionSystem()

    # デフォルトコレクターを追加
    system.add_collector(SystemMetricsCollector())
    system.add_collector(ApplicationMetricsCollector())
    system.add_collector(MLModelMetricsCollector())

    return system


# グローバルインスタンス
_metrics_system = None


def get_metrics_system() -> MetricsCollectionSystem:
    """グローバルメトリクス収集システムを取得"""
    global _metrics_system
    if _metrics_system is None:
        _metrics_system = create_default_metrics_system()
    return _metrics_system


def record_http_request(endpoint: str, method: str, duration: float, status_code: int):
    """HTTPリクエストメトリクスを記録"""
    system = get_metrics_system()
    for collector in system.collectors:
        if isinstance(collector, ApplicationMetricsCollector):
            collector.record_request(endpoint, method, duration, status_code)
            break


def record_ml_prediction(model_name: str, duration: float, accuracy: float = None):
    """ML予測メトリクスを記録"""
    system = get_metrics_system()
    for collector in system.collectors:
        if isinstance(collector, MLModelMetricsCollector):
            collector.record_prediction(model_name, duration, accuracy)
            break
