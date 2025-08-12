"""
パフォーマンスダッシュボードシステム

Grafana風のリアルタイムダッシュボード機能を提供。
メトリクスデータの可視化、アラートの管理、パフォーマンス分析を行う。
"""

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .metrics_collection_system import MetricsCollectionSystem, get_metrics_system


class ChartType(Enum):
    """チャート種別"""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"


class TimeRange(Enum):
    """時間範囲"""

    LAST_5M = "5m"
    LAST_15M = "15m"
    LAST_1H = "1h"
    LAST_6H = "6h"
    LAST_24H = "24h"
    LAST_7D = "7d"
    LAST_30D = "30d"
    CUSTOM = "custom"


class AggregationType(Enum):
    """集約種別"""

    NONE = "none"
    AVERAGE = "avg"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


@dataclass
class DashboardQuery:
    """ダッシュボードクエリ"""

    metric_name: str
    time_range: TimeRange
    aggregation: AggregationType = AggregationType.NONE
    labels: Dict[str, str] = field(default_factory=dict)
    custom_start_time: Optional[datetime] = None
    custom_end_time: Optional[datetime] = None
    group_by: List[str] = field(default_factory=list)
    interval: str = "1m"  # 集約間隔


@dataclass
class ChartWidget:
    """チャートウィジェット"""

    id: str
    title: str
    chart_type: ChartType
    queries: List[DashboardQuery]
    width: int = 6  # グリッドの幅（1-12）
    height: int = 4  # グリッドの高さ
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """ダッシュボード"""

    id: str
    title: str
    description: str
    widgets: List[ChartWidget]
    refresh_interval: int = 30  # 秒
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class DataProcessor:
    """データ処理クラス"""

    @staticmethod
    def get_time_range_bounds(
        time_range: TimeRange,
        custom_start: datetime = None,
        custom_end: datetime = None,
    ) -> Tuple[datetime, datetime]:
        """時間範囲の開始・終了時刻を取得"""
        now = datetime.utcnow()

        if time_range == TimeRange.CUSTOM:
            if custom_start and custom_end:
                return custom_start, custom_end
            else:
                # カスタム範囲が指定されていない場合は過去1時間をデフォルト
                return now - timedelta(hours=1), now

        time_deltas = {
            TimeRange.LAST_5M: timedelta(minutes=5),
            TimeRange.LAST_15M: timedelta(minutes=15),
            TimeRange.LAST_1H: timedelta(hours=1),
            TimeRange.LAST_6H: timedelta(hours=6),
            TimeRange.LAST_24H: timedelta(hours=24),
            TimeRange.LAST_7D: timedelta(days=7),
            TimeRange.LAST_30D: timedelta(days=30),
        }

        delta = time_deltas.get(time_range, timedelta(hours=1))
        return now - delta, now

    @staticmethod
    def aggregate_data(
        data: List[Tuple[datetime, float]],
        aggregation: AggregationType,
        interval: str = "1m",
    ) -> List[Tuple[datetime, float]]:
        """データを集約"""
        if aggregation == AggregationType.NONE or not data:
            return data

        # 間隔をタイムデルタに変換
        interval_seconds = DataProcessor._parse_interval(interval)

        # データを時間バケットに分割
        buckets = {}
        for timestamp, value in data:
            bucket_key = int(timestamp.timestamp() // interval_seconds) * interval_seconds
            bucket_time = datetime.utcfromtimestamp(bucket_key)

            if bucket_time not in buckets:
                buckets[bucket_time] = []
            buckets[bucket_time].append(value)

        # 各バケットで集約
        aggregated_data = []
        for bucket_time, values in sorted(buckets.items()):
            if aggregation == AggregationType.AVERAGE:
                agg_value = sum(values) / len(values)
            elif aggregation == AggregationType.SUM:
                agg_value = sum(values)
            elif aggregation == AggregationType.MAX:
                agg_value = max(values)
            elif aggregation == AggregationType.MIN:
                agg_value = min(values)
            elif aggregation == AggregationType.MEDIAN:
                sorted_values = sorted(values)
                n = len(sorted_values)
                agg_value = (sorted_values[n // 2] + sorted_values[(n - 1) // 2]) / 2
            elif aggregation == AggregationType.PERCENTILE_95:
                sorted_values = sorted(values)
                idx = int(0.95 * (len(sorted_values) - 1))
                agg_value = sorted_values[idx]
            elif aggregation == AggregationType.PERCENTILE_99:
                sorted_values = sorted(values)
                idx = int(0.99 * (len(sorted_values) - 1))
                agg_value = sorted_values[idx]
            else:
                agg_value = values[0] if values else 0

            aggregated_data.append((bucket_time, agg_value))

        return aggregated_data

    @staticmethod
    def _parse_interval(interval: str) -> int:
        """間隔文字列を秒数に変換"""
        if interval.endswith("s"):
            return int(interval[:-1])
        elif interval.endswith("m"):
            return int(interval[:-1]) * 60
        elif interval.endswith("h"):
            return int(interval[:-1]) * 3600
        elif interval.endswith("d"):
            return int(interval[:-1]) * 86400
        else:
            return 60  # デフォルト1分


class DashboardEngine:
    """ダッシュボードエンジン"""

    def __init__(self, metrics_system: MetricsCollectionSystem):
        self.metrics_system = metrics_system
        self.dashboards: Dict[str, Dashboard] = {}
        self.data_processor = DataProcessor()

    def create_dashboard(self, dashboard: Dashboard):
        """ダッシュボードを作成"""
        self.dashboards[dashboard.id] = dashboard

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """ダッシュボードを取得"""
        return self.dashboards.get(dashboard_id)

    def list_dashboards(self) -> List[Dashboard]:
        """ダッシュボード一覧を取得"""
        return list(self.dashboards.values())

    async def execute_query(self, query: DashboardQuery) -> List[Tuple[datetime, float]]:
        """クエリを実行してデータを取得"""
        start_time, end_time = self.data_processor.get_time_range_bounds(
            query.time_range, query.custom_start_time, query.custom_end_time
        )

        # メトリクスデータを取得
        raw_data = self.metrics_system.query_metrics(
            query.metric_name, start_time, end_time, query.labels
        )

        # データ変換（timestamp, value, labels）から（timestamp, value）へ
        time_series_data = [(timestamp, value) for timestamp, value, _ in raw_data]

        # データ集約
        aggregated_data = self.data_processor.aggregate_data(
            time_series_data, query.aggregation, query.interval
        )

        return aggregated_data

    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """ダッシュボードデータを取得"""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard {dashboard_id} not found")

        dashboard_data = {
            "id": dashboard.id,
            "title": dashboard.title,
            "description": dashboard.description,
            "refresh_interval": dashboard.refresh_interval,
            "widgets": [],
        }

        # 各ウィジェットのデータを並列取得
        widget_tasks = []
        for widget in dashboard.widgets:
            widget_tasks.append(self._get_widget_data(widget))

        widget_data_list = await asyncio.gather(*widget_tasks)

        for widget, widget_data in zip(dashboard.widgets, widget_data_list):
            dashboard_data["widgets"].append(
                {
                    "id": widget.id,
                    "title": widget.title,
                    "chart_type": widget.chart_type.value,
                    "width": widget.width,
                    "height": widget.height,
                    "options": widget.options,
                    "data": widget_data,
                }
            )

        return dashboard_data

    async def _get_widget_data(self, widget: ChartWidget) -> Dict[str, Any]:
        """ウィジェットデータを取得"""
        widget_data = {"series": []}

        # 各クエリを並列実行
        query_tasks = [self.execute_query(query) for query in widget.queries]
        query_results = await asyncio.gather(*query_tasks)

        for query, data in zip(widget.queries, query_results):
            series_name = f"{query.metric_name}"
            if query.labels:
                label_str = ",".join([f"{k}={v}" for k, v in query.labels.items()])
                series_name += f"({label_str})"

            widget_data["series"].append(
                {
                    "name": series_name,
                    "data": [{"timestamp": ts.isoformat(), "value": value} for ts, value in data],
                }
            )

        return widget_data


class DashboardTemplates:
    """ダッシュボードテンプレート"""

    @staticmethod
    def create_system_overview_dashboard() -> Dashboard:
        """システム概要ダッシュボードを作成"""
        widgets = [
            ChartWidget(
                id="cpu_usage",
                title="CPU使用率",
                chart_type=ChartType.LINE,
                queries=[
                    DashboardQuery(
                        metric_name="cpu_usage_percent",
                        time_range=TimeRange.LAST_1H,
                        aggregation=AggregationType.AVERAGE,
                        interval="1m",
                    )
                ],
                width=6,
                height=4,
                options={"unit": "%", "max": 100},
            ),
            ChartWidget(
                id="memory_usage",
                title="メモリ使用率",
                chart_type=ChartType.LINE,
                queries=[
                    DashboardQuery(
                        metric_name="memory_usage_percent",
                        time_range=TimeRange.LAST_1H,
                        aggregation=AggregationType.AVERAGE,
                        interval="1m",
                    )
                ],
                width=6,
                height=4,
                options={"unit": "%", "max": 100},
            ),
            ChartWidget(
                id="disk_usage",
                title="ディスク使用率",
                chart_type=ChartType.GAUGE,
                queries=[
                    DashboardQuery(
                        metric_name="disk_usage_percent",
                        time_range=TimeRange.LAST_5M,
                        aggregation=AggregationType.AVERAGE,
                    )
                ],
                width=4,
                height=4,
                options={"unit": "%", "max": 100},
            ),
            ChartWidget(
                id="network_io",
                title="ネットワークI/O",
                chart_type=ChartType.LINE,
                queries=[
                    DashboardQuery(
                        metric_name="network_io_sent_bytes",
                        time_range=TimeRange.LAST_1H,
                        aggregation=AggregationType.AVERAGE,
                        interval="1m",
                    ),
                    DashboardQuery(
                        metric_name="network_io_recv_bytes",
                        time_range=TimeRange.LAST_1H,
                        aggregation=AggregationType.AVERAGE,
                        interval="1m",
                    ),
                ],
                width=8,
                height=4,
                options={"unit": "bytes/sec"},
            ),
        ]

        return Dashboard(
            id="system_overview",
            title="システム概要",
            description="システムリソースの監視ダッシュボード",
            widgets=widgets,
        )

    @staticmethod
    def create_application_performance_dashboard() -> Dashboard:
        """アプリケーション性能ダッシュボードを作成"""
        widgets = [
            ChartWidget(
                id="request_rate",
                title="リクエスト率",
                chart_type=ChartType.LINE,
                queries=[
                    DashboardQuery(
                        metric_name="http_requests_total",
                        time_range=TimeRange.LAST_1H,
                        aggregation=AggregationType.SUM,
                        interval="1m",
                    )
                ],
                width=6,
                height=4,
                options={"unit": "req/min"},
            ),
            ChartWidget(
                id="response_time",
                title="レスポンス時間",
                chart_type=ChartType.LINE,
                queries=[
                    DashboardQuery(
                        metric_name="http_request_duration_seconds",
                        time_range=TimeRange.LAST_1H,
                        aggregation=AggregationType.AVERAGE,
                        interval="1m",
                    ),
                    DashboardQuery(
                        metric_name="http_request_duration_seconds",
                        time_range=TimeRange.LAST_1H,
                        aggregation=AggregationType.PERCENTILE_95,
                        interval="1m",
                    ),
                ],
                width=6,
                height=4,
                options={"unit": "seconds"},
            ),
            ChartWidget(
                id="error_rate",
                title="エラー率",
                chart_type=ChartType.BAR,
                queries=[
                    DashboardQuery(
                        metric_name="http_errors_total",
                        time_range=TimeRange.LAST_1H,
                        aggregation=AggregationType.SUM,
                        interval="5m",
                    )
                ],
                width=6,
                height=4,
                options={"unit": "errors"},
            ),
            ChartWidget(
                id="cache_hit_rate",
                title="キャッシュヒット率",
                chart_type=ChartType.GAUGE,
                queries=[
                    DashboardQuery(
                        metric_name="cache_hit_rate",
                        time_range=TimeRange.LAST_15M,
                        aggregation=AggregationType.AVERAGE,
                    )
                ],
                width=6,
                height=4,
                options={"unit": "%", "max": 100},
            ),
        ]

        return Dashboard(
            id="app_performance",
            title="アプリケーション性能",
            description="アプリケーションの性能監視ダッシュボード",
            widgets=widgets,
        )

    @staticmethod
    def create_ml_model_dashboard() -> Dashboard:
        """MLモデル監視ダッシュボードを作成"""
        widgets = [
            ChartWidget(
                id="prediction_latency",
                title="予測レイテンシー",
                chart_type=ChartType.HISTOGRAM,
                queries=[
                    DashboardQuery(
                        metric_name="model_prediction_duration_seconds",
                        time_range=TimeRange.LAST_1H,
                        interval="5m",
                    )
                ],
                width=8,
                height=4,
                options={"unit": "seconds"},
            ),
            ChartWidget(
                id="model_accuracy",
                title="モデル精度",
                chart_type=ChartType.LINE,
                queries=[
                    DashboardQuery(
                        metric_name="model_accuracy",
                        time_range=TimeRange.LAST_6H,
                        aggregation=AggregationType.AVERAGE,
                        interval="10m",
                    )
                ],
                width=6,
                height=4,
                options={"unit": "%", "max": 100},
            ),
            ChartWidget(
                id="prediction_count",
                title="予測実行数",
                chart_type=ChartType.BAR,
                queries=[
                    DashboardQuery(
                        metric_name="model_predictions_total",
                        time_range=TimeRange.LAST_24H,
                        aggregation=AggregationType.SUM,
                        interval="1h",
                    )
                ],
                width=6,
                height=4,
                options={"unit": "predictions"},
            ),
            ChartWidget(
                id="model_errors",
                title="モデルエラー",
                chart_type=ChartType.LINE,
                queries=[
                    DashboardQuery(
                        metric_name="model_errors_total",
                        time_range=TimeRange.LAST_6H,
                        aggregation=AggregationType.SUM,
                        interval="15m",
                    )
                ],
                width=6,
                height=4,
                options={"unit": "errors", "color": "red"},
            ),
        ]

        return Dashboard(
            id="ml_model_monitoring",
            title="MLモデル監視",
            description="機械学習モデルの性能監視ダッシュボード",
            widgets=widgets,
        )


class DashboardManager:
    """ダッシュボード管理"""

    def __init__(self, db_path: str = "dashboards.db"):
        self.db_path = db_path
        self.engine = DashboardEngine(get_metrics_system())
        self._initialize_database()
        self._load_default_dashboards()

    def _initialize_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dashboards (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    config TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            """
            )
            conn.commit()

    def _load_default_dashboards(self):
        """デフォルトダッシュボードをロード"""
        templates = DashboardTemplates()
        default_dashboards = [
            templates.create_system_overview_dashboard(),
            templates.create_application_performance_dashboard(),
            templates.create_ml_model_dashboard(),
        ]

        for dashboard in default_dashboards:
            self.engine.create_dashboard(dashboard)

    def save_dashboard(self, dashboard: Dashboard):
        """ダッシュボードを保存"""
        dashboard_config = {
            "id": dashboard.id,
            "title": dashboard.title,
            "description": dashboard.description,
            "refresh_interval": dashboard.refresh_interval,
            "widgets": [
                {
                    "id": widget.id,
                    "title": widget.title,
                    "chart_type": widget.chart_type.value,
                    "width": widget.width,
                    "height": widget.height,
                    "options": widget.options,
                    "queries": [
                        {
                            "metric_name": query.metric_name,
                            "time_range": query.time_range.value,
                            "aggregation": query.aggregation.value,
                            "labels": query.labels,
                            "group_by": query.group_by,
                            "interval": query.interval,
                        }
                        for query in widget.queries
                    ],
                }
                for widget in dashboard.widgets
            ],
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO dashboards
                (id, title, description, config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    dashboard.id,
                    dashboard.title,
                    dashboard.description,
                    json.dumps(dashboard_config, ensure_ascii=False),
                    dashboard.created_at.isoformat(),
                    dashboard.updated_at.isoformat(),
                ),
            )
            conn.commit()

        self.engine.create_dashboard(dashboard)

    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """ダッシュボードデータを取得"""
        return await self.engine.get_dashboard_data(dashboard_id)

    def list_dashboards(self) -> List[Dict[str, str]]:
        """ダッシュボード一覧を取得"""
        dashboards = []
        for dashboard in self.engine.list_dashboards():
            dashboards.append(
                {
                    "id": dashboard.id,
                    "title": dashboard.title,
                    "description": dashboard.description,
                    "created_at": dashboard.created_at.isoformat(),
                    "updated_at": dashboard.updated_at.isoformat(),
                }
            )
        return dashboards


# グローバルインスタンス
_dashboard_manager = None


def get_dashboard_manager() -> DashboardManager:
    """グローバルダッシュボード管理を取得"""
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = DashboardManager()
    return _dashboard_manager
