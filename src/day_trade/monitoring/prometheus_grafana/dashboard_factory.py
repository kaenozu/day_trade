"""
Grafanaダッシュボードファクトリー

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

Grafanaダッシュボードとパネルの作成機能を提供する。
"""

from typing import Any, Dict

from .configs import GrafanaDashboard


class DashboardFactory:
    """ダッシュボードファクトリー"""

    @staticmethod
    def create_system_overview_dashboard() -> GrafanaDashboard:
        """システム概要ダッシュボード作成"""
        return GrafanaDashboard(
            title="Day Trade System Overview",
            description="システム全体の概要監視",
            tags=["day-trade", "overview", "system"],
            time_range="1h",
            refresh_interval="30s",
            panels=[
                DashboardFactory._create_api_requests_panel(),
                DashboardFactory._create_response_time_panel(),
                DashboardFactory._create_active_trades_panel(),
                DashboardFactory._create_portfolio_value_panel(),
                DashboardFactory._create_ml_accuracy_panel(),
                DashboardFactory._create_security_events_panel(),
                DashboardFactory._create_log_levels_panel(),
                DashboardFactory._create_cache_performance_panel(),
            ],
        )

    @staticmethod
    def create_performance_dashboard() -> GrafanaDashboard:
        """パフォーマンス詳細ダッシュボード作成"""
        return GrafanaDashboard(
            title="Day Trade Performance Details",
            description="パフォーマンス詳細監視",
            tags=["day-trade", "performance", "details"],
            time_range="6h",
            refresh_interval="15s",
            panels=[
                DashboardFactory._create_request_rate_panel(),
                DashboardFactory._create_error_rate_panel(),
                DashboardFactory._create_latency_percentiles_panel(),
                DashboardFactory._create_throughput_panel(),
            ],
        )

    @staticmethod
    def _create_api_requests_panel() -> Dict[str, Any]:
        """APIリクエストパネル作成"""
        return {
            "id": 1,
            "title": "API Requests Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(day_trade_api_requests_total[5m])",
                    "legendFormat": "{{method}} {{endpoint}}",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Requests/sec", "min": 0}],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        }

    @staticmethod
    def _create_response_time_panel() -> Dict[str, Any]:
        """応答時間パネル作成"""
        return {
            "id": 2,
            "title": "API Response Time (95th percentile)",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(day_trade_api_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Seconds", "min": 0}],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        }

    @staticmethod
    def _create_active_trades_panel() -> Dict[str, Any]:
        """アクティブ取引パネル作成"""
        return {
            "id": 3,
            "title": "Active Trades",
            "type": "singlestat",
            "targets": [{"expr": "day_trade_active_trades", "refId": "A"}],
            "valueName": "current",
            "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
        }

    @staticmethod
    def _create_portfolio_value_panel() -> Dict[str, Any]:
        """ポートフォリオ価値パネル作成"""
        return {
            "id": 4,
            "title": "Portfolio Value (JPY)",
            "type": "singlestat",
            "targets": [{"expr": "day_trade_portfolio_value_jpy", "refId": "A"}],
            "valueName": "current",
            "format": "currencyJPY",
            "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
        }

    @staticmethod
    def _create_ml_accuracy_panel() -> Dict[str, Any]:
        """ML予測精度パネル作成"""
        return {
            "id": 5,
            "title": "ML Prediction Accuracy",
            "type": "gauge",
            "targets": [
                {"expr": "avg(day_trade_ml_prediction_accuracy)", "refId": "A"}
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 0.7},
                            {"color": "green", "value": 0.8},
                        ]
                    },
                }
            },
            "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8},
        }

    @staticmethod
    def _create_security_events_panel() -> Dict[str, Any]:
        """セキュリティイベントパネル作成"""
        return {
            "id": 6,
            "title": "Security Events Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(day_trade_security_events_total[5m])",
                    "legendFormat": "{{severity}} {{component}}",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Events/sec", "min": 0}],
            "gridPos": {"h": 4, "w": 6, "x": 18, "y": 8},
        }

    @staticmethod
    def _create_log_levels_panel() -> Dict[str, Any]:
        """ログレベルパネル作成"""
        return {
            "id": 7,
            "title": "Log Levels Distribution",
            "type": "piechart",
            "targets": [
                {
                    "expr": "sum by (level) (rate(day_trade_log_entries_total[5m]))",
                    "legendFormat": "{{level}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
        }

    @staticmethod
    def _create_cache_performance_panel() -> Dict[str, Any]:
        """キャッシュパフォーマンスパネル作成"""
        return {
            "id": 8,
            "title": "Cache Hit Ratio",
            "type": "graph",
            "targets": [
                {
                    "expr": "day_trade_cache_hit_ratio",
                    "legendFormat": "{{cache_type}} {{cache_level}}",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Hit Ratio", "min": 0, "max": 1}],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
        }

    @staticmethod
    def _create_request_rate_panel() -> Dict[str, Any]:
        """リクエスト率パネル作成"""
        return {
            "id": 9,
            "title": "Request Rate by Endpoint",
            "type": "graph",
            "targets": [
                {
                    "expr": "sum by (endpoint) (rate(day_trade_api_requests_total[5m]))",
                    "legendFormat": "{{endpoint}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        }

    @staticmethod
    def _create_error_rate_panel() -> Dict[str, Any]:
        """エラー率パネル作成"""
        return {
            "id": 10,
            "title": "Error Rate by Status Code",
            "type": "graph",
            "targets": [
                {
                    "expr": "sum by (status) (rate(day_trade_api_requests_total{status=~'[4-5]..'}[5m]))",
                    "legendFormat": "{{status}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        }

    @staticmethod
    def _create_latency_percentiles_panel() -> Dict[str, Any]:
        """レイテンシパーセンタイルパネル作成"""
        return {
            "id": 11,
            "title": "Response Time Percentiles",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.50, rate(day_trade_api_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile",
                    "refId": "A",
                },
                {
                    "expr": "histogram_quantile(0.95, rate(day_trade_api_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile",
                    "refId": "B",
                },
                {
                    "expr": "histogram_quantile(0.99, rate(day_trade_api_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "99th percentile",
                    "refId": "C",
                },
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        }

    @staticmethod
    def _create_throughput_panel() -> Dict[str, Any]:
        """スループットパネル作成"""
        return {
            "id": 12,
            "title": "System Throughput",
            "type": "graph",
            "targets": [
                {
                    "expr": "sum(rate(day_trade_api_requests_total[5m]))",
                    "legendFormat": "Total Throughput",
                    "refId": "A",
                }
            ],
            "yAxes": [{"label": "Requests/sec", "min": 0}],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        }