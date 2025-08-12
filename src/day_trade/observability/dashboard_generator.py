"""
APM・オブザーバビリティ統合基盤 - Issue #442 Phase 3
動的ダッシュボード生成・SLO可視化・インテリジェントアラート

このモジュールは以下を提供します:
- Grafanaダッシュボード動的生成
- SLO/SLI可視化
- HFT専用ダッシュボード
- カスタムメトリクス表示
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DashboardType(Enum):
    """ダッシュボードタイプ"""

    HFT_TRADING = "hft_trading"
    SYSTEM_OVERVIEW = "system_overview"
    SLO_MONITORING = "slo_monitoring"
    SECURITY_MONITORING = "security_monitoring"
    BUSINESS_METRICS = "business_metrics"
    ERROR_ANALYSIS = "error_analysis"


class PanelType(Enum):
    """パネルタイプ"""

    GRAPH = "graph"
    STAT = "stat"
    TABLE = "table"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    BAR_GAUGE = "bargauge"
    PIE_CHART = "piechart"
    ALERT_LIST = "alertlist"
    TEXT = "text"
    LOGS = "logs"
    NODE_GRAPH = "nodeGraph"


@dataclass
class PanelConfig:
    """パネル設定"""

    title: str
    panel_type: PanelType
    targets: List[Dict[str, Any]]
    x: int = 0
    y: int = 0
    width: int = 12
    height: int = 8
    datasource: str = "Prometheus"
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)
    transformations: List[Dict[str, Any]] = field(default_factory=list)


class DashboardGenerator:
    """
    動的ダッシュボード生成クラス

    Features:
    - テンプレートベース生成
    - SLO/SLI専用パネル
    - HFT最適化表示
    - リアルタイムアラート統合
    """

    def __init__(self, base_path: str = "/var/lib/grafana/dashboards"):
        self.base_path = base_path
        self.panel_id_counter = 1

    def _get_next_panel_id(self) -> int:
        """次のパネルID取得"""
        panel_id = self.panel_id_counter
        self.panel_id_counter += 1
        return panel_id

    def _create_base_dashboard(
        self,
        title: str,
        dashboard_type: DashboardType,
        tags: List[str] = None,
        refresh: str = "5s",
    ) -> Dict[str, Any]:
        """基本ダッシュボード構造作成"""
        return {
            "dashboard": {
                "id": None,
                "title": title,
                "tags": tags or [dashboard_type.value],
                "style": "dark",
                "timezone": "UTC",
                "refresh": refresh,
                "schemaVersion": 39,
                "version": 1,
                "time": {"from": "now-1h", "to": "now"},
                "timepicker": {
                    "refresh_intervals": [
                        "5s",
                        "10s",
                        "30s",
                        "1m",
                        "5m",
                        "15m",
                        "30m",
                        "1h",
                        "2h",
                        "1d",
                    ],
                    "time_options": [
                        "5m",
                        "15m",
                        "1h",
                        "6h",
                        "12h",
                        "24h",
                        "2d",
                        "7d",
                        "30d",
                    ],
                },
                "templating": {"list": self._create_template_variables()},
                "annotations": {"list": self._create_annotations()},
                "panels": [],
                "editable": True,
                "fiscalYearStartMonth": 0,
                "graphTooltip": 1,
                "links": [],
                "liveNow": True,  # HFT用リアルタイム表示
                "weekStart": "",
            },
            "overwrite": True,
        }

    def _create_template_variables(self) -> List[Dict[str, Any]]:
        """テンプレート変数作成"""
        return [
            {
                "name": "service",
                "type": "query",
                "datasource": "Prometheus",
                "query": "label_values(up, job)",
                "refresh": 1,
                "regex": "",
                "sort": 1,
                "multi": True,
                "includeAll": True,
                "allValue": ".*",
            },
            {
                "name": "instance",
                "type": "query",
                "datasource": "Prometheus",
                "query": 'label_values(up{job=~"$service"}, instance)',
                "refresh": 1,
                "regex": "",
                "sort": 1,
                "multi": True,
                "includeAll": True,
                "allValue": ".*",
            },
            {
                "name": "symbol",
                "type": "query",
                "datasource": "Prometheus",
                "query": "label_values(day_trade_trades_total, symbol)",
                "refresh": 1,
                "regex": "",
                "sort": 1,
                "multi": True,
                "includeAll": True,
                "allValue": ".*",
            },
            {
                "name": "timerange",
                "type": "interval",
                "query": "5s,10s,30s,1m,5m,15m,30m,1h",
                "current": {"text": "30s", "value": "30s"},
            },
        ]

    def _create_annotations(self) -> List[Dict[str, Any]]:
        """アノテーション設定作成"""
        return [
            {
                "name": "Alerts",
                "datasource": "AlertManager",
                "enable": True,
                "iconColor": "red",
                "query": "",
                "showIn": 0,
                "tags": [],
                "type": "dashboard",
            },
            {
                "name": "Deployments",
                "datasource": "Prometheus",
                "enable": True,
                "expr": "changes(up[5m]) > 0",
                "iconColor": "green",
                "showIn": 0,
                "step": "1m",
                "tags": ["deployment"],
                "titleFormat": "Deployment",
                "type": "prometheus",
            },
        ]

    def _create_panel(self, config: PanelConfig) -> Dict[str, Any]:
        """パネル作成"""
        panel = {
            "id": self._get_next_panel_id(),
            "title": config.title,
            "type": config.panel_type.value,
            "datasource": {
                "type": (
                    "prometheus"
                    if config.datasource == "Prometheus"
                    else config.datasource.lower()
                ),
                "uid": config.datasource.lower(),
            },
            "gridPos": {
                "h": config.height,
                "w": config.width,
                "x": config.x,
                "y": config.y,
            },
            "targets": config.targets,
            "options": config.options,
            "fieldConfig": config.field_config,
            "transformations": config.transformations,
        }

        return panel

    # === HFTダッシュボード生成 ===

    def create_hft_dashboard(self) -> Dict[str, Any]:
        """HFT専用ダッシュボード作成"""
        dashboard = self._create_base_dashboard(
            "🚀 HFT Trading Performance",
            DashboardType.HFT_TRADING,
            ["hft", "trading", "performance"],
            "1s",  # 1秒更新
        )

        panels = []
        y_pos = 0

        # === トップ統計パネル ===
        stats_panels = [
            # 取引レイテンシ (P99)
            PanelConfig(
                title="Trade Latency P99",
                panel_type=PanelType.STAT,
                targets=[
                    {
                        "expr": "histogram_quantile(0.99, rate(day_trade_trade_latency_microseconds_bucket[30s]))",
                        "legendFormat": "P99 Latency",
                        "refId": "A",
                    }
                ],
                x=0,
                y=y_pos,
                width=6,
                height=4,
                options={
                    "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                    "orientation": "auto",
                    "textMode": "auto",
                    "colorMode": "background",
                    "graphMode": "area",
                    "justifyMode": "auto",
                },
                field_config={
                    "defaults": {
                        "unit": "µs",
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 20},
                                {"color": "red", "value": 50},
                            ]
                        },
                    }
                },
            ),
            # 取引成功率
            PanelConfig(
                title="Trade Success Rate",
                panel_type=PanelType.STAT,
                targets=[
                    {
                        "expr": '(rate(day_trade_trades_total{status="success"}[1m]) / rate(day_trade_trades_total[1m])) * 100',
                        "legendFormat": "Success Rate",
                        "refId": "A",
                    }
                ],
                x=6,
                y=y_pos,
                width=6,
                height=4,
                options={
                    "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                    "colorMode": "background",
                },
                field_config={
                    "defaults": {
                        "unit": "percent",
                        "min": 95,
                        "max": 100,
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 99.9},
                                {"color": "green", "value": 99.95},
                            ]
                        },
                    }
                },
            ),
            # スループット
            PanelConfig(
                title="Trading Throughput",
                panel_type=PanelType.STAT,
                targets=[
                    {
                        "expr": "rate(day_trade_trades_total[30s])",
                        "legendFormat": "Trades/sec",
                        "refId": "A",
                    }
                ],
                x=12,
                y=y_pos,
                width=6,
                height=4,
                options={"colorMode": "background"},
                field_config={
                    "defaults": {
                        "unit": "ops",
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 100},
                                {"color": "green", "value": 1000},
                            ]
                        },
                    }
                },
            ),
            # P&L
            PanelConfig(
                title="Current P&L",
                panel_type=PanelType.STAT,
                targets=[
                    {
                        "expr": "increase(day_trade_pnl_dollars_sum[5m])",
                        "legendFormat": "P&L (5m)",
                        "refId": "A",
                    }
                ],
                x=18,
                y=y_pos,
                width=6,
                height=4,
                options={"colorMode": "background"},
                field_config={
                    "defaults": {
                        "unit": "currencyUSD",
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": -1000},
                                {"color": "yellow", "value": 0},
                                {"color": "green", "value": 1000},
                            ]
                        },
                    }
                },
            ),
        ]

        panels.extend([self._create_panel(panel) for panel in stats_panels])
        y_pos += 4

        # === レイテンシ詳細分析 ===
        latency_panels = [
            PanelConfig(
                title="Trade Execution Latency Distribution",
                panel_type=PanelType.HEATMAP,
                targets=[
                    {
                        "expr": "rate(day_trade_trade_latency_microseconds_bucket[1m])",
                        "format": "heatmap",
                        "legendFormat": "{{le}}",
                        "refId": "A",
                    }
                ],
                x=0,
                y=y_pos,
                width=12,
                height=8,
                options={
                    "yAxis": {"unit": "µs"},
                    "color": {"mode": "spectrum"},
                    "calculate": True,
                },
            ),
            PanelConfig(
                title="Latency Percentiles",
                panel_type=PanelType.GRAPH,
                targets=[
                    {
                        "expr": "histogram_quantile(0.50, rate(day_trade_trade_latency_microseconds_bucket[30s]))",
                        "legendFormat": "P50",
                        "refId": "A",
                    },
                    {
                        "expr": "histogram_quantile(0.95, rate(day_trade_trade_latency_microseconds_bucket[30s]))",
                        "legendFormat": "P95",
                        "refId": "B",
                    },
                    {
                        "expr": "histogram_quantile(0.99, rate(day_trade_trade_latency_microseconds_bucket[30s]))",
                        "legendFormat": "P99",
                        "refId": "C",
                    },
                    {
                        "expr": "histogram_quantile(0.999, rate(day_trade_trade_latency_microseconds_bucket[30s]))",
                        "legendFormat": "P99.9",
                        "refId": "D",
                    },
                ],
                x=12,
                y=y_pos,
                width=12,
                height=8,
                field_config={
                    "defaults": {
                        "unit": "µs",
                        "custom": {
                            "drawStyle": "line",
                            "lineWidth": 2,
                            "fillOpacity": 10,
                        },
                    }
                },
            ),
        ]

        panels.extend([self._create_panel(panel) for panel in latency_panels])
        y_pos += 8

        # === システムリソース監視 ===
        system_panels = [
            PanelConfig(
                title="CPU Usage",
                panel_type=PanelType.GRAPH,
                targets=[
                    {
                        "expr": '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle", instance=~"$instance"}[5m])) * 100)',
                        "legendFormat": "CPU {{instance}}",
                        "refId": "A",
                    }
                ],
                x=0,
                y=y_pos,
                width=8,
                height=6,
                field_config={"defaults": {"unit": "percent", "min": 0, "max": 100}},
            ),
            PanelConfig(
                title="Memory Usage",
                panel_type=PanelType.GRAPH,
                targets=[
                    {
                        "expr": "((node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes) * 100",
                        "legendFormat": "Memory {{instance}}",
                        "refId": "A",
                    }
                ],
                x=8,
                y=y_pos,
                width=8,
                height=6,
                field_config={"defaults": {"unit": "percent", "min": 0, "max": 100}},
            ),
            PanelConfig(
                title="Network I/O",
                panel_type=PanelType.GRAPH,
                targets=[
                    {
                        "expr": 'rate(node_network_receive_bytes_total{instance=~"$instance"}[5m])',
                        "legendFormat": "RX {{device}}",
                        "refId": "A",
                    },
                    {
                        "expr": '-rate(node_network_transmit_bytes_total{instance=~"$instance"}[5m])',
                        "legendFormat": "TX {{device}}",
                        "refId": "B",
                    },
                ],
                x=16,
                y=y_pos,
                width=8,
                height=6,
                field_config={"defaults": {"unit": "Bps"}},
            ),
        ]

        panels.extend([self._create_panel(panel) for panel in system_panels])

        dashboard["dashboard"]["panels"] = panels
        return dashboard

    # === SLOダッシュボード生成 ===

    def create_slo_dashboard(self) -> Dict[str, Any]:
        """SLO/SLI監視ダッシュボード作成"""
        dashboard = self._create_base_dashboard(
            "📊 SLO/SLI Monitoring",
            DashboardType.SLO_MONITORING,
            ["slo", "sli", "monitoring"],
            "30s",
        )

        panels = []
        y_pos = 0

        # SLO概要統計
        slo_stats = [
            PanelConfig(
                title="Trade Latency SLO",
                panel_type=PanelType.GAUGE,
                targets=[
                    {
                        "expr": '(rate(day_trade_trades_total{latency_bucket="le_50"}[5m]) / rate(day_trade_trades_total[5m])) * 100',
                        "refId": "A",
                    }
                ],
                x=0,
                y=y_pos,
                width=6,
                height=6,
                options={
                    "reduceOptions": {"calcs": ["lastNotNull"]},
                    "orientation": "auto",
                    "displayMode": "basic",
                },
                field_config={
                    "defaults": {
                        "unit": "percent",
                        "min": 99,
                        "max": 100,
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 99.9},
                                {"color": "green", "value": 99.95},
                            ]
                        },
                    }
                },
            )
        ]

        panels.extend([self._create_panel(panel) for panel in slo_stats])

        dashboard["dashboard"]["panels"] = panels
        return dashboard

    # === セキュリティダッシュボード生成 ===

    def create_security_dashboard(self) -> Dict[str, Any]:
        """セキュリティ監視ダッシュボード作成"""
        dashboard = self._create_base_dashboard(
            "🛡️ Security Monitoring",
            DashboardType.SECURITY_MONITORING,
            ["security", "monitoring"],
            "15s",
        )

        panels = []
        y_pos = 0

        # セキュリティメトリクス
        security_panels = [
            PanelConfig(
                title="Authentication Failures",
                panel_type=PanelType.GRAPH,
                targets=[
                    {
                        "expr": "rate(day_trade_auth_failures_total[5m])",
                        "legendFormat": "Auth failures/sec",
                        "refId": "A",
                    }
                ],
                x=0,
                y=y_pos,
                width=12,
                height=6,
                field_config={"defaults": {"unit": "ops"}},
            ),
            PanelConfig(
                title="Suspicious Activities",
                panel_type=PanelType.TABLE,
                targets=[
                    {
                        "expr": 'topk(10, rate(day_trade_requests_total{status=~"4.."}[5m]))',
                        "format": "table",
                        "instant": True,
                        "refId": "A",
                    }
                ],
                x=12,
                y=y_pos,
                width=12,
                height=6,
            ),
        ]

        panels.extend([self._create_panel(panel) for panel in security_panels])

        dashboard["dashboard"]["panels"] = panels
        return dashboard

    # === ダッシュボード保存 ===

    def save_dashboard(self, dashboard: Dict[str, Any], filename: str) -> str:
        """ダッシュボード保存"""
        filepath = os.path.join(self.base_path, f"{filename}.json")

        # ディレクトリ作成
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # メタデータ追加
        dashboard["dashboard"]["meta"] = {
            "type": "db",
            "canSave": True,
            "canEdit": True,
            "canAdmin": True,
            "canStar": True,
            "slug": filename.replace("_", "-"),
            "expires": "0001-01-01T00:00:00Z",
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
            "updatedBy": "admin",
            "createdBy": "dashboard-generator",
            "version": 1,
        }

        # JSON保存
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)

        return filepath

    def generate_all_dashboards(self) -> List[str]:
        """全ダッシュボード生成"""
        generated_files = []

        # HFTダッシュボード
        hft_dashboard = self.create_hft_dashboard()
        hft_path = self.save_dashboard(hft_dashboard, "hft_trading_performance")
        generated_files.append(hft_path)

        # SLOダッシュボード
        slo_dashboard = self.create_slo_dashboard()
        slo_path = self.save_dashboard(slo_dashboard, "slo_monitoring")
        generated_files.append(slo_path)

        # セキュリティダッシュボード
        security_dashboard = self.create_security_dashboard()
        security_path = self.save_dashboard(security_dashboard, "security_monitoring")
        generated_files.append(security_path)

        return generated_files


# 便利関数
def generate_dashboards(output_dir: str = None) -> List[str]:
    """ダッシュボード一括生成"""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "config",
            "grafana",
            "dashboards",
        )

    generator = DashboardGenerator(output_dir)
    return generator.generate_all_dashboards()


if __name__ == "__main__":
    # 実行例
    generated = generate_dashboards()
    print(f"Generated {len(generated)} dashboards:")
    for path in generated:
        print(f"  - {path}")
