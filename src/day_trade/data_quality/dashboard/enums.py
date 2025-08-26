#!/usr/bin/env python3
"""
データ品質ダッシュボード - 列挙型定義
Enum definitions for data quality dashboard components
"""

from enum import Enum


class DashboardComponentType(Enum):
    """ダッシュボードコンポーネント種別"""

    METRIC_CARD = "metric_card"
    CHART = "chart"
    TABLE = "table"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TIMELINE = "timeline"
    GRAPH = "graph"
    ALERT_PANEL = "alert_panel"


class ChartType(Enum):
    """チャート種別"""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    BOX = "box"
    RADAR = "radar"