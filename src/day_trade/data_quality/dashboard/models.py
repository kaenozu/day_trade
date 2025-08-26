#!/usr/bin/env python3
"""
データ品質ダッシュボード - データクラス定義
Data class definitions for dashboard models
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from .enums import DashboardComponentType


@dataclass
class DashboardWidget:
    """ダッシュボードウィジェット"""

    widget_id: str
    title: str
    component_type: DashboardComponentType
    data_source: str
    refresh_interval: int = 300  # 秒
    position: Dict[str, int] = field(
        default_factory=lambda: {"x": 0, "y": 0, "w": 4, "h": 4}
    )
    config: Dict[str, Any] = field(default_factory=dict)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DashboardLayout:
    """ダッシュボードレイアウト"""

    layout_id: str
    name: str
    description: str
    widgets: List[DashboardWidget]
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_default: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityKPI:
    """品質KPI定義"""

    kpi_id: str
    name: str
    description: str
    target_value: float
    current_value: float
    unit: str
    trend: str = "stable"  # "up", "down", "stable"
    status: str = "healthy"  # "healthy", "warning", "critical"
    last_updated: datetime = field(default_factory=datetime.utcnow)