#!/usr/bin/env python3
"""
データ品質ダッシュボード - パッケージ初期化
Package initialization with backward compatibility
"""

# Core imports for backward compatibility
from .dashboard_core import DataQualityDashboard
from .enums import ChartType, DashboardComponentType
from .models import DashboardLayout, DashboardWidget, QualityKPI

# Factory function for backward compatibility
def create_data_quality_dashboard(
    storage_path: str = "data/dashboard",
    enable_cache: bool = True,
    refresh_interval_seconds: int = 300,
    retention_days: int = 90,
) -> DataQualityDashboard:
    """統合データ品質ダッシュボード作成"""
    return DataQualityDashboard(
        storage_path=storage_path,
        enable_cache=enable_cache,
        refresh_interval_seconds=refresh_interval_seconds,
        retention_days=retention_days,
    )


# All exports for backward compatibility
__all__ = [
    # Main classes
    "DataQualityDashboard",
    
    # Model classes
    "DashboardWidget",
    "DashboardLayout", 
    "QualityKPI",
    
    # Enums
    "DashboardComponentType",
    "ChartType",
    
    # Factory function
    "create_data_quality_dashboard",
]