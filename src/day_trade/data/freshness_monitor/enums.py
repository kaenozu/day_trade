#!/usr/bin/env python3
"""
データ鮮度監視システム用列挙型定義
基本的な列挙型とステータス定義を提供します
"""

from enum import Enum


class DataSourceType(Enum):
    """データソース種別"""
    
    API = "api"
    DATABASE = "database"
    FILE = "file"
    STREAM = "stream"
    EXTERNAL_FEED = "external_feed"


class MonitoringLevel(Enum):
    """監視レベル"""
    
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """アラート重要度"""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """回復アクション"""
    
    RETRY = "retry"
    FALLBACK = "fallback"
    MANUAL = "manual"
    DISABLE = "disable"


try:
    from ...data.data_freshness_monitor import FreshnessStatus
except ImportError:
    # Fallback定義
    class FreshnessStatus(Enum):
        """データ鮮度ステータス"""
        
        FRESH = "fresh"
        STALE = "stale" 
        EXPIRED = "expired"