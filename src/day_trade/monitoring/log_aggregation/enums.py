#!/usr/bin/env python3
"""
ログ集約システムの列挙型とデータクラス定義
"""

from enum import Enum


class LogLevel(Enum):
    """ログレベル"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogSource(Enum):
    """ログソース種別"""

    APPLICATION = "application"
    SYSTEM = "system"
    DATABASE = "database"
    API = "api"
    TRADING = "trading"
    ML = "ml"
    SECURITY = "security"
    PERFORMANCE = "performance"


class AlertSeverity(Enum):
    """アラート重要度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"