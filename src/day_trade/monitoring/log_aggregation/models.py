#!/usr/bin/env python3
"""
ログ集約システムのデータモデル定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import AlertSeverity, LogLevel, LogSource


@dataclass
class LogEntry:
    """ログエントリ"""

    id: str
    timestamp: datetime
    level: LogLevel
    source: LogSource
    component: str
    message: str
    structured_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    raw_log: str = ""
    parsed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LogPattern:
    """ログパターン定義"""

    pattern_id: str
    name: str
    description: str
    regex_pattern: str
    source_filter: Optional[LogSource] = None
    level_filter: Optional[LogLevel] = None
    alert_threshold: int = 10  # この回数検出されたらアラート
    time_window_minutes: int = 5
    severity: AlertSeverity = AlertSeverity.MEDIUM
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LogAlert:
    """ログアラート"""

    alert_id: str
    pattern_id: str
    pattern_name: str
    severity: AlertSeverity
    message: str
    occurrence_count: int
    first_occurrence: datetime
    last_occurrence: datetime
    related_logs: List[str]  # ログIDリスト
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class LogSearchQuery:
    """ログ検索クエリ"""

    query_text: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: List[LogLevel] = field(default_factory=list)
    sources: List[LogSource] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    limit: int = 1000
    order_by: str = "timestamp"
    order_desc: bool = True