#!/usr/bin/env python3
"""
セキュリティ管制センター - 列挙型定義
"""

from enum import Enum


class SecurityLevel(Enum):
    """セキュリティレベル"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreatCategory(Enum):
    """脅威カテゴリ"""

    VULNERABILITY = "vulnerability"
    MALWARE = "malware"
    INTRUSION = "intrusion"
    DATA_BREACH = "data_breach"
    POLICY_VIOLATION = "policy_violation"
    ANOMALY = "anomaly"
    COMPLIANCE = "compliance"


class IncidentStatus(Enum):
    """インシデント状態"""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"