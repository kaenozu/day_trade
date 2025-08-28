#!/usr/bin/env python3
"""
セキュリティ管制センター - データモデル
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .enums import SecurityLevel, ThreatCategory, IncidentStatus


@dataclass
class SecurityThreat:
    """セキュリティ脅威"""

    id: str
    title: str
    description: str
    category: ThreatCategory
    severity: SecurityLevel
    source: str  # どこから検出されたか
    affected_assets: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


@dataclass
class SecurityIncident:
    """セキュリティインシデント"""

    id: str
    title: str
    description: str
    severity: SecurityLevel
    status: IncidentStatus = IncidentStatus.OPEN
    threats: List[SecurityThreat] = field(default_factory=list)
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    response_actions: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMetrics:
    """セキュリティメトリクス"""

    total_threats: int = 0
    critical_threats: int = 0
    high_threats: int = 0
    medium_threats: int = 0
    low_threats: int = 0
    open_incidents: int = 0
    resolved_incidents: int = 0
    mean_resolution_time: float = 0.0
    security_score: float = 100.0
    compliance_score: float = 100.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))