#!/usr/bin/env python3
"""
多角的データ収集システム - データモデル定義

Issue #322: ML Data Shortage Problem Resolution
データ構造とモデルクラスの定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class DataSource:
    """データソース定義"""

    name: str
    type: str  # 'price', 'news', 'sentiment', 'macro', 'fundamental'
    priority: int  # 1(最高) - 5(最低)
    api_url: str
    rate_limit: int = 60  # requests/minute
    timeout: int = 30
    retry_count: int = 3
    last_request_time: float = 0.0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectedData:
    """収集データコンテナ"""

    symbol: str
    data_type: str
    data: Any
    source: str
    timestamp: datetime
    quality_score: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """データ品質レポート"""

    completeness: float  # 完全性 0-1
    consistency: float  # 一貫性 0-1
    accuracy: float  # 正確性 0-1
    timeliness: float  # 時間性 0-1
    validity: float  # 有効性 0-1
    overall_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)