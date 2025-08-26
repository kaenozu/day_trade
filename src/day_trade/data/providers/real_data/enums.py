#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider Enums and Data Classes

データソースの列挙型とデータクラスの定義
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any
import pandas as pd


class DataSource(Enum):
    """データソース種別"""
    YAHOO_FINANCE = "yahoo_finance"
    STOOQ = "stooq"
    ALPHA_VANTAGE = "alpha_vantage"
    MOCK = "mock"


class DataQualityLevel(Enum):
    """データ品質レベル"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAILED = "failed"


@dataclass
class DataSourceConfig:
    """データソース設定"""
    name: str
    enabled: bool = True
    priority: int = 1
    timeout: int = 30
    retry_count: int = 3
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 1000
    quality_threshold: float = 70.0

    # API固有設定
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    # 品質設定
    min_data_points: int = 5
    max_price_threshold: float = 1000000
    price_consistency_check: bool = True

    # キャッシュ設定
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


@dataclass
class DataFetchResult:
    """データ取得結果"""
    data: Optional[pd.DataFrame]
    source: DataSource
    quality_level: DataQualityLevel
    quality_score: float
    fetch_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False

    @property
    def success(self) -> bool:
        """データ取得成功フラグ"""
        return self.data is not None and not self.data.empty