#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider V2 - Models and Data Classes
リアルデータプロバイダー V2 - モデルとデータクラス

データ構造、列挙型、設定クラス等を定義するモジュール
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


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


@dataclass
class ProviderStatistics:
    """プロバイダー統計情報"""
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    avg_quality: float = 0.0

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.requests == 0:
            return 0.0
        return (self.successes / self.requests) * 100

    @property
    def failure_rate(self) -> float:
        """失敗率"""
        if self.requests == 0:
            return 0.0
        return (self.failures / self.requests) * 100

    @property
    def avg_response_time(self) -> float:
        """平均応答時間"""
        if self.requests == 0:
            return 0.0
        return self.total_time / self.requests


@dataclass
class SourceStatus:
    """データソース状態"""
    enabled: bool
    priority: int
    daily_requests: int
    daily_limit: int
    requests_remaining: int
    success_rate: float
    avg_quality: float

    @property
    def utilization_rate(self) -> float:
        """利用率"""
        if self.daily_limit == 0:
            return 0.0
        return (self.daily_requests / self.daily_limit) * 100


# 定数定義
DEFAULT_PERIOD_DAYS = {
    '1d': 1,
    '5d': 5, 
    '1mo': 30,
    '3mo': 90,
    '6mo': 180,
    '1y': 365,
    '2y': 730,
    '5y': 1825
}

REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

COLUMN_MAPPING = {
    'Date': 'Date',
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume'
}

# Yahoo Finance用銘柄コードサフィックス
YAHOO_SUFFIXES = ['.T', '.JP', '.TO', '.TYO']

# Stooq用銘柄コードサフィックス  
STOOQ_SUFFIXES = ['.jp']