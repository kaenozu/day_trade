#!/usr/bin/env python3
"""
TOPIX500 Analysis System - Data Classes

TOPIX500分析に使用するデータクラスの定義
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TOPIX500Symbol:
    """TOPIX500銘柄情報"""

    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    weight_in_index: float
    listing_date: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    processing_time_seconds: float
    avg_time_per_symbol_ms: float
    peak_memory_mb: float
    cache_hit_rate: float
    throughput_symbols_per_second: float
    sector_count: int
    error_messages: List[str] = field(default_factory=list)


@dataclass
class SectorAnalysisResult:
    """セクター分析結果"""

    sector_name: str
    symbol_count: int
    symbols: List[str] = field(default_factory=list)
    avg_performance_score: float = 0.0
    sector_trend: str = "NEUTRAL"  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    sector_volatility: float = 0.0
    top_performers: List[str] = field(default_factory=list)
    sector_metrics: Dict[str, Any] = field(default_factory=dict)

    # 既存フィールドとの互換性
    sector: str = field(init=False)
    avg_volatility: float = field(init=False)
    avg_return: float = field(init=False)
    sector_momentum: float = field(init=False)
    risk_level: str = field(init=False)
    recommended_allocation: float = field(init=False)
    sector_rotation_signal: str = field(init=False)
    processing_time: float = field(init=False)

    def __post_init__(self):
        # 既存フィールドの値を設定
        self.sector = self.sector_name
        self.avg_volatility = self.sector_volatility
        self.avg_return = self.avg_performance_score
        self.sector_momentum = self.avg_performance_score
        self.risk_level = "medium"
        self.recommended_allocation = max(0.0, min(1.0, self.avg_performance_score))
        self.sector_rotation_signal = "neutral"
        self.processing_time = 0.0


@dataclass
class TOPIX500AnalysisResult:
    """TOPIX500総合分析結果"""

    analysis_timestamp: datetime
    total_symbols_analyzed: int
    successful_analyses: int
    failed_analyses: int
    sector_results: Dict[str, SectorAnalysisResult]
    top_recommendations: List[Dict[str, Any]]
    market_overview: Dict[str, float]
    risk_distribution: Dict[str, int]
    processing_performance: Dict[str, float]
    total_processing_time: float


@dataclass
class BatchProcessingTask:
    """バッチ処理タスク"""

    task_id: str
    symbols: List[str]
    analysis_types: List[str]
    priority: float = 1.0
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 2
