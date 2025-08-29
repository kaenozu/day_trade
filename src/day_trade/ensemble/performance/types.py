#!/usr/bin/env python3
"""
パフォーマンス分析用データクラス
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class AttributionResult:
    """貢献度分析結果"""
    model_attributions: Dict[str, np.ndarray]
    feature_attributions: Dict[str, np.ndarray]
    interaction_effects: Dict[str, np.ndarray]
    attribution_method: str
    confidence_intervals: Dict[str, Tuple[float, float]]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceDecomposition:
    """パフォーマンス分解結果"""
    total_performance: float
    individual_contributions: Dict[str, float]
    interaction_terms: Dict[str, float]
    residual_performance: float
    decomposition_method: str
    variance_explained: float
    component_rankings: List[Tuple[str, float]]

@dataclass
class AnalysisInsight:
    """分析洞察"""
    insight_type: str  # "performance", "attribution", "pattern", "anomaly"
    title: str
    description: str
    importance_score: float
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    timestamp: float
