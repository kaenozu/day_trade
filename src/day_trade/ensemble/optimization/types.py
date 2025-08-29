#!/usr/bin/env python3
"""
アンサンブル最適化のためのデータクラス
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class EnsembleConfiguration:
    """アンサンブル構成"""
    model_selection: List[str]
    weights: Dict[str, float]
    hyperparameters: Dict[str, Dict[str, Any]]
    aggregation_method: str = "weighted_average"
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    estimated_performance: float = 0.0
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class OptimizationResult:
    """最適化結果"""
    best_configuration: EnsembleConfiguration
    optimization_score: float
    convergence_history: List[float]
    total_evaluations: int
    optimization_time: float
    method_used: str
    cross_validation_scores: Optional[List[float]] = None

@dataclass
class ModelCandidate:
    """モデル候補"""
    model_id: str
    model_class: str
    hyperparameter_space: Dict[str, Any]
    base_performance: float = 0.0
    complexity_score: float = 1.0
    is_enabled: bool = True
