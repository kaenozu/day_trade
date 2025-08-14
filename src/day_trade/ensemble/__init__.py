#!/usr/bin/env python3
"""
高度なアンサンブル予測システム
Advanced Ensemble Prediction System

Issue #762: 高度なアンサンブル予測システムの強化
"""

from .adaptive_weighting import AdaptiveWeightingEngine, MarketRegimeDetector, PerformanceTracker, WeightOptimizer
from .meta_learning import MetaLearnerEngine, ModelKnowledgeTransfer, FewShotLearning, ContinualLearning
from .ensemble_optimizer import EnsembleOptimizer, ArchitectureSearch, HyperparameterTuning, AutoMLIntegration
from .performance_analyzer import EnsembleAnalyzer, AttributionEngine, PerformanceDecomposer, VisualDashboard
from .advanced_ensemble import AdvancedEnsembleSystem

__all__ = [
    # Dynamic Weighting
    'AdaptiveWeightingEngine',
    'MarketRegimeDetector',
    'PerformanceTracker',
    'WeightOptimizer',

    # Meta Learning
    'MetaLearnerEngine',
    'ModelKnowledgeTransfer',
    'FewShotLearning',
    'ContinualLearning',

    # Ensemble Optimization
    'EnsembleOptimizer',
    'ArchitectureSearch',
    'HyperparameterTuning',
    'AutoMLIntegration',

    # Performance Analysis
    'EnsembleAnalyzer',
    'AttributionEngine',
    'PerformanceDecomposer',
    'VisualDashboard',

    # Integrated System
    'AdvancedEnsembleSystem',
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Claude AI Assistant"
__description__ = "Advanced Ensemble Prediction System for Day Trade ML"

# 設定
DEFAULT_CONFIG = {
    "adaptive_weighting": {
        "lookback_window": 252,
        "regime_threshold": 0.05,
        "weight_momentum": 0.9,
        "min_weight": 0.01,
        "max_weight": 0.5
    },
    "meta_learning": {
        "inner_lr": 0.01,
        "outer_lr": 0.001,
        "adaptation_steps": 5,
        "meta_batch_size": 16,
        "support_shots": 5,
        "query_shots": 10
    },
    "ensemble_optimization": {
        "optimization_budget": 100,
        "population_size": 50,
        "max_generations": 20,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8
    },
    "performance_analysis": {
        "attribution_method": "shap",
        "decomposition_depth": 3,
        "visualization_format": "html",
        "dashboard_refresh_rate": 30
    }
}

def create_advanced_ensemble_system(
    models=None,
    config=None,
    enable_adaptive_weighting=True,
    enable_meta_learning=True,
    enable_optimization=True,
    enable_analysis=True
):
    """
    高度なアンサンブルシステム作成

    Args:
        models: ベースモデルのリスト
        config: システム設定
        enable_adaptive_weighting: 動的重み付け有効化
        enable_meta_learning: メタ学習有効化
        enable_optimization: アンサンブル最適化有効化
        enable_analysis: パフォーマンス分析有効化

    Returns:
        AdvancedEnsembleSystem: 設定済みシステム
    """
    if config is None:
        config = DEFAULT_CONFIG

    return AdvancedEnsembleSystem(
        models=models,
        config=config,
        enable_adaptive_weighting=enable_adaptive_weighting,
        enable_meta_learning=enable_meta_learning,
        enable_optimization=enable_optimization,
        enable_analysis=enable_analysis
    )

def get_system_info():
    """システム情報取得"""
    return {
        "name": "Advanced Ensemble Prediction System",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": len(__all__),
        "config_keys": list(DEFAULT_CONFIG.keys())
    }