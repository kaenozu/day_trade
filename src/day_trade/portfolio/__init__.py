#!/usr/bin/env python3
"""
マルチアセット・ポートフォリオ自動構築AIシステム
Issue #367 - 100+指標分析・AI駆動・企業レベル対応

Components:
- ai_portfolio_manager: AI駆動ポートフォリオマネージャー
- technical_indicators: 100+テクニカル指標分析エンジン
- automl_system: 機械学習自動化システム
- risk_parity_optimizer: リスクパリティ最適化
- style_analyzer: 投資スタイル分析システム
- multi_asset_allocator: マルチアセット配分最適化
"""

from .ai_portfolio_manager import (
    get_portfolio_manager,
    PortfolioManager,
    PortfolioConfig,
    AssetAllocation,
    PortfolioMetrics,
    OptimizationResults
)

from .technical_indicators import (
    get_indicator_engine,
    TechnicalIndicatorEngine,
    IndicatorConfig,
    IndicatorResult,
    SignalStrength
)

from .automl_system import (
    get_automl_system,
    AutoMLSystem,
    AutoMLConfig,
    ModelPerformance,
    HyperparameterResults
)

from .risk_parity_optimizer import (
    get_risk_parity_optimizer,
    RiskParityOptimizer,
    RiskParityConfig,
    RiskBudgetAllocation,
    OptimizationResult
)

from .style_analyzer import (
    get_style_analyzer,
    StyleAnalyzer,
    InvestmentStyle,
    StyleAnalysisResult,
    StyleConfiguration
)

# 利用可能性フラグ
PORTFOLIO_AI_AVAILABLE = True

try:
    # 必要な依存関係チェック
    import numpy as np
    import pandas as pd
    import scipy.optimize
    import sklearn.ensemble

except ImportError as e:
    PORTFOLIO_AI_AVAILABLE = False

__all__ = [
    # AI ポートフォリオマネージャー
    'get_portfolio_manager',
    'PortfolioManager',
    'PortfolioConfig',
    'AssetAllocation',
    'PortfolioMetrics',
    'OptimizationResults',

    # テクニカル指標エンジン
    'get_indicator_engine',
    'TechnicalIndicatorEngine',
    'IndicatorConfig',
    'IndicatorResult',
    'SignalStrength',

    # AutoMLシステム
    'get_automl_system',
    'AutoMLSystem',
    'AutoMLConfig',
    'ModelPerformance',
    'HyperparameterResults',

    # リスクパリティ最適化
    'get_risk_parity_optimizer',
    'RiskParityOptimizer',
    'RiskParityConfig',
    'RiskBudgetAllocation',
    'OptimizationResult',

    # スタイル分析
    'get_style_analyzer',
    'StyleAnalyzer',
    'InvestmentStyle',
    'StyleAnalysisResult',
    'StyleConfiguration',

    # 利用可能性フラグ
    'PORTFOLIO_AI_AVAILABLE'
]
