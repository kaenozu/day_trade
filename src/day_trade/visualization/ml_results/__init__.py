#!/usr/bin/env python3
"""
機械学習結果可視化システム - パッケージ初期化

Issue #315: 高度テクニカル指標・ML機能拡張
分割されたモジュールの統合とバックワード互換性の提供
"""

import warnings
from typing import Dict, Optional

# バージョン情報
__version__ = "1.0.0"
__author__ = "ML Results Visualization System"
__description__ = "機械学習結果可視化システム統合パッケージ"

# 警告の抑制
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 基本型・設定のインポート
from .types import (
    # 列挙型
    RiskLevel,
    TradingAction,
    SignalStrength,
    PositionSize,
    TrendDirection,
    ChartType,
    
    # データクラス
    ColorPalette,
    LSTMResults,
    CurrentMetrics,
    EnsembleForecast,
    RiskAssessment,
    VolatilityResults,
    TechnicalIndicators,
    TimeframeData,
    IntegratedSignal,
    InvestmentRecommendation,
    IntegratedAnalysis,
    MultiFrameResults,
    VisualizationConfig,
    ChartMetadata,
    DependencyStatus,
    
    # デフォルト設定・定数
    DEFAULT_CONFIG,
    ERROR_MESSAGES,
    JAPANESE_LABELS,
    TIMEFRAMES,
    CHART_SETTINGS,
    
    # ユーティリティ関数
    validate_lstm_results,
    validate_volatility_results,
    validate_multiframe_results,
    get_risk_color,
    get_signal_color,
)

# データ処理モジュール
from .data_processor import DataProcessor

# チャート生成モジュール
from .chart_generators import (
    BaseChartGenerator,
    LSTMChartGenerator,
    VolatilityChartGenerator,
)

# UI コンポーネント
from .ui_components import (
    InteractiveDashboard,
    AdvancedChartGenerator,
)

# レポート生成
from .report_builder import ReportBuilder

# メインビジュアライザー
from .main_visualizer import MLResultsVisualizer

# バックワード互換性のためのエイリアス
# 元の ml_results_visualizer.py からインポートしていたクラス
Visualizer = MLResultsVisualizer

# ロギング設定
import logging
logger = logging.getLogger(__name__)

# 依存関係チェック
_dependency_status = None

def get_dependency_status() -> DependencyStatus:
    """
    依存パッケージの状況を取得
    
    Returns:
        依存関係状況
    """
    global _dependency_status
    
    if _dependency_status is None:
        _dependency_status = DependencyStatus()
        
        # matplotlib チェック
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            _dependency_status.matplotlib_available = True
            _dependency_status.seaborn_available = True
        except ImportError:
            pass
        
        # plotly チェック
        try:
            import plotly.graph_objects as go
            _dependency_status.plotly_available = True
        except ImportError:
            pass
    
    return _dependency_status


def create_visualizer(output_dir: str = "output/ml_visualizations") -> MLResultsVisualizer:
    """
    機械学習結果可視化システムのインスタンス作成
    
    Args:
        output_dir: 出力ディレクトリ
        
    Returns:
        MLResultsVisualizerインスタンス
    """
    return MLResultsVisualizer(output_dir)


def check_dependencies() -> Dict[str, bool]:
    """
    依存関係チェック結果を辞書形式で返す
    
    Returns:
        依存関係チェック結果辞書
    """
    status = get_dependency_status()
    return {
        'matplotlib': status.matplotlib_available,
        'seaborn': status.seaborn_available,
        'plotly': status.plotly_available,
        'can_create_static_charts': status.can_create_static_charts,
        'can_create_interactive_charts': status.can_create_interactive_charts,
        'all_available': status.all_available
    }


# バックワード互換性関数群
# 元のml_results_visualizer.pyの関数と同じインターフェースを提供

def create_lstm_prediction_chart(
    data,
    lstm_results: Dict,
    symbol: str = "Unknown",
    save_path: Optional[str] = None,
    output_dir: str = "output/ml_visualizations"
) -> Optional[str]:
    """
    LSTM予測チャート作成（バックワード互換性）
    
    Args:
        data: 価格データ
        lstm_results: LSTM予測結果
        symbol: 銘柄コード
        save_path: 保存パス
        output_dir: 出力ディレクトリ
        
    Returns:
        保存されたファイルパス
    """
    visualizer = create_visualizer(output_dir)
    return visualizer.create_lstm_prediction_chart(
        data, lstm_results, symbol, save_path
    )


def create_volatility_forecast_chart(
    data,
    volatility_results: Dict,
    symbol: str = "Unknown", 
    save_path: Optional[str] = None,
    output_dir: str = "output/ml_visualizations"
) -> Optional[str]:
    """
    ボラティリティ予測チャート作成（バックワード互換性）
    
    Args:
        data: 価格データ
        volatility_results: ボラティリティ予測結果
        symbol: 銘柄コード
        save_path: 保存パス
        output_dir: 出力ディレクトリ
        
    Returns:
        保存されたファイルパス
    """
    visualizer = create_visualizer(output_dir)
    return visualizer.create_volatility_forecast_chart(
        data, volatility_results, symbol, save_path
    )


def create_multiframe_analysis_chart(
    data,
    multiframe_results: Dict,
    symbol: str = "Unknown",
    save_path: Optional[str] = None,
    output_dir: str = "output/ml_visualizations"
) -> Optional[str]:
    """
    マルチタイムフレーム分析チャート作成（バックワード互換性）
    
    Args:
        data: 価格データ
        multiframe_results: マルチタイムフレーム分析結果
        symbol: 銘柄コード
        save_path: 保存パス
        output_dir: 出力ディレクトリ
        
    Returns:
        保存されたファイルパス
    """
    visualizer = create_visualizer(output_dir)
    return visualizer.create_multiframe_analysis_chart(
        data, multiframe_results, symbol, save_path
    )


def create_comprehensive_dashboard(
    data,
    lstm_results: Optional[Dict] = None,
    volatility_results: Optional[Dict] = None,
    multiframe_results: Optional[Dict] = None,
    technical_results: Optional[Dict] = None,
    symbol: str = "Unknown",
    save_path: Optional[str] = None,
    output_dir: str = "output/ml_visualizations"
) -> Optional[str]:
    """
    総合ダッシュボード作成（バックワード互換性）
    
    Args:
        data: 価格データ
        lstm_results: LSTM予測結果
        volatility_results: ボラティリティ予測結果
        multiframe_results: マルチタイムフレーム分析結果
        technical_results: 高度テクニカル指標結果
        symbol: 銘柄コード
        save_path: 保存パス
        output_dir: 出力ディレクトリ
        
    Returns:
        保存されたファイルパス
    """
    visualizer = create_visualizer(output_dir)
    return visualizer.create_comprehensive_dashboard(
        data, lstm_results, volatility_results, 
        multiframe_results, technical_results, symbol, save_path
    )


def create_interactive_plotly_dashboard(
    data,
    results_dict: Dict,
    symbol: str = "Unknown",
    save_path: Optional[str] = None,
    output_dir: str = "output/ml_visualizations"
) -> Optional[str]:
    """
    インタラクティブダッシュボード作成（バックワード互換性）
    
    Args:
        data: 価格データ
        results_dict: 全分析結果の辞書
        symbol: 銘柄コード
        save_path: 保存パス
        output_dir: 出力ディレクトリ
        
    Returns:
        保存されたファイルパス（HTML）
    """
    visualizer = create_visualizer(output_dir)
    return visualizer.create_interactive_plotly_dashboard(
        data, results_dict, symbol, save_path
    )


def generate_analysis_report(
    symbol: str,
    results_dict: Dict,
    save_path: Optional[str] = None,
    output_dir: str = "output/ml_visualizations"
) -> Optional[str]:
    """
    分析レポート生成（バックワード互換性）
    
    Args:
        symbol: 銘柄コード
        results_dict: 全分析結果の辞書
        save_path: 保存パス
        output_dir: 出力ディレクトリ
        
    Returns:
        保存されたファイルパス
    """
    visualizer = create_visualizer(output_dir)
    return visualizer.generate_analysis_report(
        symbol, results_dict, save_path
    )


# エクスポート対象の定義
__all__ = [
    # バージョン情報
    "__version__",
    "__author__", 
    "__description__",
    
    # 主要クラス
    "MLResultsVisualizer",
    "Visualizer",  # バックワード互換性エイリアス
    
    # コンポーネントクラス
    "DataProcessor",
    "BaseChartGenerator",
    "LSTMChartGenerator",
    "VolatilityChartGenerator",
    "InteractiveDashboard",
    "AdvancedChartGenerator",
    "ReportBuilder",
    
    # 型・列挙型
    "RiskLevel",
    "TradingAction",
    "SignalStrength",
    "PositionSize", 
    "TrendDirection",
    "ChartType",
    
    # データクラス
    "ColorPalette",
    "LSTMResults",
    "CurrentMetrics", 
    "EnsembleForecast",
    "RiskAssessment",
    "VolatilityResults",
    "TechnicalIndicators",
    "TimeframeData",
    "IntegratedSignal",
    "InvestmentRecommendation",
    "IntegratedAnalysis",
    "MultiFrameResults",
    "VisualizationConfig",
    "ChartMetadata",
    "DependencyStatus",
    
    # 設定・定数
    "DEFAULT_CONFIG",
    "ERROR_MESSAGES", 
    "JAPANESE_LABELS",
    "TIMEFRAMES",
    "CHART_SETTINGS",
    
    # ユーティリティ関数
    "create_visualizer",
    "check_dependencies",
    "get_dependency_status",
    "validate_lstm_results",
    "validate_volatility_results",
    "validate_multiframe_results",
    "get_risk_color",
    "get_signal_color",
    
    # バックワード互換性関数
    "create_lstm_prediction_chart",
    "create_volatility_forecast_chart",
    "create_multiframe_analysis_chart", 
    "create_comprehensive_dashboard",
    "create_interactive_plotly_dashboard",
    "generate_analysis_report",
]

# 初期化ログ
logger.info(f"機械学習結果可視化システム v{__version__} 初期化完了")

# 依存関係の初期チェック
_initial_deps = check_dependencies()
logger.info(f"依存関係チェック: {_initial_deps}")
if not _initial_deps['can_create_static_charts']:
    logger.warning("matplotlib未インストール - 静的チャート作成不可")
if not _initial_deps['can_create_interactive_charts']:
    logger.warning("plotly未インストール - インタラクティブチャート作成不可")