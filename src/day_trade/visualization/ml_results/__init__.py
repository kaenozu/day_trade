#!/usr/bin/env python3
"""
機械学習結果可視化システム - パッケージ初期化

Issue #315: 高度テクニカル指標・ML機能拡張
分割されたモジュールの統合とバックワード互換性の提供
"""

import warnings
from typing import Dict, Optional

import pandas as pd

# バージョン情報
__version__ = "1.0.0"
__author__ = "ML Results Visualization System"
__description__ = "機械学習結果可視化システム統合パッケージ"

# 警告の抑制
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 基本クラス
from .base import BaseMLVisualizer, MATPLOTLIB_AVAILABLE, PLOTLY_AVAILABLE, logger

# 各機能モジュール
from .dashboard_interactive import InteractiveDashboardGenerator
from .dashboard_static import StaticDashboardGenerator
from .lstm_chart import LSTMChartGenerator
from .multiframe_chart import MultiframeChartGenerator
from .report_generator import ReportGenerator
from .test_runner import MLVisualizationTestRunner
from .volatility_chart import VolatilityChartGenerator

# 後方互換性のためのメインクラス
class MLResultsVisualizer(BaseMLVisualizer):
    """
    機械学習結果可視化システム - 統合クラス
    
    元のMLResultsVisualizerクラスとの完全な後方互換性を提供
    すべての機能を内部的に分割されたモジュールに委譲
    """

    def __init__(self, output_dir: str = "output/ml_visualizations"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        super().__init__(output_dir)
        
        # 各機能の生成クラスをインスタンス化
        self._lstm_generator = LSTMChartGenerator(output_dir)
        self._volatility_generator = VolatilityChartGenerator(output_dir)
        self._multiframe_generator = MultiframeChartGenerator(output_dir)
        self._static_dashboard_generator = StaticDashboardGenerator(output_dir)
        self._interactive_dashboard_generator = InteractiveDashboardGenerator(output_dir)
        self._report_generator = ReportGenerator(output_dir)
        
        logger.info("機械学習結果可視化システム初期化完了")

    def create_lstm_prediction_chart(
        self,
        data: pd.DataFrame,
        lstm_results: Dict,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        LSTM予測結果チャート作成
        
        後方互換性のため、LSTMChartGeneratorに委譲
        """
        return self._lstm_generator.create_lstm_prediction_chart(
            data, lstm_results, symbol, save_path
        )

    def create_volatility_forecast_chart(
        self,
        data: pd.DataFrame,
        volatility_results: Dict,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        ボラティリティ予測チャート作成
        
        後方互換性のため、VolatilityChartGeneratorに委譲
        """
        return self._volatility_generator.create_volatility_forecast_chart(
            data, volatility_results, symbol, save_path
        )

    def create_multiframe_analysis_chart(
        self,
        data: pd.DataFrame,
        multiframe_results: Dict,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        マルチタイムフレーム分析チャート作成
        
        後方互換性のため、MultiframeChartGeneratorに委譲
        """
        return self._multiframe_generator.create_multiframe_analysis_chart(
            data, multiframe_results, symbol, save_path
        )

    def create_comprehensive_dashboard(
        self,
        data: pd.DataFrame,
        lstm_results: Optional[Dict] = None,
        volatility_results: Optional[Dict] = None,
        multiframe_results: Optional[Dict] = None,
        technical_results: Optional[Dict] = None,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        総合ダッシュボード作成
        
        後方互換性のため、StaticDashboardGeneratorに委譲
        """
        return self._static_dashboard_generator.create_comprehensive_dashboard(
            data, lstm_results, volatility_results, multiframe_results, 
            technical_results, symbol, save_path
        )

    def create_interactive_plotly_dashboard(
        self,
        data: pd.DataFrame,
        results_dict: Dict,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plotlyによるインタラクティブダッシュボード作成
        
        後方互換性のため、InteractiveDashboardGeneratorに委譲
        """
        return self._interactive_dashboard_generator.create_interactive_plotly_dashboard(
            data, results_dict, symbol, save_path
        )

    def generate_analysis_report(
        self, symbol: str, results_dict: Dict, save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        分析レポート生成（テキスト形式）
        
        後方互換性のため、ReportGeneratorに委譲
        """
        return self._report_generator.generate_analysis_report(
            symbol, results_dict, save_path
        )

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


# 公開API定義
__all__ = [
    # バージョン情報
    "__version__",
    "__author__", 
    "__description__",
    
    # メインクラス（後方互換性）
    'MLResultsVisualizer',
    
    # 基本クラス
    'BaseMLVisualizer',
    
    # 個別生成クラス
    'LSTMChartGenerator',
    'VolatilityChartGenerator', 
    'MultiframeChartGenerator',
    'StaticDashboardGenerator',
    'InteractiveDashboardGenerator',
    'ReportGenerator',
    'MLVisualizationTestRunner',
    
    # 定数
    'MATPLOTLIB_AVAILABLE',
    'PLOTLY_AVAILABLE',
    'logger',
    
    # バックワード互換性関数
    "create_lstm_prediction_chart",
    "create_volatility_forecast_chart",
    "create_multiframe_analysis_chart", 
    "create_comprehensive_dashboard",
    "create_interactive_plotly_dashboard",
    "generate_analysis_report",
    
    # ユーティリティ関数
    "create_visualizer",
    "check_dependencies",
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