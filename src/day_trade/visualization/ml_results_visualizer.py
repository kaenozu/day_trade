#!/usr/bin/env python3
"""
機械学習結果可視化システム - 後方互換性ファイル
Issue #315: 高度テクニカル指標・ML機能拡張

このファイルは後方互換性のために残されています。
実際の機能は ml_results/ パッケージに分割されました。
"""

# 新しい分割されたモジュールからすべてのクラスと関数をインポート
from .ml_results import (
    # メインクラス
    MLResultsVisualizer,
    
    # 個別生成クラス
    BaseMLVisualizer,
    LSTMChartGenerator,
    VolatilityChartGenerator,
    MultiframeChartGenerator,
    StaticDashboardGenerator,
    InteractiveDashboardGenerator,
    ReportGenerator,
    MLVisualizationTestRunner,
    
    # 定数
    MATPLOTLIB_AVAILABLE,
    PLOTLY_AVAILABLE,
    logger,
    
    # バックワード互換性関数
    create_lstm_prediction_chart,
    create_volatility_forecast_chart,
    create_multiframe_analysis_chart,
    create_comprehensive_dashboard,
    create_interactive_plotly_dashboard,
    generate_analysis_report,
    
    # ユーティリティ関数
    create_visualizer,
    check_dependencies,
)

# テスト実行機能（元のファイルの__main__部分と互換）
def main():
    """
    テスト実行（後方互換性）
    
    新しいMLVisualizationTestRunnerを使用してテストを実行
    """
    test_runner = MLVisualizationTestRunner()
    test_runner.run_comprehensive_test()

if __name__ == "__main__":
    main()

# 公開API（元のファイルと同じインターフェース）
__all__ = [
    'MLResultsVisualizer',
    'BaseMLVisualizer',
    'LSTMChartGenerator',
    'VolatilityChartGenerator',
    'MultiframeChartGenerator',
    'StaticDashboardGenerator',
    'InteractiveDashboardGenerator',
    'ReportGenerator',
    'MLVisualizationTestRunner',
    'MATPLOTLIB_AVAILABLE',
    'PLOTLY_AVAILABLE',
    'logger',
    'create_lstm_prediction_chart',
    'create_volatility_forecast_chart',
    'create_multiframe_analysis_chart',
    'create_comprehensive_dashboard',
    'create_interactive_plotly_dashboard',
    'generate_analysis_report',
    'create_visualizer',
    'check_dependencies',
    'main',
]