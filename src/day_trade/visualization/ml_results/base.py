#!/usr/bin/env python3
"""
機械学習結果可視化システム - 基本クラス
Issue #315: 高度テクニカル指標・ML機能拡張

基本設定・共通機能・カラーパレット・依存関係チェック
"""

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon, Rectangle

    # 日本語フォント設定
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    MATPLOTLIB_AVAILABLE = True
    logger.info("matplotlib/seaborn利用可能")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning(
        "matplotlib/seaborn未インストール - pip install matplotlib seabornでインストール"
    )

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
    logger.info("plotly利用可能")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly未インストール - pip install plotlyでインストール")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class BaseMLVisualizer:
    """
    機械学習結果可視化システムの基本クラス

    LSTM・GARCH・テクニカル分析・ボラティリティ予測の結果を統合的に可視化
    """

    def __init__(self, output_dir: str = "output/ml_visualizations"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # カラーパレット設定
        self.colors = {
            "price": "#1f77b4",
            "prediction": "#ff7f0e",
            "lstm": "#2ca02c",
            "garch": "#d62728",
            "vix": "#9467bd",
            "support": "#8c564b",
            "resistance": "#e377c2",
            "bullish": "#2ca02c",
            "bearish": "#d62728",
            "neutral": "#7f7f7f",
            "volatility": "#ff9999",
            "confidence": "#66b3ff",
        }

        # プロット設定
        if MATPLOTLIB_AVAILABLE:
            sns.set_style("whitegrid")
            plt.style.use("default")

        logger.info("機械学習結果可視化システム基本クラス初期化完了")

    def check_matplotlib_available(self) -> bool:
        """
        Matplotlibの利用可能性をチェック

        Returns:
            利用可能な場合True
        """
        return MATPLOTLIB_AVAILABLE

    def check_plotly_available(self) -> bool:
        """
        Plotlyの利用可能性をチェック

        Returns:
            利用可能な場合True
        """
        return PLOTLY_AVAILABLE

    def close_figure(self, fig):
        """
        図を安全にクローズ

        Args:
            fig: Matplotlibの図オブジェクト
        """
        if MATPLOTLIB_AVAILABLE and fig is not None:
            plt.close(fig)