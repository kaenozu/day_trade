"""
基底チャートレンダラー

共通的なチャート描画機能を提供
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import pandas as pd
import numpy as np

from ...utils.logging_config import get_context_logger
from .color_palette import ColorPalette

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon, Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未インストール")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly未インストール")

warnings.filterwarnings("ignore", category=UserWarning)


class ChartRenderer(ABC):
    """
    チャートレンダラー基底クラス

    matplotlib/plotly の統一インターフェースを提供
    """

    def __init__(self, output_dir: str = "output/charts", theme: str = "default"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            theme: テーマ ("default", "dark")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.palette = ColorPalette(theme)
        self.theme = theme

        logger.info(f"チャートレンダラー初期化完了 - テーマ: {theme}")

    @abstractmethod
    def render(self, data: pd.DataFrame, **kwargs) -> Optional[str]:
        """
        チャート描画（抽象メソッド）

        Args:
            data: 描画データ
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        pass

    def create_figure(self, figsize: Tuple[int, int] = (12, 8), **kwargs) -> Any:
        """
        図の作成

        Args:
            figsize: 図のサイズ
            **kwargs: 追加パラメータ

        Returns:
            図オブジェクト
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - 図の作成不可")
            return None

        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        return fig, ax

    def create_subplots(
        self, rows: int, cols: int = 1, figsize: Tuple[int, int] = (12, 10)
    ) -> Any:
        """
        サブプロット作成

        Args:
            rows: 行数
            cols: 列数
            figsize: 図のサイズ

        Returns:
            図・軸オブジェクト
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - サブプロット作成不可")
            return None, None

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        return fig, axes

    def add_candlestick(self, ax, data: pd.DataFrame, **kwargs) -> None:
        """
        ローソク足チャート追加

        Args:
            ax: 描画軸
            data: OHLC データ
            **kwargs: 追加パラメータ
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        # 簡易ローソク足実装
        opens = data["Open"].values
        closes = data["Close"].values
        highs = data["High"].values
        lows = data["Low"].values

        up_color = self.palette.get_color("bullish")
        down_color = self.palette.get_color("bearish")

        for i, (open_price, close_price, high, low) in enumerate(
            zip(opens, closes, highs, lows)
        ):
            color = up_color if close_price > open_price else down_color

            # ヒゲ
            ax.plot([i, i], [low, high], color=color, linewidth=1)

            # 実体
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            rect = Rectangle((i - 0.3, bottom), 0.6, height, facecolor=color, alpha=0.7)
            ax.add_patch(rect)

    def add_volume_bars(self, ax, data: pd.DataFrame, **kwargs) -> None:
        """
        出来高バー追加

        Args:
            ax: 描画軸
            data: 出来高データ
            **kwargs: 追加パラメータ
        """
        if not MATPLOTLIB_AVAILABLE or "Volume" not in data.columns:
            return

        volume_color = self.palette.get_color("volume")
        ax.bar(range(len(data)), data["Volume"], color=volume_color, alpha=0.6)

    def add_moving_average(
        self, ax, data: pd.DataFrame, periods: List[int], **kwargs
    ) -> None:
        """
        移動平均線追加

        Args:
            ax: 描画軸
            data: 価格データ
            periods: 移動平均期間リスト
            **kwargs: 追加パラメータ
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        colors = ["ma_short", "ma_long"]

        for i, period in enumerate(periods):
            ma = data["Close"].rolling(window=period).mean()
            color = self.palette.get_color(colors[i % len(colors)])
            ax.plot(ma, label=f"MA({period})", color=color, linewidth=1.5)

    def add_bollinger_bands(
        self, ax, data: pd.DataFrame, period: int = 20, std_dev: float = 2, **kwargs
    ) -> None:
        """
        ボリンジャーバンド追加

        Args:
            ax: 描画軸
            data: 価格データ
            period: 移動平均期間
            std_dev: 標準偏差倍率
            **kwargs: 追加パラメータ
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        ma = data["Close"].rolling(window=period).mean()
        std = data["Close"].rolling(window=period).std()

        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)

        bb_color = self.palette.get_color("bb_upper")

        ax.plot(upper_band, color=bb_color, linestyle="--", alpha=0.7, label="BB Upper")
        ax.plot(ma, color=self.palette.get_color("ma_short"), label=f"BB Mid({period})")
        ax.plot(lower_band, color=bb_color, linestyle="--", alpha=0.7, label="BB Lower")

        # バンド間の塗りつぶし
        ax.fill_between(
            range(len(data)), upper_band, lower_band, color=bb_color, alpha=0.1
        )

    def add_rsi(self, ax, data: pd.DataFrame, period: int = 14, **kwargs) -> None:
        """
        RSI追加

        Args:
            ax: 描画軸
            data: 価格データ
            period: RSI期間
            **kwargs: 追加パラメータ
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        rsi_color = self.palette.get_color("rsi")
        ax.plot(rsi, color=rsi_color, label="RSI", linewidth=1.5)

        # オーバーボート・オーバーソールドライン
        ax.axhline(y=70, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=30, color="green", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 100)

    def save_figure(self, fig, filename: str, **kwargs) -> str:
        """
        図の保存

        Args:
            fig: 図オブジェクト
            filename: ファイル名
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - 保存不可")
            return ""

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=kwargs.get("dpi", 100), bbox_inches="tight", **kwargs)
        plt.close(fig)

        logger.info(f"チャート保存完了: {filepath}")
        return str(filepath)

    def create_plotly_figure(self, **kwargs) -> Optional[go.Figure]:
        """
        Plotly図作成

        Args:
            **kwargs: 追加パラメータ

        Returns:
            Plotly図オブジェクト
        """
        if not PLOTLY_AVAILABLE:
            logger.error("plotly未インストール - インタラクティブ図作成不可")
            return None

        return go.Figure(**kwargs)

    def save_plotly_figure(
        self, fig: go.Figure, filename: str, format: str = "html"
    ) -> str:
        """
        Plotly図保存

        Args:
            fig: Plotly図オブジェクト
            filename: ファイル名
            format: 保存形式 ("html", "png", "pdf")

        Returns:
            保存されたファイルパス
        """
        if not PLOTLY_AVAILABLE:
            logger.error("plotly未インストール - 保存不可")
            return ""

        # 拡張子調整
        name_parts = filename.split(".")
        if len(name_parts) > 1:
            name_parts[-1] = format
        else:
            name_parts.append(format)
        filename = ".".join(name_parts)

        filepath = self.output_dir / filename

        if format == "html":
            fig.write_html(str(filepath))
        elif format == "png":
            fig.write_image(str(filepath), format="png")
        elif format == "pdf":
            fig.write_image(str(filepath), format="pdf")
        else:
            logger.error(f"未対応形式: {format}")
            return ""

        logger.info(f"Plotlyチャート保存完了: {filepath}")
        return str(filepath)

    def apply_common_styling(
        self, ax, title: str = "", xlabel: str = "", ylabel: str = ""
    ) -> None:
        """
        共通スタイル適用

        Args:
            ax: 描画軸
            title: タイトル
            xlabel: X軸ラベル
            ylabel: Y軸ラベル
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        self.palette.apply_style_to_axes(ax, title)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # 日付軸の場合
        if hasattr(ax, "xaxis") and any(
            "date" in str(type(tick)).lower() for tick in ax.get_xticklabels()
        ):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 凡例
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
