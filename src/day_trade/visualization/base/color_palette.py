"""
カラーパレット・スタイル管理

一貫したカラースキーマとスタイル設定を提供
"""

import warnings
from typing import Dict, Optional

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
    logger.info("matplotlib/seaborn利用可能")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib/seaborn未インストール")

# 警告フィルター
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ColorPalette:
    """
    統一カラーパレット管理

    全ての可視化コンポーネントで一貫したカラースキーマを使用
    """

    # 基本カラーパレット
    COLORS = {
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
        "volume": "#17becf",
        "ma_short": "#ff9f40",
        "ma_long": "#4bc0c0",
        "rsi": "#9966ff",
        "macd": "#ff6384",
        "bb_upper": "#ffcd56",
        "bb_lower": "#ffcd56",
        "signal": "#36a2eb",
    }

    # ダークテーマ
    DARK_COLORS = {
        "price": "#8dd3c7",
        "prediction": "#ffffb3",
        "lstm": "#bebada",
        "garch": "#fb8072",
        "vix": "#80b1d3",
        "support": "#fdb462",
        "resistance": "#b3de69",
        "bullish": "#00ff00",
        "bearish": "#ff4444",
        "neutral": "#cccccc",
        "volatility": "#fccde5",
        "confidence": "#d9d9d9",
    }

    def __init__(self, theme: str = "default"):
        """
        初期化

        Args:
            theme: カラーテーマ ("default", "dark")
        """
        self.theme = theme
        self.colors = self.DARK_COLORS if theme == "dark" else self.COLORS

        # matplotlib設定
        if MATPLOTLIB_AVAILABLE:
            self._setup_matplotlib_style()

        logger.info(f"カラーパレット初期化完了 - テーマ: {theme}")

    def _setup_matplotlib_style(self) -> None:
        """matplotlib スタイル設定"""
        if self.theme == "dark":
            plt.style.use("dark_background")
            sns.set_style("darkgrid")
        else:
            plt.style.use("default")
            sns.set_style("whitegrid")

        # 日本語フォント設定
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3

    def get_color(self, color_name: str) -> str:
        """
        カラー取得

        Args:
            color_name: カラー名

        Returns:
            カラーコード
        """
        return self.colors.get(color_name, self.colors["neutral"])

    def get_color_sequence(self, count: int) -> list:
        """
        カラーシーケンス取得

        Args:
            count: 必要なカラー数

        Returns:
            カラーリスト
        """
        color_list = list(self.colors.values())
        return (
            color_list[:count]
            if count <= len(color_list)
            else color_list * (count // len(color_list) + 1)
        )

    def get_gradient_colors(self, start_color: str, end_color: str, steps: int) -> list:
        """
        グラデーションカラー生成

        Args:
            start_color: 開始カラー
            end_color: 終了カラー
            steps: グラデーション段階数

        Returns:
            グラデーションカラーリスト
        """
        try:
            import matplotlib.colors as mcolors

            return [
                mcolors.to_hex(c)
                for c in mcolors.LinearSegmentedColormap.from_list(
                    "", [start_color, end_color]
                )(range(steps))
            ]
        except ImportError:
            # フォールバック: 単純な繰り返し
            return [start_color, end_color] * (steps // 2 + 1)

    def get_categorical_palette(self, n_colors: int) -> list:
        """
        カテゴリカルパレット取得

        Args:
            n_colors: 必要なカラー数

        Returns:
            カテゴリカルカラーリスト
        """
        if MATPLOTLIB_AVAILABLE:
            try:
                return sns.color_palette("husl", n_colors).as_hex()
            except:
                pass

        # フォールバック
        return self.get_color_sequence(n_colors)[:n_colors]

    def get_theme_config(self) -> Dict[str, any]:
        """
        テーマ設定取得

        Returns:
            テーマ設定辞書
        """
        base_config = {
            "colors": self.colors,
            "figure_size": (12, 8),
            "dpi": 100,
            "grid": True,
            "grid_alpha": 0.3,
        }

        if self.theme == "dark":
            base_config.update(
                {
                    "background": "#2e2e2e",
                    "text_color": "#ffffff",
                    "grid_color": "#444444",
                }
            )
        else:
            base_config.update(
                {
                    "background": "#ffffff",
                    "text_color": "#000000",
                    "grid_color": "#cccccc",
                }
            )

        return base_config

    def apply_style_to_axes(self, ax, title: Optional[str] = None) -> None:
        """
        軸にスタイルを適用

        Args:
            ax: matplotlib軸オブジェクト
            title: チャートタイトル
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        config = self.get_theme_config()

        if title:
            ax.set_title(
                title, fontsize=14, fontweight="bold", color=config["text_color"]
            )

        ax.tick_params(colors=config["text_color"])
        ax.xaxis.label.set_color(config["text_color"])
        ax.yaxis.label.set_color(config["text_color"])

        if config["grid"]:
            ax.grid(True, alpha=config["grid_alpha"], color=config["grid_color"])
