"""
ローソク足チャート

高度なローソク足パターン認識とチャート可視化
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
from ..base.chart_renderer import ChartRenderer

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未インストール")


class CandlestickCharts(ChartRenderer):
    """
    ローソク足チャートクラス

    高度なローソク足パターン認識と詳細可視化を提供
    """

    def __init__(
        self, output_dir: str = "output/candlestick_charts", theme: str = "default"
    ):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            theme: テーマ
        """
        super().__init__(output_dir, theme)
        logger.info("ローソク足可視化システム初期化完了")

    def render(self, data: pd.DataFrame, patterns: Dict, **kwargs) -> Optional[str]:
        """
        ローソク足チャート描画

        Args:
            data: OHLCV データ
            patterns: 検出パターン辞書
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - ローソク足可視化不可")
            return None

        symbol = kwargs.get("symbol", "UNKNOWN")
        title = kwargs.get("title", f"{symbol} ローソク足パターン分析")

        fig, axes = self.create_subplots(3, 1, figsize=(15, 12))

        # メインローソク足チャート
        self._plot_main_candlesticks(axes[0], data, patterns, symbol)

        # パターン詳細
        self._plot_pattern_details(axes[1], data, patterns)

        # パターン統計
        self._plot_pattern_statistics(axes[2], patterns)

        plt.tight_layout()

        filename = kwargs.get("filename", f"candlestick_analysis_{symbol}.png")
        return self.save_figure(fig, filename)

    def _plot_main_candlesticks(
        self, ax, data: pd.DataFrame, patterns: Dict, symbol: str
    ) -> None:
        """
        メインローソク足プロット

        Args:
            ax: 描画軸
            data: OHLCV データ
            patterns: パターン辞書
            symbol: 銘柄シンボル
        """
        required_columns = ["Open", "High", "Low", "Close"]
        if not all(col in data.columns for col in required_columns):
            logger.warning("OHLC データが不完全です")
            return

        opens = data["Open"].values
        highs = data["High"].values
        lows = data["Low"].values
        closes = data["Close"].values

        # ローソク足描画
        for i in range(len(data)):
            open_price = opens[i]
            high_price = highs[i]
            low_price = lows[i]
            close_price = closes[i]

            # 上昇・下落の判定
            is_bullish = close_price > open_price
            color = (
                self.palette.get_color("bullish")
                if is_bullish
                else self.palette.get_color("bearish")
            )

            # ヒゲ（上下の線）
            ax.plot([i, i], [low_price, high_price], color=color, linewidth=1)

            # 実体（四角形）
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)

            rect = Rectangle(
                (i - 0.4, body_bottom),
                0.8,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8 if is_bullish else 0.9,
            )
            ax.add_patch(rect)

        # パターンハイライト
        if "detected_patterns" in patterns:
            self._highlight_patterns(ax, data, patterns["detected_patterns"])

        self.apply_common_styling(
            ax, title=f"{symbol} ローソク足チャート", xlabel="時間", ylabel="価格"
        )

    def _highlight_patterns(
        self, ax, data: pd.DataFrame, detected_patterns: List[Dict]
    ) -> None:
        """
        検出パターンのハイライト

        Args:
            ax: 描画軸
            data: データ
            detected_patterns: 検出パターンリスト
        """
        pattern_colors = {
            "doji": self.palette.get_color("neutral"),
            "hammer": self.palette.get_color("bullish"),
            "hanging_man": self.palette.get_color("bearish"),
            "shooting_star": self.palette.get_color("bearish"),
            "inverted_hammer": self.palette.get_color("bullish"),
            "engulfing_bullish": self.palette.get_color("bullish"),
            "engulfing_bearish": self.palette.get_color("bearish"),
            "morning_star": self.palette.get_color("bullish"),
            "evening_star": self.palette.get_color("bearish"),
            "three_white_soldiers": self.palette.get_color("bullish"),
            "three_black_crows": self.palette.get_color("bearish"),
        }

        for pattern in detected_patterns:
            pattern_name = pattern.get("pattern", "unknown")
            start_idx = pattern.get("start_index", 0)
            end_idx = pattern.get("end_index", start_idx)
            confidence = pattern.get("confidence", 0)

            color = pattern_colors.get(pattern_name, self.palette.get_color("warning"))

            # パターン範囲をハイライト
            if start_idx < len(data) and end_idx < len(data):
                high_price = data.iloc[start_idx:end_idx + 1]["High"].max()
                low_price = data.iloc[start_idx:end_idx + 1]["Low"].min()

                # 背景ハイライト
                existing_labels = [p.get_label() for p in ax.get_children() if hasattr(p, 'get_label')]
                should_add_label = pattern_name not in existing_labels

                ax.axvspan(
                    start_idx - 0.5,
                    end_idx + 0.5,
                    color=color,
                    alpha=0.2,
                    label=f"{pattern_name}" if should_add_label else None,
                )

                # パターン名をアノテーション
                ax.annotate(
                    f"{pattern_name}\n({confidence:.1f}%)",
                    xy=((start_idx + end_idx) / 2, high_price),
                    xytext=(10, 20),
                    textcoords="offset points",
                    fontsize=9,
                    ha="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor=color, alpha=0.7
                    ),
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3,rad=0", color=color
                    ),
                )

    def _plot_pattern_details(self, ax, data: pd.DataFrame, patterns: Dict) -> None:
        """
        パターン詳細プロット

        Args:
            ax: 描画軸
            data: データ
            patterns: パターン辞書
        """
        if "pattern_strength" in patterns:
            strength_data = patterns["pattern_strength"]

            ax.plot(
                strength_data,
                label="パターン強度",
                color=self.palette.get_color("pattern_strength"),
                linewidth=2,
            )

            # 強度閾値線
            ax.axhline(
                0.7, color=self.palette.get_color("bullish"), linestyle="--", alpha=0.7
            )
            ax.axhline(
                0.3, color=self.palette.get_color("bearish"), linestyle="--", alpha=0.7
            )

        if "pattern_reliability" in patterns:
            reliability_data = patterns["pattern_reliability"]

            ax2 = ax.twinx()
            ax2.plot(
                reliability_data,
                label="パターン信頼度",
                color=self.palette.get_color("reliability"),
                linewidth=2,
                alpha=0.7,
                linestyle=":",
            )
            ax2.set_ylabel("信頼度")

        self.apply_common_styling(ax, title="パターン強度・信頼度", ylabel="強度")

    def _plot_pattern_statistics(self, ax, patterns: Dict) -> None:
        """
        パターン統計プロット

        Args:
            ax: 描画軸
            patterns: パターン辞書
        """
        if "pattern_frequency" in patterns:
            frequency_data = patterns["pattern_frequency"]

            pattern_names = list(frequency_data.keys())
            frequencies = list(frequency_data.values())

            if pattern_names:
                colors = self.palette.get_categorical_palette(len(pattern_names))
                bars = ax.bar(pattern_names, frequencies, color=colors)

                # 頻度をバーの上に表示
                for bar, freq in zip(bars, frequencies):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + height * 0.01,
                        f"{freq}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                self.apply_common_styling(ax, title="パターン出現頻度", ylabel="回数")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        elif "detected_patterns" in patterns:
            # 検出パターンから統計を作成
            detected_patterns = patterns["detected_patterns"]
            pattern_count = {}

            for pattern in detected_patterns:
                pattern_name = pattern.get("pattern", "unknown")
                pattern_count[pattern_name] = pattern_count.get(pattern_name, 0) + 1

            if pattern_count:
                pattern_names = list(pattern_count.keys())
                counts = list(pattern_count.values())

                colors = self.palette.get_categorical_palette(len(pattern_names))
                bars = ax.bar(pattern_names, counts, color=colors)

                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + height * 0.01,
                        f"{count}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                self.apply_common_styling(ax, title="検出パターン統計", ylabel="検出回数")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def create_pattern_recognition_dashboard(
        self, data: pd.DataFrame, patterns: Dict, symbol: str = "STOCK"
    ) -> str:
        """
        パターン認識ダッシュボード作成

        Args:
            data: OHLCV データ
            patterns: パターン辞書
            symbol: 銘柄シンボル

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - ダッシュボード作成不可")
            return ""

        fig = plt.figure(figsize=(20, 15))

        # グリッドレイアウト設定
        gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 1, 1])

        # メインローソク足（全幅）
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_main_candlesticks(ax_main, data, patterns, symbol)

        # パターン詳細
        ax_details = fig.add_subplot(gs[1, :2])
        self._plot_pattern_details(ax_details, data, patterns)

        # パターン統計
        ax_stats = fig.add_subplot(gs[1, 2])
        self._plot_pattern_statistics(ax_stats, patterns)

        # パターン成功率
        ax_success = fig.add_subplot(gs[2, 0])
        self._plot_pattern_success_rate(ax_success, patterns)

        # パターン分布
        ax_distribution = fig.add_subplot(gs[2, 1])
        self._plot_pattern_distribution(ax_distribution, patterns)

        # 時間軸分析
        ax_time = fig.add_subplot(gs[2, 2])
        self._plot_temporal_analysis(ax_time, patterns)

        # リスク・リターン分析
        ax_risk = fig.add_subplot(gs[3, :])
        self._plot_risk_return_analysis(ax_risk, patterns)

        plt.suptitle(
            f"{symbol} ローソク足パターン認識ダッシュボード", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        filename = f"pattern_recognition_dashboard_{symbol}.png"
        return self.save_figure(fig, filename, dpi=300)

    def _plot_pattern_success_rate(self, ax, patterns: Dict) -> None:
        """
        パターン成功率プロット

        Args:
            ax: 描画軸
            patterns: パターン辞書
        """
        if "pattern_success_rate" in patterns:
            success_data = patterns["pattern_success_rate"]

            pattern_names = list(success_data.keys())
            success_rates = list(success_data.values())

            if pattern_names:
                # 成功率に基づく色分け
                colors = []
                for rate in success_rates:
                    if rate >= 0.7:
                        colors.append(self.palette.get_color("bullish"))
                    elif rate >= 0.5:
                        colors.append(self.palette.get_color("neutral"))
                    else:
                        colors.append(self.palette.get_color("bearish"))

                bars = ax.bar(pattern_names, success_rates, color=colors)

                # 成功率をパーセント表示
                for bar, rate in zip(bars, success_rates):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{rate:.1%}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                ax.set_ylim(0, 1)
                self.apply_common_styling(ax, title="パターン成功率", ylabel="成功率")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_pattern_distribution(self, ax, patterns: Dict) -> None:
        """
        パターン分布プロット

        Args:
            ax: 描画軸
            patterns: パターン辞書
        """
        if "pattern_frequency" in patterns:
            frequency_data = patterns["pattern_frequency"]

            pattern_names = list(frequency_data.keys())
            frequencies = list(frequency_data.values())

            if pattern_names:
                colors = self.palette.get_categorical_palette(len(pattern_names))

                # パイチャート
                wedges, texts, autotexts = ax.pie(
                    frequencies,
                    labels=pattern_names,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=90,
                )

                # テキストのフォントサイズ調整
                for text in texts:
                    text.set_fontsize(8)
                for autotext in autotexts:
                    autotext.set_fontsize(8)

                ax.set_title("パターン分布", fontsize=10)

    def _plot_temporal_analysis(self, ax, patterns: Dict) -> None:
        """
        時間軸分析プロット

        Args:
            ax: 描画軸
            patterns: パターン辞書
        """
        if "temporal_analysis" in patterns:
            temporal_data = patterns["temporal_analysis"]

            # 月別パターン出現
            if "monthly_frequency" in temporal_data:
                monthly_freq = temporal_data["monthly_frequency"]
                months = list(range(1, 13))
                frequencies = [monthly_freq.get(month, 0) for month in months]

                ax.bar(months, frequencies, color=self.palette.get_color("temporal"))

                ax.set_xlabel("月")
                ax.set_ylabel("出現回数")
                ax.set_title("月別パターン出現頻度")
                ax.set_xticks(months)

    def _plot_risk_return_analysis(self, ax, patterns: Dict) -> None:
        """
        リスク・リターン分析プロット

        Args:
            ax: 描画軸
            patterns: パターン辞書
        """
        if "risk_return_analysis" in patterns:
            rr_data = patterns["risk_return_analysis"]

            for pattern_name, analysis in rr_data.items():
                returns = analysis.get("returns", [])
                risks = analysis.get("risks", [])

                if returns and risks:
                    avg_return = np.mean(returns)
                    avg_risk = np.mean(risks)

                    # パターン別の散布図
                    ax.scatter(
                        avg_risk,
                        avg_return,
                        label=pattern_name,
                        s=100,
                        alpha=0.7,
                    )

            # リスク・リターンの基準線
            ax.axhline(0, color="black", linestyle="-", alpha=0.3)
            ax.axvline(0, color="black", linestyle="-", alpha=0.3)

            self.apply_common_styling(
                ax, title="パターン別リスク・リターン分析", xlabel="リスク", ylabel="リターン"
            )

    def export_candlestick_report(
        self, data: pd.DataFrame, patterns: Dict, symbol: str = "STOCK"
    ) -> Dict[str, str]:
        """
        ローソク足分析レポート出力

        Args:
            data: データ
            patterns: パターン辞書
            symbol: 銘柄シンボル

        Returns:
            出力ファイルパス辞書
        """
        exported_files = {}

        try:
            # メインチャート
            main_chart = self.render(data, patterns, symbol=symbol)
            if main_chart:
                exported_files["main_chart"] = main_chart

            # パターン認識ダッシュボード
            dashboard = self.create_pattern_recognition_dashboard(
                data, patterns, symbol
            )
            if dashboard:
                exported_files["dashboard"] = dashboard

            logger.info(f"ローソク足分析レポート出力完了 - {len(exported_files)}ファイル")
            return exported_files

        except Exception as e:
            logger.error(f"ローソク足分析レポート出力エラー: {e}")
            return {}
