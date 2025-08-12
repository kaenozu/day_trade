"""
出来高分析可視化

出来高パターン・価格出来高関係・流動性分析の可視化
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
from ..base.chart_renderer import ChartRenderer

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未インストール")


class VolumeAnalysis(ChartRenderer):
    """
    出来高分析可視化クラス

    出来高パターン・価格出来高関係・流動性分析の詳細可視化を提供
    """

    def __init__(self, output_dir: str = "output/volume_charts", theme: str = "default"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            theme: テーマ
        """
        super().__init__(output_dir, theme)
        logger.info("出来高分析可視化システム初期化完了")

    def render(self, data: pd.DataFrame, volume_analysis: Dict, **kwargs) -> Optional[str]:
        """
        出来高分析チャート描画

        Args:
            data: OHLCV データ
            volume_analysis: 出来高分析結果
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - 出来高分析可視化不可")
            return None

        symbol = kwargs.get("symbol", "UNKNOWN")
        title = kwargs.get("title", f"{symbol} 出来高分析")

        fig, axes = self.create_subplots(4, 1, figsize=(15, 16))

        # 価格・出来高チャート
        self._plot_price_volume(axes[0], data, volume_analysis, symbol)

        # 出来高プロファイル
        self._plot_volume_profile(axes[1], data, volume_analysis)

        # 出来高指標
        self._plot_volume_indicators(axes[2], data, volume_analysis)

        # 流動性分析
        self._plot_liquidity_analysis(axes[3], volume_analysis)

        plt.tight_layout()

        filename = kwargs.get("filename", f"volume_analysis_{symbol}.png")
        return self.save_figure(fig, filename)

    def _plot_price_volume(
        self, ax, data: pd.DataFrame, volume_analysis: Dict, symbol: str
    ) -> None:
        """
        価格・出来高プロット

        Args:
            ax: 描画軸
            data: OHLCV データ
            volume_analysis: 出来高分析結果
            symbol: 銘柄シンボル
        """
        # 価格（左軸）
        if "Close" in data.columns:
            price_data = data["Close"]
            ax.plot(
                price_data,
                label="終値",
                color=self.palette.get_color("price"),
                linewidth=2,
            )

        # 出来高（右軸）
        if "Volume" in data.columns:
            ax2 = ax.twinx()
            volume_data = data["Volume"]

            # 出来高バー
            volume_colors = []
            if "Close" in data.columns and len(data) > 1:
                for i in range(len(volume_data)):
                    if i > 0:
                        price_up = data["Close"].iloc[i] > data["Close"].iloc[i - 1]
                        volume_colors.append(
                            self.palette.get_color("bullish" if price_up else "bearish")
                        )
                    else:
                        volume_colors.append(self.palette.get_color("neutral"))
            else:
                volume_colors = [self.palette.get_color("volume")] * len(volume_data)

            ax2.bar(
                range(len(volume_data)),
                volume_data,
                color=volume_colors,
                alpha=0.6,
                label="出来高",
            )

            # 出来高移動平均
            if "volume_ma" in volume_analysis:
                volume_ma = volume_analysis["volume_ma"]
                ax2.plot(
                    volume_ma,
                    label="出来高MA",
                    color=self.palette.get_color("volume_ma"),
                    linewidth=2,
                )

            ax2.set_ylabel("出来高")

        # 異常出来高の検出・ハイライト
        if "volume_spikes" in volume_analysis:
            volume_spikes = volume_analysis["volume_spikes"]
            for spike_idx in volume_spikes:
                if spike_idx < len(data):
                    ax.axvline(
                        spike_idx,
                        color=self.palette.get_color("warning"),
                        linestyle=":",
                        alpha=0.8,
                        linewidth=2,
                    )

        self.apply_common_styling(ax, title=f"{symbol} 価格・出来高", xlabel="時間", ylabel="価格")

    def _plot_volume_profile(self, ax, data: pd.DataFrame, volume_analysis: Dict) -> None:
        """
        出来高プロファイルプロット

        Args:
            ax: 描画軸
            data: データ
            volume_analysis: 出来高分析結果
        """
        if "volume_profile" in volume_analysis:
            volume_profile = volume_analysis["volume_profile"]

            # 価格帯別出来高
            if "price_levels" in volume_profile and "volumes" in volume_profile:
                price_levels = volume_profile["price_levels"]
                volumes = volume_profile["volumes"]

                # 水平バーチャートで表示
                ax.barh(
                    price_levels,
                    volumes,
                    height=(max(price_levels) - min(price_levels)) / len(price_levels),
                    color=self.palette.get_color("volume_profile"),
                    alpha=0.7,
                )

                # POC（Point of Control）の強調
                if "poc" in volume_profile:
                    poc_price = volume_profile["poc"]
                    ax.axhline(
                        poc_price,
                        color=self.palette.get_color("poc"),
                        linestyle="-",
                        linewidth=3,
                        label=f"POC: {poc_price:.2f}",
                    )

                # VAH/VAL（Value Area High/Low）
                if "value_area" in volume_profile:
                    va = volume_profile["value_area"]
                    if "high" in va and "low" in va:
                        ax.axhline(
                            va["high"],
                            color=self.palette.get_color("va_high"),
                            linestyle="--",
                            alpha=0.8,
                            label=f"VAH: {va['high']:.2f}",
                        )
                        ax.axhline(
                            va["low"],
                            color=self.palette.get_color("va_low"),
                            linestyle="--",
                            alpha=0.8,
                            label=f"VAL: {va['low']:.2f}",
                        )

        elif "Close" in data.columns and "Volume" in data.columns:
            # 簡易出来高プロファイル作成
            price_data = data["Close"]
            volume_data = data["Volume"]

            # 価格帯を分割
            price_min, price_max = price_data.min(), price_data.max()
            price_bins = np.linspace(price_min, price_max, 20)
            volume_by_price = np.zeros(len(price_bins) - 1)

            # 各価格帯の出来高を集計
            for i in range(len(data)):
                price = price_data.iloc[i]
                volume = volume_data.iloc[i]
                bin_idx = np.digitize(price, price_bins) - 1
                if 0 <= bin_idx < len(volume_by_price):
                    volume_by_price[bin_idx] += volume

            # プロット
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            ax.barh(
                bin_centers,
                volume_by_price,
                height=(price_max - price_min) / 20,
                color=self.palette.get_color("volume_profile"),
                alpha=0.7,
            )

        self.apply_common_styling(ax, title="出来高プロファイル", xlabel="出来高", ylabel="価格")

    def _plot_volume_indicators(self, ax, data: pd.DataFrame, volume_analysis: Dict) -> None:
        """
        出来高指標プロット

        Args:
            ax: 描画軸
            data: データ
            volume_analysis: 出来高分析結果
        """
        indicators_plotted = []

        # OBV (On-Balance Volume)
        if "obv" in volume_analysis:
            obv_data = volume_analysis["obv"]
            ax.plot(
                obv_data,
                label="OBV",
                color=self.palette.get_color("obv"),
                linewidth=2,
            )
            indicators_plotted.append("OBV")

        # A/D Line (Accumulation/Distribution Line)
        if "ad_line" in volume_analysis:
            ad_line = volume_analysis["ad_line"]
            color = (
                self.palette.get_color("ad_line")
                if "OBV" not in indicators_plotted
                else self.palette.get_color("ad_line_secondary")
            )

            if indicators_plotted:
                ax2 = ax.twinx()
                ax2.plot(ad_line, label="A/D Line", color=color, linewidth=2, alpha=0.7)
                ax2.set_ylabel("A/D Line")
            else:
                ax.plot(ad_line, label="A/D Line", color=color, linewidth=2)

            indicators_plotted.append("A/D Line")

        # Chaikin Money Flow
        if "cmf" in volume_analysis:
            cmf_data = volume_analysis["cmf"]

            if len(indicators_plotted) >= 2:
                # 新しいサブプロットが必要な場合のフォールバック
                ax.plot(
                    cmf_data,
                    label="CMF",
                    color=self.palette.get_color("cmf"),
                    linewidth=1.5,
                    alpha=0.7,
                    linestyle="--",
                )
            else:
                ax3 = ax.twinx() if indicators_plotted else ax
                ax3.plot(
                    cmf_data,
                    label="CMF",
                    color=self.palette.get_color("cmf"),
                    linewidth=2,
                )

                # CMFゼロライン
                ax3.axhline(0, color="black", linestyle="-", alpha=0.3)

                if indicators_plotted:
                    ax3.set_ylabel("CMF")
                    ax3.spines["right"].set_position(("outward", 60))

            indicators_plotted.append("CMF")

        # Volume Rate of Change
        if "volume_roc" in volume_analysis:
            volume_roc = volume_analysis["volume_roc"]

            # 新しい軸に表示
            ax_roc = ax.twinx()
            if len(indicators_plotted) >= 1:
                ax_roc.spines["right"].set_position(("outward", 120))

            ax_roc.plot(
                volume_roc,
                label="Volume ROC",
                color=self.palette.get_color("volume_roc"),
                linewidth=1.5,
                alpha=0.6,
                linestyle=":",
            )
            ax_roc.set_ylabel("Volume ROC (%)")
            ax_roc.axhline(0, color="gray", linestyle="-", alpha=0.3)

        title = "出来高指標"
        if indicators_plotted:
            title += f" ({', '.join(indicators_plotted)})"

        self.apply_common_styling(ax, title=title, ylabel="値")

    def _plot_liquidity_analysis(self, ax, volume_analysis: Dict) -> None:
        """
        流動性分析プロット

        Args:
            ax: 描画軸
            volume_analysis: 出来高分析結果
        """
        if "liquidity_analysis" in volume_analysis:
            liquidity_data = volume_analysis["liquidity_analysis"]

            # 流動性スコア
            if "liquidity_score" in liquidity_data:
                liquidity_score = liquidity_data["liquidity_score"]
                ax.plot(
                    liquidity_score,
                    label="流動性スコア",
                    color=self.palette.get_color("liquidity"),
                    linewidth=2,
                )

                # 流動性閾値
                ax.axhline(
                    0.7,
                    color=self.palette.get_color("bullish"),
                    linestyle="--",
                    alpha=0.7,
                    label="高流動性閾値",
                )
                ax.axhline(
                    0.3,
                    color=self.palette.get_color("bearish"),
                    linestyle="--",
                    alpha=0.7,
                    label="低流動性閾値",
                )

            # ビッド・オファースプレッド
            if "bid_ask_spread" in liquidity_data:
                spread_data = liquidity_data["bid_ask_spread"]
                ax2 = ax.twinx()
                ax2.plot(
                    spread_data,
                    label="ビッド・オファースプレッド",
                    color=self.palette.get_color("spread"),
                    linewidth=2,
                    alpha=0.7,
                )
                ax2.set_ylabel("スプレッド")

        elif "volume_volatility" in volume_analysis:
            # 出来高ボラティリティによる流動性推定
            volume_volatility = volume_analysis["volume_volatility"]
            # 逆数で流動性を近似（低ボラティリティ = 高流動性）
            liquidity_proxy = 1 / (1 + np.array(volume_volatility))

            ax.plot(
                liquidity_proxy,
                label="流動性代理指標",
                color=self.palette.get_color("liquidity"),
                linewidth=2,
            )

            ax.axhline(
                0.5,
                color="gray",
                linestyle="--",
                alpha=0.7,
                label="中央値",
            )

        self.apply_common_styling(ax, title="流動性分析", ylabel="流動性")

    def create_comprehensive_volume_dashboard(
        self, data: pd.DataFrame, volume_analysis: Dict, symbol: str = "STOCK"
    ) -> str:
        """
        総合出来高分析ダッシュボード作成

        Args:
            data: OHLCV データ
            volume_analysis: 出来高分析結果
            symbol: 銘柄シンボル

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - ダッシュボード作成不可")
            return ""

        fig = plt.figure(figsize=(20, 15))

        # グリッドレイアウト設定
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1])

        # メイン価格・出来高チャート（全幅）
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_price_volume(ax_main, data, volume_analysis, symbol)

        # 出来高プロファイル
        ax_profile = fig.add_subplot(gs[1, 0])
        self._plot_volume_profile(ax_profile, data, volume_analysis)

        # 出来高指標
        ax_indicators = fig.add_subplot(gs[1, 1])
        self._plot_volume_indicators(ax_indicators, data, volume_analysis)

        # 流動性分析
        ax_liquidity = fig.add_subplot(gs[1, 2])
        self._plot_liquidity_analysis(ax_liquidity, volume_analysis)

        # 出来高分布
        ax_distribution = fig.add_subplot(gs[2, 0])
        self._plot_volume_distribution(ax_distribution, data, volume_analysis)

        # 出来高・価格相関
        ax_correlation = fig.add_subplot(gs[2, 1])
        self._plot_volume_price_correlation(ax_correlation, data, volume_analysis)

        # 出来高統計
        ax_stats = fig.add_subplot(gs[2, 2])
        self._plot_volume_statistics(ax_stats, data, volume_analysis)

        plt.suptitle(f"{symbol} 総合出来高分析ダッシュボード", fontsize=16, fontweight="bold")
        plt.tight_layout()

        filename = f"volume_dashboard_{symbol}.png"
        return self.save_figure(fig, filename, dpi=300)

    def _plot_volume_distribution(self, ax, data: pd.DataFrame, volume_analysis: Dict) -> None:
        """
        出来高分布プロット

        Args:
            ax: 描画軸
            data: データ
            volume_analysis: 出来高分析結果
        """
        if "Volume" in data.columns:
            volume_data = data["Volume"]

            # ヒストグラム
            ax.hist(
                volume_data,
                bins=30,
                color=self.palette.get_color("volume"),
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            # 統計値
            mean_volume = volume_data.mean()
            median_volume = volume_data.median()

            ax.axvline(
                mean_volume,
                color=self.palette.get_color("mean"),
                linestyle="--",
                linewidth=2,
                label=f"平均: {mean_volume:,.0f}",
            )
            ax.axvline(
                median_volume,
                color=self.palette.get_color("median"),
                linestyle=":",
                linewidth=2,
                label=f"中央値: {median_volume:,.0f}",
            )

            self.apply_common_styling(ax, title="出来高分布", xlabel="出来高", ylabel="頻度")

    def _plot_volume_price_correlation(self, ax, data: pd.DataFrame, volume_analysis: Dict) -> None:
        """
        出来高・価格相関プロット

        Args:
            ax: 描画軸
            data: データ
            volume_analysis: 出来高分析結果
        """
        if "Volume" in data.columns and "Close" in data.columns:
            volume_data = data["Volume"]
            price_data = data["Close"]

            # 散布図
            ax.scatter(
                volume_data,
                price_data,
                alpha=0.6,
                color=self.palette.get_color("correlation"),
                s=20,
            )

            # 相関係数
            correlation = np.corrcoef(volume_data, price_data)[0, 1]
            ax.text(
                0.05,
                0.95,
                f"相関係数: {correlation:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            self.apply_common_styling(ax, title="出来高・価格相関", xlabel="出来高", ylabel="価格")

    def _plot_volume_statistics(self, ax, data: pd.DataFrame, volume_analysis: Dict) -> None:
        """
        出来高統計プロット

        Args:
            ax: 描画軸
            data: データ
            volume_analysis: 出来高分析結果
        """
        if "volume_statistics" in volume_analysis:
            stats = volume_analysis["volume_statistics"]

            stat_names = []
            stat_values = []

            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    stat_names.append(key)
                    stat_values.append(value)

            if stat_names:
                colors = self.palette.get_categorical_palette(len(stat_names))
                bars = ax.bar(stat_names, stat_values, color=colors)

                # 値をバーの上に表示
                for bar, value in zip(bars, stat_values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + height * 0.01,
                        f"{value:.2e}" if value > 1000 else f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                self.apply_common_styling(ax, title="出来高統計", ylabel="値")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        elif "Volume" in data.columns:
            # 基本統計量を計算
            volume_data = data["Volume"]
            stats = {
                "平均": volume_data.mean(),
                "中央値": volume_data.median(),
                "標準偏差": volume_data.std(),
                "最大値": volume_data.max(),
                "最小値": volume_data.min(),
            }

            stat_names = list(stats.keys())
            stat_values = list(stats.values())

            colors = self.palette.get_categorical_palette(len(stat_names))
            bars = ax.bar(stat_names, stat_values, color=colors)

            for bar, value in zip(bars, stat_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{value:.0f}" if value > 100 else f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            self.apply_common_styling(ax, title="出来高基本統計", ylabel="値")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def export_volume_report(
        self, data: pd.DataFrame, volume_analysis: Dict, symbol: str = "STOCK"
    ) -> Dict[str, str]:
        """
        出来高分析レポート出力

        Args:
            data: データ
            volume_analysis: 出来高分析結果
            symbol: 銘柄シンボル

        Returns:
            出力ファイルパス辞書
        """
        exported_files = {}

        try:
            # メインチャート
            main_chart = self.render(data, volume_analysis, symbol=symbol)
            if main_chart:
                exported_files["main_chart"] = main_chart

            # 総合ダッシュボード
            dashboard = self.create_comprehensive_volume_dashboard(data, volume_analysis, symbol)
            if dashboard:
                exported_files["dashboard"] = dashboard

            logger.info(f"出来高分析レポート出力完了 - {len(exported_files)}ファイル")
            return exported_files

        except Exception as e:
            logger.error(f"出来高分析レポート出力エラー: {e}")
            return {}
