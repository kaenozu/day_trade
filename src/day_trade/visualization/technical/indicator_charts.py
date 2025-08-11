"""
テクニカル指標チャート

RSI、MACD、ボリンジャーバンド等のテクニカル指標可視化
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

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未インストール")


class IndicatorCharts(ChartRenderer):
    """
    テクニカル指標チャートクラス

    RSI、MACD、ボリンジャーバンド等の詳細可視化を提供
    """

    def __init__(
        self, output_dir: str = "output/technical_charts", theme: str = "default"
    ):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            theme: テーマ
        """
        super().__init__(output_dir, theme)
        logger.info("テクニカル指標可視化システム初期化完了")

    def render(self, data: pd.DataFrame, indicators: Dict, **kwargs) -> Optional[str]:
        """
        テクニカル指標チャート描画

        Args:
            data: 元データ
            indicators: 計算済み指標辞書
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - テクニカル指標可視化不可")
            return None

        symbol = kwargs.get("symbol", "UNKNOWN")
        title = kwargs.get("title", f"{symbol} テクニカル分析")

        fig, axes = self.create_subplots(4, 1, figsize=(15, 16))

        # 価格・移動平均・ボリンジャーバンド
        self._plot_price_indicators(axes[0], data, indicators, symbol)

        # RSI
        self._plot_rsi(axes[1], indicators)

        # MACD
        self._plot_macd(axes[2], indicators)

        # ストキャスティクス・その他オシレーター
        self._plot_oscillators(axes[3], indicators)

        plt.tight_layout()

        filename = kwargs.get("filename", f"technical_analysis_{symbol}.png")
        return self.save_figure(fig, filename)

    def _plot_price_indicators(
        self, ax, data: pd.DataFrame, indicators: Dict, symbol: str
    ) -> None:
        """
        価格・移動平均・ボリンジャーバンドプロット

        Args:
            ax: 描画軸
            data: 価格データ
            indicators: 指標辞書
            symbol: 銘柄シンボル
        """
        # ローソク足データ
        if "Close" in data.columns:
            ax.plot(
                data["Close"],
                label="終値",
                color=self.palette.get_color("price"),
                linewidth=2,
            )

        # 移動平均線
        if "MA" in indicators:
            ma_data = indicators["MA"]
            for period, ma_values in ma_data.items():
                if period == "5":
                    color = self.palette.get_color("ma_short")
                    label = "MA(5)"
                elif period == "25":
                    color = self.palette.get_color("ma_medium")
                    label = "MA(25)"
                elif period == "75":
                    color = self.palette.get_color("ma_long")
                    label = "MA(75)"
                else:
                    color = self.palette.get_color("neutral")
                    label = f"MA({period})"

                ax.plot(ma_values, label=label, color=color, linewidth=1.5, alpha=0.8)

        # ボリンジャーバンド
        if "Bollinger_Bands" in indicators:
            bb_data = indicators["Bollinger_Bands"]

            if "upper" in bb_data and "lower" in bb_data:
                upper_band = bb_data["upper"]
                lower_band = bb_data["lower"]

                ax.plot(
                    upper_band,
                    label="ボリンジャー上限",
                    color=self.palette.get_color("bb_upper"),
                    linewidth=1,
                    alpha=0.7,
                )
                ax.plot(
                    lower_band,
                    label="ボリンジャー下限",
                    color=self.palette.get_color("bb_lower"),
                    linewidth=1,
                    alpha=0.7,
                )

                # バンド内を塗りつぶし
                ax.fill_between(
                    range(len(upper_band)),
                    upper_band,
                    lower_band,
                    color=self.palette.get_color("bb_fill"),
                    alpha=0.1,
                    label="ボリンジャーバンド",
                )

            # バンドウォーク検出
            if "band_walk" in bb_data:
                band_walk_periods = bb_data["band_walk"]
                if "Close" in data.columns and band_walk_periods:
                    close_prices = data["Close"].values
                    for period_start, period_end in band_walk_periods:
                        ax.axvspan(
                            period_start,
                            period_end,
                            color=self.palette.get_color("warning"),
                            alpha=0.3,
                        )

        # サポート・レジスタンスライン
        if "Support_Resistance" in indicators:
            sr_data = indicators["Support_Resistance"]

            if "support_levels" in sr_data:
                for support_level in sr_data["support_levels"]:
                    ax.axhline(
                        support_level,
                        color=self.palette.get_color("bullish"),
                        linestyle="--",
                        alpha=0.6,
                        linewidth=1,
                    )

            if "resistance_levels" in sr_data:
                for resistance_level in sr_data["resistance_levels"]:
                    ax.axhline(
                        resistance_level,
                        color=self.palette.get_color("bearish"),
                        linestyle="--",
                        alpha=0.6,
                        linewidth=1,
                    )

        self.apply_common_styling(
            ax, title=f"{symbol} 価格・トレンド指標", xlabel="時間", ylabel="価格"
        )

    def _plot_rsi(self, ax, indicators: Dict) -> None:
        """
        RSIプロット

        Args:
            ax: 描画軸
            indicators: 指標辞書
        """
        if "RSI" in indicators:
            rsi_data = indicators["RSI"]

            if "rsi" in rsi_data:
                rsi_values = rsi_data["rsi"]
                ax.plot(
                    rsi_values,
                    label="RSI(14)",
                    color=self.palette.get_color("rsi"),
                    linewidth=2,
                )

                # 過買い・過売りライン
                ax.axhline(
                    70,
                    color=self.palette.get_color("bearish"),
                    linestyle="--",
                    alpha=0.7,
                )
                ax.axhline(
                    30,
                    color=self.palette.get_color("bullish"),
                    linestyle="--",
                    alpha=0.7,
                )

                # 過買い・過売り領域を塗りつぶし
                ax.fill_between(
                    range(len(rsi_values)),
                    70,
                    100,
                    where=np.array(rsi_values) > 70,
                    color=self.palette.get_color("bearish"),
                    alpha=0.2,
                    interpolate=True,
                    label="過買い領域",
                )
                ax.fill_between(
                    range(len(rsi_values)),
                    0,
                    30,
                    where=np.array(rsi_values) < 30,
                    color=self.palette.get_color("bullish"),
                    alpha=0.2,
                    interpolate=True,
                    label="過売り領域",
                )

                # ダイバージェンス検出
                if "divergences" in rsi_data:
                    divergences = rsi_data["divergences"]
                    for div_type, div_points in divergences.items():
                        color = (
                            self.palette.get_color("bullish")
                            if div_type == "bullish"
                            else self.palette.get_color("bearish")
                        )
                        for start_idx, end_idx in div_points:
                            ax.plot(
                                [start_idx, end_idx],
                                [rsi_values[start_idx], rsi_values[end_idx]],
                                color=color,
                                linestyle=":",
                                linewidth=2,
                                alpha=0.8,
                            )

                ax.set_ylim(0, 100)

        self.apply_common_styling(
            ax, title="RSI (Relative Strength Index)", ylabel="RSI"
        )

    def _plot_macd(self, ax, indicators: Dict) -> None:
        """
        MACDプロット

        Args:
            ax: 描画軸
            indicators: 指標辞書
        """
        if "MACD" in indicators:
            macd_data = indicators["MACD"]

            # MACD線
            if "macd" in macd_data:
                macd_line = macd_data["macd"]
                ax.plot(
                    macd_line,
                    label="MACD",
                    color=self.palette.get_color("macd_line"),
                    linewidth=2,
                )

            # シグナル線
            if "signal" in macd_data:
                signal_line = macd_data["signal"]
                ax.plot(
                    signal_line,
                    label="シグナル",
                    color=self.palette.get_color("macd_signal"),
                    linewidth=2,
                )

            # MACDヒストグラム
            if "histogram" in macd_data:
                histogram = macd_data["histogram"]
                colors = [
                    self.palette.get_color("bullish")
                    if h >= 0
                    else self.palette.get_color("bearish")
                    for h in histogram
                ]

                ax.bar(
                    range(len(histogram)),
                    histogram,
                    color=colors,
                    alpha=0.6,
                    label="ヒストグラム",
                )

            # ゼロライン
            ax.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=1)

            # ゴールデンクロス・デッドクロス検出
            if "crossovers" in macd_data:
                crossovers = macd_data["crossovers"]
                if "golden_cross" in crossovers:
                    for gc_point in crossovers["golden_cross"]:
                        ax.axvline(
                            gc_point,
                            color=self.palette.get_color("bullish"),
                            linestyle=":",
                            alpha=0.8,
                            linewidth=2,
                        )

                if "dead_cross" in crossovers:
                    for dc_point in crossovers["dead_cross"]:
                        ax.axvline(
                            dc_point,
                            color=self.palette.get_color("bearish"),
                            linestyle=":",
                            alpha=0.8,
                            linewidth=2,
                        )

        self.apply_common_styling(ax, title="MACD", ylabel="MACD値")

    def _plot_oscillators(self, ax, indicators: Dict) -> None:
        """
        オシレーター系指標プロット

        Args:
            ax: 描画軸
            indicators: 指標辞書
        """
        plotted_indicators = []

        # ストキャスティクス
        if "Stochastic" in indicators:
            stoch_data = indicators["Stochastic"]

            if "k" in stoch_data and "d" in stoch_data:
                k_values = stoch_data["k"]
                d_values = stoch_data["d"]

                ax.plot(
                    k_values,
                    label="%K",
                    color=self.palette.get_color("stoch_k"),
                    linewidth=2,
                )
                ax.plot(
                    d_values,
                    label="%D",
                    color=self.palette.get_color("stoch_d"),
                    linewidth=2,
                )

                plotted_indicators.append("Stochastic")

        # CCI (Commodity Channel Index)
        if "CCI" in indicators:
            cci_data = indicators["CCI"]
            if "cci" in cci_data:
                cci_values = cci_data["cci"]
                ax2 = ax.twinx() if plotted_indicators else ax

                ax2.plot(
                    cci_values,
                    label="CCI",
                    color=self.palette.get_color("cci"),
                    linewidth=2,
                    alpha=0.7,
                )

                # CCI過買い・過売りライン
                ax2.axhline(100, color="red", linestyle="--", alpha=0.5)
                ax2.axhline(-100, color="green", linestyle="--", alpha=0.5)

                if plotted_indicators:
                    ax2.set_ylabel("CCI")

                plotted_indicators.append("CCI")

        # Williams %R
        if "Williams_R" in indicators:
            wr_data = indicators["Williams_R"]
            if "williams_r" in wr_data:
                wr_values = wr_data["williams_r"]
                ax3 = ax.twinx() if len(plotted_indicators) >= 1 else ax

                ax3.plot(
                    wr_values,
                    label="Williams %R",
                    color=self.palette.get_color("williams_r"),
                    linewidth=1.5,
                    alpha=0.7,
                )

                # Williams %R過買い・過売りライン
                ax3.axhline(-20, color="red", linestyle="--", alpha=0.5)
                ax3.axhline(-80, color="green", linestyle="--", alpha=0.5)

                if len(plotted_indicators) >= 1:
                    ax3.set_ylabel("Williams %R")
                    ax3.spines["right"].set_position(("outward", 60))

                plotted_indicators.append("Williams %R")

        # デフォルトの場合：ストキャスティクス範囲設定
        if "Stochastic" in plotted_indicators:
            ax.set_ylim(0, 100)
            ax.axhline(80, color="red", linestyle="--", alpha=0.7)
            ax.axhline(20, color="green", linestyle="--", alpha=0.7)

        title = "オシレーター系指標"
        if plotted_indicators:
            title += f" ({', '.join(plotted_indicators)})"

        self.apply_common_styling(ax, title=title, ylabel="値")

    def create_comprehensive_technical_dashboard(
        self, data: pd.DataFrame, indicators: Dict, symbol: str = "STOCK"
    ) -> str:
        """
        総合テクニカル分析ダッシュボード作成

        Args:
            data: 価格データ
            indicators: 指標辞書
            symbol: 銘柄シンボル

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - ダッシュボード作成不可")
            return ""

        fig = plt.figure(figsize=(20, 24))

        # グリッドレイアウト設定（6行2列）
        gs = fig.add_gridspec(6, 2, height_ratios=[3, 2, 2, 2, 2, 1])

        # メイン価格チャート（全幅）
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_price_indicators(ax_main, data, indicators, symbol)

        # RSI
        ax_rsi = fig.add_subplot(gs[1, 0])
        self._plot_rsi(ax_rsi, indicators)

        # MACD
        ax_macd = fig.add_subplot(gs[1, 1])
        self._plot_macd(ax_macd, indicators)

        # ストキャスティクス・オシレーター
        ax_oscillators = fig.add_subplot(gs[2, 0])
        self._plot_oscillators(ax_oscillators, indicators)

        # ボラティリティ指標
        ax_volatility = fig.add_subplot(gs[2, 1])
        self._plot_volatility_indicators(ax_volatility, indicators)

        # トレンド指標
        ax_trend = fig.add_subplot(gs[3, 0])
        self._plot_trend_indicators(ax_trend, indicators)

        # モメンタム指標
        ax_momentum = fig.add_subplot(gs[3, 1])
        self._plot_momentum_indicators(ax_momentum, indicators)

        # 出来高関連指標
        ax_volume = fig.add_subplot(gs[4, :])
        self._plot_volume_indicators(ax_volume, data, indicators)

        # 指標サマリー
        ax_summary = fig.add_subplot(gs[5, :])
        self._plot_indicator_summary(ax_summary, indicators)

        plt.suptitle(
            f"{symbol} 総合テクニカル分析ダッシュボード", fontsize=18, fontweight="bold"
        )
        plt.tight_layout()

        filename = f"comprehensive_technical_{symbol}.png"
        return self.save_figure(fig, filename, dpi=300)

    def _plot_volatility_indicators(self, ax, indicators: Dict) -> None:
        """
        ボラティリティ指標プロット

        Args:
            ax: 描画軸
            indicators: 指標辞書
        """
        if "ATR" in indicators:
            atr_data = indicators["ATR"]
            if "atr" in atr_data:
                ax.plot(
                    atr_data["atr"],
                    label="ATR",
                    color=self.palette.get_color("atr"),
                    linewidth=2,
                )

        if "Bollinger_Bands" in indicators:
            bb_data = indicators["Bollinger_Bands"]
            if "bandwidth" in bb_data:
                ax.plot(
                    bb_data["bandwidth"],
                    label="BB幅",
                    color=self.palette.get_color("bb_bandwidth"),
                    linewidth=2,
                )

        self.apply_common_styling(ax, title="ボラティリティ指標", ylabel="値")

    def _plot_trend_indicators(self, ax, indicators: Dict) -> None:
        """
        トレンド指標プロット

        Args:
            ax: 描画軸
            indicators: 指標辞書
        """
        if "ADX" in indicators:
            adx_data = indicators["ADX"]
            if "adx" in adx_data:
                ax.plot(
                    adx_data["adx"],
                    label="ADX",
                    color=self.palette.get_color("adx"),
                    linewidth=2,
                )

                ax.axhline(25, color="gray", linestyle="--", alpha=0.5)

        if "Parabolic_SAR" in indicators:
            sar_data = indicators["Parabolic_SAR"]
            if "sar" in sar_data:
                ax.plot(
                    sar_data["sar"],
                    label="SAR",
                    color=self.palette.get_color("sar"),
                    linewidth=1.5,
                    marker="o",
                    markersize=2,
                )

        self.apply_common_styling(ax, title="トレンド指標", ylabel="値")

    def _plot_momentum_indicators(self, ax, indicators: Dict) -> None:
        """
        モメンタム指標プロット

        Args:
            ax: 描画軸
            indicators: 指標辞書
        """
        if "ROC" in indicators:
            roc_data = indicators["ROC"]
            if "roc" in roc_data:
                ax.plot(
                    roc_data["roc"],
                    label="ROC",
                    color=self.palette.get_color("roc"),
                    linewidth=2,
                )

                ax.axhline(0, color="black", linestyle="-", alpha=0.3)

        if "MFI" in indicators:
            mfi_data = indicators["MFI"]
            if "mfi" in mfi_data:
                ax2 = ax.twinx()
                ax2.plot(
                    mfi_data["mfi"],
                    label="MFI",
                    color=self.palette.get_color("mfi"),
                    linewidth=2,
                    alpha=0.7,
                )
                ax2.set_ylabel("MFI")

        self.apply_common_styling(ax, title="モメンタム指標", ylabel="値")

    def _plot_volume_indicators(self, ax, data: pd.DataFrame, indicators: Dict) -> None:
        """
        出来高関連指標プロット

        Args:
            ax: 描画軸
            data: 価格データ
            indicators: 指標辞書
        """
        # 出来高
        if "Volume" in data.columns:
            volume_data = data["Volume"]
            ax.bar(
                range(len(volume_data)),
                volume_data,
                color=self.palette.get_color("volume"),
                alpha=0.6,
                label="出来高",
            )

        # 出来高移動平均
        if "Volume_MA" in indicators:
            vol_ma = indicators["Volume_MA"]
            if "volume_ma" in vol_ma:
                ax.plot(
                    vol_ma["volume_ma"],
                    label="出来高MA",
                    color=self.palette.get_color("volume_ma"),
                    linewidth=2,
                )

        self.apply_common_styling(ax, title="出来高分析", ylabel="出来高")

    def _plot_indicator_summary(self, ax, indicators: Dict) -> None:
        """
        指標サマリープロット

        Args:
            ax: 描画軸
            indicators: 指標辞書
        """
        # 指標別のシグナル強度を集計
        signal_summary = {}

        if "RSI" in indicators:
            rsi_latest = (
                indicators["RSI"].get("rsi", [0])[-1]
                if indicators["RSI"].get("rsi")
                else 50
            )
            if rsi_latest > 70:
                signal_summary["RSI"] = "過買い"
            elif rsi_latest < 30:
                signal_summary["RSI"] = "過売り"
            else:
                signal_summary["RSI"] = "中立"

        if "MACD" in indicators:
            macd_hist = indicators["MACD"].get("histogram", [0])
            if macd_hist:
                latest_hist = macd_hist[-1]
                signal_summary["MACD"] = "強気" if latest_hist > 0 else "弱気"

        # サマリーテーブル風表示
        if signal_summary:
            summary_text = "\n".join([f"{k}: {v}" for k, v in signal_summary.items()])
            ax.text(
                0.1,
                0.5,
                summary_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
            )

        ax.set_title("指標サマリー")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def export_technical_report(
        self, data: pd.DataFrame, indicators: Dict, symbol: str = "STOCK"
    ) -> Dict[str, str]:
        """
        テクニカル分析レポート出力

        Args:
            data: データ
            indicators: 指標辞書
            symbol: 銘柄シンボル

        Returns:
            出力ファイルパス辞書
        """
        exported_files = {}

        try:
            # メインチャート
            main_chart = self.render(data, indicators, symbol=symbol)
            if main_chart:
                exported_files["main_chart"] = main_chart

            # 総合ダッシュボード
            dashboard = self.create_comprehensive_technical_dashboard(
                data, indicators, symbol
            )
            if dashboard:
                exported_files["dashboard"] = dashboard

            logger.info(
                f"テクニカル分析レポート出力完了 - {len(exported_files)}ファイル"
            )
            return exported_files

        except Exception as e:
            logger.error(f"テクニカル分析レポート出力エラー: {e}")
            return {}
