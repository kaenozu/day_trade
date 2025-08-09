"""
GARCH/ボラティリティモデル結果可視化

GARCHモデルによるボラティリティ予測結果の詳細可視化
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
from ..base.chart_renderer import ChartRenderer

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未インストール")


class GARCHVisualizer(ChartRenderer):
    """
    GARCH/ボラティリティ可視化クラス

    GARCHモデルによるボラティリティ予測の詳細可視化を提供
    """

    def __init__(self, output_dir: str = "output/garch_charts", theme: str = "default"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            theme: テーマ
        """
        super().__init__(output_dir, theme)
        logger.info("GARCH可視化システム初期化完了")

    def render(
        self, data: pd.DataFrame, garch_results: Dict, **kwargs
    ) -> Optional[str]:
        """
        GARCHボラティリティチャート描画

        Args:
            data: 元データ
            garch_results: GARCH予測結果
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - GARCH可視化不可")
            return None

        symbol = kwargs.get("symbol", "UNKNOWN")
        title = kwargs.get("title", f"{symbol} GARCHボラティリティ予測")

        fig, axes = self.create_subplots(4, 1, figsize=(15, 16))

        # 価格とボラティリティ
        self._plot_price_and_volatility(axes[0], data, garch_results, symbol)

        # 条件付きボラティリティ
        self._plot_conditional_volatility(axes[1], data, garch_results)

        # ボラティリティ分解
        self._plot_volatility_decomposition(axes[2], garch_results)

        # リスク指標
        self._plot_risk_metrics(axes[3], garch_results)

        plt.tight_layout()

        filename = kwargs.get("filename", f"garch_analysis_{symbol}.png")
        return self.save_figure(fig, filename)

    def _plot_price_and_volatility(
        self, ax, data: pd.DataFrame, garch_results: Dict, symbol: str
    ) -> None:
        """
        価格とボラティリティプロット

        Args:
            ax: 描画軸
            data: 価格データ
            garch_results: GARCH結果
            symbol: 銘柄シンボル
        """
        # 価格データ（左軸）
        ax2 = ax.twinx()

        if "Close" in data.columns:
            ax2.plot(
                data["Close"],
                label="価格",
                color=self.palette.get_color("price"),
                linewidth=1.5,
                alpha=0.7,
            )
            ax2.set_ylabel("価格")

        # 実現ボラティリティ（右軸）
        if "realized_volatility" in garch_results:
            realized_vol = garch_results["realized_volatility"]
            ax.plot(
                realized_vol,
                label="実現ボラティリティ",
                color=self.palette.get_color("volatility"),
                linewidth=2,
            )

        # GARCH条件付きボラティリティ
        if "conditional_volatility" in garch_results:
            cond_vol = garch_results["conditional_volatility"]
            ax.plot(
                cond_vol,
                label="GARCH条件付きボラティリティ",
                color=self.palette.get_color("garch"),
                linewidth=2,
            )

        # 予測ボラティリティ
        if "volatility_forecast" in garch_results:
            forecast_vol = garch_results["volatility_forecast"]
            forecast_x = range(
                len(realized_vol), len(realized_vol) + len(forecast_vol)
            )
            ax.plot(
                forecast_x,
                forecast_vol,
                label="ボラティリティ予測",
                color=self.palette.get_color("prediction"),
                linewidth=2,
                linestyle="--",
            )

        self.apply_common_styling(
            ax, title=f"{symbol} 価格・ボラティリティ", xlabel="時間", ylabel="ボラティリティ"
        )

    def _plot_conditional_volatility(
        self, ax, data: pd.DataFrame, garch_results: Dict
    ) -> None:
        """
        条件付きボラティリティプロット

        Args:
            ax: 描画軸
            data: データ
            garch_results: GARCH結果
        """
        if "conditional_volatility" in garch_results:
            cond_vol = garch_results["conditional_volatility"]
            ax.plot(
                cond_vol,
                color=self.palette.get_color("garch"),
                linewidth=2,
                label="条件付きボラティリティ",
            )

            # ボラティリティクラスター可視化
            if "volatility_clusters" in garch_results:
                clusters = garch_results["volatility_clusters"]
                high_vol_periods = np.where(
                    np.array(cond_vol) > np.percentile(cond_vol, 75)
                )[0]

                ax.scatter(
                    high_vol_periods,
                    np.array(cond_vol)[high_vol_periods],
                    color=self.palette.get_color("bearish"),
                    alpha=0.7,
                    s=30,
                    label="高ボラティリティ期間",
                )

            # 統計的閾値
            mean_vol = np.mean(cond_vol)
            std_vol = np.std(cond_vol)

            ax.axhline(
                mean_vol,
                color="gray",
                linestyle="-",
                alpha=0.5,
                label=f"平均: {mean_vol:.3f}",
            )
            ax.axhline(
                mean_vol + 2 * std_vol,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"+2σ: {mean_vol + 2*std_vol:.3f}",
            )

            self.apply_common_styling(
                ax,
                title="GARCH条件付きボラティリティ",
                xlabel="時間",
                ylabel="ボラティリティ",
            )

    def _plot_volatility_decomposition(self, ax, garch_results: Dict) -> None:
        """
        ボラティリティ分解プロット

        Args:
            ax: 描画軸
            garch_results: GARCH結果
        """
        if "volatility_components" in garch_results:
            components = garch_results["volatility_components"]

            # 長期・短期コンポーネント
            if "long_term" in components and "short_term" in components:
                long_term = components["long_term"]
                short_term = components["short_term"]

                ax.plot(
                    long_term,
                    label="長期ボラティリティ",
                    color=self.palette.get_color("ma_long"),
                    linewidth=2,
                )
                ax.plot(
                    short_term,
                    label="短期ボラティリティ",
                    color=self.palette.get_color("ma_short"),
                    linewidth=2,
                )

                # 比率プロット
                if len(long_term) == len(short_term):
                    ratio = np.array(short_term) / np.array(long_term)
                    ax2 = ax.twinx()
                    ax2.plot(
                        ratio,
                        label="短期/長期比",
                        color=self.palette.get_color("vix"),
                        linewidth=1,
                        alpha=0.7,
                        linestyle=":",
                    )
                    ax2.set_ylabel("短期/長期比")

        elif "garch_parameters" in garch_results:
            # GARCHパラメータの時系列変化
            params = garch_results["garch_parameters"]

            if "alpha" in params and "beta" in params:
                alpha = params["alpha"]
                beta = params["beta"]

                ax.plot(
                    alpha,
                    label=f"α (短期) 平均: {np.mean(alpha):.3f}",
                    color=self.palette.get_color("ma_short"),
                    linewidth=2,
                )
                ax.plot(
                    beta,
                    label=f"β (長期) 平均: {np.mean(beta):.3f}",
                    color=self.palette.get_color("ma_long"),
                    linewidth=2,
                )

                # 持続性指標
                persistence = np.array(alpha) + np.array(beta)
                ax.plot(
                    persistence,
                    label=f"持続性 (α+β) 平均: {np.mean(persistence):.3f}",
                    color=self.palette.get_color("neutral"),
                    linewidth=2,
                    linestyle="--",
                )

        self.apply_common_styling(
            ax, title="ボラティリティ分解", xlabel="時間", ylabel="パラメータ値"
        )

    def _plot_risk_metrics(self, ax, garch_results: Dict) -> None:
        """
        リスク指標プロット

        Args:
            ax: 描画軸
            garch_results: GARCH結果
        """
        if "risk_metrics" in garch_results:
            risk_metrics = garch_results["risk_metrics"]

            # VaR（Value at Risk）
            if "var_95" in risk_metrics:
                var_95 = risk_metrics["var_95"]
                ax.plot(
                    var_95,
                    label="VaR (95%)",
                    color=self.palette.get_color("bearish"),
                    linewidth=2,
                )

            if "var_99" in risk_metrics:
                var_99 = risk_metrics["var_99"]
                ax.plot(
                    var_99,
                    label="VaR (99%)",
                    color=self.palette.get_color("bearish"),
                    linewidth=2,
                    alpha=0.7,
                )

            # Expected Shortfall
            if "expected_shortfall" in risk_metrics:
                es = risk_metrics["expected_shortfall"]
                ax.plot(
                    es,
                    label="Expected Shortfall",
                    color=self.palette.get_color("neutral"),
                    linewidth=2,
                    linestyle=":",
                )

        else:
            # 代替: 条件付きボラティリティベースのリスク推定
            if "conditional_volatility" in garch_results:
                cond_vol = garch_results["conditional_volatility"]
                # 95% VaR近似（正規分布仮定）
                var_95_approx = -1.645 * np.array(cond_vol)
                ax.plot(
                    var_95_approx,
                    label="VaR (95%) 近似",
                    color=self.palette.get_color("bearish"),
                    linewidth=2,
                )

                # 99% VaR近似
                var_99_approx = -2.326 * np.array(cond_vol)
                ax.plot(
                    var_99_approx,
                    label="VaR (99%) 近似",
                    color=self.palette.get_color("bearish"),
                    linewidth=2,
                    alpha=0.7,
                )

        self.apply_common_styling(
            ax, title="リスク指標", xlabel="時間", ylabel="リスク値"
        )

    def create_garch_analysis_dashboard(
        self, data: pd.DataFrame, garch_results: Dict, symbol: str = "STOCK"
    ) -> str:
        """
        GARCH分析ダッシュボード作成

        Args:
            data: 価格データ
            garch_results: GARCH結果
            symbol: 銘柄シンボル

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - ダッシュボード作成不可")
            return ""

        fig = plt.figure(figsize=(20, 15))

        # グリッドレイアウト設定
        gs = fig.add_gridspec(
            3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1]
        )

        # メインボラティリティチャート
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_price_and_volatility(ax_main, data, garch_results, symbol)

        # 条件付きボラティリティ
        ax_conditional = fig.add_subplot(gs[1, :2])
        self._plot_conditional_volatility(ax_conditional, data, garch_results)

        # ボラティリティ統計
        ax_stats = fig.add_subplot(gs[1, 2])
        self._plot_volatility_statistics(ax_stats, garch_results)

        # パラメータ推定
        ax_params = fig.add_subplot(gs[2, :2])
        self._plot_garch_parameters(ax_params, garch_results)

        # モデル適合度
        ax_goodness = fig.add_subplot(gs[2, 2])
        self._plot_model_goodness(ax_goodness, garch_results)

        plt.suptitle(f"{symbol} GARCH分析ダッシュボード", fontsize=16, fontweight="bold")
        plt.tight_layout()

        filename = f"garch_dashboard_{symbol}.png"
        return self.save_figure(fig, filename, dpi=300)

    def _plot_volatility_statistics(self, ax, garch_results: Dict) -> None:
        """
        ボラティリティ統計プロット

        Args:
            ax: 描画軸
            garch_results: GARCH結果
        """
        if "volatility_statistics" in garch_results:
            stats = garch_results["volatility_statistics"]

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
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

                self.apply_common_styling(ax, title="ボラティリティ統計", ylabel="値")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_garch_parameters(self, ax, garch_results: Dict) -> None:
        """
        GARCHパラメータプロット

        Args:
            ax: 描画軸
            garch_results: GARCH結果
        """
        if "parameter_evolution" in garch_results:
            param_evolution = garch_results["parameter_evolution"]

            for param_name, values in param_evolution.items():
                ax.plot(
                    values,
                    label=f"{param_name}",
                    linewidth=2,
                )

        elif "garch_parameters" in garch_results:
            params = garch_results["garch_parameters"]

            # 静的パラメータの場合は水平線で表示
            y_pos = 0
            for param_name, value in params.items():
                if isinstance(value, (int, float)):
                    ax.axhline(
                        y=value,
                        label=f"{param_name}: {value:.4f}",
                        linewidth=2,
                    )
                    y_pos += 1

        self.apply_common_styling(ax, title="GARCHパラメータ", xlabel="時間", ylabel="値")

    def _plot_model_goodness(self, ax, garch_results: Dict) -> None:
        """
        モデル適合度プロット

        Args:
            ax: 描画軸
            garch_results: GARCH結果
        """
        if "goodness_of_fit" in garch_results:
            gof = garch_results["goodness_of_fit"]

            metrics = []
            values = []

            for metric, value in gof.items():
                if isinstance(value, (int, float)):
                    metrics.append(metric)
                    values.append(value)

            if metrics:
                colors = self.palette.get_categorical_palette(len(metrics))
                bars = ax.bar(metrics, values, color=colors)

                # 値をバーの上に表示
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + height * 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

                self.apply_common_styling(ax, title="適合度指標", ylabel="値")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def export_garch_report(
        self, data: pd.DataFrame, garch_results: Dict, symbol: str = "STOCK"
    ) -> Dict[str, str]:
        """
        GARCH分析レポート出力

        Args:
            data: データ
            garch_results: GARCH結果
            symbol: 銘柄シンボル

        Returns:
            出力ファイルパス辞書
        """
        exported_files = {}

        try:
            # メインチャート
            main_chart = self.render(data, garch_results, symbol=symbol)
            if main_chart:
                exported_files["main_chart"] = main_chart

            # ダッシュボード
            dashboard = self.create_garch_analysis_dashboard(
                data, garch_results, symbol
            )
            if dashboard:
                exported_files["dashboard"] = dashboard

            logger.info(f"GARCH分析レポート出力完了 - {len(exported_files)}ファイル")
            return exported_files

        except Exception as e:
            logger.error(f"GARCH分析レポート出力エラー: {e}")
            return {}
