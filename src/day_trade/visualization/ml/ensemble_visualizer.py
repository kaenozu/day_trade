"""
アンサンブル学習結果可視化

複数ML手法の統合結果とアンサンブル予測の詳細可視化
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


class EnsembleVisualizer(ChartRenderer):
    """
    アンサンブル学習可視化クラス

    複数ML手法の統合結果とアンサンブル予測の詳細可視化を提供
    """

    def __init__(self, output_dir: str = "output/ensemble_charts", theme: str = "default"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            theme: テーマ
        """
        super().__init__(output_dir, theme)
        logger.info("アンサンブル可視化システム初期化完了")

    def render(self, data: pd.DataFrame, ensemble_results: Dict, **kwargs) -> Optional[str]:
        """
        アンサンブル予測チャート描画

        Args:
            data: 元データ
            ensemble_results: アンサンブル予測結果
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - アンサンブル可視化不可")
            return None

        symbol = kwargs.get("symbol", "UNKNOWN")
        title = kwargs.get("title", f"{symbol} アンサンブル予測")

        fig, axes = self.create_subplots(4, 1, figsize=(15, 16))

        # アンサンブル予測 vs 実価格
        self._plot_ensemble_prediction(axes[0], data, ensemble_results, symbol)

        # 個別モデル予測比較
        self._plot_individual_model_comparison(axes[1], data, ensemble_results)

        # モデル重み・信頼度
        self._plot_model_weights(axes[2], ensemble_results)

        # 予測不確実性・アンサンブルバリアンス
        self._plot_prediction_uncertainty(axes[3], ensemble_results)

        plt.tight_layout()

        filename = kwargs.get("filename", f"ensemble_prediction_{symbol}.png")
        return self.save_figure(fig, filename)

    def _plot_ensemble_prediction(
        self, ax, data: pd.DataFrame, ensemble_results: Dict, symbol: str
    ) -> None:
        """
        アンサンブル予測プロット

        Args:
            ax: 描画軸
            data: 価格データ
            ensemble_results: アンサンブル結果
            symbol: 銘柄シンボル
        """
        # 実際の価格
        if "Close" in data.columns:
            actual_prices = data["Close"].values
            ax.plot(
                actual_prices,
                label="実際の価格",
                color=self.palette.get_color("price"),
                linewidth=2,
            )

        # アンサンブル予測
        if "ensemble_prediction" in ensemble_results:
            ensemble_pred = ensemble_results["ensemble_prediction"]
            prediction_start_idx = len(actual_prices) - len(ensemble_pred)
            pred_x = range(prediction_start_idx, len(actual_prices))

            ax.plot(
                pred_x,
                ensemble_pred,
                label="アンサンブル予測",
                color=self.palette.get_color("prediction"),
                linewidth=3,
                alpha=0.9,
            )

        # アンサンブル信頼区間
        if "ensemble_confidence_interval" in ensemble_results:
            ci = ensemble_results["ensemble_confidence_interval"]
            lower_bound = ci.get("lower", [])
            upper_bound = ci.get("upper", [])

            if lower_bound and upper_bound:
                pred_x = range(len(actual_prices) - len(lower_bound), len(actual_prices))
                ax.fill_between(
                    pred_x,
                    lower_bound,
                    upper_bound,
                    color=self.palette.get_color("confidence"),
                    alpha=0.3,
                    label="アンサンブル信頼区間",
                )

        # 将来予測
        if "future_ensemble_prediction" in ensemble_results:
            future_pred = ensemble_results["future_ensemble_prediction"]
            future_x = range(len(actual_prices), len(actual_prices) + len(future_pred))

            ax.plot(
                future_x,
                future_pred,
                label="将来アンサンブル予測",
                color=self.palette.get_color("prediction"),
                linewidth=3,
                linestyle="--",
            )

            # 将来予測の信頼区間
            if "future_confidence_interval" in ensemble_results:
                future_ci = ensemble_results["future_confidence_interval"]
                future_lower = future_ci.get("lower", [])
                future_upper = future_ci.get("upper", [])

                if future_lower and future_upper:
                    ax.fill_between(
                        future_x,
                        future_lower,
                        future_upper,
                        color=self.palette.get_color("confidence"),
                        alpha=0.2,
                        label="将来予測信頼区間",
                    )

        self.apply_common_styling(
            ax, title=f"{symbol} アンサンブル価格予測", xlabel="時間", ylabel="価格"
        )

    def _plot_individual_model_comparison(
        self, ax, data: pd.DataFrame, ensemble_results: Dict
    ) -> None:
        """
        個別モデル予測比較プロット

        Args:
            ax: 描画軸
            data: データ
            ensemble_results: アンサンブル結果
        """
        if "individual_predictions" in ensemble_results:
            individual_preds = ensemble_results["individual_predictions"]

            # 各モデルの予測をプロット
            model_colors = [
                "lstm",
                "garch",
                "arima",
                "rf",
                "svm",
                "xgboost",
                "neural_net",
                "transformer",
            ]

            for i, (model_name, prediction) in enumerate(individual_preds.items()):
                color = self.palette.get_color(
                    model_colors[i % len(model_colors)] if i < len(model_colors) else "neutral"
                )

                if len(data) >= len(prediction):
                    prediction_start_idx = len(data) - len(prediction)
                    pred_x = range(prediction_start_idx, len(data))

                    ax.plot(
                        pred_x,
                        prediction,
                        label=f"{model_name}",
                        color=color,
                        linewidth=1.5,
                        alpha=0.7,
                    )

            # アンサンブル予測を強調表示
            if "ensemble_prediction" in ensemble_results:
                ensemble_pred = ensemble_results["ensemble_prediction"]
                prediction_start_idx = len(data) - len(ensemble_pred)
                pred_x = range(prediction_start_idx, len(data))

                ax.plot(
                    pred_x,
                    ensemble_pred,
                    label="アンサンブル",
                    color=self.palette.get_color("prediction"),
                    linewidth=3,
                )

        self.apply_common_styling(ax, title="個別モデル予測比較", xlabel="時間", ylabel="予測価格")

    def _plot_model_weights(self, ax, ensemble_results: Dict) -> None:
        """
        モデル重みプロット

        Args:
            ax: 描画軸
            ensemble_results: アンサンブル結果
        """
        if "model_weights" in ensemble_results:
            weights = ensemble_results["model_weights"]

            if isinstance(weights, dict):
                # 静的重みの場合
                model_names = list(weights.keys())
                weight_values = list(weights.values())

                colors = self.palette.get_categorical_palette(len(model_names))
                bars = ax.bar(model_names, weight_values, color=colors)

                # 重みをバーの上に表示
                for bar, weight in zip(bars, weight_values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{weight:.3f}",
                        ha="center",
                        va="bottom",
                    )

                self.apply_common_styling(ax, title="モデル重み", ylabel="重み")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            elif isinstance(weights, pd.DataFrame):
                # 動的重みの場合（時系列）
                for column in weights.columns:
                    ax.plot(
                        weights[column],
                        label=column,
                        linewidth=2,
                    )

                self.apply_common_styling(ax, title="動的モデル重み", xlabel="時間", ylabel="重み")

        elif "model_performance" in ensemble_results:
            # パフォーマンスベースの重み可視化
            performance = ensemble_results["model_performance"]

            if isinstance(performance, dict):
                model_names = list(performance.keys())
                perf_values = list(performance.values())

                colors = self.palette.get_categorical_palette(len(model_names))
                bars = ax.bar(model_names, perf_values, color=colors)

                # パフォーマンス値を表示
                for bar, perf in zip(bars, perf_values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + height * 0.01,
                        f"{perf:.3f}",
                        ha="center",
                        va="bottom",
                    )

                self.apply_common_styling(ax, title="モデルパフォーマンス", ylabel="スコア")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_prediction_uncertainty(self, ax, ensemble_results: Dict) -> None:
        """
        予測不確実性プロット

        Args:
            ax: 描画軸
            ensemble_results: アンサンブル結果
        """
        if "prediction_uncertainty" in ensemble_results:
            uncertainty = ensemble_results["prediction_uncertainty"]

            # 不確実性の時系列
            ax.plot(
                uncertainty,
                color=self.palette.get_color("vix"),
                linewidth=2,
                label="予測不確実性",
            )

            # 不確実性閾値
            mean_uncertainty = np.mean(uncertainty)
            ax.axhline(
                mean_uncertainty,
                color="gray",
                linestyle="-",
                alpha=0.5,
                label=f"平均不確実性: {mean_uncertainty:.3f}",
            )

            # 高不確実性期間を強調
            high_uncertainty_threshold = np.percentile(uncertainty, 75)
            high_uncertainty_mask = np.array(uncertainty) > high_uncertainty_threshold

            if np.any(high_uncertainty_mask):
                high_uncertainty_periods = np.where(high_uncertainty_mask)[0]
                ax.scatter(
                    high_uncertainty_periods,
                    np.array(uncertainty)[high_uncertainty_periods],
                    color=self.palette.get_color("bearish"),
                    s=50,
                    alpha=0.7,
                    label="高不確実性期間",
                )

        elif "ensemble_variance" in ensemble_results:
            # アンサンブル分散による不確実性
            variance = ensemble_results["ensemble_variance"]
            std_dev = np.sqrt(variance)

            ax.plot(
                std_dev,
                color=self.palette.get_color("vix"),
                linewidth=2,
                label="アンサンブル標準偏差",
            )

            # 分散成分分解
            if "variance_components" in ensemble_results:
                components = ensemble_results["variance_components"]

                for component_name, component_values in components.items():
                    ax.plot(
                        component_values,
                        linewidth=1.5,
                        alpha=0.7,
                        label=f"{component_name}分散",
                        linestyle="--",
                    )

        self.apply_common_styling(
            ax, title="予測不確実性・アンサンブル分散", xlabel="時間", ylabel="不確実性"
        )

    def create_ensemble_analysis_dashboard(
        self, data: pd.DataFrame, ensemble_results: Dict, symbol: str = "STOCK"
    ) -> str:
        """
        アンサンブル分析ダッシュボード作成

        Args:
            data: 価格データ
            ensemble_results: アンサンブル結果
            symbol: 銘柄シンボル

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - ダッシュボード作成不可")
            return ""

        fig = plt.figure(figsize=(20, 15))

        # グリッドレイアウト設定
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])

        # メインアンサンブル予測
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_ensemble_prediction(ax_main, data, ensemble_results, symbol)

        # 個別モデル比較
        ax_models = fig.add_subplot(gs[1, :2])
        self._plot_individual_model_comparison(ax_models, data, ensemble_results)

        # モデル重み
        ax_weights = fig.add_subplot(gs[1, 2])
        self._plot_model_weights(ax_weights, ensemble_results)

        # 不確実性分析
        ax_uncertainty = fig.add_subplot(gs[2, :2])
        self._plot_prediction_uncertainty(ax_uncertainty, ensemble_results)

        # アンサンブル統計
        ax_stats = fig.add_subplot(gs[2, 2])
        self._plot_ensemble_statistics(ax_stats, ensemble_results)

        plt.suptitle(f"{symbol} アンサンブル分析ダッシュボード", fontsize=16, fontweight="bold")
        plt.tight_layout()

        filename = f"ensemble_dashboard_{symbol}.png"
        return self.save_figure(fig, filename, dpi=300)

    def _plot_ensemble_statistics(self, ax, ensemble_results: Dict) -> None:
        """
        アンサンブル統計プロット

        Args:
            ax: 描画軸
            ensemble_results: アンサンブル結果
        """
        if "ensemble_statistics" in ensemble_results:
            stats = ensemble_results["ensemble_statistics"]

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

                self.apply_common_styling(ax, title="アンサンブル統計", ylabel="値")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def plot_model_diversity(self, ensemble_results: Dict, **kwargs) -> Optional[str]:
        """
        モデル多様性可視化

        Args:
            ensemble_results: アンサンブル結果
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - 多様性可視化不可")
            return None

        fig, axes = self.create_subplots(2, 2, figsize=(15, 10))

        # 予測相関マトリックス
        if "prediction_correlation_matrix" in ensemble_results:
            corr_matrix = ensemble_results["prediction_correlation_matrix"]
            im = axes[0, 0].imshow(corr_matrix, cmap="RdBu", vmin=-1, vmax=1)
            axes[0, 0].set_title("予測相関マトリックス")

            # 相関値をテキスト表示
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    axes[0, 0].text(
                        j,
                        i,
                        f"{corr_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                    )

            plt.colorbar(im, ax=axes[0, 0])

        # 多様性指標の時系列
        if "diversity_metrics" in ensemble_results:
            diversity = ensemble_results["diversity_metrics"]

            for metric_name, values in diversity.items():
                axes[0, 1].plot(values, label=metric_name, linewidth=2)

            axes[0, 1].set_title("多様性指標推移")
            axes[0, 1].legend()

        # 予測分散分解
        if "prediction_variance_decomposition" in ensemble_results:
            var_decomp = ensemble_results["prediction_variance_decomposition"]

            components = list(var_decomp.keys())
            values = list(var_decomp.values())

            axes[1, 0].pie(
                values,
                labels=components,
                autopct="%1.1f%%",
                colors=self.palette.get_categorical_palette(len(components)),
            )
            axes[1, 0].set_title("予測分散分解")

        # アンサンブルサイズ vs パフォーマンス
        if "ensemble_size_performance" in ensemble_results:
            size_perf = ensemble_results["ensemble_size_performance"]
            sizes = size_perf.get("sizes", [])
            performance = size_perf.get("performance", [])

            if sizes and performance:
                axes[1, 1].plot(
                    sizes,
                    performance,
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    color=self.palette.get_color("prediction"),
                )
                axes[1, 1].set_title("アンサンブルサイズ vs パフォーマンス")
                axes[1, 1].set_xlabel("アンサンブルサイズ")
                axes[1, 1].set_ylabel("パフォーマンス")

        plt.tight_layout()

        filename = kwargs.get("filename", "model_diversity_analysis.png")
        return self.save_figure(fig, filename)

    def export_ensemble_report(
        self, data: pd.DataFrame, ensemble_results: Dict, symbol: str = "STOCK"
    ) -> Dict[str, str]:
        """
        アンサンブル分析レポート出力

        Args:
            data: データ
            ensemble_results: アンサンブル結果
            symbol: 銘柄シンボル

        Returns:
            出力ファイルパス辞書
        """
        exported_files = {}

        try:
            # メインチャート
            main_chart = self.render(data, ensemble_results, symbol=symbol)
            if main_chart:
                exported_files["main_chart"] = main_chart

            # ダッシュボード
            dashboard = self.create_ensemble_analysis_dashboard(data, ensemble_results, symbol)
            if dashboard:
                exported_files["dashboard"] = dashboard

            # モデル多様性分析
            diversity_chart = self.plot_model_diversity(
                ensemble_results, filename=f"ensemble_diversity_{symbol}.png"
            )
            if diversity_chart:
                exported_files["diversity_analysis"] = diversity_chart

            logger.info(f"アンサンブル分析レポート出力完了 - {len(exported_files)}ファイル")
            return exported_files

        except Exception as e:
            logger.error(f"アンサンブル分析レポート出力エラー: {e}")
            return {}
