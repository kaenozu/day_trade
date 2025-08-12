"""
LSTM予測結果可視化

LSTM時系列予測の結果を詳細に可視化
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
from ..base.chart_renderer import ChartRenderer

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未インストール")


class LSTMVisualizer(ChartRenderer):
    """
    LSTM予測結果可視化クラス

    LSTM時系列予測の詳細可視化を提供
    """

    def __init__(self, output_dir: str = "output/lstm_charts", theme: str = "default"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            theme: テーマ
        """
        super().__init__(output_dir, theme)
        logger.info("LSTM可視化システム初期化完了")

    def render(self, data: pd.DataFrame, lstm_results: Dict, **kwargs) -> Optional[str]:
        """
        LSTM予測チャート描画

        Args:
            data: 元データ
            lstm_results: LSTM予測結果
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - LSTM可視化不可")
            return None

        symbol = kwargs.get("symbol", "UNKNOWN")
        title = kwargs.get("title", f"{symbol} LSTM価格予測")

        fig, axes = self.create_subplots(3, 1, figsize=(15, 12))

        # メインプライスチャート
        self._plot_price_prediction(axes[0], data, lstm_results, symbol)

        # 予測精度・信頼区間
        self._plot_prediction_accuracy(axes[1], data, lstm_results)

        # 特徴量重要度・注意機構
        self._plot_attention_weights(axes[2], lstm_results)

        plt.tight_layout()

        filename = kwargs.get("filename", f"lstm_prediction_{symbol}.png")
        return self.save_figure(fig, filename)

    def _plot_price_prediction(
        self, ax, data: pd.DataFrame, lstm_results: Dict, symbol: str
    ) -> None:
        """
        価格予測プロット

        Args:
            ax: 描画軸
            data: 価格データ
            lstm_results: LSTM結果
            symbol: 銘柄シンボル
        """
        # 実際の価格
        actual_prices = data["Close"].values
        ax.plot(
            actual_prices,
            label="実際の価格",
            color=self.palette.get_color("price"),
            linewidth=2,
        )

        # LSTM予測
        if "predictions" in lstm_results:
            predictions = lstm_results["predictions"]
            # 予測開始点を実際の価格に接続
            prediction_start_idx = len(actual_prices) - len(predictions)
            pred_x = range(prediction_start_idx, len(actual_prices))

            ax.plot(
                pred_x,
                predictions,
                label="LSTM予測",
                color=self.palette.get_color("lstm"),
                linewidth=2,
                alpha=0.8,
            )

        # 信頼区間
        if "confidence_intervals" in lstm_results:
            intervals = lstm_results["confidence_intervals"]
            lower_bound = intervals.get("lower", [])
            upper_bound = intervals.get("upper", [])

            if lower_bound and upper_bound:
                pred_x = range(
                    len(actual_prices) - len(lower_bound), len(actual_prices)
                )
                ax.fill_between(
                    pred_x,
                    lower_bound,
                    upper_bound,
                    color=self.palette.get_color("confidence"),
                    alpha=0.3,
                    label="95% 信頼区間",
                )

        # 将来予測
        if "future_predictions" in lstm_results:
            future_pred = lstm_results["future_predictions"]
            future_x = range(len(actual_prices), len(actual_prices) + len(future_pred))

            ax.plot(
                future_x,
                future_pred,
                label="将来予測",
                color=self.palette.get_color("prediction"),
                linewidth=2,
                linestyle="--",
            )

        self.apply_common_styling(
            ax, title=f"{symbol} LSTM価格予測", xlabel="時間", ylabel="価格"
        )

    def _plot_prediction_accuracy(
        self, ax, data: pd.DataFrame, lstm_results: Dict
    ) -> None:
        """
        予測精度プロット

        Args:
            ax: 描画軸
            data: データ
            lstm_results: LSTM結果
        """
        if "accuracy_metrics" in lstm_results:
            metrics = lstm_results["accuracy_metrics"]

            # 予測誤差の時系列
            if "prediction_errors" in metrics:
                errors = metrics["prediction_errors"]
                ax.plot(
                    errors,
                    color=self.palette.get_color("bearish"),
                    linewidth=1.5,
                    label="予測誤差",
                )
                ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # 移動平均精度
        if "rolling_accuracy" in lstm_results:
            rolling_acc = lstm_results["rolling_accuracy"]
            ax.plot(
                rolling_acc,
                color=self.palette.get_color("bullish"),
                linewidth=2,
                label="移動平均精度",
            )

        self.apply_common_styling(
            ax, title="LSTM予測精度", xlabel="時間", ylabel="誤差/精度"
        )

    def _plot_attention_weights(self, ax, lstm_results: Dict) -> None:
        """
        注意機構重みプロット

        Args:
            ax: 描画軸
            lstm_results: LSTM結果
        """
        if "attention_weights" in lstm_results:
            attention = lstm_results["attention_weights"]

            # ヒートマップ風の可視化
            if isinstance(attention, np.ndarray) and attention.ndim == 2:
                im = ax.imshow(attention, cmap="Blues", aspect="auto")
                ax.set_title("注意機構重み")
                ax.set_xlabel("時系列ステップ")
                ax.set_ylabel("特徴量")

                # カラーバー
                plt.colorbar(im, ax=ax)

            elif isinstance(attention, (list, np.ndarray)) and len(attention) > 0:
                # 1次元の場合は線グラフ
                ax.plot(
                    attention,
                    color=self.palette.get_color("vix"),
                    linewidth=2,
                    label="注意重み",
                )
                self.apply_common_styling(
                    ax, title="注意機構重み", xlabel="時系列ステップ", ylabel="重み"
                )
        else:
            # 代替: 特徴量重要度
            if "feature_importance" in lstm_results:
                importance = lstm_results["feature_importance"]
                feature_names = lstm_results.get(
                    "feature_names", [f"特徴量{i + 1}" for i in range(len(importance))]
                )

                ax.barh(
                    feature_names, importance, color=self.palette.get_color("neutral")
                )
                self.apply_common_styling(
                    ax, title="特徴量重要度", xlabel="重要度", ylabel="特徴量"
                )

    def create_lstm_analysis_dashboard(
        self, data: pd.DataFrame, lstm_results: Dict, symbol: str = "STOCK"
    ) -> str:
        """
        LSTM分析ダッシュボード作成

        Args:
            data: 価格データ
            lstm_results: LSTM結果
            symbol: 銘柄シンボル

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - ダッシュボード作成不可")
            return ""

        fig = plt.figure(figsize=(20, 15))

        # グリッドレイアウト設定
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 1], width_ratios=[2, 1, 1])

        # メイン価格チャート
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_price_prediction(ax_main, data, lstm_results, symbol)

        # 予測精度
        ax_accuracy = fig.add_subplot(gs[1, :2])
        self._plot_prediction_accuracy(ax_accuracy, data, lstm_results)

        # パフォーマンス指標
        ax_metrics = fig.add_subplot(gs[1, 2])
        self._plot_performance_metrics(ax_metrics, lstm_results)

        # 注意機構/特徴量重要度
        ax_attention = fig.add_subplot(gs[2, :])
        self._plot_attention_weights(ax_attention, lstm_results)

        # 予測統計
        ax_stats = fig.add_subplot(gs[3, :])
        self._plot_prediction_statistics(ax_stats, lstm_results)

        plt.suptitle(f"{symbol} LSTM分析ダッシュボード", fontsize=16, fontweight="bold")
        plt.tight_layout()

        filename = f"lstm_dashboard_{symbol}.png"
        return self.save_figure(fig, filename, dpi=300)

    def _plot_performance_metrics(self, ax, lstm_results: Dict) -> None:
        """
        パフォーマンス指標プロット

        Args:
            ax: 描画軸
            lstm_results: LSTM結果
        """
        if "performance_metrics" in lstm_results:
            metrics = lstm_results["performance_metrics"]

            metric_names = []
            metric_values = []

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_names.append(key)
                    metric_values.append(value)

            if metric_names:
                colors = self.palette.get_categorical_palette(len(metric_names))
                bars = ax.bar(metric_names, metric_values, color=colors)

                # 値をバーの上に表示
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

                self.apply_common_styling(ax, title="パフォーマンス指標", ylabel="値")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_prediction_statistics(self, ax, lstm_results: Dict) -> None:
        """
        予測統計プロット

        Args:
            ax: 描画軸
            lstm_results: LSTM結果
        """
        if "predictions" in lstm_results:
            predictions = np.array(lstm_results["predictions"])

            # 予測値の分布をヒストグラム表示
            ax.hist(
                predictions,
                bins=30,
                color=self.palette.get_color("lstm"),
                alpha=0.7,
                edgecolor="black",
            )

            # 統計値をテキスト表示
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            ax.axvline(
                mean_pred, color="red", linestyle="--", label=f"平均: {mean_pred:.2f}"
            )
            ax.axvline(
                mean_pred + std_pred,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label=f"+1σ: {mean_pred + std_pred:.2f}",
            )
            ax.axvline(
                mean_pred - std_pred,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label=f"-1σ: {mean_pred - std_pred:.2f}",
            )

            self.apply_common_styling(
                ax, title="予測値分布", xlabel="予測価格", ylabel="頻度"
            )

    def export_lstm_report(
        self, data: pd.DataFrame, lstm_results: Dict, symbol: str = "STOCK"
    ) -> Dict[str, str]:
        """
        LSTM分析レポート出力

        Args:
            data: データ
            lstm_results: LSTM結果
            symbol: 銘柄シンボル

        Returns:
            出力ファイルパス辞書
        """
        exported_files = {}

        try:
            # メインチャート
            main_chart = self.render(data, lstm_results, symbol=symbol)
            if main_chart:
                exported_files["main_chart"] = main_chart

            # ダッシュボード
            dashboard = self.create_lstm_analysis_dashboard(data, lstm_results, symbol)
            if dashboard:
                exported_files["dashboard"] = dashboard

            logger.info(f"LSTM分析レポート出力完了 - {len(exported_files)}ファイル")
            return exported_files

        except Exception as e:
            logger.error(f"LSTM分析レポート出力エラー: {e}")
            return {}
