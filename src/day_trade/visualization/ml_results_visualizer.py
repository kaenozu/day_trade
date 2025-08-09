#!/usr/bin/env python3
"""
機械学習結果可視化システム
Issue #315: 高度テクニカル指標・ML機能拡張

LSTM・GARCH・マルチタイムフレーム分析結果の包括的可視化
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

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


class MLResultsVisualizer:
    """
    機械学習結果可視化システム

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

        logger.info("機械学習結果可視化システム初期化完了")

    def create_lstm_prediction_chart(
        self,
        data: pd.DataFrame,
        lstm_results: Dict,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        LSTM予測結果チャート作成

        Args:
            data: 価格データ
            lstm_results: LSTM予測結果
            symbol: 銘柄コード
            save_path: 保存パス

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlibが利用できません")
            return None

        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))

            # 実際の価格データ
            actual_prices = data["Close"].iloc[-100:]  # 最新100日分
            dates = actual_prices.index

            # 1. 価格チャートと予測
            ax1 = axes[0]
            ax1.plot(
                dates,
                actual_prices,
                color=self.colors["price"],
                linewidth=2,
                label="実際価格",
            )

            if "predicted_prices" in lstm_results:
                pred_prices = lstm_results["predicted_prices"]
                pred_dates = pd.to_datetime(lstm_results.get("prediction_dates", []))

                if len(pred_dates) == len(pred_prices):
                    ax1.plot(
                        pred_dates,
                        pred_prices,
                        color=self.colors["prediction"],
                        linewidth=2,
                        linestyle="--",
                        marker="o",
                        markersize=4,
                        label="LSTM予測",
                    )

                    # 予測範囲のシェーディング
                    confidence = lstm_results.get("confidence_score", 70)
                    if confidence > 0:
                        error_margin = (
                            np.array(pred_prices) * (1 - confidence / 100) * 0.5
                        )
                        ax1.fill_between(
                            pred_dates,
                            np.array(pred_prices) - error_margin,
                            np.array(pred_prices) + error_margin,
                            alpha=0.2,
                            color=self.colors["prediction"],
                            label=f"予測信頼区間({confidence:.0f}%)",
                        )

            ax1.set_title(f"{symbol} - LSTM価格予測", fontsize=14, fontweight="bold")
            ax1.set_ylabel("価格", fontsize=12)
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            # 2. 予測リターン
            ax2 = axes[1]
            if "predicted_returns" in lstm_results:
                pred_returns = lstm_results["predicted_returns"]
                pred_dates = pd.to_datetime(lstm_results.get("prediction_dates", []))

                if len(pred_dates) == len(pred_returns):
                    # バーチャート
                    colors = [
                        self.colors["bullish"] if r > 0 else self.colors["bearish"]
                        for r in pred_returns
                    ]
                    bars = ax2.bar(pred_dates, pred_returns, color=colors, alpha=0.7)

                    # ゼロライン
                    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)

            ax2.set_title("LSTM予測リターン", fontsize=14, fontweight="bold")
            ax2.set_ylabel("予測リターン (%)", fontsize=12)
            ax2.grid(True, alpha=0.3)

            # 3. 信頼度とモデル情報
            ax3 = axes[2]

            # 信頼度表示
            confidence = lstm_results.get("confidence_score", 0)
            confidence_color = (
                self.colors["bullish"]
                if confidence > 70
                else self.colors["neutral"]
                if confidence > 40
                else self.colors["bearish"]
            )

            ax3.bar(["予測信頼度"], [confidence], color=confidence_color, alpha=0.7)
            ax3.set_ylim(0, 100)
            ax3.set_ylabel("信頼度 (%)", fontsize=12)
            ax3.set_title("モデル信頼度", fontsize=14, fontweight="bold")

            # 信頼度の数値表示
            ax3.text(
                0,
                confidence + 5,
                f"{confidence:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

            # 全体レイアウト調整
            plt.tight_layout()

            # ファイル保存
            if save_path is None:
                save_path = self.output_dir / f"{symbol}_lstm_prediction.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"LSTM予測チャート保存完了: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"LSTM予測チャート作成エラー: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None

    def create_volatility_forecast_chart(
        self,
        data: pd.DataFrame,
        volatility_results: Dict,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        ボラティリティ予測チャート作成

        Args:
            data: 価格データ
            volatility_results: ボラティリティ予測結果
            symbol: 銘柄コード
            save_path: 保存パス

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlibが利用できません")
            return None

        try:
            fig, axes = plt.subplots(4, 1, figsize=(15, 16))

            # 過去60日のデータ
            recent_data = data.tail(60)
            dates = recent_data.index

            # 1. 価格とVIX風指標
            ax1 = axes[0]
            ax1_twin = ax1.twinx()

            # 価格
            ax1.plot(
                dates,
                recent_data["Close"],
                color=self.colors["price"],
                linewidth=2,
                label="価格",
            )
            ax1.set_ylabel("価格", color=self.colors["price"], fontsize=12)
            ax1.tick_params(axis="y", labelcolor=self.colors["price"])

            # VIX風指標
            current_metrics = volatility_results.get("current_metrics", {})
            if "vix_like_indicator" in current_metrics:
                # VIX系列を作成（簡易版）
                vix_value = current_metrics["vix_like_indicator"]
                vix_series = [vix_value] * len(
                    dates
                )  # 実際の実装では時系列データを使用

                ax1_twin.plot(
                    dates,
                    vix_series,
                    color=self.colors["vix"],
                    linewidth=2,
                    linestyle="--",
                    label="VIX風指標",
                )
                ax1_twin.set_ylabel("VIX風指標", color=self.colors["vix"], fontsize=12)
                ax1_twin.tick_params(axis="y", labelcolor=self.colors["vix"])

            ax1.set_title(
                f"{symbol} - 価格とボラティリティ指標", fontsize=14, fontweight="bold"
            )
            ax1.grid(True, alpha=0.3)

            # 2. ボラティリティ予測
            ax2 = axes[1]

            # アンサンブル予測
            ensemble = volatility_results.get("ensemble_forecast", {})
            if ensemble:
                current_vol = current_metrics.get("realized_volatility", 0) * 100
                ensemble_vol = ensemble.get("ensemble_volatility", current_vol)

                # 現在値と予測値
                ax2.bar(
                    ["現在のボラティリティ", "アンサンブル予測"],
                    [current_vol, ensemble_vol],
                    color=[self.colors["neutral"], self.colors["volatility"]],
                    alpha=0.7,
                )

                # 個別モデル予測
                individual = ensemble.get("individual_forecasts", {})
                if individual:
                    model_names = list(individual.keys())
                    model_values = list(individual.values())

                    x_pos = np.arange(len(model_names)) + 3
                    bars = ax2.bar(
                        model_names,
                        model_values,
                        color=[
                            self.colors["lstm"],
                            self.colors["garch"],
                            self.colors["vix"],
                        ][: len(model_names)],
                        alpha=0.6,
                    )

            ax2.set_ylabel("ボラティリティ (%)", fontsize=12)
            ax2.set_title("ボラティリティ予測比較", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

            # 3. リスク評価
            ax3 = axes[2]

            risk_assessment = volatility_results.get("risk_assessment", {})
            if risk_assessment:
                risk_level = risk_assessment.get("risk_level", "MEDIUM")
                risk_score = risk_assessment.get("risk_score", 50)

                # リスクレベルの色分け
                risk_colors = {
                    "LOW": self.colors["bullish"],
                    "MEDIUM": self.colors["neutral"],
                    "HIGH": self.colors["bearish"],
                }

                ax3.barh(
                    ["リスクスコア"],
                    [risk_score],
                    color=risk_colors.get(risk_level, self.colors["neutral"]),
                    alpha=0.7,
                )
                ax3.set_xlim(0, 100)
                ax3.set_xlabel("リスクスコア", fontsize=12)
                ax3.set_title(
                    f"リスク評価: {risk_level}", fontsize=14, fontweight="bold"
                )

                # リスクスコアの数値表示
                ax3.text(
                    risk_score + 2,
                    0,
                    f"{risk_score}",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

                # リスク要因表示
                risk_factors = risk_assessment.get("risk_factors", [])
                if risk_factors:
                    factors_text = "\n".join(
                        [f"• {factor}" for factor in risk_factors[:3]]
                    )
                    ax3.text(
                        0.02,
                        0.95,
                        factors_text,
                        transform=ax3.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    )

            # 4. 投資への示唆
            ax4 = axes[3]
            ax4.axis("off")  # 軸を非表示

            implications = volatility_results.get("investment_implications", {})
            if implications:
                y_pos = 0.9
                for category, suggestions in implications.items():
                    if suggestions and category != "error":
                        category_jp = {
                            "portfolio_adjustments": "ポートフォリオ調整",
                            "trading_strategies": "トレーディング戦略",
                            "risk_management": "リスク管理",
                            "market_timing": "マーケットタイミング",
                        }.get(category, category)

                        ax4.text(
                            0.02,
                            y_pos,
                            f"【{category_jp}】",
                            transform=ax4.transAxes,
                            fontsize=12,
                            fontweight="bold",
                            color=self.colors["price"],
                        )
                        y_pos -= 0.08

                        for suggestion in suggestions[:2]:
                            ax4.text(
                                0.05,
                                y_pos,
                                f"• {suggestion}",
                                transform=ax4.transAxes,
                                fontsize=10,
                            )
                            y_pos -= 0.06
                        y_pos -= 0.02

            ax4.set_title("投資への示唆", fontsize=14, fontweight="bold")

            # 全体レイアウト調整
            plt.tight_layout()

            # ファイル保存
            if save_path is None:
                save_path = self.output_dir / f"{symbol}_volatility_forecast.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"ボラティリティ予測チャート保存完了: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"ボラティリティチャート作成エラー: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None

    def create_multiframe_analysis_chart(
        self,
        data: pd.DataFrame,
        multiframe_results: Dict,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        マルチタイムフレーム分析チャート作成

        Args:
            data: 価格データ
            multiframe_results: マルチタイムフレーム分析結果
            symbol: 銘柄コード
            save_path: 保存パス

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlibが利用できません")
            return None

        try:
            fig = plt.figure(figsize=(20, 12))

            # グリッドレイアウト作成
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # 1. メイン価格チャート
            ax_main = fig.add_subplot(gs[0, :])
            recent_data = data.tail(100)

            ax_main.plot(
                recent_data.index,
                recent_data["Close"],
                color=self.colors["price"],
                linewidth=2,
                label="価格",
            )

            # サポート・レジスタンスライン（日足）
            timeframes = multiframe_results.get("timeframes", {})
            if "daily" in timeframes:
                daily_data = timeframes["daily"]
                support = daily_data.get("support_level")
                resistance = daily_data.get("resistance_level")

                if support:
                    ax_main.axhline(
                        y=support,
                        color=self.colors["support"],
                        linestyle="--",
                        alpha=0.7,
                        label=f"サポート: {support:.0f}",
                    )
                if resistance:
                    ax_main.axhline(
                        y=resistance,
                        color=self.colors["resistance"],
                        linestyle="--",
                        alpha=0.7,
                        label=f"レジスタンス: {resistance:.0f}",
                    )

            ax_main.set_title(
                f"{symbol} - マルチタイムフレーム分析", fontsize=16, fontweight="bold"
            )
            ax_main.set_ylabel("価格", fontsize=12)
            ax_main.legend(loc="upper left")
            ax_main.grid(True, alpha=0.3)

            # 2. 時間軸別トレンド強度
            ax_trend = fig.add_subplot(gs[1, 0])

            trend_data = {}
            for tf_key, tf_data in timeframes.items():
                trend_strength = tf_data.get("trend_strength", 50)
                trend_direction = tf_data.get("trend_direction", "sideways")
                trend_data[tf_data.get("timeframe", tf_key)] = {
                    "strength": trend_strength,
                    "direction": trend_direction,
                }

            if trend_data:
                tf_names = list(trend_data.keys())
                strengths = [trend_data[tf]["strength"] for tf in tf_names]
                colors = []

                for tf in tf_names:
                    direction = trend_data[tf]["direction"]
                    if "uptrend" in direction:
                        colors.append(self.colors["bullish"])
                    elif "downtrend" in direction:
                        colors.append(self.colors["bearish"])
                    else:
                        colors.append(self.colors["neutral"])

                bars = ax_trend.barh(tf_names, strengths, color=colors, alpha=0.7)
                ax_trend.set_xlim(0, 100)
                ax_trend.set_xlabel("トレンド強度", fontsize=10)
                ax_trend.set_title(
                    "時間軸別トレンド強度", fontsize=12, fontweight="bold"
                )

            # 3. テクニカル指標比較
            ax_tech = fig.add_subplot(gs[1, 1])

            tech_data = []
            tech_labels = []

            for tf_key, tf_data in timeframes.items():
                tf_name = tf_data.get("timeframe", tf_key)
                indicators = tf_data.get("technical_indicators", {})

                if "rsi" in indicators:
                    tech_data.append(indicators["rsi"])
                    tech_labels.append(f"{tf_name} RSI")

            if tech_data:
                ax_tech.bar(
                    tech_labels, tech_data, color=self.colors["volatility"], alpha=0.7
                )
                ax_tech.axhline(
                    y=70,
                    color=self.colors["bearish"],
                    linestyle="--",
                    alpha=0.5,
                    label="過買われ",
                )
                ax_tech.axhline(
                    y=30,
                    color=self.colors["bullish"],
                    linestyle="--",
                    alpha=0.5,
                    label="過売られ",
                )
                ax_tech.set_ylim(0, 100)
                ax_tech.set_ylabel("RSI", fontsize=10)
                ax_tech.set_title("時間軸別RSI", fontsize=12, fontweight="bold")
                ax_tech.legend(fontsize=8)
                plt.setp(ax_tech.get_xticklabels(), rotation=45, ha="right")

            # 4. 統合分析結果
            ax_integrated = fig.add_subplot(gs[1, 2])
            ax_integrated.axis("off")

            integrated = multiframe_results.get("integrated_analysis", {})
            if integrated:
                overall_trend = integrated.get("overall_trend", "unknown")
                confidence = integrated.get("trend_confidence", 0)
                consistency = integrated.get("consistency_score", 0)

                # 統合結果表示
                results_text = f"""
統合分析結果

総合トレンド: {overall_trend}
トレンド信頼度: {confidence:.1f}%
整合性スコア: {consistency:.1f}%
                """.strip()

                ax_integrated.text(
                    0.1,
                    0.9,
                    results_text,
                    transform=ax_integrated.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
                )

                # シグナル情報
                signal = integrated.get("integrated_signal", {})
                if signal:
                    signal_text = f"""
統合シグナル

アクション: {signal.get("action", "HOLD")}
強度: {signal.get("strength", "WEAK")}
シグナルスコア: {signal.get("signal_score", 0):.1f}
                    """.strip()

                    ax_integrated.text(
                        0.1,
                        0.5,
                        signal_text,
                        transform=ax_integrated.transAxes,
                        fontsize=11,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3),
                    )

            # 5. 投資推奨
            ax_recommend = fig.add_subplot(gs[2, :])
            ax_recommend.axis("off")

            recommendation = (
                integrated.get("investment_recommendation", {})
                if "integrated_analysis" in multiframe_results
                else {}
            )
            if recommendation:
                rec_action = recommendation.get("recommendation", "HOLD")
                position_size = recommendation.get("position_size", "NEUTRAL")
                confidence = recommendation.get("confidence", 0)

                # 推奨の色分け
                rec_color = (
                    self.colors["bullish"]
                    if "BUY" in rec_action
                    else self.colors["bearish"]
                    if "SELL" in rec_action
                    else self.colors["neutral"]
                )

                recommendation_text = f"【投資推奨】 {rec_action} (ポジションサイズ: {position_size}, 信頼度: {confidence:.1f}%)"
                ax_recommend.text(
                    0.5,
                    0.9,
                    recommendation_text,
                    transform=ax_recommend.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    ha="center",
                    bbox=dict(boxstyle="round", facecolor=rec_color, alpha=0.2),
                )

                # 推奨理由
                reasons = recommendation.get("reasons", [])
                if reasons:
                    reasons_text = "推奨理由:\n" + "\n".join(
                        [f"• {reason}" for reason in reasons[:4]]
                    )
                    ax_recommend.text(
                        0.1,
                        0.6,
                        reasons_text,
                        transform=ax_recommend.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                    )

                # ストップロス・利益確定
                stop_loss = recommendation.get("stop_loss_suggestion")
                take_profit = recommendation.get("take_profit_suggestion")

                if stop_loss or take_profit:
                    price_text = ""
                    if stop_loss:
                        price_text += f"ストップロス: {stop_loss:.2f}\n"
                    if take_profit:
                        price_text += f"利益確定: {take_profit:.2f}\n"

                    ax_recommend.text(
                        0.7,
                        0.6,
                        price_text,
                        transform=ax_recommend.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    )

            # ファイル保存
            if save_path is None:
                save_path = self.output_dir / f"{symbol}_multiframe_analysis.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"マルチタイムフレーム分析チャート保存完了: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"マルチタイムフレーム分析チャート作成エラー: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None

    def create_comprehensive_dashboard(
        self,
        data: pd.DataFrame,
        lstm_results: Optional[Dict] = None,
        volatility_results: Optional[Dict] = None,
        multiframe_results: Optional[Dict] = None,
        technical_results: Optional[Dict] = None,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        総合ダッシュボード作成

        全ての分析結果を1つのダッシュボードにまとめて可視化

        Args:
            data: 価格データ
            lstm_results: LSTM予測結果
            volatility_results: ボラティリティ予測結果
            multiframe_results: マルチタイムフレーム分析結果
            technical_results: 高度テクニカル指標結果
            symbol: 銘柄コード
            save_path: 保存パス

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlibが利用できません")
            return None

        try:
            # 大型ダッシュボード作成
            fig = plt.figure(figsize=(24, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

            # 1. メイン価格チャート (上段全体)
            ax_price = fig.add_subplot(gs[0, :])
            recent_data = data.tail(100)

            ax_price.plot(
                recent_data.index,
                recent_data["Close"],
                color=self.colors["price"],
                linewidth=3,
                label="価格",
                zorder=3,
            )

            # LSTM予測の重ね合わせ
            if lstm_results and "predicted_prices" in lstm_results:
                pred_prices = lstm_results["predicted_prices"]
                pred_dates = pd.to_datetime(lstm_results.get("prediction_dates", []))

                if len(pred_dates) == len(pred_prices):
                    ax_price.plot(
                        pred_dates,
                        pred_prices,
                        color=self.colors["prediction"],
                        linewidth=2,
                        linestyle="--",
                        marker="o",
                        markersize=5,
                        label="LSTM予測",
                        zorder=4,
                    )

            ax_price.set_title(
                f"{symbol} - 統合分析ダッシュボード", fontsize=20, fontweight="bold"
            )
            ax_price.set_ylabel("価格", fontsize=14)
            ax_price.legend(loc="upper left", fontsize=12)
            ax_price.grid(True, alpha=0.3)

            # 2. LSTM予測信頼度 (左上)
            ax_lstm = fig.add_subplot(gs[1, 0])
            if lstm_results:
                confidence = lstm_results.get("confidence_score", 0)
                ax_lstm.pie(
                    [confidence, 100 - confidence],
                    colors=[self.colors["lstm"], "lightgray"],
                    startangle=90,
                    counterclock=False,
                    wedgeprops=dict(width=0.3),
                )
                ax_lstm.text(
                    0,
                    0,
                    f"{confidence:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                )
                ax_lstm.set_title("LSTM信頼度", fontsize=12, fontweight="bold")
            else:
                ax_lstm.text(
                    0.5,
                    0.5,
                    "LSTM\n結果なし",
                    ha="center",
                    va="center",
                    transform=ax_lstm.transAxes,
                    fontsize=12,
                )
                ax_lstm.set_title("LSTM予測", fontsize=12, fontweight="bold")

            # 3. ボラティリティ状況 (中上)
            ax_vol = fig.add_subplot(gs[1, 1])
            if volatility_results:
                current_metrics = volatility_results.get("current_metrics", {})
                realized_vol = current_metrics.get("realized_volatility", 0) * 100
                vix_like = current_metrics.get("vix_like_indicator", 20)

                ax_vol.bar(
                    ["実現ボラティリティ", "VIX風指標"],
                    [realized_vol, vix_like],
                    color=[self.colors["volatility"], self.colors["vix"]],
                    alpha=0.7,
                )
                ax_vol.set_ylabel("値 (%)", fontsize=10)
                ax_vol.set_title("ボラティリティ指標", fontsize=12, fontweight="bold")
                plt.setp(ax_vol.get_xticklabels(), rotation=45, ha="right")
            else:
                ax_vol.text(
                    0.5,
                    0.5,
                    "ボラティリティ\n結果なし",
                    ha="center",
                    va="center",
                    transform=ax_vol.transAxes,
                    fontsize=12,
                )

            # 4. マルチタイムフレーム合意度 (右上)
            ax_agreement = fig.add_subplot(gs[1, 2])
            if multiframe_results:
                integrated = multiframe_results.get("integrated_analysis", {})
                consistency = integrated.get("consistency_score", 0)

                # 円グラフで合意度表示
                ax_agreement.pie(
                    [consistency, 100 - consistency],
                    colors=[self.colors["bullish"], "lightcoral"],
                    startangle=90,
                    counterclock=False,
                    wedgeprops=dict(width=0.3),
                )
                ax_agreement.text(
                    0,
                    0,
                    f"{consistency:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                )
                ax_agreement.set_title("時間軸合意度", fontsize=12, fontweight="bold")
            else:
                ax_agreement.text(
                    0.5,
                    0.5,
                    "マルチTF\n結果なし",
                    ha="center",
                    va="center",
                    transform=ax_agreement.transAxes,
                    fontsize=12,
                )

            # 5. 統合シグナル (右上隅)
            ax_signal = fig.add_subplot(gs[1, 3])
            ax_signal.axis("off")

            # 各分析からのシグナルを統合
            signals = []
            if lstm_results and "predicted_returns" in lstm_results:
                avg_return = np.mean(lstm_results["predicted_returns"])
                lstm_signal = (
                    "BUY" if avg_return > 1 else "SELL" if avg_return < -1 else "HOLD"
                )
                signals.append(("LSTM", lstm_signal))

            if multiframe_results:
                integrated = multiframe_results.get("integrated_analysis", {})
                signal_info = integrated.get("integrated_signal", {})
                mf_signal = signal_info.get("action", "HOLD")
                signals.append(("マルチTF", mf_signal))

            if volatility_results:
                risk_assessment = volatility_results.get("risk_assessment", {})
                risk_level = risk_assessment.get("risk_level", "MEDIUM")
                vol_signal = "CAUTION" if risk_level == "HIGH" else "NORMAL"
                signals.append(("ボラティリティ", vol_signal))

            # シグナル表示
            y_pos = 0.9
            for signal_source, signal_value in signals:
                signal_color = (
                    self.colors["bullish"]
                    if signal_value == "BUY"
                    else self.colors["bearish"]
                    if signal_value == "SELL"
                    else self.colors["neutral"]
                )

                ax_signal.text(
                    0.1,
                    y_pos,
                    f"{signal_source}: {signal_value}",
                    transform=ax_signal.transAxes,
                    fontsize=11,
                    color=signal_color,
                    fontweight="bold",
                )
                y_pos -= 0.15

            ax_signal.set_title("統合シグナル", fontsize=12, fontweight="bold")

            # 6. テクニカル指標ヒートマップ (左下)
            ax_heatmap = fig.add_subplot(gs[2, :2])

            # 複数時間軸のテクニカル指標を表示
            if multiframe_results:
                timeframes = multiframe_results.get("timeframes", {})
                tf_names = []
                indicator_names = ["RSI", "MACD", "BB位置"]
                heatmap_data = []

                for tf_key, tf_data in timeframes.items():
                    tf_name = tf_data.get("timeframe", tf_key)
                    tf_names.append(tf_name)
                    indicators = tf_data.get("technical_indicators", {})

                    row_data = []
                    # RSI (0-100を-1 to 1に正規化)
                    rsi = indicators.get("rsi", 50)
                    row_data.append((rsi - 50) / 50)

                    # MACD (正規化）
                    macd = indicators.get("macd", 0)
                    row_data.append(np.tanh(macd))  # -1 to 1に圧縮

                    # BB位置 (0-1を-1 to 1に変換)
                    bb_pos = indicators.get("bb_position", 0.5)
                    row_data.append(bb_pos * 2 - 1)

                    heatmap_data.append(row_data)

                if heatmap_data:
                    im = ax_heatmap.imshow(
                        heatmap_data, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1
                    )
                    ax_heatmap.set_xticks(range(len(indicator_names)))
                    ax_heatmap.set_xticklabels(indicator_names)
                    ax_heatmap.set_yticks(range(len(tf_names)))
                    ax_heatmap.set_yticklabels(tf_names)

                    # 数値表示
                    for i in range(len(tf_names)):
                        for j in range(len(indicator_names)):
                            text = ax_heatmap.text(
                                j,
                                i,
                                f"{heatmap_data[i][j]:.2f}",
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=9,
                            )

                    plt.colorbar(im, ax=ax_heatmap, shrink=0.8)

            ax_heatmap.set_title(
                "時間軸別テクニカル指標", fontsize=12, fontweight="bold"
            )

            # 7. リスク評価サマリー (右下)
            ax_risk = fig.add_subplot(gs[2, 2:])
            ax_risk.axis("off")

            risk_summary = "【リスク評価サマリー】\n\n"

            # ボラティリティリスク
            if volatility_results:
                risk_assessment = volatility_results.get("risk_assessment", {})
                risk_level = risk_assessment.get("risk_level", "UNKNOWN")
                risk_score = risk_assessment.get("risk_score", 0)
                risk_summary += f"ボラティリティリスク: {risk_level} ({risk_score}点)\n"

                risk_factors = risk_assessment.get("risk_factors", [])
                if risk_factors:
                    risk_summary += "主なリスク要因:\n"
                    for factor in risk_factors[:3]:
                        risk_summary += f"• {factor}\n"

            # マルチタイムフレームリスク
            if multiframe_results:
                integrated = multiframe_results.get("integrated_analysis", {})
                risk_info = integrated.get("risk_assessment", {})
                if risk_info:
                    mf_risk_level = risk_info.get("risk_level", "UNKNOWN")
                    risk_summary += f"\nマルチTFリスク: {mf_risk_level}\n"

            ax_risk.text(
                0.05,
                0.95,
                risk_summary,
                transform=ax_risk.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.2),
            )

            # 8. 総合推奨 (下段中央)
            ax_recommendation = fig.add_subplot(gs[3, :])
            ax_recommendation.axis("off")

            # 最終推奨の統合
            final_recommendation = "【統合投資推奨】\n\n"

            if multiframe_results:
                integrated = multiframe_results.get("integrated_analysis", {})
                recommendation = integrated.get("investment_recommendation", {})
                if recommendation:
                    action = recommendation.get("recommendation", "HOLD")
                    position = recommendation.get("position_size", "NEUTRAL")
                    confidence = recommendation.get("confidence", 0)

                    final_recommendation += f"推奨アクション: {action}\n"
                    final_recommendation += f"ポジションサイズ: {position}\n"
                    final_recommendation += f"信頼度: {confidence:.1f}%\n\n"

                    reasons = recommendation.get("reasons", [])
                    if reasons:
                        final_recommendation += "推奨根拠:\n"
                        for reason in reasons:
                            final_recommendation += f"• {reason}\n"

            # 推奨の色分け
            if "BUY" in final_recommendation:
                rec_color = "lightgreen"
            elif "SELL" in final_recommendation:
                rec_color = "lightcoral"
            else:
                rec_color = "lightgray"

            ax_recommendation.text(
                0.5,
                0.5,
                final_recommendation,
                transform=ax_recommendation.transAxes,
                fontsize=12,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor=rec_color, alpha=0.3),
            )

            # タイムスタンプ追加
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(
                0.99,
                0.01,
                f"生成日時: {timestamp}",
                ha="right",
                va="bottom",
                fontsize=8,
                alpha=0.7,
            )

            # ファイル保存
            if save_path is None:
                save_path = self.output_dir / f"{symbol}_comprehensive_dashboard.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"総合ダッシュボード保存完了: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"総合ダッシュボード作成エラー: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None

    def create_interactive_plotly_dashboard(
        self,
        data: pd.DataFrame,
        results_dict: Dict,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Plotlyによるインタラクティブダッシュボード作成

        Args:
            data: 価格データ
            results_dict: 全分析結果の辞書
            symbol: 銘柄コード
            save_path: 保存パス

        Returns:
            保存されたファイルパス（HTML）
        """
        if not PLOTLY_AVAILABLE:
            logger.error("plotlyが利用できません")
            return None

        try:
            # サブプロット作成
            fig = make_subplots(
                rows=4,
                cols=2,
                subplot_titles=[
                    "価格チャートとLSTM予測",
                    "ボラティリティ予測",
                    "マルチタイムフレーム分析",
                    "テクニカル指標",
                    "リスク評価",
                    "統合シグナル",
                    "予測精度",
                    "投資推奨",
                ],
                specs=[
                    [{"secondary_y": False}, {"secondary_y": True}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
                vertical_spacing=0.08,
            )

            # 1. 価格チャートとLSTM予測
            recent_data = data.tail(100)
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data["Close"],
                    mode="lines",
                    name="価格",
                    line=dict(color="blue", width=2),
                ),
                row=1,
                col=1,
            )

            lstm_results = results_dict.get("lstm", {})
            if lstm_results and "predicted_prices" in lstm_results:
                pred_dates = pd.to_datetime(lstm_results.get("prediction_dates", []))
                pred_prices = lstm_results["predicted_prices"]

                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=pred_prices,
                        mode="lines+markers",
                        name="LSTM予測",
                        line=dict(color="orange", width=2, dash="dash"),
                    ),
                    row=1,
                    col=1,
                )

            # 2. ボラティリティ予測
            vol_results = results_dict.get("volatility", {})
            if vol_results:
                ensemble = vol_results.get("ensemble_forecast", {})
                individual = ensemble.get("individual_forecasts", {})

                if individual:
                    models = list(individual.keys())
                    values = list(individual.values())

                    fig.add_trace(
                        go.Bar(
                            x=models,
                            y=values,
                            name="ボラティリティ予測",
                            marker_color="purple",
                        ),
                        row=1,
                        col=2,
                    )

            # 3. マルチタイムフレーム分析
            mf_results = results_dict.get("multiframe", {})
            if mf_results:
                timeframes = mf_results.get("timeframes", {})
                tf_names = []
                strengths = []

                for tf_key, tf_data in timeframes.items():
                    tf_names.append(tf_data.get("timeframe", tf_key))
                    strengths.append(tf_data.get("trend_strength", 50))

                if tf_names:
                    fig.add_trace(
                        go.Bar(
                            x=tf_names,
                            y=strengths,
                            name="トレンド強度",
                            marker_color="green",
                        ),
                        row=2,
                        col=1,
                    )

            # 4. テクニカル指標（レーダーチャート風）
            if mf_results:
                timeframes = mf_results.get("timeframes", {})
                daily_data = timeframes.get("daily", {})
                indicators = daily_data.get("technical_indicators", {})

                if indicators:
                    indicator_names = list(indicators.keys())[:5]  # 上位5個
                    indicator_values = [indicators[name] for name in indicator_names]

                    fig.add_trace(
                        go.Bar(
                            x=indicator_names,
                            y=indicator_values,
                            name="テクニカル指標",
                            marker_color="red",
                        ),
                        row=2,
                        col=2,
                    )

            # 5. リスク評価
            risk_scores = []
            risk_labels = []

            if vol_results:
                risk_assessment = vol_results.get("risk_assessment", {})
                risk_score = risk_assessment.get("risk_score", 0)
                risk_scores.append(risk_score)
                risk_labels.append("ボラティリティリスク")

            if mf_results:
                integrated = mf_results.get("integrated_analysis", {})
                risk_info = integrated.get("risk_assessment", {})
                if risk_info:
                    risk_score = risk_info.get("risk_score", 0)
                    risk_scores.append(risk_score)
                    risk_labels.append("マルチTFリスク")

            if risk_scores:
                fig.add_trace(
                    go.Bar(
                        x=risk_labels,
                        y=risk_scores,
                        name="リスクスコア",
                        marker_color="orange",
                    ),
                    row=3,
                    col=1,
                )

            # 6. 統合シグナル（パイチャート）
            signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}

            if lstm_results and "predicted_returns" in lstm_results:
                avg_return = np.mean(lstm_results["predicted_returns"])
                if avg_return > 1:
                    signal_counts["BUY"] += 1
                elif avg_return < -1:
                    signal_counts["SELL"] += 1
                else:
                    signal_counts["HOLD"] += 1

            if mf_results:
                integrated = mf_results.get("integrated_analysis", {})
                signal_info = integrated.get("integrated_signal", {})
                action = signal_info.get("action", "HOLD")
                signal_counts[action] += 1

            fig.add_trace(
                go.Pie(
                    labels=list(signal_counts.keys()),
                    values=list(signal_counts.values()),
                    name="統合シグナル",
                ),
                row=3,
                col=2,
            )

            # 7. 予測精度（LSTM）
            if lstm_results:
                confidence = lstm_results.get("confidence_score", 0)

                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "LSTM信頼度 (%)"},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 50], "color": "lightgray"},
                                {"range": [50, 80], "color": "gray"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 90,
                            },
                        },
                    ),
                    row=4,
                    col=1,
                )

            # レイアウト設定
            fig.update_layout(
                title=f"{symbol} - 機械学習統合分析ダッシュボード",
                height=1200,
                showlegend=True,
            )

            # ファイル保存
            if save_path is None:
                save_path = self.output_dir / f"{symbol}_interactive_dashboard.html"

            fig.write_html(str(save_path))

            logger.info(f"インタラクティブダッシュボード保存完了: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"インタラクティブダッシュボード作成エラー: {e}")
            return None

    def generate_analysis_report(
        self, symbol: str, results_dict: Dict, save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        分析レポート生成（テキスト形式）

        Args:
            symbol: 銘柄コード
            results_dict: 全分析結果の辞書
            save_path: 保存パス

        Returns:
            保存されたファイルパス
        """
        try:
            if save_path is None:
                save_path = self.output_dir / f"{symbol}_analysis_report.txt"

            with open(save_path, "w", encoding="utf-8") as f:
                f.write("機械学習統合分析レポート\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"銘柄: {symbol}\n")
                f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # LSTM分析
                lstm_results = results_dict.get("lstm", {})
                if lstm_results:
                    f.write("【LSTM時系列予測】\n")
                    f.write(
                        f"予測信頼度: {lstm_results.get('confidence_score', 0):.1f}%\n"
                    )

                    if "predicted_prices" in lstm_results:
                        pred_prices = lstm_results["predicted_prices"]
                        f.write(f"価格予測: {pred_prices}\n")

                    if "predicted_returns" in lstm_results:
                        pred_returns = lstm_results["predicted_returns"]
                        avg_return = np.mean(pred_returns)
                        f.write(f"平均予測リターン: {avg_return:.2f}%\n")

                    f.write("\n")

                # ボラティリティ分析
                vol_results = results_dict.get("volatility", {})
                if vol_results:
                    f.write("【ボラティリティ予測】\n")

                    current_metrics = vol_results.get("current_metrics", {})
                    if current_metrics:
                        f.write(
                            f"現在の実現ボラティリティ: {current_metrics.get('realized_volatility', 0) * 100:.1f}%\n"
                        )
                        f.write(
                            f"VIX風指標: {current_metrics.get('vix_like_indicator', 0):.1f}\n"
                        )

                    ensemble = vol_results.get("ensemble_forecast", {})
                    if ensemble:
                        f.write(
                            f"アンサンブル予測ボラティリティ: {ensemble.get('ensemble_volatility', 0):.1f}%\n"
                        )
                        f.write(
                            f"予測信頼度: {ensemble.get('ensemble_confidence', 0):.1f}\n"
                        )

                    risk_assessment = vol_results.get("risk_assessment", {})
                    if risk_assessment:
                        f.write(
                            f"リスクレベル: {risk_assessment.get('risk_level', 'UNKNOWN')}\n"
                        )

                        risk_factors = risk_assessment.get("risk_factors", [])
                        if risk_factors:
                            f.write("リスク要因:\n")
                            for factor in risk_factors:
                                f.write(f"  - {factor}\n")

                    f.write("\n")

                # マルチタイムフレーム分析
                mf_results = results_dict.get("multiframe", {})
                if mf_results:
                    f.write("【マルチタイムフレーム分析】\n")

                    integrated = mf_results.get("integrated_analysis", {})
                    if integrated:
                        f.write(
                            f"総合トレンド: {integrated.get('overall_trend', 'unknown')}\n"
                        )
                        f.write(
                            f"トレンド信頼度: {integrated.get('trend_confidence', 0):.1f}%\n"
                        )
                        f.write(
                            f"時間軸整合性: {integrated.get('consistency_score', 0):.1f}%\n"
                        )

                        signal = integrated.get("integrated_signal", {})
                        if signal:
                            f.write(f"統合シグナル: {signal.get('action', 'HOLD')}\n")
                            f.write(f"シグナル強度: {signal.get('strength', 'WEAK')}\n")

                        recommendation = integrated.get("investment_recommendation", {})
                        if recommendation:
                            f.write(
                                f"投資推奨: {recommendation.get('recommendation', 'HOLD')}\n"
                            )
                            f.write(
                                f"ポジションサイズ: {recommendation.get('position_size', 'NEUTRAL')}\n"
                            )

                            reasons = recommendation.get("reasons", [])
                            if reasons:
                                f.write("推奨理由:\n")
                                for reason in reasons:
                                    f.write(f"  - {reason}\n")

                    timeframes = mf_results.get("timeframes", {})
                    if timeframes:
                        f.write("\n時間軸別分析:\n")
                        for tf_key, tf_data in timeframes.items():
                            f.write(f"  {tf_data.get('timeframe', tf_key)}:\n")
                            f.write(
                                f"    トレンド: {tf_data.get('trend_direction', 'unknown')}\n"
                            )
                            f.write(
                                f"    強度: {tf_data.get('trend_strength', 0):.1f}\n"
                            )

                    f.write("\n")

                # 統合推奨
                f.write("【統合投資判断】\n")

                # 各分析からの信号を集計
                signals = []
                confidences = []

                if lstm_results:
                    if "predicted_returns" in lstm_results:
                        avg_return = np.mean(lstm_results["predicted_returns"])
                        lstm_signal = (
                            "BUY"
                            if avg_return > 1
                            else "SELL"
                            if avg_return < -1
                            else "HOLD"
                        )
                        signals.append(lstm_signal)
                        confidences.append(lstm_results.get("confidence_score", 0))

                if mf_results:
                    integrated = mf_results.get("integrated_analysis", {})
                    signal_info = integrated.get("integrated_signal", {})
                    mf_signal = signal_info.get("action", "HOLD")
                    signals.append(mf_signal)
                    confidences.append(integrated.get("trend_confidence", 0))

                # 統合判定
                buy_count = signals.count("BUY")
                sell_count = signals.count("SELL")
                hold_count = signals.count("HOLD")

                if buy_count > sell_count and buy_count > hold_count:
                    final_signal = "BUY"
                elif sell_count > buy_count and sell_count > hold_count:
                    final_signal = "SELL"
                else:
                    final_signal = "HOLD"

                avg_confidence = np.mean(confidences) if confidences else 0

                f.write(f"最終判定: {final_signal}\n")
                f.write(f"統合信頼度: {avg_confidence:.1f}%\n")
                f.write(
                    f"シグナル分布: BUY({buy_count}) SELL({sell_count}) HOLD({hold_count})\n"
                )

                f.write("\n" + "=" * 50 + "\n")
                f.write("レポート終了\n")

            logger.info(f"分析レポート保存完了: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"分析レポート生成エラー: {e}")
            return None


if __name__ == "__main__":
    # テスト実行
    print("=== 機械学習結果可視化システム テスト ===")

    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib未インストールのためテストを実行できません")
        exit(1)

    # サンプルデータ生成
    dates = pd.date_range(start="2023-06-01", end="2024-12-31", freq="D")
    np.random.seed(42)

    base_price = 2800
    prices = [base_price]
    for i in range(1, len(dates)):
        change = np.random.normal(0.0008, 0.025)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))

    sample_data = pd.DataFrame(
        {
            "Open": [p * np.random.uniform(0.995, 1.005) for p in prices],
            "High": [p * np.random.uniform(1.000, 1.025) for p in prices],
            "Low": [p * np.random.uniform(0.975, 1.000) for p in prices],
            "Close": prices,
            "Volume": np.random.randint(800000, 12000000, len(dates)),
        },
        index=dates,
    )

    # サンプル結果データ作成
    sample_lstm_results = {
        "predicted_prices": [2850, 2880, 2920, 2950, 2970],
        "predicted_returns": [1.8, 1.05, 1.4, 1.0, 0.7],
        "prediction_dates": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
        ],
        "confidence_score": 78.5,
    }

    sample_volatility_results = {
        "current_metrics": {
            "realized_volatility": 0.22,
            "vix_like_indicator": 28.5,
            "volatility_regime": "medium_vol",
        },
        "ensemble_forecast": {
            "ensemble_volatility": 26.8,
            "ensemble_confidence": 0.72,
            "individual_forecasts": {
                "GARCH": 25.2,
                "Machine Learning": 28.1,
                "VIX-like": 27.1,
            },
        },
        "risk_assessment": {
            "risk_level": "MEDIUM",
            "risk_score": 45,
            "risk_factors": ["中程度のボラティリティ環境", "一部不確実性要因"],
        },
        "investment_implications": {
            "portfolio_adjustments": ["適度なリスク管理"],
            "trading_strategies": ["レンジトレーディング機会"],
            "risk_management": ["標準的なリスク管理手法"],
        },
    }

    sample_multiframe_results = {
        "timeframes": {
            "daily": {
                "timeframe": "日足",
                "trend_direction": "uptrend",
                "trend_strength": 65.2,
                "technical_indicators": {
                    "rsi": 58.3,
                    "macd": 0.15,
                    "bb_position": 0.67,
                },
                "support_level": 2750,
                "resistance_level": 2950,
            },
            "weekly": {
                "timeframe": "週足",
                "trend_direction": "uptrend",
                "trend_strength": 72.8,
                "technical_indicators": {
                    "rsi": 62.1,
                    "macd": 0.22,
                    "bb_position": 0.71,
                },
            },
        },
        "integrated_analysis": {
            "overall_trend": "uptrend",
            "trend_confidence": 68.9,
            "consistency_score": 78.4,
            "integrated_signal": {
                "action": "BUY",
                "strength": "MODERATE",
                "signal_score": 12.5,
            },
            "investment_recommendation": {
                "recommendation": "BUY",
                "position_size": "MODERATE",
                "confidence": 68.9,
                "reasons": ["複数時間軸で上昇トレンド確認", "技術指標の良好な位置"],
            },
        },
    }

    try:
        visualizer = MLResultsVisualizer()

        print(f"サンプルデータ: {len(sample_data)}日分")
        print(f"出力ディレクトリ: {visualizer.output_dir}")

        # 1. LSTM予測チャートテスト
        print("\n1. LSTM予測チャート作成テスト")
        lstm_chart = visualizer.create_lstm_prediction_chart(
            sample_data, sample_lstm_results, symbol="TEST_STOCK"
        )
        if lstm_chart:
            print(f"✅ LSTM予測チャート作成完了: {lstm_chart}")
        else:
            print("❌ LSTM予測チャート作成失敗")

        # 2. ボラティリティ予測チャートテスト
        print("\n2. ボラティリティ予測チャート作成テスト")
        vol_chart = visualizer.create_volatility_forecast_chart(
            sample_data, sample_volatility_results, symbol="TEST_STOCK"
        )
        if vol_chart:
            print(f"✅ ボラティリティ予測チャート作成完了: {vol_chart}")
        else:
            print("❌ ボラティリティ予測チャート作成失敗")

        # 3. マルチタイムフレーム分析チャートテスト
        print("\n3. マルチタイムフレーム分析チャート作成テスト")
        mf_chart = visualizer.create_multiframe_analysis_chart(
            sample_data, sample_multiframe_results, symbol="TEST_STOCK"
        )
        if mf_chart:
            print(f"✅ マルチタイムフレーム分析チャート作成完了: {mf_chart}")
        else:
            print("❌ マルチタイムフレーム分析チャート作成失敗")

        # 4. 総合ダッシュボードテスト
        print("\n4. 総合ダッシュボード作成テスト")
        dashboard = visualizer.create_comprehensive_dashboard(
            sample_data,
            lstm_results=sample_lstm_results,
            volatility_results=sample_volatility_results,
            multiframe_results=sample_multiframe_results,
            symbol="TEST_STOCK",
        )
        if dashboard:
            print(f"✅ 総合ダッシュボード作成完了: {dashboard}")
        else:
            print("❌ 総合ダッシュボード作成失敗")

        # 5. インタラクティブダッシュボードテスト
        if PLOTLY_AVAILABLE:
            print("\n5. インタラクティブダッシュボード作成テスト")
            results_dict = {
                "lstm": sample_lstm_results,
                "volatility": sample_volatility_results,
                "multiframe": sample_multiframe_results,
            }

            interactive_dashboard = visualizer.create_interactive_plotly_dashboard(
                sample_data, results_dict, symbol="TEST_STOCK"
            )
            if interactive_dashboard:
                print(
                    f"✅ インタラクティブダッシュボード作成完了: {interactive_dashboard}"
                )
            else:
                print("❌ インタラクティブダッシュボード作成失敗")
        else:
            print(
                "\n5. インタラクティブダッシュボード - スキップ（plotly未インストール）"
            )

        # 6. 分析レポートテスト
        print("\n6. 分析レポート生成テスト")
        results_dict = {
            "lstm": sample_lstm_results,
            "volatility": sample_volatility_results,
            "multiframe": sample_multiframe_results,
        }

        report = visualizer.generate_analysis_report("TEST_STOCK", results_dict)
        if report:
            print(f"✅ 分析レポート生成完了: {report}")

            # レポート内容の一部表示
            with open(report, encoding="utf-8") as f:
                lines = f.readlines()
                print("📄 レポート内容（抜粋）:")
                for line in lines[:15]:  # 最初の15行表示
                    print(f"   {line.rstrip()}")
                if len(lines) > 15:
                    print(f"   ... (残り{len(lines) - 15}行)")
        else:
            print("❌ 分析レポート生成失敗")

        print("\n✅ 機械学習結果可視化システム テスト完了！")
        print(f"📁 生成ファイル保存場所: {visualizer.output_dir}")

        # 生成されたファイル一覧
        output_files = list(visualizer.output_dir.glob("*"))
        if output_files:
            print("📊 生成されたファイル:")
            for file_path in output_files:
                print(f"   - {file_path.name}")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
