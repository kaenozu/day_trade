#!/usr/bin/env python3
"""
機械学習結果可視化システム - 静的ダッシュボード
Issue #315: 高度テクニカル指標・ML機能拡張

総合分析結果の統合静的ダッシュボード生成機能
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseMLVisualizer, MATPLOTLIB_AVAILABLE, logger

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt


class StaticDashboardGenerator(BaseMLVisualizer):
    """静的総合ダッシュボード生成クラス"""

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
        if not self.check_matplotlib_available():
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
                    else (
                        self.colors["bearish"]
                        if signal_value == "SELL"
                        else self.colors["neutral"]
                    )
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
                self.close_figure(fig)
            return None