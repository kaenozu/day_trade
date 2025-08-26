#!/usr/bin/env python3
"""
機械学習結果可視化システム - マルチタイムフレームチャート
Issue #315: 高度テクニカル指標・ML機能拡張

複数時間軸分析結果の統合可視化機能
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseMLVisualizer, MATPLOTLIB_AVAILABLE, logger

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt


class MultiframeChartGenerator(BaseMLVisualizer):
    """マルチタイムフレーム分析チャート生成クラス"""

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
        if not self.check_matplotlib_available():
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
                    else (
                        self.colors["bearish"]
                        if "SELL" in rec_action
                        else self.colors["neutral"]
                    )
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
                self.close_figure(fig)
            return None