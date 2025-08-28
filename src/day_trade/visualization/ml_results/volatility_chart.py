#!/usr/bin/env python3
"""
機械学習結果可視化システム - ボラティリティチャート
Issue #315: 高度テクニカル指標・ML機能拡張

ボラティリティ予測・GARCH・VIX分析結果の専用可視化機能
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseMLVisualizer, MATPLOTLIB_AVAILABLE, logger

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt


class VolatilityChartGenerator(BaseMLVisualizer):
    """ボラティリティ予測チャート生成クラス"""

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
        if not self.check_matplotlib_available():
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
                self.close_figure(fig)
            return None