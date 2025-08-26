#!/usr/bin/env python3
"""
機械学習結果可視化システム - LSTMチャート
Issue #315: 高度テクニカル指標・ML機能拡張

LSTM時系列予測結果の専用可視化機能
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseMLVisualizer, MATPLOTLIB_AVAILABLE, logger

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt


class LSTMChartGenerator(BaseMLVisualizer):
    """LSTM予測結果チャート生成クラス"""

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
        if not self.check_matplotlib_available():
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
                else (
                    self.colors["neutral"]
                    if confidence > 40
                    else self.colors["bearish"]
                )
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
                self.close_figure(fig)
            return None