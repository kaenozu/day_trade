#!/usr/bin/env python3
"""
機械学習結果可視化システム - レポート生成
Issue #315: 高度テクニカル指標・ML機能拡張

分析結果のテキストレポート生成機能
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseMLVisualizer, logger


class ReportGenerator(BaseMLVisualizer):
    """分析レポート生成クラス"""

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
                            else "SELL" if avg_return < -1 else "HOLD"
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