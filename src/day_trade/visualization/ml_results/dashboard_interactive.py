#!/usr/bin/env python3
"""
機械学習結果可視化システム - インタラクティブダッシュボード
Issue #315: 高度テクニカル指標・ML機能拡張

Plotlyによるインタラクティブ分析ダッシュボード生成機能
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseMLVisualizer, PLOTLY_AVAILABLE, logger

if PLOTLY_AVAILABLE:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots


class InteractiveDashboardGenerator(BaseMLVisualizer):
    """インタラクティブダッシュボード生成クラス"""

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
        if not self.check_plotly_available():
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