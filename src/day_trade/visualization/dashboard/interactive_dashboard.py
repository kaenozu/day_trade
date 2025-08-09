"""
インタラクティブダッシュボード

Plotlyを使用したインタラクティブな分析ダッシュボード作成
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime

from ...utils.logging_config import get_context_logger
from ..base.chart_renderer import ChartRenderer

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly未インストール")


class InteractiveDashboard(ChartRenderer):
    """
    インタラクティブダッシュボードクラス

    Plotlyを使用した動的・双方向の分析ダッシュボードを提供
    """

    def __init__(
        self, output_dir: str = "output/interactive_dashboards", theme: str = "default"
    ):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            theme: テーマ
        """
        super().__init__(output_dir, theme)
        logger.info("インタラクティブダッシュボードシステム初期化完了")

    def render(
        self, data: pd.DataFrame, analysis_results: Dict, **kwargs
    ) -> Optional[str]:
        """
        インタラクティブダッシュボード描画

        Args:
            data: 元データ
            analysis_results: 分析結果
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not PLOTLY_AVAILABLE:
            logger.error("plotly未インストール - インタラクティブダッシュボード作成不可")
            return None

        symbol = kwargs.get("symbol", "UNKNOWN")
        title = kwargs.get("title", f"{symbol} インタラクティブ分析")

        # 統合ダッシュボード作成
        fig = self.create_comprehensive_dashboard(data, analysis_results, symbol)

        # HTMLファイルとして保存
        filename = kwargs.get("filename", f"interactive_dashboard_{symbol}.html")
        filepath = self.output_dir / filename

        # カスタム設定
        config = {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ],
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"dashboard_{symbol}",
                "height": 800,
                "width": 1400,
                "scale": 2,
            },
        }

        fig.write_html(
            str(filepath),
            config=config,
            include_plotlyjs=True,
            div_id="interactive-analysis-dashboard",
        )

        logger.info(f"インタラクティブダッシュボード保存完了: {filepath}")
        return str(filepath)

    def create_comprehensive_dashboard(
        self, data: pd.DataFrame, analysis_results: Dict, symbol: str = "STOCK"
    ) -> go.Figure:
        """
        総合インタラクティブダッシュボード作成

        Args:
            data: 価格データ
            analysis_results: 分析結果
            symbol: 銘柄シンボル

        Returns:
            Plotly図オブジェクト
        """
        # サブプロット構成
        subplot_titles = [
            f"{symbol} 価格・予測",
            "テクニカル指標",
            "出来高分析",
            "ML予測信頼度",
            "リスク指標",
            "パフォーマンス統計",
        ]

        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=subplot_titles,
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        # 1. メイン価格・予測チャート
        self._add_price_prediction_chart(fig, data, analysis_results, row=1, col=1)

        # 2. テクニカル指標
        self._add_technical_indicators_chart(fig, data, analysis_results, row=1, col=2)

        # 3. 出来高分析
        self._add_volume_analysis_chart(fig, data, analysis_results, row=2, col=1)

        # 4. ML予測信頼度
        self._add_ml_confidence_chart(fig, analysis_results, row=2, col=2)

        # 5. リスク指標
        self._add_risk_indicators_chart(fig, analysis_results, row=3, col=1)

        # 6. パフォーマンス統計
        self._add_performance_statistics_chart(fig, analysis_results, row=3, col=2)

        # レイアウト設定
        fig.update_layout(
            title={
                "text": f"{symbol} 統合分析ダッシュボード",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 24},
            },
            height=1000,
            showlegend=True,
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def _add_price_prediction_chart(
        self, fig: go.Figure, data: pd.DataFrame, analysis_results: Dict, row: int, col: int
    ) -> None:
        """
        価格・予測チャート追加

        Args:
            fig: 図オブジェクト
            data: データ
            analysis_results: 分析結果
            row: 行番号
            col: 列番号
        """
        # 実際の価格
        if "Close" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name="終値",
                    line=dict(color="blue", width=2),
                    hovertemplate="<b>終値</b><br>%{y:.2f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        # LSTM予測
        if "lstm_prediction" in analysis_results:
            lstm_pred = analysis_results["lstm_prediction"]
            if "predictions" in lstm_pred:
                predictions = lstm_pred["predictions"]
                pred_index = data.index[-len(predictions):]

                fig.add_trace(
                    go.Scatter(
                        x=pred_index,
                        y=predictions,
                        mode="lines",
                        name="LSTM予測",
                        line=dict(color="red", width=2, dash="dash"),
                        hovertemplate="<b>LSTM予測</b><br>%{y:.2f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

                # 信頼区間
                if "confidence_intervals" in lstm_pred:
                    ci = lstm_pred["confidence_intervals"]
                    if "lower" in ci and "upper" in ci:
                        fig.add_trace(
                            go.Scatter(
                                x=pred_index,
                                y=ci["upper"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=row,
                            col=col,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=pred_index,
                                y=ci["lower"],
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor="rgba(255,0,0,0.2)",
                                name="LSTM信頼区間",
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=row,
                            col=col,
                        )

        # アンサンブル予測
        if "ensemble_prediction" in analysis_results:
            ensemble_pred = analysis_results["ensemble_prediction"]
            if "ensemble_prediction" in ensemble_pred:
                predictions = ensemble_pred["ensemble_prediction"]
                pred_index = data.index[-len(predictions):]

                fig.add_trace(
                    go.Scatter(
                        x=pred_index,
                        y=predictions,
                        mode="lines",
                        name="アンサンブル予測",
                        line=dict(color="green", width=3),
                        hovertemplate="<b>アンサンブル予測</b><br>%{y:.2f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

    def _add_technical_indicators_chart(
        self, fig: go.Figure, data: pd.DataFrame, analysis_results: Dict, row: int, col: int
    ) -> None:
        """
        テクニカル指標チャート追加

        Args:
            fig: 図オブジェクト
            data: データ
            analysis_results: 分析結果
            row: 行番号
            col: 列番号
        """
        # RSI
        if "technical_analysis" in analysis_results:
            tech_data = analysis_results["technical_analysis"]

            if "RSI" in tech_data and "rsi" in tech_data["RSI"]:
                rsi_values = tech_data["RSI"]["rsi"]
                rsi_index = data.index[-len(rsi_values):]

                fig.add_trace(
                    go.Scatter(
                        x=rsi_index,
                        y=rsi_values,
                        mode="lines",
                        name="RSI",
                        line=dict(color="purple", width=2),
                        yaxis="y2",
                        hovertemplate="<b>RSI</b><br>%{y:.1f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                    secondary_y=True,
                )

                # RSI過買い・過売りライン
                fig.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="過買い(70)",
                    row=row,
                    col=col,
                )
                fig.add_hline(
                    y=30,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="過売り(30)",
                    row=row,
                    col=col,
                )

            # MACD
            if "MACD" in tech_data:
                macd_data = tech_data["MACD"]

                if "macd" in macd_data and "signal" in macd_data:
                    macd_line = macd_data["macd"]
                    signal_line = macd_data["signal"]
                    macd_index = data.index[-len(macd_line):]

                    fig.add_trace(
                        go.Scatter(
                            x=macd_index,
                            y=macd_line,
                            mode="lines",
                            name="MACD",
                            line=dict(color="blue", width=2),
                            hovertemplate="<b>MACD</b><br>%{y:.4f}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=macd_index,
                            y=signal_line,
                            mode="lines",
                            name="Signal",
                            line=dict(color="orange", width=2),
                            hovertemplate="<b>Signal</b><br>%{y:.4f}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

    def _add_volume_analysis_chart(
        self, fig: go.Figure, data: pd.DataFrame, analysis_results: Dict, row: int, col: int
    ) -> None:
        """
        出来高分析チャート追加

        Args:
            fig: 図オブジェクト
            data: データ
            analysis_results: 分析結果
            row: 行番号
            col: 列番号
        """
        if "Volume" in data.columns:
            # 価格変化に基づく出来高の色分け
            volume_colors = []
            if "Close" in data.columns and len(data) > 1:
                for i in range(len(data)):
                    if i > 0:
                        price_up = data["Close"].iloc[i] > data["Close"].iloc[i - 1]
                        volume_colors.append("green" if price_up else "red")
                    else:
                        volume_colors.append("gray")
            else:
                volume_colors = ["gray"] * len(data)

            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data["Volume"],
                    name="出来高",
                    marker_color=volume_colors,
                    opacity=0.7,
                    hovertemplate="<b>出来高</b><br>%{y:,.0f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # 出来高移動平均
            if "volume_analysis" in analysis_results:
                vol_analysis = analysis_results["volume_analysis"]
                if "volume_ma" in vol_analysis:
                    volume_ma = vol_analysis["volume_ma"]
                    vol_ma_index = data.index[-len(volume_ma):]

                    fig.add_trace(
                        go.Scatter(
                            x=vol_ma_index,
                            y=volume_ma,
                            mode="lines",
                            name="出来高MA",
                            line=dict(color="blue", width=2),
                            hovertemplate="<b>出来高MA</b><br>%{y:,.0f}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

    def _add_ml_confidence_chart(
        self, fig: go.Figure, analysis_results: Dict, row: int, col: int
    ) -> None:
        """
        ML予測信頼度チャート追加

        Args:
            fig: 図オブジェクト
            analysis_results: 分析結果
            row: 行番号
            col: 列番号
        """
        confidence_data = []

        # 各モデルの信頼度を収集
        if "lstm_prediction" in analysis_results:
            lstm_conf = analysis_results["lstm_prediction"].get("confidence", 0.5)
            confidence_data.append(("LSTM", lstm_conf))

        if "garch_prediction" in analysis_results:
            garch_conf = analysis_results["garch_prediction"].get("confidence", 0.5)
            confidence_data.append(("GARCH", garch_conf))

        if "ensemble_prediction" in analysis_results:
            ensemble_conf = analysis_results["ensemble_prediction"].get("confidence", 0.5)
            confidence_data.append(("Ensemble", ensemble_conf))

        if confidence_data:
            models, confidences = zip(*confidence_data)

            # 信頼度に基づく色分け
            colors = []
            for conf in confidences:
                if conf >= 0.8:
                    colors.append("green")
                elif conf >= 0.6:
                    colors.append("orange")
                else:
                    colors.append("red")

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=confidences,
                    name="予測信頼度",
                    marker_color=colors,
                    text=[f"{conf:.1%}" for conf in confidences],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>信頼度: %{y:.1%}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # 信頼度閾値線
            fig.add_hline(
                y=0.8, line_dash="dash", line_color="green", row=row, col=col
            )
            fig.add_hline(
                y=0.6, line_dash="dash", line_color="orange", row=row, col=col
            )

    def _add_risk_indicators_chart(
        self, fig: go.Figure, analysis_results: Dict, row: int, col: int
    ) -> None:
        """
        リスク指標チャート追加

        Args:
            fig: 図オブジェクト
            analysis_results: 分析結果
            row: 行番号
            col: 列番号
        """
        risk_metrics = []

        # GARCHからのリスク指標
        if "garch_prediction" in analysis_results:
            garch_data = analysis_results["garch_prediction"]
            if "risk_metrics" in garch_data:
                risk_data = garch_data["risk_metrics"]

                if "var_95" in risk_data:
                    var_95 = risk_data["var_95"]
                    if isinstance(var_95, list) and var_95:
                        risk_metrics.append(("VaR(95%)", var_95[-1]))

                if "expected_shortfall" in risk_data:
                    es = risk_data["expected_shortfall"]
                    if isinstance(es, list) and es:
                        risk_metrics.append(("Expected Shortfall", es[-1]))

        # ボラティリティ指標
        if "volatility_analysis" in analysis_results:
            vol_data = analysis_results["volatility_analysis"]
            if "current_volatility" in vol_data:
                curr_vol = vol_data["current_volatility"]
                risk_metrics.append(("現在ボラティリティ", curr_vol))

        if risk_metrics:
            risk_names, risk_values = zip(*risk_metrics)

            fig.add_trace(
                go.Bar(
                    x=risk_names,
                    y=risk_values,
                    name="リスク指標",
                    marker_color="darkred",
                    opacity=0.8,
                    text=[f"{val:.3f}" for val in risk_values],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    def _add_performance_statistics_chart(
        self, fig: go.Figure, analysis_results: Dict, row: int, col: int
    ) -> None:
        """
        パフォーマンス統計チャート追加

        Args:
            fig: 図オブジェクト
            analysis_results: 分析結果
            row: 行番号
            col: 列番号
        """
        perf_metrics = []

        # 各モデルのパフォーマンス指標を収集
        models = ["lstm_prediction", "garch_prediction", "ensemble_prediction"]
        model_names = ["LSTM", "GARCH", "Ensemble"]

        for model_key, model_name in zip(models, model_names):
            if model_key in analysis_results:
                model_data = analysis_results[model_key]

                # 精度指標
                if "accuracy_metrics" in model_data:
                    metrics = model_data["accuracy_metrics"]
                    if "mse" in metrics:
                        perf_metrics.append((f"{model_name}_MSE", metrics["mse"]))
                    if "mae" in metrics:
                        perf_metrics.append((f"{model_name}_MAE", metrics["mae"]))
                    if "r2_score" in metrics:
                        perf_metrics.append((f"{model_name}_R²", metrics["r2_score"]))

        if perf_metrics:
            metric_names, metric_values = zip(*perf_metrics)

            # パフォーマンス指標の種類に基づく色分け
            colors = []
            for name in metric_names:
                if "MSE" in name or "MAE" in name:
                    colors.append("red")  # エラー指標（低い方が良い）
                elif "R²" in name:
                    colors.append("blue")  # 決定係数（高い方が良い）
                else:
                    colors.append("gray")

            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name="パフォーマンス指標",
                    marker_color=colors,
                    opacity=0.8,
                    text=[f"{val:.4f}" for val in metric_values],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    def create_model_comparison_dashboard(
        self, data: pd.DataFrame, analysis_results: Dict, symbol: str = "STOCK"
    ) -> Optional[str]:
        """
        モデル比較ダッシュボード作成

        Args:
            data: 価格データ
            analysis_results: 分析結果
            symbol: 銘柄シンボル

        Returns:
            保存されたファイルパス
        """
        if not PLOTLY_AVAILABLE:
            logger.error("plotly未インストール - モデル比較ダッシュボード作成不可")
            return None

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "予測精度比較",
                "モデル信頼度",
                "予測vs実績",
                "リスク・リターン分析",
            ],
        )

        # モデル比較データの準備
        models_data = self._prepare_model_comparison_data(analysis_results)

        if models_data:
            # 1. 予測精度比較
            self._add_accuracy_comparison(fig, models_data, row=1, col=1)

            # 2. モデル信頼度
            self._add_confidence_comparison(fig, models_data, row=1, col=2)

            # 3. 予測vs実績
            self._add_prediction_vs_actual(fig, data, models_data, row=2, col=1)

            # 4. リスク・リターン分析
            self._add_risk_return_comparison(fig, models_data, row=2, col=2)

        fig.update_layout(
            title=f"{symbol} モデル比較ダッシュボード",
            height=800,
            showlegend=True,
        )

        filename = f"model_comparison_dashboard_{symbol}.html"
        filepath = self.output_dir / filename

        fig.write_html(str(filepath), include_plotlyjs=True)

        logger.info(f"モデル比較ダッシュボード保存完了: {filepath}")
        return str(filepath)

    def _prepare_model_comparison_data(self, analysis_results: Dict) -> Dict:
        """
        モデル比較用データ準備

        Args:
            analysis_results: 分析結果

        Returns:
            比較用データ辞書
        """
        models_data = {}

        model_keys = ["lstm_prediction", "garch_prediction", "ensemble_prediction"]
        model_names = ["LSTM", "GARCH", "Ensemble"]

        for key, name in zip(model_keys, model_names):
            if key in analysis_results:
                models_data[name] = analysis_results[key]

        return models_data

    def _add_accuracy_comparison(
        self, fig: go.Figure, models_data: Dict, row: int, col: int
    ) -> None:
        """
        精度比較チャート追加

        Args:
            fig: 図オブジェクト
            models_data: モデルデータ
            row: 行番号
            col: 列番号
        """
        model_names = []
        mse_values = []
        r2_values = []

        for model_name, model_data in models_data.items():
            if "accuracy_metrics" in model_data:
                metrics = model_data["accuracy_metrics"]
                model_names.append(model_name)
                mse_values.append(metrics.get("mse", 0))
                r2_values.append(metrics.get("r2_score", 0))

        if model_names:
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=mse_values,
                    name="MSE",
                    yaxis="y",
                    marker_color="red",
                    opacity=0.7,
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=model_names,
                    y=r2_values,
                    mode="markers+lines",
                    name="R²スコア",
                    yaxis="y2",
                    marker=dict(size=10, color="blue"),
                    line=dict(color="blue"),
                ),
                row=row,
                col=col,
            )

    def _add_confidence_comparison(
        self, fig: go.Figure, models_data: Dict, row: int, col: int
    ) -> None:
        """
        信頼度比較チャート追加

        Args:
            fig: 図オブジェクト
            models_data: モデルデータ
            row: 行番号
            col: 列番号
        """
        model_names = []
        confidence_values = []

        for model_name, model_data in models_data.items():
            confidence = model_data.get("confidence", 0.5)
            model_names.append(model_name)
            confidence_values.append(confidence)

        if model_names:
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=confidence_values,
                    name="予測信頼度",
                    marker_color="green",
                    text=[f"{conf:.1%}" for conf in confidence_values],
                    textposition="auto",
                ),
                row=row,
                col=col,
            )

    def _add_prediction_vs_actual(
        self, fig: go.Figure, data: pd.DataFrame, models_data: Dict, row: int, col: int
    ) -> None:
        """
        予測vs実績チャート追加

        Args:
            fig: 図オブジェクト
            data: 価格データ
            models_data: モデルデータ
            row: 行番号
            col: 列番号
        """
        if "Close" in data.columns:
            actual_prices = data["Close"].values

            for model_name, model_data in models_data.items():
                if "predictions" in model_data:
                    predictions = model_data["predictions"]

                    if len(predictions) <= len(actual_prices):
                        actual_subset = actual_prices[-len(predictions):]

                        fig.add_trace(
                            go.Scatter(
                                x=actual_subset,
                                y=predictions,
                                mode="markers",
                                name=f"{model_name} 予測",
                                opacity=0.7,
                                hovertemplate=f"<b>{model_name}</b><br>実績: %{{x:.2f}}<br>予測: %{{y:.2f}}<extra></extra>",
                            ),
                            row=row,
                            col=col,
                        )

            # 理想的な予測線（y=x）
            if actual_prices.size > 0:
                min_val = actual_prices.min()
                max_val = actual_prices.max()
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="理想予測線",
                        line=dict(color="black", dash="dash"),
                    ),
                    row=row,
                    col=col,
                )

    def _add_risk_return_comparison(
        self, fig: go.Figure, models_data: Dict, row: int, col: int
    ) -> None:
        """
        リスク・リターン比較チャート追加

        Args:
            fig: 図オブジェクト
            models_data: モデルデータ
            row: 行番号
            col: 列番号
        """
        model_names = []
        expected_returns = []
        risk_values = []

        for model_name, model_data in models_data.items():
            # 期待リターンとリスクの計算（簡易版）
            if "predictions" in model_data:
                predictions = model_data["predictions"]
                expected_return = np.mean(np.diff(predictions) / predictions[:-1])
                risk = np.std(np.diff(predictions) / predictions[:-1])

                model_names.append(model_name)
                expected_returns.append(expected_return)
                risk_values.append(risk)

        if model_names:
            fig.add_trace(
                go.Scatter(
                    x=risk_values,
                    y=expected_returns,
                    mode="markers+text",
                    text=model_names,
                    textposition="top center",
                    marker=dict(size=15, opacity=0.8),
                    name="モデル位置",
                    hovertemplate="<b>%{text}</b><br>リスク: %{x:.4f}<br>期待リターン: %{y:.4f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # リスク・リターン基準線
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", row=row, col=col)

    def export_interactive_report(
        self, data: pd.DataFrame, analysis_results: Dict, symbol: str = "STOCK"
    ) -> Dict[str, str]:
        """
        インタラクティブレポート出力

        Args:
            data: データ
            analysis_results: 分析結果
            symbol: 銘柄シンボル

        Returns:
            出力ファイルパス辞書
        """
        exported_files = {}

        try:
            # メインダッシュボード
            main_dashboard = self.render(data, analysis_results, symbol=symbol)
            if main_dashboard:
                exported_files["main_dashboard"] = main_dashboard

            # モデル比較ダッシュボード
            comparison_dashboard = self.create_model_comparison_dashboard(
                data, analysis_results, symbol
            )
            if comparison_dashboard:
                exported_files["comparison_dashboard"] = comparison_dashboard

            logger.info(
                f"インタラクティブレポート出力完了 - {len(exported_files)}ファイル"
            )
            return exported_files

        except Exception as e:
            logger.error(f"インタラクティブレポート出力エラー: {e}")
            return {}
