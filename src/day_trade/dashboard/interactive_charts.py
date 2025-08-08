#!/usr/bin/env python3
"""
インタラクティブチャート管理システム
Issue #319: 分析ダッシュボード強化

Plotlyを使用したリアルタイム・インタラクティブ可視化
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class InteractiveChartManager:
    """インタラクティブチャート管理システム"""

    def __init__(self):
        """初期化"""
        self.default_config = {
            "theme": "plotly_white",
            "color_palette": [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ],
            "font_family": "Arial, sans-serif",
            "font_size": 12,
        }

        self.chart_cache = {}
        logger.info("インタラクティブチャート管理システム初期化完了")

    def create_price_chart(
        self,
        data: Dict[str, pd.DataFrame],
        symbols: List[str],
        chart_type: str = "candlestick",
        show_volume: bool = True,
        indicators: List[str] = None,
    ) -> Dict[str, Any]:
        """
        価格チャート作成

        Args:
            data: 銘柄別価格データ
            symbols: 表示銘柄リスト
            chart_type: チャートタイプ（candlestick, line, ohlc）
            show_volume: 出来高表示フラグ
            indicators: テクニカル指標リスト

        Returns:
            Plotly チャートのJSON表現
        """
        try:
            # サブプロット設定
            rows = 2 if show_volume else 1
            subplot_titles = ["価格", "出来高"] if show_volume else ["価格"]

            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                subplot_titles=subplot_titles,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3] if show_volume else [1.0],
            )

            colors = self.default_config["color_palette"]

            for i, symbol in enumerate(symbols[: len(colors)]):
                if symbol not in data or data[symbol].empty:
                    continue

                df = data[symbol].copy()
                color = colors[i % len(colors)]

                # 価格チャート追加
                if chart_type == "candlestick":
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df["Open"],
                            high=df["High"],
                            low=df["Low"],
                            close=df["Close"],
                            name=symbol,
                            increasing_line_color=color,
                            decreasing_line_color=color,
                        ),
                        row=1,
                        col=1,
                    )
                elif chart_type == "line":
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["Close"],
                            mode="lines",
                            name=symbol,
                            line=dict(color=color, width=2),
                        ),
                        row=1,
                        col=1,
                    )
                elif chart_type == "ohlc":
                    fig.add_trace(
                        go.Ohlc(
                            x=df.index,
                            open=df["Open"],
                            high=df["High"],
                            low=df["Low"],
                            close=df["Close"],
                            name=symbol,
                            line=dict(color=color),
                        ),
                        row=1,
                        col=1,
                    )

                # 出来高チャート追加
                if show_volume and "Volume" in df.columns:
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df["Volume"],
                            name=f"{symbol} 出来高",
                            marker_color=color,
                            opacity=0.6,
                        ),
                        row=2,
                        col=1,
                    )

                # テクニカル指標追加
                if indicators:
                    self._add_technical_indicators(fig, df, symbol, indicators, color)

            # レイアウト設定
            fig.update_layout(
                title={
                    "text": f"価格チャート - {', '.join(symbols)}",
                    "x": 0.5,
                    "xanchor": "center",
                },
                template=self.default_config["theme"],
                font=dict(
                    family=self.default_config["font_family"],
                    size=self.default_config["font_size"],
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                dragmode="zoom",
                hovermode="x unified",
            )

            # X軸設定
            fig.update_xaxes(
                rangeslider_visible=False, type="date", tickformat="%Y-%m-%d"
            )

            # Y軸設定
            fig.update_yaxes(title_text="価格 (円)", row=1, col=1)
            if show_volume:
                fig.update_yaxes(title_text="出来高", row=2, col=1)

            return {
                "chart": fig.to_json(),
                "config": {
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                },
                "type": "price_chart",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"価格チャート作成エラー: {e}")
            return self._create_error_chart(f"価格チャート作成エラー: {e}")

    def create_performance_chart(
        self,
        performance_data: Dict[str, List[float]],
        dates: List[datetime],
        benchmark: Optional[Dict[str, List[float]]] = None,
        cumulative: bool = True,
    ) -> Dict[str, Any]:
        """
        パフォーマンスチャート作成

        Args:
            performance_data: 戦略別パフォーマンスデータ
            dates: 日付リスト
            benchmark: ベンチマークデータ
            cumulative: 累積リターン表示フラグ

        Returns:
            Plotly チャートのJSON表現
        """
        try:
            fig = go.Figure()
            colors = self.default_config["color_palette"]

            # 戦略別パフォーマンス追加
            for i, (strategy, returns) in enumerate(performance_data.items()):
                color = colors[i % len(colors)]

                if cumulative:
                    # 累積リターン計算
                    cum_returns = np.cumprod(1 + np.array(returns)) - 1
                    y_data = cum_returns
                    y_title = "累積リターン"
                else:
                    y_data = returns
                    y_title = "リターン"

                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=y_data,
                        mode="lines",
                        name=strategy,
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>{strategy}</b><br>"
                        + "日付: %{x}<br>"
                        + f"{y_title}: %{{y:.2%}}<br>"
                        + "<extra></extra>",
                    )
                )

            # ベンチマーク追加
            if benchmark:
                for bench_name, bench_returns in benchmark.items():
                    if cumulative:
                        bench_cum = np.cumprod(1 + np.array(bench_returns)) - 1
                        y_data = bench_cum
                    else:
                        y_data = bench_returns

                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=y_data,
                            mode="lines",
                            name=f"{bench_name} (ベンチマーク)",
                            line=dict(color="gray", width=2, dash="dash"),
                            opacity=0.8,
                        )
                    )

            # レイアウト設定
            title_text = (
                "累積パフォーマンス比較" if cumulative else "パフォーマンス比較"
            )
            fig.update_layout(
                title={"text": title_text, "x": 0.5, "xanchor": "center"},
                template=self.default_config["theme"],
                font=dict(
                    family=self.default_config["font_family"],
                    size=self.default_config["font_size"],
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                dragmode="zoom",
                hovermode="x unified",
                xaxis=dict(title="日付", type="date", tickformat="%Y-%m-%d"),
                yaxis=dict(title=y_title, tickformat=".1%" if cumulative else ".2%"),
            )

            return {
                "chart": fig.to_json(),
                "config": {
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                },
                "type": "performance_chart",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"パフォーマンスチャート作成エラー: {e}")
            return self._create_error_chart(f"パフォーマンスチャート作成エラー: {e}")

    def create_risk_analysis_chart(
        self, risk_data: Dict[str, Dict[str, float]], chart_type: str = "scatter"
    ) -> Dict[str, Any]:
        """
        リスク分析チャート作成

        Args:
            risk_data: 戦略別リスクデータ
            chart_type: チャートタイプ（scatter, radar, heatmap）

        Returns:
            Plotly チャートのJSON表現
        """
        try:
            if chart_type == "scatter":
                return self._create_risk_scatter_chart(risk_data)
            elif chart_type == "radar":
                return self._create_risk_radar_chart(risk_data)
            elif chart_type == "heatmap":
                return self._create_risk_heatmap(risk_data)
            else:
                raise ValueError(f"サポートされていないチャートタイプ: {chart_type}")

        except Exception as e:
            logger.error(f"リスク分析チャート作成エラー: {e}")
            return self._create_error_chart(f"リスク分析チャート作成エラー: {e}")

    def create_market_overview_dashboard(
        self, market_data: Dict[str, Any], layout: str = "grid"
    ) -> Dict[str, Any]:
        """
        市場概要ダッシュボード作成

        Args:
            market_data: 市場データ
            layout: レイアウトタイプ

        Returns:
            ダッシュボードのJSON表現
        """
        try:
            # 複数のチャートを統合したダッシュボード作成
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "市場セクター別パフォーマンス",
                    "出来高推移",
                    "価格分布",
                    "相関マトリックス",
                ],
                specs=[
                    [{"type": "pie"}, {"type": "scatter"}],
                    [{"type": "histogram"}, {"type": "heatmap"}],
                ],
            )

            # セクター別パフォーマンス (パイチャート)
            if "sector_performance" in market_data:
                sector_data = market_data["sector_performance"]
                fig.add_trace(
                    go.Pie(
                        labels=list(sector_data.keys()),
                        values=list(sector_data.values()),
                        name="セクター",
                        textinfo="label+percent",
                    ),
                    row=1,
                    col=1,
                )

            # 出来高推移
            if "volume_trend" in market_data:
                volume_data = market_data["volume_trend"]
                fig.add_trace(
                    go.Scatter(
                        x=volume_data["dates"],
                        y=volume_data["volumes"],
                        mode="lines+markers",
                        name="出来高",
                    ),
                    row=1,
                    col=2,
                )

            # 価格分布 (ヒストグラム)
            if "price_distribution" in market_data:
                prices = market_data["price_distribution"]
                fig.add_trace(
                    go.Histogram(x=prices, name="価格分布", nbinsx=30), row=2, col=1
                )

            # 相関マトリックス (ヒートマップ)
            if "correlation_matrix" in market_data:
                corr_data = market_data["correlation_matrix"]
                fig.add_trace(
                    go.Heatmap(
                        z=corr_data["values"],
                        x=corr_data["symbols"],
                        y=corr_data["symbols"],
                        colorscale="RdBu",
                        name="相関",
                    ),
                    row=2,
                    col=2,
                )

            fig.update_layout(
                title={"text": "市場概要ダッシュボード", "x": 0.5, "xanchor": "center"},
                template=self.default_config["theme"],
                showlegend=False,
                height=800,
            )

            return {
                "dashboard": fig.to_json(),
                "config": {"displayModeBar": True, "displaylogo": False},
                "type": "market_overview",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"市場概要ダッシュボード作成エラー: {e}")
            return self._create_error_chart(f"市場概要ダッシュボード作成エラー: {e}")

    def _add_technical_indicators(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        symbol: str,
        indicators: List[str],
        color: str,
    ) -> None:
        """テクニカル指標追加"""
        try:
            for indicator in indicators:
                if indicator == "sma_20":
                    sma = df["Close"].rolling(window=20).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=sma,
                            mode="lines",
                            name=f"{symbol} SMA20",
                            line=dict(color=color, dash="dash", width=1),
                            opacity=0.7,
                        ),
                        row=1,
                        col=1,
                    )
                elif indicator == "sma_50":
                    sma = df["Close"].rolling(window=50).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=sma,
                            mode="lines",
                            name=f"{symbol} SMA50",
                            line=dict(color=color, dash="dot", width=1),
                            opacity=0.7,
                        ),
                        row=1,
                        col=1,
                    )
                elif indicator == "bollinger":
                    # ボリンジャーバンド
                    sma = df["Close"].rolling(window=20).mean()
                    std = df["Close"].rolling(window=20).std()
                    upper = sma + 2 * std
                    lower = sma - 2 * std

                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=upper,
                            mode="lines",
                            name=f"{symbol} BB Upper",
                            line=dict(color=color, dash="dash", width=1),
                            opacity=0.5,
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=lower,
                            mode="lines",
                            name=f"{symbol} BB Lower",
                            line=dict(color=color, dash="dash", width=1),
                            opacity=0.5,
                            fill="tonexty",
                            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.1])}",
                        ),
                        row=1,
                        col=1,
                    )

        except Exception as e:
            logger.warning(f"テクニカル指標追加エラー ({indicator}): {e}")

    def _create_risk_scatter_chart(
        self, risk_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """リスク・リターン散布図作成"""
        fig = go.Figure()

        strategies = list(risk_data.keys())

        returns = [risk_data[s].get("return", 0) for s in strategies]
        volatilities = [risk_data[s].get("volatility", 0) for s in strategies]
        sharpe_ratios = [risk_data[s].get("sharpe_ratio", 0) for s in strategies]

        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode="markers+text",
                marker=dict(
                    size=[abs(sr) * 20 + 10 for sr in sharpe_ratios],
                    color=sharpe_ratios,
                    colorscale="RdYlGn",
                    colorbar=dict(title="シャープレシオ"),
                    line=dict(width=2, color="white"),
                ),
                text=strategies,
                textposition="top center",
                name="戦略",
                hovertemplate="<b>%{text}</b><br>"
                + "リターン: %{y:.2%}<br>"
                + "ボラティリティ: %{x:.2%}<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            title={"text": "リスク・リターン分析", "x": 0.5, "xanchor": "center"},
            template=self.default_config["theme"],
            xaxis=dict(title="ボラティリティ", tickformat=".1%"),
            yaxis=dict(title="リターン", tickformat=".1%"),
            showlegend=False,
        )

        return {
            "chart": fig.to_json(),
            "config": {"displayModeBar": True, "displaylogo": False},
            "type": "risk_scatter",
            "timestamp": datetime.now().isoformat(),
        }

    def _create_risk_radar_chart(
        self, risk_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """レーダーチャート作成"""
        fig = go.Figure()

        # 共通のメトリクス
        metrics = [
            "return",
            "sharpe_ratio",
            "max_drawdown",
            "volatility",
            "sortino_ratio",
        ]
        metric_labels = [
            "リターン",
            "シャープレシオ",
            "最大ドローダウン",
            "ボラティリティ",
            "ソルティノレシオ",
        ]

        colors = self.default_config["color_palette"]

        for i, (strategy, data) in enumerate(risk_data.items()):
            values = []
            for metric in metrics:
                value = data.get(metric, 0)
                # 正規化 (0-1スケール)
                if metric == "max_drawdown":
                    value = max(0, 1 + value)  # ドローダウンは負の値なので反転
                elif metric in ["return", "sharpe_ratio", "sortino_ratio"]:
                    value = max(0, min(1, (value + 1) / 2))  # -1 to 1 -> 0 to 1
                else:
                    value = max(0, min(1, value))

                values.append(value)

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metric_labels,
                    fill="toself",
                    name=strategy,
                    line=dict(color=colors[i % len(colors)]),
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title={"text": "戦略別リスク指標比較", "x": 0.5, "xanchor": "center"},
            template=self.default_config["theme"],
            showlegend=True,
        )

        return {
            "chart": fig.to_json(),
            "config": {"displayModeBar": True, "displaylogo": False},
            "type": "risk_radar",
            "timestamp": datetime.now().isoformat(),
        }

    def _create_risk_heatmap(
        self, risk_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """リスクヒートマップ作成"""
        strategies = list(risk_data.keys())
        metrics = [
            "return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "sortino_ratio",
        ]
        metric_labels = [
            "リターン",
            "ボラティリティ",
            "シャープレシオ",
            "最大ドローダウン",
            "ソルティノレシオ",
        ]

        # データ行列作成
        matrix = []
        for strategy in strategies:
            row = []
            for metric in metrics:
                value = risk_data[strategy].get(metric, 0)
                row.append(value)
            matrix.append(row)

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=metric_labels,
                y=strategies,
                colorscale="RdYlGn",
                hovertemplate="戦略: %{y}<br>"
                + "指標: %{x}<br>"
                + "値: %{z:.3f}<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            title={
                "text": "戦略別リスク指標ヒートマップ",
                "x": 0.5,
                "xanchor": "center",
            },
            template=self.default_config["theme"],
        )

        return {
            "chart": fig.to_json(),
            "config": {"displayModeBar": True, "displaylogo": False},
            "type": "risk_heatmap",
            "timestamp": datetime.now().isoformat(),
        }

    def _create_error_chart(self, error_message: str) -> Dict[str, Any]:
        """エラーチャート作成"""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"チャート作成エラー<br>{error_message}",
            showarrow=False,
            font=dict(size=16, color="red"),
            align="center",
        )

        fig.update_layout(
            title="エラー",
            template=self.default_config["theme"],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )

        return {
            "chart": fig.to_json(),
            "config": {"displayModeBar": False},
            "type": "error",
            "timestamp": datetime.now().isoformat(),
        }

    def update_chart_config(self, config_updates: Dict[str, Any]) -> None:
        """チャート設定更新"""
        self.default_config.update(config_updates)
        logger.info(f"チャート設定更新: {config_updates}")

    def clear_cache(self) -> None:
        """キャッシュクリア"""
        self.chart_cache.clear()
        logger.info("チャートキャッシュをクリアしました")


if __name__ == "__main__":
    # テスト実行
    print("インタラクティブチャート管理システム テスト")
    print("=" * 60)

    try:
        chart_manager = InteractiveChartManager()

        # サンプルデータ作成
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        sample_data = {
            "7203.T": pd.DataFrame(
                {
                    "Open": np.random.uniform(2000, 3000, len(dates)),
                    "High": np.random.uniform(2500, 3500, len(dates)),
                    "Low": np.random.uniform(1500, 2500, len(dates)),
                    "Close": np.random.uniform(2000, 3000, len(dates)),
                    "Volume": np.random.randint(1000000, 10000000, len(dates)),
                },
                index=dates,
            )
        }

        # 価格チャートテスト
        print("\n1. 価格チャート作成テスト")
        price_chart = chart_manager.create_price_chart(
            sample_data,
            ["7203.T"],
            chart_type="candlestick",
            show_volume=True,
            indicators=["sma_20", "bollinger"],
        )
        print(f"チャート作成: {price_chart['type']} - {len(price_chart['chart'])} 文字")

        # パフォーマンスチャートテスト
        print("\n2. パフォーマンスチャート作成テスト")
        performance_data = {
            "戦略A": np.random.normal(0.001, 0.02, len(dates)).tolist(),
            "戦略B": np.random.normal(0.0005, 0.015, len(dates)).tolist(),
        }
        benchmark = {"TOPIX": np.random.normal(0.0003, 0.018, len(dates)).tolist()}

        perf_chart = chart_manager.create_performance_chart(
            performance_data, dates.to_list(), benchmark=benchmark
        )
        print(f"チャート作成: {perf_chart['type']} - {len(perf_chart['chart'])} 文字")

        # リスク分析チャートテスト
        print("\n3. リスク分析チャート作成テスト")
        risk_data = {
            "戦略A": {
                "return": 0.12,
                "volatility": 0.18,
                "sharpe_ratio": 0.67,
                "max_drawdown": -0.15,
                "sortino_ratio": 0.82,
            },
            "戦略B": {
                "return": 0.08,
                "volatility": 0.12,
                "sharpe_ratio": 0.73,
                "max_drawdown": -0.08,
                "sortino_ratio": 0.91,
            },
        }

        risk_scatter = chart_manager.create_risk_analysis_chart(
            risk_data, chart_type="scatter"
        )
        print(f"散布図作成: {risk_scatter['type']} - {len(risk_scatter['chart'])} 文字")

        risk_radar = chart_manager.create_risk_analysis_chart(
            risk_data, chart_type="radar"
        )
        print(
            f"レーダーチャート作成: {risk_radar['type']} - {len(risk_radar['chart'])} 文字"
        )

        print("\nインタラクティブチャート管理システム テスト完了！")
        print("✅ 全ての基本機能が正常に動作しています")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
