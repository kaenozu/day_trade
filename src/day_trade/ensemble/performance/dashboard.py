#!/usr/bin/env python3
"""
視覚的ダッシュボード
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import base64
from typing import Dict, Optional
from io import BytesIO

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from .types import AttributionResult, PerformanceDecomposition

logger = logging.getLogger(__name__)

class VisualDashboard:
    """視覚的ダッシュボード"""

    def __init__(self, style: str = "plotly", output_format: str = "html"):
        self.style = style
        self.output_format = output_format
        self.chart_cache: Dict[str, str] = {}

        # スタイル設定
        if style == "seaborn":
            sns.set_style("whitegrid")
            plt.style.use("seaborn-v0_8")

        logger.info(f"VisualDashboard initialized with {style} style")

    async def generate_charts(self,
                            attributions: AttributionResult,
                            decomposition: PerformanceDecomposition,
                            time_series_data: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """チャート生成"""
        try:
            charts = {}

            # 1. モデル貢献度チャート
            charts["model_attribution"] = await self._create_attribution_chart(attributions)

            # 2. パフォーマンス分解チャート
            charts["performance_decomposition"] = await self._create_decomposition_chart(decomposition)

            # 3. 特徴量重要度チャート
            charts["feature_importance"] = await self._create_feature_importance_chart(attributions)

            # 4. 相互作用ヒートマップ
            charts["interaction_heatmap"] = await self._create_interaction_heatmap(attributions)

            # 5. 時系列チャート（データがある場合）
            if time_series_data is not None:
                charts["time_series"] = await self._create_time_series_chart(time_series_data)

            # キャッシュ更新
            self.chart_cache.update(charts)

            return charts

        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return {}

    async def _create_attribution_chart(self, attributions: AttributionResult) -> str:
        """貢献度チャート作成"""
        try:
            if self.style == "plotly":
                return await self._create_plotly_attribution_chart(attributions)
            else:
                return await self._create_matplotlib_attribution_chart(attributions)
        except Exception as e:
            logger.error(f"Attribution chart error: {e}")
            return ""

    async def _create_plotly_attribution_chart(self, attributions: AttributionResult) -> str:
        """Plotly貢献度チャート"""
        try:
            model_names = list(attributions.model_attributions.keys())
            model_contributions = [
                np.mean(attributions.model_attributions[model])
                for model in model_names
            ]

            # 円グラフ
            fig = go.Figure(data=[go.Pie(
                labels=model_names,
                values=model_contributions,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])

            fig.update_layout(
                title="Model Attribution Analysis",
                showlegend=True,
                width=600,
                height=500
            )

            if self.output_format == "html":
                return fig.to_html(include_plotlyjs=True)
            else:
                # Base64エンコード画像
                img_bytes = fig.to_image(format="png", width=600, height=500)
                img_base64 = base64.b64encode(img_bytes).decode()
                return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Plotly attribution chart error: {e}")
            return ""

    async def _create_matplotlib_attribution_chart(self, attributions: AttributionResult) -> str:
        """Matplotlib貢献度チャート"""
        try:
            model_names = list(attributions.model_attributions.keys())
            model_contributions = [
                np.mean(attributions.model_attributions[model])
                for model in model_names
            ]

            fig, ax = plt.subplots(figsize=(10, 6))

            # 棒グラフ
            bars = ax.bar(model_names, model_contributions,
                         color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))

            ax.set_title("Model Attribution Analysis")
            ax.set_ylabel("Attribution Score")
            ax.set_xlabel("Models")

            # 値ラベル
            for bar, value in zip(bars, model_contributions):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

            plt.tight_layout()

            # Base64エンコード
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Matplotlib attribution chart error: {e}")
            return ""

    async def _create_decomposition_chart(self, decomposition: PerformanceDecomposition) -> str:
        """分解チャート作成"""
        try:
            components = dict(decomposition.component_rankings[:10])  # Top 10

            if self.style == "plotly":
                fig = go.Figure([go.Bar(
                    x=list(components.keys()),
                    y=list(components.values()),
                    text=[f"{v:.3f}" for v in components.values()],
                    textposition='auto',
                )])

                fig.update_layout(
                    title="Performance Decomposition",
                    xaxis_title="Components",
                    yaxis_title="Contribution",
                    width=800,
                    height=500
                )

                if self.output_format == "html":
                    return fig.to_html(include_plotlyjs=True)
                else:
                    img_bytes = fig.to_image(format="png", width=800, height=500)
                    img_base64 = base64.b64encode(img_bytes).decode()
                    return f"data:image/png;base64,{img_base64}"

            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(range(len(components)), list(components.values()),
                             color=plt.cm.plasma(np.linspace(0, 1, len(components))))

                ax.set_xticks(range(len(components)))
                ax.set_xticklabels(list(components.keys()), rotation=45, ha='right')
                ax.set_title("Performance Decomposition")
                ax.set_ylabel("Contribution")

                plt.tight_layout()

                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Decomposition chart error: {e}")
            return ""

    async def _create_feature_importance_chart(self, attributions: AttributionResult) -> str:
        """特徴量重要度チャート"""
        try:
            feature_names = list(attributions.feature_attributions.keys())
            feature_importance = [
                np.mean(np.abs(attributions.feature_attributions[feature]))
                for feature in feature_names
            ]

            if self.style == "plotly":
                fig = go.Figure([go.Bar(
                    y=feature_names,
                    x=feature_importance,
                    orientation='h',
                    text=[f"{v:.3f}" for v in feature_importance],
                    textposition='auto',
                )])

                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    width=700,
                    height=600
                )

                if self.output_format == "html":
                    return fig.to_html(include_plotlyjs=True)
                else:
                    img_bytes = fig.to_image(format="png", width=700, height=600)
                    img_base64 = base64.b64encode(img_bytes).decode()
                    return f"data:image/png;base64,{img_base64}"

            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                bars = ax.barh(range(len(feature_names)), feature_importance,
                              color=plt.cm.viridis(np.linspace(0, 1, len(feature_names))))

                ax.set_yticks(range(len(feature_names)))
                ax.set_yticklabels(feature_names)
                ax.set_title("Feature Importance")
                ax.set_xlabel("Importance Score")

                plt.tight_layout()

                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Feature importance chart error: {e}")
            return ""

    async def _create_interaction_heatmap(self, attributions: AttributionResult) -> str:
        """相互作用ヒートマップ"""
        try:
            interactions = attributions.interaction_effects
            if not interactions:
                return ""

            # 相互作用マトリックス作成
            interaction_keys = list(interactions.keys())
            interaction_values = [np.mean(interactions[key]) for key in interaction_keys]

            if self.style == "plotly":
                # 簡易ヒートマップ（1D to 2D変換）
                n = int(np.sqrt(len(interaction_keys))) + 1
                matrix = np.zeros((n, n))

                for i, value in enumerate(interaction_values[:n*n]):
                    row, col = divmod(i, n)
                    matrix[row, col] = value

                fig = go.Figure(data=go.Heatmap(
                    z=matrix,
                    colorscale='Viridis',
                    showscale=True
                ))

                fig.update_layout(
                    title="Interaction Effects Heatmap",
                    width=600,
                    height=500
                )

                if self.output_format == "html":
                    return fig.to_html(include_plotlyjs=True)
                else:
                    img_bytes = fig.to_image(format="png", width=600, height=500)
                    img_base64 = base64.b64encode(img_bytes).decode()
                    return f"data:image/png;base64,{img_base64}"

            else:
                # Matplotlib版
                fig, ax = plt.subplots(figsize=(8, 6))

                # 簡易可視化：棒グラフ
                bars = ax.bar(range(len(interaction_keys)), interaction_values,
                             color=plt.cm.coolwarm(np.linspace(0, 1, len(interaction_keys))))

                ax.set_xticks(range(len(interaction_keys)))
                ax.set_xticklabels([key[:10] + "..." if len(key) > 10 else key
                                   for key in interaction_keys], rotation=45, ha='right')
                ax.set_title("Interaction Effects")
                ax.set_ylabel("Interaction Strength")

                plt.tight_layout()

                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Interaction heatmap error: {e}")
            return ""

    async def _create_time_series_chart(self, time_series_data: pd.DataFrame) -> str:
        """時系列チャート"""
        try:
            if 'timestamp' not in time_series_data.columns:
                return ""

            if self.style == "plotly":
                fig = go.Figure()

                for column in time_series_data.columns:
                    if column != 'timestamp':
                        fig.add_trace(go.Scatter(
                            x=time_series_data['timestamp'],
                            y=time_series_data[column],
                            name=column,
                            mode='lines'
                        ))

                fig.update_layout(
                    title="Performance Time Series",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    width=900,
                    height=500
                )

                if self.output_format == "html":
                    return fig.to_html(include_plotlyjs=True)
                else:
                    img_bytes = fig.to_image(format="png", width=900, height=500)
                    img_base64 = base64.b64encode(img_bytes).decode()
                    return f"data:image/png;base64,{img_base64}"

            else:
                fig, ax = plt.subplots(figsize=(12, 6))

                for column in time_series_data.columns:
                    if column != 'timestamp':
                        ax.plot(time_series_data['timestamp'], time_series_data[column],
                               label=column, linewidth=2)

                ax.set_title("Performance Time Series")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()

                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Time series chart error: {e}")
            return ""
