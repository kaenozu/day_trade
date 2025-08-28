#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Visualization System - 高度なビジュアライゼーション

チャート作成、予測可視化、パフォーマンスダッシュボードの作成
"""

import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Plotly関連ライブラリ
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .config import DashboardConfig, DashboardTheme


class AdvancedVisualization:
    """高度なビジュアライゼーション"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_enhanced_candlestick_chart(
        self,
        data: pd.DataFrame,
        symbol: str,
        indicators: List[str] = None
    ) -> Dict[str, Any]:
        """拡張ローソク足チャート作成"""
        if not PLOTLY_AVAILABLE:
            return {'success': False, 'error': 'Plotlyが利用できません'}
            
        try:
            # サブプロットの作成
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} Price Chart', 'Volume', 'Indicators'),
                row_width=[0.6, 0.2, 0.2]
            )

            # ローソク足チャート
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol,
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ),
                row=1, col=1
            )

            # 移動平均線の追加
            if 'SMA_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )

            if 'SMA_50' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )

            # ボリンジャーバンド
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill=None
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)'
                    ),
                    row=1, col=1
                )

            # 出来高チャート
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(0,150,255,0.6)'
                ),
                row=2, col=1
            )

            # 技術指標（RSI）
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=3, col=1
                )

                # RSIの70/30ライン
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # レイアウト設定
            fig.update_layout(
                title=f'{symbol} - Enhanced Technical Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark' if self.config.theme == DashboardTheme.DARK else 'plotly_white',
                showlegend=True,
                height=800
            )

            fig.update_xaxes(rangeslider_visible=False)

            return {
                'chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"チャート作成エラー: {e}")
            return {'success': False, 'error': str(e)}

    def create_prediction_chart(
        self,
        historical_data: pd.DataFrame,
        predictions: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """予測チャート作成"""
        if not PLOTLY_AVAILABLE:
            return {'success': False, 'error': 'Plotlyが利用できません'}
            
        try:
            fig = go.Figure()

            # 過去データ
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                )
            )

            # 予測データ
            if 'predictions' in predictions and predictions['predictions']:
                pred_dates = pd.date_range(
                    start=historical_data.index[-1],
                    periods=len(predictions['predictions']),
                    freq='D'
                )

                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=predictions['predictions'],
                        mode='lines+markers',
                        name='Prediction',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6)
                    )
                )

                # 信頼区間
                if 'confidence_intervals' in predictions:
                    upper_bound = predictions['confidence_intervals']['upper']
                    lower_bound = predictions['confidence_intervals']['lower']

                    fig.add_trace(
                        go.Scatter(
                            x=pred_dates,
                            y=upper_bound,
                            mode='lines',
                            name='Upper Bound',
                            line=dict(color='rgba(255,0,0,0.3)', width=1),
                            fill=None
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=pred_dates,
                            y=lower_bound,
                            mode='lines',
                            name='Lower Bound',
                            line=dict(color='rgba(255,0,0,0.3)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.1)'
                        )
                    )

            fig.update_layout(
                title=f'{symbol} - Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark' if self.config.theme == DashboardTheme.DARK else 'plotly_white',
                showlegend=True,
                height=500
            )

            return {
                'chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"予測チャート作成エラー: {e}")
            return {'success': False, 'error': str(e)}

    def create_performance_dashboard(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """パフォーマンスダッシュボード作成"""
        if not PLOTLY_AVAILABLE:
            return {'success': False, 'error': 'Plotlyが利用できません'}
            
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Model Accuracy Trends',
                    'Prediction Confidence',
                    'Feature Importance',
                    'Data Quality Metrics'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )

            # モデル精度トレンド
            if 'accuracy_history' in performance_data:
                accuracy_data = performance_data['accuracy_history']
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(accuracy_data))),
                        y=accuracy_data,
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='green', width=3)
                    ),
                    row=1, col=1
                )

            # 予測信頼度
            if 'confidence_distribution' in performance_data:
                conf_data = performance_data['confidence_distribution']
                fig.add_trace(
                    go.Histogram(
                        x=conf_data,
                        name='Confidence',
                        marker_color='blue',
                        opacity=0.7
                    ),
                    row=1, col=2
                )

            # 特徴量重要度
            if 'feature_importance' in performance_data:
                features = performance_data['feature_importance']
                fig.add_trace(
                    go.Bar(
                        x=list(features.values()),
                        y=list(features.keys()),
                        orientation='h',
                        name='Importance',
                        marker_color='orange'
                    ),
                    row=2, col=1
                )

            # データ品質メトリクス
            if 'data_quality' in performance_data:
                quality_metrics = performance_data['data_quality']
                fig.add_trace(
                    go.Bar(
                        x=list(quality_metrics.keys()),
                        y=list(quality_metrics.values()),
                        name='Quality Score',
                        marker_color='purple'
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                title='System Performance Dashboard',
                template='plotly_dark' if self.config.theme == DashboardTheme.DARK else 'plotly_white',
                showlegend=True,
                height=700
            )

            return {
                'chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"パフォーマンスダッシュボード作成エラー: {e}")
            return {'success': False, 'error': str(e)}

    def create_simple_line_chart(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """シンプルなラインチャート作成"""
        if not PLOTLY_AVAILABLE:
            return {'success': False, 'error': 'Plotlyが利用できません'}
            
        try:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=f'{symbol} Price',
                    line=dict(color='blue', width=2)
                )
            )

            fig.update_layout(
                title=f'{symbol} - Price Chart',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark' if self.config.theme == DashboardTheme.DARK else 'plotly_white',
                height=400
            )

            return {
                'chart': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"シンプルチャート作成エラー: {e}")
            return {'success': False, 'error': str(e)}