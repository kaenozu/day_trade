#!/usr/bin/env python3
"""
機械学習結果可視化システム - UI関連コンポーネント

Issue #315: 高度テクニカル指標・ML機能拡張
インタラクティブダッシュボード、アドバンストチャート、UI要素の生成
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

import numpy as np
import pandas as pd

from .types import (
    VisualizationConfig,
    ChartType,
    ChartMetadata,
    DependencyStatus,
    get_signal_color,
    get_risk_color,
    CHART_SETTINGS,
    ERROR_MESSAGES
)
from .chart_generators import BaseChartGenerator

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# Plotly依存性チェック
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    logger.info("plotly利用可能")
except ImportError:
    logger.warning("plotly未インストール")

# Matplotlib依存性チェック
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib未インストール")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class InteractiveDashboard:
    """
    インタラクティブダッシュボード生成クラス
    
    Plotlyを使用したインタラクティブなWebダッシュボードを生成
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        初期化
        
        Args:
            config: 可視化設定
        """
        self.config = config or VisualizationConfig("output/ml_visualizations")
        self.colors = self.config.color_palette
        logger.info("インタラクティブダッシュボード初期化")
    
    @property
    def can_create_interactive_charts(self) -> bool:
        """インタラクティブチャート作成可能かチェック"""
        return PLOTLY_AVAILABLE
    
    def create_interactive_dashboard(
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
        if not self.can_create_interactive_charts:
            logger.error(ERROR_MESSAGES['PLOTLY_UNAVAILABLE'])
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
            
            # 各セクションの作成
            self._add_price_and_prediction_section(fig, data, results_dict)
            self._add_volatility_section(fig, results_dict)
            self._add_multiframe_section(fig, results_dict)
            self._add_technical_indicators_section(fig, results_dict)
            self._add_risk_evaluation_section(fig, results_dict)
            self._add_integrated_signals_section(fig, results_dict)
            self._add_prediction_accuracy_section(fig, results_dict)
            
            # レイアウト設定
            fig.update_layout(
                title=f"{symbol} - 機械学習統合分析ダッシュボード",
                height=1200,
                showlegend=True,
            )
            
            # ファイル保存
            if save_path is None:
                save_path = self.config.output_dir / f"{symbol}_interactive_dashboard.html"
            else:
                save_path = Path(save_path)
            
            fig.write_html(str(save_path))
            
            logger.info(f"インタラクティブダッシュボード保存完了: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"インタラクティブダッシュボード作成エラー: {e}")
            return None
    
    def _add_price_and_prediction_section(self, fig, data: pd.DataFrame, results_dict: Dict):
        """価格と予測セクションの追加"""
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
        
        # LSTM予測の追加
        lstm_results = results_dict.get("lstm", {})
        if lstm_results:
            pred_data = self._extract_prediction_data(lstm_results)
            if pred_data:
                fig.add_trace(
                    go.Scatter(
                        x=pred_data['dates'],
                        y=pred_data['prices'],
                        mode="lines+markers",
                        name="LSTM予測",
                        line=dict(color="orange", width=2, dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
    
    def _add_volatility_section(self, fig, results_dict: Dict):
        """ボラティリティセクションの追加"""
        vol_results = results_dict.get("volatility", {})
        if vol_results:
            vol_data = self._extract_volatility_data(vol_results)
            if vol_data:
                fig.add_trace(
                    go.Bar(
                        x=vol_data['models'],
                        y=vol_data['values'],
                        name="ボラティリティ予測",
                        marker_color="purple",
                    ),
                    row=1,
                    col=2,
                )
    
    def _add_multiframe_section(self, fig, results_dict: Dict):
        """マルチタイムフレームセクションの追加"""
        mf_results = results_dict.get("multiframe", {})
        if mf_results:
            mf_data = self._extract_multiframe_data(mf_results)
            if mf_data:
                fig.add_trace(
                    go.Bar(
                        x=mf_data['timeframes'],
                        y=mf_data['strengths'],
                        name="トレンド強度",
                        marker_color="green",
                    ),
                    row=2,
                    col=1,
                )
    
    def _add_technical_indicators_section(self, fig, results_dict: Dict):
        """テクニカル指標セクションの追加"""
        mf_results = results_dict.get("multiframe", {})
        if mf_results:
            tech_data = self._extract_technical_data(mf_results)
            if tech_data:
                fig.add_trace(
                    go.Bar(
                        x=tech_data['indicators'],
                        y=tech_data['values'],
                        name="テクニカル指標",
                        marker_color="red",
                    ),
                    row=2,
                    col=2,
                )
    
    def _add_risk_evaluation_section(self, fig, results_dict: Dict):
        """リスク評価セクションの追加"""
        risk_data = self._extract_risk_data(results_dict)
        if risk_data:
            fig.add_trace(
                go.Bar(
                    x=risk_data['labels'],
                    y=risk_data['scores'],
                    name="リスクスコア",
                    marker_color="orange",
                ),
                row=3,
                col=1,
            )
    
    def _add_integrated_signals_section(self, fig, results_dict: Dict):
        """統合シグナルセクションの追加"""
        signal_data = self._extract_signal_data(results_dict)
        if signal_data:
            fig.add_trace(
                go.Pie(
                    labels=list(signal_data.keys()),
                    values=list(signal_data.values()),
                    name="統合シグナル",
                ),
                row=3,
                col=2,
            )
    
    def _add_prediction_accuracy_section(self, fig, results_dict: Dict):
        """予測精度セクションの追加"""
        lstm_results = results_dict.get("lstm", {})
        if lstm_results:
            confidence = self._extract_confidence_data(lstm_results)
            if confidence is not None:
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
    
    # データ抽出ヘルパーメソッド
    def _extract_prediction_data(self, lstm_results) -> Optional[Dict]:
        """LSTM予測データの抽出"""
        try:
            if hasattr(lstm_results, 'predicted_prices'):
                pred_prices = lstm_results.predicted_prices
                pred_dates = pd.to_datetime(lstm_results.prediction_dates)
            else:
                pred_prices = lstm_results.get("predicted_prices", [])
                pred_dates = pd.to_datetime(lstm_results.get("prediction_dates", []))
            
            if pred_prices and len(pred_dates) == len(pred_prices):
                return {'dates': pred_dates, 'prices': pred_prices}
        except Exception as e:
            logger.warning(f"LSTM予測データ抽出エラー: {e}")
        return None
    
    def _extract_volatility_data(self, vol_results) -> Optional[Dict]:
        """ボラティリティデータの抽出"""
        try:
            if hasattr(vol_results, 'ensemble_forecast'):
                individual = vol_results.ensemble_forecast.individual_forecasts
            else:
                ensemble = vol_results.get("ensemble_forecast", {})
                individual = ensemble.get("individual_forecasts", {})
            
            if individual:
                return {
                    'models': list(individual.keys()),
                    'values': list(individual.values())
                }
        except Exception as e:
            logger.warning(f"ボラティリティデータ抽出エラー: {e}")
        return None
    
    def _extract_multiframe_data(self, mf_results) -> Optional[Dict]:
        """マルチタイムフレームデータの抽出"""
        try:
            if hasattr(mf_results, 'timeframes'):
                timeframes = mf_results.timeframes
            else:
                timeframes = mf_results.get("timeframes", {})
            
            tf_names = []
            strengths = []
            
            for tf_key, tf_data in timeframes.items():
                if hasattr(tf_data, 'timeframe'):
                    tf_names.append(tf_data.timeframe)
                    strengths.append(tf_data.trend_strength)
                else:
                    tf_names.append(tf_data.get("timeframe", tf_key))
                    strengths.append(tf_data.get("trend_strength", 50))
            
            if tf_names:
                return {'timeframes': tf_names, 'strengths': strengths}
        except Exception as e:
            logger.warning(f"マルチタイムフレームデータ抽出エラー: {e}")
        return None
    
    def _extract_technical_data(self, mf_results) -> Optional[Dict]:
        """テクニカルデータの抽出"""
        try:
            if hasattr(mf_results, 'timeframes'):
                timeframes = mf_results.timeframes
            else:
                timeframes = mf_results.get("timeframes", {})
            
            daily_data = None
            for tf_key, tf_data in timeframes.items():
                if (hasattr(tf_data, 'timeframe') and tf_data.timeframe == "日足") or tf_key == "daily":
                    daily_data = tf_data
                    break
            
            if daily_data:
                if hasattr(daily_data, 'technical_indicators'):
                    indicators = daily_data.technical_indicators
                    indicator_dict = {
                        'RSI': indicators.rsi,
                        'MACD': indicators.macd,
                        'BB位置': indicators.bb_position
                    }
                else:
                    indicators = daily_data.get("technical_indicators", {})
                    indicator_dict = {
                        'RSI': indicators.get('rsi'),
                        'MACD': indicators.get('macd'),
                        'BB位置': indicators.get('bb_position')
                    }
                
                # None値を除去
                valid_indicators = {k: v for k, v in indicator_dict.items() if v is not None}
                
                if valid_indicators:
                    return {
                        'indicators': list(valid_indicators.keys()),
                        'values': list(valid_indicators.values())
                    }
        except Exception as e:
            logger.warning(f"テクニカルデータ抽出エラー: {e}")
        return None
    
    def _extract_risk_data(self, results_dict: Dict) -> Optional[Dict]:
        """リスクデータの抽出"""
        risk_scores = []
        risk_labels = []
        
        # ボラティリティリスク
        vol_results = results_dict.get("volatility", {})
        if vol_results:
            if hasattr(vol_results, 'risk_assessment'):
                risk_score = vol_results.risk_assessment.risk_score
            else:
                risk_assessment = vol_results.get("risk_assessment", {})
                risk_score = risk_assessment.get("risk_score", 0)
            
            risk_scores.append(risk_score)
            risk_labels.append("ボラティリティリスク")
        
        # マルチタイムフレームリスク
        mf_results = results_dict.get("multiframe", {})
        if mf_results:
            if hasattr(mf_results, 'integrated_analysis'):
                risk_info = mf_results.integrated_analysis.risk_assessment
                if risk_info and hasattr(risk_info, 'risk_score'):
                    risk_scores.append(risk_info.risk_score)
                    risk_labels.append("マルチTFリスク")
            else:
                integrated = mf_results.get("integrated_analysis", {})
                risk_info = integrated.get("risk_assessment", {})
                if risk_info:
                    risk_score = risk_info.get("risk_score", 0)
                    risk_scores.append(risk_score)
                    risk_labels.append("マルチTFリスク")
        
        if risk_scores:
            return {'scores': risk_scores, 'labels': risk_labels}
        return None
    
    def _extract_signal_data(self, results_dict: Dict) -> Optional[Dict]:
        """シグナルデータの抽出"""
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        # LSTMシグナル
        lstm_results = results_dict.get("lstm", {})
        if lstm_results:
            if hasattr(lstm_results, 'predicted_returns'):
                returns = lstm_results.predicted_returns
            else:
                returns = lstm_results.get("predicted_returns", [])
            
            if returns:
                avg_return = np.mean(returns)
                if avg_return > 1:
                    signal_counts["BUY"] += 1
                elif avg_return < -1:
                    signal_counts["SELL"] += 1
                else:
                    signal_counts["HOLD"] += 1
        
        # マルチタイムフレームシグナル
        mf_results = results_dict.get("multiframe", {})
        if mf_results:
            if hasattr(mf_results, 'integrated_analysis'):
                action = str(mf_results.integrated_analysis.integrated_signal.action)
            else:
                integrated = mf_results.get("integrated_analysis", {})
                signal_info = integrated.get("integrated_signal", {})
                action = signal_info.get("action", "HOLD")
            
            signal_counts[action] += 1
        
        return signal_counts if any(signal_counts.values()) else None
    
    def _extract_confidence_data(self, lstm_results) -> Optional[float]:
        """信頼度データの抽出"""
        try:
            if hasattr(lstm_results, 'confidence_score'):
                return lstm_results.confidence_score
            else:
                return lstm_results.get("confidence_score", 0)
        except Exception as e:
            logger.warning(f"信頼度データ抽出エラー: {e}")
        return None


class AdvancedChartGenerator(BaseChartGenerator):
    """
    高度なチャート生成クラス
    
    マルチタイムフレーム分析、総合ダッシュボードなど複雑なチャートを生成
    """
    
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
        if not MATPLOTLIB_AVAILABLE:
            logger.error(ERROR_MESSAGES['MATPLOTLIB_UNAVAILABLE'])
            return None
        
        try:
            fig = plt.figure(figsize=self.config.figsize_large)
            
            # グリッドレイアウト作成
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 各セクションの作成
            self._create_main_price_chart(fig, gs, data, multiframe_results, symbol)
            self._create_trend_strength_chart(fig, gs, multiframe_results)
            self._create_technical_comparison_chart(fig, gs, multiframe_results)
            self._create_integrated_analysis_section(fig, gs, multiframe_results)
            self._create_recommendation_section(fig, gs, multiframe_results)
            
            # ファイル保存
            if save_path is None:
                save_path = self.config.output_dir / f"{symbol}_multiframe_analysis.png"
            else:
                save_path = Path(save_path)
            
            metadata = self._save_figure(
                fig, save_path, ChartType.MULTIFRAME_ANALYSIS, symbol
            )
            
            return str(save_path) if metadata else None
            
        except Exception as e:
            logger.error(f"マルチタイムフレーム分析チャート作成エラー: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None
    
    def _create_main_price_chart(self, fig, gs, data, multiframe_results, symbol):
        """メイン価格チャートの作成"""
        ax_main = fig.add_subplot(gs[0, :])
        recent_data = data.tail(100)
        
        ax_main.plot(
            recent_data.index,
            recent_data["Close"],
            color=self.colors.price,
            linewidth=CHART_SETTINGS['line_width'],
            label="価格",
        )
        
        # サポート・レジスタンスライン
        timeframes = multiframe_results.get("timeframes", {})
        daily_data = timeframes.get("daily", {})
        
        support = daily_data.get("support_level")
        resistance = daily_data.get("resistance_level")
        
        if support:
            ax_main.axhline(
                y=support,
                color=self.colors.support,
                linestyle="--",
                alpha=0.7,
                label=f"サポート: {support:.0f}",
            )
        if resistance:
            ax_main.axhline(
                y=resistance,
                color=self.colors.resistance,
                linestyle="--",
                alpha=0.7,
                label=f"レジスタンス: {resistance:.0f}",
            )
        
        ax_main.set_title(f"{symbol} - マルチタイムフレーム分析", 
                         fontsize=16, fontweight="bold")
        ax_main.set_ylabel("価格", fontsize=12)
        ax_main.legend(loc="upper left")
        ax_main.grid(True, alpha=CHART_SETTINGS['grid_alpha'])
    
    def _create_trend_strength_chart(self, fig, gs, multiframe_results):
        """トレンド強度チャートの作成"""
        ax_trend = fig.add_subplot(gs[1, 0])
        
        timeframes = multiframe_results.get("timeframes", {})
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
                    colors.append(self.colors.bullish)
                elif "downtrend" in direction:
                    colors.append(self.colors.bearish)
                else:
                    colors.append(self.colors.neutral)
            
            ax_trend.barh(tf_names, strengths, color=colors, 
                         alpha=CHART_SETTINGS['bar_alpha'])
            ax_trend.set_xlim(0, 100)
            ax_trend.set_xlabel("トレンド強度", fontsize=10)
            ax_trend.set_title("時間軸別トレンド強度", fontsize=12, fontweight="bold")
    
    def _create_technical_comparison_chart(self, fig, gs, multiframe_results):
        """テクニカル指標比較チャートの作成"""
        ax_tech = fig.add_subplot(gs[1, 1])
        
        timeframes = multiframe_results.get("timeframes", {})
        tech_data = []
        tech_labels = []
        
        for tf_key, tf_data in timeframes.items():
            tf_name = tf_data.get("timeframe", tf_key)
            indicators = tf_data.get("technical_indicators", {})
            
            if "rsi" in indicators:
                tech_data.append(indicators["rsi"])
                tech_labels.append(f"{tf_name} RSI")
        
        if tech_data:
            ax_tech.bar(tech_labels, tech_data, 
                       color=self.colors.volatility, 
                       alpha=CHART_SETTINGS['bar_alpha'])
            ax_tech.axhline(y=70, color=self.colors.bearish, 
                           linestyle="--", alpha=0.5, label="過買われ")
            ax_tech.axhline(y=30, color=self.colors.bullish, 
                           linestyle="--", alpha=0.5, label="過売られ")
            ax_tech.set_ylim(0, 100)
            ax_tech.set_ylabel("RSI", fontsize=10)
            ax_tech.set_title("時間軸別RSI", fontsize=12, fontweight="bold")
            ax_tech.legend(fontsize=8)
            plt.setp(ax_tech.get_xticklabels(), rotation=45, ha="right")
    
    def _create_integrated_analysis_section(self, fig, gs, multiframe_results):
        """統合分析セクションの作成"""
        ax_integrated = fig.add_subplot(gs[1, 2])
        ax_integrated.axis("off")
        
        integrated = multiframe_results.get("integrated_analysis", {})
        if integrated:
            overall_trend = integrated.get("overall_trend", "unknown")
            confidence = integrated.get("trend_confidence", 0)
            consistency = integrated.get("consistency_score", 0)
            
            results_text = f"""
統合分析結果

総合トレンド: {overall_trend}
トレンド信頼度: {confidence:.1f}%
整合性スコア: {consistency:.1f}%
            """.strip()
            
            ax_integrated.text(
                0.1, 0.9, results_text,
                transform=ax_integrated.transAxes,
                fontsize=11, verticalalignment="top",
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
                    0.1, 0.5, signal_text,
                    transform=ax_integrated.transAxes,
                    fontsize=11, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3),
                )
    
    def _create_recommendation_section(self, fig, gs, multiframe_results):
        """投資推奨セクションの作成"""
        ax_recommend = fig.add_subplot(gs[2, :])
        ax_recommend.axis("off")
        
        integrated = multiframe_results.get("integrated_analysis", {})
        recommendation = integrated.get("investment_recommendation", {})
        
        if recommendation:
            rec_action = recommendation.get("recommendation", "HOLD")
            position_size = recommendation.get("position_size", "NEUTRAL")
            confidence = recommendation.get("confidence", 0)
            
            # 推奨の色分け
            rec_color = get_signal_color(rec_action)
            
            recommendation_text = f"【投資推奨】 {rec_action} (ポジションサイズ: {position_size}, 信頼度: {confidence:.1f}%)"
            ax_recommend.text(
                0.5, 0.9, recommendation_text,
                transform=ax_recommend.transAxes,
                fontsize=14, fontweight="bold",
                ha="center",
                bbox=dict(boxstyle="round", facecolor=rec_color, alpha=0.2),
            )
            
            # 推奨理由
            reasons = recommendation.get("reasons", [])
            if reasons:
                reasons_text = "推奨理由:\n" + "\n".join([
                    f"• {reason}" for reason in reasons[:4]
                ])
                ax_recommend.text(
                    0.1, 0.6, reasons_text,
                    transform=ax_recommend.transAxes,
                    fontsize=10, verticalalignment="top",
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
                    0.7, 0.6, price_text,
                    transform=ax_recommend.transAxes,
                    fontsize=10, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )