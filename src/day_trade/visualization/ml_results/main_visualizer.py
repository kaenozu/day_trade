#!/usr/bin/env python3
"""
機械学習結果可視化システム - メインビジュアライザー

Issue #315: 高度テクニカル指標・ML機能拡張
統合的な機械学習結果可視化システムのメインクラス
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any

import pandas as pd

from .types import (
    VisualizationConfig,
    ChartType,
    ChartMetadata,
    DependencyStatus,
    DEFAULT_CONFIG
)
from .data_processor import DataProcessor
from .chart_generators import LSTMChartGenerator, VolatilityChartGenerator
from .ui_components import InteractiveDashboard, AdvancedChartGenerator
from .report_builder import ReportBuilder

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class MLResultsVisualizer:
    """
    機械学習結果可視化システム - メインクラス
    
    LSTM・GARCH・テクニカル分析・ボラティリティ予測の結果を統合的に可視化
    """
    
    def __init__(self, output_dir: str = "output/ml_visualizations"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        # 設定初期化
        self.config = VisualizationConfig(output_dir=output_dir)
        
        # コンポーネント初期化
        self.data_processor = DataProcessor()
        self.lstm_generator = LSTMChartGenerator(self.config)
        self.volatility_generator = VolatilityChartGenerator(self.config)
        self.advanced_generator = AdvancedChartGenerator(self.config)
        self.interactive_dashboard = InteractiveDashboard(self.config)
        self.report_builder = ReportBuilder(self.config)
        
        # チャートメタデータ管理
        self.chart_metadata: List[ChartMetadata] = []
        
        logger.info(f"機械学習結果可視化システム初期化完了: {self.config.output_dir}")
    
    @property
    def output_dir(self) -> Path:
        """出力ディレクトリのプロパティ"""
        return self.config.output_dir
    
    @property
    def colors(self) -> Dict[str, str]:
        """カラーパレットのプロパティ"""
        return {
            "price": self.config.color_palette.price,
            "prediction": self.config.color_palette.prediction,
            "lstm": self.config.color_palette.lstm,
            "garch": self.config.color_palette.garch,
            "vix": self.config.color_palette.vix,
            "support": self.config.color_palette.support,
            "resistance": self.config.color_palette.resistance,
            "bullish": self.config.color_palette.bullish,
            "bearish": self.config.color_palette.bearish,
            "neutral": self.config.color_palette.neutral,
            "volatility": self.config.color_palette.volatility,
            "confidence": self.config.color_palette.confidence,
        }
    
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
        try:
            # データ前処理
            processed_results = self.data_processor.process_lstm_results(
                lstm_results, data
            )
            if not processed_results:
                logger.error("LSTM結果の処理に失敗")
                return None
            
            # チャート生成
            chart_path = self.lstm_generator.create_lstm_prediction_chart(
                data, lstm_results, symbol, save_path
            )
            
            if chart_path:
                # メタデータ記録
                metadata = ChartMetadata(
                    chart_type=ChartType.LSTM_PREDICTION,
                    symbol=symbol,
                    creation_time=datetime.now(),
                    file_path=chart_path
                )
                self.chart_metadata.append(metadata)
            
            return chart_path
            
        except Exception as e:
            logger.error(f"LSTM予測チャート作成エラー: {e}")
            return None
    
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
        try:
            # データ前処理
            processed_results = self.data_processor.process_volatility_results(
                volatility_results
            )
            if not processed_results:
                logger.error("ボラティリティ結果の処理に失敗")
                return None
            
            # チャート生成
            chart_path = self.volatility_generator.create_volatility_forecast_chart(
                data, volatility_results, symbol, save_path
            )
            
            if chart_path:
                # メタデータ記録
                metadata = ChartMetadata(
                    chart_type=ChartType.VOLATILITY_FORECAST,
                    symbol=symbol,
                    creation_time=datetime.now(),
                    file_path=chart_path
                )
                self.chart_metadata.append(metadata)
            
            return chart_path
            
        except Exception as e:
            logger.error(f"ボラティリティ予測チャート作成エラー: {e}")
            return None
    
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
        try:
            # データ前処理
            processed_results = self.data_processor.process_multiframe_results(
                multiframe_results
            )
            if not processed_results:
                logger.error("マルチタイムフレーム結果の処理に失敗")
                return None
            
            # チャート生成
            chart_path = self.advanced_generator.create_multiframe_analysis_chart(
                data, multiframe_results, symbol, save_path
            )
            
            if chart_path:
                # メタデータ記録
                metadata = ChartMetadata(
                    chart_type=ChartType.MULTIFRAME_ANALYSIS,
                    symbol=symbol,
                    creation_time=datetime.now(),
                    file_path=chart_path
                )
                self.chart_metadata.append(metadata)
            
            return chart_path
            
        except Exception as e:
            logger.error(f"マルチタイムフレーム分析チャート作成エラー: {e}")
            return None
    
    def create_comprehensive_dashboard(
        self,
        data: pd.DataFrame,
        lstm_results: Optional[Dict] = None,
        volatility_results: Optional[Dict] = None,
        multiframe_results: Optional[Dict] = None,
        technical_results: Optional[Dict] = None,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        総合ダッシュボード作成
        
        全ての分析結果を1つのダッシュボードにまとめて可視化
        
        Args:
            data: 価格データ
            lstm_results: LSTM予測結果
            volatility_results: ボラティリティ予測結果
            multiframe_results: マルチタイムフレーム分析結果
            technical_results: 高度テクニカル指標結果
            symbol: 銘柄コード
            save_path: 保存パス
            
        Returns:
            保存されたファイルパス
        """
        try:
            # 結果統合
            merged_results = self.data_processor.merge_analysis_results(
                lstm_results=self.data_processor.process_lstm_results(lstm_results, data) if lstm_results else None,
                volatility_results=self.data_processor.process_volatility_results(volatility_results) if volatility_results else None,
                multiframe_results=self.data_processor.process_multiframe_results(multiframe_results) if multiframe_results else None,
                technical_results=technical_results
            )
            
            # チャート作成用データ準備
            chart_data = self.data_processor.prepare_chart_data(
                data, merged_results
            )
            
            if not chart_data:
                logger.error("チャート作成用データの準備に失敗")
                return None
            
            # 総合ダッシュボードの生成は複雑なため、静的な実装を提供
            chart_path = self._create_static_comprehensive_dashboard(
                data, merged_results, symbol, save_path
            )
            
            if chart_path:
                # メタデータ記録
                metadata = ChartMetadata(
                    chart_type=ChartType.COMPREHENSIVE_DASHBOARD,
                    symbol=symbol,
                    creation_time=datetime.now(),
                    file_path=chart_path
                )
                self.chart_metadata.append(metadata)
            
            return chart_path
            
        except Exception as e:
            logger.error(f"総合ダッシュボード作成エラー: {e}")
            return None
    
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
        try:
            chart_path = self.interactive_dashboard.create_interactive_dashboard(
                data, results_dict, symbol, save_path
            )
            
            if chart_path:
                # メタデータ記録
                metadata = ChartMetadata(
                    chart_type=ChartType.INTERACTIVE_DASHBOARD,
                    symbol=symbol,
                    creation_time=datetime.now(),
                    file_path=chart_path
                )
                self.chart_metadata.append(metadata)
            
            return chart_path
            
        except Exception as e:
            logger.error(f"インタラクティブダッシュボード作成エラー: {e}")
            return None
    
    def generate_analysis_report(
        self, 
        symbol: str, 
        results_dict: Dict,
        save_path: Optional[str] = None
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
            return self.report_builder.generate_analysis_report(
                symbol, results_dict, save_path
            )
        except Exception as e:
            logger.error(f"分析レポート生成エラー: {e}")
            return None
    
    def generate_summary_report(
        self,
        symbol: str,
        results_dict: Dict,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        要約レポート生成
        
        Args:
            symbol: 銘柄コード
            results_dict: 全分析結果の辞書
            save_path: 保存パス
            
        Returns:
            保存されたファイルパス
        """
        try:
            return self.report_builder.generate_summary_report(
                symbol, results_dict, save_path
            )
        except Exception as e:
            logger.error(f"要約レポート生成エラー: {e}")
            return None
    
    def get_chart_metadata(self) -> List[ChartMetadata]:
        """
        生成されたチャートのメタデータ取得
        
        Returns:
            チャートメタデータのリスト
        """
        return self.chart_metadata.copy()
    
    def clear_metadata(self):
        """メタデータのクリア"""
        self.chart_metadata.clear()
        logger.info("チャートメタデータをクリアしました")
    
    def get_dependency_status(self) -> DependencyStatus:
        """
        依存関係の状況取得
        
        Returns:
            依存関係状況
        """
        return DependencyStatus(
            matplotlib_available=self.lstm_generator.can_create_charts,
            plotly_available=self.interactive_dashboard.can_create_interactive_charts,
            seaborn_available=True  # chart_generatorsで確認済み
        )
    
    def _create_static_comprehensive_dashboard(
        self,
        data: pd.DataFrame,
        merged_results: Dict,
        symbol: str,
        save_path: Optional[str]
    ) -> Optional[str]:
        """
        静的総合ダッシュボードの作成
        
        Args:
            data: 価格データ
            merged_results: 統合結果
            symbol: 銘柄コード
            save_path: 保存パス
            
        Returns:
            保存されたファイルパス
        """
        try:
            if not self.lstm_generator.can_create_charts:
                logger.error("matplotlibが利用できません")
                return None
            
            import matplotlib.pyplot as plt
            
            # 大型ダッシュボード作成
            fig = plt.figure(figsize=self.config.figsize_large)
            gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
            
            # 基本的なレイアウトの実装
            self._create_dashboard_price_section(fig, gs, data, merged_results, symbol)
            self._create_dashboard_metrics_section(fig, gs, merged_results)
            self._create_dashboard_summary_section(fig, gs, merged_results)
            
            # タイムスタンプ追加
            if self.config.include_timestamps:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fig.text(
                    0.99, 0.01, f"生成日時: {timestamp}",
                    ha="right", va="bottom", fontsize=8, alpha=0.7,
                )
            
            # ファイル保存
            if save_path is None:
                save_path = self.config.output_dir / f"{symbol}_comprehensive_dashboard.png"
            else:
                save_path = Path(save_path)
            
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close()
            
            logger.info(f"総合ダッシュボード保存完了: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"静的ダッシュボード作成エラー: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None
    
    def _create_dashboard_price_section(self, fig, gs, data, merged_results, symbol):
        """ダッシュボード価格セクション作成"""
        ax_price = fig.add_subplot(gs[0, :])
        recent_data = data.tail(100)
        
        ax_price.plot(
            recent_data.index,
            recent_data["Close"],
            color=self.colors["price"],
            linewidth=3,
            label="価格",
            zorder=3,
        )
        
        # LSTM予測の重ね合わせ
        lstm_data = merged_results.get('lstm')
        if lstm_data and hasattr(lstm_data, 'predicted_prices') and lstm_data.predicted_prices:
            pred_dates = pd.to_datetime(lstm_data.prediction_dates)
            
            ax_price.plot(
                pred_dates,
                lstm_data.predicted_prices,
                color=self.colors["prediction"],
                linewidth=2,
                linestyle="--",
                marker="o",
                markersize=5,
                label="LSTM予測",
                zorder=4,
            )
        
        ax_price.set_title(f"{symbol} - 統合分析ダッシュボード", 
                          fontsize=20, fontweight="bold")
        ax_price.set_ylabel("価格", fontsize=14)
        ax_price.legend(loc="upper left", fontsize=12)
        ax_price.grid(True, alpha=0.3)
    
    def _create_dashboard_metrics_section(self, fig, gs, merged_results):
        """ダッシュボードメトリクスセクション作成"""
        # LSTM信頼度
        ax_lstm = fig.add_subplot(gs[1, 0])
        lstm_data = merged_results.get('lstm')
        if lstm_data:
            confidence = lstm_data.confidence_score if hasattr(lstm_data, 'confidence_score') else 0
            ax_lstm.pie(
                [confidence, 100 - confidence],
                colors=[self.colors["lstm"], "lightgray"],
                startangle=90,
                counterclock=False,
                wedgeprops=dict(width=0.3),
            )
            ax_lstm.text(0, 0, f"{confidence:.0f}%",
                        ha="center", va="center",
                        fontsize=16, fontweight="bold")
            ax_lstm.set_title("LSTM信頼度", fontsize=12, fontweight="bold")
        
        # ボラティリティ状況
        ax_vol = fig.add_subplot(gs[1, 1])
        vol_data = merged_results.get('volatility')
        if vol_data:
            if hasattr(vol_data, 'current_metrics'):
                realized_vol = vol_data.current_metrics.realized_volatility * 100
                vix_like = vol_data.current_metrics.vix_like_indicator
            else:
                realized_vol = 20
                vix_like = 25
            
            ax_vol.bar(
                ["実現ボラティリティ", "VIX風指標"],
                [realized_vol, vix_like],
                color=[self.colors["volatility"], self.colors["vix"]],
                alpha=0.7,
            )
            ax_vol.set_ylabel("値 (%)", fontsize=10)
            ax_vol.set_title("ボラティリティ指標", fontsize=12, fontweight="bold")
            plt.setp(ax_vol.get_xticklabels(), rotation=45, ha="right")
    
    def _create_dashboard_summary_section(self, fig, gs, merged_results):
        """ダッシュボード要約セクション作成"""
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis("off")
        
        # 統合判断の表示
        final_assessment = merged_results.get('final_assessment', {})
        if final_assessment:
            final_action = final_assessment.get('final_action', 'HOLD')
            confidence = final_assessment.get('confidence', 0)
            
            summary_text = f"""
統合分析サマリー

最終判断: {final_action}
信頼度: {confidence:.1f}%
利用可能分析: {len(merged_results.get('available_analyses', []))}個

            """.strip()
            
            ax_summary.text(
                0.5, 0.5, summary_text,
                transform=ax_summary.transAxes,
                fontsize=14, ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
            )