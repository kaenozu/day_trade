#!/usr/bin/env python3
"""
機械学習結果可視化システム - LSTMチャート生成

Issue #315: 高度テクニカル指標・ML機能拡張
LSTM予測結果の専用チャート生成機能
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .types import (
    VisualizationConfig,
    ChartType,
    ChartMetadata,
    DependencyStatus,
    ERROR_MESSAGES,
    CHART_SETTINGS
)
from .data_processor import DataProcessor

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.rcParams["font.family"] = CHART_SETTINGS['font_family']
    plt.rcParams["axes.unicode_minus"] = False
    
    MATPLOTLIB_AVAILABLE = True
    logger.info("matplotlib/seaborn利用可能")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib/seaborn未インストール")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class BaseChartGenerator:
    """
    基本チャート生成クラス
    
    共通のチャート生成機能を提供
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        初期化
        
        Args:
            config: 可視化設定
        """
        self.config = config or VisualizationConfig("output/ml_visualizations")
        self.colors = self.config.color_palette
        self.data_processor = DataProcessor()
        
        # matplotlib設定
        if MATPLOTLIB_AVAILABLE:
            sns.set_style("whitegrid")
            plt.style.use("default")
        
        logger.info("基本チャート生成システム初期化完了")
    
    @property
    def can_create_charts(self) -> bool:
        """チャート作成可能かチェック"""
        return MATPLOTLIB_AVAILABLE
    
    def _setup_figure(self, figsize: Tuple[int, int]) -> Tuple:
        """図の初期設定"""
        if not self.can_create_charts:
            raise RuntimeError(ERROR_MESSAGES['MATPLOTLIB_UNAVAILABLE'])
        
        fig, axes = plt.subplots(figsize=figsize)
        return fig, axes
    
    def _save_figure(
        self, 
        fig, 
        save_path: Path, 
        chart_type: ChartType,
        symbol: str
    ) -> Optional[ChartMetadata]:
        """図の保存とメタデータ作成"""
        try:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            plt.close(fig)
            
            metadata = ChartMetadata(
                chart_type=chart_type,
                symbol=symbol,
                creation_time=datetime.now(),
                file_path=str(save_path),
                file_size=save_path.stat().st_size if save_path.exists() else None
            )
            
            logger.info(f"チャート保存完了: {save_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"チャート保存エラー: {e}")
            if fig:
                plt.close(fig)
            return None


class LSTMChartGenerator(BaseChartGenerator):
    """LSTM予測チャート生成クラス"""
    
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
        if not self.can_create_charts:
            logger.error(ERROR_MESSAGES['MATPLOTLIB_UNAVAILABLE'])
            return None
        
        try:
            fig, axes = plt.subplots(3, 1, figsize=self.config.figsize_small)
            
            # 実際の価格データ
            actual_prices = data["Close"].iloc[-100:]
            dates = actual_prices.index
            
            # 1. 価格チャートと予測
            self._create_price_prediction_subplot(
                axes[0], dates, actual_prices, lstm_results, symbol
            )
            
            # 2. 予測リターン
            self._create_return_prediction_subplot(
                axes[1], lstm_results
            )
            
            # 3. 信頼度とモデル情報
            self._create_confidence_subplot(
                axes[2], lstm_results
            )
            
            # レイアウト調整
            plt.tight_layout()
            
            # ファイル保存
            if save_path is None:
                save_path = self.config.output_dir / f"{symbol}_lstm_prediction.png"
            else:
                save_path = Path(save_path)
            
            metadata = self._save_figure(
                fig, save_path, ChartType.LSTM_PREDICTION, symbol
            )
            
            return str(save_path) if metadata else None
            
        except Exception as e:
            logger.error(f"LSTM予測チャート作成エラー: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None
    
    def _create_price_prediction_subplot(
        self, ax, dates, actual_prices, lstm_results, symbol
    ):
        """価格予測サブプロットの作成"""
        # 実際の価格
        ax.plot(
            dates,
            actual_prices,
            color=self.colors.price,
            linewidth=CHART_SETTINGS['line_width'],
            label="実際価格",
        )
        
        # 予測価格
        if "predicted_prices" in lstm_results:
            pred_prices = lstm_results["predicted_prices"]
            pred_dates = pd.to_datetime(
                lstm_results.get("prediction_dates", [])
            )
            
            if len(pred_dates) == len(pred_prices):
                ax.plot(
                    pred_dates,
                    pred_prices,
                    color=self.colors.prediction,
                    linewidth=CHART_SETTINGS['line_width'],
                    linestyle="--",
                    marker="o",
                    markersize=CHART_SETTINGS['marker_size'],
                    label="LSTM予測",
                )
                
                # 信頼区間
                self._add_confidence_band(
                    ax, pred_dates, pred_prices, lstm_results
                )
        
        ax.set_title(f"{symbol} - LSTM価格予測", fontsize=14, fontweight="bold")
        ax.set_ylabel("価格", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=CHART_SETTINGS['grid_alpha'])
    
    def _add_confidence_band(self, ax, dates, prices, lstm_results):
        """信頼区間の追加"""
        confidence = lstm_results.get("confidence_score", 70)
        if confidence > 0 and self.config.show_confidence_bands:
            error_margin = np.array(prices) * (1 - confidence / 100) * 0.5
            ax.fill_between(
                dates,
                np.array(prices) - error_margin,
                np.array(prices) + error_margin,
                alpha=CHART_SETTINGS['confidence_alpha'],
                color=self.colors.prediction,
                label=f"予測信頼区間({confidence:.0f}%)",
            )
    
    def _create_return_prediction_subplot(self, ax, lstm_results):
        """リターン予測サブプロットの作成"""
        if "predicted_returns" in lstm_results:
            pred_returns = lstm_results["predicted_returns"]
            pred_dates = pd.to_datetime(
                lstm_results.get("prediction_dates", [])
            )
            
            if len(pred_dates) == len(pred_returns):
                colors = [
                    self.colors.bullish if r > 0 else self.colors.bearish
                    for r in pred_returns
                ]
                ax.bar(
                    pred_dates, 
                    pred_returns, 
                    color=colors, 
                    alpha=CHART_SETTINGS['bar_alpha']
                )
                
                # ゼロライン
                ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        
        ax.set_title("LSTM予測リターン", fontsize=14, fontweight="bold")
        ax.set_ylabel("予測リターン (%)", fontsize=12)
        ax.grid(True, alpha=CHART_SETTINGS['grid_alpha'])
    
    def _create_confidence_subplot(self, ax, lstm_results):
        """信頼度サブプロットの作成"""
        confidence = lstm_results.get("confidence_score", 0)
        confidence_color = (
            self.colors.bullish if confidence > 70
            else (
                self.colors.neutral if confidence > 40
                else self.colors.bearish
            )
        )
        
        ax.bar(["予測信頼度"], [confidence], 
               color=confidence_color, alpha=CHART_SETTINGS['bar_alpha'])
        ax.set_ylim(0, 100)
        ax.set_ylabel("信頼度 (%)", fontsize=12)
        ax.set_title("モデル信頼度", fontsize=14, fontweight="bold")
        
        # 信頼度の数値表示
        ax.text(
            0, confidence + 5, f"{confidence:.1f}%",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
        )