#!/usr/bin/env python3
"""
機械学習結果可視化システム - ボラティリティチャート生成

Issue #315: 高度テクニカル指標・ML機能拡張
ボラティリティ予測結果の専用チャート生成機能
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .types import (
    VisualizationConfig,
    ChartType,
    ChartMetadata,
    ERROR_MESSAGES,
    CHART_SETTINGS,
    get_risk_color
)
from .lstm_charts import BaseChartGenerator

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未インストール")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class VolatilityChartGenerator(BaseChartGenerator):
    """ボラティリティ予測チャート生成クラス"""
    
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
        if not MATPLOTLIB_AVAILABLE:
            logger.error(ERROR_MESSAGES['MATPLOTLIB_UNAVAILABLE'])
            return None
        
        try:
            fig, axes = plt.subplots(4, 1, figsize=(15, 16))
            
            # 過去60日のデータ
            recent_data = data.tail(60)
            dates = recent_data.index
            
            # 1. 価格とVIX風指標
            self._create_price_vix_subplot(
                axes[0], recent_data, dates, volatility_results, symbol
            )
            
            # 2. ボラティリティ予測
            self._create_volatility_prediction_subplot(
                axes[1], volatility_results
            )
            
            # 3. リスク評価
            self._create_risk_assessment_subplot(
                axes[2], volatility_results
            )
            
            # 4. 投資への示唆
            self._create_investment_implications_subplot(
                axes[3], volatility_results
            )
            
            # レイアウト調整
            plt.tight_layout()
            
            # ファイル保存
            if save_path is None:
                save_path = self.config.output_dir / f"{symbol}_volatility_forecast.png"
            else:
                save_path = Path(save_path)
            
            metadata = self._save_figure(
                fig, save_path, ChartType.VOLATILITY_FORECAST, symbol
            )
            
            return str(save_path) if metadata else None
            
        except Exception as e:
            logger.error(f"ボラティリティチャート作成エラー: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None
    
    def _create_price_vix_subplot(
        self, ax, recent_data, dates, volatility_results, symbol
    ):
        """価格とVIX風指標サブプロット作成"""
        ax_twin = ax.twinx()
        
        # 価格
        ax.plot(
            dates, recent_data["Close"],
            color=self.colors.price, linewidth=CHART_SETTINGS['line_width'],
            label="価格",
        )
        ax.set_ylabel("価格", color=self.colors.price, fontsize=12)
        ax.tick_params(axis="y", labelcolor=self.colors.price)
        
        # VIX風指標
        current_metrics = volatility_results.get("current_metrics", {})
        if "vix_like_indicator" in current_metrics:
            vix_value = current_metrics["vix_like_indicator"]
            vix_series = [vix_value] * len(dates)
            
            ax_twin.plot(
                dates, vix_series,
                color=self.colors.vix, linewidth=CHART_SETTINGS['line_width'],
                linestyle="--", label="VIX風指標",
            )
            ax_twin.set_ylabel("VIX風指標", color=self.colors.vix, fontsize=12)
            ax_twin.tick_params(axis="y", labelcolor=self.colors.vix)
        
        ax.set_title(f"{symbol} - 価格とボラティリティ指標", 
                    fontsize=14, fontweight="bold")
        ax.grid(True, alpha=CHART_SETTINGS['grid_alpha'])
    
    def _create_volatility_prediction_subplot(self, ax, volatility_results):
        """ボラティリティ予測サブプロット作成"""
        current_metrics = volatility_results.get("current_metrics", {})
        ensemble = volatility_results.get("ensemble_forecast", {})
        
        if ensemble:
            current_vol = current_metrics.get("realized_volatility", 0) * 100
            ensemble_vol = ensemble.get("ensemble_volatility", current_vol)
            
            # 現在値と予測値
            ax.bar(
                ["現在のボラティリティ", "アンサンブル予測"],
                [current_vol, ensemble_vol],
                color=[self.colors.neutral, self.colors.volatility],
                alpha=CHART_SETTINGS['bar_alpha'],
            )
            
            # 個別モデル予測
            individual = ensemble.get("individual_forecasts", {})
            if individual:
                model_names = list(individual.keys())
                model_values = list(individual.values())
                
                colors_map = [self.colors.lstm, self.colors.garch, self.colors.vix]
                model_colors = colors_map[:len(model_names)]
                
                ax.bar(
                    model_names, model_values,
                    color=model_colors, alpha=0.6,
                )
        
        ax.set_ylabel("ボラティリティ (%)", fontsize=12)
        ax.set_title("ボラティリティ予測比較", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=CHART_SETTINGS['grid_alpha'])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    def _create_risk_assessment_subplot(self, ax, volatility_results):
        """リスク評価サブプロット作成"""
        risk_assessment = volatility_results.get("risk_assessment", {})
        if risk_assessment:
            risk_level = risk_assessment.get("risk_level", "MEDIUM")
            risk_score = risk_assessment.get("risk_score", 50)
            
            risk_color = get_risk_color(risk_level)
            
            ax.barh(
                ["リスクスコア"], [risk_score],
                color=risk_color, alpha=CHART_SETTINGS['bar_alpha'],
            )
            ax.set_xlim(0, 100)
            ax.set_xlabel("リスクスコア", fontsize=12)
            ax.set_title(f"リスク評価: {risk_level}", 
                        fontsize=14, fontweight="bold")
            
            # リスクスコアの数値表示
            ax.text(
                risk_score + 2, 0, f"{risk_score}",
                va="center", fontsize=12, fontweight="bold",
            )
            
            # リスク要因表示
            risk_factors = risk_assessment.get("risk_factors", [])
            if risk_factors:
                factors_text = "\n".join([
                    f"• {factor}" for factor in risk_factors[:3]
                ])
                ax.text(
                    0.02, 0.95, factors_text,
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
    
    def _create_investment_implications_subplot(self, ax, volatility_results):
        """投資示唆サブプロット作成"""
        ax.axis("off")
        
        implications = volatility_results.get("investment_implications", {})
        if implications:
            y_pos = 0.9
            
            category_labels = {
                "portfolio_adjustments": "ポートフォリオ調整",
                "trading_strategies": "トレーディング戦略",
                "risk_management": "リスク管理",
                "market_timing": "マーケットタイミング",
            }
            
            for category, suggestions in implications.items():
                if suggestions and category != "error":
                    category_jp = category_labels.get(category, category)
                    
                    ax.text(
                        0.02, y_pos, f"【{category_jp}】",
                        transform=ax.transAxes,
                        fontsize=12, fontweight="bold",
                        color=self.colors.price,
                    )
                    y_pos -= 0.08
                    
                    for suggestion in suggestions[:2]:
                        ax.text(
                            0.05, y_pos, f"• {suggestion}",
                            transform=ax.transAxes, fontsize=10,
                        )
                        y_pos -= 0.06
                    y_pos -= 0.02
        
        ax.set_title("投資への示唆", fontsize=14, fontweight="bold")