#!/usr/bin/env python3
"""
マルチタイムフレーム分析システム - メインクラス
Issue #315: 高度テクニカル指標・ML機能拡張

複数時間軸（日足・週足・月足）を統合した包括的トレンド分析
"""

import warnings
from datetime import datetime
from typing import Dict

import pandas as pd

from ...utils.logging_config import get_context_logger
from .integrated_analyzer import IntegratedAnalyzer
from .risk_evaluator import RiskEvaluator
from .technical_calculator import TechnicalCalculator
from .trend_analyzer import TrendAnalyzer

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class MultiTimeframeAnalyzer:
    """
    マルチタイムフレーム分析クラス

    日足・週足・月足の複数時間軸でトレンド分析を実行し、
    時間軸間の整合性チェックと統合判定を提供
    """

    def __init__(self):
        """初期化"""
        self.technical_calculator = TechnicalCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.integrated_analyzer = IntegratedAnalyzer()
        self.risk_evaluator = RiskEvaluator()
        
        self.analysis_cache = {}

        logger.info("マルチタイムフレーム分析システム初期化完了")

    def analyze_multiple_timeframes(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> Dict[str, any]:
        """
        複数時間軸統合分析

        Args:
            data: 日足価格データ
            symbol: 銘柄コード

        Returns:
            統合分析結果辞書
        """
        try:
            analysis_results = {
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "timeframes": {},
                "integrated_analysis": {},
            }

            # 各時間軸で分析実行
            timeframe_data = {}
            
            timeframes = self.technical_calculator.resampler.get_all_timeframes()

            for tf_key, tf_info in timeframes.items():
                logger.info(f"{tf_info['name']}分析開始: {symbol}")

                # 指標計算
                tf_indicators = self.technical_calculator.calculate_timeframe_indicators(
                    data, tf_key
                )

                if tf_indicators.empty:
                    logger.warning(f"{tf_key}分析スキップ: データ不足")
                    continue

                timeframe_data[tf_key] = tf_indicators

                # トレンド分析
                trend_result = self.trend_analyzer.analyze_trend_data(
                    tf_indicators, tf_key
                )
                
                # テクニカル指標サマリー取得
                technical_summary = self.technical_calculator.get_indicator_summary(
                    tf_indicators, tf_key
                )

                # 分析結果統合
                tf_analysis = {
                    "timeframe": tf_info["name"],
                    "data_points": len(tf_indicators),
                    "current_price": technical_summary.get("current_price", 0),
                    "trend_direction": trend_result.get("trend_direction", "unknown"),
                    "trend_strength": trend_result.get("trend_strength", 50),
                    "technical_indicators": technical_summary.get("technical_indicators", {}),
                    "support_level": trend_result.get("support_level"),
                    "resistance_level": trend_result.get("resistance_level"),
                    "ichimoku_signal": technical_summary.get("ichimoku_signal"),
                }

                analysis_results["timeframes"][tf_key] = tf_analysis

            # 統合分析実行
            if len(analysis_results["timeframes"]) >= 2:
                integrated = self.integrated_analyzer.perform_integrated_analysis(
                    analysis_results["timeframes"]
                )
                
                # リスク評価
                risk_assessment = self.risk_evaluator.assess_multi_timeframe_risk(
                    analysis_results["timeframes"]
                )
                
                # 投資推奨
                investment_recommendation = self.risk_evaluator.generate_investment_recommendation(
                    integrated["integrated_signal"], 
                    risk_assessment, 
                    analysis_results["timeframes"]
                )
                
                integrated["risk_assessment"] = risk_assessment
                integrated["investment_recommendation"] = investment_recommendation
                
                analysis_results["integrated_analysis"] = integrated

                logger.info(f"マルチタイムフレーム分析完了: {symbol}")
            else:
                logger.warning(
                    f"統合分析スキップ: 分析可能な時間軸が不足 ({len(analysis_results['timeframes'])})"
                )
                analysis_results["integrated_analysis"] = {
                    "overall_trend": "insufficient_data",
                    "confidence": 0,
                    "message": "統合分析に十分な時間軸データがありません",
                }

            return analysis_results

        except Exception as e:
            logger.error(f"マルチタイムフレーム分析エラー ({symbol}): {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
            }
    
    def resample_to_timeframe(
        self, data: pd.DataFrame, timeframe: str, method: str = "last"
    ) -> pd.DataFrame:
        """
        後方互換性のためのリサンプリングメソッド
        
        Args:
            data: 元の価格データ（日足想定）
            timeframe: 'daily', 'weekly', 'monthly'
            method: リサンプリング方法

        Returns:
            リサンプリングされたDataFrame
        """
        return self.technical_calculator.resampler.resample_to_timeframe(
            data, timeframe, method
        )
    
    def calculate_timeframe_indicators(
        self, data: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        後方互換性のためのテクニカル指標計算メソッド
        
        Args:
            data: 価格データ
            timeframe: 時間軸

        Returns:
            テクニカル指標を含むDataFrame
        """
        return self.technical_calculator.calculate_timeframe_indicators(
            data, timeframe
        )