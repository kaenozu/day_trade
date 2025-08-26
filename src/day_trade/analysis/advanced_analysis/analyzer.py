#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Analysis Analyzer - 高度技術分析アナライザー

メインの分析エンジンクラス
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from .types import AnalysisType, TechnicalAnalysisResult
from .indicators import AdvancedTechnicalIndicators
from .patterns import PatternRecognition
from .signals import SignalAnalyzer
from .database import DatabaseManager


class AdvancedTechnicalAnalysis:
    """高度技術分析システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # コンポーネント初期化
        self.indicators = AdvancedTechnicalIndicators()
        self.pattern_recognition = PatternRecognition()
        self.signal_analyzer = SignalAnalyzer()

        # データベース設定
        self.db_path = Path("technical_analysis_data/advanced_analysis.db")
        self.db_path.parent.mkdir(exist_ok=True)

        self.database_manager = DatabaseManager(self.db_path)
        
        self.logger.info("Advanced technical analysis system initialized")

    async def perform_comprehensive_analysis(self, symbol: str, period: str = "3mo") -> TechnicalAnalysisResult:
        """包括的技術分析実行"""

        self.logger.info(f"包括的技術分析開始: {symbol}")

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            if data is None or len(data) < 50:
                raise ValueError("分析に十分なデータがありません")

            # 各種分析実行
            all_signals = []
            all_patterns = []
            all_indicators = {}

            # 1. 高度モメンタム分析
            momentum_indicators = self.indicators.calculate_advanced_momentum(data)
            momentum_signals = self.signal_analyzer.analyze_momentum_signals(momentum_indicators, data)
            all_signals.extend(momentum_signals)
            all_indicators.update({k: v.iloc[-1] if len(v) > 0 else 0 for k, v in momentum_indicators.items()})

            # 2. 高度トレンド分析
            trend_indicators = self.indicators.calculate_advanced_trend(data)
            trend_signals = self.signal_analyzer.analyze_trend_signals(trend_indicators, data)
            all_signals.extend(trend_signals)
            all_indicators.update({k: v.iloc[-1] if len(v) > 0 else 0 for k, v in trend_indicators.items()})

            # 3. 高度ボラティリティ分析
            volatility_indicators = self.indicators.calculate_advanced_volatility(data)
            volatility_signals = self.signal_analyzer.analyze_volatility_signals(volatility_indicators, data)
            all_signals.extend(volatility_signals)
            all_indicators.update({k: v.iloc[-1] if len(v) > 0 else 0 for k, v in volatility_indicators.items()})

            # 4. 高度ボリューム分析
            volume_indicators = self.indicators.calculate_advanced_volume(data)
            volume_signals = self.signal_analyzer.analyze_volume_signals(volume_indicators, data)
            all_signals.extend(volume_signals)
            all_indicators.update({k: v.iloc[-1] if len(v) > 0 else 0 for k, v in volume_indicators.items()})

            # 5. パターン認識
            candlestick_patterns = self.pattern_recognition.detect_candlestick_patterns(data)
            price_patterns = self.pattern_recognition.detect_price_patterns(data)
            all_patterns.extend(candlestick_patterns)
            all_patterns.extend(price_patterns)

            # 6. 総合判定
            overall_sentiment, confidence_score = self.signal_analyzer.calculate_overall_sentiment(all_signals)
            risk_level = self.signal_analyzer.assess_risk_level(all_indicators, volatility_indicators)
            recommendations = self.signal_analyzer.generate_recommendations(all_signals, all_patterns, overall_sentiment)

            # 結果作成
            result = TechnicalAnalysisResult(
                symbol=symbol,
                analysis_type=AnalysisType.TREND_ANALYSIS,  # メインタイプ
                signals=all_signals,
                patterns=all_patterns,
                indicators=all_indicators,
                overall_sentiment=overall_sentiment,
                confidence_score=confidence_score,
                risk_level=risk_level,
                recommendations=recommendations,
                timestamp=datetime.now()
            )

            # 結果保存
            await self.database_manager.save_analysis_result(result)

            self.logger.info(f"包括的技術分析完了: {len(all_signals)}シグナル, {len(all_patterns)}パターン")

            return result

        except Exception as e:
            self.logger.error(f"包括的技術分析エラー: {e}")
            raise