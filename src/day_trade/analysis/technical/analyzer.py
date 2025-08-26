#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Analyzer Main Class
高度技術分析メインクラス
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from .base import AdvancedAnalysis
from .trend_indicators import TrendIndicators
from .momentum_indicators import MomentumIndicators
from .volatility_indicators import VolatilityIndicators
from .volume_indicators import VolumeIndicators
from .statistical_analysis import StatisticalAnalysis
from .ml_prediction import MLPredictor
from .data_provider import TechnicalDataProvider
from .signal_generator import SignalGenerator

try:
    from enhanced_symbol_manager import EnhancedSymbolManager
    ENHANCED_SYMBOLS_AVAILABLE = True
except ImportError:
    ENHANCED_SYMBOLS_AVAILABLE = False


class AdvancedTechnicalAnalyzer:
    """
    高度技術指標・分析手法拡張システム
    先進的技術分析の包括的実装
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データディレクトリ
        self.data_dir = Path("advanced_technical_data")
        self.data_dir.mkdir(exist_ok=True)

        # 拡張システム統合
        if ENHANCED_SYMBOLS_AVAILABLE:
            self.symbol_manager = EnhancedSymbolManager()

        # 技術指標計算用キャッシュ
        self.indicator_cache = {}

        # 各種計算クラス初期化
        self.trend_indicators = TrendIndicators()
        self.momentum_indicators = MomentumIndicators()
        self.volatility_indicators = VolatilityIndicators()
        self.volume_indicators = VolumeIndicators()
        self.statistical_analysis = StatisticalAnalysis()
        self.ml_predictor = MLPredictor()
        self.data_provider = TechnicalDataProvider()
        self.signal_generator = SignalGenerator()

        self.logger.info("Advanced technical analyzer initialized with ML capabilities")

    async def analyze_symbol(self, symbol: str, period: str = "1mo") -> AdvancedAnalysis:
        """
        銘柄の高度技術分析

        Args:
            symbol: 分析対象銘柄
            period: 分析期間

        Returns:
            高度分析結果
        """
        try:
            # 価格データ取得
            df = await self.data_provider.get_price_data(symbol, period)

            if df is None or len(df) < 20:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None

            current_price = float(df['Close'].iloc[-1])
            price_change = float((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100)
            volume_ratio = float(
                df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
            )

            # 各種技術指標計算
            trend_indicators = await self.trend_indicators.calculate_trend_indicators(df)
            momentum_indicators = await self.momentum_indicators.calculate_momentum_indicators(df)
            volatility_indicators = await self.volatility_indicators.calculate_volatility_indicators(df)
            volume_indicators = await self.volume_indicators.calculate_volume_indicators(df)

            # 複合分析
            composite_score = self.data_provider.calculate_composite_score(
                trend_indicators, momentum_indicators, volatility_indicators
            )

            trend_strength = self.trend_indicators.calculate_trend_strength(trend_indicators)
            momentum_score = self.momentum_indicators.calculate_momentum_score(momentum_indicators)
            volatility_regime = self.volatility_indicators.determine_volatility_regime(
                volatility_indicators
            )

            # シグナル生成
            primary_signals, secondary_signals = await self.signal_generator.generate_signals(
                df, trend_indicators, momentum_indicators, volatility_indicators, volume_indicators
            )

            # 統計分析
            statistical_profile = self.statistical_analysis.perform_statistical_analysis(df)
            anomaly_score = self.statistical_analysis.calculate_anomaly_score(df)

            # 機械学習予測
            ml_prediction = await self.ml_predictor.generate_ml_prediction(symbol, df)

            # パターン認識
            pattern_recognition = self.statistical_analysis.perform_pattern_recognition(df)

            return AdvancedAnalysis(
                symbol=symbol,
                analysis_time=datetime.now(),
                current_price=current_price,
                price_change=price_change,
                volume_ratio=volume_ratio,
                trend_indicators=trend_indicators,
                momentum_indicators=momentum_indicators,
                volatility_indicators=volatility_indicators,
                volume_indicators=volume_indicators,
                composite_score=composite_score,
                trend_strength=trend_strength,
                momentum_score=momentum_score,
                volatility_regime=volatility_regime,
                primary_signals=primary_signals,
                secondary_signals=secondary_signals,
                statistical_profile=statistical_profile,
                anomaly_score=anomaly_score,
                ml_prediction=ml_prediction,
                pattern_recognition=pattern_recognition
            )

        except Exception as e:
            self.logger.error(f"Advanced analysis failed for {symbol}: {e}")
            return None


# テスト関数
async def test_advanced_technical_analyzer():
    """高度技術指標・分析手法拡張システムのテスト"""
    print("=== 高度技術指標・分析手法拡張システム テスト ===")

    analyzer = AdvancedTechnicalAnalyzer()

    # テスト銘柄
    test_symbols = ["7203", "4751", "6861"]

    print(f"\n[ 高度技術分析実行: {len(test_symbols)}銘柄 ]")

    for symbol in test_symbols:
        print(f"\n--- {symbol} 高度技術分析 ---")

        # 分析実行
        analysis = await analyzer.analyze_symbol(symbol, period="3mo")

        if analysis:
            print(f"現在価格: ¥{analysis.current_price:.2f} ({analysis.price_change:+.2f}%)")
            print(f"出来高倍率: {analysis.volume_ratio:.2f}倍")
            print(f"総合スコア: {analysis.composite_score:.1f}点")
            print(f"トレンド強度: {analysis.trend_strength:+.1f}")
            print(f"モメンタムスコア: {analysis.momentum_score:+.1f}")
            print(f"ボラティリティ局面: {analysis.volatility_regime}")
            print(f"異常度スコア: {analysis.anomaly_score:.1f}")

            # 主要指標表示
            print(f"\n  [ 主要指標 ]")
            trend = analysis.trend_indicators
            momentum = analysis.momentum_indicators

            if 'RSI_14' in momentum:
                print(f"  RSI(14): {momentum['RSI_14']:.1f}")
            if 'MACD' in trend:
                print(f"  MACD: {trend['MACD']:.4f}")
            if 'BB_Position' in analysis.volatility_indicators:
                print(f"  BB位置: {analysis.volatility_indicators['BB_Position']:.1f}%")

            # シグナル表示
            print(f"\n  [ プライマリシグナル: {len(analysis.primary_signals)}件 ]")
            for signal in analysis.primary_signals[:3]:
                print(f"  • {signal.indicator_name}: {signal.signal_type} (信頼度{signal.confidence:.0f}%)")

        else:
            print(f"  ❌ {symbol} の分析に失敗しました")

    print(f"\n=== 高度技術指標・分析手法拡張システム テスト完了 ===")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    asyncio.run(test_advanced_technical_analyzer())