#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Analyzer Main - 高度技術指標・分析手法拡張システム（メインクラス）
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .types_and_enums import AdvancedAnalysis
from .data_provider import AdvancedDataProvider
from .trend_indicators import TrendIndicatorCalculator
from .momentum_indicators import MomentumIndicatorCalculator
from .volatility_volume_indicators import VolatilityVolumeCalculator
from .analysis_engine import AnalysisEngine
from .pattern_ml_features import PatternMLProcessor


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

        # 各種計算器初期化
        self.data_provider = AdvancedDataProvider()
        self.trend_calculator = TrendIndicatorCalculator()
        self.momentum_calculator = MomentumIndicatorCalculator()
        self.volatility_volume_calculator = VolatilityVolumeCalculator()
        self.analysis_engine = AnalysisEngine()
        self.pattern_ml_processor = PatternMLProcessor()

        self.logger.info("Advanced technical analyzer initialized with modular architecture")

    async def analyze_symbol(self, symbol: str, period: str = "1mo") -> Optional[AdvancedAnalysis]:
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
            volume_ratio = float(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1])

            # 各種技術指標計算
            trend_indicators = await self.trend_calculator.calculate_trend_indicators(df)
            momentum_indicators = await self.momentum_calculator.calculate_momentum_indicators(df)
            volatility_indicators = await self.volatility_volume_calculator.calculate_volatility_indicators(df)
            volume_indicators = await self.volatility_volume_calculator.calculate_volume_indicators(df)

            # 複合分析
            composite_score = self.analysis_engine.calculate_composite_score(
                trend_indicators, momentum_indicators, volatility_indicators
            )

            trend_strength = self.analysis_engine.calculate_trend_strength(trend_indicators)
            momentum_score = self.analysis_engine.calculate_momentum_score(momentum_indicators)
            volatility_regime = self.analysis_engine.determine_volatility_regime(volatility_indicators)

            # シグナル生成
            primary_signals, secondary_signals = await self.analysis_engine.generate_signals(
                df, trend_indicators, momentum_indicators, volatility_indicators, volume_indicators
            )

            # 統計分析
            statistical_profile = self.analysis_engine.perform_statistical_analysis(df)
            anomaly_score = self.analysis_engine.calculate_anomaly_score(df)

            # 機械学習予測
            ml_prediction = await self.pattern_ml_processor.generate_ml_prediction(symbol, df)

            # パターン認識
            pattern_recognition = self.pattern_ml_processor.perform_pattern_recognition(df)

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

            # 統計プロファイル
            stats = analysis.statistical_profile
            if stats:
                print(f"\n  [ 統計プロファイル ]")
                print(f"  年率リターン: {stats.get('mean_return', 0)*100:.1f}%")
                print(f"  ボラティリティ: {stats.get('volatility', 0)*100:.1f}%")
                if 'sharpe_ratio' in stats:
                    print(f"  シャープレシオ: {stats['sharpe_ratio']:.2f}")

            # 機械学習予測
            if analysis.ml_prediction:
                ml = analysis.ml_prediction
                print(f"\n  [ ML予測 ]")
                print(f"  方向性: {ml['direction']} (信頼度{ml['confidence']:.0f}%)")
                print(f"  期待リターン: {ml.get('expected_return', 0):.2f}%")
                print(f"  リスクレベル: {ml['risk_level']}")

            # パターン認識
            if analysis.pattern_recognition:
                pattern = analysis.pattern_recognition
                print(f"\n  [ パターン認識 ]")
                print(f"  検出パターン: {pattern.get('detected_pattern', 'N/A')}")
                print(f"  現在位置: {pattern.get('current_position', 'N/A')}")

                support_levels = pattern.get('support_levels', [])
                if support_levels:
                    print(f"  サポートレベル: {[f'¥{level:.0f}' for level in support_levels]}")

        else:
            print(f"  ❌ {symbol} の分析に失敗しました")

    print(f"\n=== 高度技術指標・分析手法拡張システム テスト完了 ===")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    asyncio.run(test_advanced_technical_analyzer())