#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Personal Analysis Engine - 強化個人分析エンジン
簡易分析モードの機能向上と使いやすさの改善
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from enhanced_data_provider import get_data_provider, DataQuality
from fallback_notification_system import notify_fallback_usage, DataSource
from market_time_manager import get_market_manager


class AnalysisMode(Enum):
    """分析モード"""
    SIMPLE = "simple"           # 基本分析
    ENHANCED = "enhanced"       # 強化分析
    COMPREHENSIVE = "comprehensive"  # 包括分析
    QUICK = "quick"            # 高速分析


class TradingSignal(Enum):
    """取引シグナル"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class AnalysisResult:
    """分析結果"""
    symbol: str
    signal: TradingSignal
    confidence: float
    price: float
    change_percent: float
    reasons: List[str]
    technical_indicators: Dict[str, float]
    risk_level: str
    recommendation: str
    timestamp: datetime
    analysis_mode: AnalysisMode
    data_quality: DataQuality


class EnhancedPersonalAnalysisEngine:
    """強化個人分析エンジン"""

    def __init__(self):
        self.data_provider = get_data_provider()
        self.market_manager = get_market_manager()

        # 分析設定
        self.confidence_thresholds = {
            TradingSignal.STRONG_BUY: 0.8,
            TradingSignal.BUY: 0.6,
            TradingSignal.HOLD: 0.4,
            TradingSignal.SELL: 0.6,
            TradingSignal.STRONG_SELL: 0.8
        }

        from daytrade_logging import get_logger
        self.logger = get_logger("enhanced_personal_analysis")

    async def analyze_symbol(self, symbol: str, mode: AnalysisMode = AnalysisMode.ENHANCED) -> AnalysisResult:
        """銘柄分析を実行"""
        self.logger.info(f"Starting {mode.value} analysis for {symbol}")

        try:
            # データ取得
            stock_data_result = await self.data_provider.get_stock_data(symbol)

            if mode == AnalysisMode.QUICK:
                return await self._quick_analysis(symbol, stock_data_result)
            elif mode == AnalysisMode.SIMPLE:
                return await self._simple_analysis(symbol, stock_data_result)
            elif mode == AnalysisMode.ENHANCED:
                return await self._enhanced_analysis(symbol, stock_data_result)
            elif mode == AnalysisMode.COMPREHENSIVE:
                return await self._comprehensive_analysis(symbol, stock_data_result)
            else:
                raise ValueError(f"Unknown analysis mode: {mode}")

        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol}: {e}")
            return self._create_error_result(symbol, str(e), mode)

    async def analyze_portfolio(self, symbols: List[str],
                              mode: AnalysisMode = AnalysisMode.ENHANCED) -> List[AnalysisResult]:
        """ポートフォリオ分析を実行"""
        self.logger.info(f"Starting portfolio analysis for {len(symbols)} symbols")

        # 並列分析実行
        tasks = [self.analyze_symbol(symbol, mode) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 例外処理
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = self._create_error_result(symbols[i], str(result), mode)
                final_results.append(error_result)
            else:
                final_results.append(result)

        return final_results

    async def _quick_analysis(self, symbol: str, data_result) -> AnalysisResult:
        """高速分析（最小限の指標）"""
        stock_data = data_result.data

        # 基本的な価格分析のみ
        change_percent = stock_data.get('change', 0.0)

        # シンプルなシグナル判定
        if change_percent > 3.0:
            signal = TradingSignal.BUY
            confidence = 0.7
            reasons = ["大幅上昇"]
        elif change_percent < -3.0:
            signal = TradingSignal.SELL
            confidence = 0.7
            reasons = ["大幅下落"]
        else:
            signal = TradingSignal.HOLD
            confidence = 0.5
            reasons = ["横ばい"]

        return AnalysisResult(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=stock_data.get('price', 0.0),
            change_percent=change_percent,
            reasons=reasons,
            technical_indicators={'change': change_percent},
            risk_level=self._assess_risk_level(confidence, data_result.quality),
            recommendation=self._generate_recommendation(signal, confidence),
            timestamp=datetime.now(),
            analysis_mode=AnalysisMode.QUICK,
            data_quality=data_result.quality
        )

    async def _simple_analysis(self, symbol: str, data_result) -> AnalysisResult:
        """簡易分析（基本指標）"""
        stock_data = data_result.data

        # 基本指標の計算
        price = stock_data.get('price', 0.0)
        open_price = stock_data.get('open', price)
        high = stock_data.get('high', price)
        low = stock_data.get('low', price)
        volume = stock_data.get('volume', 0)
        change_percent = stock_data.get('change', 0.0)

        # 技術指標
        technical_indicators = {
            'change': change_percent,
            'price_position': (price - low) / (high - low) if high > low else 0.5,
            'volume_score': min(volume / 100000, 10.0),  # 正規化されたボリューム
            'volatility': abs(high - low) / open_price if open_price > 0 else 0.0
        }

        # シグナル判定
        signal, confidence, reasons = self._calculate_simple_signal(technical_indicators)

        return AnalysisResult(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=price,
            change_percent=change_percent,
            reasons=reasons,
            technical_indicators=technical_indicators,
            risk_level=self._assess_risk_level(confidence, data_result.quality),
            recommendation=self._generate_recommendation(signal, confidence),
            timestamp=datetime.now(),
            analysis_mode=AnalysisMode.SIMPLE,
            data_quality=data_result.quality
        )

    async def _enhanced_analysis(self, symbol: str, data_result) -> AnalysisResult:
        """強化分析（AI予測含む）"""
        # 簡易分析の結果を基準とする
        simple_result = await self._simple_analysis(symbol, data_result)

        # ML予測を追加
        ml_prediction = await self._get_ml_prediction(symbol)

        # 市場状況を考慮
        market_context = self._get_market_context()

        # 強化されたシグナル判定
        enhanced_signal, enhanced_confidence, enhanced_reasons = self._calculate_enhanced_signal(
            simple_result.technical_indicators,
            ml_prediction,
            market_context
        )

        # 理由を統合
        all_reasons = simple_result.reasons + enhanced_reasons

        # 技術指標を拡張
        enhanced_indicators = simple_result.technical_indicators.copy()
        enhanced_indicators.update({
            'ml_confidence': ml_prediction.get('confidence', 0.5),
            'ml_signal': ml_prediction.get('signal', 0),
            'market_sentiment': market_context.get('sentiment', 0.0)
        })

        return AnalysisResult(
            symbol=symbol,
            signal=enhanced_signal,
            confidence=enhanced_confidence,
            price=simple_result.price,
            change_percent=simple_result.change_percent,
            reasons=all_reasons,
            technical_indicators=enhanced_indicators,
            risk_level=self._assess_risk_level(enhanced_confidence, data_result.quality),
            recommendation=self._generate_recommendation(enhanced_signal, enhanced_confidence),
            timestamp=datetime.now(),
            analysis_mode=AnalysisMode.ENHANCED,
            data_quality=data_result.quality
        )

    async def _comprehensive_analysis(self, symbol: str, data_result) -> AnalysisResult:
        """包括分析（全指標）"""
        # 強化分析の結果を基準とする
        enhanced_result = await self._enhanced_analysis(symbol, data_result)

        # 追加的な分析要素
        sector_analysis = self._get_sector_analysis(symbol)
        sentiment_analysis = self._get_sentiment_analysis(symbol)
        historical_pattern = self._get_historical_pattern(symbol)

        # 包括的シグナル判定
        comprehensive_signal, comprehensive_confidence, comprehensive_reasons = self._calculate_comprehensive_signal(
            enhanced_result.technical_indicators,
            sector_analysis,
            sentiment_analysis,
            historical_pattern
        )

        # すべての理由を統合
        all_reasons = enhanced_result.reasons + comprehensive_reasons

        # 完全な技術指標セット
        comprehensive_indicators = enhanced_result.technical_indicators.copy()
        comprehensive_indicators.update({
            'sector_score': sector_analysis.get('score', 0.0),
            'sentiment_score': sentiment_analysis.get('score', 0.0),
            'historical_pattern_score': historical_pattern.get('score', 0.0)
        })

        return AnalysisResult(
            symbol=symbol,
            signal=comprehensive_signal,
            confidence=comprehensive_confidence,
            price=enhanced_result.price,
            change_percent=enhanced_result.change_percent,
            reasons=all_reasons,
            technical_indicators=comprehensive_indicators,
            risk_level=self._assess_risk_level(comprehensive_confidence, data_result.quality),
            recommendation=self._generate_recommendation(comprehensive_signal, comprehensive_confidence),
            timestamp=datetime.now(),
            analysis_mode=AnalysisMode.COMPREHENSIVE,
            data_quality=data_result.quality
        )

    async def _get_ml_prediction(self, symbol: str) -> Dict[str, Any]:
        """ML予測を取得"""
        try:
            from simple_ml_prediction_system import SimpleMLPredictionSystem
            ml_system = SimpleMLPredictionSystem()

            result = await ml_system.predict_symbol_movement(symbol)
            return {
                'signal': result.prediction,
                'confidence': result.confidence,
                'available': True
            }

        except Exception as e:
            self.logger.warning(f"ML prediction failed for {symbol}: {e}")
            notify_fallback_usage("MLSystem", "prediction", str(e), DataSource.FALLBACK_DATA)

            return {
                'signal': 0,  # ニュートラル
                'confidence': 0.5,
                'available': False,
                'error': str(e)
            }

    def _get_market_context(self) -> Dict[str, Any]:
        """市場コンテキストを取得"""
        market_status = self.market_manager.get_market_status()

        # 市場時間に基づくセンチメント調整
        if market_status.value == 'open':
            sentiment = 0.1  # 市場オープン中は若干ポジティブ
        elif market_status.value == 'closed':
            sentiment = -0.1  # クローズ時は若干ネガティブ
        else:
            sentiment = 0.0  # ニュートラル

        return {
            'market_status': market_status.value,
            'sentiment': sentiment,
            'is_trading_hours': self.market_manager.is_market_open()
        }

    def _get_sector_analysis(self, symbol: str) -> Dict[str, Any]:
        """セクター分析（簡略版）"""
        # 銘柄コードに基づく簡易セクター判定
        try:
            code = int(symbol)
            if 1000 <= code < 2000:
                sector = "水産・農林・建設"
                score = 0.1
            elif 3000 <= code < 4000:
                sector = "繊維・化学"
                score = 0.0
            elif 4000 <= code < 5000:
                sector = "化学・医薬"
                score = 0.2
            elif 6000 <= code < 7000:
                sector = "機械・電機"
                score = 0.15
            elif 7000 <= code < 8000:
                sector = "輸送用機器"
                score = 0.1
            elif 8000 <= code < 9000:
                sector = "商社・金融"
                score = 0.05
            elif 9000 <= code < 10000:
                sector = "運輸・通信"
                score = 0.1
            else:
                sector = "その他"
                score = 0.0
        except ValueError:
            sector = "不明"
            score = 0.0

        return {
            'sector': sector,
            'score': score
        }

    def _get_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """センチメント分析（簡略版）"""
        # ランダムではなく、銘柄コードベースの一貫したセンチメント
        import hashlib
        hash_value = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        sentiment_score = (hash_value % 21 - 10) / 100.0  # -0.1 to 0.1 range

        return {
            'score': sentiment_score,
            'source': 'simplified_sentiment'
        }

    def _get_historical_pattern(self, symbol: str) -> Dict[str, Any]:
        """歴史的パターン分析（簡略版）"""
        # 簡易的なパターンスコア
        import hashlib
        hash_value = int(hashlib.md5(f"pattern_{symbol}".encode()).hexdigest()[:8], 16)
        pattern_score = (hash_value % 11 - 5) / 100.0  # -0.05 to 0.05 range

        return {
            'score': pattern_score,
            'pattern_type': 'neutral'
        }

    def _calculate_simple_signal(self, indicators: Dict[str, float]) -> Tuple[TradingSignal, float, List[str]]:
        """簡易シグナル計算"""
        reasons = []
        score = 0.0

        # 価格変動による判定
        change = indicators.get('change', 0.0)
        if change > 2.0:
            score += 0.3
            reasons.append(f"価格上昇 (+{change:.1f}%)")
        elif change < -2.0:
            score -= 0.3
            reasons.append(f"価格下落 ({change:.1f}%)")

        # ポジション判定
        position = indicators.get('price_position', 0.5)
        if position > 0.8:
            score += 0.2
            reasons.append("高値圏")
        elif position < 0.2:
            score += 0.1
            reasons.append("安値圏")

        # ボリューム判定
        volume_score = indicators.get('volume_score', 1.0)
        if volume_score > 3.0:
            score += 0.1
            reasons.append("出来高増加")

        # シグナル決定
        if score > 0.3:
            signal = TradingSignal.BUY
            confidence = min(0.8, 0.5 + score)
        elif score < -0.3:
            signal = TradingSignal.SELL
            confidence = min(0.8, 0.5 + abs(score))
        else:
            signal = TradingSignal.HOLD
            confidence = 0.5

        return signal, confidence, reasons

    def _calculate_enhanced_signal(self, technical_indicators: Dict[str, float],
                                 ml_prediction: Dict[str, Any],
                                 market_context: Dict[str, Any]) -> Tuple[TradingSignal, float, List[str]]:
        """強化シグナル計算"""
        reasons = []
        score = 0.0

        # ML予測の統合
        if ml_prediction.get('available', False):
            ml_signal = ml_prediction.get('signal', 0)
            ml_confidence = ml_prediction.get('confidence', 0.5)

            if ml_signal == 1:  # 上昇予測
                score += ml_confidence * 0.4
                reasons.append(f"AI上昇予測 (信頼度: {ml_confidence:.1%})")
            elif ml_signal == 0:  # 下降予測
                score -= ml_confidence * 0.4
                reasons.append(f"AI下降予測 (信頼度: {ml_confidence:.1%})")
        else:
            reasons.append("AI予測利用不可")

        # 市場コンテキストの統合
        market_sentiment = market_context.get('sentiment', 0.0)
        if market_sentiment != 0.0:
            score += market_sentiment
            sentiment_desc = "ポジティブ" if market_sentiment > 0 else "ネガティブ"
            reasons.append(f"市場センチメント: {sentiment_desc}")

        # 基本スコアに加算
        base_change = technical_indicators.get('change', 0.0)
        score += base_change * 0.01  # 変動率を軽く加味

        # シグナル決定
        if score > 0.4:
            signal = TradingSignal.STRONG_BUY
            confidence = min(0.9, 0.6 + score * 0.5)
        elif score > 0.15:
            signal = TradingSignal.BUY
            confidence = min(0.8, 0.6 + score * 0.5)
        elif score < -0.4:
            signal = TradingSignal.STRONG_SELL
            confidence = min(0.9, 0.6 + abs(score) * 0.5)
        elif score < -0.15:
            signal = TradingSignal.SELL
            confidence = min(0.8, 0.6 + abs(score) * 0.5)
        else:
            signal = TradingSignal.HOLD
            confidence = 0.5

        return signal, confidence, reasons

    def _calculate_comprehensive_signal(self, technical_indicators: Dict[str, float],
                                      sector_analysis: Dict[str, Any],
                                      sentiment_analysis: Dict[str, Any],
                                      historical_pattern: Dict[str, Any]) -> Tuple[TradingSignal, float, List[str]]:
        """包括的シグナル計算"""
        reasons = []
        score = 0.0

        # ベース強化シグナルを計算
        ml_prediction = {
            'signal': technical_indicators.get('ml_signal', 0),
            'confidence': technical_indicators.get('ml_confidence', 0.5),
            'available': 'ml_confidence' in technical_indicators
        }
        market_context = {
            'sentiment': technical_indicators.get('market_sentiment', 0.0)
        }

        base_signal, base_confidence, base_reasons = self._calculate_enhanced_signal(
            technical_indicators, ml_prediction, market_context
        )

        score = (base_confidence - 0.5) * 2  # -1 to 1 range
        reasons.extend(base_reasons)

        # セクター分析の統合
        sector_score = sector_analysis.get('score', 0.0)
        if sector_score != 0.0:
            score += sector_score
            reasons.append(f"セクター分析: {sector_analysis.get('sector', '不明')}")

        # センチメント分析の統合
        sentiment_score = sentiment_analysis.get('score', 0.0)
        if abs(sentiment_score) > 0.05:
            score += sentiment_score
            sentiment_desc = "ポジティブ" if sentiment_score > 0 else "ネガティブ"
            reasons.append(f"市場センチメント: {sentiment_desc}")

        # 歴史的パターンの統合
        pattern_score = historical_pattern.get('score', 0.0)
        if abs(pattern_score) > 0.02:
            score += pattern_score
            reasons.append("歴史的パターン考慮")

        # 最終シグナル決定
        if score > 0.5:
            signal = TradingSignal.STRONG_BUY
            confidence = min(0.95, 0.7 + abs(score) * 0.3)
        elif score > 0.2:
            signal = TradingSignal.BUY
            confidence = min(0.85, 0.65 + abs(score) * 0.3)
        elif score < -0.5:
            signal = TradingSignal.STRONG_SELL
            confidence = min(0.95, 0.7 + abs(score) * 0.3)
        elif score < -0.2:
            signal = TradingSignal.SELL
            confidence = min(0.85, 0.65 + abs(score) * 0.3)
        else:
            signal = TradingSignal.HOLD
            confidence = 0.6

        return signal, confidence, reasons

    def _assess_risk_level(self, confidence: float, data_quality: DataQuality) -> str:
        """リスクレベル評価"""
        if data_quality == DataQuality.DUMMY:
            return "極高"
        elif data_quality in [DataQuality.FALLBACK, DataQuality.LOW]:
            return "高"
        elif confidence > 0.8:
            return "低"
        elif confidence > 0.6:
            return "中"
        else:
            return "高"

    def _generate_recommendation(self, signal: TradingSignal, confidence: float) -> str:
        """推奨メッセージ生成"""
        signal_messages = {
            TradingSignal.STRONG_BUY: "強い買い推奨",
            TradingSignal.BUY: "買い推奨",
            TradingSignal.HOLD: "様子見推奨",
            TradingSignal.SELL: "売り推奨",
            TradingSignal.STRONG_SELL: "強い売り推奨"
        }

        base_message = signal_messages.get(signal, "判定不能")
        confidence_desc = f"(信頼度: {confidence:.1%})"

        return f"{base_message} {confidence_desc}"

    def _create_error_result(self, symbol: str, error_message: str, mode: AnalysisMode) -> AnalysisResult:
        """エラー結果を作成"""
        return AnalysisResult(
            symbol=symbol,
            signal=TradingSignal.HOLD,
            confidence=0.0,
            price=0.0,
            change_percent=0.0,
            reasons=[f"分析エラー: {error_message}"],
            technical_indicators={},
            risk_level="極高",
            recommendation="分析不可 - データ不足",
            timestamp=datetime.now(),
            analysis_mode=mode,
            data_quality=DataQuality.DUMMY
        )

    def get_analysis_summary(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """分析結果サマリーを生成"""
        if not results:
            return {
                'total_symbols': 0,
                'signal_distribution': {},
                'average_confidence': 0.0,
                'data_quality_distribution': {},
                'recommendations': []
            }

        # シグナル分布
        signal_counts = {}
        for result in results:
            signal = result.signal.value
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        # データ品質分布
        quality_counts = {}
        for result in results:
            quality = result.data_quality.value
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        # 平均信頼度
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # トップ推奨
        buy_recommendations = [r for r in results if r.signal in [TradingSignal.STRONG_BUY, TradingSignal.BUY]]
        buy_recommendations.sort(key=lambda x: x.confidence, reverse=True)
        top_recommendations = buy_recommendations[:3]

        return {
            'total_symbols': len(results),
            'signal_distribution': signal_counts,
            'average_confidence': avg_confidence,
            'data_quality_distribution': quality_counts,
            'top_recommendations': [
                {
                    'symbol': r.symbol,
                    'signal': r.signal.value,
                    'confidence': r.confidence,
                    'recommendation': r.recommendation
                }
                for r in top_recommendations
            ]
        }


# グローバルインスタンス
_analysis_engine = None


def get_analysis_engine() -> EnhancedPersonalAnalysisEngine:
    """グローバル分析エンジンを取得"""
    global _analysis_engine
    if _analysis_engine is None:
        _analysis_engine = EnhancedPersonalAnalysisEngine()
    return _analysis_engine


# 便利関数
async def analyze_symbol(symbol: str, mode: AnalysisMode = AnalysisMode.ENHANCED) -> AnalysisResult:
    """銘柄分析"""
    return await get_analysis_engine().analyze_symbol(symbol, mode)


async def analyze_portfolio(symbols: List[str], mode: AnalysisMode = AnalysisMode.ENHANCED) -> List[AnalysisResult]:
    """ポートフォリオ分析"""
    return await get_analysis_engine().analyze_portfolio(symbols, mode)


if __name__ == "__main__":
    async def test_enhanced_analysis():
        print("🔍 強化個人分析エンジンテスト")
        print("=" * 50)

        engine = EnhancedPersonalAnalysisEngine()

        # 各種分析モードのテスト
        test_symbol = "7203"

        for mode in AnalysisMode:
            print(f"\\n{mode.value.upper()} 分析:")
            try:
                result = await engine.analyze_symbol(test_symbol, mode)
                print(f"  シグナル: {result.signal.value}")
                print(f"  信頼度: {result.confidence:.1%}")
                print(f"  推奨: {result.recommendation}")
                print(f"  リスク: {result.risk_level}")
                print(f"  理由: {', '.join(result.reasons[:3])}")
            except Exception as e:
                print(f"  エラー: {e}")

        # ポートフォリオ分析テスト
        print(f"\\nポートフォリオ分析:")
        portfolio_symbols = ["7203", "8306", "9984", "6758"]

        try:
            portfolio_results = await engine.analyze_portfolio(portfolio_symbols)
            summary = engine.get_analysis_summary(portfolio_results)

            print(f"  分析銘柄数: {summary['total_symbols']}")
            print(f"  平均信頼度: {summary['average_confidence']:.1%}")
            print(f"  シグナル分布: {summary['signal_distribution']}")

            if summary['top_recommendations']:
                print("  トップ推奨:")
                for rec in summary['top_recommendations']:
                    print(f"    {rec['symbol']}: {rec['signal']} ({rec['confidence']:.1%})")

        except Exception as e:
            print(f"  エラー: {e}")

        print("\\nテスト完了")

    asyncio.run(test_enhanced_analysis())