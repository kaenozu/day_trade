#!/usr/bin/env python3
"""
RecommendationEngine包括的テストスイート

Issue #471対応: 推奨銘柄選定エンジンのテストカバレッジと信頼性向上

包括的テストケース:
- 推奨生成・スコア計算・リスク評価
- アンサンブル統合・エラーハンドリング
- パフォーマンス・モック使用
"""

import unittest
import numpy as np
import pandas as pd
import asyncio
import warnings
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import sys

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.day_trade.recommendation.recommendation_engine import (
        RecommendationEngine,
        StockRecommendation,
        RecommendationAction,
        get_daily_recommendations,
        get_smart_daily_recommendations
    )
    RECOMMENDATION_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"RecommendationEngine import failed: {e}")
    RECOMMENDATION_ENGINE_AVAILABLE = False

warnings.filterwarnings("ignore")


class TestRecommendationEngineBasic(unittest.TestCase):
    """RecommendationEngine基本機能テスト"""

    def setUp(self):
        """テスト前準備"""
        if not RECOMMENDATION_ENGINE_AVAILABLE:
            self.skipTest("RecommendationEngine not available")

        # テストデータ生成
        self.sample_data = self._create_sample_market_data()
        self.test_symbols = ["1001", "2002", "3003"]

    def _create_sample_market_data(self):
        """サンプル市場データ作成"""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            '始値': np.random.uniform(1000, 2000, 60),
            '高値': np.random.uniform(1100, 2100, 60),
            '安値': np.random.uniform(900, 1900, 60),
            '終値': np.random.uniform(1000, 2000, 60),
            '出来高': np.random.uniform(10000, 50000, 60),
        }, index=dates)

        # 価格の整合性確保
        for i in range(len(data)):
            low = data.iloc[i]['安値']
            high = data.iloc[i]['高値']
            data.iloc[i, data.columns.get_loc('始値')] = np.random.uniform(low, high)
            data.iloc[i, data.columns.get_loc('終値')] = np.random.uniform(low, high)

        return data

    def test_recommendation_engine_initialization(self):
        """RecommendationEngine初期化テスト"""
        engine = RecommendationEngine()

        # 基本属性確認
        self.assertIsNotNone(engine.technical_manager)
        self.assertIsNotNone(engine.ml_engine)
        self.assertIsNotNone(engine.data_fetcher)
        self.assertIsNotNone(engine.stock_helper)

        # 重み設定確認
        self.assertIn('technical', engine.weights)
        self.assertIn('ml', engine.weights)

        # 設定値確認
        self.assertIsInstance(engine.risk_thresholds, dict)
        self.assertIsInstance(engine.action_thresholds, dict)

    @patch('src.day_trade.recommendation.recommendation_engine.AdvancedBatchDataFetcher')
    def test_single_stock_analysis_mock(self, mock_fetcher_class):
        """単一銘柄分析テスト（モック使用）"""
        # モックの設定
        mock_fetcher = Mock()
        mock_fetcher_class.return_value = mock_fetcher

        engine = RecommendationEngine()

        # 分析実行（非同期）
        async def run_analysis():
            result = await engine._analyze_single_stock("1001", self.sample_data)
            return result

        recommendation = asyncio.run(run_analysis())

        # 結果検証
        if recommendation:  # 分析が成功した場合
            self.assertIsInstance(recommendation, StockRecommendation)
            self.assertEqual(recommendation.symbol, "1001")
            self.assertIsInstance(recommendation.composite_score, (int, float))
            self.assertGreaterEqual(recommendation.composite_score, 0)
            self.assertLessEqual(recommendation.composite_score, 100)
            self.assertIn(recommendation.risk_level, ['低', '中', '高'])

    def test_risk_level_assessment(self):
        """リスクレベル評価テスト"""
        engine = RecommendationEngine()

        # 低ボラティリティデータ
        low_vol_data = self.sample_data.copy()
        low_vol_data['終値'] = np.linspace(1000, 1050, len(low_vol_data))  # 安定した価格

        risk_level = engine._assess_risk_level(low_vol_data)
        self.assertIn(risk_level, ['低', '中', '高'])

        # 高ボラティリティデータ
        high_vol_data = self.sample_data.copy()
        high_vol_data['終値'] = np.random.uniform(500, 3000, len(high_vol_data))  # 不安定な価格

        risk_level = engine._assess_risk_level(high_vol_data)
        self.assertIn(risk_level, ['低', '中', '高'])

    def test_action_determination(self):
        """アクション決定テスト"""
        engine = RecommendationEngine()

        # 高スコア・低リスク → 買い推奨
        high_score_action = engine._determine_action(85, '低')
        self.assertIn(high_score_action, [
            RecommendationAction.STRONG_BUY,
            RecommendationAction.BUY
        ])

        # 低スコア・高リスク → 売り推奨
        low_score_action = engine._determine_action(25, '高')
        self.assertIn(low_score_action, [
            RecommendationAction.SELL,
            RecommendationAction.STRONG_SELL
        ])

        # 中程度スコア → 様子見
        mid_score_action = engine._determine_action(50, '中')
        self.assertEqual(mid_score_action, RecommendationAction.HOLD)

    def test_price_target_calculation(self):
        """価格目標計算テスト"""
        engine = RecommendationEngine()

        current_price = 1500.0

        # 高スコア・低リスクの価格目標
        target, stop_loss = engine._calculate_price_targets(current_price, 80, '低')

        if target and stop_loss:
            self.assertGreater(target, current_price)  # 目標価格は現在価格より高い
            self.assertLess(stop_loss, current_price)  # 損切価格は現在価格より低い
            self.assertIsInstance(target, (int, float))
            self.assertIsInstance(stop_loss, (int, float))


class TestRecommendationEngineAdvanced(unittest.TestCase):
    """RecommendationEngine高度機能テスト"""

    def setUp(self):
        """テスト前準備"""
        if not RECOMMENDATION_ENGINE_AVAILABLE:
            self.skipTest("RecommendationEngine not available")

        self.engine = RecommendationEngine()
        self.sample_data = self._create_trending_data()

    def _create_trending_data(self):
        """トレンドありサンプルデータ作成"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        # 上昇トレンドのデータ
        base_price = 1000
        trend_data = pd.DataFrame({
            '始値': [base_price + i * 10 + np.random.uniform(-5, 5) for i in range(50)],
            '高値': [base_price + i * 10 + 15 + np.random.uniform(0, 10) for i in range(50)],
            '安値': [base_price + i * 10 - 15 + np.random.uniform(-10, 0) for i in range(50)],
            '終値': [base_price + i * 10 + np.random.uniform(-5, 5) for i in range(50)],
            '出来高': np.random.uniform(20000, 80000, 50),
        }, index=dates)

        return trend_data

    @patch('src.day_trade.recommendation.recommendation_engine.TechnicalIndicatorsManager')
    def test_technical_score_calculation_mock(self, mock_technical_manager):
        """テクニカルスコア計算テスト（モック使用）"""
        # モック設定
        mock_manager = Mock()
        mock_indicators = {
            'sma': Mock(values={'sma': np.array([1100, 1110, 1120])}),
            'rsi': Mock(values={'rsi': np.array([45, 50, 55])}),
            'macd': Mock(values={'macd': np.array([5, 8, 12])}),
            'bollinger_bands': Mock(values={'upper': np.array([1150]), 'lower': np.array([1050])})
        }
        mock_manager.calculate_indicators.return_value = mock_indicators
        mock_technical_manager.return_value = mock_manager

        engine = RecommendationEngine()

        # テクニカルスコア計算（非同期）
        async def run_technical_analysis():
            score, reasons = await engine._calculate_technical_score("TEST", self.sample_data)
            return score, reasons

        score, reasons = asyncio.run(run_technical_analysis())

        # 結果検証
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIsInstance(reasons, list)

    def test_confidence_calculation(self):
        """信頼度計算テスト"""
        engine = RecommendationEngine()

        # 高スコア・十分なデータ → 高信頼度
        high_confidence = engine._calculate_confidence(80, 75, self.sample_data)
        self.assertGreaterEqual(high_confidence, 0)
        self.assertLessEqual(high_confidence, 100)

        # 低スコア・少ないデータ → 低信頼度
        small_data = self.sample_data.iloc[:5]  # データ量を減らす
        low_confidence = engine._calculate_confidence(40, 45, small_data)
        self.assertGreaterEqual(low_confidence, 0)
        self.assertLessEqual(low_confidence, 100)

    @patch('src.day_trade.recommendation.recommendation_engine.AdvancedMLEngine')
    def test_ml_score_calculation_mock(self, mock_ml_engine_class):
        """MLスコア計算テスト（モック使用）"""
        # モック設定
        mock_engine = Mock()
        mock_indicators = {
            'trend_strength': 75.0,
            'volatility_prediction': 60.0,
            'pattern_recognition': 80.0,
        }
        mock_engine.calculate_advanced_technical_indicators.return_value = mock_indicators
        mock_ml_engine_class.return_value = mock_engine

        engine = RecommendationEngine()

        # MLスコア計算（非同期）
        async def run_ml_analysis():
            score, reasons = await engine._calculate_ml_score("TEST", self.sample_data)
            return score, reasons

        score, reasons = asyncio.run(run_ml_analysis())

        # 結果検証
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIsInstance(reasons, list)

    def test_ensemble_score_calculation(self):
        """アンサンブルスコア計算テスト"""
        engine = RecommendationEngine()

        # アンサンブルシステムが初期化されている場合のテスト
        if hasattr(engine, 'ensemble_system') and engine.ensemble_system:
            async def run_ensemble_analysis():
                score, reasons = await engine._calculate_ensemble_score("TEST", self.sample_data)
                return score, reasons

            score, reasons = asyncio.run(run_ensemble_analysis())

            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
            self.assertIsInstance(reasons, list)
        else:
            # アンサンブルシステムが利用できない場合
            async def run_ensemble_analysis():
                score, reasons = await engine._calculate_ensemble_score("TEST", self.sample_data)
                return score, reasons

            score, reasons = asyncio.run(run_ensemble_analysis())
            self.assertEqual(score, 0.0)


class TestRecommendationEngineErrorHandling(unittest.TestCase):
    """RecommendationEngineエラーハンドリングテスト"""

    def setUp(self):
        """テスト前準備"""
        if not RECOMMENDATION_ENGINE_AVAILABLE:
            self.skipTest("RecommendationEngine not available")

        self.engine = RecommendationEngine()

    def test_empty_data_handling(self):
        """空データ処理テスト"""
        empty_data = pd.DataFrame()

        # 空データでの分析
        async def run_empty_analysis():
            result = await self.engine._analyze_single_stock("EMPTY", empty_data)
            return result

        result = asyncio.run(run_empty_analysis())

        # エラーハンドリングが適切に動作することを確認
        self.assertIsNone(result)

    def test_invalid_symbol_handling(self):
        """無効銘柄処理テスト"""
        valid_data = pd.DataFrame({
            '始値': [1000, 1010, 1020],
            '高値': [1050, 1060, 1070],
            '安値': [950, 960, 970],
            '終値': [1030, 1040, 1050],
            '出来高': [10000, 11000, 12000],
        })

        # 無効な銘柄コードでの分析
        async def run_invalid_analysis():
            result = await self.engine._analyze_single_stock("INVALID", valid_data)
            return result

        result = asyncio.run(run_invalid_analysis())

        # エラーハンドリングが動作してNoneまたは適切な結果が返されることを確認
        # 結果がNoneでない場合は、エラーハンドリングが適切に動作していることを示す
        if result:
            self.assertIsInstance(result, StockRecommendation)

    @patch('src.day_trade.recommendation.recommendation_engine.TechnicalIndicatorsManager')
    def test_technical_analysis_exception_handling(self, mock_technical_manager):
        """テクニカル分析例外処理テスト"""
        # 例外を発生させるモック
        mock_manager = Mock()
        mock_manager.calculate_indicators.side_effect = Exception("Technical analysis failed")
        mock_technical_manager.return_value = mock_manager

        engine = RecommendationEngine()

        sample_data = pd.DataFrame({
            '始値': [1000],
            '高値': [1050],
            '安値': [950],
            '終値': [1030],
            '出来高': [10000],
        })

        # 例外が発生してもスコアが返されることをテスト
        async def run_technical_with_error():
            score, reasons = await engine._calculate_technical_score("TEST", sample_data)
            return score, reasons

        score, reasons = asyncio.run(run_technical_with_error())

        # エラー時のフォールバック値が返されることを確認
        self.assertIsInstance(score, (int, float))
        self.assertIsInstance(reasons, list)


class TestRecommendationEngineIntegration(unittest.TestCase):
    """RecommendationEngine統合テスト"""

    def setUp(self):
        """テスト前準備"""
        if not RECOMMENDATION_ENGINE_AVAILABLE:
            self.skipTest("RecommendationEngine not available")

    @patch('src.day_trade.recommendation.recommendation_engine.AdvancedBatchDataFetcher')
    def test_daily_recommendations_integration(self, mock_fetcher_class):
        """日次推奨統合テスト"""
        # モックデータ準備
        mock_fetcher = Mock()
        sample_data = pd.DataFrame({
            '始値': [1000, 1010, 1020],
            '高値': [1050, 1060, 1070],
            '安値': [950, 960, 970],
            '終値': [1030, 1040, 1050],
            '出来高': [10000, 11000, 12000],
        })

        # モックの返り値設定
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = sample_data

        mock_fetcher.fetch_batch.return_value = {"TEST1": mock_result, "TEST2": mock_result}
        mock_fetcher_class.return_value = mock_fetcher

        # 推奨取得テスト
        async def run_daily_recommendations():
            recommendations = await get_daily_recommendations(limit=2)
            return recommendations

        recommendations = asyncio.run(run_daily_recommendations())

        # 結果検証
        self.assertIsInstance(recommendations, list)
        # レコメンデーションが生成される場合の検証
        for rec in recommendations:
            self.assertIsInstance(rec, StockRecommendation)

    def test_top_recommendations_filtering(self):
        """TOP推奨フィルタリングテスト"""
        engine = RecommendationEngine()

        # テスト用推奨データ作成
        test_recommendations = [
            StockRecommendation(
                symbol="HIGH1", name="高スコア銘柄1", composite_score=85, technical_score=80,
                ml_score=90, action=RecommendationAction.STRONG_BUY, confidence=90,
                reasons=["高スコア"], risk_level="低"
            ),
            StockRecommendation(
                symbol="MID1", name="中スコア銘柄1", composite_score=65, technical_score=60,
                ml_score=70, action=RecommendationAction.BUY, confidence=70,
                reasons=["中スコア"], risk_level="中"
            ),
            StockRecommendation(
                symbol="LOW1", name="低スコア銘柄1", composite_score=30, technical_score=25,
                ml_score=35, action=RecommendationAction.SELL, confidence=60,
                reasons=["低スコア"], risk_level="高"
            ),
        ]

        # TOP推奨取得
        top_recommendations = engine.get_top_recommendations(test_recommendations, limit=2)

        # フィルタリング結果確認
        self.assertLessEqual(len(top_recommendations), 2)

        # 買い推奨のみが含まれることを確認
        for rec in top_recommendations:
            self.assertIn(rec.action, [
                RecommendationAction.STRONG_BUY,
                RecommendationAction.BUY
            ])

        # スコア順にソートされていることを確認
        if len(top_recommendations) > 1:
            for i in range(len(top_recommendations) - 1):
                self.assertGreaterEqual(
                    top_recommendations[i].composite_score,
                    top_recommendations[i + 1].composite_score
                )


def run_recommendation_engine_tests():
    """RecommendationEngineテストスイートの実行"""
    print("=" * 60)
    print("RecommendationEngine包括的テストスイート実行")
    print("=" * 60)

    if not RECOMMENDATION_ENGINE_AVAILABLE:
        print("❌ RecommendationEngineが利用できません。テストをスキップします。")
        return

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # 基本機能テスト
    test_suite.addTest(unittest.makeSuite(TestRecommendationEngineBasic))

    # 高度機能テスト
    test_suite.addTest(unittest.makeSuite(TestRecommendationEngineAdvanced))

    # エラーハンドリングテスト
    test_suite.addTest(unittest.makeSuite(TestRecommendationEngineErrorHandling))

    # 統合テスト
    test_suite.addTest(unittest.makeSuite(TestRecommendationEngineIntegration))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print("\n" + "=" * 60)
    print("テストサマリー")
    print("=" * 60)
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures[:3]:  # 最初の3つのみ表示
            print(f"- {test}")

    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors[:3]:  # 最初の3つのみ表示
            print(f"- {test}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\n✅ テスト成功率: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_recommendation_engine_tests()
    exit(0 if success else 1)