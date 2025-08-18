#!/usr/bin/env python3
"""
自動更新最適化システム統合テスト
Issue #881: 自動更新の更新時間を考える
"""

import asyncio
import json
import logging
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from auto_update_optimizer import AutoUpdateOptimizer, MarketCondition, SymbolPriority
from symbol_expansion_system import SymbolExpansionSystem
from prediction_accuracy_enhancer import PredictionAccuracyEnhancer


class TestAutoUpdateOptimization(unittest.TestCase):
    """自動更新最適化統合テスト"""

    def setUp(self):
        """テスト前準備"""
        # 一時設定ファイル作成
        self.temp_config = self._create_temp_config()
        self.optimizer = AutoUpdateOptimizer(self.temp_config)
        self.expansion_system = SymbolExpansionSystem(self.temp_config)
        self.accuracy_enhancer = PredictionAccuracyEnhancer(self.temp_config)

    def tearDown(self):
        """テスト後クリーンアップ"""
        if hasattr(self, 'temp_config') and Path(self.temp_config).exists():
            Path(self.temp_config).unlink()

    def _create_temp_config(self) -> str:
        """一時設定ファイル作成"""
        config = {
            "watchlist": {
                "symbols": [
                    {
                        "code": "7203",
                        "name": "トヨタ自動車",
                        "group": "自動車",
                        "priority": "high",
                        "sector": "Transportation"
                    },
                    {
                        "code": "9984",
                        "name": "ソフトバンクグループ",
                        "group": "テクノロジー",
                        "priority": "high",
                        "sector": "Technology"
                    },
                    {
                        "code": "4563",
                        "name": "アンジェス",
                        "group": "遺伝子治療薬",
                        "priority": "high",
                        "sector": "BioTech"
                    }
                ],
                "update_interval_minutes": 0.5,
                "market_hours": {
                    "start": "09:00",
                    "end": "15:00",
                    "lunch_start": "11:30",
                    "lunch_end": "12:30"
                }
            }
        }

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()

        return temp_file.name

    def test_optimizer_initialization(self):
        """最適化システム初期化テスト"""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(len(self.optimizer.config['watchlist']['symbols']), 3)

        # スケジュール初期化
        self.optimizer.initialize_schedules()
        self.assertEqual(len(self.optimizer.schedules), 3)

        # 各銘柄のスケジュールチェック
        for symbol_data in self.optimizer.config['watchlist']['symbols']:
            symbol = symbol_data['code']
            self.assertIn(symbol, self.optimizer.schedules)

            schedule = self.optimizer.schedules[symbol]
            self.assertEqual(schedule.symbol, symbol)
            self.assertIsInstance(schedule.priority, SymbolPriority)
            self.assertGreater(schedule.interval_seconds, 0)

    def test_market_condition_detection(self):
        """市場状況検知テスト"""
        # 開場時間テスト
        with patch('auto_update_optimizer.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 9, 15)  # 9:15
            mock_datetime.strptime = datetime.strptime
            mock_datetime.combine = datetime.combine

            condition = self.optimizer.detect_market_condition()
            self.assertEqual(condition, MarketCondition.OPENING)

        # 昼休み時間テスト
        with patch('auto_update_optimizer.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 12, 0)  # 12:00
            mock_datetime.strptime = datetime.strptime
            mock_datetime.combine = datetime.combine

            condition = self.optimizer.detect_market_condition()
            self.assertEqual(condition, MarketCondition.LUNCH)

    def test_priority_determination(self):
        """優先度決定テスト"""
        # 高優先度銘柄（BioTech）
        biotech_symbol = {"code": "4563", "sector": "BioTech", "priority": "medium"}
        priority = self.optimizer._determine_priority(biotech_symbol)
        self.assertEqual(priority, SymbolPriority.CRITICAL)

        # 中優先度銘柄
        normal_symbol = {"code": "7203", "sector": "Transportation", "priority": "medium"}
        priority = self.optimizer._determine_priority(normal_symbol)
        self.assertEqual(priority, SymbolPriority.MEDIUM)

    def test_update_frequency_optimization(self):
        """更新頻度最適化テスト"""
        self.optimizer.initialize_schedules()

        # 市場状況変更
        original_condition = self.optimizer.market_condition
        self.optimizer.market_condition = MarketCondition.HIGH_VOLATILITY

        # 最適化実行
        asyncio.run(self.optimizer.optimize_update_frequency())

        # 間隔が短縮されていることを確認
        for schedule in self.optimizer.schedules.values():
            if schedule.priority == SymbolPriority.CRITICAL:
                self.assertLessEqual(schedule.interval_seconds, 10)  # 高ボラ時は10秒以下

    def test_expansion_system_initialization(self):
        """拡張システム初期化テスト"""
        self.assertIsNotNone(self.expansion_system)
        self.assertGreater(len(self.expansion_system.symbol_pool), 0)

        # 現在銘柄初期化
        self.expansion_system.initialize_current_symbols()
        self.assertEqual(len(self.expansion_system.current_symbols), 3)

    def test_expansion_candidate_analysis(self):
        """拡張候補分析テスト"""
        self.expansion_system.initialize_current_symbols()

        # 候補分析
        opportunities = self.expansion_system.analyze_expansion_opportunities()

        self.assertIsInstance(opportunities, list)
        # 機会スコア順でソートされていることを確認
        if len(opportunities) > 1:
            for i in range(len(opportunities) - 1):
                self.assertGreaterEqual(
                    opportunities[i].opportunity_score,
                    opportunities[i + 1].opportunity_score
                )

    def test_prediction_accuracy_initialization(self):
        """予測精度システム初期化テスト"""
        self.assertIsNotNone(self.accuracy_enhancer)

        # モデル初期化確認
        symbols = [s['code'] for s in self.accuracy_enhancer.config['watchlist']['symbols']]
        for symbol in symbols:
            self.assertIn(symbol, self.accuracy_enhancer.models)
            self.assertIn(symbol, self.accuracy_enhancer.performance_tracker)

    def test_feature_generation(self):
        """特徴量生成テスト"""
        # テスト用価格データ
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        price_data = pd.DataFrame({
            'close': np.random.uniform(1000, 1100, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates)),
            'high': np.random.uniform(1000, 1100, len(dates)),
            'low': np.random.uniform(950, 1050, len(dates))
        }, index=dates)

        features = self.accuracy_enhancer.generate_features("7203", price_data)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(price_data))

        # 基本特徴量チェック
        self.assertIn('price', features.columns)
        self.assertIn('volume', features.columns)
        self.assertIn('rsi', features.columns)
        self.assertIn('macd', features.columns)

    def test_ensemble_prediction(self):
        """アンサンブル予測テスト"""
        # テスト用データ
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        price_data = pd.DataFrame({
            'close': np.random.uniform(1000, 1100, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

        features = self.accuracy_enhancer.generate_features("7203", price_data)

        # 予測実行
        prediction = self.accuracy_enhancer.make_ensemble_prediction("7203", features)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.symbol, "7203")
        self.assertGreater(prediction.predicted_price, 0)
        self.assertGreaterEqual(prediction.confidence, 0)
        self.assertLessEqual(prediction.confidence, 1)

    def test_progress_reporting(self):
        """進捗レポートテスト"""
        self.optimizer.initialize_schedules()

        # プログレスレポート生成
        report = self.optimizer.generate_progress_report()

        self.assertIsInstance(report, dict)
        self.assertIn('timestamp', report)
        self.assertIn('market_condition', report)
        self.assertIn('total_symbols', report)
        self.assertIn('optimization_metrics', report)
        self.assertEqual(report['total_symbols'], 3)

    def test_integration_workflow(self):
        """統合ワークフローテスト"""
        # 1. 最適化システム初期化
        self.optimizer.initialize_schedules()
        self.assertEqual(len(self.optimizer.schedules), 3)

        # 2. 拡張システムで候補分析
        self.expansion_system.initialize_current_symbols()
        candidates = self.expansion_system.select_expansion_candidates(2)

        # 3. 予測システムでテスト予測
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        price_data = pd.DataFrame({
            'close': np.random.uniform(1000, 1100, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

        features = self.accuracy_enhancer.generate_features("7203", price_data)
        prediction = self.accuracy_enhancer.make_ensemble_prediction("7203", features)

        # 4. 結果確認
        self.assertIsNotNone(prediction)
        self.assertGreater(prediction.predicted_price, 0)

        # 5. レポート生成
        optimizer_report = self.optimizer.generate_progress_report()
        expansion_report = self.expansion_system.generate_expansion_report()
        accuracy_report = self.accuracy_enhancer.generate_accuracy_report()

        self.assertIsInstance(optimizer_report, dict)
        self.assertIsInstance(expansion_report, dict)
        self.assertIsInstance(accuracy_report, dict)

    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 存在しない設定ファイル
        invalid_optimizer = AutoUpdateOptimizer("nonexistent.json")
        self.assertEqual(invalid_optimizer.config, {})

        # 無効なデータでの予測
        empty_features = pd.DataFrame()
        try:
            prediction = self.accuracy_enhancer.make_ensemble_prediction("INVALID", empty_features)
            # エラーが発生しないことを確認（フォールバック動作）
            self.assertIsNotNone(prediction)
        except Exception as e:
            self.fail(f"予期しないエラー: {e}")

    def test_performance_metrics(self):
        """性能メトリクステスト"""
        # テストデータでモデル訓練
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        price_data = pd.DataFrame({
            'close': np.cumsum(np.random.randn(len(dates))) + 1000,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

        # モデル訓練
        self.accuracy_enhancer.train_models("7203", price_data)

        # 性能確認
        performance = self.accuracy_enhancer.performance_tracker["7203"]
        for model_type, perf in performance.items():
            if perf.prediction_count > 0:
                self.assertIsInstance(perf.mse, (int, float))
                self.assertIsInstance(perf.mae, (int, float))
                self.assertIsInstance(perf.r2, (int, float))


class TestPerformanceBenchmark(unittest.TestCase):
    """性能ベンチマークテスト"""

    def test_optimization_speed(self):
        """最適化速度テスト"""
        temp_config = self._create_large_config()
        optimizer = AutoUpdateOptimizer(temp_config)

        # 大量銘柄での初期化時間測定
        start_time = datetime.now()
        optimizer.initialize_schedules()
        initialization_time = (datetime.now() - start_time).total_seconds()

        # 10秒以内で完了することを確認
        self.assertLess(initialization_time, 10)

        # クリーンアップ
        Path(temp_config).unlink()

    def _create_large_config(self) -> str:
        """大量銘柄設定ファイル作成"""
        symbols = []
        for i in range(100):  # 100銘柄
            symbols.append({
                "code": f"TEST{i:04d}",
                "name": f"テスト銘柄{i}",
                "group": "テスト",
                "priority": "medium",
                "sector": "Technology"
            })

        config = {
            "watchlist": {
                "symbols": symbols,
                "update_interval_minutes": 0.5,
                "market_hours": {
                    "start": "09:00",
                    "end": "15:00",
                    "lunch_start": "11:30",
                    "lunch_end": "12:30"
                }
            }
        }

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()

        return temp_file.name


def run_integration_test():
    """統合テスト実行"""
    print("=== 自動更新最適化システム統合テスト ===")

    # テストスイート作成
    suite = unittest.TestSuite()

    # 基本機能テスト
    suite.addTest(TestAutoUpdateOptimization('test_optimizer_initialization'))
    suite.addTest(TestAutoUpdateOptimization('test_market_condition_detection'))
    suite.addTest(TestAutoUpdateOptimization('test_priority_determination'))
    suite.addTest(TestAutoUpdateOptimization('test_update_frequency_optimization'))

    # 拡張システムテスト
    suite.addTest(TestAutoUpdateOptimization('test_expansion_system_initialization'))
    suite.addTest(TestAutoUpdateOptimization('test_expansion_candidate_analysis'))

    # 予測精度テスト
    suite.addTest(TestAutoUpdateOptimization('test_prediction_accuracy_initialization'))
    suite.addTest(TestAutoUpdateOptimization('test_feature_generation'))
    suite.addTest(TestAutoUpdateOptimization('test_ensemble_prediction'))

    # 統合テスト
    suite.addTest(TestAutoUpdateOptimization('test_integration_workflow'))
    suite.addTest(TestAutoUpdateOptimization('test_progress_reporting'))
    suite.addTest(TestAutoUpdateOptimization('test_error_handling'))
    suite.addTest(TestAutoUpdateOptimization('test_performance_metrics'))

    # 性能テスト
    suite.addTest(TestPerformanceBenchmark('test_optimization_speed'))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 結果サマリー
    print(f"\n=== テスト結果サマリー ===")
    print(f"実行テスト数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")

    if result.failures:
        print(f"\n=== 失敗したテスト ===")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\n=== エラーが発生したテスト ===")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_test()
    if success:
        print("\n✅ 全テストが正常に完了しました")
    else:
        print("\n❌ 一部のテストが失敗しました")
        exit(1)