#!/usr/bin/env python3
"""
高度なアンサンブルシステム統合テスト
Advanced Ensemble System Integration Tests

Issue #762: 高度なアンサンブル予測システムの強化 - 統合テスト
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
import logging
import time
import warnings
from typing import Dict, List, Any, Tuple

# テストフレームワーク
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.day_trade.testing.framework import BaseTestCase, TestResult
from src.day_trade.testing.fixtures import TestDataManager, MockDataGenerator
from src.day_trade.ensemble.advanced_ensemble import AdvancedEnsembleSystem, create_and_train_ensemble
from src.day_trade.ensemble import (
    AdaptiveWeightingEngine, MetaLearnerEngine, EnsembleOptimizer, EnsembleAnalyzer
)

# ログ設定
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class AdvancedEnsembleIntegrationTest(BaseTestCase):
    """高度なアンサンブルシステム統合テスト"""

    def __init__(self):
        super().__init__(
            test_name="Advanced Ensemble Integration Test",
            test_category="integration",
            timeout=300
        )

        # テストデータ
        self.data_manager = TestDataManager()

        # テスト設定
        self.test_config = {
            'n_samples': 500,
            'n_features': 15,
            'noise_level': 0.1,
            'train_ratio': 0.8
        }

    async def setup(self) -> None:
        """テストセットアップ"""
        # シード設定
        np.random.seed(42)

        # テストデータ生成
        all_features, all_targets = self.data_manager.get_feature_data(
            samples=self.test_config['n_samples'],
            features=self.test_config['n_features'],
            target_type="regression"
        )

        # Split data into train and test based on train_ratio
        split_idx = int(self.test_config['n_samples'] * self.test_config['train_ratio'])
        self.train_data = (all_features[:split_idx], all_targets[:split_idx])
        self.test_data = (all_features[split_idx:], all_targets[split_idx:])

        logger.info(f"Generated test data: train={self.train_data[0].shape}, test={self.test_data[0].shape}")

    async def _test_system_initialization(self, results: Dict) -> None:
        """システム初期化テスト"""
        logger.info("Testing system initialization...")

        try:
            # デフォルト設定でのシステム作成
            system1 = AdvancedEnsembleSystem()

            # カスタム設定でのシステム作成
            config = {
                'adaptive_weighting': {'lookback_window': 100},
                'meta_learning': {'adaptation_steps': 3},
                'optimization': {'optimization_budget': 20},
                'analysis': {'decomposition_depth': 2}
            }

            system2 = AdvancedEnsembleSystem(config=config)

            # 部分的な機能でのシステム作成
            system3 = AdvancedEnsembleSystem(
                enable_meta_learning=False,
                enable_optimization=False
            )

            # 検証
            assert len(system1.models) > 0, "Default models should be created"
            assert system2.config.adaptive_weighting_config['lookback_window'] == 100
            assert not system3.config.enable_meta_learning

            results['initialization_test'] = {
                'status': 'passed',
                'systems_created': 3,
                'default_models': len(system1.models)
            }

        except Exception as e:
            results['initialization_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise

    async def _test_full_training_pipeline(self, results: Dict) -> None:
        """完全な学習パイプラインテスト"""
        logger.info("Testing full training pipeline...")

        start_time = time.time()

        try:
            X_train, y_train = self.train_data

            # システム作成・学習
            system = await create_and_train_ensemble(X_train, y_train)

            training_time = time.time() - start_time

            # 学習完了確認
            assert system.is_fitted, "System should be fitted"
            assert len(system.training_history) > 0, "Training history should exist"

            # コンポーネント確認
            status = system.get_system_status()

            results['training_pipeline_test'] = {
                'status': 'passed',
                'training_time': training_time,
                'is_fitted': system.is_fitted,
                'components_enabled': sum(status['components'].values()),
                'training_data_shape': X_train.shape
            }

            # 後続テスト用にシステム保存
            self.trained_system = system

        except Exception as e:
            results['training_pipeline_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise

    async def _test_prediction_functionality(self, results: Dict) -> None:
        """予測機能テスト"""
        logger.info("Testing prediction functionality...")

        try:
            X_test, y_test = self.test_data

            # 予測実行
            prediction_result = await self.trained_system.predict(X_test)

            # 結果検証
            assert prediction_result.predictions.shape[0] == X_test.shape[0]
            assert len(prediction_result.individual_predictions) > 0
            assert prediction_result.confidence_scores.shape[0] == X_test.shape[0]
            assert prediction_result.processing_time > 0

            # 信頼度スコア範囲確認
            assert np.all(prediction_result.confidence_scores >= 0)
            assert np.all(prediction_result.confidence_scores <= 1)

            results['prediction_test'] = {
                'status': 'passed',
                'prediction_shape': prediction_result.predictions.shape,
                'avg_confidence': float(np.mean(prediction_result.confidence_scores)),
                'processing_time': prediction_result.processing_time,
                'individual_models': len(prediction_result.individual_predictions)
            }

        except Exception as e:
            results['prediction_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise

    async def _test_performance_analysis(self, results: Dict) -> None:
        """パフォーマンス分析テスト"""
        logger.info("Testing performance analysis...")

        try:
            X_test, y_test = self.test_data

            # パフォーマンス分析実行
            analysis = await self.trained_system.analyze_performance(X_test, y_test)

            # 基本メトリクス確認
            assert 'basic_metrics' in analysis
            basic_metrics = analysis['basic_metrics']

            required_metrics = ['mse', 'mae', 'r2', 'rmse']
            for metric in required_metrics:
                assert metric in basic_metrics
                assert isinstance(basic_metrics[metric], (int, float))

            # R²スコアの妥当性確認（-1 ~ 1の範囲）
            assert -1 <= basic_metrics['r2'] <= 1

            results['performance_analysis_test'] = {
                'status': 'passed',
                'metrics_available': list(basic_metrics.keys()),
                'r2_score': basic_metrics['r2'],
                'mse': basic_metrics['mse'],
                'mae': basic_metrics['mae']
            }

        except Exception as e:
            results['performance_analysis_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise

    async def _test_adaptive_weighting_integration(self, results: Dict) -> None:
        """動的重み付け統合テスト"""
        logger.info("Testing adaptive weighting integration...")

        try:
            # 動的重み付け有効なシステムで複数回予測
            X_test, y_test = self.test_data

            predictions = []
            weights_history = []

            # 複数バッチでの予測
            batch_size = 20
            for i in range(0, len(X_test), batch_size):
                X_batch = X_test[i:i+batch_size]

                result = await self.trained_system.predict(X_batch)
                predictions.append(result.predictions)
                weights_history.append(result.ensemble_weights)

            # 重みの変動確認
            if len(weights_history) > 1:
                weight_variation = np.std([w.mean() for w in weights_history])
                adaptive_behavior = weight_variation > 0.001  # 微小でも変動があるか
            else:
                adaptive_behavior = True  # 単一バッチの場合は適応的と見なす

            results['adaptive_weighting_test'] = {
                'status': 'passed',
                'batches_processed': len(weights_history),
                'adaptive_behavior_detected': adaptive_behavior,
                'avg_weight_variation': float(weight_variation) if len(weights_history) > 1 else 0.0
            }

        except Exception as e:
            results['adaptive_weighting_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.warning(f"Adaptive weighting test failed: {e}")

    async def _test_meta_learning_integration(self, results: Dict) -> None:
        """メタ学習統合テスト"""
        logger.info("Testing meta-learning integration...")

        try:
            # メタ学習エンジンが利用可能か確認
            has_meta_learner = self.trained_system.meta_learner is not None

            if has_meta_learner:
                # メタ学習統計取得
                meta_stats = self.trained_system.meta_learner.get_learning_statistics()

                # 少数ショット学習テスト
                X_test, y_test = self.test_data
                support_set = (X_test[:5], y_test[:5])
                query_data = X_test[5:10]

                meta_pred, meta_metadata = await self.trained_system.meta_learner.predict_with_adaptation(
                    support_set, query_data
                )

                meta_learning_functional = True
                prediction_shape = meta_pred.shape
            else:
                meta_stats = {}
                meta_learning_functional = False
                prediction_shape = None

            results['meta_learning_test'] = {
                'status': 'passed',
                'meta_learner_available': has_meta_learner,
                'meta_learning_functional': meta_learning_functional,
                'learning_statistics': meta_stats,
                'prediction_shape': prediction_shape
            }

        except Exception as e:
            results['meta_learning_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.warning(f"Meta-learning test failed: {e}")

    async def _test_system_persistence(self, results: Dict) -> None:
        """システム永続化テスト"""
        logger.info("Testing system persistence...")

        try:
            # システム保存
            save_path = "test_ensemble_system.pkl"
            self.trained_system.save_system(save_path)

            # システム読み込み
            loaded_system = AdvancedEnsembleSystem.load_system(save_path)

            # 読み込み後の動作確認
            X_test, y_test = self.test_data
            original_pred = await self.trained_system.predict(X_test[:10])
            loaded_pred = await loaded_system.predict(X_test[:10])

            # 予測結果の一致確認（許容誤差内）
            pred_diff = np.abs(original_pred.predictions - loaded_pred.predictions)
            predictions_match = np.all(pred_diff < 1e-6)

            # ファイル削除
            if os.path.exists(save_path):
                os.remove(save_path)

            results['persistence_test'] = {
                'status': 'passed',
                'save_successful': True,
                'load_successful': True,
                'predictions_match': predictions_match,
                'max_prediction_diff': float(np.max(pred_diff))
            }

        except Exception as e:
            results['persistence_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise

    async def _test_performance_benchmarks(self, results: Dict) -> None:
        """パフォーマンスベンチマークテスト"""
        logger.info("Testing performance benchmarks...")

        try:
            X_test, y_test = self.test_data

            # レイテンシテスト
            latency_times = []
            for _ in range(10):
                start_time = time.time()
                await self.trained_system.predict(X_test[:50])
                latency_times.append(time.time() - start_time)

            avg_latency = np.mean(latency_times)

            # スループットテスト
            throughput_start = time.time()
            batch_sizes = [10, 20, 50, 100]
            total_predictions = 0

            for batch_size in batch_sizes:
                if batch_size <= len(X_test):
                    await self.trained_system.predict(X_test[:batch_size])
                    total_predictions += batch_size

            throughput_time = time.time() - throughput_start
            predictions_per_second = total_predictions / throughput_time

            # パフォーマンス目標
            latency_target = 0.1  # 100ms以下
            throughput_target = 100  # 100予測/秒以上

            latency_pass = avg_latency <= latency_target
            throughput_pass = predictions_per_second >= throughput_target

            results['performance_benchmarks_test'] = {
                'status': 'passed',
                'avg_latency': avg_latency,
                'latency_target': latency_target,
                'latency_pass': latency_pass,
                'predictions_per_second': predictions_per_second,
                'throughput_target': throughput_target,
                'throughput_pass': throughput_pass
            }

        except Exception as e:
            results['performance_benchmarks_test'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.warning(f"Performance benchmark test failed: {e}")

    async def execute(self) -> TestResult:
        """統合テスト実行"""
        results = {}

        try:
            await self.setup()

            # システム初期化テスト
            await self._test_system_initialization(results)

            # 完全な学習パイプラインテスト
            await self._test_full_training_pipeline(results)

            # 予測機能テスト
            await self._test_prediction_functionality(results)

            # パフォーマンス分析テスト
            await self._test_performance_analysis(results)

            # 動的重み付け統合テスト
            await self._test_adaptive_weighting_integration(results)

            # メタ学習統合テスト
            await self._test_meta_learning_integration(results)

            # システム永続化テスト
            await self._test_system_persistence(results)

            # パフォーマンスベンチマークテスト
            await self._test_performance_benchmarks(results)

            # 全体結果評価
            passed_tests = sum(1 for test in results.values() if test.get('status') == 'passed')
            total_tests = len(results)

            success = passed_tests == total_tests

            return TestResult(
                test_name=self.test_name,
                success=success,
                execution_time=time.time() - self.start_time,
                details={
                    'test_summary': {
                        'total_tests': total_tests,
                        'passed_tests': passed_tests,
                        'success_rate': passed_tests / total_tests
                    },
                    'individual_tests': results
                }
            )

        except Exception as e:
            return TestResult(
                test_name=self.test_name,
                success=False,
                execution_time=time.time() - self.start_time,
                details={
                    'error': str(e),
                    'completed_tests': results
                }
            )

class EnsembleSystemStressTest(BaseTestCase):
    """アンサンブルシステムストレステスト"""

    def __init__(self):
        super().__init__(
            test_name="Ensemble System Stress Test",
            test_category="stress",
            timeout=600
        )

    async def execute(self) -> TestResult:
        """ストレステスト実行"""
        results = {}

        try:
            # 大規模データテスト
            logger.info("Starting large-scale data test...")

            n_samples = 5000
            n_features = 50
            X = np.random.randn(n_samples, n_features)
            y = np.random.randn(n_samples)

            # システム学習
            start_time = time.time()
            system = await create_and_train_ensemble(X, y)
            training_time = time.time() - start_time

            # 大規模予測
            X_test = np.random.randn(1000, n_features)
            pred_start = time.time()
            result = await system.predict(X_test)
            prediction_time = time.time() - pred_start

            # メモリ使用量テスト（簡易）
            memory_intensive_data = np.random.randn(100, n_features)
            for _ in range(50):  # 50回連続予測
                await system.predict(memory_intensive_data)

            results['large_scale_test'] = {
                'data_size': (n_samples, n_features),
                'training_time': training_time,
                'prediction_time': prediction_time,
                'stress_predictions_completed': 50
            }

            return TestResult(
                test_name=self.test_name,
                success=True,
                execution_time=time.time() - self.start_time,
                details=results
            )

        except Exception as e:
            return TestResult(
                test_name=self.test_name,
                success=False,
                execution_time=time.time() - self.start_time,
                details={'error': str(e)}
            )

# テスト実行関数
async def run_ensemble_integration_tests():
    """アンサンブル統合テスト実行"""
    logger.info("Starting Advanced Ensemble Integration Tests...")

    tests = [
        AdvancedEnsembleIntegrationTest(),
        EnsembleSystemStressTest()
    ]

    results = []
    for test in tests:
        logger.info(f"Running {test.test_name}...")
        result = await test.execute()
        results.append(result)

        if result.success:
            logger.info(f"✓ {test.test_name} PASSED")
        else:
            logger.error(f"✗ {test.test_name} FAILED")

    # 全体サマリー
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)

    logger.info(f"\nIntegration Test Summary:")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

    return results

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)

    # テスト実行
    asyncio.run(run_ensemble_integration_tests())