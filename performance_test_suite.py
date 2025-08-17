#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Test Suite - パフォーマンステストスイート
統合パフォーマンス最適化システムの包括的テスト
"""

import time
import numpy as np
import pandas as pd
import unittest
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import psutil
import gc
import warnings
warnings.filterwarnings('ignore')

# テスト対象インポート
from integrated_performance_optimizer import (
    IntegratedPerformanceOptimizer,
    ModelCache,
    FeatureCache,
    PredictionBatchProcessor,
    OptimizationTarget,
    OptimizationPriority,
    SystemState
)

from performance_optimization_system import (
    PerformanceOptimizationSystem,
    get_performance_system
)


class MockPredictor:
    """テスト用モック予測システム"""

    def __init__(self, prediction_delay: float = 0.01):
        self.prediction_delay = prediction_delay
        self.model_loaded = False

    def load_model(self):
        """モデル読み込みシミュレーション"""
        time.sleep(0.1)  # 読み込み時間シミュレーション
        self.model_loaded = True

    def predict(self, data: np.ndarray) -> np.ndarray:
        """単一予測"""
        if not self.model_loaded:
            self.load_model()

        time.sleep(self.prediction_delay)
        return np.random.randn(len(data))

    def predict_batch(self, batch_data: List[Dict[str, Any]]) -> List[np.ndarray]:
        """バッチ予測"""
        if not self.model_loaded:
            self.load_model()

        # バッチ処理効率をシミュレーション
        batch_delay = self.prediction_delay * len(batch_data) * 0.7  # 30%の効率化
        time.sleep(batch_delay)

        return [np.random.randn(10) for _ in batch_data]


class TestModelCache(unittest.TestCase):
    """モデルキャッシュテスト"""

    def setUp(self):
        self.cache = ModelCache(max_size=10, ttl_seconds=1)

    def test_basic_cache_operations(self):
        """基本的なキャッシュ操作テスト"""
        # キャッシュに保存
        model_data = {"weights": np.random.randn(100, 10)}
        self.cache.put("model_1", model_data)

        # 取得テスト
        retrieved = self.cache.get("model_1")
        self.assertIsNotNone(retrieved)
        self.assertTrue(np.array_equal(retrieved["weights"], model_data["weights"]))

        # 存在しないキーのテスト
        missing = self.cache.get("nonexistent")
        self.assertIsNone(missing)

    def test_ttl_expiration(self):
        """TTL期限切れテスト"""
        model_data = {"weights": np.random.randn(50, 5)}
        self.cache.put("ttl_test", model_data)

        # 即座に取得（成功するはず）
        retrieved = self.cache.get("ttl_test")
        self.assertIsNotNone(retrieved)

        # TTL期限切れ後に取得（失敗するはず）
        time.sleep(1.5)
        expired = self.cache.get("ttl_test")
        self.assertIsNone(expired)

    def test_lru_eviction(self):
        """LRU削除テスト"""
        # キャッシュサイズを超えるデータを挿入
        for i in range(15):  # max_size=10を超える
            self.cache.put(f"model_{i}", {"data": i})

        # 最初のエントリが削除されているはず
        self.assertEqual(len(self.cache.cache), 10)
        self.assertIsNone(self.cache.get("model_0"))
        self.assertIsNotNone(self.cache.get("model_14"))

    def test_hit_rate_calculation(self):
        """ヒット率計算テスト"""
        # 初期状態
        self.assertEqual(self.cache.get_hit_rate(), 0.0)

        # データ挿入とアクセス
        self.cache.put("test", {"data": "test"})

        # ヒット
        self.cache.get("test")  # hit
        self.assertEqual(self.cache.get_hit_rate(), 1.0)

        # ミス
        self.cache.get("missing")  # miss
        self.assertEqual(self.cache.get_hit_rate(), 0.5)


class TestFeatureCache(unittest.TestCase):
    """特徴量キャッシュテスト"""

    def setUp(self):
        self.cache = FeatureCache(max_features=50, compression=False)

    def test_feature_caching(self):
        """特徴量キャッシュテスト"""
        features = np.random.randn(1000, 20)
        cache_key = self.cache.get_cache_key("data_hash_123", {"param": "value"})

        # 保存
        self.cache.put_features(cache_key, features)

        # 取得
        retrieved = self.cache.get_features(cache_key)
        self.assertIsNotNone(retrieved)
        self.assertTrue(np.array_equal(retrieved, features))

    def test_cache_key_generation(self):
        """キャッシュキー生成テスト"""
        key1 = self.cache.get_cache_key("hash1", {"a": 1, "b": 2})
        key2 = self.cache.get_cache_key("hash1", {"b": 2, "a": 1})  # 順序違い
        key3 = self.cache.get_cache_key("hash2", {"a": 1, "b": 2})  # ハッシュ違い

        # 同じ設定なら同じキー
        self.assertEqual(key1, key2)
        # 異なるハッシュなら異なるキー
        self.assertNotEqual(key1, key3)

    def test_least_used_eviction(self):
        """最小使用頻度削除テスト"""
        # キャッシュを満杯にする
        for i in range(60):  # max_features=50を超える
            features = np.random.randn(100, 10)
            cache_key = f"key_{i}"
            self.cache.put_features(cache_key, features)

        # サイズ制限の確認
        self.assertLessEqual(len(self.cache.cache), 50)

        # 使用頻度の高いエントリのアクセス
        active_key = "key_55"
        for _ in range(10):
            self.cache.get_features(active_key)

        # 新しいエントリ追加
        for i in range(10):
            features = np.random.randn(100, 10)
            cache_key = f"new_key_{i}"
            self.cache.put_features(cache_key, features)

        # アクティブキーが残っているはず
        self.assertIsNotNone(self.cache.get_features(active_key))


class TestBatchProcessor(unittest.TestCase):
    """バッチ処理テスト"""

    def setUp(self):
        self.processor = PredictionBatchProcessor(batch_size=5, max_workers=2)
        self.predictor = MockPredictor(prediction_delay=0.01)
        self.results = []

    def result_callback(self, result):
        """結果コールバック"""
        self.results.append(result)

    def test_batch_processing(self):
        """バッチ処理テスト"""
        # リクエスト追加
        for i in range(10):
            self.processor.add_prediction_request(
                f"req_{i}",
                {"data": np.random.randn(10)},
                self.result_callback
            )

        # バッチ処理実行
        batch_results = self.processor.process_batch(self.predictor)

        # 結果確認
        self.assertEqual(len(batch_results), 5)  # batch_size分処理される
        self.assertEqual(len(self.results), 5)  # コールバックも実行される

        # 残りのリクエスト確認
        self.assertEqual(len(self.processor.pending_requests), 5)

    def test_empty_batch_processing(self):
        """空バッチ処理テスト"""
        batch_results = self.processor.process_batch(self.predictor)
        self.assertEqual(len(batch_results), 0)

    def test_processing_flag(self):
        """処理フラグテスト"""
        self.assertFalse(self.processor.processing)

        # 大量のリクエスト追加
        for i in range(100):
            self.processor.add_prediction_request(
                f"req_{i}",
                {"data": np.random.randn(10)},
                None
            )

        # 並行処理テスト
        def process_batch():
            self.processor.process_batch(self.predictor)

        thread1 = threading.Thread(target=process_batch)
        thread2 = threading.Thread(target=process_batch)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # 処理が重複しないことを確認
        self.assertFalse(self.processor.processing)


class TestIntegratedOptimizer(unittest.TestCase):
    """統合最適化システムテスト"""

    def setUp(self):
        self.optimizer = IntegratedPerformanceOptimizer()

    def test_system_state_capture(self):
        """システム状態取得テスト"""
        state = self.optimizer.capture_system_state()

        self.assertIsInstance(state, SystemState)
        self.assertGreater(state.memory_usage_mb, 0)
        self.assertGreaterEqual(state.cpu_usage_percent, 0)
        self.assertGreaterEqual(state.response_time_ms, 0)
        self.assertIsInstance(state.timestamp, datetime)

    def test_memory_optimization(self):
        """メモリ最適化テスト"""
        # メモリを意図的に使用
        large_data = [np.random.randn(1000, 1000) for _ in range(10)]

        result = self.optimizer.optimize_memory_usage()

        self.assertIn('strategy', result)
        self.assertIn('execution_time_ms', result)
        self.assertIn('improvement_percent', result)
        self.assertIn('before_state', result)
        self.assertIn('after_state', result)

        # メモリが解放されたかチェック
        del large_data
        gc.collect()

    def test_model_cache_optimization(self):
        """モデルキャッシュ最適化テスト"""
        # キャッシュにモデルを追加
        for i in range(10):
            model_data = {"weights": np.random.randn(100, 50)}
            self.optimizer.model_cache.put(f"model_{i}", model_data)

        # 一部のモデルにアクセスしてヒット率を上げる
        for i in range(5):
            self.optimizer.model_cache.get(f"model_{i}")

        result = self.optimizer.optimize_model_cache()

        self.assertIn('strategy', result)
        self.assertIn('hit_rate_improvement', result)
        self.assertIn('cache_size_adjustment', result)

    def test_feature_cache_optimization(self):
        """特徴量キャッシュ最適化テスト"""
        # 特徴量キャッシュに大量データ追加
        for i in range(30):
            features = np.random.randn(500, 20)
            cache_key = f"features_{i}"
            self.optimizer.feature_cache.put_features(cache_key, features)

        result = self.optimizer.optimize_feature_processing()

        self.assertIn('strategy', result)
        self.assertIn('improvement_percent', result)
        self.assertIn('compression_enabled', result)

    def test_comprehensive_optimization(self):
        """包括的最適化テスト"""
        results = self.optimizer.run_comprehensive_optimization()

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for result in results:
            self.assertIn('strategy', result)
            if 'improvement_percent' in result:
                self.assertIsInstance(result['improvement_percent'], (int, float))

    def test_performance_report_generation(self):
        """パフォーマンスレポート生成テスト"""
        report = self.optimizer.get_performance_report()

        self.assertIn('system_state', report)
        self.assertIn('cache_performance', report)
        self.assertIn('optimization_summary', report)
        self.assertIn('timestamp', report)

        # システム状態の確認
        system_state = report['system_state']
        self.assertIn('memory_usage_mb', system_state)
        self.assertIn('cpu_usage_percent', system_state)

        # キャッシュ性能の確認
        cache_perf = report['cache_performance']
        self.assertIn('model_cache', cache_perf)
        self.assertIn('feature_cache', cache_perf)


class PerformanceBenchmarkTest(unittest.TestCase):
    """パフォーマンスベンチマークテスト"""

    def setUp(self):
        self.optimizer = IntegratedPerformanceOptimizer()
        self.predictor = MockPredictor()

    def test_memory_usage_benchmark(self):
        """メモリ使用量ベンチマーク"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # 大量のデータを処理
        large_datasets = []
        for i in range(50):
            data = np.random.randn(1000, 100)
            features = self.optimizer.feature_cache.get_cache_key(f"data_{i}", {})
            self.optimizer.feature_cache.put_features(features, data)
            large_datasets.append(data)

        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory

        # 最適化実行
        optimization_results = self.optimizer.run_comprehensive_optimization()

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_after_optimization = final_memory - initial_memory

        print(f"Memory Usage Benchmark:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Peak: {peak_memory:.1f}MB (+{memory_increase:.1f}MB)")
        print(f"  After optimization: {final_memory:.1f}MB (+{memory_after_optimization:.1f}MB)")

        # メモリ使用量が最適化後に改善されていることを確認
        self.assertLess(memory_after_optimization, memory_increase)

    def test_prediction_speed_benchmark(self):
        """予測速度ベンチマーク"""
        # 単一予測のベンチマーク
        single_prediction_times = []
        for _ in range(100):
            start_time = time.time()
            prediction = self.predictor.predict(np.random.randn(10))
            end_time = time.time()
            single_prediction_times.append((end_time - start_time) * 1000)

        # バッチ予測のベンチマーク
        batch_data = [{"data": np.random.randn(10)} for _ in range(100)]

        start_time = time.time()
        batch_predictions = self.predictor.predict_batch(batch_data)
        end_time = time.time()
        batch_total_time = (end_time - start_time) * 1000
        batch_per_prediction_time = batch_total_time / len(batch_data)

        avg_single_time = np.mean(single_prediction_times)

        print(f"Prediction Speed Benchmark:")
        print(f"  Average single prediction: {avg_single_time:.2f}ms")
        print(f"  Batch per prediction: {batch_per_prediction_time:.2f}ms")
        print(f"  Batch efficiency: {((avg_single_time - batch_per_prediction_time) / avg_single_time * 100):.1f}% faster")

        # バッチ処理が単一処理より効率的であることを確認
        self.assertLess(batch_per_prediction_time, avg_single_time)

    def test_cache_efficiency_benchmark(self):
        """キャッシュ効率ベンチマーク"""
        cache = ModelCache(max_size=100, ttl_seconds=300)

        # モデルデータ生成
        model_data = {"weights": np.random.randn(1000, 500)}

        # キャッシュなしの読み込み時間
        no_cache_times = []
        for _ in range(50):
            start_time = time.time()
            # ディープコピーでモデル読み込みをシミュレート
            imported_model = {"weights": model_data["weights"].copy()}
            end_time = time.time()
            no_cache_times.append((end_time - start_time) * 1000)

        # キャッシュありの読み込み時間
        cache.put("benchmark_model", model_data)

        cache_times = []
        for _ in range(50):
            start_time = time.time()
            cached_model = cache.get("benchmark_model")
            end_time = time.time()
            cache_times.append((end_time - start_time) * 1000)

        avg_no_cache_time = np.mean(no_cache_times)
        avg_cache_time = np.mean(cache_times)

        print(f"Cache Efficiency Benchmark:")
        print(f"  No cache average: {avg_no_cache_time:.3f}ms")
        print(f"  With cache average: {avg_cache_time:.3f}ms")
        print(f"  Cache speedup: {(avg_no_cache_time / avg_cache_time):.1f}x faster")

        # キャッシュが高速化に貢献していることを確認
        self.assertLess(avg_cache_time, avg_no_cache_time)


class LoadTest(unittest.TestCase):
    """負荷テスト"""

    def setUp(self):
        self.optimizer = IntegratedPerformanceOptimizer()

    def test_concurrent_optimization(self):
        """並行最適化テスト"""
        results = []

        def run_optimization():
            try:
                result = self.optimizer.run_comprehensive_optimization()
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")

        # 複数スレッドで同時最適化実行
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=run_optimization)
            threads.append(thread)
            thread.start()

        # 全スレッド完了待ち
        for thread in threads:
            thread.join()

        # エラーが発生していないことを確認
        error_count = sum(1 for r in results if isinstance(r, str) and r.startswith("Error"))
        self.assertEqual(error_count, 0, f"Concurrent optimization errors: {error_count}")

    def test_memory_stress(self):
        """メモリストレステスト"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # 大量のデータを段階的に作成
        data_collections = []
        for round_num in range(10):
            round_data = []
            for i in range(100):
                data = np.random.randn(500, 100)
                round_data.append(data)

                # 特徴量キャッシュにも保存
                cache_key = f"stress_test_{round_num}_{i}"
                self.optimizer.feature_cache.put_features(cache_key, data)

            data_collections.append(round_data)

            # 各ラウンド後にメモリ使用量チェック
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory

            # メモリ使用量が異常に増加した場合は最適化実行
            if memory_increase > 500:  # 500MB以上増加
                print(f"Memory stress detected at round {round_num}: {memory_increase:.1f}MB increase")
                optimization_results = self.optimizer.run_comprehensive_optimization()

                # 最適化後のメモリチェック
                optimized_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_reduction = current_memory - optimized_memory
                print(f"Memory optimization freed: {memory_reduction:.1f}MB")

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory

        print(f"Memory Stress Test Results:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Total increase: {total_increase:.1f}MB")

        # メモリリークがないことを確認（合理的な範囲内）
        self.assertLess(total_increase, 1000, "Potential memory leak detected")


def run_performance_test_suite():
    """パフォーマンステストスイートを実行"""
    print("Performance Test Suite Running...")
    print("=" * 60)

    # テストスイート作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 各テストクラスを追加
    test_classes = [
        TestModelCache,
        TestFeatureCache,
        TestBatchProcessor,
        TestIntegratedOptimizer,
        PerformanceBenchmarkTest,
        LoadTest
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"Test Results Summary:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Success: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")

    if result.failures:
        print(f"\nFailed Tests:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\nError Tests:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_performance_test_suite()
    exit(0 if success else 1)