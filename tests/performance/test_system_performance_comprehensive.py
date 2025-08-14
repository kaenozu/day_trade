#!/usr/bin/env python3
"""
Issue #755 Final: システムパフォーマンス包括的テスト

Issue #487完全自動化システムの性能検証
- 高負荷・大規模データ処理性能テスト
- リアルタイム処理・レスポンス時間テスト
- メモリ・CPU効率性テスト
- 本番運用パフォーマンス要件検証
"""

import unittest
import pytest
import asyncio
import threading
import time
import concurrent.futures
import psutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
import gc

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    # パフォーマンステスト対象システム
    from src.day_trade.data_fetcher import DataFetcher
    from src.day_trade.automation.smart_symbol_selector import (
        SmartSymbolSelector,
        SymbolMetrics,
        SelectionCriteria
    )
    from src.day_trade.ml.ensemble_system import (
        EnsembleSystem,
        EnsembleConfig,
        EnsemblePredictions
    )
    from src.day_trade.automation.execution_scheduler import (
        ExecutionScheduler,
        ScheduledTask,
        ScheduleType
    )

except ImportError as e:
    print(f"パフォーマンステスト用インポートエラー: {e}")
    sys.exit(1)


class TestEnsembleSystemPerformance(unittest.TestCase):
    """EnsembleSystemパフォーマンステスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.ensemble = EnsembleSystem()
        self.performance_data = []
        self.logger = logging.getLogger(__name__)

    def test_high_frequency_prediction_performance(self):
        """高頻度予測パフォーマンステスト"""
        # 訓練データ準備
        n_samples = 1000
        n_features = 20
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randn(n_samples)

        # アンサンブル訓練
        train_start = time.time()
        self.ensemble.fit(X_train, y_train)
        train_time = time.time() - train_start

        self.logger.info(f"アンサンブル訓練時間: {train_time:.2f}秒")

        # 高頻度予測実行（100回）
        prediction_times = []
        memory_usage = []

        process = psutil.Process(os.getpid())

        for i in range(100):
            # メモリ使用量記録
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB

            # 単一予測実行
            X_single = np.random.randn(1, n_features)

            pred_start = time.time()
            prediction = self.ensemble.predict(X_single)
            pred_time = time.time() - pred_start

            prediction_times.append(pred_time)

            # 予測結果確認
            self.assertIsNotNone(prediction.final_predictions)
            self.assertEqual(len(prediction.final_predictions), 1)

            # 高頻度要件確認（500ms以下）
            self.assertLess(pred_time, 0.5,
                          f"予測時間 {pred_time:.3f}秒 が高頻度要件を超えています")

        # パフォーマンス統計
        avg_prediction_time = np.mean(prediction_times)
        max_prediction_time = np.max(prediction_times)
        min_prediction_time = np.min(prediction_times)
        std_prediction_time = np.std(prediction_times)

        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        memory_increase = max_memory - memory_usage[0]

        # パフォーマンス要件検証
        self.assertLess(avg_prediction_time, 0.2,
                       f"平均予測時間 {avg_prediction_time:.3f}秒 が目標を超えています")
        self.assertLess(max_prediction_time, 0.5,
                       f"最大予測時間 {max_prediction_time:.3f}秒 が要件を超えています")
        self.assertLess(std_prediction_time, 0.1,
                       f"予測時間のばらつき {std_prediction_time:.3f}秒 が大きすぎます")
        self.assertLess(memory_increase, 50,
                       f"メモリ増加量 {memory_increase:.1f}MB が許容値を超えています")

        # パフォーマンスデータ記録
        performance_result = {
            'test_name': 'high_frequency_prediction',
            'train_time': train_time,
            'avg_prediction_time': avg_prediction_time,
            'max_prediction_time': max_prediction_time,
            'min_prediction_time': min_prediction_time,
            'std_prediction_time': std_prediction_time,
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'memory_increase_mb': memory_increase,
            'predictions_count': 100
        }

        self.performance_data.append(performance_result)

        self.logger.info(f"高頻度予測パフォーマンス結果:")
        self.logger.info(f"  平均予測時間: {avg_prediction_time:.3f}秒")
        self.logger.info(f"  最大予測時間: {max_prediction_time:.3f}秒")
        self.logger.info(f"  メモリ使用量: {avg_memory:.1f}MB (最大: {max_memory:.1f}MB)")

    def test_large_dataset_processing_performance(self):
        """大規模データセット処理パフォーマンステスト"""
        # 大規模データセット準備
        large_samples = 5000
        large_features = 50

        self.logger.info(f"大規模データセット: {large_samples}サンプル × {large_features}特徴量")

        X_large = np.random.randn(large_samples, large_features)
        y_large = np.random.randn(large_samples)

        # メモリ使用量監視開始
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 大規模データ訓練
        train_start = time.time()
        self.ensemble.fit(X_large, y_large)
        train_time = time.time() - train_start

        train_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 大規模データ予測
        X_test_large = np.random.randn(1000, large_features)

        pred_start = time.time()
        predictions = self.ensemble.predict(X_test_large)
        pred_time = time.time() - pred_start

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # パフォーマンス要件検証
        self.assertLess(train_time, 300,  # 5分以下
                       f"大規模データ訓練時間 {train_time:.1f}秒 が要件を超えています")
        self.assertLess(pred_time, 30,  # 30秒以下
                       f"大規模データ予測時間 {pred_time:.1f}秒 が要件を超えています")

        memory_increase = final_memory - initial_memory
        self.assertLess(memory_increase, 1000,  # 1GB以下
                       f"メモリ使用量増加 {memory_increase:.1f}MB が要件を超えています")

        # 予測品質確認
        self.assertEqual(len(predictions.final_predictions), 1000)
        self.assertTrue(np.all(np.isfinite(predictions.final_predictions)))

        # パフォーマンスデータ記録
        large_performance = {
            'test_name': 'large_dataset_processing',
            'samples': large_samples,
            'features': large_features,
            'train_time': train_time,
            'prediction_time': pred_time,
            'initial_memory_mb': initial_memory,
            'train_memory_mb': train_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'predictions_count': 1000
        }

        self.performance_data.append(large_performance)

        self.logger.info(f"大規模データ処理パフォーマンス結果:")
        self.logger.info(f"  訓練時間: {train_time:.1f}秒")
        self.logger.info(f"  予測時間: {pred_time:.1f}秒")
        self.logger.info(f"  メモリ増加: {memory_increase:.1f}MB")

    def test_concurrent_prediction_performance(self):
        """並行予測パフォーマンステスト"""
        # アンサンブル訓練
        X_train = np.random.randn(500, 15)
        y_train = np.random.randn(500)
        self.ensemble.fit(X_train, y_train)

        # 並行予測実行
        concurrent_results = []

        def concurrent_prediction(thread_id: int, predictions_per_thread: int = 20):
            """並行予測関数"""
            thread_times = []

            for i in range(predictions_per_thread):
                X_test = np.random.randn(1, 15)

                start_time = time.time()
                prediction = self.ensemble.predict(X_test)
                pred_time = time.time() - start_time

                thread_times.append(pred_time)

            result = {
                'thread_id': thread_id,
                'predictions_count': predictions_per_thread,
                'avg_time': np.mean(thread_times),
                'max_time': np.max(thread_times),
                'total_time': sum(thread_times)
            }

            concurrent_results.append(result)
            return result

        # 並行実行（5スレッド）
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_prediction, i, 20) for i in range(5)]
            concurrent.futures.wait(futures)

        total_concurrent_time = time.time() - start_time

        # 並行処理結果検証
        self.assertEqual(len(concurrent_results), 5)

        total_predictions = sum(r['predictions_count'] for r in concurrent_results)
        overall_avg_time = np.mean([r['avg_time'] for r in concurrent_results])
        overall_max_time = np.max([r['max_time'] for r in concurrent_results])

        # 並行処理効率確認
        sequential_time_estimate = total_predictions * overall_avg_time
        efficiency = sequential_time_estimate / total_concurrent_time

        self.assertGreater(efficiency, 2.0,
                          f"並行処理効率 {efficiency:.1f}x が期待値を下回っています")
        self.assertLess(overall_max_time, 1.0,
                       f"並行処理中の最大予測時間 {overall_max_time:.3f}秒 が要件を超えています")

        # 並行パフォーマンスデータ記録
        concurrent_performance = {
            'test_name': 'concurrent_prediction',
            'threads': 5,
            'total_predictions': total_predictions,
            'total_time': total_concurrent_time,
            'avg_prediction_time': overall_avg_time,
            'max_prediction_time': overall_max_time,
            'efficiency_ratio': efficiency,
            'predictions_per_second': total_predictions / total_concurrent_time
        }

        self.performance_data.append(concurrent_performance)

        self.logger.info(f"並行予測パフォーマンス結果:")
        self.logger.info(f"  並行効率: {efficiency:.1f}x")
        self.logger.info(f"  予測レート: {total_predictions / total_concurrent_time:.1f} 予測/秒")


class TestSmartSymbolSelectorPerformance(unittest.TestCase):
    """SmartSymbolSelectorパフォーマンステスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.selector = SmartSymbolSelector()
        self.performance_data = []
        self.logger = logging.getLogger(__name__)

    @patch('yfinance.Ticker')
    def test_large_symbol_pool_performance(self, mock_ticker):
        """大規模銘柄プールパフォーマンステスト"""
        # yfinanceモック設定
        self._setup_yfinance_mock(mock_ticker)

        # 大規模銘柄プール作成（500銘柄）
        large_pool_size = 500
        large_symbol_pool = {}

        for i in range(large_pool_size):
            symbol_code = f"{1000 + i:04d}.T"
            large_symbol_pool[symbol_code] = f"テスト企業{i}"

        self.selector.symbol_pool = large_symbol_pool

        self.logger.info(f"大規模銘柄プール: {large_pool_size}銘柄")

        # メモリ使用量監視
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 大規模銘柄選択実行
        async def large_scale_selection():
            criteria = SelectionCriteria(
                target_symbols=20,
                min_market_cap=1000000000,
                min_avg_volume=100000,
                max_volatility=0.05
            )
            return await self.selector.select_optimal_symbols(criteria)

        start_time = time.time()
        selected_symbols = asyncio.run(large_scale_selection())
        selection_time = time.time() - start_time

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # パフォーマンス要件検証
        self.assertLess(selection_time, 180,  # 3分以下
                       f"大規模銘柄選択時間 {selection_time:.1f}秒 が要件を超えています")
        self.assertLess(memory_increase, 200,  # 200MB以下
                       f"メモリ使用量増加 {memory_increase:.1f}MB が要件を超えています")

        # 選択結果品質確認
        self.assertIsInstance(selected_symbols, list)
        self.assertLessEqual(len(selected_symbols), 20)
        self.assertGreater(len(selected_symbols), 0)

        # 処理効率計算
        symbols_per_second = large_pool_size / selection_time

        # パフォーマンスデータ記録
        large_pool_performance = {
            'test_name': 'large_symbol_pool',
            'pool_size': large_pool_size,
            'selection_time': selection_time,
            'selected_count': len(selected_symbols),
            'memory_increase_mb': memory_increase,
            'symbols_per_second': symbols_per_second,
            'efficiency_score': symbols_per_second / memory_increase if memory_increase > 0 else float('inf')
        }

        self.performance_data.append(large_pool_performance)

        self.logger.info(f"大規模銘柄選択パフォーマンス結果:")
        self.logger.info(f"  選択時間: {selection_time:.1f}秒")
        self.logger.info(f"  処理レート: {symbols_per_second:.1f} 銘柄/秒")
        self.logger.info(f"  メモリ効率: {large_pool_performance['efficiency_score']:.2f}")

    def _setup_yfinance_mock(self, mock_ticker):
        """yfinanceモック設定"""
        mock_instance = Mock()

        # ランダムな企業情報
        mock_instance.info = {
            'marketCap': np.random.randint(1000000000, 10000000000),
            'averageVolume': np.random.randint(100000, 5000000),
            'beta': np.random.uniform(0.5, 2.0)
        }

        # 履歴データ生成
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        mock_instance.history.return_value = pd.DataFrame({
            'Open': np.random.uniform(1000, 1100, 60),
            'High': np.random.uniform(1100, 1200, 60),
            'Low': np.random.uniform(900, 1000, 60),
            'Close': np.random.uniform(1000, 1100, 60),
            'Volume': np.random.randint(100000, 5000000, 60)
        }, index=dates)

        mock_ticker.return_value = mock_instance

    @patch('yfinance.Ticker')
    def test_concurrent_symbol_analysis_performance(self, mock_ticker):
        """並行銘柄分析パフォーマンステスト"""
        self._setup_yfinance_mock(mock_ticker)

        # 中規模銘柄プール（100銘柄）
        medium_pool_size = 100
        medium_symbol_pool = {
            f"{2000 + i:04d}.T": f"並行テスト企業{i}"
            for i in range(medium_pool_size)
        }

        self.selector.symbol_pool = medium_symbol_pool

        # 並行分析実行
        async def concurrent_symbol_analysis():
            criteria = SelectionCriteria(target_symbols=10)
            return await self.selector.select_optimal_symbols(criteria)

        # 複数回並行実行
        concurrent_results = []

        def run_concurrent_analysis(run_id: int):
            """並行分析実行"""
            start_time = time.time()
            result = asyncio.run(concurrent_symbol_analysis())
            execution_time = time.time() - start_time

            return {
                'run_id': run_id,
                'execution_time': execution_time,
                'selected_count': len(result),
                'symbols': result
            }

        # 並行実行（3プロセス）
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_concurrent_analysis, i) for i in range(3)]
            concurrent_results = [future.result() for future in futures]

        total_concurrent_time = time.time() - start_time

        # 並行処理結果検証
        self.assertEqual(len(concurrent_results), 3)

        avg_execution_time = np.mean([r['execution_time'] for r in concurrent_results])
        max_execution_time = np.max([r['execution_time'] for r in concurrent_results])

        # 並行処理効率確認
        sequential_estimate = avg_execution_time * 3
        efficiency = sequential_estimate / total_concurrent_time

        self.assertGreater(efficiency, 1.5,
                          f"並行処理効率 {efficiency:.1f}x が期待値を下回っています")
        self.assertLess(max_execution_time, 120,  # 2分以下
                       f"並行処理中の最大実行時間 {max_execution_time:.1f}秒 が要件を超えています")

        # 並行分析パフォーマンスデータ記録
        concurrent_analysis_performance = {
            'test_name': 'concurrent_symbol_analysis',
            'pool_size': medium_pool_size,
            'concurrent_runs': 3,
            'total_time': total_concurrent_time,
            'avg_execution_time': avg_execution_time,
            'max_execution_time': max_execution_time,
            'efficiency_ratio': efficiency
        }

        self.performance_data.append(concurrent_analysis_performance)

        self.logger.info(f"並行銘柄分析パフォーマンス結果:")
        self.logger.info(f"  並行効率: {efficiency:.1f}x")
        self.logger.info(f"  平均実行時間: {avg_execution_time:.1f}秒")


class TestExecutionSchedulerPerformance(unittest.TestCase):
    """ExecutionSchedulerパフォーマンステスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()
        self.performance_data = []
        self.logger = logging.getLogger(__name__)

    def tearDown(self):
        """テスト環境クリーンアップ"""
        if self.scheduler.is_running:
            self.scheduler.stop()

    def test_high_load_task_scheduling_performance(self):
        """高負荷タスクスケジューリングパフォーマンステスト"""
        # 大量タスク作成（200タスク）
        task_count = 200
        execution_log = []

        def performance_task(task_id: int):
            """パフォーマンステスト用タスク"""
            start_time = time.time()

            # 軽量な処理を模擬
            result = sum(range(1000))

            execution_time = time.time() - start_time

            execution_log.append({
                'task_id': task_id,
                'execution_time': execution_time,
                'result': result,
                'timestamp': datetime.now()
            })

            return result

        # 大量タスク追加
        task_creation_start = time.time()

        for i in range(task_count):
            task = ScheduledTask(
                task_id=f"perf_task_{i}",
                name=f"Performance Task {i}",
                schedule_type=ScheduleType.ON_DEMAND,
                target_function=lambda tid=i: performance_task(tid)
            )

            success = self.scheduler.add_task(task)
            self.assertTrue(success)

        task_creation_time = time.time() - task_creation_start

        # メモリ使用量監視
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 全タスク実行
        execution_start = time.time()

        for task_id in self.scheduler.tasks.keys():
            task = self.scheduler.tasks[task_id]
            self.scheduler._execute_task(task)

        execution_time = time.time() - execution_start
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # パフォーマンス要件検証
        self.assertLess(task_creation_time, 30,  # 30秒以下
                       f"タスク作成時間 {task_creation_time:.1f}秒 が要件を超えています")
        self.assertLess(execution_time, 60,  # 60秒以下
                       f"全タスク実行時間 {execution_time:.1f}秒 が要件を超えています")
        self.assertLess(memory_increase, 100,  # 100MB以下
                       f"メモリ使用量増加 {memory_increase:.1f}MB が要件を超えています")

        # 実行結果確認
        self.assertEqual(len(execution_log), task_count)

        avg_task_time = np.mean([log['execution_time'] for log in execution_log])
        max_task_time = np.max([log['execution_time'] for log in execution_log])

        # タスク実行効率確認
        tasks_per_second = task_count / execution_time

        self.assertGreater(tasks_per_second, 3,
                          f"タスク実行レート {tasks_per_second:.1f} タスク/秒 が要件を下回っています")

        # 高負荷スケジューリングパフォーマンスデータ記録
        high_load_performance = {
            'test_name': 'high_load_task_scheduling',
            'task_count': task_count,
            'task_creation_time': task_creation_time,
            'execution_time': execution_time,
            'avg_task_time': avg_task_time,
            'max_task_time': max_task_time,
            'memory_increase_mb': memory_increase,
            'tasks_per_second': tasks_per_second
        }

        self.performance_data.append(high_load_performance)

        self.logger.info(f"高負荷スケジューリングパフォーマンス結果:")
        self.logger.info(f"  タスク実行レート: {tasks_per_second:.1f} タスク/秒")
        self.logger.info(f"  平均タスク時間: {avg_task_time:.3f}秒")
        self.logger.info(f"  メモリ効率: {memory_increase:.1f}MB")

    def test_concurrent_scheduler_performance(self):
        """並行スケジューラパフォーマンステスト"""
        # 複数スケジューラ同時実行
        scheduler_count = 5
        schedulers = []
        concurrent_results = []

        def create_and_run_scheduler(scheduler_id: int):
            """スケジューラ作成・実行"""
            local_scheduler = ExecutionScheduler()
            execution_log = []

            def concurrent_task(task_id: int):
                execution_log.append({
                    'scheduler_id': scheduler_id,
                    'task_id': task_id,
                    'timestamp': datetime.now()
                })
                return f"scheduler_{scheduler_id}_task_{task_id}"

            # 各スケジューラに10タスク追加
            for i in range(10):
                task = ScheduledTask(
                    task_id=f"concurrent_task_{i}",
                    name=f"Concurrent Task {i}",
                    schedule_type=ScheduleType.ON_DEMAND,
                    target_function=lambda tid=i: concurrent_task(tid)
                )
                local_scheduler.add_task(task)

            # 全タスク実行
            start_time = time.time()

            for task_id in local_scheduler.tasks.keys():
                task = local_scheduler.tasks[task_id]
                local_scheduler._execute_task(task)

            execution_time = time.time() - start_time

            result = {
                'scheduler_id': scheduler_id,
                'tasks_executed': len(execution_log),
                'execution_time': execution_time,
                'tasks_per_second': len(execution_log) / execution_time
            }

            concurrent_results.append(result)
            return result

        # 並行スケジューラ実行
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=scheduler_count) as executor:
            futures = [executor.submit(create_and_run_scheduler, i) for i in range(scheduler_count)]
            concurrent.futures.wait(futures)

        total_concurrent_time = time.time() - start_time

        # 並行処理結果検証
        self.assertEqual(len(concurrent_results), scheduler_count)

        total_tasks = sum(r['tasks_executed'] for r in concurrent_results)
        avg_execution_time = np.mean([r['execution_time'] for r in concurrent_results])
        total_tasks_per_second = total_tasks / total_concurrent_time

        # 並行処理効率確認
        sequential_estimate = avg_execution_time * scheduler_count
        efficiency = sequential_estimate / total_concurrent_time

        self.assertGreater(efficiency, 2.0,
                          f"並行スケジューラ効率 {efficiency:.1f}x が期待値を下回っています")
        self.assertGreater(total_tasks_per_second, 15,
                          f"総タスク実行レート {total_tasks_per_second:.1f} が要件を下回っています")

        # 並行スケジューラパフォーマンスデータ記録
        concurrent_scheduler_performance = {
            'test_name': 'concurrent_scheduler',
            'scheduler_count': scheduler_count,
            'total_tasks': total_tasks,
            'total_time': total_concurrent_time,
            'avg_execution_time': avg_execution_time,
            'efficiency_ratio': efficiency,
            'total_tasks_per_second': total_tasks_per_second
        }

        self.performance_data.append(concurrent_scheduler_performance)

        self.logger.info(f"並行スケジューラパフォーマンス結果:")
        self.logger.info(f"  並行効率: {efficiency:.1f}x")
        self.logger.info(f"  総タスクレート: {total_tasks_per_second:.1f} タスク/秒")


class TestSystemResourceEfficiency(unittest.TestCase):
    """システムリソース効率性テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.resource_data = []
        self.logger = logging.getLogger(__name__)

    def test_memory_efficiency_under_load(self):
        """負荷下でのメモリ効率性テスト"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 複数システム並行動作
        components = []
        memory_snapshots = []

        # 10個のEnsembleSystem並行作成・実行
        for i in range(10):
            ensemble = EnsembleSystem()

            # 訓練データ準備
            X_train = np.random.randn(200, 15)
            y_train = np.random.randn(200)

            # 訓練実行
            ensemble.fit(X_train, y_train)

            components.append(ensemble)

            # メモリ使用量記録
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_snapshots.append({
                'component_index': i,
                'memory_mb': current_memory,
                'memory_increase': current_memory - initial_memory
            })

        # 最終メモリ使用量
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = peak_memory - initial_memory

        # 各システムで予測実行
        prediction_times = []

        for i, ensemble in enumerate(components):
            X_test = np.random.randn(5, 15)

            start_time = time.time()
            prediction = ensemble.predict(X_test)
            pred_time = time.time() - start_time

            prediction_times.append(pred_time)

        # ガベージコレクション実行
        gc.collect()
        after_gc_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_recovered = peak_memory - after_gc_memory

        # メモリ効率性要件検証
        self.assertLess(total_memory_increase, 500,  # 500MB以下
                       f"総メモリ増加量 {total_memory_increase:.1f}MB が要件を超えています")

        avg_memory_per_component = total_memory_increase / len(components)
        self.assertLess(avg_memory_per_component, 50,  # 50MB以下/コンポーネント
                       f"コンポーネント当たりメモリ {avg_memory_per_component:.1f}MB が要件を超えています")

        # 予測性能維持確認
        avg_prediction_time = np.mean(prediction_times)
        self.assertLess(avg_prediction_time, 0.5,
                       f"負荷下での予測時間 {avg_prediction_time:.3f}秒 が要件を超えています")

        # メモリ効率性データ記録
        memory_efficiency = {
            'test_name': 'memory_efficiency_under_load',
            'components_count': len(components),
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'total_memory_increase_mb': total_memory_increase,
            'avg_memory_per_component_mb': avg_memory_per_component,
            'memory_recovered_mb': memory_recovered,
            'avg_prediction_time': avg_prediction_time,
            'memory_efficiency_score': len(components) / total_memory_increase if total_memory_increase > 0 else float('inf')
        }

        self.resource_data.append(memory_efficiency)

        self.logger.info(f"メモリ効率性テスト結果:")
        self.logger.info(f"  総メモリ増加: {total_memory_increase:.1f}MB")
        self.logger.info(f"  コンポーネント効率: {avg_memory_per_component:.1f}MB/コンポーネント")
        self.logger.info(f"  メモリ回復: {memory_recovered:.1f}MB")

    def test_cpu_utilization_efficiency(self):
        """CPU使用率効率性テスト"""
        # CPU使用率監視開始
        cpu_usage_samples = []

        def monitor_cpu_usage():
            """CPU使用率監視"""
            for _ in range(20):  # 10秒間監視
                cpu_percent = psutil.cpu_percent(interval=0.5)
                cpu_usage_samples.append(cpu_percent)

        # CPU監視スレッド開始
        monitor_thread = threading.Thread(target=monitor_cpu_usage)
        monitor_thread.start()

        # CPU集約的処理実行
        ensemble = EnsembleSystem()

        # 複数回の訓練・予測実行
        cpu_intensive_start = time.time()

        for i in range(5):
            # 大きめのデータセットで訓練
            X_train = np.random.randn(1000, 20)
            y_train = np.random.randn(1000)

            ensemble.fit(X_train, y_train)

            # 予測実行
            X_test = np.random.randn(100, 20)
            predictions = ensemble.predict(X_test)

            self.assertIsNotNone(predictions.final_predictions)

        cpu_intensive_time = time.time() - cpu_intensive_start

        # CPU監視終了
        monitor_thread.join()

        # CPU使用率分析
        avg_cpu_usage = np.mean(cpu_usage_samples)
        max_cpu_usage = np.max(cpu_usage_samples)
        cpu_efficiency = len(cpu_usage_samples) / cpu_intensive_time  # サンプル/秒

        # CPU効率性要件検証
        self.assertLess(max_cpu_usage, 90,  # 90%以下
                       f"最大CPU使用率 {max_cpu_usage:.1f}% が要件を超えています")
        self.assertLess(avg_cpu_usage, 70,  # 70%以下
                       f"平均CPU使用率 {avg_cpu_usage:.1f}% が要件を超えています")

        # CPU効率性データ記録
        cpu_efficiency_data = {
            'test_name': 'cpu_utilization_efficiency',
            'monitoring_duration': len(cpu_usage_samples) * 0.5,
            'intensive_processing_time': cpu_intensive_time,
            'avg_cpu_usage': avg_cpu_usage,
            'max_cpu_usage': max_cpu_usage,
            'cpu_usage_samples': len(cpu_usage_samples),
            'processing_efficiency': 5 / cpu_intensive_time  # 処理回数/秒
        }

        self.resource_data.append(cpu_efficiency_data)

        self.logger.info(f"CPU効率性テスト結果:")
        self.logger.info(f"  平均CPU使用率: {avg_cpu_usage:.1f}%")
        self.logger.info(f"  最大CPU使用率: {max_cpu_usage:.1f}%")
        self.logger.info(f"  処理効率: {cpu_efficiency_data['processing_efficiency']:.2f} 処理/秒")


if __name__ == '__main__':
    # テストスイート設定
    test_suite = unittest.TestSuite()

    # EnsembleSystemパフォーマンステスト
    test_suite.addTest(unittest.makeSuite(TestEnsembleSystemPerformance))

    # SmartSymbolSelectorパフォーマンステスト
    test_suite.addTest(unittest.makeSuite(TestSmartSymbolSelectorPerformance))

    # ExecutionSchedulerパフォーマンステスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerPerformance))

    # システムリソース効率性テスト
    test_suite.addTest(unittest.makeSuite(TestSystemResourceEfficiency))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print(f"\n{'='*80}")
    print(f"システムパフォーマンス包括的テスト完了")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗数: {len(result.failures)}")
    print(f"エラー数: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*80}")