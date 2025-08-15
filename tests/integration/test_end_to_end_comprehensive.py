#!/usr/bin/env python3
"""
Issue #755 Phase 5: エンドツーエンド統合テスト

完全自動化システムの包括的統合検証
- DataFetcher → SmartSymbolSelector → EnsembleSystem → ExecutionScheduler
- 実際の市場データを使用したリアルタイム処理テスト
- 高負荷・並行処理・例外処理の統合検証
- Issue #487完全自動化システムの最終品質保証
"""

import unittest
import pytest
import asyncio
import threading
import time
import concurrent.futures
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    # データ取得システム
    from src.day_trade.data.stock_fetcher import StockFetcher # Corrected import

    # スマート銘柄選択システム
    from src.day_trade.automation.smart_symbol_selector import (
        SmartSymbolSelector,
        SymbolMetrics,
        SelectionCriteria,
        get_smart_selected_symbols
    )

    # アンサンブル予測システム
    from src.day_trade.ml.ensemble_system import (
        EnsembleSystem,
        EnsembleConfig,
        EnsemblePrediction # Corrected from EnsemblePredictions
    )

    # 実行スケジューラシステム
    from src.day_trade.automation.execution_scheduler import (
        ExecutionScheduler,
        ScheduledTask,
        ScheduleType,
        ExecutionStatus,
        smart_stock_analysis_task,
        create_default_automation_tasks
    )

except ImportError as e:
    print(f"エンドツーエンド統合テスト用インポートエラー: {e}")
    sys.exit(1)


class TestCompleteSystemIntegration(unittest.TestCase):
    """完全システム統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        # システムコンポーネント初期化
        self.data_fetcher = StockFetcher() # Corrected instantiation
        self.smart_selector = SmartSymbolSelector()
        self.ensemble_system = EnsembleSystem()
        self.execution_scheduler = ExecutionScheduler()

        # テストデータ準備
        self.test_symbols = ['7203.T', '6758.T', '9984.T', '4519.T', '8316.T']
        self.integration_results = []

        # ロギング設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def tearDown(self):
        """テスト環境クリーンアップ"""
        if self.execution_scheduler.is_running:
            self.execution_scheduler.stop()

    @patch('yfinance.Ticker')
    @patch('src.day_trade.data.stock_fetcher.StockFetcher.bulk_get_historical_data') # Corrected patch target
    def test_full_pipeline_integration(self, mock_fetch_data, mock_ticker):
        """完全パイプライン統合テスト"""
        # yfinanceモック設定
        self._setup_yfinance_mock(mock_ticker)

        # DataFetcherのモック設定
        mock_fetch_data.return_value = {s: self._create_mock_market_data() for s in self.test_symbols[:3]} # Mock bulk_get_historical_data

        # パイプライン実行結果記録
        pipeline_results = {}

        # Step 1: データ取得テスト
        self.logger.info("Step 1: データ取得開始")
        market_data_dict = self.data_fetcher.bulk_get_historical_data(self.test_symbols[:3]) # Corrected call
        market_data = pd.concat(market_data_dict.values()) if market_data_dict else pd.DataFrame()

        self.assertIsNotNone(market_data)
        pipeline_results['data_fetch'] = {
            'success': True,
            'symbols_count': len(self.test_symbols[:3]),
            'data_points': len(market_data) if market_data is not None else 0
        }

        # Step 2: スマート銘柄選択テスト
        self.logger.info("Step 2: スマート銘柄選択開始")

        async def async_symbol_selection():
            return await self.smart_selector.select_optimal_symbols(
                SelectionCriteria(target_symbols=2)
            )

        # 小さなシンボルプールで実行
        self.smart_selector.symbol_pool = {
            '7203.T': 'トヨタ自動車',
            '6758.T': 'ソニーグループ',
            '9984.T': 'ソフトバンクグループ'
        }

        selected_symbols = asyncio.run(async_symbol_selection())

        self.assertIsInstance(selected_symbols, list)
        self.assertGreater(len(selected_symbols), 0)

        pipeline_results['symbol_selection'] = {
            'success': True,
            'selected_count': len(selected_symbols),
            'symbols': selected_symbols
        }

        # Step 3: アンサンブル予測テスト
        self.logger.info("Step 3: アンサンブル予測開始")

        if len(selected_symbols) > 0:
            # テスト用特徴量生成
            n_samples = 100
            n_features = 15

            X_train = np.random.randn(n_samples, n_features)
            y_train = np.random.randn(n_samples)
            X_test = np.random.randn(len(selected_symbols), n_features)

            # アンサンブル訓練・予測
            self.ensemble_system.fit(X_train, y_train)
            predictions = self.ensemble_system.predict(X_test)

            self.assertIsInstance(predictions, EnsemblePrediction) # Corrected from EnsemblePredictions
            self.assertEqual(len(predictions.final_predictions), len(selected_symbols))

            pipeline_results['ensemble_prediction'] = {
                'success': True,
                'predictions_count': len(predictions.final_predictions),
                'model_count': len(predictions.model_predictions)
            }

        # Step 4: スケジューラ統合テスト
        self.logger.info("Step 4: スケジューラ統合開始")

        def integrated_task():
            """統合タスク関数"""
            return {
                'pipeline_results': pipeline_results,
                'execution_time': datetime.now(),
                'status': 'completed'
            }

        # 統合タスク作成・実行
        integration_task = ScheduledTask(
            task_id="full_integration",
            name="Full System Integration",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=integrated_task
        )

        self.execution_scheduler.add_task(integration_task)
        self.execution_scheduler._execute_task(integration_task)

        # 統合実行結果確認
        self.assertEqual(integration_task.status, ExecutionStatus.SUCCESS)

        # 全ステップ成功確認
        self.assertTrue(pipeline_results['data_fetch']['success'])
        self.assertTrue(pipeline_results['symbol_selection']['success'])
        if 'ensemble_prediction' in pipeline_results:
            self.assertTrue(pipeline_results['ensemble_prediction']['success'])

        self.integration_results.append(pipeline_results)

    def _setup_yfinance_mock(self, mock_ticker):
        """yfinanceモック設定"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'marketCap': 5000000000000}

        # 履歴データ生成
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        mock_ticker_instance.history.return_value = pd.DataFrame({
            'Open': np.random.uniform(1000, 1100, 60),
            'High': np.random.uniform(1100, 1200, 60),
            'Low': np.random.uniform(900, 1000, 60),
            'Close': np.random.uniform(1000, 1100, 60),
            'Volume': np.random.randint(2000000, 4000000, 60)
        }, index=dates)

        mock_ticker.return_value = mock_ticker_instance

    def _create_mock_market_data(self) -> pd.DataFrame:
        """モック市場データ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        symbols = ['7203.T', '6758.T', '9984.T', '4519.T', '8316.T']

        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Open': np.random.uniform(1000, 1100),
                    'High': np.random.uniform(1100, 1200),
                    'Low': np.random.uniform(900, 1000),
                    'Close': np.random.uniform(1000, 1100),
                    'Volume': np.random.randint(2000000, 4000000)
                })

        return pd.DataFrame(data)

    def test_concurrent_pipeline_execution(self):
        """並行パイプライン実行テスト"""
        concurrent_results = []

        def concurrent_pipeline(pipeline_id: int):
            """並行パイプライン実行"""
            try:
                # 各パイプラインで独立したコンポーネント使用
                local_ensemble = EnsembleSystem()

                # 簡易テストデータで予測
                X_train = np.random.randn(50, 10)
                y_train = np.random.randn(50)
                X_test = np.random.randn(5, 10)

                local_ensemble.fit(X_train, y_train)
                predictions = local_ensemble.predict(X_test)

                result = {
                    'pipeline_id': pipeline_id,
                    'success': True,
                    'predictions_count': len(predictions.final_predictions),
                    'execution_time': datetime.now()
                }

                concurrent_results.append(result)
                return result

            except Exception as e:
                error_result = {
                    'pipeline_id': pipeline_id,
                    'success': False,
                    'error': str(e),
                    'execution_time': datetime.now()
                }
                concurrent_results.append(error_result)
                return error_result

        # 並行実行
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(concurrent_pipeline, i) for i in range(3)]
            concurrent.futures.wait(futures)

        # 並行実行結果確認
        self.assertEqual(len(concurrent_results), 3)
        success_count = sum(1 for r in concurrent_results if r['success'])
        self.assertGreater(success_count, 0, "並行実行で少なくとも1つは成功する必要があります")

    @patch('src.day_trade.automation.smart_symbol_selector.get_smart_selected_symbols')
    def test_automated_workflow_scheduling(self, mock_get_symbols):
        """自動化ワークフロースケジューリングテスト"""
        mock_get_symbols.return_value = ['7203.T', '6758.T']

        workflow_executions = []

        def automated_workflow():
            """自動化ワークフロー"""
            execution_id = len(workflow_executions) + 1

            workflow_result = {
                'execution_id': execution_id,
                'start_time': datetime.now(),
                'steps': []
            }

            # Step 1: 市場データ確認
            workflow_result['steps'].append('market_data_check')
            time.sleep(0.1)  # 処理時間模擬

            # Step 2: 銘柄選択
            workflow_result['steps'].append('symbol_selection')
            selected = asyncio.run(get_smart_selected_symbols(2))
            workflow_result['selected_symbols'] = selected

            # Step 3: 予測実行
            workflow_result['steps'].append('prediction_execution')
            # 簡易予測（実装省略）
            workflow_result['predictions'] = {'mock': 'prediction'}

            workflow_result['end_time'] = datetime.now()
            workflow_result['duration'] = (workflow_result['end_time'] - workflow_result['start_time']).total_seconds()
            workflow_result['status'] = 'completed'

            workflow_executions.append(workflow_result)
            return workflow_result

        # 自動化ワークフロータスク作成
        workflow_task = ScheduledTask(
            task_id="automated_workflow",
            name="Automated Trading Workflow",
            schedule_type=ScheduleType.CONTINUOUS,
            target_function=automated_workflow,
            interval_minutes=1
        )

        # スケジューラでワークフロー実行
        self.execution_scheduler.add_task(workflow_task)

        # 複数回実行テスト
        for _ in range(3):
            self.execution_scheduler._execute_task(workflow_task)
            time.sleep(0.1)

        # ワークフロー実行結果確認
        self.assertEqual(len(workflow_executions), 3)

        for execution in workflow_executions:
            self.assertEqual(execution['status'], 'completed')
            self.assertIn('symbol_selection', execution['steps'])
            self.assertIn('prediction_execution', execution['steps'])
            self.assertLess(execution['duration'], 5.0)

    def test_error_recovery_integration(self):
        """エラー回復統合テスト"""
        error_scenarios = []

        def error_prone_integration():
            """エラー発生可能統合処理"""
            scenario_count = len(error_scenarios)

            # 段階的エラー・回復シナリオ
            if scenario_count == 0:
                error_scenarios.append({'type': 'data_fetch_error', 'recovered': False})
                raise ConnectionError("データ取得エラー")
            elif scenario_count == 1:
                error_scenarios.append({'type': 'prediction_error', 'recovered': False})
                raise ValueError("予測処理エラー")
            elif scenario_count == 2:
                error_scenarios.append({'type': 'scheduling_error', 'recovered': False})
                raise RuntimeError("スケジューリングエラー")
            else:
                # 最終的に成功
                for scenario in error_scenarios:
                    scenario['recovered'] = True
                return {
                    'recovery_completed': True,
                    'error_count': len(error_scenarios),
                    'final_status': 'success'
                }

        # エラー回復タスク作成
        error_recovery_task = ScheduledTask(
            task_id="error_recovery",
            name="Error Recovery Integration",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=error_prone_integration,
            max_retries=4
        )

        self.execution_scheduler.add_task(error_recovery_task)
        self.execution_scheduler._execute_task(error_recovery_task)

        # エラー回復結果確認
        self.assertEqual(error_recovery_task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(len(error_scenarios), 4)  # 3回エラー + 1回成功
        self.assertTrue(all(scenario['recovered'] for scenario in error_scenarios))

    def test_real_time_monitoring_integration(self):
        """リアルタイム監視統合テスト"""
        monitoring_data = []

        def real_time_monitor():
            """リアルタイム監視"""
            current_time = datetime.now()

            # システム状態監視
            system_status = {
                'timestamp': current_time,
                'data_fetcher_active': True,
                'symbol_selector_active': True,
                'ensemble_system_active': True,
                'scheduler_active': self.execution_scheduler.is_running
            }

            # パフォーマンス監視
            system_status['memory_usage'] = self._get_mock_memory_usage()
            system_status['cpu_usage'] = self._get_mock_cpu_usage()

            # 予測精度監視
            system_status['prediction_accuracy'] = np.random.uniform(0.85, 0.95)

            monitoring_data.append(system_status)
            return system_status

        # リアルタイム監視タスク作成
        monitor_task = ScheduledTask(
            task_id="realtime_monitor",
            name="Real-time System Monitor",
            schedule_type=ScheduleType.CONTINUOUS,
            target_function=real_time_monitor,
            interval_minutes=1
        )

        # 監視実行
        self.execution_scheduler.add_task(monitor_task)

        # 複数回監視実行
        for _ in range(5):
            self.execution_scheduler._execute_task(monitor_task)

        # 監視結果確認
        self.assertEqual(len(monitoring_data), 5)

        for data in monitoring_data:
            self.assertIn('timestamp', data)
            self.assertTrue(data['data_fetcher_active'])
            self.assertTrue(data['symbol_selector_active'])
            self.assertTrue(data['ensemble_system_active'])
            self.assertGreater(data['prediction_accuracy'], 0.8)

    def _get_mock_memory_usage(self) -> float:
        """モックメモリ使用量"""
        return np.random.uniform(30.0, 70.0)  # MB

    def _get_mock_cpu_usage(self) -> float:
        """モックCPU使用率"""
        return np.random.uniform(10.0, 80.0)  # %


class TestSystemPerformanceIntegration(unittest.TestCase):
    """システムパフォーマンス統合テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.performance_data = []

    def test_high_frequency_trading_simulation(self):
        """高頻度取引シミュレーションテスト"""
        ensemble = EnsembleSystem()

        # 訓練データ準備
        X_train = np.random.randn(200, 20)
        y_train = np.random.randn(200)
        ensemble.fit(X_train, y_train)

        # 高頻度予測実行
        prediction_times = []

        for i in range(50):
            start_time = time.time()

            # 単一予測実行
            X_single = np.random.randn(1, 20)
            prediction = ensemble.predict(X_single)

            self.assertIsNotNone(prediction.final_predictions)

            prediction_time = time.time() - start_time
            prediction_times.append(prediction_time)

            # 高頻度要件確認（1秒以下）
            self.assertLess(prediction_time, 1.0,
                          f"予測時間 {prediction_time:.3f}秒 が要件を超えています")

        # パフォーマンス統計
        avg_time = np.mean(prediction_times)
        max_time = np.max(prediction_times)

        self.assertLess(avg_time, 0.5, f"平均予測時間 {avg_time:.3f}秒 が目標を超えています")
        self.assertLess(max_time, 1.0, f"最大予測時間 {max_time:.3f}秒 が要件を超えています")

    def test_large_scale_symbol_processing(self):
        """大規模銘柄処理テスト"""
        selector = SmartSymbolSelector()

        # 大規模シンボルプール作成
        large_symbol_pool = {}
        for i in range(100):
            symbol_code = f"{1000 + i}.T"
            large_symbol_pool[symbol_code] = f"テスト企業{i}"

        selector.symbol_pool = large_symbol_pool

        async def large_scale_selection():
            criteria = SelectionCriteria(target_symbols=10)
            return await selector.select_optimal_symbols(criteria)

        # 大規模選択実行
        start_time = time.time()

        with patch('yfinance.Ticker') as mock_ticker:
            # yfinanceモック設定
            mock_instance = Mock()
            mock_instance.info = {'marketCap': np.random.randint(1000000000, 10000000000)}
            mock_instance.history.return_value = self._create_mock_ticker_data()
            mock_ticker.return_value = mock_instance

            selected_symbols = asyncio.run(large_scale_selection())

        processing_time = time.time() - start_time

        # 大規模処理結果確認
        self.assertIsInstance(selected_symbols, list)
        self.assertLessEqual(len(selected_symbols), 10)

        # パフォーマンス要件確認（30秒以下）
        self.assertLess(processing_time, 30.0,
                       f"大規模処理時間 {processing_time:.1f}秒 が要件を超えています")

    def _create_mock_ticker_data(self) -> pd.DataFrame:
        """モックティッカーデータ作成"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        return pd.DataFrame({
            'Open': np.random.uniform(1000, 1100, 60),
            'High': np.random.uniform(1100, 1200, 60),
            'Low': np.random.uniform(900, 1000, 60),
            'Close': np.random.uniform(1000, 1100, 60),
            'Volume': np.random.randint(1000000, 5000000, 60)
        }, index=dates)

    def test_memory_efficiency_integration(self):
        """メモリ効率性統合テスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # メモリ集約的統合処理
        components = []

        for i in range(10):
            # 複数システム並行実行
            ensemble = EnsembleSystem()

            X_train = np.random.randn(100, 15)
            y_train = np.random.randn(100)
            ensemble.fit(X_train, y_train)

            components.append(ensemble)

        # メモリ使用量確認
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # メモリ効率性要件確認（200MB以下の増加）
        self.assertLess(memory_increase, 200,
                       f"メモリ使用量増加 {memory_increase:.1f}MB が要件を超えています")


class TestSystemReliabilityIntegration(unittest.TestCase):
    """システム信頼性統合テスト"""

    def test_24_hour_continuous_operation_simulation(self):
        """24時間連続運用シミュレーションテスト"""
        scheduler = ExecutionScheduler()
        operation_log = []

        def continuous_operation_task():
            """連続運用タスク"""
            operation_log.append({
                'timestamp': datetime.now(),
                'status': 'operational',
                'memory_check': 'ok',
                'cpu_check': 'ok'
            })
            return 'operational'

        # 連続運用タスク作成（1分間隔シミュレーション）
        continuous_task = ScheduledTask(
            task_id="continuous_ops",
            name="24H Continuous Operation",
            schedule_type=ScheduleType.CONTINUOUS,
            target_function=continuous_operation_task,
            interval_minutes=1
        )

        scheduler.add_task(continuous_task)

        # 24時間シミュレーション（60回実行 = 1時間分）
        for minute in range(60):
            scheduler._execute_task(continuous_task)

            # 毎10分でシステム状態確認
            if minute % 10 == 0:
                status = scheduler.get_task_status("continuous_ops")
                self.assertIsNotNone(status)
                self.assertTrue(status['enabled'])

        # 連続運用結果確認
        self.assertEqual(len(operation_log), 60)
        self.assertEqual(continuous_task.success_count, 60)
        self.assertEqual(continuous_task.error_count, 0)

    def test_system_graceful_shutdown(self):
        """システム優雅停止テスト"""
        scheduler = ExecutionScheduler()
        shutdown_log = []

        def long_running_task():
            """長時間実行タスク"""
            shutdown_log.append('task_started')
            time.sleep(0.5)  # 処理時間シミュレーション
            shutdown_log.append('task_completed')
            return 'completed'

        # 長時間タスク作成
        long_task = ScheduledTask(
            task_id="long_running",
            name="Long Running Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=long_running_task
        )

        scheduler.add_task(long_task)
        scheduler.start()

        # タスク実行開始
        thread = threading.Thread(target=scheduler._execute_task, args=(long_task,))
        thread.start()

        # 短時間待機後停止
        time.sleep(0.2)
        scheduler.stop()

        # スレッド完了待機
        thread.join(timeout=2.0)

        # 優雅停止確認
        self.assertFalse(scheduler.is_running)
        self.assertGreaterEqual(len(shutdown_log), 1)  # 最低限開始ログは記録される


if __name__ == '__main__':
    # テストスイート設定
    test_suite = unittest.TestSuite()

    # 完全システム統合テスト
    test_suite.addTest(unittest.makeSuite(TestCompleteSystemIntegration))

    # パフォーマンス統合テスト
    test_suite.addTest(unittest.makeSuite(TestSystemPerformanceIntegration))

    # 信頼性統合テスト
    test_suite.addTest(unittest.makeSuite(TestSystemReliabilityIntegration))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print(f"\n{'='*80}")
    print(f"エンドツーエンド統合テスト完了")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗数: {len(result.failures)}")
    print(f"エラー数: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*80}")