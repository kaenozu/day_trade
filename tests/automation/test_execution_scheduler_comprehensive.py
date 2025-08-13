#!/usr/bin/env python3
"""
Issue #755 Phase 4: ExecutionScheduler包括的テストスイート

実行スケジューラシステムの詳細検証
- スケジュール管理テスト
- タスク実行・管理テスト
- 市場時間・条件実行テスト
- エラーハンドリング・堅牢性テスト
- パフォーマンス・同期処理テスト
"""

import unittest
import pytest
import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from typing import Dict, List, Any

# テスト対象システムのインポート
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.day_trade.automation.execution_scheduler import (
        ExecutionScheduler,
        ScheduledTask,
        ExecutionResult,
        ScheduleType,
        ExecutionStatus,
        smart_stock_analysis_task,
        create_default_automation_tasks
    )
    from src.day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig

except ImportError as e:
    print(f"インポートエラー: {e}")
    sys.exit(1)


class TestExecutionSchedulerCore(unittest.TestCase):
    """ExecutionSchedulerコア機能テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()
        self.test_result = None

        # テスト用タスク関数
        def test_function(value: int = 1, **kwargs):
            self.test_result = value * 2
            return self.test_result

        def async_test_function(value: int = 1, **kwargs):
            return asyncio.run(self._async_helper(value))

        async def _async_helper(value: int):
            await asyncio.sleep(0.1)
            return value * 3

        self.test_function = test_function
        self.async_test_function = async_test_function

    def test_scheduler_initialization(self):
        """スケジューラ初期化テスト"""
        # 基本初期化確認
        self.assertIsInstance(self.scheduler.tasks, dict)
        self.assertEqual(len(self.scheduler.tasks), 0)
        self.assertFalse(self.scheduler.is_running)
        self.assertIsNone(self.scheduler.scheduler_thread)

        # アンサンブルシステム統合確認
        self.assertIsNotNone(self.scheduler.ensemble_system)
        self.assertIsInstance(self.scheduler.ensemble_system, EnsembleSystem)

        # 市場時間設定確認
        self.assertIn('start', self.scheduler.market_hours)
        self.assertIn('end', self.scheduler.market_hours)
        self.assertEqual(self.scheduler.market_hours['start'], (9, 0))
        self.assertEqual(self.scheduler.market_hours['end'], (15, 0))

    def test_task_addition(self):
        """タスク追加テスト"""
        # 日次スケジュールタスク作成
        task = ScheduledTask(
            task_id="test_daily",
            name="Daily Test Task",
            schedule_type=ScheduleType.DAILY,
            target_function=self.test_function,
            parameters={'value': 5},
            schedule_time="09:00"
        )

        # タスク追加
        success = self.scheduler.add_task(task)

        # 追加結果確認
        self.assertTrue(success)
        self.assertIn("test_daily", self.scheduler.tasks)
        self.assertEqual(len(self.scheduler.tasks), 1)

        # タスク詳細確認
        added_task = self.scheduler.tasks["test_daily"]
        self.assertEqual(added_task.name, "Daily Test Task")
        self.assertEqual(added_task.schedule_type, ScheduleType.DAILY)
        self.assertIsNotNone(added_task.next_execution)

    def test_next_execution_calculation(self):
        """次回実行時刻計算テスト"""
        # 日次実行の場合
        daily_task = ScheduledTask(
            task_id="test_daily",
            name="Daily Task",
            schedule_type=ScheduleType.DAILY,
            target_function=self.test_function,
            schedule_time="10:30"
        )

        next_exec = self.scheduler._calculate_next_execution(daily_task)
        self.assertIsInstance(next_exec, datetime)
        self.assertEqual(next_exec.hour, 10)
        self.assertEqual(next_exec.minute, 30)

        # 時間次実行の場合
        hourly_task = ScheduledTask(
            task_id="test_hourly",
            name="Hourly Task",
            schedule_type=ScheduleType.HOURLY,
            target_function=self.test_function
        )

        next_exec_hourly = self.scheduler._calculate_next_execution(hourly_task)
        self.assertIsInstance(next_exec_hourly, datetime)
        self.assertEqual(next_exec_hourly.minute, 0)

        # 連続実行の場合
        continuous_task = ScheduledTask(
            task_id="test_continuous",
            name="Continuous Task",
            schedule_type=ScheduleType.CONTINUOUS,
            target_function=self.test_function,
            interval_minutes=30
        )

        next_exec_continuous = self.scheduler._calculate_next_execution(continuous_task)
        expected_time = datetime.now() + timedelta(minutes=30)
        self.assertAlmostEqual(
            next_exec_continuous.timestamp(),
            expected_time.timestamp(),
            delta=60  # 1分の誤差を許容
        )

    def test_market_time_calculation(self):
        """市場時間計算テスト"""
        # 市場開始前の場合
        morning_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)

        with patch('src.day_trade.automation.execution_scheduler.datetime') as mock_datetime:
            mock_datetime.now.return_value = morning_time
            mock_datetime.return_value.replace = morning_time.replace
            mock_datetime.return_value.time = morning_time.time

            next_market = self.scheduler._next_market_time()
            self.assertEqual(next_market.hour, 9)
            self.assertEqual(next_market.minute, 0)

    def test_task_status_management(self):
        """タスクステータス管理テスト"""
        # テストタスク作成・追加
        task = ScheduledTask(
            task_id="status_test",
            name="Status Test Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=self.test_function
        )

        self.scheduler.add_task(task)

        # ステータス取得テスト
        status = self.scheduler.get_task_status("status_test")
        self.assertIsNotNone(status)
        self.assertEqual(status['task_id'], "status_test")
        self.assertEqual(status['name'], "Status Test Task")
        self.assertEqual(status['status'], ExecutionStatus.READY.value)
        self.assertTrue(status['enabled'])
        self.assertEqual(status['execution_count'], 0)

        # 全タスクステータス取得テスト
        all_status = self.scheduler.get_all_tasks_status()
        self.assertEqual(len(all_status), 1)
        self.assertEqual(all_status[0]['task_id'], "status_test")

    def test_task_pause_resume(self):
        """タスク一時停止・再開テスト"""
        # テストタスク作成・追加
        task = ScheduledTask(
            task_id="pause_test",
            name="Pause Test Task",
            schedule_type=ScheduleType.DAILY,
            target_function=self.test_function,
            schedule_time="12:00"
        )

        self.scheduler.add_task(task)

        # 一時停止テスト
        pause_success = self.scheduler.pause_task("pause_test")
        self.assertTrue(pause_success)

        paused_status = self.scheduler.get_task_status("pause_test")
        self.assertFalse(paused_status['enabled'])
        self.assertEqual(paused_status['status'], ExecutionStatus.PAUSED.value)

        # 再開テスト
        resume_success = self.scheduler.resume_task("pause_test")
        self.assertTrue(resume_success)

        resumed_status = self.scheduler.get_task_status("pause_test")
        self.assertTrue(resumed_status['enabled'])
        self.assertEqual(resumed_status['status'], ExecutionStatus.READY.value)

        # 存在しないタスクの操作テスト
        self.assertFalse(self.scheduler.pause_task("nonexistent"))
        self.assertFalse(self.scheduler.resume_task("nonexistent"))


class TestExecutionSchedulerExecution(unittest.TestCase):
    """タスク実行機能テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()
        self.execution_results = []

        # テスト用関数
        def successful_function(value: int = 1):
            self.execution_results.append(f"success_{value}")
            return value * 2

        def failing_function():
            raise ValueError("Test error")

        def slow_function():
            time.sleep(0.2)
            return "slow_result"

        self.successful_function = successful_function
        self.failing_function = failing_function
        self.slow_function = slow_function

    def test_task_execution_success(self):
        """タスク実行成功テスト"""
        # 成功タスク作成
        task = ScheduledTask(
            task_id="success_test",
            name="Success Test",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=self.successful_function,
            parameters={'value': 10}
        )

        # タスク実行
        self.scheduler._execute_task(task)

        # 実行結果確認
        self.assertEqual(len(self.execution_results), 1)
        self.assertEqual(self.execution_results[0], "success_10")

        # タスクステータス確認
        self.assertEqual(task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(task.success_count, 1)
        self.assertEqual(task.error_count, 0)
        self.assertIsNone(task.last_error)

        # 実行履歴確認
        self.assertEqual(len(self.scheduler.execution_history), 1)
        history = self.scheduler.execution_history[0]
        self.assertEqual(history.status, ExecutionStatus.SUCCESS)
        self.assertEqual(history.result_data, 20)

    def test_task_execution_failure(self):
        """タスク実行失敗テスト"""
        # 失敗タスク作成
        task = ScheduledTask(
            task_id="failure_test",
            name="Failure Test",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=self.failing_function,
            max_retries=1  # リトライ回数を制限
        )

        # タスク実行
        self.scheduler._execute_task(task)

        # タスクステータス確認
        self.assertEqual(task.status, ExecutionStatus.FAILED)
        self.assertEqual(task.success_count, 0)
        self.assertEqual(task.error_count, 1)
        self.assertIsNotNone(task.last_error)
        self.assertIn("Test error", task.last_error)

        # 実行履歴確認
        self.assertEqual(len(self.scheduler.execution_history), 1)
        history = self.scheduler.execution_history[0]
        self.assertEqual(history.status, ExecutionStatus.FAILED)
        self.assertIsNotNone(history.error_message)

    def test_task_retry_mechanism(self):
        """タスクリトライ機能テスト"""
        retry_count = 0

        def flaky_function():
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise RuntimeError(f"Attempt {retry_count} failed")
            return f"success_on_attempt_{retry_count}"

        # リトライタスク作成
        task = ScheduledTask(
            task_id="retry_test",
            name="Retry Test",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=flaky_function,
            max_retries=3
        )

        # タスク実行
        start_time = time.time()
        self.scheduler._execute_task(task)
        execution_time = time.time() - start_time

        # リトライ結果確認
        self.assertEqual(retry_count, 3)
        self.assertEqual(task.status, ExecutionStatus.SUCCESS)
        self.assertEqual(task.success_count, 1)

        # リトライによる待機時間確認（ある程度の時間がかかることを確認）
        self.assertGreater(execution_time, 0.5)  # リトライ待機時間を考慮

    def test_task_execution_timeout(self):
        """タスク実行タイムアウトテスト"""
        # 長時間実行タスク（実際のタイムアウト機能実装が必要）
        task = ScheduledTask(
            task_id="timeout_test",
            name="Timeout Test",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=self.slow_function,
            timeout_minutes=1
        )

        # タスク実行
        start_time = time.time()
        self.scheduler._execute_task(task)
        execution_time = time.time() - start_time

        # 正常完了確認（タイムアウト前）
        self.assertEqual(task.status, ExecutionStatus.SUCCESS)
        self.assertLess(execution_time, 1.0)


class TestExecutionSchedulerScheduling(unittest.TestCase):
    """スケジューリング機能テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()
        self.execution_log = []

        def log_function(task_name: str):
            self.execution_log.append(f"{task_name}_{datetime.now().isoformat()}")
            return f"executed_{task_name}"

        self.log_function = log_function

    def test_scheduler_start_stop(self):
        """スケジューラ開始・停止テスト"""
        # 初期状態確認
        self.assertFalse(self.scheduler.is_running)
        self.assertIsNone(self.scheduler.scheduler_thread)

        # 開始テスト
        self.scheduler.start()
        self.assertTrue(self.scheduler.is_running)
        self.assertIsNotNone(self.scheduler.scheduler_thread)
        self.assertTrue(self.scheduler.scheduler_thread.is_alive())

        # 重複開始テスト
        self.scheduler.start()  # 警告が出るが正常動作
        self.assertTrue(self.scheduler.is_running)

        # 停止テスト
        self.scheduler.stop()
        time.sleep(0.1)  # 停止処理の完了を待機
        self.assertFalse(self.scheduler.is_running)

    def test_immediate_execution_scheduling(self):
        """即座実行スケジューリングテスト"""
        # 即座実行タスク作成
        task = ScheduledTask(
            task_id="immediate_test",
            name="Immediate Test",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=lambda: self.log_function("immediate")
        )

        # 次回実行時刻を現在時刻に設定
        task.next_execution = datetime.now() - timedelta(seconds=1)
        self.scheduler.add_task(task)

        # スケジューラ開始
        self.scheduler.start()

        # 実行完了まで待機
        timeout = 5.0
        start_wait = time.time()
        while len(self.execution_log) == 0 and (time.time() - start_wait) < timeout:
            time.sleep(0.1)

        # スケジューラ停止
        self.scheduler.stop()

        # 実行結果確認
        self.assertGreater(len(self.execution_log), 0)
        self.assertIn("immediate", self.execution_log[0])

    def test_multiple_task_scheduling(self):
        """複数タスクスケジューリングテスト"""
        # 複数タスク作成
        tasks = []
        for i in range(3):
            task = ScheduledTask(
                task_id=f"multi_test_{i}",
                name=f"Multi Test {i}",
                schedule_type=ScheduleType.ON_DEMAND,
                target_function=lambda name=f"task_{i}": self.log_function(name)
            )
            task.next_execution = datetime.now() + timedelta(seconds=i * 0.1)
            tasks.append(task)
            self.scheduler.add_task(task)

        # スケジューラ開始
        self.scheduler.start()

        # 全タスク実行完了まで待機
        timeout = 10.0
        start_wait = time.time()
        while len(self.execution_log) < 3 and (time.time() - start_wait) < timeout:
            time.sleep(0.1)

        # スケジューラ停止
        self.scheduler.stop()

        # 実行結果確認
        self.assertEqual(len(self.execution_log), 3)
        for i in range(3):
            self.assertTrue(any(f"task_{i}" in log for log in self.execution_log))


class TestExecutionSchedulerIntegration(unittest.TestCase):
    """統合機能テスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()

    @patch('src.day_trade.automation.smart_symbol_selector.get_smart_selected_symbols')
    def test_smart_analysis_task_integration(self, mock_get_symbols):
        """スマート分析タスク統合テスト"""
        # SmartSymbolSelectorのモック設定
        mock_get_symbols.return_value = ['7203.T', '6758.T', '9984.T']

        # スマート分析タスク実行
        result = asyncio.run(smart_stock_analysis_task(target_count=3))

        # 結果検証
        self.assertIsInstance(result, dict)
        self.assertIn('timestamp', result)
        self.assertIn('selected_symbols', result)
        self.assertIn('predictions', result)
        self.assertEqual(result['status'], 'success')

        # 選定銘柄確認
        self.assertEqual(len(result['selected_symbols']), 3)
        self.assertIn('7203.T', result['selected_symbols'])

        # 予測結果確認
        self.assertEqual(len(result['predictions']), 3)
        for symbol in result['selected_symbols']:
            self.assertIn(symbol, result['predictions'])
            prediction = result['predictions'][symbol]
            self.assertTrue(prediction['prediction_ready'])
            self.assertEqual(prediction['accuracy_target'], 93.0)

    def test_default_automation_tasks_creation(self):
        """デフォルト自動化タスク作成テスト"""
        # デフォルトタスク生成
        default_tasks = create_default_automation_tasks()

        # タスク数確認
        self.assertEqual(len(default_tasks), 3)

        # タスク詳細確認
        task_ids = [task.task_id for task in default_tasks]
        self.assertIn("morning_analysis", task_ids)
        self.assertIn("lunch_analysis", task_ids)
        self.assertIn("closing_analysis", task_ids)

        # スケジュール時間確認
        morning_task = next(task for task in default_tasks if task.task_id == "morning_analysis")
        self.assertEqual(morning_task.schedule_time, "08:30")
        self.assertEqual(morning_task.schedule_type, ScheduleType.DAILY)

        lunch_task = next(task for task in default_tasks if task.task_id == "lunch_analysis")
        self.assertEqual(lunch_task.schedule_time, "12:00")

        closing_task = next(task for task in default_tasks if task.task_id == "closing_analysis")
        self.assertEqual(closing_task.schedule_time, "15:30")

    def test_ensemble_system_integration(self):
        """EnsembleSystem統合テスト"""
        # EnsembleSystem統合確認
        self.assertIsNotNone(self.scheduler.ensemble_system)
        self.assertIsInstance(self.scheduler.ensemble_system, EnsembleSystem)

        # 設定確認
        config = self.scheduler.ensemble_system.config
        self.assertTrue(config.use_xgboost)
        self.assertTrue(config.use_catboost)
        self.assertTrue(config.use_random_forest)


class TestExecutionSchedulerRobustness(unittest.TestCase):
    """堅牢性・エラーハンドリングテスト"""

    def setUp(self):
        """テスト環境セットアップ"""
        self.scheduler = ExecutionScheduler()

    def test_invalid_task_handling(self):
        """無効タスク処理テスト"""
        # 無効な関数を持つタスク
        invalid_task = ScheduledTask(
            task_id="invalid_test",
            name="Invalid Test",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=None  # 無効な関数
        )

        # タスク実行でエラーが適切に処理されることを確認
        with self.assertRaises(Exception):
            self.scheduler._run_task_function(invalid_task)

    def test_concurrent_task_execution(self):
        """並行タスク実行テスト"""
        execution_results = []

        def concurrent_function(task_id: str):
            time.sleep(0.1)
            execution_results.append(f"executed_{task_id}")
            return f"result_{task_id}"

        # 複数タスク作成
        tasks = []
        for i in range(3):
            task = ScheduledTask(
                task_id=f"concurrent_{i}",
                name=f"Concurrent Task {i}",
                schedule_type=ScheduleType.ON_DEMAND,
                target_function=lambda tid=str(i): concurrent_function(tid)
            )
            tasks.append(task)
            self.scheduler.add_task(task)

        # 並行実行シミュレート
        threads = []
        for task in tasks:
            thread = threading.Thread(target=self.scheduler._execute_task, args=(task,))
            threads.append(thread)

        # 全スレッド開始
        for thread in threads:
            thread.start()

        # 全スレッド完了待機
        for thread in threads:
            thread.join(timeout=5.0)

        # 結果確認
        self.assertEqual(len(execution_results), 3)
        for i in range(3):
            self.assertIn(f"executed_{i}", execution_results)

    def test_memory_efficiency(self):
        """メモリ効率性テスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 大量のタスク追加
        for i in range(100):
            task = ScheduledTask(
                task_id=f"memory_test_{i}",
                name=f"Memory Test {i}",
                schedule_type=ScheduleType.ON_DEMAND,
                target_function=lambda: "result"
            )
            self.scheduler.add_task(task)

        # メモリ使用量確認
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # メモリ使用量が合理的範囲内であることを確認
        self.assertLess(memory_increase, 50,
                       f"メモリ使用量増加 {memory_increase:.1f}MB が過大です")

    def test_scheduler_loop_error_recovery(self):
        """スケジューラループエラー回復テスト"""
        error_count = 0

        def error_prone_function():
            nonlocal error_count
            error_count += 1
            if error_count <= 2:
                raise RuntimeError(f"Error {error_count}")
            return "recovered"

        # エラー発生タスク作成
        task = ScheduledTask(
            task_id="error_recovery",
            name="Error Recovery Test",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=error_prone_function,
            max_retries=3
        )

        task.next_execution = datetime.now() - timedelta(seconds=1)
        self.scheduler.add_task(task)

        # スケジューラ開始
        self.scheduler.start()

        # エラー回復まで待機
        timeout = 10.0
        start_wait = time.time()
        while task.status != ExecutionStatus.SUCCESS and (time.time() - start_wait) < timeout:
            time.sleep(0.1)

        # スケジューラ停止
        self.scheduler.stop()

        # エラー回復確認
        self.assertEqual(task.status, ExecutionStatus.SUCCESS)
        self.assertGreater(error_count, 2)


if __name__ == '__main__':
    # テストスイート設定
    test_suite = unittest.TestSuite()

    # コア機能テスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerCore))

    # タスク実行テスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerExecution))

    # スケジューリングテスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerScheduling))

    # 統合テスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerIntegration))

    # 堅牢性テスト
    test_suite.addTest(unittest.makeSuite(TestExecutionSchedulerRobustness))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print(f"\n{'='*70}")
    print(f"ExecutionScheduler包括的テスト完了")
    print(f"実行テスト数: {result.testsRun}")
    print(f"失敗数: {len(result.failures)}")
    print(f"エラー数: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*70}")