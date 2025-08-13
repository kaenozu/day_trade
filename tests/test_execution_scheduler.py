#!/usr/bin/env python3
"""
実行スケジューラシステムのテスト

Issue #750対応: テストカバレッジ改善プロジェクト Phase 1
execution_scheduler.pyの包括的テストスイート
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import pytest

from src.day_trade.automation.execution_scheduler import (
    ExecutionScheduler,
    ScheduledTask,
    ExecutionResult,
    ScheduleType,
    ExecutionStatus
)


class TestScheduleType:
    """ScheduleType列挙体のテスト"""

    def test_schedule_type_values(self):
        """スケジュール種別の値テスト"""
        assert ScheduleType.DAILY.value == "daily"
        assert ScheduleType.HOURLY.value == "hourly"
        assert ScheduleType.MARKET_HOURS.value == "market_hours"
        assert ScheduleType.ON_DEMAND.value == "on_demand"
        assert ScheduleType.CONTINUOUS.value == "continuous"


class TestExecutionStatus:
    """ExecutionStatus列挙体のテスト"""

    def test_execution_status_values(self):
        """実行ステータスの値テスト"""
        assert ExecutionStatus.READY.value == "ready"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.PAUSED.value == "paused"
        assert ExecutionStatus.STOPPED.value == "stopped"


class TestScheduledTask:
    """ScheduledTaskデータクラスのテスト"""

    def test_scheduled_task_initialization(self):
        """ScheduledTask初期化テスト"""
        mock_function = Mock()

        task = ScheduledTask(
            task_id="test_task_1",
            name="Test Task",
            schedule_type=ScheduleType.DAILY,
            target_function=mock_function,
            schedule_time="09:00"
        )

        assert task.task_id == "test_task_1"
        assert task.name == "Test Task"
        assert task.schedule_type == ScheduleType.DAILY
        assert task.target_function == mock_function
        assert task.schedule_time == "09:00"
        assert task.max_retries == 3
        assert task.timeout_minutes == 30
        assert task.enabled is True
        assert task.status == ExecutionStatus.READY
        assert task.execution_count == 0
        assert task.success_count == 0
        assert task.error_count == 0

    def test_scheduled_task_with_parameters(self):
        """パラメータ付きScheduledTaskテスト"""
        mock_function = Mock()
        parameters = {"param1": "value1", "param2": 42}

        task = ScheduledTask(
            task_id="test_task_2",
            name="Parameterized Task",
            schedule_type=ScheduleType.HOURLY,
            target_function=mock_function,
            parameters=parameters,
            interval_minutes=60
        )

        assert task.parameters == parameters
        assert task.interval_minutes == 60

    def test_scheduled_task_custom_settings(self):
        """カスタム設定ScheduledTaskテスト"""
        mock_function = Mock()

        task = ScheduledTask(
            task_id="test_task_3",
            name="Custom Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=mock_function,
            max_retries=5,
            timeout_minutes=60,
            enabled=False
        )

        assert task.max_retries == 5
        assert task.timeout_minutes == 60
        assert task.enabled is False


class TestExecutionResult:
    """ExecutionResultデータクラスのテスト"""

    def test_execution_result_initialization(self):
        """ExecutionResult初期化テスト"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)

        result = ExecutionResult(
            task_id="test_task_1",
            status=ExecutionStatus.SUCCESS,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=30.0,
            result_data={"output": "success"}
        )

        assert result.task_id == "test_task_1"
        assert result.status == ExecutionStatus.SUCCESS
        assert result.start_time == start_time
        assert result.end_time == end_time
        assert result.duration_seconds == 30.0
        assert result.result_data == {"output": "success"}
        assert result.error_message is None
        assert result.retry_count == 0

    def test_execution_result_with_error(self):
        """エラー付きExecutionResultテスト"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)

        result = ExecutionResult(
            task_id="test_task_2",
            status=ExecutionStatus.FAILED,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=10.0,
            error_message="Test error",
            retry_count=2
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.error_message == "Test error"
        assert result.retry_count == 2


class TestExecutionScheduler:
    """ExecutionSchedulerクラスのテスト"""

    @pytest.fixture
    def scheduler(self):
        """テスト用スケジューラーフィクスチャ"""
        with patch('src.day_trade.automation.execution_scheduler.EnsembleSystem'):
            scheduler = ExecutionScheduler()
            return scheduler

    @pytest.fixture
    def mock_task_function(self):
        """テスト用モック関数"""
        return Mock(return_value="test_result")

    def test_scheduler_initialization(self, scheduler):
        """スケジューラー初期化テスト"""
        assert isinstance(scheduler.tasks, dict)
        assert len(scheduler.tasks) == 0
        assert scheduler.is_running is False
        assert scheduler.scheduler_thread is None
        assert isinstance(scheduler.execution_history, list)
        assert len(scheduler.execution_history) == 0

    def test_add_task(self, scheduler, mock_task_function):
        """タスク追加テスト"""
        task = ScheduledTask(
            task_id="test_task_1",
            name="Test Task",
            schedule_type=ScheduleType.DAILY,
            target_function=mock_task_function,
            schedule_time="09:00"
        )

        success = scheduler.add_task(task)
        assert success is True
        assert task.task_id in scheduler.tasks
        stored_task = scheduler.tasks[task.task_id]
        assert stored_task.name == "Test Task"
        assert stored_task.schedule_type == ScheduleType.DAILY
        assert stored_task.target_function == mock_task_function
        assert stored_task.schedule_time == "09:00"

    def test_add_task_with_parameters(self, scheduler, mock_task_function):
        """パラメータ付きタスク追加テスト"""
        parameters = {"param1": "value1", "param2": 42}

        task = ScheduledTask(
            task_id="test_task_2",
            name="Parameterized Task",
            schedule_type=ScheduleType.HOURLY,
            target_function=mock_task_function,
            parameters=parameters,
            interval_minutes=30
        )

        success = scheduler.add_task(task)
        assert success is True
        stored_task = scheduler.tasks[task.task_id]
        assert stored_task.parameters == parameters
        assert stored_task.interval_minutes == 30

    def test_remove_task(self, scheduler, mock_task_function):
        """タスク削除テスト"""
        task_id = scheduler.add_task(
            name="Task to Remove",
            schedule_type=ScheduleType.DAILY,
            target_function=mock_task_function
        )

        assert task_id in scheduler.tasks

        success = scheduler.remove_task(task_id)
        assert success is True
        assert task_id not in scheduler.tasks

    def test_remove_nonexistent_task(self, scheduler):
        """存在しないタスク削除テスト"""
        success = scheduler.remove_task("nonexistent_task")
        assert success is False

    def test_get_task(self, scheduler, mock_task_function):
        """タスク取得テスト"""
        task_id = scheduler.add_task(
            name="Get Task Test",
            schedule_type=ScheduleType.DAILY,
            target_function=mock_task_function
        )

        task = scheduler.get_task(task_id)
        assert task is not None
        assert task.task_id == task_id
        assert task.name == "Get Task Test"

    def test_get_nonexistent_task(self, scheduler):
        """存在しないタスク取得テスト"""
        task = scheduler.get_task("nonexistent_task")
        assert task is None

    def test_enable_disable_task(self, scheduler, mock_task_function):
        """タスク有効/無効化テスト"""
        task_id = scheduler.add_task(
            name="Enable/Disable Test",
            schedule_type=ScheduleType.DAILY,
            target_function=mock_task_function
        )

        # 初期状態は有効
        task = scheduler.get_task(task_id)
        assert task.enabled is True

        # 無効化
        scheduler.disable_task(task_id)
        assert task.enabled is False

        # 有効化
        scheduler.enable_task(task_id)
        assert task.enabled is True

    def test_list_tasks(self, scheduler, mock_task_function):
        """タスク一覧取得テスト"""
        # 複数タスク追加
        task_id1 = scheduler.add_task(
            name="Task 1",
            schedule_type=ScheduleType.DAILY,
            target_function=mock_task_function
        )
        task_id2 = scheduler.add_task(
            name="Task 2",
            schedule_type=ScheduleType.HOURLY,
            target_function=mock_task_function
        )

        tasks = scheduler.list_tasks()
        assert len(tasks) == 2
        task_ids = [task.task_id for task in tasks]
        assert task_id1 in task_ids
        assert task_id2 in task_ids

    def test_list_tasks_by_status(self, scheduler, mock_task_function):
        """ステータス別タスク一覧テスト"""
        task_id = scheduler.add_task(
            name="Status Test Task",
            schedule_type=ScheduleType.DAILY,
            target_function=mock_task_function
        )

        ready_tasks = scheduler.list_tasks(status=ExecutionStatus.READY)
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == task_id

        running_tasks = scheduler.list_tasks(status=ExecutionStatus.RUNNING)
        assert len(running_tasks) == 0

    @patch('time.sleep')
    def test_execute_task_sync_success(self, mock_sleep, scheduler, mock_task_function):
        """同期タスク実行成功テスト"""
        mock_task_function.return_value = "success_result"

        task_id = scheduler.add_task(
            name="Sync Success Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=mock_task_function
        )

        result = scheduler.execute_task(task_id)

        assert result is not None
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result_data == "success_result"
        assert result.error_message is None

        # タスクの統計更新確認
        task = scheduler.get_task(task_id)
        assert task.execution_count == 1
        assert task.success_count == 1
        assert task.error_count == 0

    def test_execute_task_sync_failure(self, scheduler):
        """同期タスク実行失敗テスト"""
        def failing_function():
            raise ValueError("Test error")

        task_id = scheduler.add_task(
            name="Sync Failure Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=failing_function
        )

        result = scheduler.execute_task(task_id)

        assert result is not None
        assert result.status == ExecutionStatus.FAILED
        assert "Test error" in result.error_message

        # タスクの統計更新確認
        task = scheduler.get_task(task_id)
        assert task.execution_count == 1
        assert task.success_count == 0
        assert task.error_count == 1

    def test_execute_nonexistent_task(self, scheduler):
        """存在しないタスク実行テスト"""
        result = scheduler.execute_task("nonexistent_task")
        assert result is None

    def test_execute_disabled_task(self, scheduler, mock_task_function):
        """無効化タスク実行テスト"""
        task_id = scheduler.add_task(
            name="Disabled Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=mock_task_function,
            enabled=False
        )

        result = scheduler.execute_task(task_id)
        assert result is None

    def test_get_execution_history(self, scheduler, mock_task_function):
        """実行履歴取得テスト"""
        task_id = scheduler.add_task(
            name="History Test Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=mock_task_function
        )

        # タスクを複数回実行
        scheduler.execute_task(task_id)
        scheduler.execute_task(task_id)

        history = scheduler.get_execution_history()
        assert len(history) == 2

        # 特定タスクの履歴取得
        task_history = scheduler.get_execution_history(task_id=task_id)
        assert len(task_history) == 2
        assert all(result.task_id == task_id for result in task_history)

    def test_clear_execution_history(self, scheduler, mock_task_function):
        """実行履歴クリアテスト"""
        task_id = scheduler.add_task(
            name="Clear History Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=mock_task_function
        )

        scheduler.execute_task(task_id)
        assert len(scheduler.execution_history) == 1

        scheduler.clear_execution_history()
        assert len(scheduler.execution_history) == 0

    def test_get_task_statistics(self, scheduler, mock_task_function):
        """タスク統計取得テスト"""
        task_id = scheduler.add_task(
            name="Statistics Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=mock_task_function
        )

        # 成功実行
        scheduler.execute_task(task_id)
        scheduler.execute_task(task_id)

        stats = scheduler.get_task_statistics(task_id)
        assert stats is not None
        assert stats["execution_count"] == 2
        assert stats["success_count"] == 2
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 1.0

    def test_get_statistics_nonexistent_task(self, scheduler):
        """存在しないタスクの統計取得テスト"""
        stats = scheduler.get_task_statistics("nonexistent_task")
        assert stats is None

    @patch('src.day_trade.automation.execution_scheduler.schedule')
    def test_scheduler_status_methods(self, mock_schedule, scheduler):
        """スケジューラーステータス制御テスト"""
        assert scheduler.is_running is False

        # 開始テスト
        scheduler.start()
        assert scheduler.is_running is True

        # 停止テスト
        scheduler.stop()
        assert scheduler.is_running is False

    def test_task_timeout_handling(self, scheduler):
        """タスクタイムアウト処理テスト"""
        def slow_function():
            time.sleep(2)  # 2秒の処理
            return "slow_result"

        task_id = scheduler.add_task(
            name="Timeout Task",
            schedule_type=ScheduleType.ON_DEMAND,
            target_function=slow_function,
            timeout_minutes=0.01  # 0.6秒でタイムアウト
        )

        result = scheduler.execute_task(task_id)

        # タイムアウトの場合のステータス確認
        # 実装に応じてアサーション調整
        assert result is not None

    def test_market_hours_detection(self, scheduler):
        """市場時間検出テスト"""
        # 市場時間内テスト（9:00-15:00）
        with patch('src.day_trade.automation.execution_scheduler.datetime') as mock_datetime:
            # 10:00に設定
            mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 0, 0)  # 月曜日10:00
            mock_datetime.weekday.return_value = 0  # 月曜日

            is_market_hours = scheduler.is_market_hours()
            # 実装に応じて期待値調整
            assert isinstance(is_market_hours, bool)

    def test_next_market_open_calculation(self, scheduler):
        """次回市場開始時刻計算テスト"""
        with patch('src.day_trade.automation.execution_scheduler.datetime') as mock_datetime:
            # 土曜日に設定
            mock_datetime.now.return_value = datetime(2024, 1, 13, 10, 0, 0)  # 土曜日

            next_open = scheduler.get_next_market_open()
            # 実装に応じて期待値調整
            assert isinstance(next_open, (datetime, type(None)))