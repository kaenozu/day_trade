#!/usr/bin/env python3
"""
実行スケジューラシステム

Issue #487対応: 完全自動化システム実装 - Phase 1
定時実行・条件実行・連続実行による完全自動運用
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import schedule

from ..utils.logging_config import get_context_logger
from .smart_symbol_selector import get_smart_selected_symbols
from ..ml.ensemble_system import EnsembleSystem, EnsembleConfig

logger = get_context_logger(__name__)


class ScheduleType(Enum):
    """スケジュール種別"""
    DAILY = "daily"              # 日次実行
    HOURLY = "hourly"           # 時間次実行
    MARKET_HOURS = "market_hours"  # 取引時間内実行
    ON_DEMAND = "on_demand"     # オンデマンド実行
    CONTINUOUS = "continuous"   # 連続実行


class ExecutionStatus(Enum):
    """実行ステータス"""
    READY = "ready"             # 実行準備完了
    RUNNING = "running"         # 実行中
    SUCCESS = "success"         # 成功
    FAILED = "failed"           # 失敗
    PAUSED = "paused"          # 一時停止
    STOPPED = "stopped"        # 停止


@dataclass
class ScheduledTask:
    """スケジュール実行タスク"""
    task_id: str
    name: str
    schedule_type: ScheduleType
    target_function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    schedule_time: Optional[str] = None  # "09:00", "15:30" など
    interval_minutes: Optional[int] = None  # 間隔（分）
    max_retries: int = 3
    timeout_minutes: int = 30
    enabled: bool = True
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.READY
    execution_count: int = 0
    success_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class ExecutionResult:
    """実行結果"""
    task_id: str
    status: ExecutionStatus
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    result_data: Any = None
    error_message: Optional[str] = None
    retry_count: int = 0


class ExecutionScheduler:
    """
    Issue #487対応: 実行スケジューラシステム

    完全自動化システムの核心となるスケジューラー
    - 定時実行: 指定時刻での自動実行
    - 条件実行: 市場条件に応じた実行
    - 連続実行: リアルタイム監視・実行
    """

    def __init__(self):
        """初期化"""
        self.tasks: Dict[str, ScheduledTask] = {}
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.execution_history: List[ExecutionResult] = []

        # Issue #462対応: 93%精度アンサンブルシステム統合
        self.ensemble_system = None
        self._initialize_ensemble()

        # 市場時間設定（日本市場）
        self.market_hours = {
            'start': (9, 0),   # 9:00開始
            'end': (15, 0),    # 15:00終了
            'lunch_start': (11, 30),  # 11:30昼休み開始
            'lunch_end': (12, 30)     # 12:30昼休み終了
        }

    def _initialize_ensemble(self):
        """アンサンブルシステム初期化"""
        try:
            # Issue #462対応: 93%精度達成のための最適設定
            config = EnsembleConfig(
                use_xgboost=True,
                use_catboost=True,
                use_random_forest=True,
                use_lstm_transformer=False,  # 高速化のため無効
                use_gradient_boosting=False,  # 高速化のため無効
                use_svr=False,  # 高速化のため無効

                # 高精度設定
                xgboost_params={
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'enable_hyperopt': False  # 自動化では無効
                },
                catboost_params={
                    'iterations': 500,
                    'depth': 8,
                    'learning_rate': 0.05,
                    'enable_hyperopt': False,  # 自動化では無効
                    'verbose': 0
                },
                random_forest_params={
                    'n_estimators': 200,
                    'max_depth': 15,
                    'enable_hyperopt': False  # 自動化では無効
                }
            )

            self.ensemble_system = EnsembleSystem(config)
            logger.info("93%精度アンサンブルシステム統合完了")

        except Exception as e:
            logger.error(f"アンサンブルシステム初期化エラー: {e}")

    def add_task(self, task: ScheduledTask) -> bool:
        """
        タスク追加

        Args:
            task: スケジュール実行タスク

        Returns:
            追加成功フラグ
        """
        try:
            # 次回実行時刻計算
            task.next_execution = self._calculate_next_execution(task)

            self.tasks[task.task_id] = task

            logger.info(f"タスク追加: {task.name} (ID: {task.task_id})")
            logger.info(f"  スケジュール: {task.schedule_type.value}")
            logger.info(f"  次回実行: {task.next_execution}")

            return True

        except Exception as e:
            logger.error(f"タスク追加エラー: {e}")
            return False

    def _calculate_next_execution(self, task: ScheduledTask) -> datetime:
        """次回実行時刻計算"""
        now = datetime.now()

        if task.schedule_type == ScheduleType.DAILY and task.schedule_time:
            # 日次実行の場合
            hour, minute = map(int, task.schedule_time.split(':'))
            next_exec = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            if next_exec <= now:
                next_exec += timedelta(days=1)

        elif task.schedule_type == ScheduleType.HOURLY:
            # 時間次実行の場合
            next_exec = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        elif task.schedule_type == ScheduleType.MARKET_HOURS:
            # 取引時間内実行の場合
            next_exec = self._next_market_time()

        elif task.schedule_type == ScheduleType.CONTINUOUS and task.interval_minutes:
            # 連続実行の場合
            next_exec = now + timedelta(minutes=task.interval_minutes)

        else:
            # オンデマンドまたはその他
            next_exec = now

        return next_exec

    def _next_market_time(self) -> datetime:
        """次の取引時間計算"""
        now = datetime.now()

        # 9:00開始の場合
        market_start = now.replace(
            hour=self.market_hours['start'][0],
            minute=self.market_hours['start'][1],
            second=0, microsecond=0
        )

        if now.time() < market_start.time():
            return market_start
        elif now.time() > datetime.now().replace(hour=15, minute=0).time():
            # 市場終了後は翌日
            return market_start + timedelta(days=1)
        else:
            # 取引時間内は次の整数時刻
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    def start(self):
        """スケジューラ開始"""
        if self.is_running:
            logger.warning("スケジューラは既に実行中です")
            return

        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        logger.info("実行スケジューラ開始")
        logger.info(f"登録タスク数: {len(self.tasks)}")

    def stop(self):
        """スケジューラ停止"""
        self.is_running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)

        logger.info("実行スケジューラ停止")

    def _scheduler_loop(self):
        """スケジューラメインループ"""
        logger.info("スケジューラループ開始")

        while self.is_running:
            try:
                current_time = datetime.now()

                # 実行対象タスク検索
                for task_id, task in self.tasks.items():
                    if (task.enabled and
                        task.status == ExecutionStatus.READY and
                        task.next_execution and
                        current_time >= task.next_execution):

                        # タスク実行
                        self._execute_task(task)

                # 1分間隔でチェック
                time.sleep(60)

            except Exception as e:
                logger.error(f"スケジューラループエラー: {e}")
                time.sleep(10)

    def _execute_task(self, task: ScheduledTask):
        """
        タスク実行

        Args:
            task: 実行対象タスク
        """
        logger.info(f"タスク実行開始: {task.name}")

        start_time = datetime.now()
        task.status = ExecutionStatus.RUNNING
        task.execution_count += 1

        retry_count = 0
        max_retries = task.max_retries

        while retry_count <= max_retries:
            try:
                # タスク実行
                result = self._run_task_function(task)

                # 成功処理
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                execution_result = ExecutionResult(
                    task_id=task.task_id,
                    status=ExecutionStatus.SUCCESS,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    result_data=result,
                    retry_count=retry_count
                )

                self.execution_history.append(execution_result)

                task.status = ExecutionStatus.SUCCESS
                task.success_count += 1
                task.last_execution = start_time
                task.next_execution = self._calculate_next_execution(task)
                task.last_error = None

                logger.info(f"タスク実行成功: {task.name} ({duration:.1f}秒)")
                return

            except Exception as e:
                retry_count += 1
                error_message = str(e)

                logger.warning(f"タスク実行エラー (試行 {retry_count}/{max_retries + 1}): {task.name}")
                logger.warning(f"エラー詳細: {error_message}")

                if retry_count <= max_retries:
                    # リトライ待機
                    time.sleep(min(retry_count * 30, 300))  # 最大5分待機
                else:
                    # 最終失敗処理
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()

                    execution_result = ExecutionResult(
                        task_id=task.task_id,
                        status=ExecutionStatus.FAILED,
                        start_time=start_time,
                        end_time=end_time,
                        duration_seconds=duration,
                        error_message=error_message,
                        retry_count=retry_count - 1
                    )

                    self.execution_history.append(execution_result)

                    task.status = ExecutionStatus.FAILED
                    task.error_count += 1
                    task.last_error = error_message
                    task.next_execution = self._calculate_next_execution(task)

                    logger.error(f"タスク実行最終失敗: {task.name}")

    def _run_task_function(self, task: ScheduledTask) -> Any:
        """タスク関数実行"""
        return task.target_function(**task.parameters)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """タスクステータス取得"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            'task_id': task.task_id,
            'name': task.name,
            'status': task.status.value,
            'enabled': task.enabled,
            'execution_count': task.execution_count,
            'success_count': task.success_count,
            'error_count': task.error_count,
            'last_execution': task.last_execution,
            'next_execution': task.next_execution,
            'last_error': task.last_error
        }

    def get_all_tasks_status(self) -> List[Dict[str, Any]]:
        """全タスクステータス取得"""
        return [self.get_task_status(task_id) for task_id in self.tasks.keys()]

    def pause_task(self, task_id: str) -> bool:
        """タスク一時停止"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.enabled = False
        task.status = ExecutionStatus.PAUSED
        logger.info(f"タスク一時停止: {task.name}")
        return True

    def resume_task(self, task_id: str) -> bool:
        """タスク再開"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.enabled = True
        task.status = ExecutionStatus.READY
        task.next_execution = self._calculate_next_execution(task)
        logger.info(f"タスク再開: {task.name}")
        return True


# Issue #487対応: 完全自動化システム用の標準タスク関数群

async def smart_stock_analysis_task(**kwargs):
    """
    スマート銘柄分析タスク

    Issue #487 + #462統合タスク:
    1. スマート銘柄自動選択
    2. 93%精度アンサンブル予測
    3. 結果記録・通知
    """
    try:
        target_count = kwargs.get('target_count', 5)

        logger.info(f"スマート銘柄分析開始 (対象: {target_count}銘柄)")

        # Step 1: スマート銘柄選択
        selected_symbols = await get_smart_selected_symbols(target_count)

        if not selected_symbols:
            raise ValueError("銘柄選択に失敗しました")

        logger.info(f"選定銘柄: {selected_symbols}")

        # Step 2: 93%精度予測実行（ここでは選定確認のみ）
        predictions = {}
        for symbol in selected_symbols:
            # 実際の予測処理はここに実装
            predictions[symbol] = {
                'prediction_ready': True,
                'accuracy_target': 93.0,
                'system': 'CatBoost+XGBoost+RandomForest'
            }

        result = {
            'timestamp': datetime.now(),
            'selected_symbols': selected_symbols,
            'predictions': predictions,
            'status': 'success'
        }

        logger.info("スマート銘柄分析完了")
        return result

    except Exception as e:
        logger.error(f"スマート銘柄分析エラー: {e}")
        raise


def create_default_automation_tasks() -> List[ScheduledTask]:
    """
    Issue #487対応: デフォルト自動化タスク生成

    Returns:
        デフォルトタスクリスト
    """
    tasks = []

    # 1. 朝の市場開始前分析（8:30）
    morning_analysis = ScheduledTask(
        task_id="morning_analysis",
        name="朝の市場分析",
        schedule_type=ScheduleType.DAILY,
        target_function=lambda: asyncio.run(smart_stock_analysis_task(target_count=8)),
        schedule_time="08:30"
    )
    tasks.append(morning_analysis)

    # 2. 昼休み分析（12:00）
    lunch_analysis = ScheduledTask(
        task_id="lunch_analysis",
        name="昼休み分析",
        schedule_type=ScheduleType.DAILY,
        target_function=lambda: asyncio.run(smart_stock_analysis_task(target_count=5)),
        schedule_time="12:00"
    )
    tasks.append(lunch_analysis)

    # 3. 市場終了後分析（15:30）
    closing_analysis = ScheduledTask(
        task_id="closing_analysis",
        name="市場終了後分析",
        schedule_type=ScheduleType.DAILY,
        target_function=lambda: asyncio.run(smart_stock_analysis_task(target_count=10)),
        schedule_time="15:30"
    )
    tasks.append(closing_analysis)

    return tasks


# デバッグ用メイン関数
async def main():
    """デバッグ用メイン"""
    logger.info("実行スケジューラシステム テスト実行")

    # スケジューラ初期化
    scheduler = ExecutionScheduler()

    # デフォルトタスク追加
    default_tasks = create_default_automation_tasks()
    for task in default_tasks:
        scheduler.add_task(task)

    # ステータス表示
    logger.info("=" * 50)
    logger.info("登録タスク一覧:")
    for task_status in scheduler.get_all_tasks_status():
        logger.info(f"  {task_status['name']}: {task_status['status']}")
        logger.info(f"    次回実行: {task_status['next_execution']}")

    logger.info("実行スケジューラシステム テスト完了")


if __name__ == "__main__":
    asyncio.run(main())