"""
進捗状況表示ユーティリティ
Rich ライブラリを使用した美しい進捗表示機能を提供
"""

import logging
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)
console = Console()


class ProgressType(Enum):
    """進捗表示タイプ"""

    DETERMINATE = "determinate"  # 確定的（進捗率が分かる）
    INDETERMINATE = "indeterminate"  # 不確定的（スピナー表示）
    MULTI_STEP = "multi_step"  # 複数ステップ
    BATCH = "batch"  # バッチ処理


class ProgressUpdater:
    """進捗更新クラス"""

    def __init__(self, progress: Progress, task_id: int):
        self.progress = progress
        self.task_id = task_id

    def update(self, advance: int = 1, description: Optional[str] = None):
        """進捗を更新"""
        if description:
            self.progress.update(self.task_id, description=description, advance=advance)
        else:
            self.progress.update(self.task_id, advance=advance)

    def set_total(self, total: int):
        """合計値を設定"""
        self.progress.update(self.task_id, total=total)

    def complete(self):
        """完了状態に設定"""
        self.progress.update(self.task_id, completed=True)

    def set_description(self, description: str):
        """説明を設定"""
        self.progress.update(self.task_id, description=description)


@contextmanager
def progress_context(
    description: str,
    total: Optional[int] = None,
    progress_type: ProgressType = ProgressType.INDETERMINATE,
):
    """
    進捗表示のコンテキストマネージャー

    Args:
        description: 作業の説明
        total: 総作業量（確定的進捗の場合）
        progress_type: 進捗表示タイプ

    Yields:
        ProgressUpdater: 進捗更新オブジェクト

    Example:
        with progress_context("データ処理中", total=100) as progress:
            for i in range(100):
                # 何らかの処理
                progress.update(1)
    """

    if progress_type == ProgressType.DETERMINATE and total:
        # 確定的進捗（プログレスバー）
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        ) as progress:
            task = progress.add_task(description, total=total)
            yield ProgressUpdater(progress, task)

    elif progress_type == ProgressType.INDETERMINATE:
        # 不確定的進捗（スピナー）
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(description)
            yield ProgressUpdater(progress, task)

    elif progress_type == ProgressType.MULTI_STEP:
        # マルチステップ進捗
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        ) as progress:
            task = progress.add_task(description, total=total or 1)
            yield ProgressUpdater(progress, task)

    else:
        # バッチ処理進捗（シンプル）
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(description, total=total or 100)
            yield ProgressUpdater(progress, task)


class BatchProgressTracker:
    """バッチ処理用進捗トラッカー"""

    def __init__(self, batch_name: str, total_items: int):
        self.batch_name = batch_name
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.progress = None
        self.task_id = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        self.progress.__enter__()
        self.task_id = self.progress.add_task(
            f"{self.batch_name} 処理中...", total=self.total_items
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)

    def update_success(self, item_name: Optional[str] = None):
        """成功時の更新"""
        self.processed_items += 1
        description = f"{self.batch_name} 処理中..."
        if item_name:
            description += f" ({item_name})"

        self.progress.update(self.task_id, advance=1, description=description)

    def update_failure(
        self, item_name: Optional[str] = None, error: Optional[str] = None
    ):
        """失敗時の更新"""
        self.processed_items += 1
        self.failed_items += 1
        description = f"{self.batch_name} 処理中... (エラー: {error or '不明'})"
        if item_name:
            description += f" ({item_name})"

        self.progress.update(self.task_id, advance=1, description=description)

        if error:
            logger.error(f"バッチ処理エラー [{item_name}]: {error}")

    def get_summary(self) -> Dict[str, Any]:
        """処理サマリーを取得"""
        elapsed_time = time.time() - self.start_time
        success_rate = (
            (self.processed_items - self.failed_items) / self.processed_items * 100
            if self.processed_items > 0
            else 0
        )

        return {
            "batch_name": self.batch_name,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "success_rate": success_rate,
            "elapsed_time": elapsed_time,
            "items_per_second": (
                self.processed_items / elapsed_time if elapsed_time > 0 else 0
            ),
        }


class MultiStepProgressTracker:
    """マルチステップ進捗トラッカー"""

    def __init__(self, steps: List[str], overall_description: str = "処理中"):
        self.steps = steps
        self.overall_description = overall_description
        self.current_step = 0
        self.progress = None
        self.task_id = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        )
        self.progress.__enter__()
        self.task_id = self.progress.add_task(
            f"{self.overall_description}: {self.steps[0]}", total=len(self.steps)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)

    def next_step(self, custom_message: Optional[str] = None):
        """次のステップに進む"""
        if self.current_step < len(self.steps):
            self.progress.update(self.task_id, advance=1)
            self.current_step += 1

            if self.current_step < len(self.steps):
                next_step_msg = custom_message or self.steps[self.current_step]
                self.progress.update(
                    self.task_id,
                    description=f"{self.overall_description}: {next_step_msg}",
                )

    def complete(self):
        """全ステップ完了"""
        remaining = len(self.steps) - self.current_step
        if remaining > 0:
            self.progress.update(self.task_id, advance=remaining)

        self.progress.update(
            self.task_id, description=f"{self.overall_description}: 完了"
        )

    def complete_step(self):
        """現在のステップを完了して次に進む"""
        self.next_step()


def show_progress_summary(
    results: List[Dict[str, Any]], title: str = "処理結果サマリー"
):
    """処理結果のサマリーを表示"""
    from rich.panel import Panel
    from rich.table import Table

    if not results:
        console.print(Panel("処理結果がありません", title=title))
        return

    table = Table(title=title)
    table.add_column("項目", style="cyan")
    table.add_column("値", style="white")

    # 統計計算
    total_processed = sum(r.get("processed_items", 0) for r in results)
    total_failed = sum(r.get("failed_items", 0) for r in results)
    total_elapsed = sum(r.get("elapsed_time", 0) for r in results)
    avg_success_rate = sum(r.get("success_rate", 0) for r in results) / len(results)

    table.add_row("総処理件数", f"{total_processed:,}")
    table.add_row("失敗件数", f"{total_failed:,}")
    table.add_row("成功率", f"{avg_success_rate:.1f}%")
    table.add_row("総処理時間", f"{total_elapsed:.2f}秒")
    if total_elapsed > 0:
        table.add_row("処理速度", f"{total_processed / total_elapsed:.1f}件/秒")

    console.print(table)


# 便利な関数
def simple_progress(func: Callable, *args, description: str = "処理中", **kwargs):
    """
    簡単な進捗表示付き関数実行

    Args:
        func: 実行する関数
        *args: 関数の引数
        description: 進捗表示の説明
        **kwargs: 関数のキーワード引数

    Returns:
        関数の実行結果
    """
    with progress_context(description, progress_type=ProgressType.INDETERMINATE):
        return func(*args, **kwargs)


@contextmanager
def multi_step_progress(description: str, steps: List[str]):
    """
    マルチステップ進捗表示のコンテキストマネージャー

    Args:
        description: 全体の説明
        steps: ステップ名のリスト

    Yields:
        MultiStepProgressTracker: マルチステップ進捗トラッカー

    Example:
        steps = ["データロード", "分析実行", "結果保存"]
        with multi_step_progress("バックテスト実行", steps) as progress:
            # ステップ1
            progress.next_step()
            # ステップ2
            progress.next_step()
            # 完了
            progress.complete()
    """
    tracker = MultiStepProgressTracker(steps, description)
    with tracker:
        yield tracker
