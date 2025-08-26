"""
並列処理管理
高度な並列AI分析パイプライン処理を管理
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...utils.logging_config import get_context_logger

# 並列処理システム
try:
    from ...utils.parallel_executor_manager import (
        ExecutionResult,
        ExecutorType,
        ParallelExecutorManager,
        TaskType,
    )
    PARALLEL_EXECUTOR_AVAILABLE = True
    CONCURRENT_AVAILABLE = True
except ImportError:
    # フォールバック用レガシー並列処理
    from concurrent.futures import ThreadPoolExecutor, as_completed
    PARALLEL_EXECUTOR_AVAILABLE = False
    CONCURRENT_AVAILABLE = True
    # ダミークラス定義
    class ExecutionResult:
        def __init__(self, task_id, result, execution_time_ms, executor_type, success, error):
            self.task_id = task_id
            self.result = result
            self.execution_time_ms = execution_time_ms
            self.executor_type = executor_type
            self.success = success
            self.error = error
    
    class ExecutorType:
        THREAD_POOL = "thread_pool"

logger = get_context_logger(__name__)


class ParallelProcessor:
    """並列処理プロセッサークラス"""

    def __init__(self, config, parallel_manager=None):
        """
        初期化

        Args:
            config: オーケストレーション設定
            parallel_manager: 並列実行マネージャー
        """
        self.config = config
        self.parallel_manager = parallel_manager

    def execute_parallel_analysis(
        self,
        symbols: List[str],
        analysis_functions: List[Tuple[Callable, Dict[str, Any]]],
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, List[ExecutionResult]]:
        """
        並列分析実行

        CPU/I/Oバウンドタスクを適切に分離して効率的に並列実行

        Args:
            symbols: 分析対象銘柄
            analysis_functions: (関数, 引数辞書) のリスト
            max_concurrent: 最大同時実行数

        Returns:
            シンボル別実行結果辞書
        """
        if not self.parallel_manager:
            logger.warning(
                "並列マネージャーが無効です。シーケンシャル実行にフォールバック"
            )
            return self._execute_sequential_fallback(symbols, analysis_functions)

        results = {}
        all_tasks = []

        # 各銘柄×各分析関数の組み合わせでタスクを生成
        for symbol in symbols:
            symbol_tasks = []

            for analysis_func, kwargs in analysis_functions:
                # シンボル固有の引数を設定
                task_kwargs = kwargs.copy()
                task_kwargs["symbol"] = symbol

                # タスクタイプをヒント
                if (
                    "fetch" in analysis_func.__name__
                    or "download" in analysis_func.__name__
                ):
                    task_type = TaskType.IO_BOUND
                elif (
                    "compute" in analysis_func.__name__
                    or "calculate" in analysis_func.__name__
                ):
                    task_type = TaskType.CPU_BOUND
                else:
                    task_type = TaskType.MIXED

                task = (analysis_func, (), task_kwargs)
                all_tasks.append((symbol, analysis_func.__name__, task))
                symbol_tasks.append(task)

            results[symbol] = []

        # バッチ実行
        batch_tasks = [task for _, _, task in all_tasks]
        execution_results = self.parallel_manager.execute_batch(
            batch_tasks, max_concurrent=max_concurrent or self.config.max_workers
        )

        # 結果を銘柄別に整理
        for (symbol, func_name, _), exec_result in zip(all_tasks, execution_results):
            results[symbol].append(exec_result)

        # 統計情報をログ出力
        successful_tasks = sum(1 for r in execution_results if r.success)
        total_tasks = len(execution_results)

        logger.info(f"並列分析完了: {successful_tasks}/{total_tasks} 成功")
        if self.parallel_manager:
            perf_stats = self.parallel_manager.get_performance_stats()
            for executor_name, stats in perf_stats.items():
                logger.info(
                    f"{executor_name}: 平均時間={stats['average_time_ms']:.1f}ms, "
                    f"成功率={stats['success_rate']:.1%}"
                )

        return results

    def _execute_sequential_fallback(
        self,
        symbols: List[str],
        analysis_functions: List[Tuple[Callable, Dict[str, Any]]],
    ) -> Dict[str, List[ExecutionResult]]:
        """シーケンシャル実行フォールバック"""
        results = {}

        for symbol in symbols:
            symbol_results = []

            for analysis_func, kwargs in analysis_functions:
                task_kwargs = kwargs.copy()
                task_kwargs["symbol"] = symbol

                start_time = time.perf_counter()
                try:
                    result = analysis_func(**task_kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = e
                    logger.error(f"Sequential execution failed for {symbol}: {e}")

                execution_time = (time.perf_counter() - start_time) * 1000

                exec_result = ExecutionResult(
                    task_id=f"{symbol}_{analysis_func.__name__}",
                    result=result,
                    execution_time_ms=execution_time,
                    executor_type=ExecutorType.THREAD_POOL,  # フォールバック
                    success=success,
                    error=error,
                )

                symbol_results.append(exec_result)

            results[symbol] = symbol_results

        return results

    def execute_parallel_ai_analysis(
        self,
        symbols: List[str],
        analysis_func: Callable,
        max_workers: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """
        並列AI分析実行

        Args:
            symbols: 分析対象銘柄リスト
            analysis_func: 分析関数
            max_workers: 最大ワーカー数
            timeout: タイムアウト時間

        Returns:
            シンボル別分析結果辞書
        """
        results = {}
        max_workers = max_workers or self.config.max_workers
        timeout = timeout or self.config.timeout_seconds

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 分析タスク投入
            future_to_symbol = {
                executor.submit(analysis_func, symbol): symbol
                for symbol in symbols
            }

            # 結果収集
            for future in as_completed(future_to_symbol, timeout=timeout):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=60)
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"並列AI分析エラー {symbol}: {e}")
                    results[symbol] = {
                        "success": False,
                        "errors": [str(e)],
                        "analysis": None,
                        "signals": [],
                        "alerts": [],
                    }

        return results

    def execute_sequential_ai_analysis(
        self, symbols: List[str], analysis_func: Callable
    ) -> Dict[str, Dict]:
        """
        逐次AI分析実行

        Args:
            symbols: 分析対象銘柄リスト
            analysis_func: 分析関数

        Returns:
            シンボル別分析結果辞書
        """
        results = {}

        for symbol in symbols:
            try:
                result = analysis_func(symbol)
                results[symbol] = result
            except Exception as e:
                logger.error(f"逐次AI分析エラー {symbol}: {e}")
                results[symbol] = {
                    "success": False,
                    "errors": [str(e)],
                    "analysis": None,
                    "signals": [],
                    "alerts": [],
                }

        return results

    @staticmethod
    def is_parallel_available() -> bool:
        """並列処理機能が利用可能かチェック"""
        return PARALLEL_EXECUTOR_AVAILABLE and CONCURRENT_AVAILABLE