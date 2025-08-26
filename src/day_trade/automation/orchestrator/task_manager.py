"""
タスク管理・並列実行エンジンモジュール
並列処理、フォールバック、リソース管理を担当
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...utils.logging_config import get_context_logger
from .config import OrchestrationConfig

logger = get_context_logger(__name__)

# 並列処理システムのインポート試行
try:
    from ...utils.parallel_executor_manager import (
        ExecutionResult,
        ExecutorType,
        ParallelExecutorManager,
        TaskType,
    )
    PARALLEL_EXECUTOR_AVAILABLE = True
except ImportError:
    # フォールバック用ダミークラス
    class ExecutionResult:
        def __init__(self, task_id: str, result: Any, execution_time_ms: float, 
                     executor_type: str, success: bool, error: Optional[Exception] = None):
            self.task_id = task_id
            self.result = result
            self.execution_time_ms = execution_time_ms
            self.executor_type = executor_type
            self.success = success
            self.error = error

    class ExecutorType:
        THREAD_POOL = "thread_pool"
        PROCESS_POOL = "process_pool"
        
    class TaskType:
        IO_BOUND = "io_bound"
        CPU_BOUND = "cpu_bound"
        MIXED = "mixed"

    PARALLEL_EXECUTOR_AVAILABLE = False


class TaskManager:
    """
    タスク管理・並列実行マネージャー
    
    高度な並列処理システムとフォールバック機能を提供し、
    効率的なタスクスケジューリングと実行を行います。
    """

    def __init__(self, config: OrchestrationConfig):
        """
        初期化
        
        Args:
            config: オーケストレーション設定
        """
        self.config = config
        
        # 並列実行マネージャー初期化
        if (self.config.enable_parallel_optimization and 
            PARALLEL_EXECUTOR_AVAILABLE):
            try:
                self.parallel_manager = ParallelExecutorManager(
                    max_thread_workers=self.config.max_thread_workers,
                    max_process_workers=self.config.max_process_workers,
                    enable_adaptive_sizing=True,
                    performance_monitoring=self.config.enable_performance_monitoring,
                )
                logger.info(
                    f"並列実行最適化有効: Thread={self.config.max_thread_workers}, "
                    f"Process={self.config.max_process_workers}"
                )
            except Exception as e:
                logger.warning(f"並列マネージャー初期化失敗: {e}")
                self.parallel_manager = None
        else:
            self.parallel_manager = None
            
        if not PARALLEL_EXECUTOR_AVAILABLE:
            logger.warning(
                "ParallelExecutorManagerが利用できません。レガシー並列処理を使用"
            )

    def execute_parallel_analysis(
        self,
        symbols: List[str],
        analysis_functions: List[Tuple[Callable, Dict[str, Any]]],
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, List[ExecutionResult]]:
        """
        並列分析実行
        
        CPU/I/Oバウンドタスクを適切に分離して効率的に並列実行します。
        
        Args:
            symbols: 分析対象銘柄
            analysis_functions: (関数, 引数辞書) のリスト
            max_concurrent: 最大同時実行数
            
        Returns:
            Dict[str, List[ExecutionResult]]: シンボル別実行結果辞書
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
        """
        シーケンシャル実行フォールバック
        
        並列処理が利用できない場合のフォールバック処理を実行します。
        
        Args:
            symbols: 分析対象銘柄
            analysis_functions: (関数, 引数辞書) のリスト
            
        Returns:
            Dict[str, List[ExecutionResult]]: シンボル別実行結果辞書
        """
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
        batch_data: Dict[str, Any],
        analysis_type: str,
        include_predictions: bool,
        analysis_func: Callable,
    ) -> Dict[str, Dict]:
        """
        並列AI分析実行
        
        Args:
            symbols: 分析対象銘柄リスト
            batch_data: バッチデータ
            analysis_type: 分析タイプ
            include_predictions: 予測分析を含むか
            analysis_func: 実行する分析関数
            
        Returns:
            Dict[str, Dict]: 分析結果辞書
        """
        results = {}

        if len(symbols) > 1:  # 複数銘柄の場合は並列実行
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # 分析タスク投入
                future_to_symbol = {
                    executor.submit(
                        analysis_func,
                        symbol,
                        batch_data.get(symbol),
                        analysis_type,
                        include_predictions,
                    ): symbol
                    for symbol in symbols
                }

                # 結果収集
                for future in as_completed(
                    future_to_symbol, timeout=self.config.timeout_seconds
                ):
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
        else:
            # 単一銘柄の場合は逐次実行
            results = self.execute_sequential_ai_analysis(
                symbols, batch_data, analysis_type, include_predictions, analysis_func
            )

        return results

    def execute_sequential_ai_analysis(
        self,
        symbols: List[str],
        batch_data: Dict[str, Any],
        analysis_type: str,
        include_predictions: bool,
        analysis_func: Callable,
    ) -> Dict[str, Dict]:
        """
        逐次AI分析実行
        
        Args:
            symbols: 分析対象銘柄リスト  
            batch_data: バッチデータ
            analysis_type: 分析タイプ
            include_predictions: 予測分析を含むか
            analysis_func: 実行する分析関数
            
        Returns:
            Dict[str, Dict]: 分析結果辞書
        """
        results = {}

        for symbol in symbols:
            try:
                result = analysis_func(
                    symbol, batch_data.get(symbol), analysis_type, include_predictions
                )
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

    def cleanup(self) -> Dict[str, Any]:
        """
        タスクマネージャーのクリーンアップ
        
        Returns:
            Dict[str, Any]: クリーンアップ結果サマリー
        """
        cleanup_summary = {
            "parallel_manager": False,
            "errors": []
        }

        try:
            # 並列マネージャークリーンアップ
            if hasattr(self, 'parallel_manager') and self.parallel_manager:
                try:
                    self.parallel_manager.shutdown()
                    self.parallel_manager = None
                    cleanup_summary["parallel_manager"] = True
                    logger.debug("並列マネージャー クリーンアップ完了")
                except Exception as e:
                    error_msg = f"並列マネージャー クリーンアップエラー: {e}"
                    logger.warning(error_msg)
                    cleanup_summary["errors"].append(error_msg)

        except Exception as e:
            error_msg = f"TaskManager クリーンアップ致命的エラー: {e}"
            logger.error(error_msg)
            cleanup_summary["errors"].append(error_msg)

        return cleanup_summary

    def get_status(self) -> Dict[str, Any]:
        """
        タスクマネージャーのステータス取得
        
        Returns:
            Dict[str, Any]: ステータス情報
        """
        status = {
            "parallel_manager_available": self.parallel_manager is not None,
            "parallel_executor_available": PARALLEL_EXECUTOR_AVAILABLE,
            "max_workers": self.config.max_workers,
            "max_thread_workers": self.config.max_thread_workers,
            "max_process_workers": self.config.max_process_workers,
            "timeout_seconds": self.config.timeout_seconds,
        }
        
        if self.parallel_manager:
            try:
                perf_stats = self.parallel_manager.get_performance_stats()
                status["performance_stats"] = perf_stats
            except Exception as e:
                status["performance_stats_error"] = str(e)
                
        return status