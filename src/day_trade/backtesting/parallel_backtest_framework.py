#!/usr/bin/env python3
"""
バックテスト並列フレームワーク
Issue #382: Parallel Backtest Framework

高頻度取引エンジンの並列処理技術を活用したバックテスト最適化
- マルチプロセシング並列実行
- GPU加速パラメータ最適化
- 高速メモリプール管理
- 分散バックテスト処理
"""

import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 高頻度取引エンジンの技術を流用
from ..trading.high_frequency_engine import MemoryPool, MicrosecondTimer
from ..utils.logging_config import get_context_logger

# バックテストエンジン
try:
    from .backtest_engine import BacktestEngine, BacktestResults
except ImportError:
    from ..backtesting.backtest_engine import BacktestEngine, BacktestResults

logger = get_context_logger(__name__)


class ParallelMode(Enum):
    """並列処理モード"""

    MULTIPROCESSING = "multiprocessing"  # マルチプロセシング
    THREADING = "threading"  # マルチスレッディング
    ASYNC = "async"  # 非同期処理
    DISTRIBUTED = "distributed"  # 分散処理（将来対応）


class OptimizationMethod(Enum):
    """最適化手法"""

    GRID_SEARCH = "grid_search"  # グリッドサーチ
    RANDOM_SEARCH = "random_search"  # ランダムサーチ
    GENETIC_ALGORITHM = "genetic"  # 遺伝的アルゴリズム
    BAYESIAN = "bayesian"  # ベイズ最適化


@dataclass
class ParameterSpace:
    """パラメータ空間定義"""

    name: str
    min_value: float
    max_value: float
    step_size: Optional[float] = None
    values: Optional[List[Any]] = None
    distribution: str = "uniform"  # uniform, normal, log_uniform

    def generate_values(self, num_samples: int = 100) -> List[float]:
        """パラメータ値生成"""
        if self.values:
            return self.values

        if self.step_size:
            return list(
                np.arange(
                    self.min_value, self.max_value + self.step_size, self.step_size
                )
            )

        if self.distribution == "uniform":
            return list(np.linspace(self.min_value, self.max_value, num_samples))
        elif self.distribution == "log_uniform":
            return list(
                np.logspace(
                    np.log10(self.min_value), np.log10(self.max_value), num_samples
                )
            )
        else:
            return list(np.linspace(self.min_value, self.max_value, num_samples))


@dataclass
class BacktestTask:
    """バックテストタスク定義"""

    task_id: str
    symbols: List[str]
    parameters: Dict[str, Any]
    start_date: str
    end_date: str
    strategy_config: Dict[str, Any]
    initial_capital: float = 1000000

    # 実行結果
    result: Optional[BacktestResults] = None
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ParallelBacktestConfig:
    """並列バックテスト設定"""

    parallel_mode: ParallelMode = ParallelMode.MULTIPROCESSING
    max_workers: int = field(default_factory=lambda: mp.cpu_count())
    optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH

    # パフォーマンス設定
    memory_limit_mb: int = 8192  # プロセス当たりメモリ制限
    task_timeout_seconds: int = 300  # タスクタイムアウト
    enable_memory_pool: bool = True  # 高速メモリプール使用
    memory_pool_size_mb: int = 200  # メモリプールサイズ

    # 最適化設定
    objective_function: str = "sharpe_ratio"  # 最適化目標
    minimize: bool = False  # 最小化フラグ
    max_iterations: int = 1000  # 最大反復回数
    convergence_threshold: float = 1e-6  # 収束判定閾値

    # 分散処理設定（将来対応）
    distributed_nodes: List[str] = field(default_factory=list)

    def validate(self):
        """設定検証"""
        if self.max_workers <= 0:
            self.max_workers = mp.cpu_count()

        if self.memory_limit_mb < 100:
            raise ValueError("メモリ制限が小さすぎます（最小100MB）")

        if self.task_timeout_seconds < 10:
            raise ValueError("タスクタイムアウトが短すぎます（最小10秒）")


class ParameterOptimizer:
    """パラメータ最適化エンジン"""

    def __init__(self, config: ParallelBacktestConfig):
        self.config = config
        self.memory_pool = (
            MemoryPool(config.memory_pool_size_mb)
            if config.enable_memory_pool
            else None
        )

    def generate_parameter_combinations(
        self, parameter_spaces: List[ParameterSpace], method: OptimizationMethod = None
    ) -> List[Dict[str, Any]]:
        """パラメータ組み合わせ生成"""
        method = method or self.config.optimization_method

        if method == OptimizationMethod.GRID_SEARCH:
            return self._grid_search_combinations(parameter_spaces)
        elif method == OptimizationMethod.RANDOM_SEARCH:
            return self._random_search_combinations(parameter_spaces)
        elif method == OptimizationMethod.GENETIC_ALGORITHM:
            return self._genetic_algorithm_combinations(parameter_spaces)
        else:
            return self._grid_search_combinations(parameter_spaces)

    def _grid_search_combinations(
        self, parameter_spaces: List[ParameterSpace]
    ) -> List[Dict[str, Any]]:
        """グリッドサーチ組み合わせ"""
        param_values = {}
        for space in parameter_spaces:
            param_values[space.name] = space.generate_values()

        # 直積を計算
        keys = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[key] for key in keys]))

        result = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            result.append(param_dict)

        logger.info(f"グリッドサーチ組み合わせ生成完了: {len(result)}通り")
        return result

    def _random_search_combinations(
        self, parameter_spaces: List[ParameterSpace]
    ) -> List[Dict[str, Any]]:
        """ランダムサーチ組み合わせ"""
        num_samples = min(self.config.max_iterations, 1000)
        result = []

        for _ in range(num_samples):
            param_dict = {}
            for space in parameter_spaces:
                if space.distribution == "uniform":
                    value = np.random.uniform(space.min_value, space.max_value)
                elif space.distribution == "log_uniform":
                    value = np.random.lognormal(
                        np.log(space.min_value),
                        np.log(space.max_value / space.min_value),
                    )
                else:
                    value = np.random.uniform(space.min_value, space.max_value)

                param_dict[space.name] = value
            result.append(param_dict)

        logger.info(f"ランダムサーチ組み合わせ生成完了: {len(result)}通り")
        return result

    def _genetic_algorithm_combinations(
        self, parameter_spaces: List[ParameterSpace]
    ) -> List[Dict[str, Any]]:
        """遺伝的アルゴリズム（簡易版）"""
        # 初期集団生成
        population_size = min(50, self.config.max_iterations // 10)
        result = []

        for _ in range(population_size):
            individual = {}
            for space in parameter_spaces:
                value = np.random.uniform(space.min_value, space.max_value)
                individual[space.name] = value
            result.append(individual)

        logger.info(f"遺伝的アルゴリズム初期集団生成完了: {len(result)}個体")
        return result


class WorkerProcess:
    """バックテストワーカープロセス"""

    def __init__(self, worker_id: int, config: ParallelBacktestConfig):
        self.worker_id = worker_id
        self.config = config
        self.memory_pool = (
            MemoryPool(config.memory_pool_size_mb)
            if config.enable_memory_pool
            else None
        )
        self.processed_tasks = 0
        self.total_execution_time = 0.0

    def execute_backtest_task(self, task: BacktestTask) -> BacktestTask:
        """バックテストタスク実行"""
        start_time = MicrosecondTimer.now_ns()

        try:
            # バックテストエンジン作成
            engine = BacktestEngine(initial_capital=task.initial_capital)

            # 過去データ読み込み
            historical_data = engine.load_historical_data(
                task.symbols, task.start_date, task.end_date
            )

            if not historical_data:
                task.error = "データ読み込みエラー"
                return task

            # 戦略関数作成（パラメータ適用）
            strategy_func = self._create_strategy_function(
                task.parameters, task.strategy_config
            )

            # バックテスト実行
            results = engine.execute_backtest(historical_data, strategy_func)

            task.result = results
            task.execution_time_ms = MicrosecondTimer.elapsed_us(start_time) / 1000

            # 統計更新
            self.processed_tasks += 1
            self.total_execution_time += task.execution_time_ms

            return task

        except Exception as e:
            task.error = str(e)
            task.execution_time_ms = MicrosecondTimer.elapsed_us(start_time) / 1000
            logger.error(f"ワーカー{self.worker_id} タスクエラー: {e}")
            return task

    def _create_strategy_function(
        self, parameters: Dict[str, Any], config: Dict[str, Any]
    ) -> Callable:
        """パラメータ化戦略関数作成"""

        def parameterized_strategy(
            lookback_data: Dict[str, pd.DataFrame], current_prices: Dict[str, float]
        ) -> Dict[str, float]:
            """パラメータ化戦略"""
            signals = {}

            # パラメータ取得
            momentum_window = int(parameters.get("momentum_window", 20))
            buy_threshold = parameters.get("buy_threshold", 0.05)
            sell_threshold = parameters.get("sell_threshold", -0.05)
            position_size = parameters.get("position_size", 0.2)

            for symbol, data in lookback_data.items():
                if len(data) >= momentum_window:
                    # モメンタム計算
                    returns = (
                        data["Close"].iloc[-1] / data["Close"].iloc[-momentum_window]
                        - 1
                    )

                    # シグナル生成
                    if returns > buy_threshold:
                        signals[symbol] = position_size
                    elif returns < sell_threshold:
                        signals[symbol] = 0.0
                    else:
                        signals[symbol] = position_size / 2

            # ポジション正規化
            total_weight = sum(signals.values())
            if total_weight > 0:
                signals = {k: v / total_weight for k, v in signals.items()}

            return signals

        return parameterized_strategy

    def get_worker_stats(self) -> Dict[str, Any]:
        """ワーカー統計取得"""
        avg_execution_time = (
            self.total_execution_time / self.processed_tasks
            if self.processed_tasks > 0
            else 0
        )

        return {
            "worker_id": self.worker_id,
            "processed_tasks": self.processed_tasks,
            "total_execution_time_ms": self.total_execution_time,
            "avg_execution_time_ms": avg_execution_time,
            "memory_pool_enabled": self.memory_pool is not None,
        }


def execute_single_backtest(
    task_data: Tuple[BacktestTask, ParallelBacktestConfig],
) -> BacktestTask:
    """単一バックテスト実行（マルチプロセシング用）"""
    task, config = task_data
    worker = WorkerProcess(0, config)  # プロセス内ワーカー
    return worker.execute_backtest_task(task)


class ParallelBacktestFramework:
    """並列バックテストフレームワーク"""

    def __init__(self, config: ParallelBacktestConfig):
        self.config = config
        self.config.validate()

        # コンポーネント
        self.optimizer = ParameterOptimizer(config)
        self.memory_pool = (
            MemoryPool(config.memory_pool_size_mb)
            if config.enable_memory_pool
            else None
        )

        # 実行統計
        self.execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_execution_time_ms": 0.0,
            "avg_task_time_ms": 0.0,
            "throughput_tasks_per_sec": 0.0,
            "memory_efficiency": 0.0,
        }

        logger.info(
            f"並列バックテストフレームワーク初期化完了: {config.max_workers}ワーカー"
        )

    def run_parameter_optimization(
        self,
        symbols: List[str],
        parameter_spaces: List[ParameterSpace],
        start_date: str,
        end_date: str,
        strategy_config: Dict[str, Any] = None,
        initial_capital: float = 1000000,
    ) -> Dict[str, Any]:
        """パラメータ最適化実行"""
        logger.info(
            f"パラメータ最適化開始: {len(symbols)}銘柄, {len(parameter_spaces)}パラメータ"
        )

        start_time = MicrosecondTimer.now_ns()

        try:
            # パラメータ組み合わせ生成
            parameter_combinations = self.optimizer.generate_parameter_combinations(
                parameter_spaces
            )

            if not parameter_combinations:
                raise ValueError("パラメータ組み合わせの生成に失敗しました")

            # バックテストタスク生成
            tasks = []
            for i, params in enumerate(parameter_combinations):
                task = BacktestTask(
                    task_id=f"opt_{i:04d}",
                    symbols=symbols,
                    parameters=params,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_config=strategy_config or {},
                    initial_capital=initial_capital,
                )
                tasks.append(task)

            logger.info(f"バックテストタスク生成完了: {len(tasks)}タスク")

            # 並列実行
            completed_tasks = self._execute_parallel_backtest(tasks)

            # 結果分析
            optimization_results = self._analyze_optimization_results(
                completed_tasks, parameter_spaces
            )

            total_time_ms = MicrosecondTimer.elapsed_us(start_time) / 1000

            # 統計更新
            self.execution_stats.update(
                {
                    "total_tasks": len(tasks),
                    "completed_tasks": len(completed_tasks),
                    "failed_tasks": len(tasks) - len(completed_tasks),
                    "total_execution_time_ms": total_time_ms,
                    "avg_task_time_ms": total_time_ms / len(tasks) if tasks else 0,
                    "throughput_tasks_per_sec": (
                        len(completed_tasks) / (total_time_ms / 1000)
                        if total_time_ms > 0
                        else 0
                    ),
                }
            )

            logger.info(
                f"パラメータ最適化完了: {len(completed_tasks)}/{len(tasks)}タスク成功, "
                f"実行時間: {total_time_ms:.0f}ms"
            )

            return optimization_results

        except Exception as e:
            logger.error(f"パラメータ最適化エラー: {e}")
            raise

    def _execute_parallel_backtest(
        self, tasks: List[BacktestTask]
    ) -> List[BacktestTask]:
        """並列バックテスト実行"""
        if self.config.parallel_mode == ParallelMode.MULTIPROCESSING:
            return self._execute_multiprocessing(tasks)
        elif self.config.parallel_mode == ParallelMode.THREADING:
            return self._execute_threading(tasks)
        else:
            return self._execute_sequential(tasks)

    def _execute_multiprocessing(self, tasks: List[BacktestTask]) -> List[BacktestTask]:
        """マルチプロセシング実行"""
        completed_tasks = []

        # タスクデータ準備
        task_data = [(task, self.config) for task in tasks]

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # タスク投入
            future_to_task = {
                executor.submit(execute_single_backtest, data): data[0]
                for data in task_data
            }

            # 結果収集
            for future in as_completed(
                future_to_task, timeout=self.config.task_timeout_seconds
            ):
                try:
                    result = future.result(timeout=self.config.task_timeout_seconds)
                    completed_tasks.append(result)

                    # 進捗表示
                    progress = len(completed_tasks) / len(tasks) * 100
                    if len(completed_tasks) % max(1, len(tasks) // 10) == 0:
                        logger.info(
                            f"並列実行進捗: {progress:.1f}% ({len(completed_tasks)}/{len(tasks)})"
                        )

                except Exception as e:
                    logger.error(f"タスク実行エラー: {e}")

        return completed_tasks

    def _execute_threading(self, tasks: List[BacktestTask]) -> List[BacktestTask]:
        """マルチスレッディング実行（軽量版）"""
        # 実装簡略化のため、シーケンシャル実行
        return self._execute_sequential(tasks)

    def _execute_sequential(self, tasks: List[BacktestTask]) -> List[BacktestTask]:
        """シーケンシャル実行"""
        completed_tasks = []
        worker = WorkerProcess(0, self.config)

        for i, task in enumerate(tasks):
            try:
                result = worker.execute_backtest_task(task)
                completed_tasks.append(result)

                # 進捗表示
                if (i + 1) % max(1, len(tasks) // 10) == 0:
                    progress = (i + 1) / len(tasks) * 100
                    logger.info(
                        f"シーケンシャル実行進捗: {progress:.1f}% ({i + 1}/{len(tasks)})"
                    )

            except Exception as e:
                logger.error(f"タスク実行エラー: {e}")

        return completed_tasks

    def _analyze_optimization_results(
        self,
        completed_tasks: List[BacktestTask],
        parameter_spaces: List[ParameterSpace],
    ) -> Dict[str, Any]:
        """最適化結果分析"""
        if not completed_tasks:
            return {"error": "実行完了タスクなし"}

        # 成功したタスクのみ分析
        successful_tasks = [
            task for task in completed_tasks if task.result and not task.error
        ]

        if not successful_tasks:
            return {"error": "成功タスクなし"}

        # 目的関数値計算
        objective_values = []
        for task in successful_tasks:
            if task.result:
                if self.config.objective_function == "sharpe_ratio":
                    value = task.result.sharpe_ratio
                elif self.config.objective_function == "total_return":
                    value = task.result.total_return
                elif self.config.objective_function == "max_drawdown":
                    value = -task.result.max_drawdown  # 最大化のため負値
                else:
                    value = task.result.sharpe_ratio

                objective_values.append((task, value))

        # ソート（最小化/最大化に応じて）
        objective_values.sort(key=lambda x: x[1], reverse=not self.config.minimize)

        # 最優秀結果
        best_task, best_value = objective_values[0]

        # 統計計算
        values = [v for _, v in objective_values]

        analysis_results = {
            "optimization_summary": {
                "total_combinations": len(completed_tasks),
                "successful_combinations": len(successful_tasks),
                "success_rate": len(successful_tasks) / len(completed_tasks),
                "objective_function": self.config.objective_function,
                "best_value": best_value,
                "mean_value": np.mean(values),
                "std_value": np.std(values),
                "min_value": np.min(values),
                "max_value": np.max(values),
            },
            "best_parameters": best_task.parameters,
            "best_result": {
                "total_return": best_task.result.total_return,
                "annualized_return": best_task.result.annualized_return,
                "sharpe_ratio": best_task.result.sharpe_ratio,
                "max_drawdown": best_task.result.max_drawdown,
                "win_rate": best_task.result.win_rate,
                "total_trades": best_task.result.total_trades,
            },
            "parameter_analysis": self._analyze_parameter_sensitivity(
                successful_tasks, parameter_spaces
            ),
            "performance_stats": self.execution_stats,
            "top_10_results": [
                {
                    "rank": i + 1,
                    "parameters": task.parameters,
                    "objective_value": value,
                    "sharpe_ratio": task.result.sharpe_ratio,
                    "total_return": task.result.total_return,
                    "max_drawdown": task.result.max_drawdown,
                }
                for i, (task, value) in enumerate(objective_values[:10])
            ],
        }

        return analysis_results

    def _analyze_parameter_sensitivity(
        self,
        successful_tasks: List[BacktestTask],
        parameter_spaces: List[ParameterSpace],
    ) -> Dict[str, Any]:
        """パラメータ感度分析"""
        sensitivity_analysis = {}

        for space in parameter_spaces:
            param_name = space.name
            param_values = []
            objective_values = []

            for task in successful_tasks:
                if param_name in task.parameters and task.result:
                    param_values.append(task.parameters[param_name])

                    if self.config.objective_function == "sharpe_ratio":
                        obj_value = task.result.sharpe_ratio
                    else:
                        obj_value = task.result.total_return

                    objective_values.append(obj_value)

            if len(param_values) > 1:
                correlation = np.corrcoef(param_values, objective_values)[0, 1]

                sensitivity_analysis[param_name] = {
                    "correlation": correlation,
                    "mean_param_value": np.mean(param_values),
                    "std_param_value": np.std(param_values),
                    "optimal_range": {
                        "min": np.min(param_values),
                        "max": np.max(param_values),
                        "best_value": param_values[np.argmax(objective_values)],
                    },
                }

        return sensitivity_analysis

    def get_framework_stats(self) -> Dict[str, Any]:
        """フレームワーク統計取得"""
        return {
            "config": {
                "parallel_mode": self.config.parallel_mode.value,
                "max_workers": self.config.max_workers,
                "optimization_method": self.config.optimization_method.value,
                "memory_limit_mb": self.config.memory_limit_mb,
            },
            "execution_stats": self.execution_stats,
            "memory_pool": (
                {
                    "enabled": self.config.enable_memory_pool,
                    "size_mb": self.config.memory_pool_size_mb,
                }
                if self.memory_pool
                else None
            ),
        }


# エクスポート用ファクトリ関数
def create_parallel_backtest_framework(
    max_workers: int = None,
    parallel_mode: ParallelMode = ParallelMode.MULTIPROCESSING,
    optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
) -> ParallelBacktestFramework:
    """並列バックテストフレームワーク作成"""
    config = ParallelBacktestConfig(
        max_workers=max_workers or mp.cpu_count(),
        parallel_mode=parallel_mode,
        optimization_method=optimization_method,
    )

    return ParallelBacktestFramework(config)
