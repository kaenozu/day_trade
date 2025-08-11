#!/usr/bin/env python3
"""
統合バッチデータフロー最適化システム
全てのバッチ処理システムを統合し、最適化されたデータフローを提供

主要機能:
- 複数バッチシステムの統合管理
- インテリジェント・ルーティング
- エンドツーエンド最適化
- 自動フェイルオーバー
- リアルタイム性能調整
- データパイプライン可視化
"""

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from ..utils.logging_config import get_context_logger
from .api_request_consolidator import APIRequestConsolidator
from .api_request_consolidator import RequestPriority as APIRequestPriority
from .database_bulk_optimizer import (
    BulkOperation,
    BulkOperationConfig,
    DatabaseBulkOptimizer,
)
from .integrated_data_fetcher import (
    IntegratedDataFetcher,
    IntegratedDataRequest,
    IntegratedDataResponse,
)
from .parallel_batch_engine import ParallelBatchEngine, ProcessingMode, TaskPriority

logger = get_context_logger(__name__)


class FlowPriority(Enum):
    """フロー優先度"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    REALTIME = 5


class FlowStatus(Enum):
    """フロー状態"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class OptimizationTarget(Enum):
    """最適化目標"""

    LATENCY = "latency"  # レイテンシー最小化
    THROUGHPUT = "throughput"  # スループット最大化
    MEMORY = "memory"  # メモリ効率重視
    COST = "cost"  # コスト最適化
    BALANCED = "balanced"  # バランス重視


@dataclass
class DataFlowStage:
    """データフロー段階"""

    stage_id: str
    name: str
    handler: str  # "api_consolidator", "data_fetcher", "db_optimizer", "batch_engine"
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # 依存する段階
    timeout: float = 300.0
    retry_count: int = 2
    critical: bool = False  # 失敗時にフロー全体を停止するか


@dataclass
class UnifiedDataFlowRequest:
    """統合データフローリクエスト"""

    flow_id: str
    symbols: List[str]
    stages: List[DataFlowStage]
    priority: FlowPriority = FlowPriority.NORMAL
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED
    max_parallel_stages: int = 4
    enable_caching: bool = True
    enable_failover: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class StageResult:
    """段階実行結果"""

    stage_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used_mb: float = 0.0
    records_processed: int = 0
    cache_hit_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedDataFlowResult:
    """統合データフロー結果"""

    flow_id: str
    success: bool
    final_data: Any = None
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    total_execution_time: float = 0.0
    total_records_processed: int = 0
    overall_cache_hit_rate: float = 0.0
    optimization_metrics: Dict[str, float] = field(default_factory=dict)
    error_summary: List[str] = field(default_factory=list)
    status: FlowStatus = FlowStatus.COMPLETED


@dataclass
class FlowPerformanceMetrics:
    """フロー性能指標"""

    total_flows: int = 0
    successful_flows: int = 0
    failed_flows: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    average_throughput: float = 0.0  # records/sec
    average_cache_hit_rate: float = 0.0
    peak_memory_usage_mb: float = 0.0
    total_data_processed_mb: float = 0.0


class IntelligentRouter:
    """インテリジェント・ルーター"""

    def __init__(self):
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.routing_cache = {}
        self.lock = threading.Lock()

    def select_optimal_handler(
        self,
        stage: DataFlowStage,
        symbols: List[str],
        optimization_target: OptimizationTarget,
        available_handlers: Dict[str, Any],
    ) -> str:
        """最適なハンドラー選択"""

        # 基本ルーティングロジック
        if stage.handler in available_handlers:
            return stage.handler

        # フォールバック選択
        if optimization_target == OptimizationTarget.LATENCY:
            # レイテンシー重視の場合はキャッシュ利用率が高いハンドラー優先
            return self._select_by_cache_hit_rate(available_handlers)
        elif optimization_target == OptimizationTarget.THROUGHPUT:
            # スループット重視の場合は並列処理能力が高いハンドラー優先
            return self._select_by_throughput(available_handlers)
        elif optimization_target == OptimizationTarget.MEMORY:
            # メモリ効率重視の場合はメモリ使用量が少ないハンドラー優先
            return self._select_by_memory_efficiency(available_handlers)
        else:
            # バランス重視
            return self._select_balanced(available_handlers)

    def _select_by_cache_hit_rate(self, handlers: Dict[str, Any]) -> str:
        """キャッシュヒット率による選択"""
        # 簡略化: 利用可能なハンドラーから最初のものを選択
        return list(handlers.keys())[0] if handlers else "batch_engine"

    def _select_by_throughput(self, handlers: Dict[str, Any]) -> str:
        """スループットによる選択"""
        return list(handlers.keys())[0] if handlers else "batch_engine"

    def _select_by_memory_efficiency(self, handlers: Dict[str, Any]) -> str:
        """メモリ効率による選択"""
        return list(handlers.keys())[0] if handlers else "batch_engine"

    def _select_balanced(self, handlers: Dict[str, Any]) -> str:
        """バランスによる選択"""
        return list(handlers.keys())[0] if handlers else "batch_engine"

    def record_performance(self, handler: str, metrics: Dict[str, float]):
        """性能記録"""
        with self.lock:
            self.performance_history[handler].append(
                {"timestamp": time.time(), **metrics}
            )


class UnifiedBatchDataFlow:
    """
    統合バッチデータフロー最適化システム

    全てのバッチ処理システムを統合し、
    最適化されたデータフローパイプラインを提供
    """

    def __init__(
        self,
        enable_api_consolidator: bool = True,
        enable_data_fetcher: bool = True,
        enable_db_optimizer: bool = True,
        enable_batch_engine: bool = True,
        db_path: str = None,
        monitoring_interval: float = 30.0,
    ):
        # 各システムの初期化
        self.systems = {}

        # APIリクエスト統合システム
        if enable_api_consolidator:
            self.systems["api_consolidator"] = APIRequestConsolidator(
                base_batch_size=50, max_workers=6, enable_caching=True
            )

        # 統合データフェッチャー
        if enable_data_fetcher:
            self.systems["data_fetcher"] = IntegratedDataFetcher(
                consolidator_config={"base_batch_size": 30, "max_workers": 4},
                enable_advanced_batch_fetcher=True,
            )

        # データベースバルク最適化
        if enable_db_optimizer and db_path:
            self.systems["db_optimizer"] = DatabaseBulkOptimizer(
                db_path=db_path,
                pool_size=8,
                config=BulkOperationConfig(batch_size=1000, max_workers=4),
            )

        # 並列バッチエンジン
        if enable_batch_engine:
            self.systems["batch_engine"] = ParallelBatchEngine(
                initial_workers=4,
                processing_mode=ProcessingMode.THREAD_BASED,
                enable_adaptive_scaling=True,
            )

        # ルーターとモニタリング
        self.router = IntelligentRouter()
        self.monitoring_interval = monitoring_interval

        # フロー管理
        self.active_flows = {}  # flow_id -> Future
        self.completed_flows = {}  # flow_id -> UnifiedDataFlowResult
        self.flow_executor = None

        # 性能メトリクス
        self.performance_metrics = FlowPerformanceMetrics()
        self.performance_history = deque(maxlen=1000)

        # 制御フラグ
        self.running = False
        self.monitor_thread = None

        # ロック
        self.metrics_lock = threading.Lock()

        logger.info(
            f"統合バッチデータフロー初期化: systems={list(self.systems.keys())}"
        )

    def start(self):
        """システム開始"""
        if self.running:
            return

        self.running = True

        # 各システム開始
        for name, system in self.systems.items():
            if hasattr(system, "start"):
                system.start()
                logger.info(f"{name} started")

        # モニタリングスレッド開始
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()

        logger.info("統合バッチデータフロー開始")

    def stop(self, timeout: float = 30.0):
        """システム停止"""
        if not self.running:
            return

        logger.info("統合バッチデータフロー停止開始...")

        self.running = False

        # モニタリングスレッド停止
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        # アクティブフローの完了待機
        self._wait_for_active_flows(timeout=timeout)

        # 各システム停止
        for name, system in self.systems.items():
            try:
                if hasattr(system, "stop"):
                    system.stop()
                elif hasattr(system, "close"):
                    system.close()
                logger.info(f"{name} stopped")
            except Exception as e:
                logger.error(f"{name} stop error: {e}")

        logger.info("統合バッチデータフロー停止完了")

    def execute_flow(
        self,
        symbols: List[str],
        stages: List[DataFlowStage],
        flow_id: str = None,
        priority: FlowPriority = FlowPriority.NORMAL,
        optimization_target: OptimizationTarget = OptimizationTarget.BALANCED,
        **kwargs,
    ) -> UnifiedDataFlowResult:
        """
        データフロー実行（同期）

        Args:
            symbols: 銘柄コードリスト
            stages: フロー段階リスト
            flow_id: フローID
            priority: 優先度
            optimization_target: 最適化目標

        Returns:
            統合データフロー結果
        """
        if not self.running:
            raise RuntimeError("システムが開始されていません")

        if flow_id is None:
            flow_id = f"flow_{int(time.time() * 1000)}_{len(self.active_flows)}"

        request = UnifiedDataFlowRequest(
            flow_id=flow_id,
            symbols=symbols,
            stages=stages,
            priority=priority,
            optimization_target=optimization_target,
            **kwargs,
        )

        return self._execute_flow_internal(request)

    def execute_flow_async(
        self,
        symbols: List[str],
        stages: List[DataFlowStage],
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> str:
        """
        データフロー実行（非同期）

        Args:
            symbols: 銘柄コードリスト
            stages: フロー段階リスト
            callback: 完了時コールバック

        Returns:
            フローID
        """
        flow_id = f"flow_async_{int(time.time() * 1000)}_{len(self.active_flows)}"

        request = UnifiedDataFlowRequest(
            flow_id=flow_id, symbols=symbols, stages=stages, **kwargs
        )

        # ThreadPoolExecutorを使用して非同期実行
        if not hasattr(self, "flow_executor") or not self.flow_executor:
            from concurrent.futures import ThreadPoolExecutor

            self.flow_executor = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="FlowExec"
            )

        future = self.flow_executor.submit(self._execute_flow_internal, request)
        self.active_flows[flow_id] = future

        # コールバック設定
        if callback:

            def handle_completion(f):
                try:
                    result = f.result()
                    callback(flow_id, result)
                except Exception as e:
                    logger.error(f"フロー実行エラー {flow_id}: {e}")
                finally:
                    if flow_id in self.active_flows:
                        del self.active_flows[flow_id]

            future.add_done_callback(handle_completion)

        logger.info(
            f"非同期フロー開始: {flow_id} - {len(symbols)} symbols, {len(stages)} stages"
        )
        return flow_id

    def _execute_flow_internal(
        self, request: UnifiedDataFlowRequest
    ) -> UnifiedDataFlowResult:
        """内部フロー実行"""
        start_time = time.time()
        result = UnifiedDataFlowResult(
            flow_id=request.flow_id, success=False, status=FlowStatus.RUNNING
        )

        try:
            logger.info(
                f"フロー実行開始: {request.flow_id} - {len(request.symbols)} symbols"
            )

            # 段階依存関係の解決
            execution_order = self._resolve_stage_dependencies(request.stages)

            # 段階実行
            stage_data = {}  # stage_id -> data
            for stage_group in execution_order:
                # 並列実行可能な段階をグループ化
                if len(stage_group) == 1:
                    # 単一段階実行
                    stage = stage_group[0]
                    stage_result = self._execute_stage(
                        stage, request.symbols, stage_data, request
                    )
                    result.stage_results[stage.stage_id] = stage_result

                    if stage_result.success:
                        stage_data[stage.stage_id] = stage_result.data
                    elif stage.critical:
                        # クリティカル段階失敗時はフロー全体停止
                        result.error_summary.append(
                            f"Critical stage {stage.stage_id} failed"
                        )
                        break
                else:
                    # 並列段階実行
                    parallel_results = self._execute_stages_parallel(
                        stage_group, request.symbols, stage_data, request
                    )

                    for stage_id, stage_result in parallel_results.items():
                        result.stage_results[stage_id] = stage_result
                        if stage_result.success:
                            stage_data[stage_id] = stage_result.data

            # 最終データ決定
            if result.stage_results:
                # 最後に成功した段階のデータを最終データとする
                last_successful_stage = None
                for stage in reversed(request.stages):
                    if (
                        stage.stage_id in result.stage_results
                        and result.stage_results[stage.stage_id].success
                    ):
                        last_successful_stage = stage.stage_id
                        break

                if last_successful_stage:
                    result.final_data = stage_data.get(last_successful_stage)
                    result.success = True
                    result.status = FlowStatus.COMPLETED

            # メトリクス計算
            result.total_execution_time = time.time() - start_time
            result.total_records_processed = sum(
                r.records_processed for r in result.stage_results.values()
            )

            if result.stage_results:
                cache_hit_rates = [
                    r.cache_hit_rate
                    for r in result.stage_results.values()
                    if r.cache_hit_rate > 0
                ]
                result.overall_cache_hit_rate = (
                    sum(cache_hit_rates) / len(cache_hit_rates)
                    if cache_hit_rates
                    else 0.0
                )

            # 最適化メトリクス計算
            result.optimization_metrics = self._calculate_optimization_metrics(
                result, request.optimization_target
            )

            # 統計更新
            self._update_performance_metrics(result)

            logger.info(
                f"フロー実行完了: {request.flow_id} - success={result.success}, "
                f"time={result.total_execution_time:.2f}s, stages={len(result.stage_results)}"
            )

        except Exception as e:
            result.total_execution_time = time.time() - start_time
            result.error_summary.append(str(e))
            result.status = FlowStatus.FAILED
            logger.error(f"フロー実行エラー {request.flow_id}: {e}")

        # 完了フローとして保存
        self.completed_flows[request.flow_id] = result

        # アクティブフローから削除
        if request.flow_id in self.active_flows:
            del self.active_flows[request.flow_id]

        return result

    def _resolve_stage_dependencies(
        self, stages: List[DataFlowStage]
    ) -> List[List[DataFlowStage]]:
        """段階依存関係解決（トポロジカルソート）"""
        # 簡略化された実装: 依存関係を考慮した実行順序
        stage_map = {stage.stage_id: stage for stage in stages}
        resolved = []
        remaining = stages.copy()

        while remaining:
            # 依存関係のない段階を探す
            ready_stages = []
            for stage in remaining:
                dependencies_resolved = all(
                    dep_id in [s.stage_id for group in resolved for s in group]
                    for dep_id in stage.depends_on
                )
                if dependencies_resolved:
                    ready_stages.append(stage)

            if not ready_stages:
                # 循環依存の可能性
                ready_stages = [remaining[0]]  # 強制的に次に進む

            resolved.append(ready_stages)
            for stage in ready_stages:
                remaining.remove(stage)

        return resolved

    def _execute_stage(
        self,
        stage: DataFlowStage,
        symbols: List[str],
        stage_data: Dict[str, Any],
        request: UnifiedDataFlowRequest,
    ) -> StageResult:
        """単一段階実行"""
        start_time = time.time()
        result = StageResult(stage_id=stage.stage_id, success=False)

        try:
            # ルーターによるハンドラー選択
            handler_name = self.router.select_optimal_handler(
                stage, symbols, request.optimization_target, self.systems
            )

            if handler_name not in self.systems:
                raise ValueError(f"Handler {handler_name} not available")

            handler = self.systems[handler_name]

            # 段階実行
            if handler_name == "api_consolidator":
                data = self._execute_api_consolidator_stage(handler, stage, symbols)
            elif handler_name == "data_fetcher":
                data = self._execute_data_fetcher_stage(handler, stage, symbols)
            elif handler_name == "db_optimizer":
                data = self._execute_db_optimizer_stage(handler, stage, stage_data)
            elif handler_name == "batch_engine":
                data = self._execute_batch_engine_stage(
                    handler, stage, symbols, stage_data
                )
            else:
                raise ValueError(f"Unknown handler: {handler_name}")

            result.success = True
            result.data = data
            result.records_processed = self._count_records(data)

        except Exception as e:
            result.error = str(e)
            logger.error(f"段階実行エラー {stage.stage_id}: {e}")

        result.execution_time = time.time() - start_time
        return result

    def _execute_stages_parallel(
        self,
        stages: List[DataFlowStage],
        symbols: List[str],
        stage_data: Dict[str, Any],
        request: UnifiedDataFlowRequest,
    ) -> Dict[str, StageResult]:
        """並列段階実行"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        with ThreadPoolExecutor(
            max_workers=min(len(stages), request.max_parallel_stages)
        ) as executor:
            # Future作成
            future_to_stage = {
                executor.submit(
                    self._execute_stage, stage, symbols, stage_data, request
                ): stage.stage_id
                for stage in stages
            }

            # 結果取得
            for future in as_completed(
                future_to_stage, timeout=max(stage.timeout for stage in stages)
            ):
                stage_id = future_to_stage[future]
                try:
                    result = future.result()
                    results[stage_id] = result
                except Exception as e:
                    results[stage_id] = StageResult(
                        stage_id=stage_id, success=False, error=str(e)
                    )

        return results

    def _execute_api_consolidator_stage(
        self, handler: APIRequestConsolidator, stage: DataFlowStage, symbols: List[str]
    ) -> Any:
        """APIリクエスト統合段階実行"""
        config = stage.config
        endpoint = config.get("endpoint", "stock_data")
        parameters = config.get("parameters", {})
        priority = config.get("priority", APIRequestPriority.NORMAL)

        # 結果収集用
        results = {}
        completed_count = 0

        def response_callback(response):
            nonlocal completed_count
            if response.success:
                results.update(response.data or {})
            completed_count += 1

        # リクエスト投入
        request_id = handler.submit_request(
            endpoint=endpoint,
            symbols=symbols,
            parameters=parameters,
            priority=priority,
            callback=response_callback,
        )

        # 完了待機
        timeout = stage.timeout
        start_wait = time.time()
        while completed_count == 0 and (time.time() - start_wait) < timeout:
            time.sleep(0.01)

        return results

    def _execute_data_fetcher_stage(
        self, handler: IntegratedDataFetcher, stage: DataFlowStage, symbols: List[str]
    ) -> Any:
        """統合データフェッチャー段階実行"""
        config = stage.config
        period = config.get("period", "60d")
        interval = config.get("interval", "1d")

        response = handler.fetch_data(
            symbols=symbols, period=period, interval=interval, timeout=stage.timeout
        )

        return response.data if response.success_count > 0 else {}

    def _execute_db_optimizer_stage(
        self,
        handler: DatabaseBulkOptimizer,
        stage: DataFlowStage,
        stage_data: Dict[str, Any],
    ) -> Any:
        """データベース最適化段階実行"""
        config = stage.config
        table_name = config.get("table_name", "stock_data")
        operation = BulkOperation(config.get("operation", "insert"))

        # 前段階のデータを取得
        data_source_stage = config.get("data_source_stage")
        if data_source_stage and data_source_stage in stage_data:
            data = stage_data[data_source_stage]

            # データフレーム変換
            if isinstance(data, dict):
                records = []
                for symbol, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        df_dict = df.to_dict("records")
                        for record in df_dict:
                            record["symbol"] = symbol
                        records.extend(df_dict)
                data = records

            # バルク操作実行
            if operation == BulkOperation.INSERT:
                result = handler.bulk_insert(
                    table_name=table_name, data=data, **config.get("insert_params", {})
                )
            elif operation == BulkOperation.UPSERT:
                result = handler.bulk_upsert(
                    table_name=table_name,
                    data=data,
                    key_columns=config.get("key_columns", ["symbol", "date"]),
                    **config.get("upsert_params", {}),
                )

            return {
                "operation_result": result,
                "records_processed": result.processed_records,
            }

        return {}

    def _execute_batch_engine_stage(
        self,
        handler: ParallelBatchEngine,
        stage: DataFlowStage,
        symbols: List[str],
        stage_data: Dict[str, Any],
    ) -> Any:
        """バッチエンジン段階実行"""
        config = stage.config
        function_name = config.get("function")
        task_args = config.get("args", ())
        task_kwargs = config.get("kwargs", {})

        if not function_name:
            raise ValueError("Batch engine stage requires 'function' in config")

        # 関数を動的に取得（実際の実装ではより安全な方法を使用）
        function = globals().get(function_name)
        if not function:
            raise ValueError(f"Function {function_name} not found")

        # タスク投入
        task_ids = []
        for symbol in symbols:
            task_id = handler.submit_task(
                function=function,
                *task_args,  # noqa: B026
                symbol=symbol,
                **task_kwargs,
            )
            task_ids.append(task_id)

        # 結果取得
        results = handler.get_results_batch(task_ids, timeout=stage.timeout)

        return {
            symbol: result.result
            for symbol, result in results.items()
            if result.success
        }

    def _count_records(self, data: Any) -> int:
        """レコード数カウント"""
        if isinstance(data, dict):
            if all(isinstance(v, pd.DataFrame) for v in data.values()):
                return sum(len(df) for df in data.values())
            elif isinstance(data, dict):
                return len(data)
        elif isinstance(data, list):
            return len(data)
        elif isinstance(data, pd.DataFrame):
            return len(data)

        return 0

    def _calculate_optimization_metrics(
        self, result: UnifiedDataFlowResult, target: OptimizationTarget
    ) -> Dict[str, float]:
        """最適化メトリクス計算"""
        metrics = {}

        if target == OptimizationTarget.LATENCY:
            metrics["latency_score"] = 1.0 / max(result.total_execution_time, 0.001)
        elif target == OptimizationTarget.THROUGHPUT:
            metrics["throughput_score"] = result.total_records_processed / max(
                result.total_execution_time, 0.001
            )
        elif target == OptimizationTarget.MEMORY:
            total_memory = sum(r.memory_used_mb for r in result.stage_results.values())
            metrics["memory_efficiency_score"] = 1.0 / max(total_memory, 1.0)
        else:  # BALANCED
            latency_score = 1.0 / max(result.total_execution_time, 0.001)
            throughput_score = result.total_records_processed / max(
                result.total_execution_time, 0.001
            )
            cache_score = result.overall_cache_hit_rate

            metrics["balance_score"] = (
                latency_score + throughput_score + cache_score
            ) / 3.0

        return metrics

    def _update_performance_metrics(self, result: UnifiedDataFlowResult):
        """性能メトリクス更新"""
        with self.metrics_lock:
            self.performance_metrics.total_flows += 1

            if result.success:
                self.performance_metrics.successful_flows += 1
            else:
                self.performance_metrics.failed_flows += 1

            # 成功率
            self.performance_metrics.success_rate = (
                self.performance_metrics.successful_flows
                / self.performance_metrics.total_flows
            )

            # 平均実行時間
            total_time = (
                self.performance_metrics.average_execution_time
                * (self.performance_metrics.total_flows - 1)
                + result.total_execution_time
            )
            self.performance_metrics.average_execution_time = (
                total_time / self.performance_metrics.total_flows
            )

            # 平均スループット
            if result.total_execution_time > 0:
                flow_throughput = (
                    result.total_records_processed / result.total_execution_time
                )
                total_throughput = (
                    self.performance_metrics.average_throughput
                    * (self.performance_metrics.total_flows - 1)
                    + flow_throughput
                )
                self.performance_metrics.average_throughput = (
                    total_throughput / self.performance_metrics.total_flows
                )

            # キャッシュヒット率
            total_cache = (
                self.performance_metrics.average_cache_hit_rate
                * (self.performance_metrics.total_flows - 1)
                + result.overall_cache_hit_rate
            )
            self.performance_metrics.average_cache_hit_rate = (
                total_cache / self.performance_metrics.total_flows
            )

    def _monitoring_loop(self):
        """モニタリングループ"""
        while self.running:
            try:
                # 全システムの統計収集
                system_stats = self._collect_system_stats()

                # パフォーマンス履歴記録
                snapshot = {
                    "timestamp": time.time(),
                    "active_flows": len(self.active_flows),
                    "completed_flows": len(self.completed_flows),
                    "performance_metrics": self.performance_metrics.__dict__,
                    "system_stats": system_stats,
                }

                self.performance_history.append(snapshot)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                if self.running:
                    logger.error(f"モニタリングループエラー: {e}")

    def _collect_system_stats(self) -> Dict[str, Any]:
        """システム統計収集"""
        stats = {}

        for name, system in self.systems.items():
            try:
                if hasattr(system, "get_stats"):
                    stats[name] = system.get_stats().__dict__
                elif hasattr(system, "get_performance_stats"):
                    stats[name] = system.get_performance_stats()
            except Exception as e:
                logger.warning(f"{name}統計収集エラー: {e}")
                stats[name] = {"error": str(e)}

        return stats

    def _wait_for_active_flows(self, timeout: float):
        """アクティブフロー完了待機"""
        start_time = time.time()

        while self.active_flows and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.active_flows:
            logger.warning(f"タイムアウト: {len(self.active_flows)} flows still active")

    def get_flow_result(self, flow_id: str) -> Optional[UnifiedDataFlowResult]:
        """フロー結果取得"""
        return self.completed_flows.get(flow_id)

    def get_performance_metrics(self) -> FlowPerformanceMetrics:
        """性能メトリクス取得"""
        with self.metrics_lock:
            return self.performance_metrics

    def get_system_status(self) -> Dict[str, Any]:
        """システムステータス取得"""
        return {
            "running": self.running,
            "systems": list(self.systems.keys()),
            "active_flows": len(self.active_flows),
            "completed_flows": len(self.completed_flows),
            "performance_metrics": self.performance_metrics.__dict__,
            "recent_performance": list(self.performance_history)[-5:]
            if self.performance_history
            else [],
        }


# 便利関数
def create_unified_dataflow(
    db_path: str = None, enable_all: bool = True, **kwargs
) -> UnifiedBatchDataFlow:
    """統合データフロー作成"""
    return UnifiedBatchDataFlow(
        enable_api_consolidator=enable_all,
        enable_data_fetcher=enable_all,
        enable_db_optimizer=enable_all and db_path is not None,
        enable_batch_engine=enable_all,
        db_path=db_path,
        **kwargs,
    )


def create_standard_flow_stages() -> List[DataFlowStage]:
    """標準フロー段階作成"""
    return [
        DataFlowStage(
            stage_id="data_fetch",
            name="データ取得",
            handler="data_fetcher",
            config={"period": "60d", "interval": "1d"},
        ),
        DataFlowStage(
            stage_id="data_storage",
            name="データ保存",
            handler="db_optimizer",
            config={
                "table_name": "stock_data",
                "operation": "upsert",
                "data_source_stage": "data_fetch",
                "key_columns": ["symbol", "date"],
            },
            depends_on=["data_fetch"],
        ),
    ]


if __name__ == "__main__":
    # テスト実行
    print("=== 統合バッチデータフロー最適化システム テスト ===")

    import os
    import tempfile

    # テストDB作成
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        test_db_path = tmp_file.name

    try:
        # 統合システム作成
        dataflow = create_unified_dataflow(
            db_path=test_db_path, enable_all=True, monitoring_interval=5.0
        )

        dataflow.start()

        # テストフロー実行
        test_symbols = ["7203", "8306", "9984"]
        stages = create_standard_flow_stages()

        print(f"フロー実行開始: {test_symbols}")

        result = dataflow.execute_flow(
            symbols=test_symbols,
            stages=stages,
            flow_id="test_flow_001",
            priority=FlowPriority.HIGH,
            optimization_target=OptimizationTarget.BALANCED,
        )

        print("\nフロー結果:")
        print(f"  フローID: {result.flow_id}")
        print(f"  成功: {result.success}")
        print(f"  ステータス: {result.status.value}")
        print(f"  実行時間: {result.total_execution_time:.2f}秒")
        print(f"  処理レコード数: {result.total_records_processed}")
        print(f"  キャッシュヒット率: {result.overall_cache_hit_rate:.1%}")
        print(f"  エラー: {result.error_summary}")

        # 段階結果詳細
        print("\n段階実行結果:")
        for stage_id, stage_result in result.stage_results.items():
            print(
                f"  {stage_id}: success={stage_result.success}, "
                f"time={stage_result.execution_time:.2f}s, "
                f"records={stage_result.records_processed}"
            )
            if stage_result.error:
                print(f"    Error: {stage_result.error}")

        # 性能メトリクス
        metrics = dataflow.get_performance_metrics()
        print("\n性能メトリクス:")
        print(f"  総フロー: {metrics.total_flows}")
        print(f"  成功率: {metrics.success_rate:.1%}")
        print(f"  平均実行時間: {metrics.average_execution_time:.2f}秒")
        print(f"  平均スループット: {metrics.average_throughput:.1f} records/sec")
        print(f"  平均キャッシュヒット率: {metrics.average_cache_hit_rate:.1%}")

        # システムステータス
        status = dataflow.get_system_status()
        print("\nシステムステータス:")
        print(f"  稼働中: {status['running']}")
        print(f"  利用可能システム: {status['systems']}")
        print(f"  アクティブフロー: {status['active_flows']}")
        print(f"  完了フロー: {status['completed_flows']}")

    finally:
        # クリーンアップ
        try:
            dataflow.stop(timeout=15.0)
        except Exception as e:
            print(f"停止エラー: {e}")

        try:
            os.unlink(test_db_path)
        except:
            pass

    print("\n=== テスト完了 ===")
