#!/usr/bin/env python3
"""
統一バッチ処理エンジン
Issue #376: バッチ処理の強化

データ取得・処理・保存の全体をバッチ最適化する統合エンジン
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# プロジェクトモジュール
try:
    from ..data.batch_data_processor import BatchDataProcessor
    from ..data.batch_data_processor import BatchOperationType as DataBatchType
    from ..data.batch_data_processor import BatchRequest as DataBatchRequest
    from ..data.unified_api_adapter import (
        APIProvider,
        APIRequest,
        RequestMethod,
        UnifiedAPIAdapter,
    )
    from ..models.advanced_batch_database import (
        AdvancedBatchDatabase,
        OptimizationLevel,
    )
    from ..models.advanced_batch_database import BatchOperation as DBBatchOperation
    from ..models.advanced_batch_database import BatchOperationType as DBBatchType
    from ..models.database import get_database_manager
    from ..utils.cache_utils import generate_safe_cache_key
    from ..utils.logging_config import get_context_logger, log_performance_metric
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    def get_database_manager():
        return None

    def generate_safe_cache_key(*args, **kwargs):
        return str(hash(str(args) + str(kwargs)))

    # モッククラス
    class BatchDataProcessor:
        def __init__(self, **kwargs):
            pass

        def shutdown(self):
            pass

    class UnifiedAPIAdapter:
        def __init__(self, **kwargs):
            pass

        async def cleanup(self):
            pass

        def get_stats(self):
            return {"mock": True}

    class AdvancedBatchDatabase:
        def __init__(self, **kwargs):
            pass

        def cleanup(self):
            pass

        def get_stats(self):
            return {"mock": True}

    class OptimizationLevel(Enum):
        BASIC = "basic"
        ADVANCED = "advanced"
        EXTREME = "extreme"


logger = get_context_logger(__name__)


class WorkflowStage(Enum):
    """ワークフローステージ"""

    DATA_FETCH = "data_fetch"
    DATA_TRANSFORM = "data_transform"
    DATA_VALIDATE = "data_validate"
    DATA_STORE = "data_store"
    DATA_ANALYZE = "data_analyze"


class PipelineStatus(Enum):
    """パイプライン状態"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """バッチジョブ定義"""

    job_id: str
    job_name: str
    workflow_stages: List[WorkflowStage]
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout_seconds: int = 3600
    retry_count: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: PipelineStatus = PipelineStatus.PENDING

    def is_expired(self) -> bool:
        """有効期限チェック"""
        return time.time() - self.created_at > self.timeout_seconds


@dataclass
class StageResult:
    """ステージ実行結果"""

    stage: WorkflowStage
    success: bool
    data: Any = None
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """ジョブ実行結果"""

    job_id: str
    job_name: str
    success: bool
    total_processing_time_ms: float
    stage_results: List[StageResult] = field(default_factory=list)
    final_data: Any = None
    error_message: Optional[str] = None
    completed_at: float = field(default_factory=time.time)


class StageProcessor:
    """ステージプロセッサー基底クラス"""

    def __init__(self, stage: WorkflowStage):
        self.stage = stage

    async def process(self, data: Any, parameters: Dict[str, Any]) -> StageResult:
        """ステージ処理実行"""
        raise NotImplementedError


class DataFetchProcessor(StageProcessor):
    """データ取得プロセッサー"""

    def __init__(self, api_adapter: UnifiedAPIAdapter):
        super().__init__(WorkflowStage.DATA_FETCH)
        self.api_adapter = api_adapter

    async def process(self, data: Any, parameters: Dict[str, Any]) -> StageResult:
        """データ取得処理"""
        start_time = time.time()

        try:
            # パラメータ解析
            symbols = parameters.get("symbols", [])
            data_types = parameters.get("data_types", ["current_price"])

            if not symbols:
                return StageResult(
                    stage=self.stage,
                    success=False,
                    error_message="Symbols required for data fetch",
                )

            fetch_results = {}

            # データタイプ別に取得
            for data_type in data_types:
                if data_type == "current_price":
                    requests = []
                    for symbol in symbols:
                        requests.append(
                            APIRequest(
                                provider=APIProvider.YFINANCE,
                                endpoint="current_price",
                                parameters={"symbol": symbol},
                                batch_key="price_batch",
                            )
                        )

                    responses = await self.api_adapter.execute_multiple(requests)
                    prices = {}
                    for response in responses:
                        if response.success and response.data:
                            symbol = response.data.get("symbol")
                            if symbol:
                                prices[symbol] = response.data

                    fetch_results[data_type] = prices

                elif data_type == "historical_data":
                    period = parameters.get("period", "1mo")
                    interval = parameters.get("interval", "1d")

                    requests = []
                    for symbol in symbols:
                        requests.append(
                            APIRequest(
                                provider=APIProvider.YFINANCE,
                                endpoint="historical_data",
                                parameters={
                                    "symbol": symbol,
                                    "period": period,
                                    "interval": interval,
                                },
                            )
                        )

                    responses = await self.api_adapter.execute_multiple(requests)
                    historical = {}
                    for response in responses:
                        if response.success and response.data:
                            symbol = response.data.get("symbol")
                            if symbol:
                                historical[symbol] = response.data

                    fetch_results[data_type] = historical

            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=True,
                data=fetch_results,
                processing_time_ms=processing_time_ms,
                statistics={
                    "symbols_requested": len(symbols),
                    "data_types_fetched": len(data_types),
                    "total_records": sum(
                        len(v) if isinstance(v, dict) else 0
                        for v in fetch_results.values()
                    ),
                },
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=False,
                processing_time_ms=processing_time_ms,
                error_message=str(e),
            )


class DataTransformProcessor(StageProcessor):
    """データ変換プロセッサー"""

    def __init__(self):
        super().__init__(WorkflowStage.DATA_TRANSFORM)

    async def process(self, data: Any, parameters: Dict[str, Any]) -> StageResult:
        """データ変換処理"""
        start_time = time.time()

        try:
            transform_rules = parameters.get("transform_rules", {})
            output_format = parameters.get("output_format", "dict")

            if not data or not isinstance(data, dict):
                return StageResult(
                    stage=self.stage,
                    success=False,
                    error_message="Invalid input data for transformation",
                )

            transformed_data = {}

            # データタイプ別変換
            for data_type, type_data in data.items():
                if data_type == "current_price":
                    # 価格データの正規化
                    normalized_prices = []
                    for symbol, price_data in type_data.items():
                        normalized = {
                            "symbol": symbol,
                            "price": float(price_data.get("price", 0)),
                            "change": float(price_data.get("change", 0)),
                            "change_percent": float(
                                price_data.get("change_percent", 0)
                            ),
                            "volume": int(price_data.get("volume", 0)),
                            "timestamp": price_data.get("timestamp", time.time()),
                        }

                        # カスタム変換ルール適用
                        if "price_scaling" in transform_rules:
                            normalized["price"] *= transform_rules["price_scaling"]

                        normalized_prices.append(normalized)

                    transformed_data[data_type] = normalized_prices

                elif data_type == "historical_data":
                    # ヒストリカルデータの変換
                    processed_historical = []
                    for symbol, hist_data in type_data.items():
                        if "data" in hist_data and hist_data["data"]:
                            for record in hist_data["data"]:
                                processed_record = {
                                    "symbol": symbol,
                                    "date": record.get("date", ""),
                                    "open": float(record.get("Open", 0)),
                                    "high": float(record.get("High", 0)),
                                    "low": float(record.get("Low", 0)),
                                    "close": float(record.get("Close", 0)),
                                    "volume": int(record.get("Volume", 0)),
                                }
                                processed_historical.append(processed_record)

                    transformed_data[data_type] = processed_historical

            # 出力形式変換
            if output_format == "flat":
                # フラット形式（全データを単一リストに）
                flat_data = []
                for type_data in transformed_data.values():
                    if isinstance(type_data, list):
                        flat_data.extend(type_data)
                final_data = flat_data
            else:
                final_data = transformed_data

            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=True,
                data=final_data,
                processing_time_ms=processing_time_ms,
                statistics={
                    "input_data_types": len(data),
                    "output_format": output_format,
                    "total_records_transformed": sum(
                        len(v) if isinstance(v, list) else 0
                        for v in transformed_data.values()
                    ),
                },
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=False,
                processing_time_ms=processing_time_ms,
                error_message=str(e),
            )


class DataValidateProcessor(StageProcessor):
    """データ検証プロセッサー"""

    def __init__(self):
        super().__init__(WorkflowStage.DATA_VALIDATE)

    async def process(self, data: Any, parameters: Dict[str, Any]) -> StageResult:
        """データ検証処理"""
        start_time = time.time()

        try:
            validation_rules = parameters.get("validation_rules", {})
            strict_mode = parameters.get("strict_mode", False)

            validation_errors = []
            valid_records = []
            invalid_records = []

            if not data:
                return StageResult(
                    stage=self.stage,
                    success=False,
                    error_message="No data provided for validation",
                )

            # データが辞書形式の場合（データタイプ別）
            if isinstance(data, dict):
                validated_data = {}

                for data_type, type_data in data.items():
                    if not isinstance(type_data, list):
                        continue

                    valid_type_records = []

                    for record in type_data:
                        validation_result = self._validate_record(
                            record, validation_rules, data_type
                        )

                        if validation_result["valid"]:
                            valid_type_records.append(record)
                            valid_records.append(record)
                        else:
                            validation_errors.extend(validation_result["errors"])
                            invalid_records.append(record)

                            if strict_mode:
                                return StageResult(
                                    stage=self.stage,
                                    success=False,
                                    error_message=f"Validation failed in strict mode: {validation_result['errors']}",
                                )

                    validated_data[data_type] = valid_type_records

                final_data = validated_data

            # データがリスト形式の場合（フラット）
            elif isinstance(data, list):
                validated_records = []

                for record in data:
                    validation_result = self._validate_record(record, validation_rules)

                    if validation_result["valid"]:
                        validated_records.append(record)
                        valid_records.append(record)
                    else:
                        validation_errors.extend(validation_result["errors"])
                        invalid_records.append(record)

                        if strict_mode:
                            return StageResult(
                                stage=self.stage,
                                success=False,
                                error_message=f"Validation failed: {validation_result['errors']}",
                            )

                final_data = validated_records

            else:
                return StageResult(
                    stage=self.stage,
                    success=False,
                    error_message="Unsupported data format for validation",
                )

            processing_time_ms = (time.time() - start_time) * 1000

            success = len(validation_errors) == 0 or not strict_mode

            return StageResult(
                stage=self.stage,
                success=success,
                data=final_data,
                processing_time_ms=processing_time_ms,
                statistics={
                    "total_records": len(valid_records) + len(invalid_records),
                    "valid_records": len(valid_records),
                    "invalid_records": len(invalid_records),
                    "validation_errors": validation_errors,
                    "validation_success_rate": len(valid_records)
                    / max(len(valid_records) + len(invalid_records), 1),
                },
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=False,
                processing_time_ms=processing_time_ms,
                error_message=str(e),
            )

    def _validate_record(
        self, record: Dict[str, Any], rules: Dict[str, Any], data_type: str = None
    ) -> Dict[str, Any]:
        """個別レコード検証"""
        errors = []

        try:
            # 必須フィールド検証
            required_fields = rules.get("required_fields", [])
            for field in required_fields:
                if field not in record or record[field] is None:
                    errors.append(f"Required field missing: {field}")

            # データタイプ別検証
            if data_type == "current_price":
                # 価格データ検証
                if "price" in record:
                    price = record["price"]
                    if not isinstance(price, (int, float)) or price < 0:
                        errors.append(f"Invalid price: {price}")

                if "volume" in record:
                    volume = record["volume"]
                    if not isinstance(volume, int) or volume < 0:
                        errors.append(f"Invalid volume: {volume}")

            # カスタム検証ルール
            custom_validations = rules.get("custom_validations", [])
            for validation in custom_validations:
                field = validation.get("field")
                condition = validation.get("condition")

                if field in record:
                    value = record[field]
                    if condition == "positive" and value <= 0:
                        errors.append(f"Field {field} must be positive")
                    elif condition == "not_empty" and (value is None or value == ""):
                        errors.append(f"Field {field} must not be empty")

            return {"valid": len(errors) == 0, "errors": errors}

        except Exception as e:
            return {"valid": False, "errors": [f"Validation error: {str(e)}"]}


class DataStoreProcessor(StageProcessor):
    """データ保存プロセッサー"""

    def __init__(self, batch_db: AdvancedBatchDatabase):
        super().__init__(WorkflowStage.DATA_STORE)
        self.batch_db = batch_db

    async def process(self, data: Any, parameters: Dict[str, Any]) -> StageResult:
        """データ保存処理"""
        start_time = time.time()

        try:
            table_mapping = parameters.get("table_mapping", {})
            operation_type = parameters.get("operation_type", "insert")

            if not data:
                return StageResult(
                    stage=self.stage,
                    success=False,
                    error_message="No data provided for storage",
                )

            total_stored = 0
            store_results = []

            # データタイプ別保存処理
            if isinstance(data, dict):
                for data_type, type_data in data.items():
                    if not isinstance(type_data, list) or not type_data:
                        continue

                    table_name = table_mapping.get(data_type, f"{data_type}_data")

                    # バッチ操作作成
                    if operation_type == "insert":
                        db_operation_type = DBBatchType.INSERT
                    elif operation_type == "upsert":
                        db_operation_type = DBBatchType.UPSERT
                    else:
                        db_operation_type = DBBatchType.INSERT

                    batch_op = DBBatchOperation(
                        operation_type=db_operation_type,
                        table_name=table_name,
                        data=type_data,
                    )

                    result = await self.batch_db.execute_batch_operation(batch_op)
                    store_results.append(
                        {
                            "data_type": data_type,
                            "table_name": table_name,
                            "success": result.success,
                            "affected_rows": result.affected_rows,
                            "error": result.error_message,
                        }
                    )

                    if result.success:
                        total_stored += result.affected_rows

            # フラットデータの場合
            elif isinstance(data, list):
                table_name = table_mapping.get("default", "batch_data")

                db_operation_type = (
                    DBBatchType.INSERT
                    if operation_type == "insert"
                    else DBBatchType.UPSERT
                )

                batch_op = DBBatchOperation(
                    operation_type=db_operation_type, table_name=table_name, data=data
                )

                result = await self.batch_db.execute_batch_operation(batch_op)
                store_results.append(
                    {
                        "data_type": "flat",
                        "table_name": table_name,
                        "success": result.success,
                        "affected_rows": result.affected_rows,
                        "error": result.error_message,
                    }
                )

                if result.success:
                    total_stored += result.affected_rows

            processing_time_ms = (time.time() - start_time) * 1000

            success = all(result["success"] for result in store_results)
            error_messages = [r["error"] for r in store_results if r["error"]]

            return StageResult(
                stage=self.stage,
                success=success,
                data={"stored_records": total_stored},
                processing_time_ms=processing_time_ms,
                error_message="; ".join(error_messages) if error_messages else None,
                statistics={
                    "total_records_stored": total_stored,
                    "store_operations": len(store_results),
                    "successful_operations": sum(
                        1 for r in store_results if r["success"]
                    ),
                    "operation_details": store_results,
                },
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=False,
                processing_time_ms=processing_time_ms,
                error_message=str(e),
            )


class BatchProcessingEngine:
    """統一バッチ処理エンジン"""

    def __init__(
        self,
        max_concurrent_jobs: int = 5,
        enable_caching: bool = True,
        optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED,
    ):
        """
        初期化

        Args:
            max_concurrent_jobs: 最大同時ジョブ数
            enable_caching: キャッシュ有効化
            optimization_level: 最適化レベル
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.enable_caching = enable_caching
        self.optimization_level = optimization_level

        # 核心コンポーネント
        self.api_adapter = UnifiedAPIAdapter(enable_caching=enable_caching)

        # データベース関連
        self.db_manager = get_database_manager()
        self.batch_db = None
        if self.db_manager:
            self.batch_db = AdvancedBatchDatabase(
                self.db_manager, optimization_level=optimization_level
            )

        # プロセッサー初期化
        self.processors = {
            WorkflowStage.DATA_FETCH: DataFetchProcessor(self.api_adapter),
            WorkflowStage.DATA_TRANSFORM: DataTransformProcessor(),
            WorkflowStage.DATA_VALIDATE: DataValidateProcessor(),
        }

        if self.batch_db:
            self.processors[WorkflowStage.DATA_STORE] = DataStoreProcessor(
                self.batch_db
            )

        # ジョブ管理
        self.active_jobs: Dict[str, BatchJob] = {}
        self.job_results: Dict[str, JobResult] = {}
        self.job_lock = threading.RLock()

        # 統計情報
        self.stats = {
            "jobs_processed": 0,
            "jobs_succeeded": 0,
            "jobs_failed": 0,
            "average_processing_time_ms": 0.0,
            "total_records_processed": 0,
        }
        self.stats_lock = threading.RLock()

        # 並行実行制御
        self.semaphore = asyncio.Semaphore(max_concurrent_jobs)

        logger.info(
            f"BatchProcessingEngine初期化完了: "
            f"max_jobs={max_concurrent_jobs}, "
            f"optimization={optimization_level.value}"
        )

    async def execute_job(self, job: BatchJob) -> JobResult:
        """ジョブ実行"""
        async with self.semaphore:
            return await self._execute_job_internal(job)

    async def _execute_job_internal(self, job: BatchJob) -> JobResult:
        """内部ジョブ実行"""
        job_start_time = time.time()

        with self.job_lock:
            self.active_jobs[job.job_id] = job
            job.status = PipelineStatus.RUNNING
            job.started_at = job_start_time

        try:
            stage_results = []
            current_data = None

            # ワークフローステージ順次実行
            for stage in job.workflow_stages:
                processor = self.processors.get(stage)
                if not processor:
                    error_msg = f"Processor not found for stage: {stage}"
                    logger.error(error_msg)

                    stage_results.append(
                        StageResult(stage=stage, success=False, error_message=error_msg)
                    )
                    break

                # ステージ実行
                logger.debug(f"ジョブ {job.job_id} ステージ {stage.value} 実行開始")

                stage_result = await processor.process(current_data, job.parameters)
                stage_results.append(stage_result)

                logger.debug(
                    f"ジョブ {job.job_id} ステージ {stage.value} 完了: "
                    f"success={stage_result.success}, "
                    f"time={stage_result.processing_time_ms:.2f}ms"
                )

                if not stage_result.success:
                    # ステージ失敗時は処理中断
                    break

                # 次のステージにデータを渡す
                current_data = stage_result.data

            # 結果判定
            success = all(result.success for result in stage_results)
            total_processing_time_ms = (time.time() - job_start_time) * 1000

            # ジョブ結果作成
            job_result = JobResult(
                job_id=job.job_id,
                job_name=job.job_name,
                success=success,
                total_processing_time_ms=total_processing_time_ms,
                stage_results=stage_results,
                final_data=current_data,
                error_message=None if success else "One or more stages failed",
            )

            # 統計更新
            self._update_stats(job_result)

            # ジョブ状態更新
            with self.job_lock:
                job.status = (
                    PipelineStatus.COMPLETED if success else PipelineStatus.FAILED
                )
                job.completed_at = time.time()
                self.job_results[job.job_id] = job_result
                del self.active_jobs[job.job_id]

            logger.info(
                f"ジョブ {job.job_id} 完了: success={success}, "
                f"time={total_processing_time_ms:.2f}ms"
            )

            return job_result

        except Exception as e:
            total_processing_time_ms = (time.time() - job_start_time) * 1000
            error_msg = f"Job execution failed: {str(e)}"
            logger.error(f"ジョブ {job.job_id} 実行エラー: {e}")

            job_result = JobResult(
                job_id=job.job_id,
                job_name=job.job_name,
                success=False,
                total_processing_time_ms=total_processing_time_ms,
                stage_results=[],
                error_message=error_msg,
            )

            # 統計更新
            self._update_stats(job_result)

            # ジョブ状態更新
            with self.job_lock:
                job.status = PipelineStatus.FAILED
                job.completed_at = time.time()
                self.job_results[job.job_id] = job_result
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]

            return job_result

    def create_stock_data_pipeline_job(
        self,
        symbols: List[str],
        job_name: str = "stock_data_pipeline",
        include_historical: bool = False,
        store_data: bool = True,
    ) -> BatchJob:
        """株価データパイプラインジョブ作成"""

        # ワークフローステージ決定
        workflow_stages = [
            WorkflowStage.DATA_FETCH,
            WorkflowStage.DATA_TRANSFORM,
            WorkflowStage.DATA_VALIDATE,
        ]

        if store_data and self.batch_db:
            workflow_stages.append(WorkflowStage.DATA_STORE)

        # パラメータ設定
        data_types = ["current_price"]
        if include_historical:
            data_types.append("historical_data")

        parameters = {
            "symbols": symbols,
            "data_types": data_types,
            "period": "1mo",
            "interval": "1d",
            "transform_rules": {
                "price_scaling": 1.0  # 必要に応じて調整
            },
            "output_format": "dict",
            "validation_rules": {
                "required_fields": ["symbol"],
                "custom_validations": [
                    {"field": "price", "condition": "positive"},
                    {"field": "symbol", "condition": "not_empty"},
                ],
            },
            "table_mapping": {
                "current_price": "stock_prices",
                "historical_data": "stock_historical",
            },
            "operation_type": "upsert",
        }

        job_id = f"stock_pipeline_{int(time.time() * 1000)}"

        return BatchJob(
            job_id=job_id,
            job_name=job_name,
            workflow_stages=workflow_stages,
            parameters=parameters,
            priority=1,
            timeout_seconds=3600,
        )

    def _update_stats(self, job_result: JobResult):
        """統計更新"""
        with self.stats_lock:
            self.stats["jobs_processed"] += 1

            if job_result.success:
                self.stats["jobs_succeeded"] += 1
            else:
                self.stats["jobs_failed"] += 1

            # 移動平均による処理時間更新
            alpha = 0.1
            self.stats["average_processing_time_ms"] = (
                self.stats["average_processing_time_ms"] * (1 - alpha)
                + job_result.total_processing_time_ms * alpha
            )

            # レコード数統計
            for stage_result in job_result.stage_results:
                if "total_records" in stage_result.statistics:
                    self.stats["total_records_processed"] += stage_result.statistics[
                        "total_records"
                    ]

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """ジョブステータス取得"""
        with self.job_lock:
            # アクティブジョブ確認
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                return {
                    "job_id": job.job_id,
                    "job_name": job.job_name,
                    "status": job.status.value,
                    "started_at": job.started_at,
                    "elapsed_seconds": time.time() - job.started_at
                    if job.started_at
                    else 0,
                }

            # 完了ジョブ確認
            if job_id in self.job_results:
                result = self.job_results[job_id]
                return {
                    "job_id": result.job_id,
                    "job_name": result.job_name,
                    "status": "completed" if result.success else "failed",
                    "success": result.success,
                    "total_processing_time_ms": result.total_processing_time_ms,
                    "completed_at": result.completed_at,
                    "stages_completed": len(result.stage_results),
                    "error_message": result.error_message,
                }

        return None

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self.stats_lock:
            stats = self.stats.copy()

        with self.job_lock:
            stats.update(
                {
                    "active_jobs": len(self.active_jobs),
                    "completed_jobs": len(self.job_results),
                    "max_concurrent_jobs": self.max_concurrent_jobs,
                }
            )

        # コンポーネント統計追加
        stats["api_adapter_stats"] = self.api_adapter.get_stats()
        if self.batch_db:
            stats["database_stats"] = self.batch_db.get_stats()

        return stats

    def get_health_status(self) -> Dict[str, Any]:
        """ヘルスステータス取得"""
        stats = self.get_stats()

        job_success_rate = stats["jobs_succeeded"] / max(stats["jobs_processed"], 1)
        active_jobs_ratio = stats["active_jobs"] / self.max_concurrent_jobs

        health = "healthy"
        if job_success_rate < 0.8:
            health = "degraded"
        if job_success_rate < 0.5 or active_jobs_ratio > 0.9:
            health = "unhealthy"

        return {
            "status": health,
            "job_success_rate": job_success_rate,
            "active_jobs_utilization": active_jobs_ratio,
            "average_processing_time_ms": stats["average_processing_time_ms"],
            "total_records_processed": stats["total_records_processed"],
            "optimization_level": self.optimization_level.value,
        }

    async def cleanup(self):
        """クリーンアップ"""
        logger.info("BatchProcessingEngine クリーンアップ開始")

        try:
            # APIアダプタークリーンアップ
            await self.api_adapter.cleanup()

            # データベースクリーンアップ
            if self.batch_db:
                self.batch_db.cleanup()

            logger.info("BatchProcessingEngine クリーンアップ完了")

        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")


# 便利関数
async def execute_stock_batch_pipeline(
    symbols: List[str],
    include_historical: bool = False,
    store_data: bool = True,
    engine: BatchProcessingEngine = None,
) -> JobResult:
    """株価データバッチパイプライン実行"""

    if not engine:
        engine = BatchProcessingEngine()

    try:
        # ジョブ作成
        job = engine.create_stock_data_pipeline_job(
            symbols=symbols,
            include_historical=include_historical,
            store_data=store_data,
        )

        # 実行
        result = await engine.execute_job(job)
        return result

    finally:
        if not engine:  # 一時的に作成したエンジンの場合
            await engine.cleanup()


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #376 統一バッチ処理エンジンテスト ===")

        engine = BatchProcessingEngine(max_concurrent_jobs=3)

        try:
            # 株価データパイプラインテスト
            print("\n1. 株価データパイプライン実行")
            test_symbols = ["AAPL", "GOOGL", "MSFT"]

            result = await execute_stock_batch_pipeline(
                symbols=test_symbols,
                include_historical=False,
                store_data=False,  # テストではデータベース保存をスキップ
                engine=engine,
            )

            print(f"ジョブ結果: success={result.success}")
            print(f"処理時間: {result.total_processing_time_ms:.2f}ms")
            print(f"ステージ数: {len(result.stage_results)}")

            for stage_result in result.stage_results:
                print(
                    f"  - {stage_result.stage.value}: "
                    f"success={stage_result.success}, "
                    f"time={stage_result.processing_time_ms:.2f}ms"
                )

            # 統計情報
            print("\n2. 統計情報")
            stats = engine.get_stats()
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

            # ヘルスステータス
            print("\n3. ヘルスステータス")
            health = engine.get_health_status()
            for key, value in health.items():
                print(f"  {key}: {value}")

        finally:
            await engine.cleanup()

    asyncio.run(main())
    print("\n=== 統一バッチ処理エンジンテスト完了 ===")
