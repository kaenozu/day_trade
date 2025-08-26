#!/usr/bin/env python3
"""
バッチ処理エンジン
Issue #376: バッチ処理の強化

メインバッチ処理エンジンクラス
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from .core_types import BatchJob, JobResult, PipelineStatus, WorkflowStage
from .data_fetch_processor import DataFetchProcessor
from .data_store_processor import DataStoreProcessor
from .data_transform_processor import DataTransformProcessor
from .data_validate_processor import DataValidateProcessor
from .stage_processor import StageProcessor

# プロジェクトモジュール
try:
    from ...data.unified_api_adapter import UnifiedAPIAdapter
    from ...models.advanced_batch_database import (
        AdvancedBatchDatabase,
        OptimizationLevel,
    )
    from ...models.database import get_database_manager
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def get_database_manager():
        return None

    # モッククラス
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

    class OptimizationLevel:
        ADVANCED = "advanced"


logger = get_context_logger(__name__)


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
            f"optimization={optimization_level}"
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

                    from .core_types import StageResult
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
            "transform_rules": {"price_scaling": 1.0},  # 必要に応じて調整
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
                    "elapsed_seconds": (
                        time.time() - job.started_at if job.started_at else 0
                    ),
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
            "optimization_level": self.optimization_level,
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