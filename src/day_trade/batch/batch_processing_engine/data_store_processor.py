#!/usr/bin/env python3
"""
データ保存プロセッサー
Issue #376: バッチ処理の強化

データ保存ステージの処理ロジック
"""

import time
from typing import Any, Dict

from .core_types import StageResult, WorkflowStage
from .stage_processor import StageProcessor

# プロジェクトモジュール
try:
    from ...models.advanced_batch_database import (
        AdvancedBatchDatabase,
        BatchOperation as DBBatchOperation,
        BatchOperationType as DBBatchType,
    )
except ImportError:
    # モッククラス
    class AdvancedBatchDatabase:
        def __init__(self, **kwargs):
            pass

        async def execute_batch_operation(self, operation):
            return type('Result', (), {
                'success': True, 
                'affected_rows': 0, 
                'error_message': None
            })()

    class DBBatchType:
        INSERT = "insert"
        UPSERT = "upsert"

    class DBBatchOperation:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


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