"""
基本的な一括操作モジュール
大量データの一括挿入・更新・削除機能

Issue #120: declarative_base()の定義場所の最適化対応
- 基本的な一括操作の責務を明確化
- パフォーマンス最適化
"""

import time
from typing import List

from ...utils.exceptions import handle_database_exception
from ...utils.logging_config import get_context_logger, log_database_operation
from .transaction import TransactionManager

logger = get_context_logger(__name__)


class BasicBulkOperations:
    """基本的な一括操作クラス"""

    def __init__(self, transaction_manager: TransactionManager):
        """
        基本一括操作の初期化

        Args:
            transaction_manager: トランザクション管理
        """
        self.transaction_manager = transaction_manager

    def bulk_insert(self, model_class, data_list: list, batch_size: int = 1000):
        """
        大量データの一括挿入（堅牢性向上版）

        Args:
            model_class: 挿入するモデルクラス
            data_list: 挿入するデータのリスト（辞書形式）
            batch_size: バッチサイズ

        Raises:
            DatabaseError: 一括挿入に失敗した場合
        """
        if not data_list:
            return

        operation_logger = logger
        operation_logger.info("Starting bulk insert")

        try:
            with self.transaction_manager.transaction_scope() as session:
                start_time = time.time()
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i : i + batch_size]
                    batch_number = i // batch_size + 1

                    batch_start_time = time.time()
                    try:
                        session.bulk_insert_mappings(model_class, batch)
                        session.flush()
                        batch_end_time = time.time()
                        log_database_operation(
                            "bulk_insert_batch",
                            duration=batch_end_time - batch_start_time,
                            table_name=str(model_class.__table__.name),
                            batch_number=batch_number,
                            batch_size=len(batch),
                        )
                    except Exception as batch_error:
                        operation_logger.error(
                            "Bulk insert batch failed",
                            batch_number=batch_number,
                            batch_size=len(batch),
                            error=str(batch_error),
                        )
                        raise

            operation_logger.info("Bulk insert completed successfully")
        except Exception as e:
            converted_error = handle_database_exception(e)
            operation_logger.error(
                "Bulk insert operation failed", extra={"error": str(converted_error)}
            )
            raise converted_error from e

    def bulk_update(self, model_class, data_list: list, batch_size: int = 1000):
        """
        大量データの一括更新（堅牢性向上版）

        Args:
            model_class: 更新するモデルクラス
            data_list: 更新するデータのリスト（辞書形式、idが必要）
            batch_size: バッチサイズ

        Raises:
            DatabaseError: 一括更新に失敗した場合
        """
        if not data_list:
            return

        operation_logger = logger
        operation_logger.info("Starting bulk update")

        try:
            with self.transaction_manager.transaction_scope() as session:
                start_time = time.time()
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i : i + batch_size]
                    batch_number = i // batch_size + 1

                    batch_start_time = time.time()
                    try:
                        session.bulk_update_mappings(model_class, batch)
                        session.flush()
                        batch_end_time = time.time()
                        log_database_operation(
                            "bulk_update_batch",
                            duration=batch_end_time - batch_start_time,
                            table_name=str(model_class.__table__.name),
                            batch_number=batch_number,
                            batch_size=len(batch),
                        )
                    except Exception as batch_error:
                        operation_logger.error(
                            "Bulk update batch failed",
                            batch_number=batch_number,
                            batch_size=len(batch),
                            error=str(batch_error),
                        )
                        raise

            operation_logger.info("Bulk update completed successfully")
        except Exception as e:
            converted_error = handle_database_exception(e)
            operation_logger.error(
                "Bulk update operation failed", extra={"error": str(converted_error)}
            )
            raise converted_error from e

    def bulk_delete(self, model_class, filter_conditions: list, batch_size: int = 1000):
        """
        大量データの一括削除

        Args:
            model_class: 削除するモデルクラス
            filter_conditions: 削除条件のリスト
            batch_size: バッチサイズ

        Raises:
            DatabaseError: 一括削除に失敗した場合
        """
        if not filter_conditions:
            return

        operation_logger = logger
        operation_logger.info("Starting bulk delete")

        try:
            with self.transaction_manager.transaction_scope() as session:
                query = session.query(model_class)
                for condition in filter_conditions:
                    query = query.filter(condition)

                # 削除前に対象レコード数を取得
                total_count = query.count()
                operation_logger.info(f"Deleting {total_count} records")

                # バッチ削除実行
                deleted_count = 0
                while True:
                    # 一度に削除する量を制限
                    subquery = query.limit(batch_size)
                    batch_records = subquery.all()

                    if not batch_records:
                        break

                    # バッチ削除
                    for record in batch_records:
                        session.delete(record)

                    session.flush()
                    deleted_count += len(batch_records)

                    operation_logger.debug(
                        f"Deleted batch: {len(batch_records)} records, "
                        f"total deleted: {deleted_count}/{total_count}"
                    )

            operation_logger.info(f"Bulk delete completed: {deleted_count} records deleted")
        except Exception as e:
            converted_error = handle_database_exception(e)
            operation_logger.error(
                "Bulk delete operation failed", extra={"error": str(converted_error)}
            )
            raise converted_error from e