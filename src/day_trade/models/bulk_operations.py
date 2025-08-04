"""
高性能バルク操作モジュール（SQLAlchemy 2.0対応）
大量データの効率的な挿入・更新・削除を提供
"""

import logging
import time
from typing import Any, Dict, List, Optional, Type, Union
from contextlib import contextmanager

from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .database import DatabaseManager, Base
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="bulk_operations")


class BulkOperationError(Exception):
    """バルク操作固有のエラー"""

    def __init__(self, message: str, operation: str, affected_records: int = 0, details: Optional[Dict] = None):
        super().__init__(message)
        self.operation = operation
        self.affected_records = affected_records
        self.details = details or {}


class AdvancedBulkOperations:
    """高度なバルク操作クラス（SQLAlchemy 2.0対応）"""

    def __init__(self, db_manager: DatabaseManager):
        """
        Args:
            db_manager: DatabaseManagerインスタンス
        """
        self.db_manager = db_manager
        self.logger = logger.bind(component="bulk_operations")

    def bulk_insert_with_conflict_resolution(
        self,
        model_class: Type[Base],
        data: List[Dict[str, Any]],
        conflict_strategy: str = "ignore",
        chunk_size: int = 1000,
        unique_columns: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        競合解決付きバルク挿入

        Args:
            model_class: モデルクラス
            data: 挿入するデータ
            conflict_strategy: 競合時の戦略（ignore, update, error）
            chunk_size: チャンクサイズ
            unique_columns: ユニーク制約のカラム名リスト

        Returns:
            結果統計（inserted, updated, skipped, errors）
        """
        if not data:
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}
        table_name = model_class.__tablename__

        import time
        start_time = time.perf_counter()
        try:
            with self.db_manager.session_scope() as session:
                    if conflict_strategy == "ignore":
                        stats = self._bulk_insert_ignore(session, model_class, data, chunk_size)
                    elif conflict_strategy == "update":
                        stats = self._bulk_upsert(session, model_class, data, chunk_size, unique_columns)
                    elif conflict_strategy == "error":
                        stats = self._bulk_insert_strict(session, model_class, data, chunk_size)
                    else:
                        raise BulkOperationError(
                            f"Unknown conflict strategy: {conflict_strategy}",
                            "bulk_insert_with_conflict_resolution"
                        )

            elapsed_time = time.perf_counter() - start_time
            self.logger.info(
                "Bulk insert with conflict resolution completed",
                table=table_name,
                strategy=conflict_strategy,
                elapsed_ms=round(elapsed_time * 1000, 2),
                **stats
            )

        except Exception as e:
            self.logger.error(
                "Bulk insert with conflict resolution failed",
                table=table_name,
                strategy=conflict_strategy,
                error=str(e)
            )
            raise BulkOperationError(
                f"Bulk insert failed: {str(e)}",
                "bulk_insert_with_conflict_resolution",
                details={"table": table_name, "strategy": conflict_strategy}
            ) from e

        return stats

    def _bulk_insert_ignore(
        self,
        session: Session,
        model_class: Type[Base],
        data: List[Dict],
        chunk_size: int
    ) -> Dict[str, int]:
        """競合を無視するバルク挿入"""
        stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            try:
                session.bulk_insert_mappings(model_class, chunk)
                stats["inserted"] += len(chunk)
            except IntegrityError:
                # 個別に処理
                for item in chunk:
                    try:
                        session.bulk_insert_mappings(model_class, [item])
                        stats["inserted"] += 1
                    except IntegrityError:
                        stats["skipped"] += 1
                    except Exception:
                        stats["errors"] += 1

        return stats

    def _bulk_upsert(
        self,
        session: Session,
        model_class: Type[Base],
        data: List[Dict],
        chunk_size: int,
        unique_columns: Optional[List[str]]
    ) -> Dict[str, int]:
        """アップサート（挿入または更新）"""
        stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        if not unique_columns:
            # 主キーを使用
            unique_columns = [pk.name for pk in inspect(model_class).primary_key]

        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]

            # 既存レコードを特定
            existing_keys = self._get_existing_keys(session, model_class, chunk, unique_columns)

            inserts = []
            updates = []

            for item in chunk:
                key_values = tuple(item.get(col) for col in unique_columns)
                if key_values in existing_keys:
                    updates.append(item)
                else:
                    inserts.append(item)

            # 挿入処理
            if inserts:
                try:
                    session.bulk_insert_mappings(model_class, inserts)
                    stats["inserted"] += len(inserts)
                except Exception as e:
                    self.logger.warning(f"Bulk insert failed in upsert: {e}")
                    stats["errors"] += len(inserts)

            # 更新処理
            if updates:
                try:
                    session.bulk_update_mappings(model_class, updates)
                    stats["updated"] += len(updates)
                except Exception as e:
                    self.logger.warning(f"Bulk update failed in upsert: {e}")
                    stats["errors"] += len(updates)

        return stats

    def _bulk_insert_strict(
        self,
        session: Session,
        model_class: Type[Base],
        data: List[Dict],
        chunk_size: int
    ) -> Dict[str, int]:
        """厳密なバルク挿入（競合時はエラー）"""
        stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            session.bulk_insert_mappings(model_class, chunk)
            stats["inserted"] += len(chunk)

        return stats

    def _get_existing_keys(
        self,
        session: Session,
        model_class: Type[Base],
        data: List[Dict],
        unique_columns: List[str]
    ) -> set:
        """既存キーを効率的に取得"""
        if not data or not unique_columns:
            return set()

        # IN句を使って既存キーを一括取得
        key_values = [
            tuple(item.get(col) for col in unique_columns)
            for item in data
        ]

        # クエリを構築
        columns = [getattr(model_class, col) for col in unique_columns]
        query = session.query(*columns)

        if len(unique_columns) == 1:
            # 単一カラムの場合
            values = [kv[0] for kv in key_values if kv[0] is not None]
            if values:
                query = query.filter(columns[0].in_(values))
        else:
            # 複数カラムの場合（tuple inを使用）
            from sqlalchemy import tuple_
            valid_tuples = [kv for kv in key_values if all(v is not None for v in kv)]
            if valid_tuples:
                query = query.filter(tuple_(*columns).in_(valid_tuples))

        return set(query.all())

    def bulk_delete_with_conditions(
        self,
        model_class: Type[Base],
        conditions: Dict[str, Any],
        chunk_size: int = 1000,
        dry_run: bool = False
    ) -> Dict[str, int]:
        """
        条件付きバルク削除

        Args:
            model_class: モデルクラス
            conditions: 削除条件
            chunk_size: チャンクサイズ
            dry_run: 実際には削除せず、削除対象数のみ返す

        Returns:
            削除統計
        """
        table_name = model_class.__tablename__

        start_time = time.perf_counter()
        try:
            with self.db_manager.session_scope() as session:
                # 削除対象を特定
                query = session.query(model_class)
                for column, value in conditions.items():
                    if hasattr(model_class, column):
                        attr = getattr(model_class, column)
                        if isinstance(value, (list, tuple)):
                            query = query.filter(attr.in_(value))
                        else:
                            query = query.filter(attr == value)

                if dry_run:
                    count = query.count()
                    return {"deleted": 0, "would_delete": count}

                # 実際の削除処理
                deleted_count = query.delete(synchronize_session=False)

                elapsed_time = time.perf_counter() - start_time
                self.logger.info(
                    "Bulk delete completed",
                    table=table_name,
                    deleted=deleted_count,
                    elapsed_ms=round(elapsed_time * 1000, 2),
                    conditions=conditions
                )

                return {"deleted": deleted_count}

        except Exception as e:
            self.logger.error(
                "Bulk delete failed",
                table=table_name,
                conditions=conditions,
                error=str(e)
            )
            raise BulkOperationError(
                f"Bulk delete failed: {str(e)}",
                "bulk_delete_with_conditions",
                details={"table": table_name, "conditions": conditions}
            ) from e

    @contextmanager
    def batch_processor(self, batch_size: int = 1000, commit_interval: int = 10):
        """
        バッチ処理コンテキストマネージャー
        大量データを効率的に処理するためのヘルパー
        """
        processed = 0
        batch_count = 0

        class BatchProcessor:
            def __init__(self, db_manager, batch_size, commit_interval, logger):
                self.db_manager = db_manager
                self.batch_size = batch_size
                self.commit_interval = commit_interval
                self.logger = logger
                self.session = None
                self.processed = 0
                self.batch_count = 0

            def __enter__(self):
                self.session = self.db_manager.get_session()
                return self

            def process_batch(self, operation_func, data_batch):
                """バッチを処理"""
                try:
                    operation_func(self.session, data_batch)
                    self.processed += len(data_batch)
                    self.batch_count += 1

                    # 定期的にコミット
                    if self.batch_count % self.commit_interval == 0:
                        self.session.commit()
                        self.logger.debug(
                            "Intermediate commit",
                            processed=self.processed,
                            batch_count=self.batch_count
                        )

                except Exception as e:
                    self.session.rollback()
                    self.logger.error(f"Batch processing failed: {e}")
                    raise

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.session:
                    try:
                        if exc_type is None:
                            self.session.commit()
                            self.logger.info(
                                "Batch processing completed",
                                total_processed=self.processed,
                                total_batches=self.batch_count
                            )
                        else:
                            self.session.rollback()
                    finally:
                        self.session.close()

        yield BatchProcessor(self.db_manager, batch_size, commit_interval, self.logger)


# 便利な関数
def create_bulk_operations(db_manager: Optional[DatabaseManager] = None) -> AdvancedBulkOperations:
    """AdvancedBulkOperationsインスタンスを作成"""
    if db_manager is None:
        from .database import db_manager as default_db_manager
        db_manager = default_db_manager

    return AdvancedBulkOperations(db_manager)
