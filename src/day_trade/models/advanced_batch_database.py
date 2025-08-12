#!/usr/bin/env python3
"""
高度バッチデータベース処理システム
Issue #376: バッチ処理の強化 - データベースI/O最適化

大量データの効率的なバッチ処理とI/O最適化を実現
"""

import asyncio
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# プロジェクトモジュール
try:
    from ..utils.exceptions import DatabaseError
    from ..utils.logging_config import get_context_logger, log_performance_metric
    from .bulk_operations import BulkOperationManager
    from .database import DatabaseManager, get_database_manager
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    def get_database_manager():
        return None

    class DatabaseManager:
        def __init__(self):
            pass

        def get_session(self):
            return None

    class DatabaseError(Exception):
        pass

    class BulkOperationManager:
        def __init__(self, db):
            pass


logger = get_context_logger(__name__)


class BatchOperationType(Enum):
    """バッチ操作タイプ"""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    UPSERT = "upsert"
    BULK_READ = "bulk_read"
    AGGREGATE = "aggregate"


class OptimizationLevel(Enum):
    """最適化レベル"""

    BASIC = "basic"  # 基本的なバッチ処理
    ADVANCED = "advanced"  # 高度な最適化（インデックス、統計）
    EXTREME = "extreme"  # 極限最適化（WAL、並列処理）


@dataclass
class BatchOperation:
    """バッチ操作定義"""

    operation_type: BatchOperationType
    table_name: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    operation_id: str = field(
        default_factory=lambda: f"batch_{int(time.time() * 1000)}"
    )

    def size(self) -> int:
        """操作サイズ"""
        return len(self.data)


@dataclass
class BatchResult:
    """バッチ処理結果"""

    operation_id: str
    operation_type: BatchOperationType
    success: bool
    affected_rows: int
    processing_time_ms: float
    error_message: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    completed_at: float = field(default_factory=time.time)


class DatabaseConnectionPool:
    """データベース接続プール（高度版）"""

    def __init__(self, db_manager, pool_size: int = 10):
        self.db_manager = db_manager
        self.pool_size = pool_size
        self._pool = deque()
        self._pool_lock = threading.RLock()
        self._active_connections = 0
        self._stats = {
            "total_requests": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "active_connections": 0,
        }

        # プール初期化
        self._initialize_pool()

    def _initialize_pool(self):
        """プール初期化"""
        try:
            for _ in range(min(3, self.pool_size)):  # 初期接続数
                session = self.db_manager.get_session()
                self._pool.append(session)
                logger.debug(f"データベース接続プール初期化: {len(self._pool)}接続")
        except Exception as e:
            logger.error(f"接続プール初期化失敗: {e}")

    @contextmanager
    def get_connection(self):
        """接続取得（コンテキストマネージャー）"""
        session = None
        try:
            with self._pool_lock:
                self._stats["total_requests"] += 1

                if self._pool:
                    session = self._pool.popleft()
                    self._stats["pool_hits"] += 1
                else:
                    session = self.db_manager.get_session()
                    self._stats["pool_misses"] += 1

                self._active_connections += 1
                self._stats["active_connections"] = self._active_connections

            yield session

        except Exception:
            if session:
                try:
                    session.rollback()
                except Exception:
                    pass
            raise
        finally:
            with self._pool_lock:
                self._active_connections -= 1
                self._stats["active_connections"] = self._active_connections

                if session and len(self._pool) < self.pool_size:
                    try:
                        # 接続の健全性チェック
                        session.execute(text("SELECT 1"))
                        self._pool.append(session)
                    except Exception:
                        session.close()
                elif session:
                    session.close()

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self._pool_lock:
            return self._stats.copy()

    def cleanup(self):
        """プールクリーンアップ"""
        with self._pool_lock:
            while self._pool:
                session = self._pool.popleft()
                try:
                    session.close()
                except Exception:
                    pass


class SQLOptimizer:
    """SQL最適化エンジン"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.dialect = db_manager.engine.dialect.name
        self._table_stats = {}
        self._index_cache = {}

    def optimize_insert_statement(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        optimization_level: OptimizationLevel,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """INSERT文最適化"""
        if not data:
            return "", []

        if optimization_level == OptimizationLevel.EXTREME and self.dialect == "sqlite":
            # SQLite極限最適化
            return self._optimize_sqlite_insert(table_name, data)
        elif optimization_level in [
            OptimizationLevel.ADVANCED,
            OptimizationLevel.EXTREME,
        ]:
            # 高度最適化
            return self._optimize_advanced_insert(table_name, data)
        else:
            # 基本最適化
            return self._optimize_basic_insert(table_name, data)

    def _optimize_sqlite_insert(
        self, table_name: str, data: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """SQLite極限最適化INSERT"""
        columns = list(data[0].keys())
        placeholders = ", ".join([f":{col}" for col in columns])

        # 複数レコードを一度にINSERT
        if len(data) > 1000:
            # 大量データ用VALUES構文
            values_list = []
            for i in range(0, len(data), 1000):  # 1000件ずつ
                batch = data[i : i + 1000]
                batch_values = []
                for row in batch:
                    row_values = ", ".join(
                        [f":{col}{j}" for j, col in enumerate(columns)]
                    )
                    batch_values.append(f"({row_values})")
                values_list.extend(batch_values)

            sql = f"""
            INSERT OR IGNORE INTO {table_name} ({', '.join(columns)})
            VALUES {', '.join(values_list[:1000])}
            """
        else:
            sql = f"INSERT OR IGNORE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

        return sql, data

    def _optimize_advanced_insert(
        self, table_name: str, data: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """高度最適化INSERT"""
        columns = list(data[0].keys())

        if self.dialect == "postgresql":
            # PostgreSQL COPY最適化
            placeholders = ", ".join([f"%({col})s" for col in columns])
            sql = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
            """
        elif self.dialect == "sqlite":
            placeholders = ", ".join([f":{col}" for col in columns])
            sql = f"INSERT OR IGNORE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        else:
            # MySQL
            placeholders = ", ".join([f"%({col})s" for col in columns])
            sql = f"""
            INSERT IGNORE INTO {table_name} ({', '.join(columns)})
            VALUES ({placeholders})
            """

        return sql, data

    def _optimize_basic_insert(
        self, table_name: str, data: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """基本最適化INSERT"""
        columns = list(data[0].keys())
        placeholders = ", ".join([f":{col}" for col in columns])
        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

        return sql, data

    def analyze_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """テーブル統計解析"""
        try:
            with self.db_manager.get_session() as session:
                # レコード数
                count_result = session.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                )
                row_count = count_result.scalar()

                # テーブルサイズ（SQLiteの場合）
                table_size = 0
                if self.dialect == "sqlite":
                    size_result = session.execute(
                        text(
                            f"SELECT SUM(pgsize) FROM dbstat WHERE name = '{table_name}'"
                        )
                    )
                    table_size = size_result.scalar() or 0

                stats = {
                    "row_count": row_count,
                    "table_size_bytes": table_size,
                    "analyzed_at": time.time(),
                }

                self._table_stats[table_name] = stats
                return stats

        except Exception as e:
            logger.warning(f"テーブル統計解析失敗 {table_name}: {e}")
            return {"row_count": 0, "table_size_bytes": 0, "analyzed_at": time.time()}

    def recommend_batch_size(self, table_name: str, data_size: int) -> int:
        """最適バッチサイズ推奨"""
        stats = self._table_stats.get(table_name)
        if not stats:
            stats = self.analyze_table_statistics(table_name)

        row_count = stats.get("row_count", 0)

        # テーブルサイズに応じた動的調整
        if row_count < 10000:
            return min(1000, data_size)  # 小さなテーブル
        elif row_count < 100000:
            return min(500, data_size)  # 中サイズテーブル
        else:
            return min(200, data_size)  # 大きなテーブル


class AdvancedBatchDatabase:
    """高度バッチデータベース処理システム"""

    def __init__(
        self,
        db_manager: DatabaseManager = None,
        optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED,
        connection_pool_size: int = 10,
        max_batch_size: int = 1000,
    ):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
            optimization_level: 最適化レベル
            connection_pool_size: 接続プールサイズ
            max_batch_size: 最大バッチサイズ
        """
        self.db_manager = db_manager or get_database_manager()
        self.optimization_level = optimization_level
        self.max_batch_size = max_batch_size

        if not self.db_manager:
            raise DatabaseError("DatabaseManager が必要です")

        # コンポーネント初期化
        self.connection_pool = DatabaseConnectionPool(
            self.db_manager, connection_pool_size
        )
        self.sql_optimizer = SQLOptimizer(self.db_manager)

        # バッチキュー
        self._batch_queue = deque()
        self._queue_lock = threading.RLock()

        # 統計情報
        self.stats = {
            "operations_processed": 0,
            "total_rows_affected": 0,
            "average_processing_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "optimization_savings_percent": 0.0,
        }
        self._stats_lock = threading.RLock()

        # パフォーマンス最適化設定
        self._setup_database_optimizations()

        logger.info(
            f"AdvancedBatchDatabase初期化完了: "
            f"optimization_level={optimization_level.value}, "
            f"pool_size={connection_pool_size}"
        )

    def _setup_database_optimizations(self):
        """データベース最適化設定"""
        if self.optimization_level in [
            OptimizationLevel.ADVANCED,
            OptimizationLevel.EXTREME,
        ]:
            try:
                with self.connection_pool.get_connection() as session:
                    # SQLite最適化
                    if self.db_manager.engine.dialect.name == "sqlite":
                        optimizations = [
                            "PRAGMA journal_mode = WAL",
                            "PRAGMA synchronous = NORMAL",
                            "PRAGMA cache_size = 10000",
                            "PRAGMA temp_store = memory",
                            "PRAGMA mmap_size = 268435456",  # 256MB
                        ]

                        if self.optimization_level == OptimizationLevel.EXTREME:
                            optimizations.extend(
                                [
                                    "PRAGMA locking_mode = EXCLUSIVE",
                                    "PRAGMA count_changes = OFF",
                                    "PRAGMA checkpoint_fullfsync = OFF",
                                ]
                            )

                        for pragma in optimizations:
                            session.execute(text(pragma))
                            logger.debug(f"SQLite最適化適用: {pragma}")

                    session.commit()

            except Exception as e:
                logger.warning(f"データベース最適化設定失敗: {e}")

    async def execute_batch_operation(self, operation: BatchOperation) -> BatchResult:
        """バッチ操作実行"""
        start_time = time.time()

        try:
            if operation.operation_type == BatchOperationType.INSERT:
                result = await self._execute_batch_insert(operation)
            elif operation.operation_type == BatchOperationType.UPDATE:
                result = await self._execute_batch_update(operation)
            elif operation.operation_type == BatchOperationType.DELETE:
                result = await self._execute_batch_delete(operation)
            elif operation.operation_type == BatchOperationType.UPSERT:
                result = await self._execute_batch_upsert(operation)
            elif operation.operation_type == BatchOperationType.BULK_READ:
                result = await self._execute_bulk_read(operation)
            else:
                raise ValueError(f"未サポート操作タイプ: {operation.operation_type}")

            processing_time_ms = (time.time() - start_time) * 1000

            # パフォーマンス統計更新
            self._update_stats(processing_time_ms, result.affected_rows, True)

            log_performance_metric(
                f"batch_{operation.operation_type.value}", processing_time_ms, "ms"
            )

            result.processing_time_ms = processing_time_ms
            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats(processing_time_ms, 0, False)

            logger.error(f"バッチ操作エラー ({operation.operation_id}): {e}")

            return BatchResult(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                success=False,
                affected_rows=0,
                processing_time_ms=processing_time_ms,
                error_message=str(e),
            )

    async def _execute_batch_insert(self, operation: BatchOperation) -> BatchResult:
        """バッチINSERT実行"""
        if not operation.data:
            return BatchResult(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                success=True,
                affected_rows=0,
                processing_time_ms=0,
            )

        # バッチサイズ最適化
        recommended_batch_size = self.sql_optimizer.recommend_batch_size(
            operation.table_name, len(operation.data)
        )

        total_affected = 0

        # データをチャンクに分割
        for i in range(0, len(operation.data), recommended_batch_size):
            chunk = operation.data[i : i + recommended_batch_size]

            # SQL最適化
            (
                optimized_sql,
                optimized_data,
            ) = self.sql_optimizer.optimize_insert_statement(
                operation.table_name, chunk, self.optimization_level
            )

            # 実行
            with self.connection_pool.get_connection() as session:
                try:
                    if self.optimization_level == OptimizationLevel.EXTREME:
                        # 極限最適化：トランザクション分離
                        session.execute(text("BEGIN IMMEDIATE"))

                    # バルクインサート実行
                    result = session.execute(text(optimized_sql), optimized_data)
                    affected_rows = (
                        result.rowcount if hasattr(result, "rowcount") else len(chunk)
                    )

                    session.commit()
                    total_affected += affected_rows

                    logger.debug(
                        f"バッチINSERT完了: {len(chunk)}件, 影響行数: {affected_rows}"
                    )

                except IntegrityError:
                    session.rollback()
                    # 整合性エラーは個別処理で回復試行
                    individual_affected = await self._fallback_individual_insert(
                        session, operation.table_name, chunk
                    )
                    total_affected += individual_affected
                    logger.warning(
                        f"バッチINSERT整合性エラー、個別処理で{individual_affected}件処理"
                    )

                except Exception:
                    session.rollback()
                    raise

        return BatchResult(
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            success=True,
            affected_rows=total_affected,
            processing_time_ms=0,  # 後で設定される
            statistics={"recommended_batch_size": recommended_batch_size},
        )

    async def _fallback_individual_insert(
        self, session: Session, table_name: str, data: List[Dict[str, Any]]
    ) -> int:
        """個別INSERT フォールバック"""
        affected_count = 0
        columns = list(data[0].keys())
        placeholders = ", ".join([f":{col}" for col in columns])
        sql = f"INSERT OR IGNORE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

        for row in data:
            try:
                result = session.execute(text(sql), row)
                if result.rowcount > 0:
                    affected_count += 1
            except Exception as e:
                logger.debug(f"個別INSERT失敗: {e}")
                continue

        session.commit()
        return affected_count

    async def _execute_batch_update(self, operation: BatchOperation) -> BatchResult:
        """バッチUPDATE実行"""
        # 実装はINSERTと類似だが、UPDATE用に最適化
        total_affected = 0

        with self.connection_pool.get_connection() as session:
            for row in operation.data:
                # 更新条件とデータの分離
                conditions = {k: v for k, v in row.items() if k in operation.conditions}
                update_data = {
                    k: v for k, v in row.items() if k not in operation.conditions
                }

                if not conditions or not update_data:
                    continue

                # 動的UPDATE文生成
                set_clause = ", ".join(
                    [f"{col} = :{col}" for col in update_data.keys()]
                )
                where_clause = " AND ".join(
                    [f"{col} = :where_{col}" for col in conditions.keys()]
                )

                sql = f"UPDATE {operation.table_name} SET {set_clause} WHERE {where_clause}"

                # パラメータ準備
                params = {**update_data}
                params.update({f"where_{k}": v for k, v in conditions.items()})

                try:
                    result = session.execute(text(sql), params)
                    total_affected += result.rowcount
                except Exception as e:
                    logger.debug(f"UPDATE失敗: {e}")

            session.commit()

        return BatchResult(
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            success=True,
            affected_rows=total_affected,
            processing_time_ms=0,
        )

    async def _execute_batch_delete(self, operation: BatchOperation) -> BatchResult:
        """バッチDELETE実行"""
        if not operation.conditions:
            raise ValueError("DELETE操作には条件が必要です")

        total_affected = 0

        with self.connection_pool.get_connection() as session:
            # 条件からWHERE句生成
            where_clauses = []
            params = {}

            for key, value in operation.conditions.items():
                if isinstance(value, list):
                    # IN句
                    placeholders = ", ".join(
                        [f":param_{key}_{i}" for i in range(len(value))]
                    )
                    where_clauses.append(f"{key} IN ({placeholders})")
                    for i, val in enumerate(value):
                        params[f"param_{key}_{i}"] = val
                else:
                    where_clauses.append(f"{key} = :{key}")
                    params[key] = value

            where_clause = " AND ".join(where_clauses)
            sql = f"DELETE FROM {operation.table_name} WHERE {where_clause}"

            try:
                result = session.execute(text(sql), params)
                total_affected = result.rowcount
                session.commit()
            except Exception:
                session.rollback()
                raise

        return BatchResult(
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            success=True,
            affected_rows=total_affected,
            processing_time_ms=0,
        )

    async def _execute_batch_upsert(self, operation: BatchOperation) -> BatchResult:
        """バッチUPSERT実行（INSERT + UPDATE）"""
        # データベース依存のUPSERT実装
        dialect = self.db_manager.engine.dialect.name

        if dialect == "sqlite":
            return await self._execute_sqlite_upsert(operation)
        elif dialect == "postgresql":
            return await self._execute_postgresql_upsert(operation)
        else:
            # フォールバック：INSERT試行 → 失敗時UPDATE
            return await self._execute_fallback_upsert(operation)

    async def _execute_sqlite_upsert(self, operation: BatchOperation) -> BatchResult:
        """SQLite UPSERT実行"""
        if not operation.data:
            return BatchResult(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                success=True,
                affected_rows=0,
                processing_time_ms=0,
            )

        columns = list(operation.data[0].keys())
        conflict_columns = operation.options.get("conflict_columns", ["id"])

        # INSERT ... ON CONFLICT構文
        placeholders = ", ".join([f":{col}" for col in columns])
        update_set = ", ".join(
            [
                f"{col} = excluded.{col}"
                for col in columns
                if col not in conflict_columns
            ]
        )

        sql = f"""
        INSERT INTO {operation.table_name} ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT({', '.join(conflict_columns)}) DO UPDATE SET {update_set}
        """

        total_affected = 0

        with self.connection_pool.get_connection() as session:
            try:
                for row in operation.data:
                    result = session.execute(text(sql), row)
                    total_affected += 1  # UPSERTは常に1行影響

                session.commit()
            except Exception:
                session.rollback()
                raise

        return BatchResult(
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            success=True,
            affected_rows=total_affected,
            processing_time_ms=0,
        )

    async def _execute_postgresql_upsert(
        self, operation: BatchOperation
    ) -> BatchResult:
        """PostgreSQL UPSERT実行"""
        # PostgreSQL ON CONFLICT実装
        return await self._execute_sqlite_upsert(operation)  # 同じロジック

    async def _execute_fallback_upsert(self, operation: BatchOperation) -> BatchResult:
        """フォールバックUPSERT実行"""
        # INSERT試行 → 失敗時UPDATE
        insert_operation = BatchOperation(
            operation_type=BatchOperationType.INSERT,
            table_name=operation.table_name,
            data=operation.data,
            operation_id=f"{operation.operation_id}_insert",
        )

        insert_result = await self._execute_batch_insert(insert_operation)

        # 失敗した分をUPDATEで処理
        failed_count = len(operation.data) - insert_result.affected_rows
        if failed_count > 0:
            update_operation = BatchOperation(
                operation_type=BatchOperationType.UPDATE,
                table_name=operation.table_name,
                data=operation.data[-failed_count:],  # 失敗分
                conditions=operation.options.get("update_conditions", {}),
                operation_id=f"{operation.operation_id}_update",
            )

            update_result = await self._execute_batch_update(update_operation)
            total_affected = insert_result.affected_rows + update_result.affected_rows
        else:
            total_affected = insert_result.affected_rows

        return BatchResult(
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            success=True,
            affected_rows=total_affected,
            processing_time_ms=0,
        )

    async def _execute_bulk_read(self, operation: BatchOperation) -> BatchResult:
        """バルク読み込み実行"""
        conditions = operation.conditions
        options = operation.options

        with self.connection_pool.get_connection() as session:
            # 動的SELECT文生成
            select_columns = options.get("columns", ["*"])
            columns_str = ", ".join(select_columns) if select_columns != ["*"] else "*"

            base_sql = f"SELECT {columns_str} FROM {operation.table_name}"

            # WHERE句生成
            if conditions:
                where_clauses = []
                params = {}

                for key, value in conditions.items():
                    if isinstance(value, list):
                        placeholders = ", ".join(
                            [f":param_{key}_{i}" for i in range(len(value))]
                        )
                        where_clauses.append(f"{key} IN ({placeholders})")
                        for i, val in enumerate(value):
                            params[f"param_{key}_{i}"] = val
                    else:
                        where_clauses.append(f"{key} = :{key}")
                        params[key] = value

                where_clause = " AND ".join(where_clauses)
                sql = f"{base_sql} WHERE {where_clause}"
            else:
                sql = base_sql
                params = {}

            # LIMIT/OFFSET
            if "limit" in options:
                sql += f" LIMIT {options['limit']}"
            if "offset" in options:
                sql += f" OFFSET {options['offset']}"

            try:
                result = session.execute(text(sql), params)
                rows = result.fetchall()

                # 辞書形式に変換
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in rows]

                return BatchResult(
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type,
                    success=True,
                    affected_rows=len(data),
                    processing_time_ms=0,
                    statistics={"data": data},
                )

            except Exception:
                raise

    def _update_stats(
        self, processing_time_ms: float, affected_rows: int, success: bool
    ):
        """統計更新"""
        with self._stats_lock:
            self.stats["operations_processed"] += 1
            if success:
                self.stats["total_rows_affected"] += affected_rows

            # 移動平均
            alpha = 0.1
            self.stats["average_processing_time_ms"] = (
                self.stats["average_processing_time_ms"] * (1 - alpha)
                + processing_time_ms * alpha
            )

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self._stats_lock:
            stats = self.stats.copy()

        # 接続プール統計追加
        stats["connection_pool"] = self.connection_pool.get_stats()

        return stats

    def get_health_status(self) -> Dict[str, Any]:
        """ヘルスステータス"""
        stats = self.get_stats()

        pool_stats = stats.get("connection_pool", {})
        pool_utilization = (
            pool_stats.get("active_connections", 0) / self.connection_pool.pool_size
        )

        health = "healthy"
        if pool_utilization > 0.8:
            health = "degraded"
        if pool_utilization > 0.95:
            health = "critical"

        return {
            "status": health,
            "pool_utilization": pool_utilization,
            "operations_processed": stats["operations_processed"],
            "average_processing_time_ms": stats["average_processing_time_ms"],
            "optimization_level": self.optimization_level.value,
        }

    def cleanup(self):
        """クリーンアップ"""
        try:
            self.connection_pool.cleanup()
            logger.info("AdvancedBatchDatabase クリーンアップ完了")
        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")


# 便利関数
async def batch_insert_stocks(
    stock_data: List[Dict[str, Any]], batch_db: AdvancedBatchDatabase = None
) -> BatchResult:
    """株価データバッチINSERT"""
    if not batch_db:
        db_manager = get_database_manager()
        batch_db = AdvancedBatchDatabase(db_manager)

    operation = BatchOperation(
        operation_type=BatchOperationType.INSERT, table_name="stocks", data=stock_data
    )

    return await batch_db.execute_batch_operation(operation)


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #376 高度バッチデータベース処理テスト ===")

        # ダミーデータ生成
        test_data = [
            {"symbol": "AAPL", "price": 150.0, "volume": 1000000},
            {"symbol": "GOOGL", "price": 2500.0, "volume": 500000},
            {"symbol": "MSFT", "price": 300.0, "volume": 750000},
        ]

        try:
            db_manager = get_database_manager()
            if db_manager:
                batch_db = AdvancedBatchDatabase(
                    db_manager, optimization_level=OptimizationLevel.ADVANCED
                )

                # INSERT操作テスト
                print("\n1. バッチINSERTテスト")
                operation = BatchOperation(
                    operation_type=BatchOperationType.INSERT,
                    table_name="test_stocks",
                    data=test_data,
                )

                result = await batch_db.execute_batch_operation(operation)
                print(
                    f"処理結果: success={result.success}, affected_rows={result.affected_rows}"
                )
                print(f"処理時間: {result.processing_time_ms:.2f}ms")

                # 統計情報
                print("\n2. 統計情報")
                stats = batch_db.get_stats()
                for key, value in stats.items():
                    print(f"  {key}: {value}")

                # ヘルスステータス
                print("\n3. ヘルスステータス")
                health = batch_db.get_health_status()
                for key, value in health.items():
                    print(f"  {key}: {value}")

                batch_db.cleanup()
            else:
                print("DatabaseManagerが利用できません（モック実行）")

        except Exception as e:
            print(f"テスト実行エラー: {e}")

    asyncio.run(main())
    print("\n=== 高度バッチデータベース処理テスト完了 ===")
