#!/usr/bin/env python3
"""
データベースバルクオペレーション最適化システム
大量データの効率的なデータベース操作を提供

主要機能:
- 大量データの高速一括挿入/更新
- トランザクション最適化
- 並列バッチ処理
- 自動インデックス最適化
- データベース接続プーリング
- メモリ効率的なストリーミング処理
"""

import csv
import io
import sqlite3
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class BulkOperation(Enum):
    """バルク操作タイプ"""

    INSERT = "insert"
    UPDATE = "update"
    UPSERT = "upsert"
    DELETE = "delete"
    MERGE = "merge"


class OptimizationStrategy(Enum):
    """最適化戦略"""

    MEMORY_OPTIMIZED = "memory"  # メモリ効率重視
    SPEED_OPTIMIZED = "speed"  # 速度重視
    BALANCED = "balanced"  # バランス重視
    STREAMING = "streaming"  # ストリーミング処理


@dataclass
class BulkOperationConfig:
    """バルク操作設定"""

    batch_size: int = 1000
    max_workers: int = 4
    transaction_size: int = 10000
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    enable_wal_mode: bool = True
    enable_synchronous_off: bool = True
    enable_journal_memory: bool = True
    vacuum_threshold_mb: int = 100
    analyze_after_bulk: bool = True
    create_temp_indexes: bool = True


@dataclass
class BulkOperationResult:
    """バルク操作結果"""

    operation: BulkOperation
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    success_rate: float = 0.0
    total_time: float = 0.0
    throughput_rps: float = 0.0
    memory_peak_mb: float = 0.0
    optimizations_applied: List[str] = field(default_factory=list)
    error_details: List[str] = field(default_factory=list)


class DatabaseConnectionPool:
    """データベース接続プール"""

    def __init__(self, db_path: str, pool_size: int = 8, enable_wal: bool = True):
        self.db_path = db_path
        self.pool_size = pool_size
        self.enable_wal = enable_wal
        self.connections = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """接続プール初期化"""
        for _ in range(self.pool_size):
            conn = self._create_optimized_connection()
            self.connections.put(conn)

    def _create_optimized_connection(self) -> sqlite3.Connection:
        """最適化されたDB接続作成"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)

        # WALモード有効化
        if self.enable_wal:
            conn.execute("PRAGMA journal_mode=WAL")

        # パフォーマンス最適化
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB

        return conn

    @contextmanager
    def get_connection(self):
        """接続取得（コンテキストマネージャー）"""
        conn = None
        try:
            conn = self.connections.get(timeout=5.0)
            yield conn
        finally:
            if conn:
                self.connections.put(conn)

    def close_all(self):
        """全接続クローズ"""
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                conn.close()
            except:
                break


class DatabaseBulkOptimizer:
    """
    データベースバルクオペレーション最適化システム

    大量データの効率的なデータベース操作を提供し、
    パフォーマンスとメモリ使用量を最適化
    """

    def __init__(
        self, db_path: str, pool_size: int = 8, config: BulkOperationConfig = None
    ):
        self.db_path = db_path
        self.pool_size = pool_size
        self.config = config or BulkOperationConfig()

        # 接続プール
        self.connection_pool = DatabaseConnectionPool(
            db_path=db_path, pool_size=pool_size, enable_wal=self.config.enable_wal_mode
        )

        # 並列処理
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # パフォーマンス統計
        self.total_operations = 0
        self.successful_operations = 0
        self.total_records_processed = 0
        self.total_processing_time = 0.0

        # 最適化キャッシュ
        self.table_schemas = {}
        self.index_cache = {}
        self.temp_indexes = set()

        logger.info(f"DB バルク最適化システム初期化: {db_path}, pool_size={pool_size}")

    def bulk_insert(
        self,
        table_name: str,
        data: Union[pd.DataFrame, List[Dict], Iterator[Dict]],
        columns: List[str] = None,
        conflict_resolution: str = "IGNORE",
        **kwargs,
    ) -> BulkOperationResult:
        """
        大量データ一括挿入

        Args:
            table_name: テーブル名
            data: 挿入データ（DataFrame, Dict List, Iterator）
            columns: 列名リスト
            conflict_resolution: 競合解決（IGNORE, REPLACE, ABORT）

        Returns:
            バルク操作結果
        """
        return self._execute_bulk_operation(
            operation=BulkOperation.INSERT,
            table_name=table_name,
            data=data,
            columns=columns,
            conflict_resolution=conflict_resolution,
            **kwargs,
        )

    def bulk_update(
        self,
        table_name: str,
        data: Union[pd.DataFrame, List[Dict]],
        where_columns: List[str],
        update_columns: List[str] = None,
        **kwargs,
    ) -> BulkOperationResult:
        """
        大量データ一括更新

        Args:
            table_name: テーブル名
            data: 更新データ
            where_columns: WHERE条件列
            update_columns: 更新対象列

        Returns:
            バルク操作結果
        """
        return self._execute_bulk_operation(
            operation=BulkOperation.UPDATE,
            table_name=table_name,
            data=data,
            where_columns=where_columns,
            update_columns=update_columns,
            **kwargs,
        )

    def bulk_upsert(
        self,
        table_name: str,
        data: Union[pd.DataFrame, List[Dict]],
        key_columns: List[str],
        **kwargs,
    ) -> BulkOperationResult:
        """
        大量データ一括UPSERT（INSERT or UPDATE）

        Args:
            table_name: テーブル名
            data: データ
            key_columns: キー列

        Returns:
            バルク操作結果
        """
        return self._execute_bulk_operation(
            operation=BulkOperation.UPSERT,
            table_name=table_name,
            data=data,
            key_columns=key_columns,
            **kwargs,
        )

    def _execute_bulk_operation(
        self,
        operation: BulkOperation,
        table_name: str,
        data: Union[pd.DataFrame, List[Dict], Iterator[Dict]],
        **kwargs,
    ) -> BulkOperationResult:
        """バルク操作実行"""
        start_time = time.time()
        result = BulkOperationResult(operation=operation)

        try:
            # データサイズ推定
            total_records = self._estimate_data_size(data)
            result.total_records = total_records

            logger.info(
                f"バルク{operation.value}開始: {table_name} - {total_records} records"
            )

            # 前処理最適化
            self._apply_pre_optimizations(table_name, operation, result)

            # データ処理戦略選択
            if self.config.optimization_strategy == OptimizationStrategy.STREAMING:
                processed = self._execute_streaming_operation(
                    operation, table_name, data, result, **kwargs
                )
            else:
                processed = self._execute_batched_operation(
                    operation, table_name, data, result, **kwargs
                )

            result.processed_records = processed

            # 後処理最適化
            self._apply_post_optimizations(table_name, operation, result)

            # 結果計算
            result.total_time = time.time() - start_time
            result.success_rate = (
                result.processed_records / result.total_records
                if result.total_records > 0
                else 0.0
            )
            result.throughput_rps = (
                result.processed_records / result.total_time
                if result.total_time > 0
                else 0.0
            )

            # 統計更新
            self._update_stats(result)

            logger.info(
                f"バルク{operation.value}完了: {result.processed_records}/{result.total_records} "
                f"records, {result.total_time:.2f}s, {result.throughput_rps:.1f} RPS"
            )

        except Exception as e:
            result.total_time = time.time() - start_time
            result.error_details.append(str(e))
            logger.error(f"バルク{operation.value}エラー: {e}")

        return result

    def _execute_batched_operation(
        self,
        operation: BulkOperation,
        table_name: str,
        data: Union[pd.DataFrame, List[Dict], Iterator[Dict]],
        result: BulkOperationResult,
        **kwargs,
    ) -> int:
        """バッチ処理による操作実行"""
        processed_records = 0

        # データをバッチに分割
        batches = self._create_batches(data, self.config.batch_size)

        # 並列処理
        if self.config.max_workers > 1:
            processed_records = self._execute_parallel_batches(
                operation, table_name, batches, result, **kwargs
            )
        else:
            # シーケンシャル処理
            for batch in batches:
                batch_result = self._execute_single_batch(
                    operation, table_name, batch, **kwargs
                )
                processed_records += batch_result

        return processed_records

    def _execute_streaming_operation(
        self,
        operation: BulkOperation,
        table_name: str,
        data: Union[pd.DataFrame, List[Dict], Iterator[Dict]],
        result: BulkOperationResult,
        **kwargs,
    ) -> int:
        """ストリーミング処理による操作実行"""
        processed_records = 0

        # ストリーミングバッチ処理
        with self.connection_pool.get_connection() as conn:
            try:
                # トランザクション開始
                conn.execute("BEGIN TRANSACTION")

                batch = []
                batch_count = 0

                for record in self._iterate_data(data):
                    batch.append(record)

                    if len(batch) >= self.config.batch_size:
                        # バッチ処理
                        batch_processed = self._process_streaming_batch(
                            conn, operation, table_name, batch, **kwargs
                        )
                        processed_records += batch_processed

                        batch = []
                        batch_count += 1

                        # トランザクションサイズチェック
                        if (
                            batch_count * self.config.batch_size
                            >= self.config.transaction_size
                        ):
                            conn.execute("COMMIT")
                            conn.execute("BEGIN TRANSACTION")
                            batch_count = 0

                # 残りのバッチ処理
                if batch:
                    batch_processed = self._process_streaming_batch(
                        conn, operation, table_name, batch, **kwargs
                    )
                    processed_records += batch_processed

                # トランザクションコミット
                conn.execute("COMMIT")

            except Exception as e:
                conn.execute("ROLLBACK")
                raise e

        return processed_records

    def _execute_parallel_batches(
        self,
        operation: BulkOperation,
        table_name: str,
        batches: List[List[Dict]],
        result: BulkOperationResult,
        **kwargs,
    ) -> int:
        """並列バッチ処理実行"""
        processed_records = 0

        # Future作成
        futures = []
        for batch in batches:
            future = self.executor.submit(
                self._execute_single_batch, operation, table_name, batch, **kwargs
            )
            futures.append(future)

        # 結果取得
        for future in futures:
            try:
                batch_result = future.result(timeout=30.0)
                processed_records += batch_result
            except Exception as e:
                result.error_details.append(str(e))
                logger.error(f"並列バッチ処理エラー: {e}")

        return processed_records

    def _execute_single_batch(
        self, operation: BulkOperation, table_name: str, batch: List[Dict], **kwargs
    ) -> int:
        """単一バッチ処理実行"""
        if not batch:
            return 0

        with self.connection_pool.get_connection() as conn:
            try:
                if operation == BulkOperation.INSERT:
                    return self._batch_insert(conn, table_name, batch, **kwargs)
                elif operation == BulkOperation.UPDATE:
                    return self._batch_update(conn, table_name, batch, **kwargs)
                elif operation == BulkOperation.UPSERT:
                    return self._batch_upsert(conn, table_name, batch, **kwargs)
                else:
                    raise ValueError(f"未対応操作: {operation}")

            except Exception as e:
                logger.error(f"バッチ処理エラー: {e}")
                return 0

    def _batch_insert(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        batch: List[Dict],
        columns: List[str] = None,
        conflict_resolution: str = "IGNORE",
        **kwargs,
    ) -> int:
        """バッチ挿入実行"""
        if not batch:
            return 0

        # 列名決定
        if not columns:
            columns = list(batch[0].keys())

        # SQL作成
        placeholders = ",".join(["?" for _ in columns])
        sql = f"INSERT OR {conflict_resolution} INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"

        # データ準備
        values_list = []
        for record in batch:
            values = [record.get(col) for col in columns]
            values_list.append(values)

        # 実行
        conn.executemany(sql, values_list)
        return len(values_list)

    def _batch_update(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        batch: List[Dict],
        where_columns: List[str],
        update_columns: List[str] = None,
        **kwargs,
    ) -> int:
        """バッチ更新実行"""
        if not batch:
            return 0

        # 更新列決定
        if not update_columns:
            update_columns = [
                col for col in batch[0].keys() if col not in where_columns
            ]

        # SQL作成
        set_clause = ",".join([f"{col}=?" for col in update_columns])
        where_clause = " AND ".join([f"{col}=?" for col in where_columns])
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"

        # データ準備
        values_list = []
        for record in batch:
            values = []
            # SET句の値
            values.extend([record.get(col) for col in update_columns])
            # WHERE句の値
            values.extend([record.get(col) for col in where_columns])
            values_list.append(values)

        # 実行
        conn.executemany(sql, values_list)
        return len(values_list)

    def _batch_upsert(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        batch: List[Dict],
        key_columns: List[str],
        **kwargs,
    ) -> int:
        """バッチUPSERT実行"""
        if not batch:
            return 0

        # 列名取得
        all_columns = list(batch[0].keys())
        update_columns = [col for col in all_columns if col not in key_columns]

        # SQL作成（SQLite 3.24以降のUPSERT構文）
        placeholders = ",".join(["?" for _ in all_columns])
        conflict_columns = ",".join(key_columns)
        update_set = ",".join([f"{col}=excluded.{col}" for col in update_columns])

        sql = f"""
            INSERT INTO {table_name} ({",".join(all_columns)})
            VALUES ({placeholders})
            ON CONFLICT({conflict_columns}) DO UPDATE SET {update_set}
        """

        # データ準備
        values_list = []
        for record in batch:
            values = [record.get(col) for col in all_columns]
            values_list.append(values)

        # 実行
        conn.executemany(sql, values_list)
        return len(values_list)

    def _process_streaming_batch(
        self,
        conn: sqlite3.Connection,
        operation: BulkOperation,
        table_name: str,
        batch: List[Dict],
        **kwargs,
    ) -> int:
        """ストリーミングバッチ処理"""
        # 単一バッチ処理と同じロジック（トランザクション制御は呼び出し元で行う）
        if operation == BulkOperation.INSERT:
            return self._batch_insert(conn, table_name, batch, **kwargs)
        elif operation == BulkOperation.UPDATE:
            return self._batch_update(conn, table_name, batch, **kwargs)
        elif operation == BulkOperation.UPSERT:
            return self._batch_upsert(conn, table_name, batch, **kwargs)
        else:
            return 0

    def _apply_pre_optimizations(
        self, table_name: str, operation: BulkOperation, result: BulkOperationResult
    ):
        """前処理最適化"""
        optimizations = []

        with self.connection_pool.get_connection() as conn:
            try:
                # 一時インデックス作成（大量更新/UPSERT時）
                if (
                    operation in [BulkOperation.UPDATE, BulkOperation.UPSERT]
                    and self.config.create_temp_indexes
                ):
                    self._create_temporary_indexes(conn, table_name)
                    optimizations.append("temporary_indexes")

                # SQLiteの最適化設定
                if self.config.enable_synchronous_off:
                    conn.execute("PRAGMA synchronous=OFF")
                    optimizations.append("synchronous_off")

                if self.config.enable_journal_memory:
                    conn.execute("PRAGMA journal_mode=MEMORY")
                    optimizations.append("journal_memory")

                result.optimizations_applied.extend(optimizations)

            except Exception as e:
                logger.warning(f"前処理最適化エラー: {e}")

    def _apply_post_optimizations(
        self, table_name: str, operation: BulkOperation, result: BulkOperationResult
    ):
        """後処理最適化"""
        optimizations = []

        with self.connection_pool.get_connection() as conn:
            try:
                # 一時インデックス削除
                self._drop_temporary_indexes(conn)
                optimizations.append("temp_indexes_dropped")

                # ANALYZE実行
                if self.config.analyze_after_bulk:
                    conn.execute(f"ANALYZE {table_name}")
                    optimizations.append("analyze")

                # VACUUM実行（必要に応じて）
                db_size_mb = self._get_database_size_mb()
                if db_size_mb > self.config.vacuum_threshold_mb:
                    conn.execute("VACUUM")
                    optimizations.append("vacuum")

                result.optimizations_applied.extend(optimizations)

            except Exception as e:
                logger.warning(f"後処理最適化エラー: {e}")

    def _create_temporary_indexes(self, conn: sqlite3.Connection, table_name: str):
        """一時インデックス作成"""
        try:
            # テーブル構造取得
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]

            # よく使われる列に一時インデックス作成
            common_index_columns = ["symbol", "date", "timestamp", "id", "code"]

            for col in common_index_columns:
                if col in columns:
                    index_name = f"temp_idx_{table_name}_{col}"
                    sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({col})"
                    conn.execute(sql)
                    self.temp_indexes.add(index_name)

        except Exception as e:
            logger.warning(f"一時インデックス作成エラー: {e}")

    def _drop_temporary_indexes(self, conn: sqlite3.Connection):
        """一時インデックス削除"""
        for index_name in list(self.temp_indexes):
            try:
                conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                self.temp_indexes.remove(index_name)
            except Exception as e:
                logger.warning(f"一時インデックス削除エラー {index_name}: {e}")

    def _create_batches(
        self, data: Union[pd.DataFrame, List[Dict], Iterator[Dict]], batch_size: int
    ) -> List[List[Dict]]:
        """データをバッチに分割"""
        batches = []
        current_batch = []

        for record in self._iterate_data(data):
            current_batch.append(record)

            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        return batches

    def _iterate_data(
        self, data: Union[pd.DataFrame, List[Dict], Iterator[Dict]]
    ) -> Iterator[Dict]:
        """データイテレーター"""
        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                yield row.to_dict()
        elif isinstance(data, list):
            for record in data:
                yield record
        else:
            # Iterator
            for record in data:
                yield record

    def _estimate_data_size(
        self, data: Union[pd.DataFrame, List[Dict], Iterator[Dict]]
    ) -> int:
        """データサイズ推定"""
        if isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, list):
            return len(data)
        else:
            # Iteratorの場合は推定困難なので0を返す
            return 0

    def _get_database_size_mb(self) -> float:
        """データベースサイズ取得（MB）"""
        try:
            import os

            size_bytes = os.path.getsize(self.db_path)
            return size_bytes / (1024 * 1024)
        except:
            return 0.0

    def _update_stats(self, result: BulkOperationResult):
        """統計更新"""
        self.total_operations += 1
        if result.success_rate > 0.9:  # 90%以上成功で成功とみなす
            self.successful_operations += 1
        self.total_records_processed += result.processed_records
        self.total_processing_time += result.total_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "success_rate": (
                self.successful_operations / self.total_operations
                if self.total_operations > 0
                else 0.0
            ),
            "total_records_processed": self.total_records_processed,
            "average_throughput_rps": (
                self.total_records_processed / self.total_processing_time
                if self.total_processing_time > 0
                else 0.0
            ),
            "average_processing_time": (
                self.total_processing_time / self.total_operations
                if self.total_operations > 0
                else 0.0
            ),
            "database_size_mb": self._get_database_size_mb(),
            "active_temp_indexes": len(self.temp_indexes),
        }

    def optimize_table_structure(self, table_name: str) -> Dict[str, Any]:
        """テーブル構造最適化"""
        optimizations = {}

        with self.connection_pool.get_connection() as conn:
            try:
                # インデックス分析
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
                    (table_name,),
                )
                existing_indexes = [row[0] for row in cursor.fetchall()]

                # テーブル統計
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                # 推奨インデックス提案（簡易版）
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]

                recommended_indexes = []
                for col in ["symbol", "date", "timestamp", "id", "code"]:
                    if col in columns:
                        index_name = f"idx_{table_name}_{col}"
                        if index_name not in existing_indexes:
                            recommended_indexes.append(index_name)

                optimizations = {
                    "table_name": table_name,
                    "row_count": row_count,
                    "existing_indexes": existing_indexes,
                    "recommended_indexes": recommended_indexes,
                    "database_size_mb": self._get_database_size_mb(),
                }

            except Exception as e:
                logger.error(f"テーブル構造最適化エラー: {e}")
                optimizations["error"] = str(e)

        return optimizations

    def close(self):
        """システム終了"""
        # 一時インデックス削除
        if self.temp_indexes:
            with self.connection_pool.get_connection() as conn:
                self._drop_temporary_indexes(conn)

        # リソース解放
        self.executor.shutdown(wait=True)
        self.connection_pool.close_all()

        logger.info("DBバルク最適化システム終了")


# 便利関数
def create_bulk_optimizer(
    db_path: str,
    batch_size: int = 1000,
    max_workers: int = 4,
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    **kwargs,
) -> DatabaseBulkOptimizer:
    """バルク最適化システム作成"""
    config = BulkOperationConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        optimization_strategy=optimization_strategy,
        **kwargs,
    )

    return DatabaseBulkOptimizer(
        db_path=db_path, pool_size=max_workers * 2, config=config
    )


if __name__ == "__main__":
    # テスト実行
    print("=== データベースバルク最適化システム テスト ===")

    # テストデータベース作成
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        test_db_path = tmp_file.name

    try:
        # バルク最適化システム作成
        optimizer = create_bulk_optimizer(
            db_path=test_db_path,
            batch_size=500,
            max_workers=2,
            optimization_strategy=OptimizationStrategy.SPEED_OPTIMIZED,
        )

        # テストテーブル作成
        with optimizer.connection_pool.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT,
                    date TEXT,
                    price REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            """
            )
            conn.commit()

        # テストデータ生成
        test_data = []
        symbols = ["7203", "8306", "9984", "6758", "4689"]

        for i in range(10000):  # 10,000レコード
            symbol = symbols[i % len(symbols)]
            test_data.append(
                {
                    "symbol": symbol,
                    "date": f"2024-01-{(i % 30) + 1:02d}",
                    "price": 100.0 + (i % 1000),
                    "volume": 1000000 + (i % 500000),
                }
            )

        print(f"テストデータ作成完了: {len(test_data)} records")

        # バルク挿入テスト
        print("\nバルク挿入テスト開始...")
        insert_result = optimizer.bulk_insert(
            table_name="stock_data", data=test_data, conflict_resolution="REPLACE"
        )

        print("バルク挿入結果:")
        print(
            f"  処理済み: {insert_result.processed_records}/{insert_result.total_records}"
        )
        print(f"  成功率: {insert_result.success_rate:.1%}")
        print(f"  処理時間: {insert_result.total_time:.2f}秒")
        print(f"  スループット: {insert_result.throughput_rps:.1f} RPS")
        print(f"  最適化適用: {insert_result.optimizations_applied}")

        # バルクUPSERTテスト
        print("\nバルクUPSERTテスト開始...")
        update_data = test_data[:1000].copy()  # 最初の1000レコードを更新
        for record in update_data:
            record["price"] *= 1.1  # 価格を10%上昇

        upsert_result = optimizer.bulk_upsert(
            table_name="stock_data", data=update_data, key_columns=["symbol", "date"]
        )

        print("バルクUPSERT結果:")
        print(
            f"  処理済み: {upsert_result.processed_records}/{upsert_result.total_records}"
        )
        print(f"  成功率: {upsert_result.success_rate:.1%}")
        print(f"  処理時間: {upsert_result.total_time:.2f}秒")
        print(f"  スループット: {upsert_result.throughput_rps:.1f} RPS")

        # パフォーマンス統計
        stats = optimizer.get_performance_stats()
        print("\nパフォーマンス統計:")
        print(f"  総操作数: {stats['total_operations']}")
        print(f"  成功率: {stats['success_rate']:.1%}")
        print(f"  処理済みレコード総数: {stats['total_records_processed']}")
        print(f"  平均スループット: {stats['average_throughput_rps']:.1f} RPS")
        print(f"  平均処理時間: {stats['average_processing_time']:.3f}秒")
        print(f"  DBサイズ: {stats['database_size_mb']:.2f} MB")

        # テーブル最適化分析
        optimization_info = optimizer.optimize_table_structure("stock_data")
        print("\nテーブル最適化分析:")
        print(f"  テーブル: {optimization_info['table_name']}")
        print(f"  行数: {optimization_info['row_count']}")
        print(f"  既存インデックス: {optimization_info['existing_indexes']}")
        print(f"  推奨インデックス: {optimization_info['recommended_indexes']}")

    finally:
        # クリーンアップ
        optimizer.close()
        try:
            os.unlink(test_db_path)
        except:
            pass

    print("\n=== テスト完了 ===")
