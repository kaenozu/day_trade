#!/usr/bin/env python3
"""
データベースアクセス最適化

コネクションプール、バッチ処理、インデックス最適化
"""

import sqlite3
import threading
import time
from contextlib import contextmanager
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class ConnectionPool:
    """データベース接続プール"""

    def __init__(self, db_path: str, pool_size: int = 10):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created_connections = 0

        # 初期接続作成
        for _ in range(min(3, pool_size)):  # 最初は3個だけ作成
            self._create_connection()

    def _create_connection(self) -> sqlite3.Connection:
        """新しい接続作成"""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0
        )

        # パフォーマンス最適化設定
        conn.execute("PRAGMA journal_mode=WAL")  # WALモード
        conn.execute("PRAGMA synchronous=NORMAL")  # 同期レベル調整
        conn.execute("PRAGMA cache_size=10000")  # キャッシュサイズ増加
        conn.execute("PRAGMA temp_store=MEMORY")  # 一時ファイルをメモリに

        self._created_connections += 1
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """接続取得"""
        try:
            # プールから取得試行
            return self._pool.get_nowait()
        except Empty:
            # プールが空の場合、新しい接続作成
            with self._lock:
                if self._created_connections < self.pool_size:
                    return self._create_connection()
                else:
                    # 最大数に達している場合は待機
                    return self._pool.get(timeout=5.0)

    def return_connection(self, conn: sqlite3.Connection):
        """接続返却"""
        try:
            self._pool.put_nowait(conn)
        except:
            # プールが満杯の場合は接続を閉じる
            conn.close()

    @contextmanager
    def connection(self):
        """接続コンテキストマネージャー"""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)

    def close_all(self):
        """全接続を閉じる"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break


class BatchProcessor:
    """バッチ処理最適化"""

    def __init__(self, connection_pool: ConnectionPool, batch_size: int = 1000):
        self.pool = connection_pool
        self.batch_size = batch_size

    def batch_insert(self, table: str, columns: List[str], 
                    data: List[Tuple]) -> int:
        """バッチインサート"""
        placeholders = ', '.join(['?' for _ in columns])
        columns_str = ', '.join(columns)
        sql = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

        total_inserted = 0

        with self.pool.connection() as conn:
            cursor = conn.cursor()

            # バッチサイズごとに分割して処理
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                cursor.executemany(sql, batch)
                conn.commit()
                total_inserted += len(batch)

        return total_inserted

    def batch_update(self, table: str, set_clause: str, where_clause: str,
                    data: List[Tuple]) -> int:
        """バッチアップデート"""
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

        total_updated = 0

        with self.pool.connection() as conn:
            cursor = conn.cursor()

            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                cursor.executemany(sql, batch)
                conn.commit()
                total_updated += cursor.rowcount

        return total_updated

    def batch_delete(self, table: str, where_clause: str,
                    data: List[Tuple]) -> int:
        """バッチ削除"""
        sql = f"DELETE FROM {table} WHERE {where_clause}"

        total_deleted = 0

        with self.pool.connection() as conn:
            cursor = conn.cursor()

            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                cursor.executemany(sql, batch)
                conn.commit()
                total_deleted += cursor.rowcount

        return total_deleted


class QueryOptimizer:
    """クエリ最適化"""

    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self._query_cache = {}

    def create_indexes(self, db_path: str):
        """インデックス作成"""
        index_definitions = [
            "CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_stocks_date ON stocks(date)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions(symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)",
        ]

        with self.pool.connection() as conn:
            for index_sql in index_definitions:
                try:
                    conn.execute(index_sql)
                    print(f"    インデックス作成: {index_sql.split()[-1]}")
                except sqlite3.Error as e:
                    print(f"    インデックス作成エラー: {e}")

    def optimize_query(self, sql: str) -> str:
        """クエリ最適化"""
        # 基本的な最適化ルール
        optimized_sql = sql

        # LIMIT句がない大きなクエリに警告
        if 'SELECT' in sql.upper() and 'LIMIT' not in sql.upper():
            if 'COUNT' not in sql.upper():
                print(f"    警告: LIMIT句のないクエリ: {sql[:50]}...")

        # ORDER BY最適化
        if 'ORDER BY' in sql.upper() and 'INDEX' not in sql.upper():
            print("    推奨: ORDER BY句にインデックス使用を検討")

        return optimized_sql

    def explain_query(self, sql: str) -> List[Dict]:
        """クエリ実行計画分析"""
        explain_sql = f"EXPLAIN QUERY PLAN {sql}"

        with self.pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(explain_sql)

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'parent': row[1],
                    'notused': row[2],
                    'detail': row[3]
                })

            return results

    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """テーブル分析"""
        with self.pool.connection() as conn:
            cursor = conn.cursor()
            
            # テーブル情報取得
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # 行数取得
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            return {
                'table_name': table_name,
                'columns': columns,
                'row_count': row_count
            }


class DatabaseManager:
    """データベースマネージャー"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pool = ConnectionPool(db_path)
        self.batch_processor = BatchProcessor(self.pool)
        self.query_optimizer = QueryOptimizer(self.pool)

        # 初期最適化
        self._initialize_optimizations()

    def _initialize_optimizations(self):
        """初期最適化設定"""
        # インデックス作成
        self.query_optimizer.create_indexes(self.db_path)

        # VACUUM実行（定期メンテナンス）
        with self.pool.connection() as conn:
            conn.execute("VACUUM")
            print("    データベース最適化完了")

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計"""
        with self.pool.connection() as conn:
            cursor = conn.cursor()

            # データベースサイズ
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]

            # キャッシュ統計
            cursor.execute("PRAGMA cache_size")
            cache_size = cursor.fetchone()[0]

            return {
                'database_size': db_size,
                'cache_size': cache_size,
                'connection_pool_size': self.pool._created_connections,
                'wal_mode': True  # WALモード有効
            }

    def cleanup(self):
        """リソースクリーンアップ"""
        self.pool.close_all()


# グローバルマネージャー
_db_managers = {}


def get_db_manager(db_path: str) -> DatabaseManager:
    """データベースマネージャー取得"""
    if db_path not in _db_managers:
        _db_managers[db_path] = DatabaseManager(db_path)
    return _db_managers[db_path]


def cleanup_all_managers():
    """全マネージャーのクリーンアップ"""
    for manager in _db_managers.values():
        manager.cleanup()
    _db_managers.clear()