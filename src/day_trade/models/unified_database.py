"""
統合データベースアクセス層

Phase 4: エラーハンドリング統合とデータベース層統合の一環として、
分散していたデータベースアクセスパターンを統一し、保守性を向上させます。

主な機能:
- 統一されたデータベースセッション管理
- トランザクション処理の自動化
- エラーハンドリングとリカバリ機能
- パフォーマンス監視とメトリクス収集
- 接続プール管理とリソース最適化
- バックアップとリストア機能
"""

import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, TypeVar, Union

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ..config.unified_config import get_unified_config_manager
from ..core.unified_error_handler import (
    get_unified_error_handler,
    unified_database_session,
)
from ..utils.logging_config import get_context_logger
from ..utils.unified_utils import ThreadSafeCounter

logger = get_context_logger(__name__, component="unified_database")

T = TypeVar("T")


class DatabaseMetrics:
    """データベース操作メトリクス"""

    def __init__(self):
        self._operation_counts = ThreadSafeCounter()
        self._error_counts = ThreadSafeCounter()
        self._connection_counts = ThreadSafeCounter()
        self._transaction_counts = ThreadSafeCounter()
        self._rollback_counts = ThreadSafeCounter()
        self._query_times = []
        self._lock = threading.RLock()

    def record_operation(self, duration: float = 0.0) -> None:
        """データベース操作を記録"""
        self._operation_counts.increment()
        if duration > 0:
            with self._lock:
                self._query_times.append(duration)
                # 最新1000件のみ保持
                if len(self._query_times) > 1000:
                    self._query_times = self._query_times[-1000:]

    def record_error(self) -> None:
        """エラー発生を記録"""
        self._error_counts.increment()

    def record_connection(self) -> None:
        """接続作成を記録"""
        self._connection_counts.increment()

    def record_transaction(self) -> None:
        """トランザクション開始を記録"""
        self._transaction_counts.increment()

    def record_rollback(self) -> None:
        """ロールバック実行を記録"""
        self._rollback_counts.increment()

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        with self._lock:
            query_times = self._query_times.copy()

        avg_query_time = sum(query_times) / max(len(query_times), 1)
        max_query_time = max(query_times) if query_times else 0

        return {
            "total_operations": self._operation_counts.get(),
            "total_errors": self._error_counts.get(),
            "total_connections": self._connection_counts.get(),
            "total_transactions": self._transaction_counts.get(),
            "total_rollbacks": self._rollback_counts.get(),
            "error_rate": self._error_counts.get() / max(self._operation_counts.get(), 1),
            "rollback_rate": self._rollback_counts.get() / max(self._transaction_counts.get(), 1),
            "avg_query_time": avg_query_time,
            "max_query_time": max_query_time,
            "recent_query_count": len(query_times),
        }


class DatabaseConnectionPool:
    """統合データベース接続プール管理"""

    def __init__(self, config_manager=None):
        """
        Args:
            config_manager: 統合設定マネージャー
        """
        self.config_manager = config_manager or get_unified_config_manager()
        self._engines = {}
        self._session_factories = {}
        self._lock = threading.RLock()
        self.metrics = DatabaseMetrics()

        # 設定読み込み
        self._load_config()

        # イベントリスナー設定
        self._setup_event_listeners()

    def _load_config(self):
        """データベース設定を読み込み"""
        try:
            db_config = self.config_manager.get_database_config()
            self.database_url = getattr(db_config, "url", "sqlite:///day_trade.db")
            self.pool_size = getattr(db_config, "pool_size", 5)
            self.max_overflow = getattr(db_config, "max_overflow", 10)
            self.pool_timeout = getattr(db_config, "pool_timeout", 30)
            self.pool_recycle = getattr(db_config, "pool_recycle", 3600)
            self.echo = getattr(db_config, "echo", False)
        except Exception as e:
            logger.warning(f"データベース設定読み込み失敗、デフォルト値使用: {e}")
            self.database_url = "sqlite:///day_trade.db"
            self.pool_size = 5
            self.max_overflow = 10
            self.pool_timeout = 30
            self.pool_recycle = 3600
            self.echo = False

    def _setup_event_listeners(self):
        """SQLAlchemyイベントリスナー設定"""

        @event.listens_for(self, "connect", propagate=True)
        def receive_connect(dbapi_connection, connection_record):
            """接続作成時のイベント"""
            self.metrics.record_connection()
            logger.debug("データベース接続作成", extra={"connection_id": id(dbapi_connection)})

    def get_engine(self, database_url: Optional[str] = None):
        """データベースエンジンを取得"""
        url = database_url or self.database_url

        with self._lock:
            if url not in self._engines:
                self._engines[url] = self._create_engine(url)

            return self._engines[url]

    def _create_engine(self, database_url: str):
        """データベースエンジンを作成"""
        engine_kwargs = {
            "echo": self.echo,
            "pool_pre_ping": True,  # 接続健全性チェック
        }

        # SQLiteの場合の特別設定
        if database_url.startswith("sqlite"):
            engine_kwargs.update(
                {
                    "poolclass": StaticPool,
                    "connect_args": {
                        "check_same_thread": False,
                        "timeout": 20,
                    },
                }
            )
        else:
            # その他のDBの場合
            engine_kwargs.update(
                {
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_timeout": self.pool_timeout,
                    "pool_recycle": self.pool_recycle,
                }
            )

        engine = create_engine(database_url, **engine_kwargs)

        # エンジンイベントリスナー
        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            context._query_start_time = time.time()

        @event.listens_for(engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - context._query_start_time
            self.metrics.record_operation(total)

            if total > 1.0:  # 1秒以上の重いクエリをログ
                logger.warning(
                    f"Slow query detected: {total:.2f}s",
                    extra={
                        "query_time": total,
                        "statement": statement[:200],  # クエリの先頭200文字
                    },
                )

        return engine

    def get_session_factory(self, database_url: Optional[str] = None):
        """セッションファクトリを取得"""
        url = database_url or self.database_url

        with self._lock:
            if url not in self._session_factories:
                engine = self.get_engine(url)
                self._session_factories[url] = sessionmaker(bind=engine)

            return self._session_factories[url]

    @contextmanager
    def get_session(
        self,
        database_url: Optional[str] = None,
        auto_commit: bool = True,
        operation: str = "database_operation",
    ):
        """
        統合データベースセッションを取得

        Args:
            database_url: データベースURL
            auto_commit: 自動コミットを行うか
            operation: 操作名（ログ用）
        """
        session_factory = self.get_session_factory(database_url)

        with unified_database_session(
            session_factory=session_factory,
            operation=operation,
            auto_commit=auto_commit,
        ) as session:
            self.metrics.record_transaction()
            yield session

    def execute_query(
        self,
        query: Union[str, text],
        parameters: Optional[Dict] = None,
        database_url: Optional[str] = None,
        operation: str = "execute_query",
    ) -> Any:
        """
        クエリを実行

        Args:
            query: 実行するクエリ
            parameters: クエリパラメータ
            database_url: データベースURL
            operation: 操作名

        Returns:
            クエリ結果
        """
        with self.get_session(database_url, operation=operation) as session:
            if isinstance(query, str):
                query = text(query)

            if parameters:
                result = session.execute(query, parameters)
            else:
                result = session.execute(query)

            return result.fetchall()

    def execute_transaction(
        self,
        operations: List[Dict[str, Any]],
        database_url: Optional[str] = None,
        operation: str = "bulk_transaction",
    ) -> bool:
        """
        複数の操作をトランザクションで実行

        Args:
            operations: 実行する操作のリスト
            database_url: データベースURL
            operation: 操作名

        Returns:
            成功したかどうか
        """
        try:
            with self.get_session(database_url, auto_commit=False, operation=operation) as session:
                for op in operations:
                    query = op.get("query")
                    parameters = op.get("parameters", {})

                    if isinstance(query, str):
                        query = text(query)

                    session.execute(query, parameters)

                session.commit()
                return True

        except Exception as e:
            self.metrics.record_error()
            self.metrics.record_rollback()
            logger.error(f"トランザクション実行失敗: {e}", exc_info=True)
            return False

    def health_check(self, database_url: Optional[str] = None) -> Dict[str, Any]:
        """データベース健全性チェック"""
        url = database_url or self.database_url
        health_status = {
            "database_url": url,
            "status": "unknown",
            "response_time": None,
            "error": None,
            "timestamp": time.time(),
        }

        start_time = time.time()

        try:
            with self.get_session(url, operation="health_check") as session:
                # 簡単なクエリでテスト
                result = session.execute(text("SELECT 1"))
                result.fetchone()

                response_time = time.time() - start_time
                health_status.update(
                    {
                        "status": "healthy",
                        "response_time": response_time,
                    }
                )

                if response_time > 1.0:
                    health_status["status"] = "slow"

        except Exception as e:
            self.metrics.record_error()
            health_status.update(
                {
                    "status": "unhealthy",
                    "response_time": time.time() - start_time,
                    "error": str(e),
                }
            )

        return health_status

    def get_connection_info(self, database_url: Optional[str] = None) -> Dict[str, Any]:
        """接続情報を取得"""
        url = database_url or self.database_url
        engine = self.get_engine(url)

        try:
            pool = engine.pool
            connection_info = {
                "database_url": url,
                "pool_size": getattr(pool, "size", lambda: "N/A")(),
                "checked_in_connections": getattr(pool, "checkedin", lambda: "N/A")(),
                "checked_out_connections": getattr(pool, "checkedout", lambda: "N/A")(),
                "overflow": getattr(pool, "overflow", lambda: "N/A")(),
                "total_connections": getattr(pool, "size", lambda: 0)()
                + getattr(pool, "overflow", lambda: 0)(),
            }
        except Exception as e:
            logger.warning(f"接続情報取得失敗: {e}")
            connection_info = {
                "database_url": url,
                "error": str(e),
            }

        return connection_info

    def close_all_connections(self):
        """全ての接続を閉じる"""
        with self._lock:
            for engine in self._engines.values():
                try:
                    engine.dispose()
                except Exception as e:
                    logger.error(f"エンジンの廃棄に失敗: {e}")

            self._engines.clear()
            self._session_factories.clear()

        logger.info("全データベース接続を閉じました")


class UnifiedDatabaseManager:
    """統合データベースマネージャー"""

    def __init__(self, config_manager=None):
        """
        Args:
            config_manager: 統合設定マネージャー
        """
        self.config_manager = config_manager or get_unified_config_manager()
        self.connection_pool = DatabaseConnectionPool(config_manager)
        self.error_handler = get_unified_error_handler()

    def create_all_tables(self, database_url: Optional[str] = None):
        """全テーブルを作成"""
        from .base import Base

        engine = self.connection_pool.get_engine(database_url)

        try:
            Base.metadata.create_all(engine)
            logger.info("データベーステーブル作成完了")
        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                operation="create_tables",
                component="database_manager",
            )
            raise

    def drop_all_tables(self, database_url: Optional[str] = None):
        """全テーブルを削除（注意して使用）"""
        from .base import Base

        engine = self.connection_pool.get_engine(database_url)

        try:
            Base.metadata.drop_all(engine)
            logger.warning("データベーステーブル削除完了")
        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                operation="drop_tables",
                component="database_manager",
            )
            raise

    def backup_database(self, backup_path: str, database_url: Optional[str] = None) -> bool:
        """データベースバックアップ"""
        try:
            # 簡易バックアップ実装（SQLiteの場合）
            if not database_url:
                database_url = self.connection_pool.database_url

            if database_url.startswith("sqlite"):
                import os
                import shutil

                # SQLiteファイルパス抽出
                db_file = database_url.replace("sqlite:///", "")
                if os.path.exists(db_file):
                    shutil.copy2(db_file, backup_path)
                    logger.info(f"データベースバックアップ完了: {backup_path}")
                    return True
                else:
                    logger.warning(f"データベースファイルが見つかりません: {db_file}")
                    return False
            else:
                logger.warning("SQLite以外のデータベースバックアップは未対応")
                return False

        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                operation="backup_database",
                component="database_manager",
                user_data={"backup_path": backup_path},
            )
            return False

    def restore_database(self, backup_path: str, database_url: Optional[str] = None) -> bool:
        """データベースリストア"""
        try:
            if not database_url:
                database_url = self.connection_pool.database_url

            if database_url.startswith("sqlite"):
                import os
                import shutil

                if not os.path.exists(backup_path):
                    logger.error(f"バックアップファイルが見つかりません: {backup_path}")
                    return False

                # 全接続を閉じる
                self.connection_pool.close_all_connections()

                # リストア実行
                db_file = database_url.replace("sqlite:///", "")
                shutil.copy2(backup_path, db_file)

                logger.info(f"データベースリストア完了: {backup_path}")
                return True
            else:
                logger.warning("SQLite以外のデータベースリストアは未対応")
                return False

        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                operation="restore_database",
                component="database_manager",
                user_data={"backup_path": backup_path},
            )
            return False

    def get_system_stats(self) -> Dict[str, Any]:
        """システム統計情報を取得"""
        db_metrics = self.connection_pool.metrics.get_stats()
        connection_info = self.connection_pool.get_connection_info()
        health_status = self.connection_pool.health_check()

        return {
            "database_metrics": db_metrics,
            "connection_info": connection_info,
            "health_status": health_status,
            "error_handler_stats": self.error_handler.metrics.get_stats(),
        }

    def cleanup_resources(self):
        """リソースのクリーンアップ"""
        self.connection_pool.close_all_connections()
        logger.info("データベースリソースクリーンアップ完了")


# グローバルインスタンス管理

_global_database_manager: Optional[UnifiedDatabaseManager] = None
_manager_lock = threading.RLock()


def get_unified_database_manager() -> UnifiedDatabaseManager:
    """統合データベースマネージャーのグローバルインスタンスを取得"""
    global _global_database_manager

    if _global_database_manager is None:
        with _manager_lock:
            if _global_database_manager is None:
                _global_database_manager = UnifiedDatabaseManager()

    return _global_database_manager


def set_unified_database_manager(manager: UnifiedDatabaseManager) -> None:
    """統合データベースマネージャーのグローバルインスタンスを設定"""
    global _global_database_manager

    with _manager_lock:
        _global_database_manager = manager


# 便利関数


@contextmanager
def get_database_session(
    database_url: Optional[str] = None,
    auto_commit: bool = True,
    operation: str = "database_operation",
):
    """データベースセッションを取得する便利関数"""
    manager = get_unified_database_manager()
    with manager.connection_pool.get_session(
        database_url=database_url, auto_commit=auto_commit, operation=operation
    ) as session:
        yield session


def execute_database_query(
    query: Union[str, text],
    parameters: Optional[Dict] = None,
    database_url: Optional[str] = None,
    operation: str = "execute_query",
) -> Any:
    """データベースクエリを実行する便利関数"""
    manager = get_unified_database_manager()
    return manager.connection_pool.execute_query(
        query=query,
        parameters=parameters,
        database_url=database_url,
        operation=operation,
    )


def get_database_health() -> Dict[str, Any]:
    """データベース健全性チェックの便利関数"""
    manager = get_unified_database_manager()
    return manager.connection_pool.health_check()


def get_database_stats() -> Dict[str, Any]:
    """データベース統計情報を取得する便利関数"""
    manager = get_unified_database_manager()
    return manager.get_system_stats()
