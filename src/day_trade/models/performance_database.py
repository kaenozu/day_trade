"""
パフォーマンス最適化データベースマネージャー

SQLAlchemy 2.0の最新機能を活用したハイパフォーマンスデータベース操作。
Phase 2: パフォーマンス最適化プロジェクト対応
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import event, insert, text, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ClauseElement

from ..utils.logging_config import get_context_logger
from ..utils.performance_analyzer import profile_performance
from ..utils.performance_config import get_performance_config
from .base import Base
from .database import DatabaseConfig, DatabaseManager

logger = get_context_logger(__name__)


class PerformanceOptimizedDatabaseManager(DatabaseManager):
    """パフォーマンス最適化されたデータベースマネージャー"""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self.perf_config = get_performance_config()
        self._setup_performance_optimizations()

    def _setup_performance_optimizations(self) -> None:
        """パフォーマンス最適化の設定を適用"""

        # SQLiteの場合はPRAGMA設定を適用
        if self.config.is_sqlite():
            self._apply_sqlite_optimizations()

        # エンジンイベントハンドラーを設定
        self._setup_engine_events()

        logger.info(
            "Performance optimizations applied",
            optimization_level=self.perf_config.optimization_level,
            pool_size=self.perf_config.database.pool_size,
            max_overflow=self.perf_config.database.max_overflow,
        )

    def _apply_sqlite_optimizations(self) -> None:
        """SQLite固有の最適化設定を適用"""

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """SQLite接続時にPRAGMA設定を適用"""
            cursor = dbapi_connection.cursor()

            pragma_settings = self.perf_config.get_sqlite_pragma_settings()

            for pragma, value in pragma_settings.items():
                if isinstance(value, str):
                    cursor.execute(f"PRAGMA {pragma} = {value}")
                else:
                    cursor.execute(f"PRAGMA {pragma} = {value}")

            cursor.close()

            logger.debug("SQLite PRAGMA settings applied", settings=pragma_settings)

    def _setup_engine_events(self) -> None:
        """エンジンイベントハンドラーを設定"""

        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            """クエリ実行前の処理（パフォーマンス測定開始）"""
            context._query_start_time = time.time()

        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            """クエリ実行後の処理（パフォーマンス測定終了）"""
            if hasattr(context, "_query_start_time"):
                execution_time = time.time() - context._query_start_time

                # 遅いクエリをログ出力
                if execution_time > 1.0:  # 1秒以上かかったクエリ
                    logger.warning(
                        "Slow query detected",
                        execution_time=execution_time,
                        statement=statement[:200],  # 長いクエリは短縮
                    )

    @profile_performance
    def bulk_insert_optimized(
        self,
        model_class: Type[Base],
        data: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
    ) -> int:
        """
        最適化されたバルク挿入操作

        Args:
            model_class: SQLAlchemyモデルクラス
            data: 挿入するデータのリスト
            chunk_size: 処理チャンクサイズ（Noneの場合は設定から取得）

        Returns:
            int: 挿入された行数
        """
        if not data:
            return 0

        chunk_size = chunk_size or self.perf_config.database.bulk_batch_size
        total_inserted = 0

        with self.session_scope() as session:
            # SQLAlchemy 2.0のbulk insert構文を使用
            stmt = insert(model_class)

            # データをチャンクに分割して処理
            for i in range(0, len(data), chunk_size):
                chunk = data[i : i + chunk_size]

                try:
                    result = session.execute(stmt, chunk)
                    total_inserted += result.rowcount
                    session.commit()

                    logger.debug(
                        "Bulk insert chunk completed",
                        chunk_size=len(chunk),
                        total_inserted=total_inserted,
                    )

                except Exception as e:
                    session.rollback()
                    logger.error(
                        "Bulk insert chunk failed",
                        chunk_start=i,
                        chunk_size=len(chunk),
                        error=str(e),
                    )
                    raise

        logger.info(
            "Bulk insert completed",
            total_rows=total_inserted,
            chunks_processed=len(data) // chunk_size + 1,
        )

        return total_inserted

    @profile_performance
    def bulk_update_optimized(
        self,
        model_class: Type[Base],
        updates: List[Dict[str, Any]],
        id_column: str = "id",
    ) -> int:
        """
        最適化されたバルク更新操作

        Args:
            model_class: SQLAlchemyモデルクラス
            updates: 更新するデータのリスト（IDを含む）
            id_column: ID列名

        Returns:
            int: 更新された行数
        """
        if not updates:
            return 0

        total_updated = 0

        with self.session_scope() as session:
            try:
                # SQLAlchemy 2.0のbulk update構文を使用
                stmt = update(model_class)

                for update_data in updates:
                    id_value = update_data.pop(id_column)

                    result = session.execute(
                        stmt.where(getattr(model_class, id_column) == id_value).values(
                            **update_data
                        )
                    )
                    total_updated += result.rowcount

                session.commit()

            except Exception as e:
                session.rollback()
                logger.error(
                    "Bulk update failed", updates_count=len(updates), error=str(e)
                )
                raise

        logger.info("Bulk update completed", total_rows=total_updated)

        return total_updated

    @profile_performance
    def execute_optimized_select(
        self, stmt: Select, use_cache: bool = True
    ) -> List[Any]:
        """
        最適化されたSELECTクエリ実行

        Args:
            stmt: SQLAlchemy SELECT文
            use_cache: クエリキャッシュ使用有無

        Returns:
            List[Any]: クエリ結果
        """
        with self.session_scope() as session:
            if use_cache:
                # クエリキャッシュを有効化
                stmt = stmt.execution_options(compiled_cache={})

            result = session.scalars(stmt).all()

            logger.debug(
                "Optimized select executed",
                result_count=len(result),
                cache_enabled=use_cache,
            )

            return result

    @profile_performance
    def execute_batch_operations(self, operations: List[ClauseElement]) -> List[Any]:
        """
        バッチ操作の実行（複数のSQL文を1つのトランザクションで実行）

        Args:
            operations: 実行するSQL操作のリスト

        Returns:
            List[Any]: 各操作の結果
        """
        results = []

        with self.session_scope() as session:
            try:
                for operation in operations:
                    result = session.execute(operation)
                    results.append(result)

                session.commit()

                logger.info(
                    "Batch operations completed", operations_count=len(operations)
                )

            except Exception as e:
                session.rollback()
                logger.error(
                    "Batch operations failed",
                    operations_count=len(operations),
                    error=str(e),
                )
                raise

        return results

    @contextmanager
    def performance_session(self, **kwargs):
        """
        パフォーマンス最適化されたセッションコンテキストマネージャー

        Args:
            **kwargs: セッションオプション
        """
        # パフォーマンス最適化オプションを設定
        session_options = {
            "expire_on_commit": False,  # コミット後のオブジェクト失効を無効化
            "autoflush": False,  # 自動フラッシュを無効化（手動制御でパフォーマンス向上）
            **kwargs,
        }

        session_class = sessionmaker(bind=self.engine, **session_options)

        session = session_class()
        start_time = time.time()

        try:
            yield session
            session.commit()

            execution_time = time.time() - start_time
            logger.debug("Performance session completed", execution_time=execution_time)

        except Exception as e:
            session.rollback()
            logger.error("Performance session failed", error=str(e))
            raise
        finally:
            session.close()

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        データベースパフォーマンス統計を取得

        Returns:
            Dict[str, Any]: パフォーマンス統計情報
        """
        pool = self.engine.pool

        stats = {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": pool.size() + pool.overflow(),
            "config": {
                "pool_size": self.perf_config.database.pool_size,
                "max_overflow": self.perf_config.database.max_overflow,
                "optimization_level": self.perf_config.optimization_level,
            },
        }

        return stats

    def optimize_table_maintenance(self, table_name: str) -> Dict[str, Any]:
        """
        テーブルメンテナンス最適化

        Args:
            table_name: テーブル名

        Returns:
            Dict[str, Any]: 最適化結果
        """
        results = {"operations": [], "improvements": []}

        with self.session_scope() as session:
            if self.config.is_sqlite():
                # SQLite固有の最適化
                operations = [
                    f"REINDEX {table_name}",
                    f"ANALYZE {table_name}",
                    "VACUUM",
                ]

                for operation in operations:
                    try:
                        start_time = time.time()
                        session.execute(text(operation))
                        execution_time = time.time() - start_time

                        results["operations"].append(
                            {
                                "operation": operation,
                                "execution_time": execution_time,
                                "status": "success",
                            }
                        )

                        logger.debug(
                            "Table maintenance operation completed",
                            operation=operation,
                            execution_time=execution_time,
                        )

                    except Exception as e:
                        results["operations"].append(
                            {
                                "operation": operation,
                                "status": "failed",
                                "error": str(e),
                            }
                        )

                        logger.warning(
                            "Table maintenance operation failed",
                            operation=operation,
                            error=str(e),
                        )

        return results
