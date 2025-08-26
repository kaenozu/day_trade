"""
トランザクション管理モジュール
セッション管理、トランザクションスコープ、デッドロック対応

Issue #120: declarative_base()の定義場所の最適化対応
- トランザクション管理の責務を明確化
- デッドロック検出とリトライ機能
"""

import time
from contextlib import contextmanager
from typing import Generator

from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from ...utils.exceptions import DatabaseError, handle_database_exception
from ...utils.logging_config import get_context_logger, log_error_with_context
from .connection import ConnectionManager

logger = get_context_logger(__name__)


class TransactionManager:
    """トランザクション管理クラス"""

    def __init__(self, connection_manager: ConnectionManager):
        """
        トランザクション管理の初期化

        Args:
            connection_manager: データベース接続管理
        """
        self.connection_manager = connection_manager
        self.session_factory = None
        self._initialize_session_factory()

    def _initialize_session_factory(self):
        """セッションファクトリーの初期化"""
        if not self.connection_manager.engine:
            raise DatabaseError(
                "Database engine not initialized",
                error_code="DB_ENGINE_NOT_INITIALIZED",
            )
        
        self.session_factory = sessionmaker(bind=self.connection_manager.engine)

    def get_session(self) -> Session:
        """
        新しいセッションを取得

        Returns:
            SQLAlchemyセッション

        Raises:
            DatabaseError: セッションファクトリーが初期化されていない場合
        """
        if self.session_factory is None:
            raise DatabaseError(
                "TransactionManager not properly initialized",
                error_code="TXN_MGR_NOT_INITIALIZED",
            )
        return self.session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        セッションのコンテキストマネージャー

        Usage:
            with transaction_manager.session_scope() as session:
                # セッションを使用した処理
                pass

        Yields:
            SQLAlchemyセッション

        Raises:
            DatabaseError: データベースエラーが発生した場合
        """
        if self.session_factory is None:
            raise DatabaseError(
                "TransactionManager not properly initialized",
                error_code="TXN_MGR_NOT_INITIALIZED",
            )

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            # エラーを適切な例外に変換
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error,
                {"operation": "database_session", "session_id": id(session)},
            )
            raise converted_error from e
        finally:
            session.close()

    @contextmanager
    def transaction_scope(
        self, retry_count: int = 3, retry_delay: float = 0.1
    ) -> Generator[Session, None, None]:
        """
        明示的なトランザクション管理とデッドロック対応

        Args:
            retry_count: デッドロック時の再試行回数
            retry_delay: 再試行時の待機時間（秒）

        Usage:
            with transaction_manager.transaction_scope() as session:
                # 複数のDB操作を含む処理
                session.add(obj1)
                session.add(obj2)
                # 明示的にflushして中間結果を確認
                session.flush()
                # さらに複雑な処理...

        Yields:
            SQLAlchemyセッション

        Raises:
            DatabaseError: データベースエラーが発生した場合
        """
        if self.session_factory is None:
            raise DatabaseError(
                "TransactionManager not properly initialized",
                error_code="TXN_MGR_NOT_INITIALIZED",
            )

        for attempt in range(retry_count + 1):
            session = self.session_factory()
            try:
                # session.begin()を明示的に呼び出し
                transaction = session.begin()
                try:
                    yield session
                    transaction.commit()
                    break
                except Exception:
                    transaction.rollback()
                    raise
            except (OperationalError, IntegrityError) as e:
                if attempt < retry_count and self._is_retriable_error(e):
                    logger.warning(
                        "Retriable database error, retrying",
                        attempt=attempt + 1,
                        max_attempts=retry_count + 1,
                        retry_delay=retry_delay,
                        error=str(e),
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数バックオフ
                    continue
                else:
                    converted_error = handle_database_exception(e)
                    log_error_with_context(
                        converted_error,
                        {
                            "operation": "database_transaction",
                            "attempts": attempt + 1,
                            "retry_count": retry_count,
                        },
                    )
                    raise converted_error from e
            except Exception as e:
                converted_error = handle_database_exception(e)
                log_error_with_context(
                    converted_error,
                    {"operation": "database_transaction", "error_type": "unexpected"},
                )
                raise converted_error from e
            finally:
                session.close()

    def _is_retriable_error(self, error: Exception) -> bool:
        """
        再試行可能なエラーかどうかを判定

        Args:
            error: 発生したエラー

        Returns:
            再試行可能な場合True
        """
        error_msg = str(error).lower()
        retriable_patterns = [
            "deadlock",
            "lock timeout",
            "database is locked",
            "connection was dropped",
            "connection pool",
        ]
        return any(pattern in error_msg for pattern in retriable_patterns)

    def atomic_operation(self, operations: list, retry_count: int = 3):
        """
        複数のDB操作をアトミックに実行

        Args:
            operations: 実行する操作のリスト（callable）
            retry_count: 再試行回数

        Usage:
            def operation1(session):
                session.add(obj1)

            def operation2(session):
                session.add(obj2)

            transaction_manager.atomic_operation([operation1, operation2])
        """
        operation_logger = logger
        operation_logger.info("Starting atomic operation")

        with self.transaction_scope(retry_count=retry_count) as session:
            for i, operation in enumerate(operations):
                operation(session)
                # 各操作後に中間状態をflush（デバッグ時などに有用）
                session.flush()
                operation_logger.debug("Operation completed", extra={"step": i + 1})

        operation_logger.info("Atomic operation completed")

    def execute_with_retry(self, operation, retry_count: int = 3, retry_delay: float = 0.1):
        """
        リトライ機能付きでデータベース操作を実行

        Args:
            operation: 実行する操作（session引数を受け取る関数）
            retry_count: 最大再試行回数
            retry_delay: 再試行時の初期遅延時間

        Returns:
            操作の実行結果

        Usage:
            def my_operation(session):
                return session.query(Model).count()

            result = transaction_manager.execute_with_retry(my_operation)
        """
        for attempt in range(retry_count + 1):
            try:
                with self.session_scope() as session:
                    return operation(session)
            except (OperationalError, IntegrityError) as e:
                if attempt < retry_count and self._is_retriable_error(e):
                    logger.warning(
                        f"Database operation failed, retrying (attempt {attempt + 1}/{retry_count + 1})",
                        extra={"error": str(e), "retry_delay": retry_delay}
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数バックオフ
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"Non-retriable database error: {e}")
                raise

    @contextmanager
    def read_only_session(self) -> Generator[Session, None, None]:
        """
        読み取り専用セッションのコンテキストマネージャー

        読み取り専用のセッションを提供し、コミットを実行しない

        Yields:
            SQLAlchemyセッション（読み取り専用）
        """
        if self.session_factory is None:
            raise DatabaseError(
                "TransactionManager not properly initialized",
                error_code="TXN_MGR_NOT_INITIALIZED",
            )

        session = self.session_factory()
        try:
            # 読み取り専用であることを明示
            logger.debug("Starting read-only session")
            yield session
            # 読み取り専用のため、明示的にcommitしない
        except Exception as e:
            # 読み取り専用でもrollbackは安全
            session.rollback()
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error,
                {"operation": "database_readonly_session", "session_id": id(session)},
            )
            raise converted_error from e
        finally:
            session.close()
            logger.debug("Read-only session closed")

    def close(self):
        """トランザクション管理をクリーンアップ"""
        self.session_factory = None
        logger.info("Transaction manager closed")

    def __del__(self):
        """デストラクタでクリーンアップ"""
        try:
            self.close()
        except Exception:
            pass  # デストラクタでの例外は無視