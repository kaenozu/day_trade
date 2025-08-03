"""
データベース基盤モジュール
SQLAlchemyを使用したデータベース接続とセッション管理
"""

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from alembic import command
from alembic.config import Config

from ..utils.exceptions import DatabaseError, handle_database_exception
from ..utils.logging_config import (
    get_context_logger,
    log_database_operation,
    log_error_with_context,
)

logger = get_context_logger(__name__)

# ベースクラスの作成
Base = declarative_base()

# テスト用のデータベースURL
TEST_DATABASE_URL = "sqlite:///:memory:"


class DatabaseConfig:
    """データベース設定クラス"""

    def __init__(
        self,
        database_url: Optional[str] = None,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        connect_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            database_url: データベースURL
            echo: SQLログ出力フラグ
            pool_size: 接続プールサイズ
            max_overflow: 最大オーバーフロー接続数
            pool_timeout: 接続タイムアウト（秒）
            pool_recycle: 接続リサイクル時間（秒）
            connect_args: 接続引数
        """
        self.database_url = database_url or os.environ.get(
            "DATABASE_URL", "sqlite:///./day_trade.db"
        )
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.connect_args = connect_args or {"check_same_thread": False}

    @classmethod
    def for_testing(cls) -> "DatabaseConfig":
        """テスト用の設定を作成"""
        return cls(
            database_url="sqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
        )

    @classmethod
    def for_production(cls) -> "DatabaseConfig":
        """本番用の設定を作成"""
        return cls(
            database_url=os.environ.get("DATABASE_URL", "sqlite:///./day_trade.db"),
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_timeout=60,
            pool_recycle=1800,
        )

    def is_sqlite(self) -> bool:
        """SQLiteデータベースかどうかを判定"""
        return self.database_url.startswith("sqlite")

    def is_in_memory(self) -> bool:
        """インメモリデータベースかどうかを判定"""
        return ":memory:" in self.database_url


class DatabaseManager:
    """データベース管理クラス（改善版）"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Args:
            config: データベース設定
        """
        self.config = config or DatabaseConfig()
        self.engine = None
        self.session_factory = None
        self._initialize_engine()

    def _get_database_type(self) -> str:
        """データベースタイプを取得"""
        if self.config.is_sqlite():
            return "sqlite"
        elif "postgresql" in self.config.database_url:
            return "postgresql"
        elif "mysql" in self.config.database_url:
            return "mysql"
        else:
            return "other"

    def _initialize_engine(self):
        """エンジンの初期化"""
        try:
            # エンジン作成時の引数を設定に基づいて構築
            engine_kwargs = {
                "echo": self.config.echo,
                "connect_args": self.config.connect_args,
            }

            if self.config.is_in_memory():
                # インメモリDBの場合はStaticPoolを使用
                engine_kwargs.update(
                    {
                        "poolclass": StaticPool,
                        "pool_pre_ping": True,
                    }
                )
            elif self.config.is_sqlite():
                # SQLiteファイルの場合
                engine_kwargs.update(
                    {
                        "pool_pre_ping": True,
                        "pool_recycle": self.config.pool_recycle,
                    }
                )
            else:
                # その他のDB（PostgreSQL、MySQLなど）
                engine_kwargs.update(
                    {
                        "pool_size": self.config.pool_size,
                        "max_overflow": self.config.max_overflow,
                        "pool_timeout": self.config.pool_timeout,
                        "pool_recycle": self.config.pool_recycle,
                        "pool_pre_ping": True,
                    }
                )

            self.engine = create_engine(self.config.database_url, **engine_kwargs)

            # SQLiteの場合は外部キー制約を有効化
            if self.config.is_sqlite():
                event.listen(self.engine, "connect", self._set_sqlite_pragma)

            # セッションファクトリーの作成
            self.session_factory = sessionmaker(bind=self.engine)

            logger.info("Database engine initialized",
                       database_url=self.config.database_url,
                       database_type=self._get_database_type())

        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(converted_error, {
                "operation": "database_initialization",
                "database_url": self.config.database_url
            })
            raise converted_error

    @staticmethod
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        """SQLiteの設定（パフォーマンス最適化）"""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")  # WALモードでパフォーマンス向上
        cursor.execute("PRAGMA synchronous=NORMAL")  # 同期レベルを調整
        cursor.execute("PRAGMA cache_size=10000")  # キャッシュサイズを増加
        cursor.execute("PRAGMA temp_store=memory")  # 一時テーブルをメモリに保存
        cursor.execute("PRAGMA mmap_size=268435456")  # メモリマップサイズ (256MB)
        cursor.close()

    def create_tables(self):
        """全テーブルを作成"""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """全テーブルを削除"""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """新しいセッションを取得"""
        if self.session_factory is None:
            raise DatabaseError(
                "DatabaseManager not properly initialized",
                error_code="DB_NOT_INITIALIZED",
            )
        return self.session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        セッションのコンテキストマネージャー

        Usage:
            with db_manager.session_scope() as session:
                # セッションを使用した処理
                pass
        """
        if self.session_factory is None:
            raise DatabaseError(
                "DatabaseManager not properly initialized",
                error_code="DB_NOT_INITIALIZED",
            )

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            # エラーを適切な例外に変換
            converted_error = handle_database_exception(e)
            log_error_with_context(converted_error, {
                "operation": "database_session",
                "session_id": id(session)
            })
            raise converted_error
        finally:
            session.close()

    @contextmanager
    def transaction_scope(self, retry_count: int = 3, retry_delay: float = 0.1) -> Generator[Session, None, None]:
        """
        明示的なトランザクション管理とデッドロック対応

        Args:
            retry_count: デッドロック時の再試行回数
            retry_delay: 再試行時の待機時間（秒）

        Usage:
            with db_manager.transaction_scope() as session:
                # 複数のDB操作を含む処理
                session.add(obj1)
                session.add(obj2)
                # 明示的にflushして中間結果を確認
                session.flush()
                # さらに複雑な処理...
        """
        if self.session_factory is None:
            raise DatabaseError(
                "DatabaseManager not properly initialized",
                error_code="DB_NOT_INITIALIZED",
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
                    logger.warning("Retriable database error, retrying",
                                 attempt=attempt + 1,
                                 max_attempts=retry_count + 1,
                                 retry_delay=retry_delay,
                                 error=str(e))
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数バックオフ
                    continue
                else:
                    converted_error = handle_database_exception(e)
                    log_error_with_context(converted_error, {
                        "operation": "database_transaction",
                        "attempts": attempt + 1,
                        "retry_count": retry_count
                    })
                    raise converted_error
            except Exception as e:
                converted_error = handle_database_exception(e)
                log_error_with_context(converted_error, {
                    "operation": "database_transaction",
                    "error_type": "unexpected"
                })
                raise converted_error
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

    def get_alembic_config(self) -> Config:
        """Alembic設定を取得"""
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", self.config.database_url)
        return alembic_cfg

    def init_alembic(self):
        """Alembicの初期化（初回マイグレーション作成）"""
        try:
            alembic_cfg = self.get_alembic_config()
            command.revision(
                alembic_cfg, autogenerate=True, message="Initial migration"
            )
            logger.info("Alembic initialized successfully")
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(converted_error, {
                "operation": "alembic_initialization"
            })
            raise converted_error

    def migrate(self, message: str = "Auto migration"):
        """新しいマイグレーションを作成"""
        try:
            alembic_cfg = self.get_alembic_config()
            command.revision(alembic_cfg, autogenerate=True, message=message)
            logger.info("Migration created", message=message)
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(converted_error, {
                "operation": "migration_creation",
                "message": message
            })
            raise converted_error

    def upgrade(self, revision: str = "head"):
        """マイグレーションを適用"""
        try:
            alembic_cfg = self.get_alembic_config()
            command.upgrade(alembic_cfg, revision)
            logger.info("Database upgraded", revision=revision)
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(converted_error, {
                "operation": "database_upgrade",
                "revision": revision
            })
            raise converted_error

    def downgrade(self, revision: str = "-1"):
        """マイグレーションをロールバック"""
        try:
            alembic_cfg = self.get_alembic_config()
            command.downgrade(alembic_cfg, revision)
            logger.info("Database downgraded", revision=revision)
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(converted_error, {
                "operation": "database_downgrade",
                "revision": revision
            })
            raise converted_error

    def current_revision(self) -> str:
        """現在のリビジョンを取得"""
        try:
            from alembic.runtime.migration import MigrationContext

            with self.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision() or "None"
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(converted_error, {
                "operation": "current_revision_retrieval"
            })
            raise converted_error

    def bulk_insert(self, model_class, data_list: list, batch_size: int = 1000):
        """
        大量データの一括挿入

        Args:
            model_class: 挿入するモデルクラス
            data_list: 挿入するデータのリスト（辞書形式）
            batch_size: バッチサイズ
        """
        if not data_list:
            return

        operation_logger = logger.bind(
            operation="bulk_insert",
            model_class=model_class.__name__,
            total_records=len(data_list),
            batch_size=batch_size
        )
        operation_logger.info("Starting bulk insert")

        with self.transaction_scope() as session:
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i : i + batch_size]
                session.bulk_insert_mappings(model_class, batch)
                session.flush()
                log_database_operation(
                    "bulk_insert_batch",
                    model_class.__table__.name,
                    batch_number=i // batch_size + 1,
                    batch_size=len(batch)
                )

        operation_logger.info("Bulk insert completed")

    def bulk_update(self, model_class, data_list: list, batch_size: int = 1000):
        """
        大量データの一括更新

        Args:
            model_class: 更新するモデルクラス
            data_list: 更新するデータのリスト（辞書形式、idが必要）
            batch_size: バッチサイズ
        """
        if not data_list:
            return

        operation_logger = logger.bind(
            operation="bulk_update",
            model_class=model_class.__name__,
            total_records=len(data_list),
            batch_size=batch_size
        )
        operation_logger.info("Starting bulk update")

        with self.transaction_scope() as session:
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i : i + batch_size]
                session.bulk_update_mappings(model_class, batch)
                session.flush()
                log_database_operation(
                    "bulk_update_batch",
                    model_class.__table__.name,
                    batch_number=i // batch_size + 1,
                    batch_size=len(batch)
                )

        operation_logger.info("Bulk update completed")

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

            db_manager.atomic_operation([operation1, operation2])
        """
        operation_logger = logger.bind(
            operation="atomic_operation",
            operation_count=len(operations),
            retry_count=retry_count
        )
        operation_logger.info("Starting atomic operation")

        with self.transaction_scope(retry_count=retry_count) as session:
            for i, operation in enumerate(operations):
                operation(session)
                # 各操作後に中間状態をflush（デバッグ時などに有用）
                session.flush()
                operation_logger.debug("Operation completed", step=i + 1)

        operation_logger.info("Atomic operation completed")

    def execute_query(self, query: str, params: dict = None):
        """
        生のSQLクエリを実行（最適化されたクエリ用）

        Args:
            query: 実行するSQLクエリ
            params: クエリパラメータ

        Returns:
            クエリ結果
        """
        operation_logger = logger.bind(
            operation="execute_query",
            query_length=len(query),
            has_params=bool(params)
        )
        operation_logger.info("Executing raw SQL query")

        with self.engine.connect() as connection:
            result = connection.execute(query, params or {})
            results = result.fetchall()
            operation_logger.info("Query executed", result_count=len(results))
            return results

    def optimize_database(self):
        """
        データベースの最適化を実行
        """
        operation_logger = logger.bind(
            operation="optimize_database",
            database_type=self._get_database_type()
        )
        operation_logger.info("Starting database optimization")

        if "sqlite" in self.config.database_url:
            with self.engine.connect() as connection:
                # VACUUM操作でデータベースファイルを最適化
                operation_logger.info("Executing VACUUM")
                connection.execute(text("VACUUM"))
                # ANALYZE操作で統計情報を更新
                operation_logger.info("Executing ANALYZE")
                connection.execute(text("ANALYZE"))

        operation_logger.info("Database optimization completed")


# デフォルトのデータベースマネージャー
db_manager = DatabaseManager()


# 便利な関数
def get_db() -> Generator[Session, None, None]:
    """
    依存性注入用のデータベースセッション取得関数

    Usage:
        def some_function(db: Session = Depends(get_db)):
            # データベース操作
            pass
    """
    with db_manager.session_scope() as session:
        yield session


def init_db():
    """データベースの初期化"""
    db_manager.create_tables()


def reset_db():
    """データベースのリセット（開発用）"""
    db_manager.drop_tables()
    db_manager.create_tables()


def init_migration():
    """マイグレーションの初期化"""
    db_manager.init_alembic()


def create_migration(message: str = "Auto migration"):
    """新しいマイグレーションファイルを作成"""
    db_manager.migrate(message)


def upgrade_db(revision: str = "head"):
    """データベースをアップグレード"""
    db_manager.upgrade(revision)


def downgrade_db(revision: str = "-1"):
    """データベースをダウングレード"""
    db_manager.downgrade(revision)


def get_current_revision() -> str:
    """現在のリビジョンを取得"""
    return db_manager.current_revision()
