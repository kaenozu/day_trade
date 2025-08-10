"""
データベース基盤モジュール
SQLAlchemyを使用したデータベース接続とセッション管理

Issue #120: declarative_base()の定義場所の最適化
- Baseクラスをbase.pyからインポートするように変更
- database.pyはデータベース接続管理に責務を特化
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from alembic import command
from alembic.config import Config

from ..utils.exceptions import DatabaseError, handle_database_exception
from ..utils.logging_config import (
    get_context_logger,
    log_database_operation,
    log_error_with_context,
)
from ..utils.performance_config import get_performance_config

# Issue #120: Baseクラスをbase.pyからインポート（責務の明確化）
from .base import Base

# Global Trading Engine モデル追加
from .forex_models import ForexPrice
from .crypto_models import CryptoPrice
from .global_models import GlobalMarketData

logger = get_context_logger(__name__)

# グローバルデータベースマネージャー（シングルトン）
_global_db_manager: Optional['DatabaseManager'] = None

# テスト用のデータベースURL
TEST_DATABASE_URL = "sqlite:///:memory:"


class DatabaseConfig:
    """データベース設定クラス（ConfigManager統合対応）"""

    def __init__(
        self,
        database_url: Optional[str] = None,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        connect_args: Optional[Dict[str, Any]] = None,
        config_manager=None,
        use_performance_config: bool = True,
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
            config_manager: ConfigManagerインスタンス（依存性注入）
            use_performance_config: パフォーマンス設定を使用するか
        """
        self._config_manager = config_manager
        self._use_performance_config = use_performance_config

        # パフォーマンス設定の統合
        if use_performance_config:
            try:
                perf_config = get_performance_config()
                # パフォーマンス設定からデフォルト値を取得
                if pool_size == 5:  # デフォルト値の場合のみ上書き
                    pool_size = perf_config.database.pool_size
                if max_overflow == 10:
                    max_overflow = perf_config.database.max_overflow
                if pool_timeout == 30:
                    pool_timeout = perf_config.database.pool_timeout
                if pool_recycle == 3600:
                    pool_recycle = perf_config.database.pool_recycle

                logger.debug(
                    "パフォーマンス設定を統合",
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    optimization_level=perf_config.optimization_level,
                )
            except Exception as e:
                logger.warning(f"パフォーマンス設定の統合に失敗: {e}")

        # 設定の優先順位: 引数 > config_manager > 環境変数 > デフォルト
        self.database_url = self._get_config_value(
            "database_url", database_url, "DATABASE_URL", "sqlite:///./day_trade.db"
        )
        self.echo = self._get_config_value("echo", echo, "DB_ECHO", False, bool)
        self.pool_size = self._get_config_value(
            "pool_size", pool_size, "DB_POOL_SIZE", 5, int
        )
        self.max_overflow = self._get_config_value(
            "max_overflow", max_overflow, "DB_MAX_OVERFLOW", 10, int
        )
        self.pool_timeout = self._get_config_value(
            "pool_timeout", pool_timeout, "DB_POOL_TIMEOUT", 30, int
        )
        self.pool_recycle = self._get_config_value(
            "pool_recycle", pool_recycle, "DB_POOL_RECYCLE", 3600, int
        )
        self.connect_args = connect_args or {"check_same_thread": False}

        # SQLite最適化パラメータ（設定化対応）
        self.sqlite_cache_size = self._get_config_value(
            "sqlite_cache_size", None, "DB_SQLITE_CACHE_SIZE", 10000, int
        )
        self.sqlite_mmap_size = self._get_config_value(
            "sqlite_mmap_size", None, "DB_SQLITE_MMAP_SIZE", 268435456, int
        )  # 256MB
        self.sqlite_temp_store = self._get_config_value(
            "sqlite_temp_store", None, "DB_SQLITE_TEMP_STORE", "memory", str
        )
        self.sqlite_journal_mode = self._get_config_value(
            "sqlite_journal_mode", None, "DB_SQLITE_JOURNAL_MODE", "WAL", str
        )
        self.sqlite_synchronous = self._get_config_value(
            "sqlite_synchronous", None, "DB_SQLITE_SYNCHRONOUS", "NORMAL", str
        )

    def _get_config_value(
        self, key: str, explicit_value, env_key: str, default_value, type_converter=str
    ):
        """設定値を優先順位に従って取得（優先度: 明示的引数 > 環境変数 > ConfigManager > デフォルト値）"""
        # 1. 明示的な引数（最優先）
        if explicit_value is not None and explicit_value != (
            False if type_converter is bool else 0
        ):
            return explicit_value

        # 2. 環境変数（ConfigManagerより優先）
        env_value = os.environ.get(env_key)
        if env_value is not None:
            try:
                if type_converter is bool:
                    return env_value.lower() in ("true", "1", "yes", "on")
                return type_converter(env_value)
            except (ValueError, TypeError):
                pass

        # 3. ConfigManagerからの設定値
        if self._config_manager:
            try:
                database_settings = self._config_manager.get_database_settings()
                if key == "database_url":
                    config_value = database_settings.url
                elif key == "echo":
                    # デフォルトでFalse（ログ出力はデフォルトで無効）
                    config_value = getattr(database_settings, "echo", False)
                else:
                    # その他の設定項目は現在の設定ファイルには含まれていないため、デフォルト値を使用
                    config_value = None

                if config_value is not None:
                    return (
                        type_converter(config_value)
                        if type_converter is not str
                        else config_value
                    )
            except Exception:
                pass  # ConfigManagerが利用できない場合は無視

        # 4. デフォルト値（最低優先度）
        return default_value

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
    """データベース管理クラス（改善版・依存性注入対応）"""

    def __init__(self, config: Optional[DatabaseConfig] = None, config_manager=None):
        """
        Args:
            config: データベース設定
            config_manager: ConfigManagerインスタンス（依存性注入）
        """
        self._config_manager = config_manager
        self.config = config or DatabaseConfig(config_manager=config_manager)
        self.engine = None
        self.session_factory = None
        self._connection_pool_stats = {
            "created_connections": 0,
            "closed_connections": 0,
            "active_sessions": 0,
        }
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
                event.listen(
                    self.engine,
                    "connect",
                    lambda conn, rec: self._set_sqlite_pragma(conn, rec),
                )

            # セッションファクトリーの作成
            self.session_factory = sessionmaker(bind=self.engine)

            logger.info(
                "Database engine initialized",
                extra={
                    "database_url": self.config.database_url,
                    "database_type": self._get_database_type(),
                },
            )

        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error,
                {
                    "operation": "database_initialization",
                    "database_url": self.config.database_url,
                },
            )
            raise converted_error from e

    def _set_sqlite_pragma(self, dbapi_connection, connection_record):
        """SQLiteの設定（SQLインジェクション対策・セキュリティ強化版）"""
        cursor = dbapi_connection.cursor()

        try:
            # 固定の外部キー制約設定（常に安全）
            cursor.execute("PRAGMA foreign_keys=ON")

            # 各PRAGMA設定値の安全性検証と実行
            self._execute_safe_pragma(cursor, "journal_mode", self.config.sqlite_journal_mode)
            self._execute_safe_pragma(cursor, "synchronous", self.config.sqlite_synchronous)
            self._execute_safe_pragma(cursor, "cache_size", self.config.sqlite_cache_size)
            self._execute_safe_pragma(cursor, "temp_store", self.config.sqlite_temp_store)
            self._execute_safe_pragma(cursor, "mmap_size", self.config.sqlite_mmap_size)

            logger.debug("SQLite PRAGMA設定完了", extra={
                "journal_mode": self.config.sqlite_journal_mode,
                "synchronous": self.config.sqlite_synchronous,
                "cache_size": self.config.sqlite_cache_size,
                "temp_store": self.config.sqlite_temp_store,
                "mmap_size": self.config.sqlite_mmap_size
            })

        except Exception as e:
            logger.error(f"SQLite PRAGMA設定エラー: {e}")
            # PRAGMA設定の失敗は致命的ではないため、接続は継続
        finally:
            cursor.close()

    def _execute_safe_pragma(self, cursor, pragma_name: str, pragma_value):
        """
        安全なPRAGMA実行（SQLインジェクション対策）

        Args:
            cursor: データベースカーソル
            pragma_name: PRAGMA名
            pragma_value: PRAGMA値
        """
        # 1. PRAGMA値のホワイトリスト検証
        validated_value = self._validate_pragma_value(pragma_name, pragma_value)

        if validated_value is None:
            logger.warning(f"無効なPRAGMA値をスキップ: {pragma_name}={pragma_value}")
            return

        # 2. 安全なPRAGMA実行（文字列結合を避けたパラメータ化）
        try:
            # SQLiteのPRAGMAは動的パラメータに対応していないため、
            # 検証済み値のみを使用した安全な文字列構築を実行
            pragma_sql = f"PRAGMA {pragma_name}={validated_value}"
            cursor.execute(pragma_sql)

            logger.debug(f"PRAGMA実行成功: {pragma_name}={validated_value}")

        except Exception as e:
            logger.warning(f"PRAGMA実行失敗: {pragma_name}={validated_value} - {e}")

    def _validate_pragma_value(self, pragma_name: str, pragma_value) -> str:
        """
        PRAGMA値のホワイトリスト検証（SQLインジェクション対策）

        Args:
            pragma_name: PRAGMA名
            pragma_value: 検証対象の値

        Returns:
            str: 検証済み安全な値、または無効な場合はNone
        """
        if pragma_value is None:
            return None

        # PRAGMA別のホワイトリスト検証
        if pragma_name == "journal_mode":
            allowed_values = {"DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"}
            value_str = str(pragma_value).upper()
            if value_str in allowed_values:
                return value_str
            else:
                logger.warning(f"journal_mode無効値: {pragma_value}, 許可値: {allowed_values}")
                return "WAL"  # デフォルト安全値

        elif pragma_name == "synchronous":
            allowed_values = {"OFF", "NORMAL", "FULL", "EXTRA", "0", "1", "2", "3"}
            value_str = str(pragma_value).upper()
            if value_str in allowed_values:
                return value_str
            else:
                logger.warning(f"synchronous無効値: {pragma_value}, 許可値: {allowed_values}")
                return "NORMAL"  # デフォルト安全値

        elif pragma_name == "cache_size":
            try:
                # 整数値の検証（負の値も許可、SQLiteの仕様に準拠）
                cache_size = int(pragma_value)
                # 範囲制限: -1000000 to 1000000（メモリ枯渇攻撃防止）
                if -1000000 <= cache_size <= 1000000:
                    return str(cache_size)
                else:
                    logger.warning(f"cache_size範囲外: {pragma_value}, 範囲: -1000000~1000000")
                    return "10000"  # デフォルト安全値
            except (ValueError, TypeError):
                logger.warning(f"cache_size無効形式: {pragma_value}")
                return "10000"  # デフォルト安全値

        elif pragma_name == "temp_store":
            allowed_values = {"DEFAULT", "FILE", "MEMORY", "0", "1", "2"}
            value_str = str(pragma_value).upper()
            if value_str in allowed_values:
                return value_str
            else:
                logger.warning(f"temp_store無効値: {pragma_value}, 許可値: {allowed_values}")
                return "MEMORY"  # デフォルト安全値

        elif pragma_name == "mmap_size":
            try:
                # 整数値の検証
                mmap_size = int(pragma_value)
                # 範囲制限: 0 to 1GB（メモリマップサイズ制限）
                if 0 <= mmap_size <= 1073741824:  # 1GB = 1024^3
                    return str(mmap_size)
                else:
                    logger.warning(f"mmap_size範囲外: {pragma_value}, 範囲: 0~1073741824")
                    return "268435456"  # デフォルト安全値（256MB）
            except (ValueError, TypeError):
                logger.warning(f"mmap_size無効形式: {pragma_value}")
                return "268435456"  # デフォルト安全値

        else:
            # 未知のPRAGMAは拒否
            logger.error(f"未サポートのPRAGMA: {pragma_name}")
            return None

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

    def get_alembic_config(self, config_path: Optional[str] = None) -> Config:
        """
        Alembic設定を取得（TOCTOU脆弱性対策・セキュリティ強化版）

        Args:
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）
        """
        if config_path is None:
            # 設定ファイルの自動検索（セキュリティ強化）
            config_path = self._find_secure_alembic_config()

        # TOCTOU脆弱性対策: 安全なファイルパス検証
        validated_config_path = self._validate_alembic_config_path(config_path)

        # 安全なAlembic設定作成
        try:
            alembic_cfg = Config(validated_config_path)
            alembic_cfg.set_main_option("sqlalchemy.url", self.config.database_url)

            logger.debug(f"Alembic設定読み込み成功: {validated_config_path}")
            return alembic_cfg

        except Exception as e:
            logger.error(f"Alembic設定読み込みエラー: {validated_config_path} - {e}")
            raise DatabaseError(
                f"Failed to load Alembic config: {config_path}",
                error_code="ALEMBIC_CONFIG_LOAD_ERROR",
            ) from e

    def _find_secure_alembic_config(self) -> str:
        """
        安全なAlembic設定ファイル検索（TOCTOU対策）

        Returns:
            str: 検証済み安全な設定ファイルパス

        Raises:
            DatabaseError: 設定ファイルが見つからない場合
        """

        # 許可された検索ベースディレクトリ
        allowed_base_dirs = [
            Path.cwd(),                           # 現在の作業ディレクトリ
            Path(__file__).parent.parent.parent,  # プロジェクトルート
        ]

        # 検索対象ファイル名パターン
        config_filenames = ["alembic.ini"]

        for base_dir in allowed_base_dirs:
            try:
                # ベースディレクトリの正規化と検証
                base_dir_resolved = base_dir.resolve()

                for filename in config_filenames:
                    config_path = base_dir_resolved / filename

                    # 原子的なファイル存在・読み取り可能チェック
                    if self._is_safe_readable_file(config_path):
                        logger.debug(f"Alembic設定ファイル発見: {config_path}")
                        return str(config_path)

            except Exception as e:
                logger.debug(f"ディレクトリ検索エラー: {base_dir} - {e}")
                continue

        # 設定ファイルが見つからない場合
        raise DatabaseError(
            "alembic.ini not found in any secure locations",
            error_code="ALEMBIC_CONFIG_NOT_FOUND",
        )

    def _validate_alembic_config_path(self, config_path: str) -> str:
        """
        Alembic設定ファイルパスの安全性検証（TOCTOU対策）

        Args:
            config_path: 検証対象のファイルパス

        Returns:
            str: 検証済み安全なファイルパス

        Raises:
            DatabaseError: 危険なパスまたはファイルアクセス不可の場合
        """

        try:
            # 1. パス正規化とセキュリティチェック
            path_obj = Path(config_path).resolve()

            # 2. 危険なパスパターンの検出
            path_str = str(path_obj).lower()
            dangerous_patterns = [
                "/etc/", "/usr/", "/var/", "/root/", "/boot/",  # Unix系システムディレクトリ
                "c:\\windows\\", "c:\\program files\\",        # Windowsシステムディレクトリ
                "\\\\", "/..", "\\..",                         # UNCパス・パストラバーサル
            ]

            for pattern in dangerous_patterns:
                if pattern in path_str:
                    logger.warning(f"危険なAlembicパスパターン検出: {config_path}")
                    raise DatabaseError(
                        f"Dangerous path pattern detected: {config_path}",
                        error_code="ALEMBIC_DANGEROUS_PATH",
                    )

            # 3. 許可されたベースディレクトリ内かチェック
            allowed_base_dirs = [
                Path.cwd().resolve(),                           # 現在の作業ディレクトリ
                Path(__file__).parent.parent.parent.resolve(), # プロジェクトルート
            ]

            is_allowed = False
            for allowed_base in allowed_base_dirs:
                try:
                    # 許可されたベースディレクトリ内またはその配下かチェック
                    if (path_obj == allowed_base or allowed_base in path_obj.parents):
                        is_allowed = True
                        break
                except Exception:
                    continue

            if not is_allowed:
                logger.warning(f"許可されていないAlembicパス: {path_obj}")
                raise DatabaseError(
                    f"Path outside allowed directories: {config_path}",
                    error_code="ALEMBIC_PATH_NOT_ALLOWED",
                )

            # 4. ファイルの存在と読み取り可能性を原子的にチェック
            if not self._is_safe_readable_file(path_obj):
                raise DatabaseError(
                    f"Alembic config file not accessible: {config_path}",
                    error_code="ALEMBIC_CONFIG_NOT_ACCESSIBLE",
                )

            logger.debug(f"Alembicパス検証完了: {path_obj}")
            return str(path_obj)

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Alembicパス検証エラー: {config_path} - {e}")
            raise DatabaseError(
                f"Path validation failed: {config_path}",
                error_code="ALEMBIC_PATH_VALIDATION_ERROR",
            ) from e

    def _is_safe_readable_file(self, file_path: Path) -> bool:
        """
        ファイルの安全な読み取り可能性チェック（TOCTOU対策）

        Args:
            file_path: チェック対象ファイルパス

        Returns:
            bool: 安全に読み取り可能な場合True
        """
        try:
            # 原子的操作: ファイル存在・タイプ・読み取り権限のチェック
            if not file_path.exists():
                return False

            if not file_path.is_file():
                logger.warning(f"通常ファイルではない: {file_path}")
                return False

            # シンボリックリンク攻撃対策
            if file_path.is_symlink():
                logger.warning(f"シンボリックリンクのためスキップ: {file_path}")
                return False

            # ファイルサイズ制限（設定ファイルが異常に大きい場合を検出）
            stat_info = file_path.stat()
            if stat_info.st_size > 10 * 1024 * 1024:  # 10MB制限
                logger.warning(f"設定ファイルが大きすぎます: {file_path} ({stat_info.st_size} bytes)")
                return False

            # 読み取り権限の確認
            with open(file_path, 'r', encoding='utf-8') as f:
                # ファイルの先頭を少し読んで読み取り可能性を確認
                f.read(1)

            return True

        except (PermissionError, OSError, FileNotFoundError):
            # ファイルアクセス不可の場合
            return False
        except UnicodeDecodeError:
            # 不正な文字エンコーディングの場合
            logger.warning(f"不正なエンコーディング: {file_path}")
            return False
        except Exception as e:
            logger.debug(f"ファイル読み取りチェックエラー: {file_path} - {e}")
            return False

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
            log_error_with_context(
                converted_error, {"operation": "alembic_initialization"}
            )
            raise converted_error from e

    def migrate(self, message: str = "Auto migration"):
        """新しいマイグレーションを作成"""
        try:
            alembic_cfg = self.get_alembic_config()
            command.revision(alembic_cfg, autogenerate=True, message=message)
            logger.info("Migration created", extra={"message": message})
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error, {"operation": "migration_creation", "message": message}
            )
            raise converted_error from e

    def upgrade(self, revision: str = "head"):
        """マイグレーションを適用"""
        try:
            alembic_cfg = self.get_alembic_config()
            command.upgrade(alembic_cfg, revision)
            logger.info("Database upgraded", extra={"revision": revision})
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error, {"operation": "database_upgrade", "revision": revision}
            )
            raise converted_error from e

    def downgrade(self, revision: str = "-1"):
        """マイグレーションをロールバック"""
        try:
            alembic_cfg = self.get_alembic_config()
            command.downgrade(alembic_cfg, revision)
            logger.info("Database downgraded", extra={"revision": revision})
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error,
                {"operation": "database_downgrade", "revision": revision},
            )
            raise converted_error from e

    def current_revision(self) -> str:
        """現在のリビジョンを取得"""
        try:
            from alembic.runtime.migration import MigrationContext

            with self.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision() or "None"
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error, {"operation": "current_revision_retrieval"}
            )
            raise converted_error from e

    def bulk_insert(self, model_class, data_list: list, batch_size: int = 1000):
        """
        大量データの一括挿入（堅牢性向上版）

        Args:
            model_class: 挿入するモデルクラス
            data_list: 挿入するデータのリスト（辞書形式）
            batch_size: バッチサイズ
        """
        if not data_list:
            return

        operation_logger = logger
        operation_logger.info("Starting bulk insert")

        try:
            with self.transaction_scope() as session:
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i : i + batch_size]
                    batch_number = i // batch_size + 1

                    try:
                        session.bulk_insert_mappings(model_class, batch)
                        session.flush()
                        log_database_operation(
                            "bulk_insert_batch",
                            duration=0.0,
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
        """
        if not data_list:
            return

        operation_logger = logger
        operation_logger.info("Starting bulk update")

        try:
            with self.transaction_scope() as session:
                for i in range(0, len(data_list), batch_size):
                    batch = data_list[i : i + batch_size]
                    batch_number = i // batch_size + 1

                    try:
                        session.bulk_update_mappings(model_class, batch)
                        session.flush()
                        log_database_operation(
                            "bulk_update_batch",
                            model_class.__table__.name,
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
        operation_logger = logger
        operation_logger.info("Starting atomic operation")

        with self.transaction_scope(retry_count=retry_count) as session:
            for i, operation in enumerate(operations):
                operation(session)
                # 各操作後に中間状態をflush（デバッグ時などに有用）
                session.flush()
                operation_logger.debug("Operation completed", extra={"step": i + 1})

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
        operation_logger = logger
        operation_logger.info("Executing raw SQL query")

        with self.engine.connect() as connection:
            result = connection.execute(query, params or {})
            results = result.fetchall()
            operation_logger.info(
                "Query executed", extra={"result_count": len(results)}
            )
            return results

    def optimize_database(self):
        """
        データベースの最適化を実行
        """
        operation_logger = logger
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


# デフォルトのデータベースマネージャー（後方互換性のため）
# 注意: 本番環境では依存性注入の使用を推奨
_default_db_manager = None


def get_default_database_manager(config_manager=None) -> DatabaseManager:
    """
    デフォルトのデータベースマネージャーを取得（依存性注入対応）

    Args:
        config_manager: ConfigManagerインスタンス（オプション）

    Returns:
        DatabaseManager: データベースマネージャーインスタンス
    """
    global _default_db_manager
    if _default_db_manager is None:
        _default_db_manager = DatabaseManager(config_manager=config_manager)
    return _default_db_manager


def set_default_database_manager(manager: DatabaseManager):
    """
    デフォルトのデータベースマネージャーを設定（テスト用）

    Args:
        manager: データベースマネージャーインスタンス
    """
    global _default_db_manager
    _default_db_manager = manager


# 後方互換性のためのグローバルインスタンス
# ConfigManagerを使用してデータベース設定を管理
try:
    from ..config.config_manager import ConfigManager

    _config_manager = ConfigManager()
    db_manager = get_default_database_manager(_config_manager)
except ImportError:
    # ConfigManagerが利用できない場合はデフォルト設定で作成
    db_manager = get_default_database_manager()
except Exception:
    # ConfigManagerの初期化に失敗した場合はデフォルト設定で作成
    db_manager = get_default_database_manager()


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


# 拡張機能をDatabaseManagerクラスに追加
def _add_enhanced_features():
    """DatabaseManagerクラスに拡張機能を動的に追加"""

    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """接続プール統計情報を取得"""
        if not self.engine:
            return {}

        pool = self.engine.pool

        # 基本統計情報
        stats = {
            "pool_size": getattr(pool, "size", lambda: 0)(),
            "checked_in": getattr(pool, "checkedin", lambda: 0)(),
            "checked_out": getattr(pool, "checkedout", lambda: 0)(),
            "overflow": getattr(pool, "overflow", lambda: 0)(),
            "invalid": getattr(pool, "invalid", lambda: 0)(),
        }

        # 詳細統計（利用可能な場合）
        stats.update(self._connection_pool_stats)

        return stats

    def health_check(self) -> Dict[str, Any]:
        """データベース接続のヘルスチェック"""
        health_status = {
            "status": "unknown",
            "database_type": self._get_database_type(),
            "database_url": self.config.database_url,
            "connection_pool": {},
            "last_check": time.time(),
            "errors": [],
        }

        try:
            # 基本接続テスト
            with self.session_scope() as session:
                result = session.execute(text("SELECT 1")).scalar()
                if result == 1:
                    health_status["status"] = "healthy"
                else:
                    health_status["status"] = "unhealthy"
                    health_status["errors"].append("Unexpected query result")

            # 接続プール統計
            health_status["connection_pool"] = self.get_connection_pool_stats()

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["errors"].append(str(e))
            log_error_with_context(e, {"operation": "database_health_check"})

        return health_status

    def optimize_performance(self):
        """パフォーマンス最適化設定を適用"""
        if not self.config.is_sqlite():
            return  # SQLite以外は現在対応なし

        try:
            with self.session_scope() as session:
                # SQLiteの高度な最適化設定
                optimizations = [
                    "PRAGMA journal_mode=WAL",
                    "PRAGMA synchronous=NORMAL",
                    "PRAGMA cache_size=10000",
                    "PRAGMA temp_store=memory",
                    "PRAGMA mmap_size=268435456",
                    "PRAGMA page_size=4096",
                    "PRAGMA auto_vacuum=INCREMENTAL",
                    "PRAGMA incremental_vacuum(1000)",
                ]

                for pragma in optimizations:
                    session.execute(text(pragma))

                logger.info("Database performance optimizations applied")

        except Exception as e:
            log_error_with_context(e, {"operation": "database_optimization"})

    @contextmanager
    def performance_monitor(self, operation_name: str):
        """パフォーマンス監視コンテキストマネージャー"""
        start_time = time.perf_counter()
        pool_stats_before = self.get_connection_pool_stats()

        try:
            yield
        finally:
            elapsed_time = time.perf_counter() - start_time
            pool_stats_after = self.get_connection_pool_stats()

            # パフォーマンス情報をログ出力
            logger.info(
                "Database operation performance",
                operation=operation_name,
                elapsed_ms=round(elapsed_time * 1000, 2),
                pool_before=pool_stats_before,
                pool_after=pool_stats_after,
            )

    def create_factory(self, config_manager=None) -> "DatabaseManager":
        """ファクトリー方式でDatabaseManagerインスタンスを作成"""
        return DatabaseManager(
            config=DatabaseConfig(config_manager=config_manager),
            config_manager=config_manager,
        )

    def create_performance_optimized_factory(self, config_manager=None):
        """パフォーマンス最適化されたDatabaseManagerを作成"""
        try:
            from .performance_database import PerformanceOptimizedDatabaseManager

            config = DatabaseConfig(
                config_manager=config_manager, use_performance_config=True
            )

            return PerformanceOptimizedDatabaseManager(config)
        except ImportError:
            logger.warning(
                "パフォーマンス最適化モジュールが利用できません。通常版を使用します。"
            )
            return self.create_factory(config_manager)

    # メソッドを動的に追加
    DatabaseManager.get_connection_pool_stats = get_connection_pool_stats
    DatabaseManager.health_check = health_check
    DatabaseManager.optimize_performance = optimize_performance
    DatabaseManager.performance_monitor = performance_monitor
    DatabaseManager.create_factory = create_factory


# 拡張機能を適用
_add_enhanced_features()


# 依存性注入用のファクトリー関数
def create_database_manager(
    config_manager=None, use_performance_optimization: bool = True
) -> DatabaseManager:
    """
    ConfigManager統合版のDatabaseManagerを作成

    Args:
        config_manager: ConfigManagerインスタンス
        use_performance_optimization: パフォーマンス最適化版を使用するか

    Returns:
        DatabaseManager: 作成されたデータベースマネージャー
    """
    if use_performance_optimization:
        try:
            from .performance_database import PerformanceOptimizedDatabaseManager

            config = DatabaseConfig(
                config_manager=config_manager, use_performance_config=True
            )

            return PerformanceOptimizedDatabaseManager(config)
        except ImportError:
            logger.warning(
                "パフォーマンス最適化モジュールが利用できません。通常版を使用します。"
            )

    return DatabaseManager(
        config=DatabaseConfig(config_manager=config_manager),
        config_manager=config_manager,
    )

# Global Trading Engine用のグローバル関数
def get_global_db_manager() -> DatabaseManager:
    """グローバルデータベースマネージャー取得"""
    global _global_db_manager
    if _global_db_manager is None:
        _global_db_manager = create_database_manager()
    return _global_db_manager

def get_session() -> Session:
    """データベースセッション取得（グローバル）"""
    return get_global_db_manager().get_session()

def init_global_database():
    """グローバルデータベース初期化"""
    db_manager = get_global_db_manager()
    db_manager.create_all_tables()
    logger.info("Global database initialized for Global Trading Engine")
