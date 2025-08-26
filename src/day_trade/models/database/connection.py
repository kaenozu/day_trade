"""
データベース接続管理モジュール
エンジン初期化、接続プール管理、SQLite PRAGMA設定

Issue #120: declarative_base()の定義場所の最適化対応
- 接続管理の責務を明確化
- セキュリティ強化（SQLインジェクション対策）
"""

from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from ...utils.exceptions import handle_database_exception
from ...utils.logging_config import (
    get_context_logger,
    log_error_with_context,
)
from .config import DatabaseConfig

logger = get_context_logger(__name__)


class ConnectionManager:
    """データベース接続管理クラス"""

    def __init__(self, config: DatabaseConfig):
        """
        接続管理の初期化

        Args:
            config: データベース設定
        """
        self.config = config
        self.engine = None
        self._initialize_engine()

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

            logger.info(
                "Database engine initialized",
                extra={
                    "database_url": self.config.database_url,
                    "database_type": self.config.get_database_type(),
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
            self._execute_safe_pragma(
                cursor, "journal_mode", self.config.sqlite_journal_mode
            )
            self._execute_safe_pragma(
                cursor, "synchronous", self.config.sqlite_synchronous
            )
            self._execute_safe_pragma(
                cursor, "cache_size", self.config.sqlite_cache_size
            )
            self._execute_safe_pragma(
                cursor, "temp_store", self.config.sqlite_temp_store
            )
            self._execute_safe_pragma(cursor, "mmap_size", self.config.sqlite_mmap_size)

            logger.debug(
                "SQLite PRAGMA設定完了",
                extra={
                    "journal_mode": self.config.sqlite_journal_mode,
                    "synchronous": self.config.sqlite_synchronous,
                    "cache_size": self.config.sqlite_cache_size,
                    "temp_store": self.config.sqlite_temp_store,
                    "mmap_size": self.config.sqlite_mmap_size,
                },
            )

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
                logger.warning(
                    f"journal_mode無効値: {pragma_value}, 許可値: {allowed_values}"
                )
                return "WAL"  # デフォルト安全値

        elif pragma_name == "synchronous":
            allowed_values = {"OFF", "NORMAL", "FULL", "EXTRA", "0", "1", "2", "3"}
            value_str = str(pragma_value).upper()
            if value_str in allowed_values:
                return value_str
            else:
                logger.warning(
                    f"synchronous無効値: {pragma_value}, 許可値: {allowed_values}"
                )
                return "NORMAL"  # デフォルト安全値

        elif pragma_name == "cache_size":
            try:
                # 整数値の検証（負の値も許可、SQLiteの仕様に準拠）
                cache_size = int(pragma_value)
                # 範囲制限: -1000000 to 1000000（メモリ枯渇攻撃防止）
                if -1000000 <= cache_size <= 1000000:
                    return str(cache_size)
                else:
                    logger.warning(
                        f"cache_size範囲外: {pragma_value}, 範囲: -1000000~1000000"
                    )
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
                logger.warning(
                    f"temp_store無効値: {pragma_value}, 許可値: {allowed_values}"
                )
                return "MEMORY"  # デフォルト安全値

        elif pragma_name == "mmap_size":
            try:
                # 整数値の検証
                mmap_size = int(pragma_value)
                # 範囲制限: 0 to 1GB（メモリマップサイズ制限）
                if 0 <= mmap_size <= 1073741824:  # 1GB = 1024^3
                    return str(mmap_size)
                else:
                    logger.warning(
                        f"mmap_size範囲外: {pragma_value}, 範囲: 0~1073741824"
                    )
                    return "268435456"  # デフォルト安全値（256MB）
            except (ValueError, TypeError):
                logger.warning(f"mmap_size無効形式: {pragma_value}")
                return "268435456"  # デフォルト安全値

        else:
            # 未知のPRAGMAは拒否
            logger.error(f"未サポートのPRAGMA: {pragma_name}")
            return None

    def get_connection_pool_stats(self) -> dict:
        """
        接続プール統計情報を取得

        Returns:
            接続プール統計情報の辞書
        """
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
            "database_type": self.config.get_database_type(),
            "database_url": self.config.database_url[:50] + "...",  # セキュリティ考慮
        }

        return stats

    def close(self):
        """接続とエンジンを閉じる"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine disposed")

    def __del__(self):
        """デストラクタで接続をクリーンアップ"""
        try:
            self.close()
        except Exception:
            pass  # デストラクタでの例外は無視