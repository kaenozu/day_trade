"""
データベース設定モジュール
DatabaseConfigクラスを使用したデータベース接続設定管理

Issue #120: declarative_base()の定義場所の最適化対応
- 設定管理の責務を明確化
- パフォーマンス設定統合
"""

import os
from typing import Any, Dict, Optional

from ...utils.logging_config import get_context_logger
from ...utils.performance_config import get_performance_config

logger = get_context_logger(__name__)

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
        データベース設定の初期化

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
        """
        設定値を優先順位に従って取得

        優先度: 明示的引数 > 環境変数 > ConfigManager > デフォルト値

        Args:
            key: 設定キー
            explicit_value: 明示的に渡された値
            env_key: 環境変数キー
            default_value: デフォルト値
            type_converter: 型変換関数

        Returns:
            設定値
        """
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
        """
        テスト用の設定を作成

        Returns:
            テスト用DatabaseConfigインスタンス
        """
        return cls(
            database_url="sqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
        )

    @classmethod
    def for_production(cls) -> "DatabaseConfig":
        """
        本番用の設定を作成

        Returns:
            本番用DatabaseConfigインスタンス
        """
        return cls(
            database_url=os.environ.get("DATABASE_URL", "sqlite:///./day_trade.db"),
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_timeout=60,
            pool_recycle=1800,
        )

    def is_sqlite(self) -> bool:
        """
        SQLiteデータベースかどうかを判定

        Returns:
            SQLiteの場合True
        """
        return self.database_url.startswith("sqlite")

    def is_in_memory(self) -> bool:
        """
        インメモリデータベースかどうかを判定

        Returns:
            インメモリデータベースの場合True
        """
        return ":memory:" in self.database_url

    def get_database_type(self) -> str:
        """
        データベースタイプを取得

        Returns:
            データベースタイプ文字列
        """
        if self.is_sqlite():
            return "sqlite"
        elif "postgresql" in self.database_url:
            return "postgresql"
        elif "mysql" in self.database_url:
            return "mysql"
        else:
            return "other"

    def validate_config(self) -> bool:
        """
        設定値の妥当性を検証

        Returns:
            設定が有効な場合True

        Raises:
            ValueError: 設定値が無効な場合
        """
        if not self.database_url:
            raise ValueError("database_url is required")

        if self.pool_size < 1:
            raise ValueError("pool_size must be greater than 0")

        if self.max_overflow < 0:
            raise ValueError("max_overflow must be non-negative")

        if self.pool_timeout <= 0:
            raise ValueError("pool_timeout must be positive")

        if self.pool_recycle <= 0:
            raise ValueError("pool_recycle must be positive")

        return True

    def __repr__(self) -> str:
        """
        設定の文字列表現

        Returns:
            設定の概要文字列
        """
        return (
            f"DatabaseConfig("
            f"url={self.database_url[:20]}..., "
            f"type={self.get_database_type()}, "
            f"pool_size={self.pool_size}, "
            f"echo={self.echo})"
        )