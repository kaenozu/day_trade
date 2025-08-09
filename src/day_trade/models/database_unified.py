"""
データベース統合システム（Strategy Pattern実装）

標準データベース管理と最適化版を統一し、設定ベースで選択可能なアーキテクチャ
"""

import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Type
from sqlalchemy import create_engine, event, text, Index
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..core.optimization_strategy import (
    OptimizationStrategy,
    OptimizationLevel,
    OptimizationConfig,
    optimization_strategy,
    get_optimized_implementation
)
from ..utils.logging_config import (
    get_context_logger,
    log_database_operation,
    log_error_with_context,
)
from ..utils.exceptions import DatabaseError, handle_database_exception

logger = get_context_logger(__name__)

# 基底クラスとAlembic設定の読み込み
try:
    from .base import Base
    from alembic import command
    from alembic.config import Config
    ALEMBIC_AVAILABLE = True
except ImportError:
    Base = None
    ALEMBIC_AVAILABLE = False
    logger.warning("Alembic未利用 - マイグレーション機能は無効")

try:
    from ..utils.performance_config import get_performance_config
    from ..utils.performance_analyzer import profile_performance
    PERFORMANCE_UTILS_AVAILABLE = True
except ImportError:
    PERFORMANCE_UTILS_AVAILABLE = False
    logger.info("パフォーマンス分析ユーティリティ未利用")

warnings.filterwarnings("ignore", category=DeprecationWarning)

TEST_DATABASE_URL = "sqlite:///:memory:"


@dataclass
class DatabaseConfig:
    """統合データベース設定クラス"""
    database_url: Optional[str] = None
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

    # 最適化固有設定
    enable_query_cache: bool = True
    enable_batch_processing: bool = True
    enable_connection_pooling: bool = True
    enable_index_optimization: bool = True

    # クエリキャッシュ設定
    query_cache_size: int = 1000
    query_cache_ttl_minutes: int = 30

    # バッチ処理設定
    batch_size: int = 1000
    bulk_insert_threshold: int = 100

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """環境変数からの設定読み込み"""
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
            pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DATABASE_POOL_RECYCLE", "3600")),
        )


@dataclass
class QueryPerformanceMetrics:
    """クエリパフォーマンス指標"""
    query_hash: str
    execution_time: float
    rows_affected: int
    cache_hit: bool = False
    optimization_level: str = "none"
    index_usage: List[str] = field(default_factory=list)
    execution_plan: Optional[str] = None


@dataclass
class DatabaseOperationResult:
    """データベース操作結果"""
    success: bool
    execution_time: float
    affected_rows: int
    strategy_used: str
    error_message: Optional[str] = None
    performance_metrics: Optional[QueryPerformanceMetrics] = None


class DatabaseBase(OptimizationStrategy):
    """データベース管理の基底戦略クラス"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.db_config = DatabaseConfig.from_env()
        self.engine = None
        self.SessionLocal = None
        self._setup_database()

    def _setup_database(self):
        """データベースセットアップ"""
        database_url = self.db_config.database_url or self._get_default_database_url()

        # エンジンの作成
        engine_kwargs = {'echo': self.db_config.echo}

        # SQLite用の特別設定
        if database_url.startswith('sqlite'):
            engine_kwargs.update({
                'poolclass': StaticPool,
                'connect_args': {'check_same_thread': False}
            })
        else:
            # PostgreSQL, MySQL等用の設定
            engine_kwargs.update({
                'pool_size': self.db_config.pool_size,
                'max_overflow': self.db_config.max_overflow,
                'pool_timeout': self.db_config.pool_timeout,
                'pool_recycle': self.db_config.pool_recycle,
            })

        self.engine = create_engine(database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # イベントハンドラーの設定
        self._setup_event_handlers()

        logger.info(f"データベース初期化完了: {database_url}")

    def _get_default_database_url(self) -> str:
        """デフォルトデータベースURL"""
        return "sqlite:///./day_trade.db"

    def _setup_event_handlers(self):
        """イベントハンドラーの設定"""
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            if self.engine.url.drivername == "sqlite":
                cursor = dbapi_connection.cursor()
                # SQLiteの最適化設定
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()

    def execute(self, operation: str, *args, **kwargs) -> DatabaseOperationResult:
        """データベース操作の実行"""
        start_time = time.time()

        try:
            if operation == "create_tables":
                result = self._create_tables()
            elif operation == "execute_query":
                result = self._execute_query(*args, **kwargs)
            elif operation == "bulk_insert":
                result = self._bulk_insert(*args, **kwargs)
            elif operation == "migrate":
                result = self._migrate_database(*args, **kwargs)
            else:
                raise ValueError(f"未サポートの操作: {operation}")

            execution_time = time.time() - start_time
            self.record_execution(execution_time, True)

            return DatabaseOperationResult(
                success=True,
                execution_time=execution_time,
                affected_rows=result.get("affected_rows", 0),
                strategy_used=self.get_strategy_name(),
                performance_metrics=result.get("performance_metrics")
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.record_execution(execution_time, False)
            error_msg = str(e)
            logger.error(f"データベース操作エラー: {error_msg}")

            return DatabaseOperationResult(
                success=False,
                execution_time=execution_time,
                affected_rows=0,
                strategy_used=self.get_strategy_name(),
                error_message=error_msg
            )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """セッション管理コンテキスト"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"データベースセッションエラー: {e}")
            raise
        finally:
            session.close()

    def _create_tables(self) -> Dict[str, Any]:
        """テーブル作成"""
        if Base is None:
            raise DatabaseError("Base クラスが利用できません")

        Base.metadata.create_all(bind=self.engine)

        # 作成されたテーブル数を取得
        inspector = self.engine.dialect.get_inspector(self.engine)
        table_count = len(inspector.get_table_names())

        return {"affected_rows": table_count}

    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """クエリ実行"""
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            if result.returns_rows:
                rows = result.fetchall()
                return {"affected_rows": len(rows), "data": rows}
            else:
                return {"affected_rows": result.rowcount}

    def _bulk_insert(self, table_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """バルクインサート"""
        if not data:
            return {"affected_rows": 0}

        with self.get_session() as session:
            # テーブルクラスの動的取得
            table_class = self._get_table_class(table_name)
            if table_class:
                objects = [table_class(**row) for row in data]
                session.bulk_save_objects(objects)
            else:
                # 生のSQLでのバルクインサート
                columns = list(data[0].keys())
                placeholders = ", ".join([f":{col}" for col in columns])
                query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                session.execute(text(query), data)

            return {"affected_rows": len(data)}

    def _migrate_database(self, revision: str = "head") -> Dict[str, Any]:
        """データベースマイグレーション"""
        if not ALEMBIC_AVAILABLE:
            logger.warning("Alembic未利用 - マイグレーションスキップ")
            return {"affected_rows": 0}

        try:
            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, revision)
            return {"affected_rows": 1}  # マイグレーション成功を示す
        except Exception as e:
            raise DatabaseError(f"マイグレーション失敗: {e}")

    def _get_table_class(self, table_name: str) -> Optional[Type]:
        """テーブルクラスの動的取得"""
        if Base is None:
            return None

        # Base.registryから該当テーブルを検索
        for mapper in Base.registry._class_registry.values():
            if hasattr(mapper, 'mapped_class'):
                cls = mapper.mapped_class
                if hasattr(cls, '__tablename__') and cls.__tablename__ == table_name:
                    return cls

        return None


@optimization_strategy("database", OptimizationLevel.STANDARD)
class StandardDatabase(DatabaseBase):
    """標準データベース管理実装"""

    def get_strategy_name(self) -> str:
        return "標準データベース"


@optimization_strategy("database", OptimizationLevel.OPTIMIZED)
class OptimizedDatabase(DatabaseBase):
    """最適化データベース管理実装"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)

        # 最適化機能の初期化
        if PERFORMANCE_UTILS_AVAILABLE:
            self.performance_config = get_performance_config()

        # クエリキャッシュ
        self.query_cache = {}
        self.query_cache_ttl = {}

        # バッチ処理キュー
        self.batch_queue = {}

        logger.info("最適化データベース初期化完了")

    def get_strategy_name(self) -> str:
        return "最適化データベース"

    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """キャッシュ機能付きクエリ実行"""
        if not self.db_config.enable_query_cache:
            return super()._execute_query(query, params)

        # クエリキャッシュキーの生成
        cache_key = self._generate_cache_key(query, params)

        # キャッシュからの取得を試行
        if cache_key in self.query_cache:
            cache_time = self.query_cache_ttl.get(cache_key, 0)
            if time.time() - cache_time < self.db_config.query_cache_ttl_minutes * 60:
                logger.debug("クエリキャッシュヒット")
                result = self.query_cache[cache_key].copy()
                if "performance_metrics" in result:
                    result["performance_metrics"].cache_hit = True
                return result

        # キャッシュミス時は通常実行
        start_time = time.time()
        result = super()._execute_query(query, params)
        execution_time = time.time() - start_time

        # パフォーマンス指標の記録
        performance_metrics = QueryPerformanceMetrics(
            query_hash=cache_key,
            execution_time=execution_time,
            rows_affected=result.get("affected_rows", 0),
            cache_hit=False,
            optimization_level="optimized"
        )

        result["performance_metrics"] = performance_metrics

        # 結果をキャッシュに保存
        if len(self.query_cache) < self.db_config.query_cache_size:
            self.query_cache[cache_key] = result.copy()
            self.query_cache_ttl[cache_key] = time.time()

        return result

    def _bulk_insert(self, table_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """最適化バルクインサート"""
        if not data:
            return {"affected_rows": 0}

        # バッチ処理の有効性チェック
        if (self.db_config.enable_batch_processing and
            len(data) >= self.db_config.bulk_insert_threshold):
            return self._optimized_bulk_insert(table_name, data)
        else:
            return super()._bulk_insert(table_name, data)

    def _optimized_bulk_insert(self, table_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """最適化されたバルクインサート"""
        total_inserted = 0
        batch_size = self.db_config.batch_size

        logger.info(f"最適化バルクインサート開始: {len(data)}件 -> {batch_size}件ずつ")

        with self.get_session() as session:
            # トランザクション最適化
            session.execute(text("PRAGMA synchronous=OFF"))  # SQLite用

            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]

                # テーブルクラスの使用を試行
                table_class = self._get_table_class(table_name)
                if table_class:
                    objects = [table_class(**row) for row in batch]
                    session.bulk_save_objects(objects)
                else:
                    # Core APIを使用した高速インサート
                    if Base and hasattr(Base.metadata.tables, table_name):
                        table = Base.metadata.tables[table_name]
                        session.execute(table.insert(), batch)
                    else:
                        # フォールバック: 生のSQL
                        columns = list(batch[0].keys())
                        placeholders = ", ".join([f":{col}" for col in columns])
                        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                        session.execute(text(query), batch)

                total_inserted += len(batch)

                # 定期的なコミット
                if i % (batch_size * 10) == 0:
                    session.commit()

            # 設定を復元
            session.execute(text("PRAGMA synchronous=NORMAL"))

        logger.info(f"最適化バルクインサート完了: {total_inserted}件")
        return {"affected_rows": total_inserted}

    def _generate_cache_key(self, query: str, params: Optional[Dict[str, Any]]) -> str:
        """キャッシュキー生成"""
        import hashlib

        combined = query
        if params:
            sorted_params = sorted(params.items())
            combined += str(sorted_params)

        return hashlib.md5(combined.encode()).hexdigest()

    def optimize_indexes(self, table_name: str, columns: List[str]) -> None:
        """インデックス最適化"""
        if not self.db_config.enable_index_optimization:
            return

        with self.get_session() as session:
            # 既存インデックスの確認
            inspector = self.engine.dialect.get_inspector(self.engine)
            existing_indexes = inspector.get_indexes(table_name)

            for column in columns:
                # インデックス名
                index_name = f"idx_{table_name}_{column}"

                # 既存インデックスをチェック
                index_exists = any(
                    index_name == idx['name'] or column in idx['column_names']
                    for idx in existing_indexes
                )

                if not index_exists:
                    # インデックス作成
                    session.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column})"))
                    logger.info(f"インデックス作成: {index_name}")

    def clear_query_cache(self) -> None:
        """クエリキャッシュのクリア"""
        self.query_cache.clear()
        self.query_cache_ttl.clear()
        logger.info("クエリキャッシュクリア完了")

    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計の取得"""
        return {
            "cache_size": len(self.query_cache),
            "max_cache_size": self.db_config.query_cache_size,
            "cache_hit_keys": list(self.query_cache.keys()),
        }


# 統合インターフェース
class DatabaseManager:
    """データベース統合マネージャー"""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig.from_env()
        self._strategy = None

    def get_strategy(self) -> OptimizationStrategy:
        """現在の戦略を取得"""
        if self._strategy is None:
            self._strategy = get_optimized_implementation("database", self.config)
        return self._strategy

    def create_tables(self) -> DatabaseOperationResult:
        """テーブル作成"""
        strategy = self.get_strategy()
        return strategy.execute("create_tables")

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseOperationResult:
        """クエリ実行"""
        strategy = self.get_strategy()
        return strategy.execute("execute_query", query, params)

    def bulk_insert(self, table_name: str, data: List[Dict[str, Any]]) -> DatabaseOperationResult:
        """バルクインサート"""
        strategy = self.get_strategy()
        return strategy.execute("bulk_insert", table_name, data)

    def migrate_database(self, revision: str = "head") -> DatabaseOperationResult:
        """データベースマイグレーション"""
        strategy = self.get_strategy()
        return strategy.execute("migrate", revision)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """セッション取得"""
        strategy = self.get_strategy()
        with strategy.get_session() as session:
            yield session

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要"""
        if self._strategy:
            return self._strategy.get_performance_metrics()
        return {}


# 便利関数
def get_database_manager(config: Optional[OptimizationConfig] = None) -> DatabaseManager:
    """データベースマネージャーのファクトリ関数"""
    return DatabaseManager(config)
