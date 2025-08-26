"""
データベース管理コアモジュール
DatabaseManagerクラスとテーブル管理機能

Issue #120: declarative_base()の定義場所の最適化対応
- データベース管理の責務を明確化
- 各種機能の統合ポイント
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy.orm import Session

from ...utils.exceptions import DatabaseError
from ...utils.logging_config import get_context_logger
from ..base import Base
from .bulk_operations import BulkOperationsManager
from .config import DatabaseConfig
from .connection import ConnectionManager
from .migration import MigrationManager
from .transaction import TransactionManager
from .utils import DatabaseUtils

logger = get_context_logger(__name__)


class DatabaseManager:
    """データベース管理クラス（改善版・依存性注入対応）"""

    def __init__(self, config: Optional[DatabaseConfig] = None, config_manager=None):
        """
        DatabaseManagerの初期化

        Args:
            config: データベース設定
            config_manager: ConfigManagerインスタンス（依存性注入）
        """
        self._config_manager = config_manager
        self.config = config or DatabaseConfig(config_manager=config_manager)
        
        # 各種管理クラスの初期化
        self.connection_manager = ConnectionManager(self.config)
        self.transaction_manager = TransactionManager(self.connection_manager)
        self.migration_manager = MigrationManager(self.connection_manager)
        self.bulk_operations_manager = BulkOperationsManager(self.transaction_manager)
        self.utils = DatabaseUtils(self.connection_manager, self.transaction_manager)
        
        # 統計情報管理
        self._connection_pool_stats = {
            "created_connections": 0,
            "closed_connections": 0,
            "active_sessions": 0,
        }

        logger.info(
            "DatabaseManager initialized",
            extra={
                "database_type": self.config.get_database_type(),
                "database_url": self.config.database_url[:50] + "...",
            },
        )

    # プロパティとして各種管理機能を公開
    @property
    def engine(self):
        """データベースエンジンを取得"""
        return self.connection_manager.engine

    @property
    def session_factory(self):
        """セッションファクトリーを取得"""
        return self.transaction_manager.session_factory

    # テーブル管理機能
    def create_tables(self):
        """
        全テーブルを作成

        Raises:
            DatabaseError: テーブル作成に失敗した場合
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("All tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(
                "Failed to create database tables",
                error_code="TABLE_CREATION_ERROR"
            ) from e

    def drop_tables(self):
        """
        全テーブルを削除

        Raises:
            DatabaseError: テーブル削除に失敗した場合
        """
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("All tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise DatabaseError(
                "Failed to drop database tables",
                error_code="TABLE_DROP_ERROR"
            ) from e

    def create_all_tables(self):
        """create_tablesのエイリアス（後方互換性）"""
        self.create_tables()

    # セッション管理（TransactionManagerへの委譲）
    def get_session(self) -> Session:
        """
        新しいセッションを取得

        Returns:
            SQLAlchemyセッション

        Raises:
            DatabaseError: セッションの取得に失敗した場合
        """
        return self.transaction_manager.get_session()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        セッションのコンテキストマネージャー

        Usage:
            with db_manager.session_scope() as session:
                # セッションを使用した処理
                pass

        Yields:
            SQLAlchemyセッション

        Raises:
            DatabaseError: セッション処理中にエラーが発生した場合
        """
        with self.transaction_manager.session_scope() as session:
            yield session

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

        Yields:
            SQLAlchemyセッション

        Raises:
            DatabaseError: トランザクション処理中にエラーが発生した場合
        """
        with self.transaction_manager.transaction_scope(
            retry_count=retry_count, retry_delay=retry_delay
        ) as session:
            yield session

    # マイグレーション機能（MigrationManagerへの委譲）
    def get_alembic_config(self, config_path: Optional[str] = None):
        """Alembic設定を取得"""
        return self.migration_manager.get_alembic_config(config_path)

    def init_alembic(self):
        """Alembicの初期化（初回マイグレーション作成）"""
        self.migration_manager.init_alembic()

    def migrate(self, message: str = "Auto migration"):
        """新しいマイグレーションを作成"""
        self.migration_manager.migrate(message)

    def upgrade(self, revision: str = "head"):
        """マイグレーションを適用"""
        self.migration_manager.upgrade(revision)

    def downgrade(self, revision: str = "-1"):
        """マイグレーションをロールバック"""
        self.migration_manager.downgrade(revision)

    def current_revision(self) -> str:
        """現在のリビジョンを取得"""
        return self.migration_manager.current_revision()

    # 一括操作機能（BulkOperationsManagerへの委譲）
    def bulk_insert(self, model_class, data_list: list, batch_size: int = 1000):
        """大量データの一括挿入"""
        self.bulk_operations_manager.bulk_insert(model_class, data_list, batch_size)

    def bulk_update(self, model_class, data_list: list, batch_size: int = 1000):
        """大量データの一括更新"""
        self.bulk_operations_manager.bulk_update(model_class, data_list, batch_size)

    def read_in_chunks(
        self,
        model_class,
        chunk_size: int = 1000,
        filters: Optional[List[Any]] = None,
        order_by: Optional[Any] = None,
    ):
        """チャンク単位でのデータ読み込み"""
        return self.bulk_operations_manager.read_in_chunks(
            model_class, chunk_size, filters, order_by
        )

    # アトミック操作
    def atomic_operation(self, operations: list, retry_count: int = 3):
        """複数のDB操作をアトミックに実行"""
        self.transaction_manager.atomic_operation(operations, retry_count)

    # ユーティリティ機能（DatabaseUtilsへの委譲）
    def execute_query(self, query: str, params: dict = None):
        """生のSQLクエリを実行"""
        return self.utils.execute_query(query, params)

    def optimize_database(self):
        """データベースの最適化を実行"""
        self.utils.optimize_database()

    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """接続プール統計情報を取得"""
        stats = self.connection_manager.get_connection_pool_stats()
        # 内部統計情報を追加
        stats.update(self._connection_pool_stats)
        return stats

    def health_check(self) -> Dict[str, Any]:
        """データベース接続のヘルスチェック"""
        return self.utils.health_check()

    def optimize_performance(self):
        """パフォーマンス最適化設定を適用"""
        self.utils.optimize_performance()

    @contextmanager
    def performance_monitor(self, operation_name: str):
        """パフォーマンス監視コンテキストマネージャー"""
        with self.utils.performance_monitor(operation_name):
            yield

    # ファクトリーメソッド
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

    # リソース管理
    def close(self):
        """DatabaseManagerとそのリソースをクリーンアップ"""
        logger.info("Closing DatabaseManager")
        
        # 各管理クラスのクリーンアップ
        if hasattr(self, 'transaction_manager'):
            self.transaction_manager.close()
        if hasattr(self, 'connection_manager'):
            self.connection_manager.close()
        
        logger.info("DatabaseManager closed successfully")

    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーのイグジット"""
        self.close()

    def __del__(self):
        """デストラクタでのクリーンアップ"""
        try:
            self.close()
        except Exception:
            pass  # デストラクタでの例外は無視

    def __repr__(self) -> str:
        """DatabaseManagerの文字列表現"""
        return (
            f"DatabaseManager("
            f"type={self.config.get_database_type()}, "
            f"url={self.config.database_url[:30]}..., "
            f"pool_size={self.config.pool_size})"
        )