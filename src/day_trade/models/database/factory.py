"""
データベースファクトリーモジュール
DatabaseManagerの作成とグローバル管理、依存性注入対応

Issue #120: declarative_base()の定義場所の最適化対応
- ファクトリー機能の責務を明確化
- 依存性注入パターンの実装
"""

from typing import Generator, Optional

from sqlalchemy.orm import Session

from ...utils.logging_config import get_context_logger
from .config import DatabaseConfig
from .manager import DatabaseManager

logger = get_context_logger(__name__)

# グローバルデータベースマネージャー（シングルトン）
_global_db_manager: Optional[DatabaseManager] = None
_default_db_manager: Optional[DatabaseManager] = None


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
            from ..performance_database import PerformanceOptimizedDatabaseManager

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


def get_global_db_manager() -> DatabaseManager:
    """
    グローバルデータベースマネージャー取得

    Returns:
        DatabaseManager: グローバルデータベースマネージャー
    """
    global _global_db_manager
    if _global_db_manager is None:
        _global_db_manager = create_database_manager()
    return _global_db_manager


def set_global_db_manager(manager: DatabaseManager):
    """
    グローバルデータベースマネージャーを設定

    Args:
        manager: データベースマネージャーインスタンス
    """
    global _global_db_manager
    _global_db_manager = manager


def get_session() -> Session:
    """
    データベースセッション取得（グローバル）

    Returns:
        Session: データベースセッション
    """
    return get_global_db_manager().get_session()


def create_test_database_manager() -> DatabaseManager:
    """
    テスト用のデータベースマネージャーを作成

    Returns:
        DatabaseManager: テスト用データベースマネージャー
    """
    config = DatabaseConfig.for_testing()
    return DatabaseManager(config=config)


def create_production_database_manager() -> DatabaseManager:
    """
    本番用のデータベースマネージャーを作成

    Returns:
        DatabaseManager: 本番用データベースマネージャー
    """
    config = DatabaseConfig.for_production()
    return DatabaseManager(config=config)


# 便利な関数
def get_db() -> Generator[Session, None, None]:
    """
    依存性注入用のデータベースセッション取得関数

    Usage:
        def some_function(db: Session = Depends(get_db)):
            # データベース操作
            pass

    Yields:
        Session: データベースセッション
    """
    db_manager = get_default_database_manager()
    with db_manager.session_scope() as session:
        yield session


def init_db():
    """データベースの初期化"""
    db_manager = get_default_database_manager()
    db_manager.create_tables()


def reset_db():
    """データベースのリセット（開発用）"""
    db_manager = get_default_database_manager()
    db_manager.drop_tables()
    db_manager.create_tables()


def init_global_database():
    """グローバルデータベース初期化"""
    db_manager = get_global_db_manager()
    db_manager.create_all_tables()
    logger.info("Global database initialized for Global Trading Engine")


def init_migration():
    """マイグレーションの初期化"""
    db_manager = get_default_database_manager()
    db_manager.init_alembic()


def create_migration(message: str = "Auto migration"):
    """新しいマイグレーションファイルを作成"""
    db_manager = get_default_database_manager()
    db_manager.migrate(message)


def upgrade_db(revision: str = "head"):
    """データベースをアップグレード"""
    db_manager = get_default_database_manager()
    db_manager.upgrade(revision)


def downgrade_db(revision: str = "-1"):
    """データベースをダウングレード"""
    db_manager = get_default_database_manager()
    db_manager.downgrade(revision)


def get_current_revision() -> str:
    """現在のリビジョンを取得"""
    db_manager = get_default_database_manager()
    return db_manager.current_revision()


def cleanup_global_managers():
    """
    グローバルマネージャーのクリーンアップ

    テスト終了時やアプリケーション終了時に呼び出し
    """
    global _global_db_manager, _default_db_manager

    if _global_db_manager:
        try:
            _global_db_manager.close()
        except Exception as e:
            logger.warning(f"Error closing global database manager: {e}")
        finally:
            _global_db_manager = None

    if _default_db_manager:
        try:
            _default_db_manager.close()
        except Exception as e:
            logger.warning(f"Error closing default database manager: {e}")
        finally:
            _default_db_manager = None

    logger.info("Global database managers cleaned up")


# 後方互換性のためのグローバルインスタンス
# ConfigManagerを使用してデータベース設定を管理
try:
    from ...config.config_manager import ConfigManager

    _config_manager = ConfigManager()
    db_manager = get_default_database_manager(_config_manager)
    logger.info("Database manager initialized with ConfigManager")
except ImportError:
    # ConfigManagerが利用できない場合はデフォルト設定で作成
    db_manager = get_default_database_manager()
    logger.info("Database manager initialized with default settings (ConfigManager not available)")
except Exception:
    # ConfigManagerの初期化に失敗した場合はデフォルト設定で作成
    db_manager = get_default_database_manager()
    logger.info("Database manager initialized with default settings (ConfigManager initialization failed)")


# アプリケーション終了時のクリーンアップ登録
import atexit
atexit.register(cleanup_global_managers)