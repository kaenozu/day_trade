"""
マイグレーション管理モジュール（統合版）
Alembic統合とデータベーススキーマ管理

Issue #120: declarative_base()の定義場所の最適化対応
- マイグレーション管理の責務を明確化、モジュール分割で300行以下に最適化
- セキュリティ強化（TOCTOU攻撃対策）
"""

from typing import Optional

from alembic.config import Config

from ...utils.logging_config import get_context_logger
from .config_security import ConfigSecurityManager
from .migration_executor import MigrationExecutor
from .connection import ConnectionManager

logger = get_context_logger(__name__)


class MigrationManager:
    """マイグレーション統合管理クラス（300行以下に最適化）"""

    def __init__(self, connection_manager: ConnectionManager):
        """
        マイグレーション管理の初期化

        Args:
            connection_manager: データベース接続管理
        """
        self.connection_manager = connection_manager
        self.config_security = ConfigSecurityManager(connection_manager.config.database_url)
        self.executor = MigrationExecutor(connection_manager)

    # 設定管理のデリゲート
    def get_alembic_config(self, config_path: Optional[str] = None) -> Config:
        """
        Alembic設定を取得（TOCTOU脆弱性対策・セキュリティ強化版）

        Args:
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Returns:
            Alembic設定オブジェクト

        Raises:
            DatabaseError: 設定ファイルの読み込みに失敗した場合
        """
        return self.config_security.get_secure_alembic_config(config_path)

    def validate_config_security(self, config_path: str) -> dict:
        """設定ファイルのセキュリティ状態を検証"""
        return self.config_security.validate_config_security(config_path)

    # マイグレーション実行のデリゲート
    def init_alembic(self, config_path: Optional[str] = None):
        """
        Alembicの初期化（初回マイグレーション作成）

        Args:
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Raises:
            DatabaseError: 初期化に失敗した場合
        """
        return self.executor.init_alembic(config_path)

    def migrate(self, message: str = "Auto migration", config_path: Optional[str] = None):
        """
        新しいマイグレーションを作成

        Args:
            message: マイグレーションメッセージ
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Raises:
            DatabaseError: マイグレーション作成に失敗した場合
        """
        return self.executor.create_migration(message, config_path)

    def upgrade(self, revision: str = "head", config_path: Optional[str] = None):
        """
        マイグレーションを適用

        Args:
            revision: 適用するリビジョン
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Raises:
            DatabaseError: マイグレーション適用に失敗した場合
        """
        return self.executor.upgrade_database(revision, config_path)

    def downgrade(self, revision: str = "-1", config_path: Optional[str] = None):
        """
        マイグレーションをロールバック

        Args:
            revision: ロールバック先のリビジョン
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Raises:
            DatabaseError: ロールバックに失敗した場合
        """
        return self.executor.downgrade_database(revision, config_path)

    def current_revision(self) -> str:
        """
        現在のリビジョンを取得

        Returns:
            現在のリビジョン文字列

        Raises:
            DatabaseError: リビジョン取得に失敗した場合
        """
        return self.executor.get_current_revision()

    def get_migration_history(self, config_path: Optional[str] = None) -> list:
        """
        マイグレーション履歴を取得

        Args:
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Returns:
            マイグレーション履歴のリスト

        Raises:
            DatabaseError: 履歴取得に失敗した場合
        """
        return self.executor.get_migration_history(config_path)

    # 統合便利メソッド
    def get_migration_status(self) -> dict:
        """マイグレーション状態を取得"""
        return self.executor.check_migration_status()

    def validate_migration_environment(self, config_path: Optional[str] = None) -> dict:
        """マイグレーション環境の検証"""
        return self.executor.validate_migration_environment(config_path)

    def execute_custom_sql_migration(self, sql_statements: list, description: str = "Custom SQL migration"):
        """カスタムSQLマイグレーションを実行"""
        return self.executor.execute_sql_migration(sql_statements, description)

    def get_manager_info(self) -> dict:
        """マイグレーション管理の情報を取得"""
        return {
            "components": {
                "config_security": "ConfigSecurityManager",
                "executor": "MigrationExecutor",
                "connection_manager": type(self.connection_manager).__name__
            },
            "capabilities": [
                "secure_config_management",
                "migration_execution",
                "environment_validation",
                "custom_sql_migration",
                "security_validation"
            ],
            "database_url": self.connection_manager.config.database_url,
            "current_status": self.get_migration_status()
        }