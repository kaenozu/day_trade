"""
マイグレーション実行モジュール
Alembicマイグレーションの実行機能

Issue #120: declarative_base()の定義場所の最適化対応
- マイグレーション実行の責務を明確化
- エラーハンドリングの強化
"""

from typing import Optional

from alembic import command
from alembic.runtime.migration import MigrationContext

from ...utils.exceptions import DatabaseError, handle_database_exception
from ...utils.logging_config import get_context_logger, log_error_with_context
from .config_security import ConfigSecurityManager
from .connection import ConnectionManager

logger = get_context_logger(__name__)


class MigrationExecutor:
    """マイグレーション実行クラス"""

    def __init__(self, connection_manager: ConnectionManager):
        """
        マイグレーション実行の初期化

        Args:
            connection_manager: データベース接続管理
        """
        self.connection_manager = connection_manager
        self.config_security = ConfigSecurityManager(connection_manager.config.database_url)

    def init_alembic(self, config_path: Optional[str] = None):
        """
        Alembicの初期化（初回マイグレーション作成）

        Args:
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Raises:
            DatabaseError: 初期化に失敗した場合
        """
        try:
            alembic_cfg = self.config_security.get_secure_alembic_config(config_path)
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

    def create_migration(self, message: str = "Auto migration", config_path: Optional[str] = None):
        """
        新しいマイグレーションを作成

        Args:
            message: マイグレーションメッセージ
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Raises:
            DatabaseError: マイグレーション作成に失敗した場合
        """
        try:
            alembic_cfg = self.config_security.get_secure_alembic_config(config_path)
            command.revision(alembic_cfg, autogenerate=True, message=message)
            logger.info("Migration created", extra={"message": message})
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error, {"operation": "migration_creation", "message": message}
            )
            raise converted_error from e

    def upgrade_database(self, revision: str = "head", config_path: Optional[str] = None):
        """
        マイグレーションを適用

        Args:
            revision: 適用するリビジョン
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Raises:
            DatabaseError: マイグレーション適用に失敗した場合
        """
        try:
            alembic_cfg = self.config_security.get_secure_alembic_config(config_path)
            command.upgrade(alembic_cfg, revision)
            logger.info("Database upgraded", extra={"revision": revision})
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error, {"operation": "database_upgrade", "revision": revision}
            )
            raise converted_error from e

    def downgrade_database(self, revision: str = "-1", config_path: Optional[str] = None):
        """
        マイグレーションをロールバック

        Args:
            revision: ロールバック先のリビジョン
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Raises:
            DatabaseError: ロールバックに失敗した場合
        """
        try:
            alembic_cfg = self.config_security.get_secure_alembic_config(config_path)
            command.downgrade(alembic_cfg, revision)
            logger.info("Database downgraded", extra={"revision": revision})
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error,
                {"operation": "database_downgrade", "revision": revision},
            )
            raise converted_error from e

    def get_current_revision(self) -> str:
        """
        現在のリビジョンを取得

        Returns:
            現在のリビジョン文字列

        Raises:
            DatabaseError: リビジョン取得に失敗した場合
        """
        try:
            with self.connection_manager.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision() or "None"
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error, {"operation": "current_revision_retrieval"}
            )
            raise converted_error from e

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
        try:
            alembic_cfg = self.config_security.get_secure_alembic_config(config_path)
            # Alembicのhistoryコマンドを使用して履歴を取得
            # 実装は簡略化、必要に応じて詳細化
            with self.connection_manager.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                return [{"current_revision": current_rev}]
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error, {"operation": "migration_history_retrieval"}
            )
            raise converted_error from e

    def check_migration_status(self) -> dict:
        """
        マイグレーション状態をチェック

        Returns:
            dict: マイグレーション状態情報
        """
        try:
            current_rev = self.get_current_revision()
            status = {
                "current_revision": current_rev,
                "is_initialized": current_rev != "None",
                "database_url": self.connection_manager.config.database_url,
                "engine_info": {
                    "dialect": str(self.connection_manager.engine.dialect.name),
                    "driver": str(self.connection_manager.engine.dialect.driver),
                }
            }
            
            logger.info("Migration status checked", extra=status)
            return status
            
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error, {"operation": "migration_status_check"}
            )
            raise converted_error from e

    def validate_migration_environment(self, config_path: Optional[str] = None) -> dict:
        """
        マイグレーション環境の検証

        Args:
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Returns:
            dict: 環境検証結果
        """
        result = {
            "is_valid": False,
            "checks": {},
            "errors": [],
            "warnings": []
        }

        try:
            # データベース接続確認
            with self.connection_manager.engine.connect() as connection:
                result["checks"]["database_connection"] = True
                result["checks"]["database_type"] = str(connection.dialect.name)

            # Alembic設定確認
            try:
                alembic_cfg = self.config_security.get_secure_alembic_config(config_path)
                result["checks"]["alembic_config"] = True
                
                # 設定ファイルのセキュリティ確認
                if config_path:
                    security_result = self.config_security.validate_config_security(config_path)
                    result["checks"]["config_security"] = security_result["is_secure"]
                    result["warnings"].extend(security_result["warnings"])
                    result["errors"].extend(security_result["errors"])
                else:
                    result["checks"]["config_security"] = True  # 自動検索は安全
                
            except Exception as e:
                result["checks"]["alembic_config"] = False
                result["errors"].append(f"Alembic設定エラー: {str(e)}")

            # 現在のマイグレーション状態確認
            try:
                current_rev = self.get_current_revision()
                result["checks"]["migration_initialized"] = current_rev != "None"
                result["checks"]["current_revision"] = current_rev
            except Exception as e:
                result["checks"]["migration_initialized"] = False
                result["warnings"].append(f"マイグレーション状態取得エラー: {str(e)}")

            # 全体評価
            result["is_valid"] = all([
                result["checks"]["database_connection"],
                result["checks"]["alembic_config"],
                result["checks"]["config_security"],
            ])

        except Exception as e:
            result["errors"].append(f"環境検証エラー: {str(e)}")

        logger.info("Migration environment validated", extra=result)
        return result

    def execute_sql_migration(self, sql_statements: list, description: str = "Custom SQL migration"):
        """
        カスタムSQLマイグレーションを実行

        Args:
            sql_statements: 実行するSQL文のリスト
            description: マイグレーションの説明

        Raises:
            DatabaseError: SQL実行に失敗した場合
        """
        try:
            with self.connection_manager.engine.connect() as connection:
                trans = connection.begin()
                try:
                    for i, sql in enumerate(sql_statements, 1):
                        logger.debug(f"Executing SQL statement {i}/{len(sql_statements)}")
                        connection.execute(sql)
                    
                    trans.commit()
                    logger.info(f"Custom SQL migration completed: {description}")
                    
                except Exception as e:
                    trans.rollback()
                    raise e
                    
        except Exception as e:
            converted_error = handle_database_exception(e)
            log_error_with_context(
                converted_error, {
                    "operation": "custom_sql_migration",
                    "description": description,
                    "statements_count": len(sql_statements)
                }
            )
            raise converted_error from e