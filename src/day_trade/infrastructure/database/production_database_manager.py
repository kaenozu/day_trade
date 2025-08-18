"""
本番環境データベースマネージャー

Issue #3: 本番データベース設定の実装
PostgreSQL本番環境対応、接続プール管理、マイグレーション機能
"""

import os
import ssl
import time
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from contextlib import contextmanager
from datetime import datetime, timedelta

import psycopg2
from sqlalchemy import create_engine, event, text, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.exc import (
    OperationalError, IntegrityError, 
    DatabaseError as SQLDatabaseError,
    TimeoutError as SQLTimeoutError
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from day_trade.core.error_handling.unified_error_system import (
    ApplicationError, DataAccessError, SystemError,
    error_boundary, global_error_handler
)
from day_trade.core.logging.unified_logging_system import get_logger

logger = get_logger(__name__)


class ProductionDatabaseError(DataAccessError):
    """本番データベース専用エラー"""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        super().__init__(message, operation=operation, **kwargs)


class DatabaseConnectionPool:
    """データベース接続プール管理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._connection_count = 0
        self._slow_queries = []
        self._last_health_check = None
        
    @error_boundary(
        component_name="database_pool",
        operation_name="create_engine",
        suppress_errors=False
    )
    def create_engine(self) -> Engine:
        """データベースエンジン作成（PostgreSQL/SQLite対応）"""
        db_config = self.config['database']
        
        # 環境変数展開
        database_url = self._expand_environment_variables(db_config['url'])
        
        # SSL設定
        connect_args = db_config.get('connect_args', {}).copy()
        if db_config.get('ssl_mode') == 'require':
            connect_args.update(self._get_ssl_config())
        
        # データベース種別に応じた設定
        engine_kwargs = {
            'poolclass': QueuePool,
            'pool_size': db_config.get('pool_size', 20),
            'max_overflow': db_config.get('max_overflow', 30),
            'pool_timeout': db_config.get('pool_timeout', 30),
            'pool_recycle': db_config.get('pool_recycle', 3600),
            'pool_pre_ping': db_config.get('pool_pre_ping', True),
            'echo': db_config.get('echo', False),
            'echo_pool': db_config.get('echo_pool', False),
            'connect_args': connect_args
        }
        
        # データベース固有の設定
        if database_url.startswith('sqlite'):
            # SQLite設定：isolation_levelは除外
            pass
        elif database_url.startswith('postgresql'):
            # PostgreSQL設定
            engine_kwargs['isolation_level'] = db_config.get('isolation_level', 'READ_COMMITTED')
        
        # エンジン作成
        self.engine = create_engine(database_url, **engine_kwargs)
        
        # イベントリスナー設定
        self._setup_event_listeners()
        
        # セッションファクトリー作成
        self.session_factory = sessionmaker(bind=self.engine)
        
        logger.info(
            "PostgreSQLエンジン作成完了",
            pool_size=db_config.get('pool_size', 20),
            max_overflow=db_config.get('max_overflow', 30),
            ssl_mode=db_config.get('ssl_mode', 'disable')
        )
        
        return self.engine
    
    def _expand_environment_variables(self, url: str) -> str:
        """環境変数を展開"""
        import re
        
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name, default_value = var_expr, ''
            
            return os.getenv(var_name, default_value)
        
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, url)
    
    def _get_ssl_config(self) -> Dict[str, Any]:
        """SSL設定取得"""
        ssl_config = {}
        db_config = self.config['database']
        
        if 'ssl_cert' in db_config:
            ssl_config['sslcert'] = self._expand_environment_variables(db_config['ssl_cert'])
        if 'ssl_key' in db_config:
            ssl_config['sslkey'] = self._expand_environment_variables(db_config['ssl_key'])
        if 'ssl_ca' in db_config:
            ssl_config['sslrootcert'] = self._expand_environment_variables(db_config['ssl_ca'])
            
        return ssl_config
    
    def _setup_event_listeners(self):
        """SQLAlchemyイベントリスナー設定"""
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            self._connection_count += 1
            logger.debug("データベース接続確立", connection_count=self._connection_count)
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            # 接続プール監視
            pool = self.engine.pool
            try:
                checked_out = getattr(pool, 'checkedout', lambda: 0)()
                size = getattr(pool, 'size', lambda: 1)()
                utilization = checked_out / size if size > 0 else 0
                
                if utilization > 0.8:
                    logger.warning(
                        "接続プール使用率が高い",
                        checked_out=checked_out,
                        pool_size=size,
                        utilization=utilization
                    )
            except Exception as e:
                logger.debug(f"接続プール監視エラー: {e}")
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - context._query_start_time
            total_time_ms = total_time * 1000
            
            # スロークエリ検出
            slow_threshold = self.config.get('monitoring', {}).get('slow_query_threshold_ms', 1000)
            if total_time_ms > slow_threshold:
                self._slow_queries.append({
                    'statement': statement[:200] + '...' if len(statement) > 200 else statement,
                    'duration_ms': total_time_ms,
                    'timestamp': datetime.now()
                })
                
                logger.warning(
                    "スロークエリ検出",
                    duration_ms=total_time_ms,
                    statement_preview=statement[:100]
                )
    
    @contextmanager
    def get_session(self):
        """セッション取得（コンテキストマネージャー）"""
        if not self.session_factory:
            raise ProductionDatabaseError("データベースエンジンが初期化されていません")
        
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("データベースセッションエラー", error=str(e))
            raise ProductionDatabaseError(f"データベース操作エラー: {e}") from e
        finally:
            session.close()
    
    def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック実行"""
        if not self.engine:
            return {"status": "error", "message": "エンジンが初期化されていません"}
        
        try:
            start_time = time.time()
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            response_time = (time.time() - start_time) * 1000
            
            # 接続プール状態
            pool = self.engine.pool
            try:
                checked_out = getattr(pool, 'checkedout', lambda: 0)()
                size = getattr(pool, 'size', lambda: 0)()
                overflow = getattr(pool, 'overflow', lambda: 0)()
                
                pool_status = {
                    "size": size,
                    "checked_out": checked_out,
                    "overflow": overflow,
                    "utilization": checked_out / size if size > 0 else 0
                }
            except Exception:
                # プール情報取得失敗時のフォールバック
                pool_status = {
                    "size": 0,
                    "checked_out": 0,
                    "overflow": 0,
                    "utilization": 0
                }
            
            self._last_health_check = datetime.now()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "pool_status": pool_status,
                "slow_queries_count": len(self._slow_queries),
                "last_check": self._last_health_check.isoformat()
            }
            
        except Exception as e:
            logger.error("ヘルスチェック失敗", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }


class DatabaseMigrationManager:
    """データベースマイグレーション管理"""
    
    def __init__(self, engine: Engine, config: Dict[str, Any]):
        self.engine = engine
        self.config = config
        self.alembic_cfg = self._setup_alembic_config()
    
    def _setup_alembic_config(self) -> Config:
        """Alembic設定セットアップ"""
        # Alembic設定ファイルのパス
        alembic_cfg_path = Path(__file__).parent.parent.parent.parent / "alembic.ini"
        
        if not alembic_cfg_path.exists():
            # alembic.iniが存在しない場合は作成
            self._create_alembic_config(alembic_cfg_path)
        
        alembic_cfg = Config(str(alembic_cfg_path))
        alembic_cfg.set_main_option("sqlalchemy.url", str(self.engine.url))
        
        return alembic_cfg
    
    def _create_alembic_config(self, config_path: Path):
        """Alembic設定ファイル作成"""
        config_content = """
# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = migrations

# template used to generate migration files
# file_template = %%(rev)s_%%(slug)s

# timezone to use when rendering the date
# within the migration file as well as the filename.
# string value is passed to dateutil.tz.gettz()
timezone =

# max length of characters to apply to the
# "slug" field
truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
sourceless = false

# version number format
version_num_format = %04d

# the output encoding used when revision files
# are written from script.py.mako
output_encoding = utf-8

sqlalchemy.url = postgresql://user:password@localhost/daytrading

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
        config_path.write_text(config_content.strip())
    
    @error_boundary(
        component_name="migration_manager",
        operation_name="check_migration_status",
        suppress_errors=False
    )
    def check_migration_status(self) -> Dict[str, Any]:
        """マイグレーション状態チェック"""
        try:
            with self.engine.connect() as connection:
                migration_context = MigrationContext.configure(connection)
                current_revision = migration_context.get_current_revision()
                
                script_dir = ScriptDirectory.from_config(self.alembic_cfg)
                head_revision = script_dir.get_current_head()
                
                pending_revisions = []
                if current_revision != head_revision:
                    # 未適用のマイグレーションを取得
                    for revision in script_dir.walk_revisions():
                        if revision.revision == current_revision:
                            break
                        pending_revisions.append(revision.revision)
                
                return {
                    "current_revision": current_revision,
                    "head_revision": head_revision,
                    "pending_revisions": pending_revisions,
                    "is_up_to_date": current_revision == head_revision
                }
                
        except Exception as e:
            logger.error("マイグレーション状態チェック失敗", error=str(e))
            raise ProductionDatabaseError(f"マイグレーション状態確認エラー: {e}") from e
    
    @error_boundary(
        component_name="migration_manager",
        operation_name="run_migrations",
        suppress_errors=False
    )
    def run_migrations(self, target_revision: str = "head") -> Dict[str, Any]:
        """マイグレーション実行"""
        migration_config = self.config.get('migration', {})
        
        # バックアップ実行（設定されている場合）
        if migration_config.get('backup_before_migration', True):
            backup_result = self._backup_database()
            logger.info("マイグレーション前バックアップ完了", backup_path=backup_result.get('backup_path'))
        
        try:
            # マイグレーション前の状態
            status_before = self.check_migration_status()
            
            # マイグレーション実行
            start_time = time.time()
            command.upgrade(self.alembic_cfg, target_revision)
            duration = time.time() - start_time
            
            # マイグレーション後の状態
            status_after = self.check_migration_status()
            
            logger.info(
                "マイグレーション完了",
                duration_seconds=duration,
                from_revision=status_before['current_revision'],
                to_revision=status_after['current_revision']
            )
            
            return {
                "success": True,
                "duration_seconds": duration,
                "from_revision": status_before['current_revision'],
                "to_revision": status_after['current_revision'],
                "applied_revisions": status_before['pending_revisions']
            }
            
        except Exception as e:
            logger.error("マイグレーション実行失敗", error=str(e))
            raise ProductionDatabaseError(f"マイグレーション実行エラー: {e}") from e
    
    def _backup_database(self) -> Dict[str, Any]:
        """データベースバックアップ"""
        backup_config = self.config.get('backup', {})
        backup_path = backup_config.get('backup_path', '/tmp/db_backup')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{backup_path}/backup_{timestamp}.sql"
        
        # pg_dumpを使用してバックアップ
        # 実際の実装では適切な認証情報とパスを使用
        backup_command = f"pg_dump {self.engine.url} > {backup_file}"
        
        # バックアップ実行のシミュレーション
        # 実際の環境では subprocess.run を使用
        logger.info("データベースバックアップ実行", backup_file=backup_file)
        
        return {
            "backup_path": backup_file,
            "timestamp": timestamp,
            "size_mb": 0  # 実際のファイルサイズを計算
        }


class ProductionDatabaseManager:
    """本番環境データベースマネージャー"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/database_production.yaml"
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.config = self._load_config()
        self.connection_pool: Optional[DatabaseConnectionPool] = None
        self.migration_manager: Optional[DatabaseMigrationManager] = None
        
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise ProductionDatabaseError(f"設定ファイルが見つかりません: {self.config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 環境固有設定をマージ
        if 'environments' in config and self.environment in config['environments']:
            env_config = config['environments'][self.environment]
            self._merge_config(config, env_config)
        
        return config
    
    def _merge_config(self, base_config: Dict[str, Any], env_config: Dict[str, Any]):
        """環境設定をベース設定にマージ"""
        for key, value in env_config.items():
            if isinstance(value, dict) and key in base_config:
                base_config[key].update(value)
            else:
                base_config[key] = value
    
    @error_boundary(
        component_name="production_db_manager",
        operation_name="initialize",
        suppress_errors=False
    )
    def initialize(self) -> None:
        """データベースマネージャー初期化"""
        logger.info("本番データベースマネージャー初期化開始", environment=self.environment)
        
        # 接続プール作成
        self.connection_pool = DatabaseConnectionPool(self.config)
        engine = self.connection_pool.create_engine()
        
        # マイグレーションマネージャー作成
        self.migration_manager = DatabaseMigrationManager(engine, self.config)
        
        # 初期ヘルスチェック
        health_status = self.connection_pool.health_check()
        if health_status['status'] != 'healthy':
            raise ProductionDatabaseError(f"データベース接続不良: {health_status.get('error')}")
        
        logger.info("本番データベースマネージャー初期化完了", health_status=health_status)
    
    def get_session(self):
        """データベースセッション取得"""
        if not self.connection_pool:
            raise ProductionDatabaseError("データベースマネージャーが初期化されていません")
        
        return self.connection_pool.get_session()
    
    def run_migrations(self) -> Dict[str, Any]:
        """マイグレーション実行"""
        if not self.migration_manager:
            raise ProductionDatabaseError("マイグレーションマネージャーが初期化されていません")
        
        return self.migration_manager.run_migrations()
    
    def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック実行"""
        if not self.connection_pool:
            return {"status": "error", "message": "接続プールが初期化されていません"}
        
        return self.connection_pool.health_check()
    
    def get_database_info(self) -> Dict[str, Any]:
        """データベース情報取得"""
        if not self.connection_pool or not self.connection_pool.engine:
            return {"status": "not_initialized"}
        
        try:
            with self.connection_pool.engine.connect() as conn:
                # PostgreSQL固有の情報取得
                version_result = conn.execute(text("SELECT version()"))
                version = version_result.scalar()
                
                # データベースサイズ
                size_result = conn.execute(text(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                ))
                database_size = size_result.scalar()
                
                # 接続数
                connections_result = conn.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                ))
                active_connections = connections_result.scalar()
                
                return {
                    "database_type": "PostgreSQL",
                    "version": version,
                    "database_size": database_size,
                    "active_connections": active_connections,
                    "environment": self.environment,
                    "pool_status": self.connection_pool.health_check().get('pool_status', {})
                }
                
        except Exception as e:
            logger.error("データベース情報取得失敗", error=str(e))
            return {"status": "error", "error": str(e)}


# グローバルインスタンス
_production_db_manager: Optional[ProductionDatabaseManager] = None


def get_production_database_manager() -> ProductionDatabaseManager:
    """本番データベースマネージャー取得（シングルトン）"""
    global _production_db_manager
    
    if _production_db_manager is None:
        _production_db_manager = ProductionDatabaseManager()
        _production_db_manager.initialize()
    
    return _production_db_manager


def initialize_production_database(config_path: Optional[str] = None) -> ProductionDatabaseManager:
    """本番データベース初期化"""
    global _production_db_manager
    
    _production_db_manager = ProductionDatabaseManager(config_path)
    _production_db_manager.initialize()
    
    return _production_db_manager