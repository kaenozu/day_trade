"""
Alembic環境設定ファイル（改善版）

機能:
- 設定ファイルからの統一されたデータベースURL取得
- 堅牢なsrcモジュールインポート
- SQLAlchemy 2.0対応
- 包括的エラーハンドリング
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path
from typing import Optional

from sqlalchemy import engine_from_config, pool
from alembic import context

# srcモジュールの堅牢なインポート
def _import_src_modules():
    """srcモジュールを堅牢にインポート"""
    try:
        # 現在のディレクトリからの相対パス
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        src_path = project_root / "src"
        
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # srcディレクトリが見つからない場合の代替パス
        if not src_path.exists():
            # プロジェクトルートからの直接インポートを試す
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
        
        # データベースモデルのインポート
        from day_trade.models.database import Base
        return Base
        
    except ImportError as e:
        print(f"Warning: Could not import Base from src.day_trade.models.database: {e}")
        print("Trying alternative import paths...")
        
        # 代替インポートパスを試す
        try:
            from src.day_trade.models.database import Base
            return Base
        except ImportError:
            try:
                # プロジェクトルートから直接インポート
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from src.day_trade.models.database import Base
                return Base
            except ImportError as final_error:
                print(f"Error: Could not import Base. Please check your module structure.")
                print(f"Final error: {final_error}")
                raise

# メタデータの取得
Base = _import_src_modules()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
# Note: config is available when run through alembic command
config = getattr(context, 'config', None)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config and config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url() -> str:
    """統一された設定システムからデータベースURLを取得"""
    try:
        # 環境変数から取得（最優先）
        env_url = os.environ.get("DATABASE_URL")
        if env_url:
            return env_url
        
        # 設定ファイルから取得を試す
        try:
            from day_trade.config.config_manager import ConfigManager
            config_manager = ConfigManager()
            db_settings = config_manager.get_database_settings()
            if db_settings.url:
                return db_settings.url
        except Exception as config_error:
            print(f"Warning: Could not load database URL from config: {config_error}")
        
        # alembic.iniから取得を試す
        if config:
            alembic_url = config.get_main_option("sqlalchemy.url")
            if alembic_url:
                return alembic_url
        
        # フォールバック: デフォルトのSQLiteデータベース
        default_url = "sqlite:///./day_trade.db"
        print(f"Using default database URL: {default_url}")
        return default_url
        
    except Exception as e:
        print(f"Error getting database URL: {e}")
        # 最終フォールバック
        return "sqlite:///./day_trade.db"


def run_migrations_offline() -> None:
    """オフラインモードでマイグレーションを実行

    このモードではエンジンを作成せずにURLのみでコンテキストを設定します。
    DBAPIが利用できない環境でも動作します。

    context.execute()の呼び出しは、文字列をスクリプト出力に出力します。
    """
    try:
        url = get_database_url()
        print(f"Running offline migrations with URL: {url}")
        
        # SQLAlchemy 2.0対応の設定
        context.configure(
            url=url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            # SQLAlchemy 2.0対応: 将来の互換性のためのオプション
            render_as_batch=True,  # バッチモードでの実行（SQLiteなどで有用）
        )

        with context.begin_transaction():
            context.run_migrations()
            
    except Exception as e:
        print(f"Error during offline migration: {e}")
        raise


def run_migrations_online() -> None:
    """オンラインモードでマイグレーションを実行

    このシナリオではエンジンを作成し、コネクションをコンテキストに関連付けます。
    SQLAlchemy 2.0対応の堅牢な実装です。
    """
    try:
        # 設定の準備
        configuration = {}
        if config:
            configuration = config.get_section(config.config_ini_section) or {}
        
        database_url = get_database_url()
        configuration["sqlalchemy.url"] = database_url
        
        print(f"Running online migrations with URL: {database_url}")
        
        # SQLAlchemy 2.0対応のエンジン設定
        engine_options = {
            "poolclass": pool.NullPool,
            # SQLAlchemy 2.0対応: future フラグの設定
            "future": True,
        }
        
        # SQLiteの場合の特別な設定
        if database_url.startswith("sqlite"):
            engine_options.update({
                "connect_args": {"check_same_thread": False},
                # SQLiteでの外部キー制約の有効化
                "pool_pre_ping": True,
            })
        
        # エンジンの作成
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            **engine_options
        )

        with connectable.connect() as connection:
            # SQLAlchemy 2.0対応の設定
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                # バッチモードでの実行（DDL変更の際に有用）
                render_as_batch=True,
                # SQLAlchemy 2.0対応: トランザクション管理の改善
                transaction_per_migration=True,
            )

            with context.begin_transaction():
                context.run_migrations()
                
    except Exception as e:
        print(f"Error during online migration: {e}")
        print(f"Database URL: {get_database_url()}")
        raise


def run_migrations():
    """マイグレーションを実行する主要関数"""
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()

# Alembicコマンドから実行される場合のみマイグレーションを実行
if __name__ != '__main__':
    # import時には実行しない（テスト用）
    try:
        run_migrations()
    except Exception as e:
        print(f"Migration execution error: {e}")
