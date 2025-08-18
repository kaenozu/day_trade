"""
Alembic環境設定

Issue #3: 本番データベース設定の実装
PostgreSQL対応マイグレーション環境
"""

import asyncio
import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import AsyncEngine
from alembic import context

# DayTradingシステムのモデルインポート
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from day_trade.models.base import Base

# ロギング設定
from day_trade.core.logging.unified_logging_system import get_logger
logger = get_logger(__name__)

# ProductionDatabaseManagerは一時的にコメントアウト
# from day_trade.infrastructure.database.production_database_manager import ProductionDatabaseManager

# 基本的なモデルのみインポート（テスト用）
try:
    # SQLAlchemy 2.0互換のサンプルモデルを作成
    from sqlalchemy import String, DECIMAL, DateTime, Boolean, Integer
    from sqlalchemy.orm import Mapped, mapped_column
    from datetime import datetime
    
    class SampleStock(Base):
        __tablename__ = 'sample_stocks'
        
        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        code: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
        name: Mapped[str] = mapped_column(String(100), nullable=False)
        market: Mapped[str] = mapped_column(String(20), nullable=True)
        created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
        updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    class SamplePriceData(Base):
        __tablename__ = 'sample_price_data'
        
        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        stock_code: Mapped[str] = mapped_column(String(10), nullable=False)
        price: Mapped[float] = mapped_column(DECIMAL(10, 2), nullable=False)
        volume: Mapped[int] = mapped_column(Integer, nullable=True)
        timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
        
    logger.info("サンプルモデル作成完了")
except Exception as e:
    logger.warning(f"モデル作成失敗: {e}")
    # Baseのみで進行

# Alembic設定オブジェクト
config = context.config

# ログ設定
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# メタデータの設定
target_metadata = Base.metadata


def get_database_url():
    """データベースURL取得"""
    # 環境変数からURL取得を試行
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        return db_url
    
    # フォールバック: 設定ファイルの値を使用
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """オフラインモードでマイグレーション実行"""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """オンラインモードでマイグレーション実行"""
    # エンジン設定を更新
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_database_url()
    
    # データベースURLからDBタイプを判定
    db_url = get_database_url()
    connect_args = {}
    
    if db_url.startswith('postgresql'):
        # PostgreSQL固有設定
        connect_args = {
            "options": "-c timezone=UTC",
            "application_name": "DayTrading_Migration"
        }
    elif db_url.startswith('sqlite'):
        # SQLite設定
        connect_args = {}
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_schemas=True,
            # PostgreSQL固有のオプション
            render_as_batch=False,
            transaction_per_migration=True
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()