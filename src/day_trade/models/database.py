"""
データベース基盤モジュール
SQLAlchemyを使用したデータベース接続とセッション管理
"""

import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from alembic.config import Config
from alembic import command

# ベースクラスの作成
Base = declarative_base()

# データベースURLの設定
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./day_trade.db")

# テスト用のインメモリデータベースURL
TEST_DATABASE_URL = "sqlite:///:memory:"


class DatabaseManager:
    """データベース管理クラス"""

    def __init__(self, database_url: str = DATABASE_URL, echo: bool = False):
        """
        Args:
            database_url: データベースURL
            echo: SQLログ出力フラグ
        """
        self.database_url = database_url
        self.echo = echo

        # エンジンの作成（コネクションプーリング最適化）
        if database_url == TEST_DATABASE_URL:
            # テスト用インメモリDBの場合
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=echo,
                pool_pre_ping=True,  # 接続の健全性チェック
            )
        else:
            # 通常のSQLiteファイルの場合（プーリング最適化）
            self.engine = create_engine(
                database_url,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20,  # タイムアウト設定
                },
                echo=echo,
                pool_size=10,  # コネクションプール サイズ
                max_overflow=20,  # 最大オーバーフロー接続数
                pool_pre_ping=True,  # 接続の健全性チェック
                pool_recycle=3600,  # 1時間でコネクションをリサイクル
            )

        # SQLiteの外部キー制約を有効化
        if "sqlite" in database_url:
            event.listen(self.engine, "connect", self._set_sqlite_pragma)

        # セッションファクトリの作成
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    @staticmethod
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        """SQLiteの設定（パフォーマンス最適化）"""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")  # WALモードでパフォーマンス向上
        cursor.execute("PRAGMA synchronous=NORMAL")  # 同期レベルを調整
        cursor.execute("PRAGMA cache_size=10000")  # キャッシュサイズを増加
        cursor.execute("PRAGMA temp_store=memory")  # 一時テーブルをメモリに保存
        cursor.execute("PRAGMA mmap_size=268435456")  # メモリマップサイズ (256MB)
        cursor.close()

    def create_tables(self):
        """全テーブルを作成"""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """全テーブルを削除"""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """新しいセッションを取得"""
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        セッションのコンテキストマネージャー

        Usage:
            with db_manager.session_scope() as session:
                # セッションを使用した処理
                pass
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_alembic_config(self) -> Config:
        """Alembic設定を取得"""
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)
        return alembic_cfg

    def init_alembic(self):
        """Alembicの初期化（初回マイグレーション作成）"""
        alembic_cfg = self.get_alembic_config()
        command.revision(alembic_cfg, autogenerate=True, message="Initial migration")

    def migrate(self, message: str = "Auto migration"):
        """新しいマイグレーションを作成"""
        alembic_cfg = self.get_alembic_config()
        command.revision(alembic_cfg, autogenerate=True, message=message)

    def upgrade(self, revision: str = "head"):
        """マイグレーションを適用"""
        alembic_cfg = self.get_alembic_config()
        command.upgrade(alembic_cfg, revision)

    def downgrade(self, revision: str = "-1"):
        """マイグレーションをロールバック"""
        alembic_cfg = self.get_alembic_config()
        command.downgrade(alembic_cfg, revision)

    def current_revision(self) -> str:
        """現在のリビジョンを取得"""
        from alembic.runtime.migration import MigrationContext

        with self.engine.connect() as connection:
            context = MigrationContext.configure(connection)
            return context.get_current_revision() or "None"

    def bulk_insert(self, model_class, data_list: list, batch_size: int = 1000):
        """
        大量データの一括挿入

        Args:
            model_class: 挿入するモデルクラス
            data_list: 挿入するデータのリスト（辞書形式）
            batch_size: バッチサイズ
        """
        if not data_list:
            return

        with self.session_scope() as session:
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i : i + batch_size]
                session.bulk_insert_mappings(model_class, batch)
                session.flush()

    def bulk_update(self, model_class, data_list: list, batch_size: int = 1000):
        """
        大量データの一括更新

        Args:
            model_class: 更新するモデルクラス
            data_list: 更新するデータのリスト（辞書形式、idが必要）
            batch_size: バッチサイズ
        """
        if not data_list:
            return

        with self.session_scope() as session:
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i : i + batch_size]
                session.bulk_update_mappings(model_class, batch)
                session.flush()

    def execute_query(self, query: str, params: dict = None):
        """
        生のSQLクエリを実行（最適化されたクエリ用）

        Args:
            query: 実行するSQLクエリ
            params: クエリパラメータ

        Returns:
            クエリ結果
        """
        with self.engine.connect() as connection:
            result = connection.execute(query, params or {})
            return result.fetchall()

    def optimize_database(self):
        """
        データベースの最適化を実行
        """
        if "sqlite" in self.database_url:
            with self.engine.connect() as connection:
                # VACUUM操作でデータベースファイルを最適化
                connection.execute("VACUUM")
                # ANALYZE操作で統計情報を更新
                connection.execute("ANALYZE")


# デフォルトのデータベースマネージャー
db_manager = DatabaseManager()


# 便利な関数
def get_db() -> Generator[Session, None, None]:
    """
    依存性注入用のデータベースセッション取得関数

    Usage:
        def some_function(db: Session = Depends(get_db)):
            # データベース操作
            pass
    """
    with db_manager.session_scope() as session:
        yield session


def init_db():
    """データベースの初期化"""
    db_manager.create_tables()


def reset_db():
    """データベースのリセット（開発用）"""
    db_manager.drop_tables()
    db_manager.create_tables()


def init_migration():
    """マイグレーションの初期化"""
    db_manager.init_alembic()


def create_migration(message: str = "Auto migration"):
    """新しいマイグレーションファイルを作成"""
    db_manager.migrate(message)


def upgrade_db(revision: str = "head"):
    """データベースをアップグレード"""
    db_manager.upgrade(revision)


def downgrade_db(revision: str = "-1"):
    """データベースをダウングレード"""
    db_manager.downgrade(revision)


def get_current_revision() -> str:
    """現在のリビジョンを取得"""
    return db_manager.current_revision()
