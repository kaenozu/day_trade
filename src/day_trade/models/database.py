"""
データベース基盤モジュール
SQLAlchemyを使用したデータベース接続とセッション管理
"""
import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# ベースクラスの作成
Base = declarative_base()

# データベースURLの設定
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./day_trade.db"
)

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
        
        # エンジンの作成
        if database_url == TEST_DATABASE_URL:
            # テスト用インメモリDBの場合
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=echo
            )
        else:
            # 通常のSQLiteファイルの場合
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                echo=echo
            )
        
        # SQLiteの外部キー制約を有効化
        if "sqlite" in database_url:
            event.listen(self.engine, "connect", self._set_sqlite_pragma)
        
        # セッションファクトリの作成
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    @staticmethod
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        """SQLiteの設定"""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
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