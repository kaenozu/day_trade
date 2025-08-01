"""
データベースモデルの基底クラス
"""

from datetime import datetime
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.ext.declarative import declared_attr

from .database import Base


class TimestampMixin:
    """作成日時・更新日時を持つMixinクラス"""

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )


class BaseModel(Base, TimestampMixin):
    """全モデルの基底クラス"""

    __abstract__ = True

    id = Column(Integer, primary_key=True, index=True)

    @declared_attr
    def __tablename__(cls):
        """テーブル名を自動生成（クラス名を小文字に）"""
        return cls.__name__.lower()

    def to_dict(self):
        """モデルを辞書に変換"""
        return {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }

    def __repr__(self):
        """オブジェクトの文字列表現"""
        class_name = self.__class__.__name__
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"{class_name}({attrs})"
