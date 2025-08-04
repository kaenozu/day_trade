"""
データベースモデルの基底クラス（SQLAlchemy 2.0対応）
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, func
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class TimestampMixin:
    """作成日時・更新日時を持つMixinクラス（SQLAlchemy 2.0対応）"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        nullable=False,
        doc="レコード作成日時"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="レコード更新日時"
    )


class BaseModel(Base, TimestampMixin):
    """全モデルの基底クラス（SQLAlchemy 2.0対応）"""

    __abstract__ = True

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
        doc="主キー（自動採番）"
    )

    @declared_attr
    def __tablename__(cls) -> str:
        """テーブル名を自動生成（クラス名を小文字に）"""
        return cls.__name__.lower()

    def to_dict(self) -> dict:
        """モデルを辞書に変換（SQLAlchemy 2.0対応）"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }

    def to_dict_safe(self) -> dict:
        """安全な辞書変換（リレーション除外、None値対応）"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name, None)
            if value is not None:
                # datetime型の場合はISO形式文字列に変換
                if isinstance(value, datetime):
                    result[column.name] = value.isoformat()
                else:
                    result[column.name] = value
        return result

    def update_from_dict(self, data: dict, exclude_keys: Optional[set] = None) -> None:
        """辞書からモデルを更新"""
        exclude_keys = exclude_keys or {'id', 'created_at'}

        for key, value in data.items():
            if key not in exclude_keys and hasattr(self, key):
                # カラムが存在する場合のみ更新
                if hasattr(self.__table__.columns, key):
                    setattr(self, key, value)

    def __repr__(self) -> str:
        """オブジェクトの文字列表現（改善版）"""
        class_name = self.__class__.__name__

        # 主要な属性のみ表示（長すぎる場合を考慮）
        key_attrs = []
        if hasattr(self, 'id') and self.id is not None:
            key_attrs.append(f"id={self.id}")
        if hasattr(self, 'code') and getattr(self, 'code', None):
            key_attrs.append(f"code={getattr(self, 'code')!r}")
        if hasattr(self, 'name') and getattr(self, 'name', None):
            key_attrs.append(f"name={getattr(self, 'name')!r}")

        attrs_str = ", ".join(key_attrs) if key_attrs else "no_key_attrs"
        return f"{class_name}({attrs_str})"
