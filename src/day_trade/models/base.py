"""
データベースモデルの基底クラス（SQLAlchemy 2.0対応・改善版）

改善点:
- タイムゾーン対応（UTC保存・ローカル表示）
- Pydantic連携強化
- to_dict拡張性向上
- バリデーション機能追加
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from sqlalchemy import DateTime, Integer
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, mapped_column

try:
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field, create_model
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    PydanticBaseModel = None

from ..utils.logging_config import get_context_logger
from .database import Base

logger = get_context_logger(__name__)

# TypeVar for generic typing
T = TypeVar('T', bound='BaseModel')


class TimestampMixin:
    """作成日時・更新日時を持つMixinクラス（SQLAlchemy 2.0対応・タイムゾーン対応版）"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),  # タイムゾーン情報を保持
        default=lambda: datetime.now(timezone.utc),  # UTC時刻で保存（callable）
        nullable=False,
        doc="レコード作成日時（UTC）"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),  # タイムゾーン情報を保持
        default=lambda: datetime.now(timezone.utc),  # UTC時刻で保存（callable）
        onupdate=lambda: datetime.now(timezone.utc),  # UTC時刻で更新（callable）
        nullable=False,
        doc="レコード更新日時（UTC）"
    )

    def get_created_at_local(self, tz: Optional[timezone] = None) -> datetime:
        """作成日時をローカルタイムゾーンで取得"""
        if tz is None:
            tz = datetime.now().astimezone().tzinfo  # システムのローカルタイムゾーン
        return self.created_at.astimezone(tz) if self.created_at else None

    def get_updated_at_local(self, tz: Optional[timezone] = None) -> datetime:
        """更新日時をローカルタイムゾーンで取得"""
        if tz is None:
            tz = datetime.now().astimezone().tzinfo  # システムのローカルタイムゾーン
        return self.updated_at.astimezone(tz) if self.updated_at else None


class BaseModel(Base, TimestampMixin):
    """全モデルの基底クラス（SQLAlchemy 2.0対応・改善版）"""

    __abstract__ = True

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
        doc="主キー（自動採番）"
    )

    @declared_attr
    def __tablename__(cls) -> str:  # noqa: N805
        """テーブル名を自動生成（クラス名を小文字に）"""
        return cls.__name__.lower()

    # キャッシュ用クラス変数
    _pydantic_model_cache: Dict[str, Type[PydanticBaseModel]] = {}

    def __init__(self, **kwargs):
        """初期化（タイムスタンプ設定を含む）"""
        super().__init__(**kwargs)
        # タイムスタンプを明示的に設定（既に値が設定されている場合は上書きしない）
        now_utc = datetime.now(timezone.utc)

        # created_atの処理：kwargsで指定されているか、既に値があるかチェック
        if 'created_at' not in kwargs and (not hasattr(self, 'created_at') or getattr(self, 'created_at', None) is None):
            self.created_at = now_utc

        # updated_atの処理：kwargsで指定されているか、既に値があるかチェック
        if 'updated_at' not in kwargs and (not hasattr(self, 'updated_at') or getattr(self, 'updated_at', None) is None):
            self.updated_at = now_utc

    def to_dict(
        self,
        include_relations: bool = False,
        relation_depth: int = 1,
        exclude_keys: Optional[Union[set, List[str]]] = None,
        include_keys: Optional[Union[set, List[str]]] = None,
        convert_datetime: bool = True,
        local_timezone: bool = False
    ) -> Dict[str, Any]:
        """
        モデルを辞書に変換（拡張版）

        Args:
            include_relations: リレーションを含めるか
            relation_depth: リレーションの深度（無限ループ防止）
            exclude_keys: 除外するキー
            include_keys: 含めるキー（指定時は指定キーのみ）
            convert_datetime: datetime型を文字列に変換するか
            local_timezone: datetime型をローカルタイムゾーンで出力するか
        """
        if relation_depth <= 0:
            return {}

        exclude_keys = set(exclude_keys) if exclude_keys else set()
        include_keys = set(include_keys) if include_keys else None

        result = {}

        # カラムデータを処理
        for column in self.__table__.columns:
            column_name = column.name

            # キーフィルタリング
            if include_keys and column_name not in include_keys:
                continue
            if column_name in exclude_keys:
                continue

            value = getattr(self, column_name, None)
            if value is not None:
                # datetime型の処理
                if isinstance(value, datetime) and convert_datetime:
                    if local_timezone and value.tzinfo:
                        # ローカルタイムゾーンに変換
                        local_tz = datetime.now().astimezone().tzinfo
                        value = value.astimezone(local_tz)
                    result[column_name] = value.isoformat()
                # Decimal型の処理
                elif isinstance(value, Decimal):
                    result[column_name] = float(value)
                else:
                    result[column_name] = value

        # リレーションデータを処理
        if include_relations and relation_depth > 1:
            try:
                for attr_name in dir(self):
                    if attr_name.startswith('_') or attr_name in exclude_keys:
                        continue
                    if include_keys and attr_name not in include_keys:
                        continue

                    attr_value = getattr(self, attr_name, None)
                    if attr_value is None:
                        continue

                    # リレーションオブジェクトの場合
                    if hasattr(attr_value, '__table__'):
                        result[attr_name] = attr_value.to_dict(
                            include_relations=True,
                            relation_depth=relation_depth - 1,
                            exclude_keys=exclude_keys,
                            convert_datetime=convert_datetime,
                            local_timezone=local_timezone
                        )
                    # リストの場合（一対多リレーション）
                    elif isinstance(attr_value, list) and attr_value and hasattr(attr_value[0], '__table__'):
                        result[attr_name] = [
                            item.to_dict(
                                include_relations=True,
                                relation_depth=relation_depth - 1,
                                exclude_keys=exclude_keys,
                                convert_datetime=convert_datetime,
                                local_timezone=local_timezone
                            ) for item in attr_value[:10]  # 最大10件制限
                        ]
            except Exception as e:
                logger.warning(f"リレーション処理でエラー: {e}")

        return result

    def to_dict_safe(self, **kwargs) -> Dict[str, Any]:
        """安全な辞書変換（後方互換性のため残存）"""
        return self.to_dict(
            include_relations=False,
            exclude_keys={'password', 'secret', 'token'},
            convert_datetime=True,
            local_timezone=True,
            **kwargs
        )

    def update_from_dict(
        self,
        data: Dict[str, Any],
        exclude_keys: Optional[Union[set, List[str]]] = None,
        validate: bool = True,
        auto_convert: bool = True
    ) -> None:
        """
        辞書からモデルを更新（拡張版）

        Args:
            data: 更新データ
            exclude_keys: 除外するキー
            validate: バリデーションを実行するか
            auto_convert: 自動型変換を行うか
        """
        exclude_keys = set(exclude_keys) if exclude_keys is not None else {'id', 'created_at'}

        for key, value in data.items():
            logger.debug(f"Processing key: {key}, value: {value}, in exclude_keys: {key in exclude_keys}")
            if key not in exclude_keys and hasattr(self, key) and hasattr(self.__table__.columns, key):
                column = self.__table__.columns[key]

                # 自動型変換
                if auto_convert and value is not None:
                    try:
                        # datetime型の変換
                        if isinstance(column.type, DateTime):
                            if isinstance(value, str):
                                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                # タイムゾーン情報がない場合はUTCとして扱う
                                if value.tzinfo is None:
                                    value = value.replace(tzinfo=timezone.utc)
                                logger.debug(f"DateTime変換成功: {key} = {value}")
                            elif not isinstance(value, datetime):
                                # datetime以外の場合はスキップするかエラーとする
                                logger.warning(f"DateTimeフィールド{key}に非datetime型が指定されました: {type(value)}")
                        # Decimal型の変換
                        elif hasattr(column.type, 'scale') and isinstance(value, (int, float, str)):
                            value = Decimal(str(value))
                    except (ValueError, TypeError) as e:
                        if validate:
                            raise ValueError(f"{key}の値変換に失敗: {e}") from e
                        logger.warning(f"値変換警告 {key}: {e}")

                # バリデーション
                if validate:
                    self._validate_field(key, value, column)

                # 値を設定
                setattr(self, key, value)

    def _validate_field(self, key: str, value: Any, column) -> None:
        """フィールドバリデーション"""
        # NULL制約チェック
        if not column.nullable and value is None:
            raise ValueError(f"{key}はNULLにできません")

        # 長さ制約チェック（String型）
        if hasattr(column.type, 'length') and column.type.length and isinstance(value, str) and len(value) > column.type.length:
            raise ValueError(f"{key}の長さが制限を超えています ({len(value)} > {column.type.length})")

    # Pydantic連携機能
    @classmethod
    def create_pydantic_model(
        cls,
        name: Optional[str] = None,
        include_relations: bool = False,
        exclude_fields: Optional[Union[set, List[str]]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Optional[Type[PydanticBaseModel]]:
        """
        SQLAlchemyモデルからPydanticモデルを動的に作成

        Args:
            name: Pydanticモデル名（未指定の場合はSQLAlchemyモデル名を使用）
            include_relations: リレーションフィールドを含めるか
            exclude_fields: 除外するフィールド
            config_overrides: Pydantic設定のオーバーライド
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("Pydanticがインストールされていません")
            return None

        model_name = name or f"{cls.__name__}Pydantic"
        cache_key = f"{cls.__name__}_{model_name}_{include_relations}_{hash(frozenset(exclude_fields) if exclude_fields else frozenset())}"

        # キャッシュから検索
        if cache_key in cls._pydantic_model_cache:
            return cls._pydantic_model_cache[cache_key]

        exclude_fields = set(exclude_fields) if exclude_fields else set()
        fields = {}

        # カラムフィールドを処理
        for column in cls.__table__.columns:
            if column.name in exclude_fields:
                continue

            # Python型を推定
            python_type = cls._get_python_type_from_column(column)

            # デフォルト値を設定（idやtimestampはオプショナルにする）
            if column.name in ['id', 'created_at', 'updated_at']:
                field_info = Field(default=None, description=getattr(column, 'doc', None))
                if not column.nullable:
                    # nullable=Falseでもデフォルト値があるフィールドはOptionalにする
                    python_type = Optional[python_type.__args__[0] if hasattr(python_type, '__args__') else python_type]
            else:
                field_info = Field(description=getattr(column, 'doc', None))

            fields[column.name] = (python_type, field_info)

        # リレーションフィールドを処理（簡単な実装）
        if include_relations:
            try:
                for attr_name in dir(cls):
                    if attr_name.startswith('_') or attr_name in exclude_fields:
                        continue
                    attr = getattr(cls, attr_name, None)
                    if hasattr(attr, 'property') and hasattr(attr.property, 'mapper'):
                        # リレーションフィールドの場合はオプショナルとして追加
                        fields[attr_name] = (Optional[Any], Field(default=None))
            except Exception as e:
                logger.warning(f"リレーションフィールド処理エラー: {e}")

        # 設定クラスを作成（Pydantic v2対応）
        from pydantic import ConfigDict

        config_dict = ConfigDict(
            from_attributes=True,  # Pydantic v2の新しい設定
            arbitrary_types_allowed=True,
        )
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config_dict, key, value)

        # Pydanticモデルを作成
        pydantic_model = create_model(
            model_name,
            __config__=config_dict,
            **fields
        )

        # キャッシュに保存
        cls._pydantic_model_cache[cache_key] = pydantic_model
        return pydantic_model

    @classmethod
    def _get_python_type_from_column(cls, column) -> Type:
        """カラムからPython型を推定"""
        from sqlalchemy import Boolean, Float, Integer, String, Text
        from sqlalchemy.types import DECIMAL

        sql_type = column.type
        python_type = Any  # デフォルト

        if isinstance(sql_type, (String, Text)):
            python_type = str
        elif isinstance(sql_type, Integer):
            python_type = int
        elif isinstance(sql_type, Boolean):
            python_type = bool
        elif isinstance(sql_type, Float):
            python_type = float
        elif isinstance(sql_type, DECIMAL):
            python_type = Decimal
        elif isinstance(sql_type, DateTime):
            python_type = datetime

        # NULL制約に基づいてOptionalを付与
        if column.nullable:
            python_type = Optional[python_type]

        return python_type

    def to_pydantic(
        self,
        model_class: Optional[Type[PydanticBaseModel]] = None,
        **kwargs
    ) -> Optional[PydanticBaseModel]:
        """
        SQLAlchemyモデルインスタンスをPydanticモデルに変換

        Args:
            model_class: 使用するPydanticモデルクラス（未指定の場合は自動作成）
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("Pydanticがインストールされていません")
            return None

        if model_class is None:
            model_class = self.create_pydantic_model(**kwargs)
            if model_class is None:
                return None

        try:
            return model_class.model_validate(self)
        except Exception as e:
            logger.error(f"Pydantic変換エラー: {e}")
            return None

    @classmethod
    def from_pydantic(
        cls: Type[T],
        pydantic_obj: PydanticBaseModel,
        exclude_unset: bool = True,
        **kwargs
    ) -> T:
        """
        PydanticモデルからSQLAlchemyモデルインスタンスを作成

        Args:
            pydantic_obj: Pydanticモデルインスタンス
            exclude_unset: 設定されていないフィールドを除外するか
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydanticがインストールされていません")

        # Pydanticモデルから辞書へ変換
        if exclude_unset:
            data = pydantic_obj.model_dump(exclude_unset=True)
        else:
            data = pydantic_obj.model_dump()

        # SQLAlchemyモデルインスタンスを作成
        instance = cls(**kwargs)
        instance.update_from_dict(data, validate=True, auto_convert=True)
        return instance

    # ユーティリティメソッド
    def clone(self: T, exclude_keys: Optional[Union[set, List[str]]] = None) -> T:
        """モデルインスタンスのクローンを作成"""
        exclude_keys = set(exclude_keys) if exclude_keys else {'id', 'created_at', 'updated_at'}

        data = self.to_dict(
            include_relations=False,
            exclude_keys=exclude_keys,
            convert_datetime=False
        )

        return self.__class__(**data)

    def refresh_timestamps(self) -> None:
        """タイムスタンプを現在時刻に更新"""
        now_utc = datetime.now(timezone.utc)
        if not self.created_at:
            self.created_at = now_utc
        self.updated_at = now_utc

    def is_new_record(self) -> bool:
        """新しいレコードかどうかを判定"""
        return self.id is None

    def get_table_name(self) -> str:
        """テーブル名を取得"""
        return self.__table__.name

    @classmethod
    def get_column_names(cls) -> List[str]:
        """カラム名のリストを取得"""
        return [column.name for column in cls.__table__.columns]

    @classmethod
    def get_primary_key_columns(cls) -> List[str]:
        """主キーカラムのリストを取得"""
        return [column.name for column in cls.__table__.primary_key.columns]

    def __repr__(self) -> str:
        """オブジェクトの文字列表現（改善版）"""
        class_name = self.__class__.__name__

        # 主要な属性のみ表示（長すぎる場合を考慮）
        key_attrs = []
        if hasattr(self, 'id') and self.id is not None:
            key_attrs.append(f"id={self.id}")
        if hasattr(self, 'code') and getattr(self, 'code', None):
            key_attrs.append(f"code={self.code!r}")
        if hasattr(self, 'name') and getattr(self, 'name', None):
            key_attrs.append(f"name={self.name!r}")

        attrs_str = ", ".join(key_attrs) if key_attrs else "no_key_attrs"
        return f"{class_name}({attrs_str})"

    def __eq__(self, other) -> bool:
        """等価比較（主キーベース）"""
        if not isinstance(other, self.__class__):
            return False

        # 主キーがある場合は主キーで比較
        if self.id is not None and other.id is not None:
            return self.id == other.id

        # 主キーがない場合は主要カラムで比較（タイムスタンプは除外）
        self_dict = self.to_dict(convert_datetime=False, exclude_keys={'created_at', 'updated_at'})
        other_dict = other.to_dict(convert_datetime=False, exclude_keys={'created_at', 'updated_at'})
        return self_dict == other_dict

    def __hash__(self) -> int:
        """ハッシュ値（主キーベース）"""
        if self.id is not None:
            return hash((self.__class__, self.id))
        # IDがない場合はオブジェクトIDを使用
        return hash(id(self))
