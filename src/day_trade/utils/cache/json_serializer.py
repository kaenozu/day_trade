"""
JSONシリアライザモジュール

キャッシュキー生成のためのカスタムJSONシリアライザを提供します。
Pydantic、Decimal、Enum、datetime等の主要な型に対応します。
"""

import json
import time
from decimal import Decimal
from enum import Enum
from typing import Any

from ..logging_config import get_logger
from .config import get_cache_config
from .constants import CacheConstants, CacheTimeoutError

logger = get_logger(__name__)

# オプショナル依存関係のインポート
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None


def _json_serializer(obj: Any) -> Any:
    """
    JSON シリアライザーのカスタムハンドラー（堅牢版・Pydantic v2対応・シンプルタイムアウト機能付き）

    Args:
        obj: シリアライズするオブジェクト

    Returns:
        シリアライズ可能な値

    Raises:
        CacheTimeoutError: シリアライゼーションがタイムアウトした場合
    """
    cache_config = get_cache_config()
    start_time = time.time()
    timeout_seconds = cache_config.serialization_timeout

    def check_timeout():
        """タイムアウトチェック"""
        if time.time() - start_time > timeout_seconds:
            raise CacheTimeoutError(
                f"Serialization timeout after {timeout_seconds} seconds",
                timeout_seconds,
                "json_serialization",
            )

    try:
        # Pydanticモデル（v2対応強化）
        if PYDANTIC_AVAILABLE and isinstance(obj, BaseModel):
            return _serialize_pydantic_model(obj, check_timeout)

        # Decimal型（精度保持）
        elif isinstance(obj, Decimal):
            return _serialize_decimal(obj, check_timeout)

        # Enum型
        elif isinstance(obj, Enum):
            return _serialize_enum(obj, check_timeout)

        # datetime/date/time オブジェクト
        elif hasattr(obj, "isoformat"):
            return _serialize_datetime(obj, check_timeout)

        # set型
        elif isinstance(obj, set):
            return _serialize_set(obj, check_timeout)

        # frozenset型
        elif isinstance(obj, frozenset):
            return _serialize_frozenset(obj, check_timeout)

        # bytes型
        elif isinstance(obj, bytes):
            return _serialize_bytes(obj, check_timeout)

        # 関数またはメソッド
        elif callable(obj):
            return _serialize_callable(obj, check_timeout)

        # __dict__ を持つオブジェクト
        elif hasattr(obj, "__dict__"):
            return _serialize_object_with_dict(obj, check_timeout)

        # NumPy配列（オプション対応）
        elif hasattr(obj, "tolist") and hasattr(obj, "dtype"):
            return _serialize_numpy_array(obj, check_timeout)

        # その他の場合は文字列表現（型情報付き）
        else:
            check_timeout()
            return {
                "__unknown_type__": obj.__class__.__name__,
                "__module__": getattr(obj.__class__, "__module__", "unknown"),
                "value": str(obj),
            }

    except CacheTimeoutError:
        # 既にCacheTimeoutErrorの場合はそのまま再発生
        raise
    except Exception as e:
        # 予期しないエラーのフォールバック
        logger.warning(f"Serialization fallback for {type(obj).__name__}: {e}")
        return {
            "__serialization_fallback__": True,
            "__type__": obj.__class__.__name__,
            "__error__": str(e),
            "value": str(obj),
        }


def _serialize_pydantic_model(obj: Any, check_timeout) -> dict:
    """Pydanticモデルのシリアライゼーション"""
    check_timeout()
    try:
        # Pydantic v2の場合
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json", exclude_unset=False)
        # Pydantic v1の場合
        elif hasattr(obj, "dict"):
            return obj.dict()
        else:
            # フォールバック
            return {
                "__pydantic_model__": obj.__class__.__name__,
                "data": str(obj),
            }
    except Exception as e:
        logger.warning(f"Pydantic model serialization failed: {e}")
        return {"__pydantic_model__": obj.__class__.__name__, "error": str(e)}


def _serialize_decimal(obj: Decimal, check_timeout) -> dict:
    """Decimalのシリアライゼーション"""
    check_timeout()
    # 特殊値のチェック
    if obj.is_nan():
        return {"__decimal__": "NaN"}
    elif obj.is_infinite():
        return {"__decimal__": "Infinity" if obj > 0 else "-Infinity"}
    else:
        return {"__decimal__": str(obj)}


def _serialize_enum(obj: Enum, check_timeout) -> dict:
    """Enumのシリアライゼーション"""
    check_timeout()
    return {
        "__enum__": obj.__class__.__name__,
        "__module__": getattr(obj.__class__, "__module__", "unknown"),
        "name": obj.name,
        "value": obj.value,
    }


def _serialize_datetime(obj: Any, check_timeout) -> dict:
    """datetime系オブジェクトのシリアライゼーション"""
    check_timeout()
    try:
        iso_string = obj.isoformat()
        return {
            "__datetime__": iso_string,
            "__type__": obj.__class__.__name__,
            "__timezone__": (
                str(getattr(obj, "tzinfo", None))
                if hasattr(obj, "tzinfo")
                else None
            ),
        }
    except Exception:
        return {"__datetime__": str(obj), "__type__": obj.__class__.__name__}


def _serialize_set(obj: set, check_timeout) -> dict:
    """setのシリアライゼーション"""
    check_timeout()
    try:
        sorted_items = []
        unsorted_items = []

        for item in obj:
            check_timeout()
            try:
                # シリアライズ可能かテスト（再帰呼び出しを避ける）
                json.dumps(item, default=str)
                sorted_items.append(item)
            except Exception:
                unsorted_items.append(str(item))

        try:
            sorted_items.sort()
        except TypeError:
            # ソートできない場合は文字列変換してソート
            sorted_items = sorted([str(item) for item in sorted_items])

        return {"__set__": sorted_items + sorted(unsorted_items)}
    except Exception:
        return {"__set__": [str(item) for item in obj]}


def _serialize_frozenset(obj: frozenset, check_timeout) -> dict:
    """frozensetのシリアライゼーション"""
    check_timeout()
    try:
        return {"__frozenset__": _serialize_set(set(obj), check_timeout)["__set__"]}
    except Exception:
        return {"__frozenset__": [str(item) for item in obj]}


def _serialize_bytes(obj: bytes, check_timeout) -> dict:
    """bytesのシリアライゼーション"""
    check_timeout()
    try:
        # UTF-8を最初に試行
        return {"__bytes__": obj.decode("utf-8"), "encoding": "utf-8"}
    except UnicodeDecodeError:
        try:
            # 他のエンコーディングを試行
            try:
                import chardet
                detected = chardet.detect(obj)
                if (
                    detected
                    and detected["encoding"]
                    and detected.get("confidence", 0)
                    > CacheConstants.CHARSET_DETECTION_CONFIDENCE_THRESHOLD
                ):
                    return {
                        "__bytes__": obj.decode(detected["encoding"]),
                        "encoding": detected["encoding"],
                        "confidence": detected["confidence"],
                    }
            except (ImportError, UnicodeDecodeError):
                pass
            # フォールバック: hexエンコーディング
            return {"__bytes__": obj.hex(), "encoding": "hex"}
        except Exception:
            # 最終フォールバック
            return {
                "__bytes__": "<binary_data>",
                "encoding": "unknown",
                "size": len(obj),
            }


def _serialize_callable(obj: Any, check_timeout) -> dict:
    """関数/メソッドのシリアライゼーション"""
    check_timeout()
    return {
        "__callable__": getattr(obj, "__name__", repr(obj)),
        "__module__": getattr(obj, "__module__", "unknown"),
        "__type__": "function" if hasattr(obj, "__name__") else "callable",
    }


def _serialize_object_with_dict(obj: Any, check_timeout) -> dict:
    """__dict__を持つオブジェクトのシリアライゼーション"""
    check_timeout()
    try:
        # __dict__の内容を安全にシリアライズ
        safe_dict = {}
        for key, value in obj.__dict__.items():
            check_timeout()
            try:
                safe_dict[str(key)] = _json_serializer(value)
            except Exception as e:
                safe_dict[str(key)] = f"<serialization_error: {e}>"

        return {
            "__object__": obj.__class__.__name__,
            "__module__": getattr(obj.__class__, "__module__", "unknown"),
            "data": safe_dict,
        }
    except Exception as e:
        return {
            "__object__": obj.__class__.__name__,
            "__module__": getattr(obj.__class__, "__module__", "unknown"),
            "error": str(e),
        }


def _serialize_numpy_array(obj: Any, check_timeout) -> dict:
    """NumPy配列のシリアライゼーション"""
    check_timeout()
    try:
        return {
            "__numpy_array__": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": obj.shape,
        }
    except Exception:
        return {"__numpy_array__": str(obj)}