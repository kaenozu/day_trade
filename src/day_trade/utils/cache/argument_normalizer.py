"""
引数正規化モジュール

キャッシュキー生成のための引数正規化機能を提供します。
循環参照検出、深い再帰処理、Pydantic対応等を含みます。
"""

import logging
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Union, Tuple, Set

from ..logging_config import get_logger
from .config import get_cache_config
from .constants import CacheConstants

logger = get_logger(__name__)

# オプショナル依存関係のインポート
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None


def _estimate_data_complexity(args: Union[Tuple, Dict, Any]) -> int:
    """
    データの複雑度を推定（再帰制限の動的調整用）

    Args:
        args: 複雑度を推定するデータ

    Returns:
        推定複雑度（オブジェクト数）
    """
    try:
        complexity = 0

        if isinstance(args, (dict, list, tuple)):
            complexity += len(args)

            # ネストした構造の場合は追加でカウント
            for item in args.values() if isinstance(args, dict) else args:
                if isinstance(item, (dict, list, tuple)):
                    complexity += len(item)

        elif hasattr(args, "__dict__"):
            complexity += len(args.__dict__)

        return complexity

    except Exception:
        # エラーの場合は中程度の複雑度を返す
        return 50


def _normalize_arguments(
    args: Union[Tuple, Dict, Any],
    max_depth: int = None,
    current_depth: int = 0,
    seen_objects: Optional[Set[int]] = None,
) -> Any:
    """
    引数を正規化してシリアライズ可能な形式に変換

    Args:
        args: 正規化する引数
        max_depth: 最大再帰深度
        current_depth: 現在の再帰深度
        seen_objects: 既に処理したオブジェクトのセット（循環参照検出用）

    Returns:
        正規化された引数
    """
    cache_config = get_cache_config()
    
    if max_depth is None:
        max_depth = cache_config.max_recursion_depth
    if seen_objects is None:
        seen_objects = set()

    if current_depth >= max_depth:
        if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logging.WARNING):
            logger.warning(
                f"Recursion depth limit reached ({max_depth}), truncating object of type {type(args).__name__}"
            )
        if isinstance(args, (dict, list, tuple)):
            return f"<truncated {type(args).__name__} with {len(args)} items at depth {max_depth}>"
        else:
            return f"<truncated {type(args).__name__} at depth {max_depth}>"

    try:
        obj_id = id(args)
    except TypeError:
        obj_id = None

    if obj_id is not None:
        if obj_id in seen_objects:
            return f"<circular reference to {type(args).__name__}>"
        seen_objects.add(obj_id)

    try:
        # Pydantic モデルの処理
        if PYDANTIC_AVAILABLE and isinstance(args, BaseModel):
            try:
                model_data = args.model_dump() if hasattr(args, "model_dump") else args.dict()
                return _normalize_arguments(model_data, max_depth, current_depth + 1, seen_objects)
            except Exception:
                return f"<pydantic model: {args.__class__.__name__}>"

        # Decimal の処理
        elif isinstance(args, Decimal):
            return str(args)

        # Enum の処理
        elif isinstance(args, Enum):
            return f"<enum {args.__class__.__name__}: {args.value}>"

        # tuple, list の処理
        elif isinstance(args, (tuple, list)):
            return _normalize_sequence(args, max_depth, current_depth, seen_objects)

        # dict の処理
        elif isinstance(args, dict):
            return _normalize_dict(args, max_depth, current_depth, seen_objects)

        # set の処理
        elif isinstance(args, set):
            return _normalize_set(args, max_depth, current_depth, seen_objects)

        # datetime オブジェクトの処理
        elif hasattr(args, "isoformat"):
            return args.isoformat()

        # __dict__ を持つオブジェクトの処理
        elif hasattr(args, "__dict__"):
            obj_dict = getattr(args, "__dict__", {})
            if obj_dict:
                return _normalize_arguments(obj_dict, max_depth, current_depth + 1, seen_objects)
            else:
                return f"<empty_object: {args.__class__.__name__}>"

        # 関数/メソッドの処理
        elif callable(args):
            name = getattr(args, "__name__", repr(args))
            return f"<callable: {name}>"

        # その他
        else:
            if isinstance(args, str) and len(args) > CacheConstants.DEFAULT_MAX_KEY_LENGTH:
                return f"<long_string: {len(args)} chars>"
            return args

    finally:
        if obj_id is not None:
            seen_objects.discard(obj_id)


def _normalize_sequence(
    seq: Union[tuple, list],
    max_depth: int,
    current_depth: int,
    seen_objects: Set[int]
) -> list:
    """
    シーケンス型（tuple, list）の正規化
    
    Args:
        seq: 正規化するシーケンス
        max_depth: 最大再帰深度
        current_depth: 現在の再帰深度
        seen_objects: 既に処理したオブジェクトのセット
        
    Returns:
        正規化されたリスト
    """
    normalized_items = []
    for i, arg in enumerate(seq):
        normalized_items.append(_normalize_arguments(arg, max_depth, current_depth + 1, seen_objects))
        if i >= CacheConstants.DEFAULT_MAX_OPERATION_HISTORY:
            normalized_items.append(f"<truncated: {len(seq) - i - 1} more items>")
            break
    return normalized_items


def _normalize_dict(
    d: dict,
    max_depth: int,
    current_depth: int,
    seen_objects: Set[int]
) -> dict:
    """
    辞書の正規化
    
    Args:
        d: 正規化する辞書
        max_depth: 最大再帰深度
        current_depth: 現在の再帰深度
        seen_objects: 既に処理したオブジェクトのセット
        
    Returns:
        正規化された辞書
    """
    normalized_dict = {}
    processed_count = 0
    for k, v in d.items():
        key_str = str(k)
        normalized_dict[key_str] = _normalize_arguments(v, max_depth, current_depth + 1, seen_objects)
        processed_count += 1
        if processed_count >= CacheConstants.DEFAULT_MAX_OPERATION_HISTORY:
            if len(d) - processed_count > 0:
                normalized_dict["<truncated>"] = f"{len(d) - processed_count} more keys"
            break
    return normalized_dict


def _normalize_set(
    s: set,
    max_depth: int,
    current_depth: int,
    seen_objects: Set[int]
) -> dict:
    """
    セットの正規化
    
    Args:
        s: 正規化するセット
        max_depth: 最大再帰深度
        current_depth: 現在の再帰深度
        seen_objects: 既に処理したオブジェクトのセット
        
    Returns:
        正規化されたセット（辞書形式）
    """
    normalized_items = []
    for i, item in enumerate(s):
        normalized_items.append(_normalize_arguments(item, max_depth, current_depth + 1, seen_objects))
        if i >= CacheConstants.DEFAULT_MAX_OPERATION_HISTORY:
            normalized_items.append(f"<truncated: {len(s) - i - 1} more items>")
            break
    return {"__set__": sorted(normalized_items, key=str)}