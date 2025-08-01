"""
キャッシュユーティリティ
安全で効率的なキャッシュキー生成とキャッシュ操作を提供
"""

import hashlib
import json
import logging
from typing import Any, Dict, Tuple, Union

logger = logging.getLogger(__name__)


def generate_safe_cache_key(func_name: str, *args, **kwargs) -> str:
    """
    安全なキャッシュキー生成

    Args:
        func_name: 関数名
        *args: 位置引数
        **kwargs: キーワード引数

    Returns:
        生成されたキャッシュキー

    Raises:
        CacheError: キャッシュキー生成に失敗した場合
    """
    try:
        # 引数を正規化してシリアライズ可能な形式に変換
        normalized_args = _normalize_arguments(args)
        normalized_kwargs = _normalize_arguments(kwargs)

        # シリアライズしてハッシュ化
        serializable_data = {
            "function": func_name,
            "args": normalized_args,
            "kwargs": (
                sorted(normalized_kwargs.items())
                if isinstance(normalized_kwargs, dict)
                else normalized_kwargs
            ),
        }

        serialized = json.dumps(
            serializable_data, sort_keys=True, default=_json_serializer
        )
        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        logger.debug(f"Generated cache key for {func_name}: {cache_key}")
        return f"{func_name}:{cache_key}"

    except Exception as e:
        # フォールバック: ハッシュベースのキー生成
        logger.warning(f"JSON serialization failed for cache key, using fallback: {e}")
        return _generate_fallback_cache_key(func_name, args, kwargs)


def _normalize_arguments(args: Union[Tuple, Dict, Any]) -> Any:
    """
    引数を正規化してシリアライズ可能にする

    Args:
        args: 正規化する引数

    Returns:
        正規化された引数
    """
    if isinstance(args, (tuple, list)):
        return [_normalize_arguments(arg) for arg in args]
    elif isinstance(args, dict):
        return {str(k): _normalize_arguments(v) for k, v in args.items()}
    elif hasattr(args, "__dict__"):
        # オブジェクトの場合は辞書に変換
        return _normalize_arguments(args.__dict__)
    elif callable(args):
        # 関数の場合は名前を使用
        return getattr(args, "__name__", str(args))
    else:
        return args


def _json_serializer(obj: Any) -> str:
    """
    JSON シリアライザーのカスタムハンドラー

    Args:
        obj: シリアライズするオブジェクト

    Returns:
        シリアライズされた文字列
    """
    if hasattr(obj, "isoformat"):
        # datetime オブジェクト
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        # 一般的なオブジェクト
        return str(obj.__dict__)
    else:
        return str(obj)


def _generate_fallback_cache_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
    """
    フォールバック用のキャッシュキー生成

    Args:
        func_name: 関数名
        args: 位置引数
        kwargs: キーワード引数

    Returns:
        生成されたフォールバックキー
    """
    try:
        # ハッシュベースの安全なキー生成
        args_hash = hash(args) if args else 0
        kwargs_hash = hash(tuple(sorted(kwargs.items()))) if kwargs else 0

        combined_hash = hashlib.md5(
            f"{func_name}:{args_hash}:{kwargs_hash}".encode(), usedforsecurity=False
        ).hexdigest()

        return f"{func_name}:fallback:{combined_hash}"

    except Exception as e:
        # 最終フォールバック
        logger.error(f"Fallback cache key generation failed: {e}")
        import time

        timestamp = int(time.time() * 1000000)  # マイクロ秒タイムスタンプ
        return f"{func_name}:emergency:{timestamp}"


class CacheStats:
    """キャッシュ統計情報"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.errors = 0

    @property
    def hit_rate(self) -> float:
        """ヒット率を計算"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_hit(self):
        """キャッシュヒットを記録"""
        self.hits += 1

    def record_miss(self):
        """キャッシュミスを記録"""
        self.misses += 1

    def record_set(self):
        """キャッシュセットを記録"""
        self.sets += 1

    def record_eviction(self):
        """キャッシュエビクションを記録"""
        self.evictions += 1

    def record_error(self):
        """キャッシュエラーを記録"""
        self.errors += 1

    def reset(self):
        """統計をリセット"""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.errors = 0

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """統計情報を辞書として返す"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "total_requests": self.hits + self.misses,
        }


def validate_cache_key(key: str) -> bool:
    """
    キャッシュキーの妥当性を検証

    Args:
        key: 検証するキー

    Returns:
        キーが有効かどうか
    """
    if not key or not isinstance(key, str):
        return False

    # 長すぎるキーを拒否（メモリ効率のため）
    if len(key) > 1000:
        return False

    # 制御文字を含むキーを拒否
    if any(ord(c) < 32 or ord(c) == 127 for c in key):
        return False

    return True


def sanitize_cache_value(value: Any) -> Any:
    """
    キャッシュ値のサニタイズ

    Args:
        value: サニタイズする値

    Returns:
        サニタイズされた値
    """
    # None値はそのまま許可
    if value is None:
        return value

    # 大きすぎるオブジェクトの警告
    try:
        import sys

        size = sys.getsizeof(value)
        if size > 10 * 1024 * 1024:  # 10MB
            logger.warning(f"Large cache value detected: {size} bytes")
    except Exception:
        pass  # サイズ取得に失敗した場合は無視

    return value
