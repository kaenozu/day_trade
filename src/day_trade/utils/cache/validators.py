"""
キャッシュバリデーション・サニタイゼーションモジュール

キャッシュキーとキャッシュ値の妥当性検証とサニタイゼーション機能を提供します。
安全で効率的なキャッシュ操作をサポートします。
"""

import sys
from typing import Any, Optional

from ..logging_config import get_logger
from .config import get_cache_config
from .constants import CacheValidationError

logger = get_logger(__name__)


def validate_cache_key(key: str, config: Optional["CacheConfig"] = None) -> bool:
    """
    キャッシュキーの妥当性を検証

    Args:
        key: 検証するキー
        config: キャッシュ設定（Noneの場合はデフォルト設定を使用）

    Returns:
        キーが有効かどうか

    Raises:
        CacheValidationError: キーが無効な場合の詳細な情報
    """
    if config is None:
        config = get_cache_config()

    # 基本的な型チェック
    if not key or not isinstance(key, str):
        raise CacheValidationError(
            "Cache key must be a non-empty string",
            "key_type_validation",
            key
        )

    # 長さチェック
    if len(key) > config.max_key_length:
        raise CacheValidationError(
            f"Cache key length ({len(key)}) exceeds maximum allowed length ({config.max_key_length})",
            "key_length_validation",
            key
        )

    # 制御文字チェック
    has_control_chars = any(ord(c) < 32 or ord(c) == 127 for c in key)
    if has_control_chars:
        raise CacheValidationError(
            "Cache key contains control characters",
            "key_control_chars_validation",
            key
        )

    # 空白文字のみのキーをチェック
    if key.isspace():
        raise CacheValidationError(
            "Cache key cannot consist only of whitespace characters",
            "key_whitespace_validation",
            key
        )

    return True


def is_valid_cache_key(key: str, config: Optional["CacheConfig"] = None) -> bool:
    """
    キャッシュキーの妥当性を検証（例外を発生させない版）

    Args:
        key: 検証するキー
        config: キャッシュ設定（Noneの場合はデフォルト設定を使用）

    Returns:
        キーが有効かどうか
    """
    try:
        validate_cache_key(key, config)
        return True
    except CacheValidationError:
        return False


def sanitize_cache_key(key: str, config: Optional["CacheConfig"] = None) -> str:
    """
    キャッシュキーをサニタイズして安全な形式に変換

    Args:
        key: サニタイズするキー
        config: キャッシュ設定（Noneの場合はデフォルト設定を使用）

    Returns:
        サニタイズされたキー

    Raises:
        CacheValidationError: サニタイズできない場合
    """
    if config is None:
        config = get_cache_config()

    if not key or not isinstance(key, str):
        raise CacheValidationError(
            "Cannot sanitize non-string or empty cache key",
            "key_sanitization",
            key
        )

    # 制御文字を除去
    sanitized = ''.join(c for c in key if ord(c) >= 32 and ord(c) != 127)

    # 空白文字を正規化
    sanitized = ' '.join(sanitized.split())

    # 長さを制限
    if len(sanitized) > config.max_key_length:
        sanitized = sanitized[:config.max_key_length].rstrip()

    # サニタイズ後にキーが空になった場合
    if not sanitized:
        raise CacheValidationError(
            "Cache key became empty after sanitization",
            "key_empty_after_sanitization",
            key
        )

    return sanitized


def validate_cache_value(value: Any, config: Optional["CacheConfig"] = None) -> bool:
    """
    キャッシュ値の妥当性を検証

    Args:
        value: 検証する値
        config: キャッシュ設定（Noneの場合はデフォルト設定を使用）

    Returns:
        値が有効かどうか

    Raises:
        CacheValidationError: 値が無効な場合
    """
    if config is None:
        config = get_cache_config()

    # None値は許可
    if value is None:
        return True

    # サイズチェック
    try:
        size_bytes = sys.getsizeof(value)
        if size_bytes > config.max_value_size_bytes:
            if config.enable_size_warnings:
                logger.warning(
                    f"Cache value size ({size_bytes} bytes) exceeds recommended limit "
                    f"({config.max_value_size_bytes} bytes)"
                )
            # 警告のみで、エラーにはしない（将来的にはエラーにすることも可能）

    except (TypeError, OverflowError, RecursionError) as e:
        # サイズ取得でエラーが発生した場合
        logger.debug(f"Failed to get cache value size: {e}")

    # 循環参照チェック（簡易版）
    try:
        # 簡単な循環参照チェックとして、repr()を試行
        repr(value)
    except RecursionError:
        raise CacheValidationError(
            "Cache value contains circular references",
            "value_circular_reference",
            value
        )

    return True


def is_valid_cache_value(value: Any, config: Optional["CacheConfig"] = None) -> bool:
    """
    キャッシュ値の妥当性を検証（例外を発生させない版）

    Args:
        value: 検証する値
        config: キャッシュ設定（Noneの場合はデフォルト設定を使用）

    Returns:
        値が有効かどうか
    """
    try:
        validate_cache_value(value, config)
        return True
    except CacheValidationError:
        return False


def sanitize_cache_value(value: Any, config: Optional["CacheConfig"] = None) -> Any:
    """
    キャッシュ値のサニタイズ

    Args:
        value: サニタイズする値
        config: キャッシュ設定（Noneの場合はデフォルト設定を使用）

    Returns:
        サニタイズされた値

    Raises:
        CacheValidationError: 値が不正な場合
    """
    if config is None:
        config = get_cache_config()

    # None値はそのまま許可
    if value is None:
        return value

    # 大きすぎるオブジェクトの警告と制限
    try:
        size = sys.getsizeof(value)
        if (
            size > config.max_value_size_bytes
            and config.enable_size_warnings
            and hasattr(logger, "isEnabledFor")
            and logger.isEnabledFor(logger.WARNING)
        ):
            logger.warning(
                f"Large cache value detected: {size} bytes (limit: {config.max_value_size_bytes})"
            )

            # 設定によっては大きすぎる値をエラーとして扱う
            # 現在は警告のみだが、将来的にはCacheErrorを発生させることも可能

    except Exception as e:
        # サイズ取得に失敗した場合はデバッグログに記録
        if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logger.DEBUG):
            logger.debug(f"Failed to get cache value size: {e}")

    return value


def validate_ttl(ttl: Optional[int]) -> bool:
    """
    TTL（Time To Live）値の妥当性を検証

    Args:
        ttl: 検証するTTL値（秒）

    Returns:
        TTL値が有効かどうか

    Raises:
        CacheValidationError: TTL値が無効な場合
    """
    if ttl is None:
        return True

    if not isinstance(ttl, (int, float)):
        raise CacheValidationError(
            f"TTL must be a number, got {type(ttl).__name__}",
            "ttl_type_validation",
            ttl
        )

    if ttl < 0:
        raise CacheValidationError(
            f"TTL cannot be negative, got {ttl}",
            "ttl_negative_validation",
            ttl
        )

    if ttl > 86400 * 365:  # 1年以上
        logger.warning(f"Very long TTL specified: {ttl} seconds ({ttl/86400:.1f} days)")

    return True


def sanitize_ttl(ttl: Optional[int]) -> Optional[int]:
    """
    TTL値をサニタイズ

    Args:
        ttl: サニタイズするTTL値

    Returns:
        サニタイズされたTTL値
    """
    if ttl is None:
        return None

    try:
        # 数値に変換
        sanitized_ttl = int(float(ttl))
        
        # 負の値は0に設定
        if sanitized_ttl < 0:
            logger.warning(f"Negative TTL ({ttl}) converted to 0")
            sanitized_ttl = 0
        
        # 極端に大きな値に警告
        if sanitized_ttl > 86400 * 365:  # 1年
            logger.warning(f"Very large TTL: {sanitized_ttl} seconds")
        
        return sanitized_ttl
    
    except (ValueError, TypeError, OverflowError) as e:
        raise CacheValidationError(
            f"Cannot convert TTL to valid integer: {ttl} ({e})",
            "ttl_conversion",
            ttl
        )


def validate_cache_size(size: int) -> bool:
    """
    キャッシュサイズの妥当性を検証

    Args:
        size: 検証するキャッシュサイズ

    Returns:
        サイズが有効かどうか

    Raises:
        CacheValidationError: サイズが無効な場合
    """
    if not isinstance(size, int):
        raise CacheValidationError(
            f"Cache size must be an integer, got {type(size).__name__}",
            "cache_size_type_validation",
            size
        )

    if size <= 0:
        raise CacheValidationError(
            f"Cache size must be positive, got {size}",
            "cache_size_positive_validation",
            size
        )

    if size > 1000000:  # 100万エントリ以上
        logger.warning(f"Very large cache size specified: {size:,} entries")

    return True


def get_safe_cache_key_preview(key: str, max_length: int = 50) -> str:
    """
    キャッシュキーの安全なプレビューを生成（ログ出力用）

    Args:
        key: プレビューを生成するキー
        max_length: 最大長（デフォルト: 50）

    Returns:
        安全なプレビュー文字列
    """
    if not isinstance(key, str):
        return f"<non-string: {type(key).__name__}>"

    # 制御文字を可視化
    safe_key = ''.join(
        c if ord(c) >= 32 and ord(c) != 127
        else f"\\x{ord(c):02x}"
        for c in key
    )

    # 長さを制限
    if len(safe_key) > max_length:
        safe_key = safe_key[:max_length - 3] + "..."

    return safe_key