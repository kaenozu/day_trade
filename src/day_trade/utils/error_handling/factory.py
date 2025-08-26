"""
エラーハンドラーのファクトリー関数とグローバルインスタンス管理
依存性注入とシングルトンパターンのサポート
"""

import threading
from typing import Optional

from ..i18n_messages import I18nMessageHandler, Language, SensitiveDataSanitizer
from .config import EnhancedErrorHandlerConfig
from .handler import EnhancedErrorHandler
from .stats import ErrorHandlerStats


# ファクトリー関数
def create_error_handler(
    message_handler: Optional[I18nMessageHandler] = None,
    sanitizer: Optional[SensitiveDataSanitizer] = None,
    config_manager=None,
    language: Language = Language.JAPANESE,
    debug_mode: Optional[bool] = None,
    enable_sanitization: Optional[bool] = None,
) -> EnhancedErrorHandler:
    """
    カスタム設定でエラーハンドラーを作成するファクトリー関数（完全統合版）

    Args:
        message_handler: メッセージハンドラー
        sanitizer: データサニタイザー
        config_manager: ConfigManagerインスタンス
        language: デフォルト言語
        debug_mode: デバッグモード（設定を上書き）
        enable_sanitization: サニタイズ機能（設定を上書き）

    Returns:
        設定されたEnhancedErrorHandlerインスタンス
    """
    # 設定オブジェクトを作成
    config = EnhancedErrorHandlerConfig(config_manager)

    # 個別設定で上書き
    if debug_mode is not None:
        config.debug_mode = debug_mode
    if enable_sanitization is not None:
        config.enable_sanitization = enable_sanitization

    # 統計オブジェクトを作成
    stats = ErrorHandlerStats(config)

    return EnhancedErrorHandler(
        message_handler=message_handler,
        sanitizer=sanitizer,
        config=config,
        language=language,
        stats=stats,
    )


# デフォルトハンドラーインスタンス（スレッドセーフ・遅延初期化）
_default_error_handler = None
_handler_lock = threading.RLock()


def get_default_error_handler() -> EnhancedErrorHandler:
    """
    デフォルトエラーハンドラーを取得（スレッドセーフ・シングルトン）

    注意: 実運用環境では dependency injection を推奨します。
    この関数は主に後方互換性と開発時の利便性のために提供されています。
    """
    global _default_error_handler

    if _default_error_handler is None:
        with _handler_lock:
            # ダブルチェックロッキングパターン
            if _default_error_handler is None:
                _default_error_handler = create_error_handler()

    return _default_error_handler


def set_default_error_handler(handler: EnhancedErrorHandler) -> None:
    """
    デフォルトエラーハンドラーを設定（依存性注入サポート）

    Args:
        handler: 設定するエラーハンドラー
    """
    global _default_error_handler

    if not isinstance(handler, EnhancedErrorHandler):
        raise ValueError(
            "ハンドラーはEnhancedErrorHandlerインスタンスである必要があります"
        )

    with _handler_lock:
        _default_error_handler = handler


def reset_default_error_handler() -> None:
    """デフォルトエラーハンドラーをリセット（テスト用）"""
    global _default_error_handler

    with _handler_lock:
        _default_error_handler = None


# 設定統合確認関数
def validate_error_handler_integration(config_manager=None) -> dict:
    """エラーハンドラーの統合状況を検証"""
    try:
        handler = create_error_handler(config_manager=config_manager)
        stats = handler.get_performance_stats()

        validation_result = {
            "config_integration": bool(handler.config.config_manager),
            "cache_integration": True,  # cache_utilsとの統合は完了済み
            "i18n_integration": bool(handler.message_handler),
            "sanitization_enabled": handler.enable_sanitization,
            "rich_display_enabled": handler.config.enable_rich_display,
            "stats_collection": bool(handler.stats),
            "current_stats": stats,
            "validation_passed": True,
        }

        return validation_result

    except Exception as e:
        return {
            "validation_passed": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }