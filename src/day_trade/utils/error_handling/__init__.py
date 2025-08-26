"""
エラーハンドリングパッケージ
enhanced_error_handler.pyの機能をモジュール化し、後方互換性を保持

分割されたモジュール:
- config: 設定クラス（EnhancedErrorHandlerConfig）
- stats: 統計情報クラス（ErrorHandlerStats）
- handler: メインハンドラークラス（EnhancedErrorHandler）
- factory: ファクトリー関数とグローバルインスタンス管理
- utils: 便利関数とCLI専用ハンドラー
"""

# メインクラスとファクトリー関数のエクスポート
from .config import EnhancedErrorHandlerConfig
from .factory import (
    create_error_handler,
    get_default_error_handler,
    reset_default_error_handler,
    set_default_error_handler,
    validate_error_handler_integration,
)
from .handler import EnhancedErrorHandler
from .stats import ErrorHandlerStats
from .utils import (
    create_friendly_error_panel,
    create_user_friendly_message,
    get_error_handler_performance_stats,
    handle_cli_error,
    handle_config_error,
    handle_database_error,
    handle_stock_fetch_error,
    log_error_for_debugging,
)

# 後方互換性のため、元のファイルで直接インポートされていたものをすべてエクスポート
__all__ = [
    # メインクラス
    "EnhancedErrorHandler",
    "EnhancedErrorHandlerConfig",
    "ErrorHandlerStats",
    # ファクトリー関数
    "create_error_handler",
    "get_default_error_handler",
    "set_default_error_handler",
    "reset_default_error_handler",
    # 便利関数
    "handle_cli_error",
    "create_user_friendly_message",
    "create_friendly_error_panel",
    "log_error_for_debugging",
    # 専用ハンドラー
    "handle_stock_fetch_error",
    "handle_database_error",
    "handle_config_error",
    # 統計・検証関数
    "get_error_handler_performance_stats",
    "validate_error_handler_integration",
]