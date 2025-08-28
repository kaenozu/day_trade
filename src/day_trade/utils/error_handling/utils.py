"""
エラーハンドリングユーティリティ関数群
便利関数と後方互換性のためのエイリアス
CLI専用のエラーハンドラー関数
"""

from typing import Any, Dict, List, Optional, Tuple

from rich.panel import Panel

from ..i18n_messages import Language
from .factory import create_error_handler, get_default_error_handler
from .handler import EnhancedErrorHandler


# 便利関数（依存性注入対応）
def handle_cli_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_action: Optional[str] = None,
    show_technical: bool = False,
    language: Language = Language.JAPANESE,
    handler: Optional[EnhancedErrorHandler] = None,
) -> None:
    """
    CLI用エラーハンドリングの便利関数（依存性注入対応）

    Args:
        error: 例外オブジェクト
        context: 追加コンテキスト
        user_action: ユーザーアクション
        show_technical: 技術的詳細を表示するか
        language: 言語設定
        handler: カスタムエラーハンドラー（指定されなければデフォルトを使用）
    """
    # カスタムハンドラーが指定されていない場合はデフォルトを使用
    error_handler = handler or get_default_error_handler()

    # 言語設定が違う場合は新しいハンドラーを作成
    if error_handler.language != language:
        error_handler = create_error_handler(
            language=language, debug_mode=show_technical
        )

    error_handler.display_and_log_error(error, context, user_action, show_technical)


# 後方互換性のためのエイリアス（friendly_error_handlerの機能）
def create_user_friendly_message(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> Tuple[str, str, List[str]]:
    """
    friendly_error_handlerとの後方互換性のための関数

    Args:
        error: 例外オブジェクト
        context: コンテキスト情報

    Returns:
        (タイトル, メッセージ, 解決策のリスト)
    """
    from ..exceptions import DayTradeError

    handler = get_default_error_handler()

    if isinstance(error, DayTradeError):
        error_code = error.error_code or handler._infer_error_code(error)
        message_data = handler._get_message_with_fallback(error_code, context or {})
    else:
        # 一般的な例外の処理は_handle_general_errorのロジックを使用
        general_mappings = {
            "FileNotFoundError": (
                "ファイルエラー",
                "指定されたファイルが見つかりません。",
                ["ファイルパスが正しいか確認してください"],
            ),
            "PermissionError": (
                "権限エラー",
                "アクセス権限がありません。",
                ["管理者権限で実行してください"],
            ),
            "KeyError": (
                "データエラー",
                "必要なデータが見つかりません。",
                ["入力データの形式を確認してください"],
            ),
            "ValueError": (
                "値エラー",
                "入力された値が正しくありません。",
                ["入力値の形式を確認してください"],
            ),
        }

        error_type = type(error).__name__
        if error_type in general_mappings:
            title, message, solutions = general_mappings[error_type]
            return title, message, solutions
        else:
            return (
                "予期しないエラー",
                "システムで予期しないエラーが発生しました。",
                ["サポートにお問い合わせください"],
            )

    message = message_data["message"]
    if context and context.get("user_input"):
        message = f"入力値 '{context['user_input']}' で{message}"

    return message_data["title"], message, message_data["solutions"]


# friendly_error_handlerから統合された便利関数
def create_friendly_error_panel(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    show_technical: bool = False,
) -> Panel:
    """
    ユーザーフレンドリーなエラーパネルを作成する便利関数
    friendly_error_handlerとの後方互換性のため
    """
    handler = create_error_handler(debug_mode=show_technical)
    return handler.handle_error(error, context, show_technical=show_technical)


def log_error_for_debugging(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> None:
    """
    デバッグ用にエラーの詳細をログに記録
    friendly_error_handlerとの後方互換性のため
    """
    handler = get_default_error_handler()
    handler.log_error(error, context)


# CLIで使いやすいラッパー関数
def handle_stock_fetch_error(
    error: Exception, stock_code: Optional[str] = None, show_technical: bool = False
) -> None:
    """株価取得エラーの専用ハンドラー"""
    context = {"user_input": stock_code} if stock_code else {}
    handle_cli_error(
        error=error,
        context=context,
        user_action="株価情報の取得",
        show_technical=show_technical,
    )


def handle_database_error(
    error: Exception, operation: Optional[str] = None, show_technical: bool = False
) -> None:
    """データベースエラーの専用ハンドラー"""
    context = {"operation": operation} if operation else {}
    handle_cli_error(
        error=error,
        context=context,
        user_action="データベース操作",
        show_technical=show_technical,
    )


def handle_config_error(
    error: Exception,
    config_key: Optional[str] = None,
    config_value: Optional[str] = None,
    show_technical: bool = False,
) -> None:
    """設定エラーの専用ハンドラー"""
    context = {}
    if config_key:
        context["config_key"] = config_key
    if config_value:
        context["user_input"] = config_value

    handle_cli_error(
        error=error,
        context=context,
        user_action="設定の変更",
        show_technical=show_technical,
    )


# パフォーマンス統計取得関数
def get_error_handler_performance_stats(
    handler: Optional[EnhancedErrorHandler] = None,
) -> Dict[str, Any]:
    """エラーハンドラーのパフォーマンス統計を取得"""
    error_handler = handler or get_default_error_handler()
    return error_handler.get_performance_stats()