"""
統合されたユーザーフレンドリーエラーハンドリングシステム
多言語対応、コンテキスト情報、解決策提示を統合
"""

import logging
from typing import Dict, Optional, Any
from rich.console import Console
from rich.panel import Panel

from .exceptions import DayTradeError
from .i18n_messages import Language, I18nMessageHandler

console = Console()
logger = logging.getLogger(__name__)


class EnhancedErrorHandler:
    """拡張されたエラーハンドリングシステム"""

    def __init__(
        self, language: Language = Language.JAPANESE, debug_mode: bool = False
    ):
        """
        Args:
            language: デフォルト言語
            debug_mode: デバッグモード（技術的詳細を表示）
        """
        self.language = language
        self.debug_mode = debug_mode
        self.message_handler = I18nMessageHandler(language)

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None,
        show_technical: Optional[bool] = None,
    ) -> Panel:
        """
        エラーを包括的に処理してユーザーフレンドリーなパネルを作成

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト情報
            user_action: ユーザーが実行していたアクション
            show_technical: 技術的詳細を表示するか

        Returns:
            Rich Panel オブジェクト
        """
        show_tech = show_technical if show_technical is not None else self.debug_mode
        context = context or {}

        # ユーザーアクションをコンテキストに追加
        if user_action:
            context["user_action"] = user_action

        # カスタム例外の場合
        if isinstance(error, DayTradeError):
            return self._handle_custom_error(error, context, show_tech)

        # 一般的な例外の場合
        return self._handle_general_error(error, context, show_tech)

    def _handle_custom_error(
        self, error: DayTradeError, context: Dict[str, Any], show_technical: bool
    ) -> Panel:
        """カスタム例外を処理"""

        error_code = error.error_code or "UNKNOWN_ERROR"
        message_data = self.message_handler.get_message(error_code, context=context)

        return self._create_enhanced_panel(
            error=error,
            title=message_data["title"],
            message=message_data["message"],
            solutions=message_data["solutions"],
            emoji=message_data.get("emoji", "❌"),
            context=context,
            show_technical=show_technical,
        )

    def _handle_general_error(
        self, error: Exception, context: Dict[str, Any], show_technical: bool
    ) -> Panel:
        """一般的な例外を処理"""

        message_data = self.message_handler.get_message_for_exception(
            error, context=context
        )

        return self._create_enhanced_panel(
            error=error,
            title=message_data["title"],
            message=message_data["message"],
            solutions=message_data["solutions"],
            emoji=message_data.get("emoji", "❌"),
            context=context,
            show_technical=show_technical,
        )

    def _create_enhanced_panel(
        self,
        error: Exception,
        title: str,
        message: str,
        solutions: list,
        emoji: str,
        context: Dict[str, Any],
        show_technical: bool,
    ) -> Panel:
        """拡張されたエラーパネルを作成"""

        content_lines = []

        # メインメッセージ
        content_lines.append(f"[bold red]{emoji} {message}[/bold red]")

        # ユーザーアクションコンテキスト
        if context.get("user_action"):
            content_lines.append(
                f"[dim]実行中のアクション: {context['user_action']}[/dim]"
            )

        # 追加コンテキスト情報
        if context.get("user_input"):
            content_lines.append(f"[dim]入力値: {context['user_input']}[/dim]")

        content_lines.append("")

        # 解決策
        content_lines.append("[bold yellow]💡 解決方法:[/bold yellow]")
        for i, solution in enumerate(solutions, 1):
            content_lines.append(f"  {i}. {solution}")

        # 技術的詳細（デバッグモード時）
        if show_technical:
            content_lines.extend(
                [
                    "",
                    "[dim]── 技術的詳細 ──[/dim]",
                    f"[dim]エラータイプ: {type(error).__name__}[/dim]",
                    f"[dim]メッセージ: {str(error)}[/dim]",
                ]
            )

            if isinstance(error, DayTradeError):
                if error.error_code:
                    content_lines.append(f"[dim]エラーコード: {error.error_code}[/dim]")
                if error.details:
                    content_lines.append(f"[dim]詳細情報: {error.details}[/dim]")

        # ヘルプメッセージ
        content_lines.extend(
            [
                "",
                "[dim]💬 さらにサポートが必要な場合は、上記の技術的詳細と共にお問い合わせください。[/dim]",
            ]
        )

        content = "\n".join(content_lines)

        return Panel(
            content,
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            padding=(1, 2),
        )

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None,
    ) -> None:
        """
        エラーの詳細をログに記録

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト
            user_action: ユーザーアクション
        """
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "user_action": user_action,
        }

        if isinstance(error, DayTradeError):
            log_data["error_code"] = error.error_code
            log_data["error_details"] = error.details

        logger.error("User-facing error occurred", exc_info=True, extra=log_data)

    def display_and_log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None,
        show_technical: Optional[bool] = None,
    ) -> None:
        """
        エラーを表示してログに記録

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト
            user_action: ユーザーアクション
            show_technical: 技術的詳細を表示するか
        """
        # パネルを作成して表示
        error_panel = self.handle_error(error, context, user_action, show_technical)
        console.print(error_panel)

        # ログに記録
        self.log_error(error, context, user_action)


# デフォルトハンドラーインスタンス
default_error_handler = EnhancedErrorHandler()


# 便利関数
def handle_cli_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_action: Optional[str] = None,
    show_technical: bool = False,
    language: Language = Language.JAPANESE,
) -> None:
    """
    CLI用エラーハンドリングの便利関数

    Args:
        error: 例外オブジェクト
        context: 追加コンテキスト
        user_action: ユーザーアクション
        show_technical: 技術的詳細を表示するか
        language: 言語
    """
    if language != default_error_handler.language:
        # 一時的にハンドラーを作成
        handler = EnhancedErrorHandler(language, show_technical)
        handler.display_and_log_error(error, context, user_action, show_technical)
    else:
        default_error_handler.display_and_log_error(
            error, context, user_action, show_technical
        )


def create_friendly_error_panel(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_action: Optional[str] = None,
    show_technical: bool = False,
    language: Language = Language.JAPANESE,
) -> Panel:
    """
    ユーザーフレンドリーなエラーパネル作成の便利関数

    Args:
        error: 例外オブジェクト
        context: 追加コンテキスト
        user_action: ユーザーアクション
        show_technical: 技術的詳細を表示するか
        language: 言語

    Returns:
        Rich Panel オブジェクト
    """
    if language != default_error_handler.language:
        handler = EnhancedErrorHandler(language, show_technical)
        return handler.handle_error(error, context, user_action, show_technical)
    else:
        return default_error_handler.handle_error(
            error, context, user_action, show_technical
        )


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
