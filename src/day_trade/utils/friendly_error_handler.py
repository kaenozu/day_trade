"""
ユーザーフレンドリーなエラーメッセージハンドラー
専門用語を避け、具体的な解決策を提示する
"""

import logging
from typing import Dict, Optional, Tuple, Any
from rich.console import Console
from rich.panel import Panel

from .exceptions import (
    DayTradeError,
    NetworkError,
    DatabaseError,
    ValidationError,
    ConfigurationError,
    RateLimitError,
    AuthenticationError,
    TimeoutError,
)

console = Console()
logger = logging.getLogger(__name__)


class FriendlyErrorHandler:
    """ユーザーフレンドリーなエラーメッセージハンドラー"""

    # エラーコードとメッセージのマッピング
    ERROR_MESSAGES = {
        # ネットワーク関連エラー
        "NETWORK_CONNECTION_ERROR": {
            "title": "接続エラー",
            "message": "インターネット接続を確認してください。",
            "solutions": [
                "Wi-Fi または有線LAN接続を確認してください",
                "ファイアウォールやプロキシ設定を確認してください",
                "しばらく時間をおいて再度お試しください",
            ],
            "emoji": "🌐",
        },
        "NETWORK_TIMEOUT_ERROR": {
            "title": "タイムアウトエラー",
            "message": "サーバーからの応答がありませんでした。",
            "solutions": [
                "インターネット接続が安定しているか確認してください",
                "時間をおいて再度お試しください",
                "市場時間中に実行してください",
            ],
            "emoji": "⏰",
        },
        "API_RATE_LIMIT_ERROR": {
            "title": "アクセス制限エラー",
            "message": "データ取得の頻度が制限に達しました。",
            "solutions": [
                "しばらく時間をおいて再度お試しください（推奨: 1分程度）",
                "短時間での連続実行を避けてください",
                "必要に応じて有料プランへのアップグレードを検討してください",
            ],
            "emoji": "⚠️",
        },
        # データベース関連エラー
        "DB_CONNECTION_ERROR": {
            "title": "データベース接続エラー",
            "message": "データベースに接続できませんでした。",
            "solutions": [
                "アプリケーションを再起動してください",
                "データベースファイルの権限を確認してください",
                "ディスク容量を確認してください",
            ],
            "emoji": "💾",
        },
        "DB_INTEGRITY_ERROR": {
            "title": "データ整合性エラー",
            "message": "データベースの整合性に問題があります。",
            "solutions": [
                "データベースを初期化してください（daytrade init）",
                "バックアップから復元してください",
                "サポートにお問い合わせください",
            ],
            "emoji": "🔧",
        },
        # 入力検証エラー
        "VALIDATION_ERROR": {
            "title": "入力エラー",
            "message": "入力された情報に問題があります。",
            "solutions": [
                "入力形式を確認してください",
                "有効な値の範囲内で入力してください",
                "必須項目が入力されているか確認してください",
            ],
            "emoji": "✏️",
        },
        # 設定エラー
        "CONFIG_ERROR": {
            "title": "設定エラー",
            "message": "アプリケーションの設定に問題があります。",
            "solutions": [
                "設定ファイルの形式を確認してください",
                "設定をデフォルトにリセットしてください（daytrade config reset）",
                "設定項目の値が正しいか確認してください",
            ],
            "emoji": "⚙️",
        },
        # 認証エラー
        "API_AUTH_ERROR": {
            "title": "認証エラー",
            "message": "APIの認証に失敗しました。",
            "solutions": [
                "APIキーが正しく設定されているか確認してください",
                "APIキーの有効期限を確認してください",
                "アカウントがアクティブであることを確認してください",
            ],
            "emoji": "🔐",
        },
    }

    @classmethod
    def create_user_friendly_message(
        cls, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, list]:
        """
        ユーザーフレンドリーなエラーメッセージを作成

        Args:
            error: 例外オブジェクト
            context: 追加のコンテキスト情報

        Returns:
            tuple: (タイトル, メッセージ, 解決策のリスト)
        """
        context = context or {}

        # カスタム例外の場合
        if isinstance(error, DayTradeError):
            error_code = error.error_code or cls._infer_error_code(error)
            error_info = cls.ERROR_MESSAGES.get(error_code, {})

            title = error_info.get("title", "エラー")
            message = error_info.get("message", str(error))
            solutions = error_info.get("solutions", ["サポートにお問い合わせください"])

            # コンテキスト情報を追加
            if context.get("user_input"):
                message = f"入力値 '{context['user_input']}' で{message}"

            return title, message, solutions

        # 一般的な例外の場合
        error_type = type(error).__name__
        return cls._handle_general_exception(error, error_type, context)

    @classmethod
    def _infer_error_code(cls, error: DayTradeError) -> str:
        """例外の種類からエラーコードを推測"""
        if isinstance(error, NetworkError):
            return "NETWORK_CONNECTION_ERROR"
        elif isinstance(error, TimeoutError):
            return "NETWORK_TIMEOUT_ERROR"
        elif isinstance(error, RateLimitError):
            return "API_RATE_LIMIT_ERROR"
        elif isinstance(error, AuthenticationError):
            return "API_AUTH_ERROR"
        elif isinstance(error, DatabaseError):
            return "DB_CONNECTION_ERROR"
        elif isinstance(error, ValidationError):
            return "VALIDATION_ERROR"
        elif isinstance(error, ConfigurationError):
            return "CONFIG_ERROR"
        else:
            return "UNKNOWN_ERROR"

    @classmethod
    def _handle_general_exception(
        cls, error: Exception, error_type: str, context: Dict[str, Any]
    ) -> Tuple[str, str, list]:
        """一般的な例外を処理"""

        # よくある例外のマッピング
        general_mappings = {
            "FileNotFoundError": {
                "title": "ファイルエラー",
                "message": "指定されたファイルが見つかりません。",
                "solutions": [
                    "ファイルパスが正しいか確認してください",
                    "ファイルが存在するか確認してください",
                    "読み取り権限があるか確認してください",
                ],
            },
            "PermissionError": {
                "title": "権限エラー",
                "message": "ファイルまたはディレクトリへのアクセス権限がありません。",
                "solutions": [
                    "管理者権限で実行してください",
                    "ファイルやフォルダの権限を確認してください",
                    "他のプログラムがファイルを使用していないか確認してください",
                ],
            },
            "KeyError": {
                "title": "データエラー",
                "message": "必要なデータが見つかりません。",
                "solutions": [
                    "入力データの形式を確認してください",
                    "最新版のアプリケーションを使用してください",
                    "データを再取得してください",
                ],
            },
            "ValueError": {
                "title": "値エラー",
                "message": "入力された値が正しくありません。",
                "solutions": [
                    "入力値の形式を確認してください",
                    "数値が正しい範囲内か確認してください",
                    "文字列が正しい形式か確認してください",
                ],
            },
        }

        error_info = general_mappings.get(
            error_type,
            {
                "title": "予期しないエラー",
                "message": "システムで予期しないエラーが発生しました。",
                "solutions": [
                    "アプリケーションを再起動してください",
                    "最新版にアップデートしてください",
                    "サポートにお問い合わせください",
                ],
            },
        )

        return error_info["title"], error_info["message"], error_info["solutions"]

    @classmethod
    def format_error_panel(
        cls,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        show_technical_details: bool = False,
    ) -> Panel:
        """
        Rich形式のエラーパネルを作成

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト
            show_technical_details: 技術的詳細を表示するか

        Returns:
            Rich Panel オブジェクト
        """
        title, message, solutions = cls.create_user_friendly_message(error, context)

        # エラーコードに基づく絵文字の取得
        emoji = "❌"
        if isinstance(error, DayTradeError) and error.error_code:
            error_info = cls.ERROR_MESSAGES.get(error.error_code, {})
            emoji = error_info.get("emoji", "❌")

        # パネル内容を構築
        content_lines = [
            f"[bold red]{emoji} {message}[/bold red]",
            "",
            "[bold yellow]💡 解決方法:[/bold yellow]",
        ]

        for i, solution in enumerate(solutions, 1):
            content_lines.append(f"  {i}. {solution}")

        # 技術的詳細を表示する場合
        if show_technical_details:
            content_lines.extend(
                [
                    "",
                    "[dim]技術的詳細:[/dim]",
                    f"[dim]エラータイプ: {type(error).__name__}[/dim]",
                    f"[dim]詳細: {str(error)}[/dim]",
                ]
            )

            if isinstance(error, DayTradeError) and error.details:
                content_lines.append(f"[dim]追加情報: {error.details}[/dim]")

        content = "\n".join(content_lines)

        return Panel(
            content,
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            padding=(1, 2),
        )

    @classmethod
    def log_error_for_debugging(
        cls, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        デバッグ用にエラーの詳細をログに記録

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト
        """
        logger.error(
            "Error occurred: %s",
            type(error).__name__,
            exc_info=True,
            extra={
                "error_message": str(error),
                "context": context or {},
                "error_type": type(error).__name__,
            },
        )

    @classmethod
    def handle_cli_error(
        cls,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        show_technical: bool = False,
    ) -> None:
        """
        CLI用のエラーハンドリング（表示とログ）

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト
            show_technical: 技術的詳細を表示するか
        """
        # ユーザーフレンドリーなエラーパネルを表示
        error_panel = cls.format_error_panel(error, context, show_technical)
        console.print(error_panel)

        # デバッグ用ログ記録
        cls.log_error_for_debugging(error, context)


# 便利な関数として公開
def create_friendly_error_panel(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    show_technical: bool = False,
) -> Panel:
    """ユーザーフレンドリーなエラーパネルを作成する便利関数"""
    return FriendlyErrorHandler.format_error_panel(error, context, show_technical)


def handle_cli_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    show_technical: bool = False,
) -> None:
    """CLI用エラーハンドリングの便利関数"""
    FriendlyErrorHandler.handle_cli_error(error, context, show_technical)
