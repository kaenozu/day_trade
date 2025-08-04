"""
統合されたユーザーフレンドリーエラーハンドリングシステム
多言語対応、コンテキスト情報、解決策提示を統合
機密情報のサニタイズとセキュリティ強化を含む
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel

from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    DatabaseError,
    DayTradeError,
    NetworkError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)
from .i18n_messages import I18nMessageHandler, Language

console = Console()
logger = logging.getLogger(__name__)


class SensitiveDataSanitizer:
    """機密情報のサニタイズを行うユーティリティクラス"""

    # 機密情報を表す可能性のあるキーワード（小文字）
    SENSITIVE_KEYWORDS = {
        "password", "passwd", "pwd", "secret", "token", "key", "api_key",
        "apikey", "access_token", "refresh_token", "auth", "authorization",
        "credential", "private", "pin", "ssn", "social", "credit_card",
        "card_number", "cvv", "ccv", "cvc", "bank", "account_number"
    }

    # 機密情報パターン（正規表現）
    SENSITIVE_PATTERNS = [
        # API キー形式
        (re.compile(r'\b[A-Za-z0-9]{32,}\b'), "***API_KEY***"),
        # パスワード形式（8文字以上の英数字記号）
        (re.compile(r'\b(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*?&]{8,}\b'), "***PASSWORD***"),
        # JWT トークン形式
        (re.compile(r'\beyJ[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=_-]+\b'), "***JWT_TOKEN***"),
        # クレジットカード番号形式
        (re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'), "***CARD_NUMBER***"),
        # 電話番号形式（日本）
        (re.compile(r'\b0\d{1,4}-?\d{1,4}-?\d{4}\b'), "***PHONE_NUMBER***"),
    ]

    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], max_depth: int = 5) -> Dict[str, Any]:
        """
        辞書内の機密情報をサニタイズ

        Args:
            data: サニタイズ対象の辞書
            max_depth: 再帰処理の最大深度

        Returns:
            サニタイズされた辞書
        """
        if max_depth <= 0:
            return {"sanitized": "max_depth_reached"}

        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()

            # キー名が機密情報を示している場合
            if any(keyword in key_lower for keyword in cls.SENSITIVE_KEYWORDS):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value, max_depth - 1)
            elif isinstance(value, list):
                sanitized[key] = cls.sanitize_list(value, max_depth - 1)
            elif isinstance(value, str):
                sanitized[key] = cls.sanitize_string(value)
            else:
                sanitized[key] = value

        return sanitized

    @classmethod
    def sanitize_list(cls, data: List[Any], max_depth: int = 5) -> List[Any]:
        """
        リスト内の機密情報をサニタイズ

        Args:
            data: サニタイズ対象のリスト
            max_depth: 再帰処理の最大深度

        Returns:
            サニタイズされたリスト
        """
        if max_depth <= 0:
            return ["sanitized: max_depth_reached"]

        sanitized = []
        for item in data:
            if isinstance(item, dict):
                sanitized.append(cls.sanitize_dict(item, max_depth - 1))
            elif isinstance(item, list):
                sanitized.append(cls.sanitize_list(item, max_depth - 1))
            elif isinstance(item, str):
                sanitized.append(cls.sanitize_string(item))
            else:
                sanitized.append(item)

        return sanitized

    @classmethod
    def sanitize_string(cls, text: str) -> str:
        """
        文字列内の機密情報をサニタイズ

        Args:
            text: サニタイズ対象の文字列

        Returns:
            サニタイズされた文字列
        """
        if not isinstance(text, str):
            return text

        sanitized_text = text
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            sanitized_text = pattern.sub(replacement, sanitized_text)

        return sanitized_text

    @classmethod
    def sanitize_context(cls, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        コンテキスト全体をサニタイズ

        Args:
            context: サニタイズ対象のコンテキスト

        Returns:
            サニタイズされたコンテキスト
        """
        if not context:
            return {}

        return cls.sanitize_dict(context)


class EnhancedErrorHandler:
    """拡張されたエラーハンドリングシステム（依存性注入対応）"""

    # friendly_error_handlerから統合されたエラーメッセージマッピング
    BUILTIN_ERROR_MESSAGES = {
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

    def __init__(
        self,
        message_handler: Optional[I18nMessageHandler] = None,
        sanitizer: Optional[SensitiveDataSanitizer] = None,
        language: Language = Language.JAPANESE,
        debug_mode: bool = False,
        enable_sanitization: bool = True,
    ):
        """
        Args:
            message_handler: メッセージハンドラー（依存性注入）
            sanitizer: データサニタイザー（依存性注入）
            language: デフォルト言語
            debug_mode: デバッグモード（技術的詳細を表示）
            enable_sanitization: サニタイズ機能を有効にするか
        """
        self.language = language
        self.debug_mode = debug_mode
        self.enable_sanitization = enable_sanitization

        # 依存性注入 - なければデフォルトインスタンスを作成
        self.message_handler = message_handler or I18nMessageHandler(language)
        self.sanitizer = sanitizer or SensitiveDataSanitizer()

    def _infer_error_code(self, error: DayTradeError) -> str:
        """
        例外の種類からエラーコードを推測（friendly_error_handlerから統合）

        Args:
            error: カスタム例外オブジェクト

        Returns:
            推測されたエラーコード
        """
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

    def _get_message_with_fallback(self, error_code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        メッセージハンドラーから取得し、失敗時はビルトインメッセージにフォールバック

        Args:
            error_code: エラーコード
            context: コンテキスト情報

        Returns:
            メッセージ辞書
        """
        try:
            # まずi18nメッセージハンドラーから取得を試行
            message_data = self.message_handler.get_message(error_code, context=context)

            # 必要なキーが存在するかチェック
            required_keys = ["title", "message", "solutions"]
            if all(key in message_data for key in required_keys):
                return message_data

        except Exception as e:
            logger.warning(f"I18nMessageHandlerからの取得に失敗: {e}")

        # フォールバック: ビルトインメッセージを使用
        builtin_message = self.BUILTIN_ERROR_MESSAGES.get(error_code)
        if builtin_message:
            return builtin_message.copy()

        # 最終フォールバック: デフォルトメッセージ
        return {
            "title": "エラー",
            "message": "予期しないエラーが発生しました。",
            "solutions": [
                "アプリケーションを再起動してください",
                "最新版にアップデートしてください",
                "サポートにお問い合わせください"
            ],
            "emoji": "❌"
        }

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

        # コンテキストのサニタイズ（セキュリティ強化）
        if self.enable_sanitization:
            context = self.sanitizer.sanitize_context(context)

        # カスタム例外の場合
        if isinstance(error, DayTradeError):
            return self._handle_custom_error(error, context, show_tech)

        # 一般的な例外の場合
        return self._handle_general_error(error, context, show_tech)

    def _handle_custom_error(
        self, error: DayTradeError, context: Dict[str, Any], show_technical: bool
    ) -> Panel:
        """カスタム例外を処理（friendly_error_handlerの機能を統合）"""

        # エラーコードの取得・推論
        error_code = error.error_code or self._infer_error_code(error)

        # メッセージデータをフォールバック付きで取得
        message_data = self._get_message_with_fallback(error_code, context)

        # コンテキスト情報を追加（user_inputがある場合）
        message = message_data["message"]
        if context.get("user_input"):
            message = f"入力値 '{context['user_input']}' で{message}"

        return self._create_enhanced_panel(
            error=error,
            title=message_data["title"],
            message=message,
            solutions=message_data["solutions"],
            emoji=message_data.get("emoji", "❌"),
            context=context,
            show_technical=show_technical,
        )

    def _handle_general_error(
        self, error: Exception, context: Dict[str, Any], show_technical: bool
    ) -> Panel:
        """一般的な例外を処理（friendly_error_handlerの機能を統合）"""

        # よくある例外のマッピング（friendly_error_handlerから統合）
        general_mappings = {
            "FileNotFoundError": {
                "title": "ファイルエラー",
                "message": "指定されたファイルが見つかりません。",
                "solutions": [
                    "ファイルパスが正しいか確認してください",
                    "ファイルが存在するか確認してください",
                    "読み取り権限があるか確認してください",
                ],
                "emoji": "📁"
            },
            "PermissionError": {
                "title": "権限エラー",
                "message": "ファイルまたはディレクトリへのアクセス権限がありません。",
                "solutions": [
                    "管理者権限で実行してください",
                    "ファイルやフォルダの権限を確認してください",
                    "他のプログラムがファイルを使用していないか確認してください",
                ],
                "emoji": "🔒"
            },
            "KeyError": {
                "title": "データエラー",
                "message": "必要なデータが見つかりません。",
                "solutions": [
                    "入力データの形式を確認してください",
                    "最新版のアプリケーションを使用してください",
                    "データを再取得してください",
                ],
                "emoji": "🔑"
            },
            "ValueError": {
                "title": "値エラー",
                "message": "入力された値が正しくありません。",
                "solutions": [
                    "入力値の形式を確認してください",
                    "数値が正しい範囲内か確認してください",
                    "文字列が正しい形式か確認してください",
                ],
                "emoji": "⚠️"
            },
        }

        error_type = type(error).__name__

        # ビルトインマッピングから取得
        builtin_error_info = general_mappings.get(error_type)

        if builtin_error_info:
            message_data = builtin_error_info.copy()
        else:
            # i18nメッセージハンドラーで試行
            try:
                message_data = self.message_handler.get_message_for_exception(
                    error, context=context
                )
            except Exception:
                # 最終フォールバック
                message_data = {
                    "title": "予期しないエラー",
                    "message": "システムで予期しないエラーが発生しました。",
                    "solutions": [
                        "アプリケーションを再起動してください",
                        "最新版にアップデートしてください",
                        "サポートにお問い合わせください",
                    ],
                    "emoji": "❌"
                }

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
        エラーの詳細をログに記録（機密情報サニタイズ付き）

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト
            user_action: ユーザーアクション
        """
        # コンテキストのサニタイズ
        sanitized_context = {}
        if context:
            sanitized_context = (
                self.sanitizer.sanitize_context(context)
                if self.enable_sanitization
                else context
            )

        # ユーザーアクションのサニタイズ
        sanitized_user_action = user_action
        if user_action and self.enable_sanitization:
            sanitized_user_action = self.sanitizer.sanitize_string(user_action)

        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": sanitized_context,
            "user_action": sanitized_user_action,
        }

        if isinstance(error, DayTradeError):
            log_data["error_code"] = error.error_code

            # error.detailsもサニタイズ
            error_details = error.details or {}
            if self.enable_sanitization:
                log_data["error_details"] = self.sanitizer.sanitize_dict(error_details)
            else:
                log_data["error_details"] = error_details

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


# デフォルトハンドラーインスタンス（下位互換性のため）
default_error_handler = EnhancedErrorHandler()


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
    error_handler = handler or default_error_handler

    # 言語設定が違う場合は新しいハンドラーを作成
    if error_handler.language != language:
        error_handler = EnhancedErrorHandler(language=language, debug_mode=show_technical)

    error_handler.display_and_log_error(error, context, user_action, show_technical)


def create_error_handler(
    message_handler: Optional[I18nMessageHandler] = None,
    sanitizer: Optional[SensitiveDataSanitizer] = None,
    language: Language = Language.JAPANESE,
    debug_mode: bool = False,
    enable_sanitization: bool = True,
) -> EnhancedErrorHandler:
    """
    カスタム設定でエラーハンドラーを作成するファクトリー関数

    Args:
        message_handler: メッセージハンドラー
        sanitizer: データサニタイザー
        language: デフォルト言語
        debug_mode: デバッグモード
        enable_sanitization: サニタイズ機能を有効にするか

    Returns:
        設定されたEnhancedErrorHandlerインスタンス
    """
    return EnhancedErrorHandler(
        message_handler=message_handler,
        sanitizer=sanitizer,
        language=language,
        debug_mode=debug_mode,
        enable_sanitization=enable_sanitization,
    )


# 後方互換性のためのエイリアス（friendly_error_handlerの機能）
def create_user_friendly_message(
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[str, str, List[str]]:
    """
    friendly_error_handlerとの後方互換性のための関数

    Args:
        error: 例外オブジェクト
        context: コンテキスト情報

    Returns:
        (タイトル, メッセージ, 解決策のリスト)
    """
    handler = EnhancedErrorHandler()

    if isinstance(error, DayTradeError):
        error_code = error.error_code or handler._infer_error_code(error)
        message_data = handler._get_message_with_fallback(error_code, context or {})
    else:
        # 一般的な例外の処理は_handle_general_errorのロジックを使用
        general_mappings = {
            "FileNotFoundError": ("ファイルエラー", "指定されたファイルが見つかりません。",
                                ["ファイルパスが正しいか確認してください"]),
            "PermissionError": ("権限エラー", "アクセス権限がありません。",
                              ["管理者権限で実行してください"]),
            "KeyError": ("データエラー", "必要なデータが見つかりません。",
                        ["入力データの形式を確認してください"]),
            "ValueError": ("値エラー", "入力された値が正しくありません。",
                          ["入力値の形式を確認してください"]),
        }

        error_type = type(error).__name__
        if error_type in general_mappings:
            title, message, solutions = general_mappings[error_type]
            return title, message, solutions
        else:
            return "予期しないエラー", "システムで予期しないエラーが発生しました。", ["サポートにお問い合わせください"]

    message = message_data["message"]
    if context and context.get("user_input"):
        message = f"入力値 '{context['user_input']}' で{message}"

    return message_data["title"], message, message_data["solutions"]


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
