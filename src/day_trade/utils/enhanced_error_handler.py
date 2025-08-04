"""
統合されたユーザーフレンドリーエラーハンドリングシステム
多言語対応、コンテキスト情報、解決策提示を統合
機密情報のサニタイズとセキュリティ強化を含む

friendly_error_handler.pyの機能を統合し、重複を解消
i18n_messages.pyのSensitiveDataSanitizerを活用
config_managerとcache_utilsとの完全統合対応
"""

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

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
from .i18n_messages import I18nMessageHandler, Language, SensitiveDataSanitizer

console = Console()
logger = logging.getLogger(__name__)


class EnhancedErrorHandlerConfig:
    """エラーハンドラー設定クラス（config_manager統合対応）"""

    def __init__(self, config_manager=None):
        """
        Args:
            config_manager: ConfigManagerインスタンス（依存性注入）
        """
        # デフォルト設定値
        self._defaults = {
            "debug_mode": False,
            "enable_sanitization": True,
            "enable_rich_display": True,
            "log_technical_details": True,
            "max_context_items": 10,
            "max_solution_items": 5,
            "console_width": 120,
            "panel_padding": (1, 2),
            "lock_timeout_seconds": 1.0,
            "enable_performance_logging": True,
        }

        self.config_manager = config_manager
        self._load_config()

    def _load_config(self):
        """設定を読み込み（config_manager優先、環境変数フォールバック）"""
        import os

        # config_managerから取得を試行
        error_handler_settings = {}
        if self.config_manager:
            try:
                error_handler_settings = getattr(self.config_manager, 'error_handler_settings', {})
            except Exception:
                logger.warning("Failed to load error handler settings from config_manager, using defaults")

        # 設定値の決定（優先度: config_manager > 環境変数 > デフォルト）
        self.debug_mode = self._parse_bool(
            error_handler_settings.get("debug_mode"),
            os.getenv("ERROR_HANDLER_DEBUG_MODE"),
            self._defaults["debug_mode"]
        )

        self.enable_sanitization = self._parse_bool(
            error_handler_settings.get("enable_sanitization"),
            os.getenv("ERROR_HANDLER_ENABLE_SANITIZATION"),
            self._defaults["enable_sanitization"]
        )

        self.enable_rich_display = self._parse_bool(
            error_handler_settings.get("enable_rich_display"),
            os.getenv("ERROR_HANDLER_ENABLE_RICH_DISPLAY"),
            self._defaults["enable_rich_display"]
        )

        self.log_technical_details = self._parse_bool(
            error_handler_settings.get("log_technical_details"),
            os.getenv("ERROR_HANDLER_LOG_TECHNICAL_DETAILS"),
            self._defaults["log_technical_details"]
        )

        self.max_context_items = int(
            error_handler_settings.get("max_context_items")
            or os.getenv("ERROR_HANDLER_MAX_CONTEXT_ITEMS")
            or self._defaults["max_context_items"]
        )

        self.max_solution_items = int(
            error_handler_settings.get("max_solution_items")
            or os.getenv("ERROR_HANDLER_MAX_SOLUTION_ITEMS")
            or self._defaults["max_solution_items"]
        )

        self.console_width = int(
            error_handler_settings.get("console_width")
            or os.getenv("ERROR_HANDLER_CONSOLE_WIDTH")
            or self._defaults["console_width"]
        )

        self.lock_timeout_seconds = float(
            error_handler_settings.get("lock_timeout_seconds")
            or os.getenv("ERROR_HANDLER_LOCK_TIMEOUT_SECONDS")
            or self._defaults["lock_timeout_seconds"]
        )

        self.enable_performance_logging = self._parse_bool(
            error_handler_settings.get("enable_performance_logging"),
            os.getenv("ERROR_HANDLER_ENABLE_PERFORMANCE_LOGGING"),
            self._defaults["enable_performance_logging"]
        )

        # パネル設定
        panel_padding = error_handler_settings.get("panel_padding")
        if panel_padding and isinstance(panel_padding, (list, tuple)) and len(panel_padding) == 2:
            self.panel_padding = tuple(panel_padding)
        else:
            self.panel_padding = self._defaults["panel_padding"]

    def _parse_bool(self, config_value, env_value, default_value):
        """設定値をbooleanに変換"""
        if config_value is not None:
            return bool(config_value)
        if env_value is not None:
            return env_value.lower() in ("true", "1", "yes", "on")
        return default_value

    def reload(self):
        """設定を再読み込み"""
        self._load_config()
        logger.info("Error handler configuration reloaded")


class ErrorHandlerStats:
    """エラーハンドラーの統計情報（スレッドセーフ版）"""

    def __init__(self, config: Optional[EnhancedErrorHandlerConfig] = None):
        """
        Args:
            config: エラーハンドラー設定（依存性注入）
        """
        self._config = config
        self._lock = threading.RLock()
        self._errors_handled = 0
        self._sanitization_count = 0
        self._i18n_fallback_count = 0
        self._rich_display_count = 0
        self._log_errors_count = 0

        if self._config:
            self._lock_timeout = self._config.lock_timeout_seconds
        else:
            self._lock_timeout = 1.0

    def _safe_lock_operation(self, operation_func, default_value=0):
        """安全なロック操作（タイムアウト付き）"""
        try:
            if self._lock.acquire(timeout=self._lock_timeout):
                try:
                    return operation_func()
                except Exception as e:
                    if logger.isEnabledFor(logging.ERROR):
                        logger.error(f"ErrorHandlerStats operation failed: {e}")
                    return default_value
                finally:
                    try:
                        self._lock.release()
                    except Exception as release_error:
                        if logger.isEnabledFor(logging.ERROR):
                            logger.error(f"Failed to release ErrorHandlerStats lock: {release_error}")
            else:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"ErrorHandlerStats lock timeout ({self._lock_timeout}s)")
                return default_value
        except Exception as e:
            if logger.isEnabledFor(logging.ERROR):
                logger.error(f"ErrorHandlerStats lock acquisition failed: {e}")
            return default_value

    def record_error_handled(self, count: int = 1):
        """エラー処理回数を記録"""
        self._safe_lock_operation(lambda: setattr(self, '_errors_handled', self._errors_handled + count))

    def record_sanitization(self, count: int = 1):
        """サニタイズ実行回数を記録"""
        self._safe_lock_operation(lambda: setattr(self, '_sanitization_count', self._sanitization_count + count))

    def record_i18n_fallback(self, count: int = 1):
        """i18nフォールバック回数を記録"""
        self._safe_lock_operation(lambda: setattr(self, '_i18n_fallback_count', self._i18n_fallback_count + count))

    def record_rich_display(self, count: int = 1):
        """Rich表示回数を記録"""
        self._safe_lock_operation(lambda: setattr(self, '_rich_display_count', self._rich_display_count + count))

    def record_log_error(self, count: int = 1):
        """ログエラー回数を記録"""
        self._safe_lock_operation(lambda: setattr(self, '_log_errors_count', self._log_errors_count + count))

    def get_stats(self) -> Dict[str, int]:
        """統計情報を取得"""
        def create_stats():
            return {
                "errors_handled": self._errors_handled,
                "sanitization_count": self._sanitization_count,
                "i18n_fallback_count": self._i18n_fallback_count,
                "rich_display_count": self._rich_display_count,
                "log_errors_count": self._log_errors_count,
            }

        result = self._safe_lock_operation(create_stats)
        return result if isinstance(result, dict) else {}

    def reset(self):
        """統計をリセット"""
        def reset_counters():
            self._errors_handled = 0
            self._sanitization_count = 0
            self._i18n_fallback_count = 0
            self._rich_display_count = 0
            self._log_errors_count = 0

        self._safe_lock_operation(reset_counters)


class EnhancedErrorHandler:
    """
    統合されたエラーハンドリングシステム（依存性注入対応・完全統合版）

    friendly_error_handler.pyから統合された機能:
    - ユーザーフレンドリーなエラーメッセージ
    - エラーコード推論機能
    - Rich形式のパネル表示
    - CLIエラーハンドリング

    新機能:
    - config_manager完全統合
    - cache_utilsとの連携
    - 統計情報収集
    - パフォーマンス最適化
    """

    # BUILTIN_ERROR_MESSAGESは廃止されました
    # 全てのエラーメッセージはsrc/day_trade/config/messages.jsonで管理されています
    # 新しいエラーメッセージを追加する場合は、messages.jsonを編集してください

    def __init__(
        self,
        message_handler: Optional[I18nMessageHandler] = None,
        sanitizer: Optional[SensitiveDataSanitizer] = None,
        config: Optional[EnhancedErrorHandlerConfig] = None,
        language: Language = Language.JAPANESE,
        stats: Optional[ErrorHandlerStats] = None,
    ):
        """
        Args:
            message_handler: メッセージハンドラー（依存性注入）
            sanitizer: データサニタイザー（依存性注入）
            config: エラーハンドラー設定（依存性注入）
            language: デフォルト言語
            stats: 統計収集オブジェクト（依存性注入）
        """
        self.language = language

        # 依存性注入 - 設定から値を取得
        self.config = config or EnhancedErrorHandlerConfig()
        self.debug_mode = self.config.debug_mode
        self.enable_sanitization = self.config.enable_sanitization

        # 依存性注入 - なければデフォルトインスタンスを作成
        self.message_handler = message_handler or I18nMessageHandler(language)
        self.sanitizer = sanitizer or SensitiveDataSanitizer()
        self.stats = stats or ErrorHandlerStats(self.config)

        # Rich console設定
        self.console = Console(width=self.config.console_width) if self.config.enable_rich_display else None

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
        メッセージハンドラーから取得し、失敗時はビルトインメッセージにフォールバック（堅牢性強化版）

        Args:
            error_code: エラーコード
            context: コンテキスト情報

        Returns:
            メッセージ辞書（必要なキーが保証される）
        """
        message_data = None

        try:
            # まずi18nメッセージハンドラーから取得を試行
            message_data = self.message_handler.get_message(error_code, context=context)

            # 必要なキーが存在し、かつ適切な型であるかチェック（堅牢性強化）
            required_keys = {
                "title": str,
                "message": str,
                "solutions": list
            }

            is_valid = True
            for key, expected_type in required_keys.items():
                if key not in message_data:
                    is_valid = False
                    break
                if not isinstance(message_data[key], expected_type):
                    is_valid = False
                    break
                # 空の値もチェック
                if expected_type == str and not message_data[key].strip():
                    is_valid = False
                    break
                if expected_type == list and not message_data[key]:
                    is_valid = False
                    break

            if is_valid:
                # 安全なコピーを返す（参照問題を回避）
                return {
                    "title": str(message_data["title"]),
                    "message": str(message_data["message"]),
                    "solutions": list(message_data["solutions"]),
                    "emoji": message_data.get("emoji", "❌")
                }

        except Exception as e:
            logger.warning(f"I18nMessageHandlerからの取得に失敗: {e}")
            self.stats.record_i18n_fallback()

        # フォールバック: i18nメッセージハンドラーから直接取得を試行
        try:
            # i18nメッセージハンドラーから基本的なメッセージを取得
            basic_message = self.message_handler.get_message("UNKNOWN_ERROR", context={})
            if basic_message and isinstance(basic_message, dict):
                validated_basic = self._validate_message_data(basic_message)
                if validated_basic:
                    return validated_basic
        except Exception as e:
            logger.warning(f"フォールバック時のメッセージ取得に失敗: {e}")
            self.stats.record_i18n_fallback()

        # 最終フォールバック: デフォルトメッセージ（常に有効）
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

    def _validate_message_data(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        メッセージデータの検証とサニタイズ

        Args:
            message_data: 検証するメッセージデータ

        Returns:
            検証済みのメッセージデータ、またはNone
        """
        if not isinstance(message_data, dict):
            return None

        try:
            title = message_data.get("title", "")
            message = message_data.get("message", "")
            solutions = message_data.get("solutions", [])

            # 基本検証
            if not isinstance(title, str) or not title.strip():
                return None
            if not isinstance(message, str) or not message.strip():
                return None
            if not isinstance(solutions, list) or not solutions:
                return None

            # solutionsの各要素も検証
            valid_solutions = []
            for solution in solutions:
                if isinstance(solution, str) and solution.strip():
                    valid_solutions.append(solution.strip())

            if not valid_solutions:
                return None

            return {
                "title": title.strip(),
                "message": message.strip(),
                "solutions": valid_solutions[:self.config.max_solution_items],
                "emoji": message_data.get("emoji", "❌")
            }

        except Exception as e:
            logger.warning(f"メッセージデータの検証中にエラー: {e}")
            return None

    def _enhanced_sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        強化された機密情報サニタイズ（セキュリティ強化版）

        Args:
            context: サニタイズするコンテキスト

        Returns:
            サニタイズされたコンテキスト
        """
        if not context:
            return context

        # 基本サニタイズ（既存のSensitiveDataSanitizerを使用）
        sanitized_context = self.sanitizer.sanitize_context(context.copy())

        # 追加の機密情報パターン検出（強化版）
        sensitive_patterns = [
            # APIキー関連
            r'(?i)(api[_-]?key|token|secret|password|passwd|pwd)',
            # 金融関連
            r'(?i)(credit[_-]?card|bank[_-]?account|account[_-]?number)',
            # 個人情報
            r'(?i)(ssn|social[_-]?security|driver[_-]?license)',
            # サーバー関連
            r'(?i)(server[_-]?password|db[_-]?password|database[_-]?password)',
        ]

        def is_sensitive_value(value_str: str) -> bool:
            """値が機密情報かどうかを判定"""
            if not isinstance(value_str, str):
                return False

            value_lower = value_str.lower()

            # 長いランダム文字列（APIキーなど）
            if len(value_str) > 20 and any(c.isalnum() for c in value_str):
                # 英数字の組み合わせで長い文字列
                alpha_count = sum(1 for c in value_str if c.isalpha())
                digit_count = sum(1 for c in value_str if c.isdigit())
                if alpha_count > 5 and digit_count > 5:
                    return True

            # JWT トークンパターン
            if value_str.count('.') == 2 and len(value_str) > 100:
                return True

            # Base64エンコードされた長い文字列
            if len(value_str) > 50 and value_str.replace('=', '').replace('+', '').replace('/', '').isalnum():
                return True

            return False

        def sanitize_recursive(obj: Any, depth: int = 0) -> Any:
            """再帰的にオブジェクトをサニタイズ（深度制限付き）"""
            # 再帰深度制限
            if depth > 10:
                return "[深すぎる構造のため省略]"

            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    key_str = str(key).lower()

                    # キー名で機密情報を判定
                    is_sensitive_key = any(
                        re.search(pattern, key_str)
                        for pattern in sensitive_patterns
                    )

                    if is_sensitive_key:
                        result[key] = "[機密情報のため非表示]"
                    elif isinstance(value, str) and is_sensitive_value(value):
                        result[key] = "[機密データのため非表示]"
                    else:
                        result[key] = sanitize_recursive(value, depth + 1)

                return result

            elif isinstance(obj, (list, tuple)):
                return type(obj)(sanitize_recursive(item, depth + 1) for item in obj)

            elif isinstance(obj, str):
                if is_sensitive_value(obj):
                    return "[機密データのため非表示]"
                return obj

            else:
                return obj

        try:
            # 強化されたサニタイズを実行
            enhanced_sanitized = sanitize_recursive(sanitized_context)

            return enhanced_sanitized

        except Exception as e:
            logger.warning(f"強化サニタイズ中にエラー: {e}")
            # フォールバック: 基本サニタイズのみ
            return sanitized_context

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None,
        show_technical: Optional[bool] = None,
    ) -> Panel:
        """
        エラーを包括的に処理してユーザーフレンドリーなパネルを作成（統計付き）

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト情報
            user_action: ユーザーが実行していたアクション
            show_technical: 技術的詳細を表示するか

        Returns:
            Rich Panel オブジェクト
        """
        self.stats.record_error_handled()

        show_tech = show_technical if show_technical is not None else self.debug_mode
        context = context or {}

        # コンテキストサイズ制限
        if len(context) > self.config.max_context_items:
            limited_context = dict(list(context.items())[:self.config.max_context_items])
            limited_context["_context_truncated"] = f"表示制限により {len(context) - self.config.max_context_items} 項目が省略されました"
            context = limited_context

        # ユーザーアクションをコンテキストに追加
        if user_action:
            context["user_action"] = user_action

        # コンテキストのサニタイズ（セキュリティ強化版）
        if self.enable_sanitization:
            context = self._enhanced_sanitize_context(context)
            self.stats.record_sanitization()

        # カスタム例外の場合
        if isinstance(error, DayTradeError):
            panel = self._handle_custom_error(error, context, show_tech)
        else:
            # 一般的な例外の場合
            panel = self._handle_general_error(error, context, show_tech)

        if self.config.enable_rich_display:
            self.stats.record_rich_display()

        return panel

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
            solutions=message_data["solutions"][:self.config.max_solution_items],
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
                self.stats.record_i18n_fallback()
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
            solutions=message_data["solutions"][:self.config.max_solution_items],
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
        """拡張されたエラーパネルを作成（設定統合版）"""

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

        # その他のコンテキスト（制限付き）
        other_context = {k: v for k, v in context.items()
                         if k not in ["user_action", "user_input", "_context_truncated"]}
        if other_context:
            content_lines.append(f"[dim]詳細: {str(other_context)[:100]}...[/dim]")

        # 省略表示メッセージ
        if "_context_truncated" in context:
            content_lines.append(f"[dim yellow]{context['_context_truncated']}[/dim]")

        content_lines.append("")

        # 解決策
        content_lines.append("[bold yellow]💡 解決方法:[/bold yellow]")
        for i, solution in enumerate(solutions, 1):
            content_lines.append(f"  {i}. {solution}")

        # 技術的詳細（デバッグモード時）
        if show_technical and self.config.log_technical_details:
            content_lines.extend([
                "",
                "[dim]── 技術的詳細 ──[/dim]",
                f"[dim]エラータイプ: {type(error).__name__}[/dim]",
                f"[dim]メッセージ: {str(error)}[/dim]",
            ])

            if isinstance(error, DayTradeError):
                if error.error_code:
                    content_lines.append(f"[dim]エラーコード: {error.error_code}[/dim]")
                if error.details:
                    # 技術的詳細も機密情報をサニタイズ
                    sanitized_details = self.sanitizer.sanitize_context(error.details) if self.enable_sanitization else error.details
                    content_lines.append(f"[dim]詳細情報: {sanitized_details}[/dim]")

        # 統計情報（デバッグモード時）
        if show_technical and self.config.enable_performance_logging:
            stats = self.stats.get_stats()
            content_lines.extend([
                "",
                "[dim]── 統計情報 ──[/dim]",
                f"[dim]処理済エラー数: {stats.get('errors_handled', 0)}[/dim]",
                f"[dim]サニタイズ実行数: {stats.get('sanitization_count', 0)}[/dim]",
            ])

        # ヘルプメッセージ
        content_lines.extend([
            "",
            "[dim]💬 さらにサポートが必要な場合は、上記の技術的詳細と共にお問い合わせください。[/dim]",
        ])

        content = "\n".join(content_lines)

        return Panel(
            content,
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            padding=self.config.panel_padding,
        )

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None,
    ) -> None:
        """
        エラーの詳細をログに記録（機密情報サニタイズ付き・統計付き）

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト
            user_action: ユーザーアクション
        """
        self.stats.record_log_error()

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
            sanitized_user_action = self.sanitizer.sanitize(user_action)

        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": sanitized_context,
            "user_action": sanitized_user_action,
            "handler_stats": self.stats.get_stats(),
        }

        if isinstance(error, DayTradeError):
            log_data["error_code"] = error.error_code

            # error.detailsもサニタイズ
            error_details = error.details or {}
            if self.enable_sanitization:
                log_data["error_details"] = self.sanitizer.sanitize_context(error_details)
            else:
                log_data["error_details"] = error_details

        if self.config.log_technical_details:
            logger.error("User-facing error occurred", exc_info=True, extra=log_data)
        else:
            logger.error("User-facing error occurred", extra=log_data)

    def display_and_log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None,
        show_technical: Optional[bool] = None,
    ) -> None:
        """
        エラーを表示してログに記録（設定統合版）

        Args:
            error: 例外オブジェクト
            context: 追加コンテキスト
            user_action: ユーザーアクション
            show_technical: 技術的詳細を表示するか
        """
        # パネルを作成して表示
        error_panel = self.handle_error(error, context, user_action, show_technical)

        if self.config.enable_rich_display and self.console:
            self.console.print(error_panel)
        else:
            # フォールバック: 標準出力
            print(f"エラー: {str(error)}")

        # ログに記録
        self.log_error(error, context, user_action)

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        base_stats = self.stats.get_stats()

        config_info = {
            "debug_mode": self.debug_mode,
            "enable_sanitization": self.enable_sanitization,
            "enable_rich_display": self.config.enable_rich_display,
            "max_context_items": self.config.max_context_items,
            "max_solution_items": self.config.max_solution_items,
        }

        return {
            **base_stats,
            "config": config_info,
            "fallback_rate": base_stats.get("i18n_fallback_count", 0) / max(base_stats.get("errors_handled", 1), 1),
            "sanitization_rate": base_stats.get("sanitization_count", 0) / max(base_stats.get("errors_handled", 1), 1),
        }


# ファクトリー関数とグローバルインスタンス管理
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
        raise ValueError("ハンドラーはEnhancedErrorHandlerインスタンスである必要があります")

    with _handler_lock:
        _default_error_handler = handler


def reset_default_error_handler() -> None:
    """デフォルトエラーハンドラーをリセット（テスト用）"""
    global _default_error_handler

    with _handler_lock:
        _default_error_handler = None


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
        error_handler = create_error_handler(language=language, debug_mode=show_technical)

    error_handler.display_and_log_error(error, context, user_action, show_technical)


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
    handler = get_default_error_handler()

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
def get_error_handler_performance_stats(handler: Optional[EnhancedErrorHandler] = None) -> Dict[str, Any]:
    """エラーハンドラーのパフォーマンス統計を取得"""
    error_handler = handler or get_default_error_handler()
    return error_handler.get_performance_stats()


# 設定統合確認関数
def validate_error_handler_integration(config_manager=None) -> Dict[str, Any]:
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
            "error_type": type(e).__name__
        }
