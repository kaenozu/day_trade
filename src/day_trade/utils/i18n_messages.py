"""
多言語対応エラーメッセージシステム（改善版）
ユーザーフレンドリーなメッセージの国際化対応

改善版：
- 外部JSONファイルからのメッセージ動的ロード
- 機密情報のサニタイズ・マスキング機能
- 依存性注入対応
- カスタム例外の網羅的対応
"""

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Language(Enum):
    """サポート言語"""

    JAPANESE = "ja"
    ENGLISH = "en"


class MessageCategory(Enum):
    """メッセージカテゴリ"""

    NETWORK = "network"
    DATABASE = "database"
    VALIDATION = "validation"
    CONFIG = "config"
    AUTH = "auth"
    FILE = "file"
    GENERAL = "general"


class SensitiveDataSanitizer:
    """機密情報のサニタイズクラス"""

    def __init__(self, sensitive_patterns: Optional[List[str]] = None):
        """
        Args:
            sensitive_patterns: 機密と見なすパターンのリスト
        """
        self.sensitive_patterns = sensitive_patterns or [
            "password",
            "passwd",
            "pwd",
            "token",
            "key",
            "secret",
            "credential",
            "auth",
            "api_key",
            "access_token",
            "private_key",
        ]

        # 正規表現パターンをコンパイル
        self.pattern_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.sensitive_patterns
        ]

    def is_sensitive(self, text: str) -> bool:
        """テキストが機密情報かどうかを判定"""
        if not text or not isinstance(text, str):
            return False

        # パターンマッチング
        for pattern in self.pattern_regex:
            if pattern.search(text):
                return True

        # 一般的な機密情報の形式をチェック
        # APIキーっぽい形式 (32文字以上の英数字)
        if re.match(r"^[a-zA-Z0-9]{32,}$", text):
            return True

        # JWTトークンっぽい形式
        return bool(text.count(".") == 2 and len(text) > 100)

    def sanitize(self, text: str, mask_char: str = "*") -> str:
        """機密情報をマスキング"""
        if not text or not isinstance(text, str):
            return text

        if not self.is_sensitive(text):
            return text

        # 短い文字列は完全マスキング
        if len(text) <= 8:
            return mask_char * len(text)

        # 長い文字列は前後を残してマスキング
        return text[:2] + mask_char * (len(text) - 4) + text[-2:]

    def sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """コンテキスト辞書内の機密情報をサニタイズ"""
        if not context:
            return context

        sanitized = {}
        for key, value in context.items():
            # キー名が機密情報を示唆する場合
            if any(pattern.search(key) for pattern in self.pattern_regex):
                sanitized[key] = self.sanitize(str(value))
            # 値が機密情報の場合
            elif isinstance(value, str) and self.is_sensitive(value):
                sanitized[key] = self.sanitize(value)
            else:
                sanitized[key] = value

        return sanitized


class MessageLoader:
    """メッセージファイルローダー"""

    def __init__(self, messages_file: Optional[str] = None):
        """
        Args:
            messages_file: メッセージファイルのパス
        """
        if messages_file is None:
            # デフォルトのメッセージファイルパス
            current_dir = Path(__file__).parent.parent
            messages_file = current_dir / "config" / "messages.json"

        self.messages_file = Path(messages_file)
        self._messages = {}
        self._exception_mapping = {}
        self._sensitive_patterns = []
        self._ui_messages = {}

        self.load_messages()

    def load_messages(self) -> None:
        """メッセージファイルをロード"""
        try:
            if not self.messages_file.exists():
                raise FileNotFoundError(f"Messages file not found: {self.messages_file}")

            with open(self.messages_file, encoding="utf-8") as f:
                data = json.load(f)

            # 各セクションを統合
            self._messages = {}
            for category_name, category_data in data.items():
                if isinstance(category_data, dict) and category_name.endswith("_errors"):
                    for key, value in category_data.items():
                        self._messages[key] = value

            # 例外マッピング
            self._exception_mapping = data.get("exception_mapping", {})

            # 機密パターン
            self._sensitive_patterns = data.get("sensitive_patterns", [])

            # UIメッセージ
            self._ui_messages = data.get("ui_messages", {})

            # general_errorsから一般エラーを追加
            general_errors = data.get("general_errors", {})
            for key, value in general_errors.items():
                self._messages[key] = value

        except Exception as e:
            # フォールバック: 最小限のメッセージセット
            self._load_fallback_messages()
            print(f"Warning: Could not load messages file: {e}")

    def _load_fallback_messages(self) -> None:
        """フォールバック用の最小メッセージセット"""
        self._messages = {
            "UNKNOWN_ERROR": {
                "ja": {
                    "title": "予期しないエラー",
                    "message": "予期しないエラーが発生しました。",
                    "solutions": ["アプリケーションを再起動してください"],
                    "emoji": "❓",
                    "category": "general",
                },
                "en": {
                    "title": "Unknown Error",
                    "message": "An unexpected error occurred.",
                    "solutions": ["Restart the application"],
                    "emoji": "❓",
                    "category": "general",
                },
            }
        }

        self._exception_mapping = {"Exception": "UNKNOWN_ERROR"}

        self._ui_messages = {"solutions_header": {"ja": "💡 解決方法:", "en": "💡 Solutions:"}}

    @property
    def messages(self) -> Dict[str, Any]:
        return self._messages

    @property
    def exception_mapping(self) -> Dict[str, str]:
        return self._exception_mapping

    @property
    def sensitive_patterns(self) -> List[str]:
        return self._sensitive_patterns

    @property
    def ui_messages(self) -> Dict[str, Any]:
        return self._ui_messages


class EnhancedI18nMessageHandler:
    """改善された多言語メッセージハンドラー"""

    def __init__(
        self,
        language: Language = Language.JAPANESE,
        message_loader: Optional[MessageLoader] = None,
        sanitizer: Optional[SensitiveDataSanitizer] = None,
    ):
        """
        Args:
            language: デフォルト言語
            message_loader: メッセージローダー（依存性注入）
            sanitizer: 機密データサニタイザー（依存性注入）
        """
        self.language = language

        # 依存性注入対応
        self.message_loader = message_loader or MessageLoader()
        self.sanitizer = sanitizer or SensitiveDataSanitizer(self.message_loader.sensitive_patterns)

    def get_message(
        self,
        error_code: str,
        language: Optional[Language] = None,
        context: Optional[Dict[str, Any]] = None,
        sanitize_context: bool = True,
    ) -> Dict[str, Any]:
        """
        エラーコードから多言語メッセージを取得（機密情報保護付き）

        Args:
            error_code: エラーコード
            language: 言語（指定されない場合はデフォルト言語）
            context: 追加コンテキスト
            sanitize_context: コンテキストの機密情報をサニタイズするか

        Returns:
            メッセージ辞書
        """
        lang = language or self.language
        context = context or {}

        # 機密情報のサニタイズ
        if sanitize_context:
            context = self.sanitizer.sanitize_context(context)

        # メッセージを取得
        message_data = self.message_loader.messages.get(error_code, {})
        lang_data = message_data.get(lang.value)

        # 指定された言語にメッセージがない場合は日本語を試す
        if not lang_data and lang != Language.JAPANESE:
            lang_data = message_data.get(Language.JAPANESE.value)

        # それでもない場合は英語を試す
        if not lang_data and lang != Language.ENGLISH:
            lang_data = message_data.get(Language.ENGLISH.value)

        # それでもない場合はデフォルトメッセージ
        if not lang_data:
            unknown_error = self.message_loader.messages.get("UNKNOWN_ERROR", {})
            lang_data = unknown_error.get(
                self.language.value,
                {
                    "title": "Unknown Error",
                    "message": "An unexpected error occurred.",
                    "solutions": ["Restart the application"],
                    "emoji": "❓",
                },
            )

        # コンテキスト情報を適用
        result = lang_data.copy()
        if context.get("user_input"):
            user_input = self.sanitizer.sanitize(str(context["user_input"]))
            if self.language == Language.JAPANESE:
                result["message"] = f"入力値 '{user_input}' で{result['message']}"
            else:
                result["message"] = f"Input '{user_input}': {result['message']}"

        return result

    def get_message_for_exception(
        self,
        exception: Exception,
        language: Optional[Language] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        例外オブジェクトからメッセージを取得

        Args:
            exception: 例外オブジェクト
            language: 言語
            context: 追加コンテキスト

        Returns:
            メッセージ辞書
        """
        exception_name = type(exception).__name__
        error_code = self.message_loader.exception_mapping.get(exception_name, "UNKNOWN_ERROR")

        # カスタム例外の対応
        if hasattr(exception, "error_code"):
            error_code = exception.error_code

        return self.get_message(error_code, language, context)

    def format_solutions_list(
        self, solutions: List[str], language: Optional[Language] = None
    ) -> str:
        """
        解決策リストをフォーマット（外部化されたヘッダー使用）

        Args:
            solutions: 解決策のリスト
            language: 言語

        Returns:
            フォーマット済み解決策文字列
        """
        lang = language or self.language

        # UIメッセージから動的にヘッダーを取得
        ui_messages = self.message_loader.ui_messages
        solutions_header = ui_messages.get("solutions_header", {})
        header = solutions_header.get(lang.value, "💡 Solutions:")

        formatted_solutions = [f"  {i + 1}. {solution}" for i, solution in enumerate(solutions)]

        return f"{header}\n" + "\n".join(formatted_solutions)

    def reload_messages(self) -> bool:
        """メッセージファイルを再読み込み"""
        try:
            self.message_loader.load_messages()
            # サニタイザーのパターンも更新
            self.sanitizer = SensitiveDataSanitizer(self.message_loader.sensitive_patterns)
            return True
        except Exception:
            return False

    def get_supported_languages(self) -> List[Language]:
        """サポートされている言語のリストを取得"""
        return [Language.JAPANESE, Language.ENGLISH]

    def validate_message_completeness(self) -> Dict[str, List[str]]:
        """メッセージの完全性を検証"""
        missing_messages = {}

        for error_code, messages in self.message_loader.messages.items():
            missing_langs = []
            for lang in self.get_supported_languages():
                if lang.value not in messages:
                    missing_langs.append(lang.value)

            if missing_langs:
                missing_messages[error_code] = missing_langs

        return missing_messages


# 後方互換性のためのエイリアス
I18nMessageHandler = EnhancedI18nMessageHandler


# グローバルインスタンス（後方互換性のため）
_default_handler = None


def get_default_handler() -> EnhancedI18nMessageHandler:
    """デフォルトハンドラーを取得（シングルトン）"""
    global _default_handler
    if _default_handler is None:
        _default_handler = EnhancedI18nMessageHandler()
    return _default_handler


def create_handler(
    language: Language = Language.JAPANESE, messages_file: Optional[str] = None
) -> EnhancedI18nMessageHandler:
    """新しいハンドラーインスタンスを作成（依存性注入対応）"""
    loader = MessageLoader(messages_file) if messages_file else None
    return EnhancedI18nMessageHandler(language=language, message_loader=loader)


# 後方互換性のための変数（非推奨）
# 新しいコードでは create_handler() や EnhancedI18nMessageHandler() を直接使用してください
message_handler = get_default_handler()
