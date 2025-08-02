"""
多言語対応エラーメッセージシステム
ユーザーフレンドリーなメッセージの国際化対応
"""

from enum import Enum
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


# 多言語メッセージデータベース
MESSAGES = {
    # ネットワーク関連
    "NETWORK_CONNECTION_ERROR": {
        Language.JAPANESE: {
            "title": "接続エラー",
            "message": "インターネット接続を確認してください。",
            "solutions": [
                "Wi-Fi または有線LAN接続を確認してください",
                "ファイアウォールやプロキシ設定を確認してください",
                "しばらく時間をおいて再度お試しください",
            ],
            "emoji": "🌐",
        },
        Language.ENGLISH: {
            "title": "Connection Error",
            "message": "Please check your internet connection.",
            "solutions": [
                "Check your Wi-Fi or ethernet connection",
                "Verify firewall and proxy settings",
                "Try again after waiting a moment",
            ],
            "emoji": "🌐",
        },
    },
    "NETWORK_TIMEOUT_ERROR": {
        Language.JAPANESE: {
            "title": "タイムアウトエラー",
            "message": "サーバーからの応答がありませんでした。",
            "solutions": [
                "インターネット接続が安定しているか確認してください",
                "時間をおいて再度お試しください",
                "市場時間中に実行してください",
            ],
            "emoji": "⏰",
        },
        Language.ENGLISH: {
            "title": "Timeout Error",
            "message": "No response received from server.",
            "solutions": [
                "Check if your internet connection is stable",
                "Try again after waiting",
                "Execute during market hours",
            ],
            "emoji": "⏰",
        },
    },
    "API_RATE_LIMIT_ERROR": {
        Language.JAPANESE: {
            "title": "アクセス制限エラー",
            "message": "データ取得の頻度が制限に達しました。",
            "solutions": [
                "しばらく時間をおいて再度お試しください（推奨: 1分程度）",
                "短時間での連続実行を避けてください",
                "必要に応じて有料プランへのアップグレードを検討してください",
            ],
            "emoji": "⚠️",
        },
        Language.ENGLISH: {
            "title": "Rate Limit Error",
            "message": "Data retrieval frequency limit reached.",
            "solutions": [
                "Please wait before trying again (recommended: ~1 minute)",
                "Avoid consecutive executions in short periods",
                "Consider upgrading to a paid plan if needed",
            ],
            "emoji": "⚠️",
        },
    },
    # データベース関連
    "DB_CONNECTION_ERROR": {
        Language.JAPANESE: {
            "title": "データベース接続エラー",
            "message": "データベースに接続できませんでした。",
            "solutions": [
                "アプリケーションを再起動してください",
                "データベースファイルの権限を確認してください",
                "ディスク容量を確認してください",
            ],
            "emoji": "💾",
        },
        Language.ENGLISH: {
            "title": "Database Connection Error",
            "message": "Could not connect to database.",
            "solutions": [
                "Restart the application",
                "Check database file permissions",
                "Verify disk space availability",
            ],
            "emoji": "💾",
        },
    },
    "DB_INTEGRITY_ERROR": {
        Language.JAPANESE: {
            "title": "データ整合性エラー",
            "message": "データベースの整合性に問題があります。",
            "solutions": [
                "データベースを初期化してください（daytrade init）",
                "バックアップから復元してください",
                "サポートにお問い合わせください",
            ],
            "emoji": "🔧",
        },
        Language.ENGLISH: {
            "title": "Data Integrity Error",
            "message": "Database integrity issue detected.",
            "solutions": [
                "Initialize database (daytrade init)",
                "Restore from backup",
                "Contact support",
            ],
            "emoji": "🔧",
        },
    },
    # 入力検証エラー
    "VALIDATION_ERROR": {
        Language.JAPANESE: {
            "title": "入力エラー",
            "message": "入力された情報に問題があります。",
            "solutions": [
                "入力形式を確認してください",
                "有効な値の範囲内で入力してください",
                "必須項目が入力されているか確認してください",
            ],
            "emoji": "✏️",
        },
        Language.ENGLISH: {
            "title": "Input Error",
            "message": "There is an issue with the provided input.",
            "solutions": [
                "Check input format",
                "Enter values within valid range",
                "Ensure required fields are filled",
            ],
            "emoji": "✏️",
        },
    },
    # 設定エラー
    "CONFIG_ERROR": {
        Language.JAPANESE: {
            "title": "設定エラー",
            "message": "アプリケーションの設定に問題があります。",
            "solutions": [
                "設定ファイルの形式を確認してください",
                "設定をデフォルトにリセットしてください（daytrade config reset）",
                "設定項目の値が正しいか確認してください",
            ],
            "emoji": "⚙️",
        },
        Language.ENGLISH: {
            "title": "Configuration Error",
            "message": "Application configuration issue detected.",
            "solutions": [
                "Check configuration file format",
                "Reset to default settings (daytrade config reset)",
                "Verify configuration values are correct",
            ],
            "emoji": "⚙️",
        },
    },
    # 認証エラー
    "API_AUTH_ERROR": {
        Language.JAPANESE: {
            "title": "認証エラー",
            "message": "APIの認証に失敗しました。",
            "solutions": [
                "APIキーが正しく設定されているか確認してください",
                "APIキーの有効期限を確認してください",
                "アカウントがアクティブであることを確認してください",
            ],
            "emoji": "🔐",
        },
        Language.ENGLISH: {
            "title": "Authentication Error",
            "message": "API authentication failed.",
            "solutions": [
                "Verify API key is correctly configured",
                "Check API key expiration date",
                "Ensure account is active",
            ],
            "emoji": "🔐",
        },
    },
    # ファイル操作エラー
    "FILE_NOT_FOUND": {
        Language.JAPANESE: {
            "title": "ファイルエラー",
            "message": "指定されたファイルが見つかりません。",
            "solutions": [
                "ファイルパスが正しいか確認してください",
                "ファイルが存在するか確認してください",
                "読み取り権限があるか確認してください",
            ],
            "emoji": "📁",
        },
        Language.ENGLISH: {
            "title": "File Error",
            "message": "Specified file not found.",
            "solutions": [
                "Verify file path is correct",
                "Check if file exists",
                "Ensure read permissions are available",
            ],
            "emoji": "📁",
        },
    },
    "PERMISSION_ERROR": {
        Language.JAPANESE: {
            "title": "権限エラー",
            "message": "ファイルまたはディレクトリへのアクセス権限がありません。",
            "solutions": [
                "管理者権限で実行してください",
                "ファイルやフォルダの権限を確認してください",
                "他のプログラムがファイルを使用していないか確認してください",
            ],
            "emoji": "🔐",
        },
        Language.ENGLISH: {
            "title": "Permission Error",
            "message": "No access permission to file or directory.",
            "solutions": [
                "Run with administrator privileges",
                "Check file and folder permissions",
                "Ensure file is not in use by another program",
            ],
            "emoji": "🔐",
        },
    },
    # 一般的なエラー
    "UNKNOWN_ERROR": {
        Language.JAPANESE: {
            "title": "予期しないエラー",
            "message": "システムで予期しないエラーが発生しました。",
            "solutions": [
                "アプリケーションを再起動してください",
                "最新版にアップデートしてください",
                "サポートにお問い合わせください",
            ],
            "emoji": "❓",
        },
        Language.ENGLISH: {
            "title": "Unexpected Error",
            "message": "An unexpected system error occurred.",
            "solutions": [
                "Restart the application",
                "Update to the latest version",
                "Contact support",
            ],
            "emoji": "❓",
        },
    },
}

# 一般的なPython例外のマッピング
EXCEPTION_MAPPING = {
    "FileNotFoundError": "FILE_NOT_FOUND",
    "PermissionError": "PERMISSION_ERROR",
    "ConnectionError": "NETWORK_CONNECTION_ERROR",
    "TimeoutError": "NETWORK_TIMEOUT_ERROR",
    "KeyError": "VALIDATION_ERROR",
    "ValueError": "VALIDATION_ERROR",
}


class I18nMessageHandler:
    """多言語メッセージハンドラー"""

    def __init__(self, language: Language = Language.JAPANESE):
        """
        Args:
            language: デフォルト言語
        """
        self.language = language

    def get_message(
        self,
        error_code: str,
        language: Optional[Language] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        エラーコードから多言語メッセージを取得

        Args:
            error_code: エラーコード
            language: 言語（指定されない場合はデフォルト言語）
            context: 追加コンテキスト

        Returns:
            メッセージ辞書
        """
        lang = language or self.language
        context = context or {}

        # メッセージを取得
        message_data = MESSAGES.get(error_code, {})
        lang_data = message_data.get(lang)

        # 指定された言語にメッセージがない場合は日本語を試す
        if not lang_data and lang != Language.JAPANESE:
            lang_data = message_data.get(Language.JAPANESE)

        # それでもない場合は英語を試す
        if not lang_data and lang != Language.ENGLISH:
            lang_data = message_data.get(Language.ENGLISH)

        # それでもない場合はデフォルトメッセージ
        if not lang_data:
            lang_data = MESSAGES["UNKNOWN_ERROR"].get(self.language, {})

        # コンテキスト情報を適用
        result = lang_data.copy()
        if context.get("user_input"):
            result["message"] = (
                f"入力値 '{context['user_input']}' で{result['message']}"
            )

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
        error_code = EXCEPTION_MAPPING.get(exception_name, "UNKNOWN_ERROR")

        return self.get_message(error_code, language, context)

    def format_solutions_list(
        self,
        solutions: List[str],
        language: Optional[Language] = None,
    ) -> str:
        """
        解決策リストをフォーマット

        Args:
            solutions: 解決策のリスト
            language: 言語

        Returns:
            フォーマット済み解決策文字列
        """
        lang = language or self.language

        header = "💡 解決方法:" if lang == Language.JAPANESE else "💡 Solutions:"

        formatted_solutions = [
            f"  {i + 1}. {solution}" for i, solution in enumerate(solutions)
        ]

        return f"{header}\n" + "\n".join(formatted_solutions)


# グローバルハンドラーインスタンス
default_message_handler = I18nMessageHandler(Language.JAPANESE)


def get_user_friendly_message(
    error_code: str,
    language: Language = Language.JAPANESE,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    ユーザーフレンドリーなメッセージを取得する便利関数

    Args:
        error_code: エラーコード
        language: 言語
        context: 追加コンテキスト

    Returns:
        メッセージ辞書
    """
    return default_message_handler.get_message(error_code, language, context)


def get_message_for_exception(
    exception: Exception,
    language: Language = Language.JAPANESE,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    例外からユーザーフレンドリーなメッセージを取得する便利関数

    Args:
        exception: 例外オブジェクト
        language: 言語
        context: 追加コンテキスト

    Returns:
        メッセージ辞書
    """
    return default_message_handler.get_message_for_exception(
        exception, language, context
    )