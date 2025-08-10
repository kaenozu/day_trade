"""
Day Trade システム用カスタム例外クラス
エラーハンドリングの統一化と分類を行う
"""

import re
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    import requests.exceptions
    import sqlalchemy.exc

# モジュールレベルでライブラリをインポート
try:
    import requests.exceptions as req_exc
except ImportError:
    req_exc = None

try:
    import sqlalchemy.exc as sa_exc
except ImportError:
    sa_exc = None


def _sanitize_sensitive_info(text: str) -> str:
    """
    機密情報を含む可能性のあるテキストをサニタイズする

    Args:
        text: サニタイズ対象のテキスト

    Returns:
        サニタイズされたテキスト
    """
    if not text:
        return text

    # 機密情報パターンのリスト
    sensitive_patterns = [
        # パスワード関連
        (r'password[=:]\s*[^\s,;]+', 'password=***'),
        (r'pwd[=:]\s*[^\s,;]+', 'pwd=***'),
        (r'pass[=:]\s*[^\s,;]+', 'pass=***'),

        # APIキー・トークン関連
        (r'api_key[=:]\s*[^\s,;]+', 'api_key=***'),
        (r'apikey[=:]\s*[^\s,;]+', 'apikey=***'),
        (r'token[=:]\s*[^\s,;]+', 'token=***'),
        (r'access_token[=:]\s*[^\s,;]+', 'access_token=***'),
        (r'refresh_token[=:]\s*[^\s,;]+', 'refresh_token=***'),
        (r'secret[=:]\s*[^\s,;]+', 'secret=***'),

        # データベース接続文字列
        (r'postgresql://[^@]+:[^@]+@', 'postgresql://***:***@'),
        (r'mysql://[^@]+:[^@]+@', 'mysql://***:***@'),
        (r'mongodb://[^@]+:[^@]+@', 'mongodb://***:***@'),
        (r'sqlite:///[^\s]+', 'sqlite:///***'),

        # IP・URL・ポート情報（部分的にマスク）
        (r'://[^:@/]+:[^@/]+@', '://***:***@'),

        # 一般的な機密情報キーワード
        (r'key[=:]\s*[^\s,;]+', 'key=***'),
        (r'credential[=:]\s*[^\s,;]+', 'credential=***'),
        (r'auth[=:]\s*[^\s,;]+', 'auth=***'),
    ]

    sanitized_text = text
    for pattern, replacement in sensitive_patterns:
        sanitized_text = re.sub(pattern, replacement, sanitized_text, flags=re.IGNORECASE)

    return sanitized_text


def _sanitize_details_dict(details: Dict[str, Any]) -> Dict[str, Any]:
    """
    詳細情報辞書内の機密情報をサニタイズする

    Args:
        details: 詳細情報辞書

    Returns:
        サニタイズされた詳細情報辞書
    """
    if not details:
        return details

    sanitized_details = {}
    for key, value in details.items():
        if isinstance(value, str):
            sanitized_details[key] = _sanitize_sensitive_info(value)
        elif isinstance(value, dict):
            sanitized_details[key] = _sanitize_details_dict(value)
        else:
            sanitized_details[key] = value

    return sanitized_details


class DayTradeError(Exception):
    """Day Trade システムの基底例外クラス"""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """
        Args:
            message: エラーメッセージ
            error_code: エラーコード
            details: 追加の詳細情報
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self, sanitize: bool = True) -> dict:
        """
        例外情報を辞書形式で返す

        Args:
            sanitize: 機密情報をサニタイズするかどうか

        Returns:
            例外情報辞書
        """
        message = _sanitize_sensitive_info(self.message) if sanitize else self.message
        details = _sanitize_details_dict(self.details) if sanitize else self.details

        return {
            "error_type": self.__class__.__name__,
            "message": message,
            "error_code": self.error_code,
            "details": details,
        }

    def get_safe_message(self) -> str:
        """
        ログ出力用の安全なメッセージを取得

        Returns:
            サニタイズされたメッセージ
        """
        return _sanitize_sensitive_info(self.message)

    def get_safe_details(self) -> dict:
        """
        ログ出力用の安全な詳細情報を取得

        Returns:
            サニタイズされた詳細情報
        """
        return _sanitize_details_dict(self.details)


class DatabaseError(DayTradeError):
    """データベース関連エラー"""

    pass


class DatabaseConnectionError(DatabaseError):
    """データベース接続エラー"""

    pass


class DatabaseIntegrityError(DatabaseError):
    """データベース整合性エラー"""

    pass


class DatabaseOperationalError(DatabaseError):
    """データベース操作エラー"""

    pass


class APIError(DayTradeError):
    """外部API関連エラー"""

    pass


class NetworkError(APIError):
    """ネットワーク関連エラー"""

    pass


class RateLimitError(APIError):
    """レート制限エラー"""

    pass


class AuthenticationError(APIError):
    """認証エラー"""

    pass


class ResourceNotFoundError(APIError):
    """リソース未検出エラー"""

    pass


class ServerError(APIError):
    """サーバーエラー"""

    pass


class BadRequestError(APIError):
    """不正なリクエストエラー"""

    pass


class DataError(DayTradeError):
    """データ関連エラー"""

    pass


class DataNotFoundError(DataError):
    """データ未検出エラー"""

    pass


class ValidationError(DataError):
    """データ検証エラー"""

    pass


class ConfigurationError(DayTradeError):
    """設定関連エラー"""

    pass


class TradingError(DayTradeError):
    """取引関連エラー"""

    pass


class AnalysisError(DayTradeError):
    """分析関連エラー"""

    pass


class IndicatorCalculationError(AnalysisError):
    """テクニカル指標計算エラー"""

    pass


class PatternRecognitionError(AnalysisError):
    """パターン認識エラー"""

    pass


class BacktestError(AnalysisError):
    """バックテストエラー"""

    pass


class SignalGenerationError(AnalysisError):
    """シグナル生成エラー"""

    pass


class AlertError(DayTradeError):
    """アラート関連エラー"""

    pass


class PortfolioError(DayTradeError):
    """ポートフォリオ関連エラー"""

    pass


class FileOperationError(DayTradeError):
    """ファイル操作エラー"""

    pass


class CacheError(DayTradeError):
    """キャッシュ関連エラー"""

    pass


class TimeoutError(NetworkError):
    """タイムアウトエラー（ネットワーク関連）"""

    pass


# データ取得関連例外
class StockFetcherError(APIError):
    """株価データ取得エラー"""
    pass


class InvalidSymbolError(DataError):
    """無効なシンボルエラー"""
    pass


def handle_database_exception(exc: "sqlalchemy.exc.SQLAlchemyError") -> DatabaseError:
    """
    SQLAlchemy例外を適切なDayTradeError例外に変換

    Args:
        exc: SQLAlchemy例外

    Returns:
        変換されたDayTradeError例外
    """
    if sa_exc is None:
        return DatabaseError(
            message="データベースエラーが発生しました（SQLAlchemyライブラリが利用できません）",
            error_code="DB_LIBRARY_UNAVAILABLE",
            details={"error_type": type(exc).__name__},
        )

    # 機密情報を含む可能性のあるエラーメッセージの安全な処理
    exc_type_name = type(exc).__name__

    # 一般的なエラーメッセージ（詳細な例外情報を含まない）
    safe_error_message = f"データベース{exc_type_name}が発生しました"

    # 詳細情報は機密情報をサニタイズして格納
    details = {
        "error_type": exc_type_name,
        "sanitized_error": _sanitize_sensitive_info(str(exc))
    }

    if isinstance(exc, sa_exc.IntegrityError):
        return DatabaseIntegrityError(
            message="データ整合性エラーが発生しました",
            error_code="DB_INTEGRITY_ERROR",
            details=details,
        )
    elif isinstance(exc, sa_exc.OperationalError):
        return DatabaseOperationalError(
            message="データベース操作エラーが発生しました",
            error_code="DB_OPERATIONAL_ERROR",
            details=details,
        )
    elif isinstance(exc, (sa_exc.DisconnectionError, sa_exc.TimeoutError)):
        return DatabaseConnectionError(
            message="データベース接続エラーが発生しました",
            error_code="DB_CONNECTION_ERROR",
            details=details,
        )
    else:
        return DatabaseError(
            message=safe_error_message,
            error_code="DB_UNKNOWN_ERROR",
            details=details,
        )


def handle_network_exception(exc: "requests.exceptions.RequestException") -> APIError:
    """
    ネットワーク関連例外を適切なAPIError例外に変換

    Args:
        exc: ネットワーク関連例外

    Returns:
        変換されたAPIError例外
    """
    if req_exc is None:
        return APIError(
            message="ネットワークエラーが発生しました（requestsライブラリが利用できません）",
            error_code="NETWORK_LIBRARY_UNAVAILABLE",
            details={"error_type": type(exc).__name__},
        )

    # 機密情報を含む可能性のあるエラーメッセージの安全な処理
    exc_type_name = type(exc).__name__

    # 詳細情報は機密情報をサニタイズして格納
    details = {
        "error_type": exc_type_name,
        "sanitized_error": _sanitize_sensitive_info(str(exc))
    }

    if isinstance(exc, req_exc.ConnectionError):
        return NetworkError(
            message="ネットワーク接続エラーが発生しました",
            error_code="NETWORK_CONNECTION_ERROR",
            details=details,
        )
    elif isinstance(exc, req_exc.Timeout):
        return TimeoutError(
            message="リクエストタイムアウトが発生しました",
            error_code="NETWORK_TIMEOUT_ERROR",
            details=details,
        )
    elif isinstance(exc, req_exc.HTTPError):
        status_code = getattr(exc.response, "status_code", None)
        details["status_code"] = status_code

        if status_code == 429:
            return RateLimitError(
                message="APIレート制限エラーが発生しました",
                error_code="API_RATE_LIMIT_ERROR",
                details=details,
            )
        elif status_code in (401, 403):
            return AuthenticationError(
                message="API認証エラーが発生しました",
                error_code="API_AUTH_ERROR",
                details=details,
            )
        elif status_code == 404:
            return ResourceNotFoundError(
                message="APIリソース未検出エラーが発生しました",
                error_code="API_NOT_FOUND_ERROR",
                details=details,
            )
        elif status_code == 400:
            return BadRequestError(
                message="不正なAPIリクエストエラーが発生しました",
                error_code="API_BAD_REQUEST_ERROR",
                details=details,
            )
        elif status_code >= 500:
            return ServerError(
                message="APIサーバーエラーが発生しました",
                error_code="API_SERVER_ERROR",
                details=details,
            )
        else:
            return APIError(
                message="HTTPエラーが発生しました",
                error_code="API_HTTP_ERROR",
                details=details,
            )
    else:
        return NetworkError(
            message="ネットワークエラーが発生しました",
            error_code="NETWORK_UNKNOWN_ERROR",
            details=details,
        )
