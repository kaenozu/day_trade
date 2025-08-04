"""
Day Trade システム用カスタム例外クラス
エラーハンドリングの統一化と分類を行う
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import requests.exceptions
    import sqlalchemy.exc


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

    def to_dict(self) -> dict:
        """例外情報を辞書形式で返す"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


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


# NOTE: レガシー例外マッピングは削除されました
# StockFetcherError, InvalidSymbolError, DataNotFoundError は
# stock_fetcher.py で直接継承して定義されているため、
# ここでのマッピングは不要です


def handle_database_exception(exc: "sqlalchemy.exc.SQLAlchemyError") -> DatabaseError:
    """
    SQLAlchemy例外を適切なDayTradeError例外に変換

    Args:
        exc: SQLAlchemy例外

    Returns:
        変換されたDayTradeError例外
    """
    import sqlalchemy.exc as sa_exc

    if isinstance(exc, sa_exc.IntegrityError):
        return DatabaseIntegrityError(
            message=f"データ整合性エラー: {exc}",
            error_code="DB_INTEGRITY_ERROR",
            details={"original_error": str(exc)},
        )
    elif isinstance(exc, sa_exc.OperationalError):
        return DatabaseOperationalError(
            message=f"データベース操作エラー: {exc}",
            error_code="DB_OPERATIONAL_ERROR",
            details={"original_error": str(exc)},
        )
    elif isinstance(exc, (sa_exc.DisconnectionError, sa_exc.TimeoutError)):
        return DatabaseConnectionError(
            message=f"データベース接続エラー: {exc}",
            error_code="DB_CONNECTION_ERROR",
            details={"original_error": str(exc)},
        )
    else:
        return DatabaseError(
            message=f"データベースエラー: {exc}",
            error_code="DB_UNKNOWN_ERROR",
            details={"original_error": str(exc)},
        )


def handle_network_exception(exc: "requests.exceptions.RequestException") -> APIError:
    """
    ネットワーク関連例外を適切なAPIError例外に変換

    Args:
        exc: ネットワーク関連例外

    Returns:
        変換されたAPIError例外
    """
    import requests.exceptions as req_exc

    if isinstance(exc, req_exc.ConnectionError):
        return NetworkError(
            message=f"ネットワーク接続エラー: {exc}",
            error_code="NETWORK_CONNECTION_ERROR",
            details={"original_error": str(exc)},
        )
    elif isinstance(exc, req_exc.Timeout):
        return TimeoutError(
            message=f"リクエストタイムアウト: {exc}",
            error_code="NETWORK_TIMEOUT_ERROR",
            details={"original_error": str(exc)},
        )
    elif isinstance(exc, req_exc.HTTPError):
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 429:
            return RateLimitError(
                message=f"レート制限エラー: {exc}",
                error_code="API_RATE_LIMIT_ERROR",
                details={"original_error": str(exc), "status_code": status_code},
            )
        elif status_code in (401, 403):
            return AuthenticationError(
                message=f"認証エラー: {exc}",
                error_code="API_AUTH_ERROR",
                details={"original_error": str(exc), "status_code": status_code},
            )
        elif status_code == 404:
            return ResourceNotFoundError(
                message=f"リソース未検出エラー: {exc}",
                error_code="API_NOT_FOUND_ERROR",
                details={"original_error": str(exc), "status_code": status_code},
            )
        elif status_code == 400:
            return BadRequestError(
                message=f"不正なリクエストエラー: {exc}",
                error_code="API_BAD_REQUEST_ERROR",
                details={"original_error": str(exc), "status_code": status_code},
            )
        elif status_code >= 500:
            return ServerError(
                message=f"サーバーエラー: {exc}",
                error_code="API_SERVER_ERROR",
                details={"original_error": str(exc), "status_code": status_code},
            )
        else:
            return APIError(
                message=f"HTTPエラー: {exc}",
                error_code="API_HTTP_ERROR",
                details={"original_error": str(exc), "status_code": status_code},
            )
    else:
        return NetworkError(
            message=f"ネットワークエラー: {exc}",
            error_code="NETWORK_UNKNOWN_ERROR",
            details={"original_error": str(exc)},
        )
