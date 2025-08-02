"""
例外システムのテスト
"""

import pytest
from unittest.mock import Mock

from src.day_trade.utils.exceptions import (
    DayTradeError,
    DatabaseError,
    DatabaseConnectionError,
    DatabaseIntegrityError,
    DatabaseOperationalError,
    APIError,
    NetworkError,
    RateLimitError,
    AuthenticationError,
    ResourceNotFoundError,
    BadRequestError,
    ServerError,
    handle_database_exception,
    handle_network_exception,
)


class TestCustomExceptions:
    """カスタム例外クラスのテスト"""

    def test_day_trade_error_basic(self):
        """基底例外の基本機能テスト"""
        error = DayTradeError("テストエラー")
        assert str(error) == "テストエラー"
        assert error.message == "テストエラー"
        assert error.error_code is None
        assert error.details == {}

    def test_day_trade_error_with_details(self):
        """詳細情報付き例外のテスト"""
        details = {"key": "value", "number": 123}
        error = DayTradeError(
            message="詳細エラー",
            error_code="TEST_ERROR",
            details=details
        )

        assert error.message == "詳細エラー"
        assert error.error_code == "TEST_ERROR"
        assert error.details == details

    def test_day_trade_error_to_dict(self):
        """例外の辞書変換テスト"""
        error = DayTradeError(
            message="辞書テスト",
            error_code="DICT_TEST",
            details={"test": True}
        )

        result = error.to_dict()
        expected = {
            "error_type": "DayTradeError",
            "message": "辞書テスト",
            "error_code": "DICT_TEST",
            "details": {"test": True}
        }

        assert result == expected

    def test_exception_inheritance(self):
        """例外の継承関係テスト"""
        # データベース例外
        db_error = DatabaseError("DB エラー")
        assert isinstance(db_error, DayTradeError)

        db_conn_error = DatabaseConnectionError("接続エラー")
        assert isinstance(db_conn_error, DatabaseError)
        assert isinstance(db_conn_error, DayTradeError)

        # API例外
        api_error = APIError("API エラー")
        assert isinstance(api_error, DayTradeError)

        net_error = NetworkError("ネットワークエラー")
        assert isinstance(net_error, APIError)
        assert isinstance(net_error, DayTradeError)


class TestDatabaseExceptionHandler:
    """データベース例外ハンドラーのテスト"""

    def test_handle_integrity_error(self):
        """整合性エラーの処理テスト"""
        # 実際のSQLAlchemyの例外をインポートして使用
        try:
            from sqlalchemy.exc import IntegrityError as SQLIntegrityError
            mock_exc = SQLIntegrityError("constraint violation", None, None)
        except ImportError:
            # SQLAlchemyが利用できない場合はモックを使用
            from unittest.mock import Mock
            mock_exc = Mock()
            mock_exc.__class__ = type("IntegrityError", (Exception,), {})
            mock_exc.__class__.__module__ = "sqlalchemy.exc"
            mock_exc.__str__ = lambda: "constraint violation"
            # isinstance チェックが通るようにモジュール情報を設定
            import sys
            from unittest.mock import MagicMock
            mock_sa_exc = MagicMock()
            mock_sa_exc.IntegrityError = mock_exc.__class__
            sys.modules['sqlalchemy.exc'] = mock_sa_exc

        result = handle_database_exception(mock_exc)

        assert isinstance(result, DatabaseIntegrityError)
        assert "constraint violation" in result.message
        assert result.error_code == "DB_INTEGRITY_ERROR"

    def test_handle_operational_error(self):
        """操作エラーの処理テスト"""
        try:
            from sqlalchemy.exc import OperationalError as SQLOperationalError
            mock_exc = SQLOperationalError("operation failed", None, None)
        except ImportError:
            from unittest.mock import Mock
            import sys
            from unittest.mock import MagicMock
            mock_exc = Mock()
            mock_exc.__class__ = type("OperationalError", (Exception,), {})
            mock_exc.__class__.__module__ = "sqlalchemy.exc"
            mock_exc.__str__ = lambda: "operation failed"
            mock_sa_exc = MagicMock()
            mock_sa_exc.OperationalError = mock_exc.__class__
            sys.modules['sqlalchemy.exc'] = mock_sa_exc

        result = handle_database_exception(mock_exc)

        assert isinstance(result, DatabaseOperationalError)
        assert "operation failed" in result.message
        assert result.error_code == "DB_OPERATIONAL_ERROR"

    def test_handle_connection_error(self):
        """接続エラーの処理テスト"""
        try:
            from sqlalchemy.exc import DisconnectionError as SQLDisconnectionError
            mock_exc = SQLDisconnectionError("connection lost")
        except ImportError:
            from unittest.mock import Mock
            import sys
            from unittest.mock import MagicMock
            mock_exc = Mock()
            mock_exc.__class__ = type("DisconnectionError", (Exception,), {})
            mock_exc.__class__.__module__ = "sqlalchemy.exc"
            mock_exc.__str__ = lambda: "connection lost"
            mock_sa_exc = MagicMock()
            mock_sa_exc.DisconnectionError = mock_exc.__class__
            sys.modules['sqlalchemy.exc'] = mock_sa_exc

        result = handle_database_exception(mock_exc)

        assert isinstance(result, DatabaseConnectionError)
        assert "connection lost" in result.message
        assert result.error_code == "DB_CONNECTION_ERROR"


class TestNetworkExceptionHandler:
    """ネットワーク例外ハンドラーのテスト"""

    def test_handle_connection_error(self):
        """接続エラーの処理テスト"""
        try:
            from requests.exceptions import ConnectionError as ReqConnectionError
            mock_exc = ReqConnectionError("connection failed")
        except ImportError:
            from unittest.mock import Mock
            import sys
            from unittest.mock import MagicMock
            mock_exc = Mock()
            mock_exc.__class__ = type("ConnectionError", (Exception,), {})
            mock_exc.__class__.__module__ = "requests.exceptions"
            mock_exc.__str__ = lambda: "connection failed"
            mock_req_exc = MagicMock()
            mock_req_exc.ConnectionError = mock_exc.__class__
            sys.modules['requests.exceptions'] = mock_req_exc

        result = handle_network_exception(mock_exc)

        assert isinstance(result, NetworkError)
        assert "connection failed" in result.message
        assert result.error_code == "NETWORK_CONNECTION_ERROR"

    def test_handle_http_error_404(self):
        """404エラーの処理テスト"""
        try:
            from requests.exceptions import HTTPError as ReqHTTPError
            from unittest.mock import Mock
            mock_exc = ReqHTTPError("not found")
            mock_response = Mock()
            mock_response.status_code = 404
            mock_exc.response = mock_response
        except ImportError:
            from unittest.mock import Mock
            import sys
            from unittest.mock import MagicMock
            mock_exc = Mock()
            mock_exc.__class__ = type("HTTPError", (Exception,), {})
            mock_exc.__class__.__module__ = "requests.exceptions"
            mock_exc.__str__ = lambda: "not found"
            mock_response = Mock()
            mock_response.status_code = 404
            mock_exc.response = mock_response
            mock_req_exc = MagicMock()
            mock_req_exc.HTTPError = mock_exc.__class__
            sys.modules['requests.exceptions'] = mock_req_exc

        result = handle_network_exception(mock_exc)

        assert isinstance(result, ResourceNotFoundError)
        assert "not found" in result.message
        assert result.error_code == "API_NOT_FOUND_ERROR"
        assert result.details["status_code"] == 404

    def test_handle_http_error_400(self):
        """400エラーの処理テスト"""
        try:
            from requests.exceptions import HTTPError as ReqHTTPError
            from unittest.mock import Mock
            mock_exc = ReqHTTPError("bad request")
            mock_response = Mock()
            mock_response.status_code = 400
            mock_exc.response = mock_response
        except ImportError:
            from unittest.mock import Mock
            import sys
            from unittest.mock import MagicMock
            mock_exc = Mock()
            mock_exc.__class__ = type("HTTPError", (Exception,), {})
            mock_exc.__class__.__module__ = "requests.exceptions"
            mock_exc.__str__ = lambda: "bad request"
            mock_response = Mock()
            mock_response.status_code = 400
            mock_exc.response = mock_response
            mock_req_exc = MagicMock()
            mock_req_exc.HTTPError = mock_exc.__class__
            sys.modules['requests.exceptions'] = mock_req_exc

        result = handle_network_exception(mock_exc)

        assert isinstance(result, BadRequestError)
        assert "bad request" in result.message
        assert result.error_code == "API_BAD_REQUEST_ERROR"

    def test_handle_http_error_429(self):
        """レート制限エラーの処理テスト"""
        try:
            from requests.exceptions import HTTPError as ReqHTTPError
            from unittest.mock import Mock
            mock_exc = ReqHTTPError("rate limit exceeded")
            mock_response = Mock()
            mock_response.status_code = 429
            mock_exc.response = mock_response
        except ImportError:
            from unittest.mock import Mock
            import sys
            from unittest.mock import MagicMock
            mock_exc = Mock()
            mock_exc.__class__ = type("HTTPError", (Exception,), {})
            mock_exc.__class__.__module__ = "requests.exceptions"
            mock_exc.__str__ = lambda: "rate limit exceeded"
            mock_response = Mock()
            mock_response.status_code = 429
            mock_exc.response = mock_response
            mock_req_exc = MagicMock()
            mock_req_exc.HTTPError = mock_exc.__class__
            sys.modules['requests.exceptions'] = mock_req_exc

        result = handle_network_exception(mock_exc)

        assert isinstance(result, RateLimitError)
        assert "rate limit exceeded" in result.message
        assert result.error_code == "API_RATE_LIMIT_ERROR"

    def test_handle_http_error_500(self):
        """500エラーの処理テスト"""
        try:
            from requests.exceptions import HTTPError as ReqHTTPError
            from unittest.mock import Mock
            mock_exc = ReqHTTPError("internal server error")
            mock_response = Mock()
            mock_response.status_code = 500
            mock_exc.response = mock_response
        except ImportError:
            from unittest.mock import Mock
            import sys
            from unittest.mock import MagicMock
            mock_exc = Mock()
            mock_exc.__class__ = type("HTTPError", (Exception,), {})
            mock_exc.__class__.__module__ = "requests.exceptions"
            mock_exc.__str__ = lambda: "internal server error"
            mock_response = Mock()
            mock_response.status_code = 500
            mock_exc.response = mock_response
            mock_req_exc = MagicMock()
            mock_req_exc.HTTPError = mock_exc.__class__
            sys.modules['requests.exceptions'] = mock_req_exc

        result = handle_network_exception(mock_exc)

        assert isinstance(result, ServerError)
        assert "internal server error" in result.message
        assert result.error_code == "API_SERVER_ERROR"

    def test_handle_http_error_401(self):
        """認証エラーの処理テスト"""
        try:
            from requests.exceptions import HTTPError as ReqHTTPError
            from unittest.mock import Mock
            mock_exc = ReqHTTPError("unauthorized")
            mock_response = Mock()
            mock_response.status_code = 401
            mock_exc.response = mock_response
        except ImportError:
            from unittest.mock import Mock
            import sys
            from unittest.mock import MagicMock
            mock_exc = Mock()
            mock_exc.__class__ = type("HTTPError", (Exception,), {})
            mock_exc.__class__.__module__ = "requests.exceptions"
            mock_exc.__str__ = lambda: "unauthorized"
            mock_response = Mock()
            mock_response.status_code = 401
            mock_exc.response = mock_response
            mock_req_exc = MagicMock()
            mock_req_exc.HTTPError = mock_exc.__class__
            sys.modules['requests.exceptions'] = mock_req_exc

        result = handle_network_exception(mock_exc)

        assert isinstance(result, AuthenticationError)
        assert "unauthorized" in result.message
        assert result.error_code == "API_AUTH_ERROR"
