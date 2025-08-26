"""
外部APIクライアント URL構築
セキュアなURL構築とパラメータサニタイゼーション
"""

import urllib.parse
from typing import Dict, Any

from ...utils.logging_config import get_context_logger
from .models import APIRequest

# セキュア APIクライアント統合（オプショナル）
try:
    from ...api.secure_api_client import SecureErrorHandler, SecureURLBuilder

    SECURE_API_AVAILABLE = True
except ImportError:
    SECURE_API_AVAILABLE = False
    SecureErrorHandler = None
    SecureURLBuilder = None

logger = get_context_logger(__name__)


class URLBuilder:
    """URL構築クラス（セキュリティ強化版）"""

    def __init__(self, secure_url_builder: SecureURLBuilder = None):
        self.secure_url_builder = secure_url_builder
        self.request_stats = {"security_violations": 0, "sanitized_errors": 0}

    def build_url(self, request: APIRequest) -> str:
        """URL構築（セキュリティ強化版）"""
        try:
            # セキュアURL構築器が利用可能な場合
            if self.secure_url_builder and SECURE_API_AVAILABLE:
                # パスパラメータとクエリパラメータを分離
                path_params = {}
                query_params = {}

                for key, value in request.params.items():
                    placeholder = f"{{{key}}}"
                    if placeholder in request.endpoint.endpoint_url:
                        path_params[key] = str(value)
                    else:
                        query_params[key] = value

                # セキュアURL構築
                secure_url = self.secure_url_builder.build_secure_url(
                    base_url=request.endpoint.endpoint_url,
                    params=query_params if query_params else None,
                    path_params=path_params if path_params else None,
                )

                logger.debug(f"セキュアURL構築成功: {len(secure_url)}文字")
                return secure_url

            else:
                # フォールバック: 既存のセキュリティ機能を使用
                url = request.endpoint.endpoint_url

                # パラメータ置換（安全なエンコーディングで実施）
                for key, value in request.params.items():
                    placeholder = f"{{{key}}}"
                    if placeholder in url:
                        # セキュリティ強化: 適切なURLエンコーディングを適用
                        safe_value = self._sanitize_url_parameter(str(value), key)
                        url = url.replace(placeholder, safe_value)

                return url

        except ValueError as e:
            # セキュリティポリシー違反
            self.request_stats["security_violations"] += 1
            logger.error(f"URL構築セキュリティエラー: {e}")
            raise ValueError("URL構築に失敗しました: セキュリティポリシー違反")

        except Exception as e:
            logger.error(f"URL構築中に予期しないエラー: {e}")
            # セキュアエラーハンドリングで機密情報を除去
            if SECURE_API_AVAILABLE:
                safe_error_msg = SecureErrorHandler.sanitize_error_message(e, "URL構築")
                self.request_stats["sanitized_errors"] += 1
                raise ValueError(safe_error_msg)
            else:
                raise

    def _sanitize_url_parameter(self, value: str, param_name: str) -> str:
        """URLパラメータのサニタイゼーション（セキュリティ強化）"""
        # 1. 危険な文字パターンチェック
        dangerous_patterns = [
            "../",  # パストラバーサル攻撃
            "..\\",  # Windows パストラバーサル
            "%2e%2e",  # エンコード済みパストラバーサル
            "%2e%2e%2f",  # エンコード済みパストラバーサル
            "//",  # プロトコル相対URL
            "\\\\",  # UNCパス
            "\x00",  # NULLバイト
            "<",  # HTMLタグ
            ">",  # HTMLタグ
            "'",  # SQLインジェクション対策
            '"',  # SQLインジェクション対策
            "javascript:",  # JavaScriptスキーム
            "data:",  # データスキーム
            "file:",  # ファイルスキーム
            "ftp:",  # FTPスキーム
        ]

        # 危険パターン検出
        for pattern in dangerous_patterns:
            if pattern in value.lower():
                logger.warning(
                    f"危険なURLパラメータパターンを検出: {param_name}={value[:50]}..."
                )
                raise ValueError(
                    f"URLパラメータに危険な文字が含まれています: {param_name}"
                )

        # 2. パラメータ長さ制限
        if len(value) > 200:
            logger.warning(f"URLパラメータが長すぎます: {param_name}={len(value)}文字")
            raise ValueError(f"URLパラメータが長すぎます: {param_name}")

        # 3. 英数字・記号のみ許可（特殊用途パラメータ以外）
        if param_name in ["symbol", "ticker"]:
            # 株式コード用: 英数字・ピリオド・ハイフンのみ許可
            if not all(c.isalnum() or c in ".-" for c in value):
                logger.warning(f"不正な株式コード形式: {param_name}={value}")
                raise ValueError(
                    f"株式コードに不正な文字が含まれています: {param_name}"
                )

        # 4. URLエンコーディング適用
        try:
            encoded_value = urllib.parse.quote(value, safe="")
            logger.debug(
                f"URLパラメータエンコード: {param_name}: {value} -> {encoded_value}"
            )
            return encoded_value
        except Exception as e:
            logger.error(f"URLエンコーディングエラー: {param_name}={value}, error={e}")
            raise ValueError(f"URLパラメータのエンコードに失敗: {param_name}")

    def get_request_stats(self) -> Dict[str, int]:
        """リクエスト統計取得"""
        return self.request_stats.copy()