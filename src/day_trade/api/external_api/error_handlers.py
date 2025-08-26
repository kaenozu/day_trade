#!/usr/bin/env python3
"""
外部APIクライアント - エラーハンドリング
"""

import re

from ...utils.logging_config import get_context_logger
from .models import APIResponse

# セキュアAPIクライアント機能のインポート
try:
    from ..secure_api_client import SecureErrorHandler
    SECURE_API_AVAILABLE = True
except ImportError:
    SECURE_API_AVAILABLE = False

logger = get_context_logger(__name__)


class ErrorHandler:
    """エラーハンドリングクラス"""

    def __init__(self):
        self.sanitized_error_count = 0

    def sanitize_error_message(self, error_message: str, error_type: str) -> str:
        """エラーメッセージの機密情報サニタイゼーション（セキュリティ強化）"""
        # セキュアエラーハンドラーが利用可能な場合
        if SECURE_API_AVAILABLE:
            try:
                # 高度なセキュリティサニタイゼーション
                sanitized_msg = SecureErrorHandler.sanitize_error_message(
                    Exception(error_message), f"API通信[{error_type}]"
                )
                self.sanitized_error_count += 1
                return sanitized_msg

            except Exception as e:
                logger.error(f"セキュアエラーハンドラーエラー: {e}")
                # フォールバック: 既存のサニタイゼーション

        # 従来のサニタイゼーション
        # 内部ログに詳細エラーを記録（セキュアなマスキング付き）
        try:
            from ...core.trade_manager import mask_sensitive_info

            logger.error(
                f"内部APIエラー詳細[{error_type}]: {mask_sensitive_info(error_message)}"
            )
        except ImportError:
            logger.error(f"内部APIエラー[{error_type}]: [マスキング機能無効]")

        # 公開用の安全なエラーメッセージ生成
        safe_messages = {
            "ClientError": "外部APIとの通信でエラーが発生しました",
            "TimeoutError": "外部APIからの応答がタイムアウトしました",
            "ConnectionError": "外部APIサーバーとの接続に失敗しました",
            "JSONDecodeError": "外部APIからの応答形式が不正です",
            "ValueError": "リクエストパラメータが不正です",
            "KeyError": "APIレスポンスの形式が予期しないものです",
            "default": "外部API処理でエラーが発生しました",
        }

        # エラータイプに応じた安全なメッセージを返す
        safe_message = safe_messages.get(error_type, safe_messages["default"])

        # 機密情報が含まれる可能性のあるパターンをチェック
        sensitive_patterns = [
            r"/[a-zA-Z]:/[^/]+",  # Windowsファイルパス
            r"/[^/]+/.+",  # Unixファイルパス
            r"[a-zA-Z0-9]{20,}",  # APIキー様文字列
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IPアドレス
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # メールアドレス
            r"[a-zA-Z]+://[^\s]+",  # URL
            r"(?:password|token|key|secret)[:=]\s*[^\s]+",  # 認証情報
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                logger.warning(
                    f"エラーメッセージに機密情報が含まれる可能性を検出: {error_type}"
                )
                # より汎用的なメッセージにさらに変更
                return f"{safe_message}（詳細はシステムログを確認してください）"

        # 一般的なエラーメッセージはそのまま返す
        self.sanitized_error_count += 1
        return safe_message

    def should_retry(self, response: APIResponse, attempt: int) -> bool:
        """リトライ判定"""
        # 成功レスポンスはリトライしない
        if response.success:
            return False

        # 最大リトライ数に達した場合はリトライしない
        if attempt >= response.request.endpoint.max_retries:
            return False

        # 4xx エラー（クライアントエラー）はリトライしない
        if 400 <= response.status_code < 500:
            logger.debug(
                f"クライアントエラーのためリトライしません: {response.status_code}"
            )
            return False

        # 5xx エラー（サーバーエラー）はリトライする
        if 500 <= response.status_code < 600:
            logger.debug(f"サーバーエラーのためリトライします: {response.status_code}")
            return True

        # ネットワークエラー（status_code == 0）はリトライする
        if response.status_code == 0:
            logger.debug("ネットワークエラーのためリトライします")
            return True

        # その他の場合はリトライしない
        return False

    def calculate_retry_delay(self, attempt: int, config: dict) -> float:
        """リトライ遅延時間計算"""
        base_delay = config.get("retry_delay_seconds", 1.0)
        use_exponential_backoff = config.get("exponential_backoff", True)
        max_backoff = config.get("max_backoff_seconds", 60.0)

        if use_exponential_backoff:
            # 指数バックオフ
            delay = base_delay * (2**attempt)
            return min(delay, max_backoff)
        else:
            return base_delay

    def categorize_error(self, error: Exception) -> str:
        """エラーの分類"""
        error_type = type(error).__name__

        # 一般的なエラータイプの分類
        if "timeout" in error_type.lower():
            return "TimeoutError"
        elif "connection" in error_type.lower():
            return "ConnectionError"
        elif "json" in error_type.lower():
            return "JSONDecodeError"
        elif "value" in error_type.lower():
            return "ValueError"
        elif "key" in error_type.lower():
            return "KeyError"
        else:
            return error_type

    def get_error_statistics(self) -> dict:
        """エラー統計情報の取得"""
        return {
            "sanitized_errors": self.sanitized_error_count,
            "secure_handler_available": SECURE_API_AVAILABLE,
        }