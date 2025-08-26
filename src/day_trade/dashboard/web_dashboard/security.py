#!/usr/bin/env python3
"""
Webダッシュボード セキュリティモジュール

セキュリティヘッダー、バリデーション、エラーハンドリング機能
"""

import os
from pathlib import Path

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SecurityManager:
    """セキュリティ管理クラス"""

    def __init__(self, app, debug: bool = False):
        """
        初期化

        Args:
            app: Flaskアプリケーション
            debug: デバッグモード
        """
        self.app = app
        self.debug = debug

    def setup_security_headers(self):
        """セキュリティヘッダー設定"""

        @self.app.after_request
        def set_security_headers(response):
            # XSS攻撃対策
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # HTTPS強制（本番環境）
            if not self.debug:
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains"
                )

            # CSP（Content Security Policy）
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; "
                "font-src 'self' cdnjs.cloudflare.com; "
                "img-src 'self' data:; "
                "connect-src 'self'"
            )
            response.headers["Content-Security-Policy"] = csp

            return response

    def sanitize_error_message(self, error: Exception) -> str:
        """エラーメッセージのサニタイズ"""
        # セキュリティ: 詳細なエラー情報を隠蔽し、一般的なメッセージを返す
        error_str = str(error).lower()

        # 機密情報が含まれる可能性のあるエラーパターン
        sensitive_patterns = [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "database",
            "connection",
            "path",
            "file",
            "directory",
        ]

        for pattern in sensitive_patterns:
            if pattern in error_str:
                return (
                    "システム内部エラーが発生しました。管理者にお問い合わせください。"
                )

        # デバッグモード時のみ詳細表示
        if self.debug:
            return str(error)

        return "処理中にエラーが発生しました。"

    def validate_metric_type(self, metric_type: str) -> bool:
        """メトリクスタイプの検証"""
        allowed_metrics = [
            "portfolio",
            "system",
            "trading",
            "risk",
            "performance",
            "alerts",
            "status",
        ]
        return metric_type in allowed_metrics

    def validate_chart_type(self, chart_type: str) -> bool:
        """チャートタイプの検証"""
        allowed_charts = ["portfolio", "system", "trading", "risk", "comprehensive"]
        return chart_type in allowed_charts

    def validate_hours_parameter(self, hours: int) -> bool:
        """時間パラメータの検証"""
        # 1時間から30日間（720時間）までを許可
        return 1 <= hours <= 720

    def validate_tier_parameter(self, tier: int) -> bool:
        """ティアパラメータの検証"""
        return tier in [1, 2, 3, 4]

    def create_secure_file(
        self, file_path: Path, content: str, permissions: int = 0o644
    ):
        """セキュアなファイル作成"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # ファイル権限設定（Unix系OS）
            if os.name != "nt":  # Windows以外
                os.chmod(file_path, permissions)
                logger.info(f"ファイル権限設定: {file_path} -> {oct(permissions)}")
            else:
                logger.info(
                    f"ファイル作成: {file_path} (Windows環境のため権限設定スキップ)"
                )

        except Exception as e:
            logger.error(f"セキュアファイル作成エラー {file_path}: {e}")
            raise