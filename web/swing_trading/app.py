#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Web Application
スイングトレード用Webアプリケーション
"""

import os
import secrets
from flask import Flask
from typing import Optional

from .routes import setup_routes


class SwingTradingWebUI:
    """スイングトレード Web UI アプリケーション"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5001):
        """
        初期化
        
        Args:
            host: ホストアドレス
            port: ポート番号
        """
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
        # セキュアなsecret key設定
        self._setup_secret_key()
        
        # 依存関係の取得
        self._setup_dependencies()
        
        # ルート設定
        self._setup_routes()

        print(f"Swing Trading Web UI initialized on {host}:{port}")

    def _setup_secret_key(self) -> None:
        """セキュアなSecret Key設定"""
        secret_key = os.environ.get('FLASK_SECRET_KEY')
        if not secret_key:
            secret_key = secrets.token_urlsafe(32)
            print("⚠️  本番環境では環境変数FLASK_SECRET_KEYを設定してください")
        self.app.secret_key = secret_key

    def _setup_dependencies(self) -> None:
        """依存関係の設定"""
        # スイングトレードスケジューラ
        try:
            from swing_trading_scheduler import swing_trading_scheduler
            self.swing_trading_scheduler = swing_trading_scheduler
        except ImportError:
            print("⚠️  Swing Trading Scheduler not available")
            self.swing_trading_scheduler = None

        # バージョン情報
        try:
            import version
            self.version_module = version
        except ImportError:
            print("⚠️  Version module not available")
            self.version_module = None

        # パフォーマンスモニター
        try:
            from performance_monitor import performance_monitor
            self.performance_monitor = performance_monitor
        except ImportError:
            print("⚠️  Performance Monitor not available")
            self.performance_monitor = None

        # 監査ログ
        try:
            from audit_logger import audit_logger
            self.audit_logger = audit_logger
        except ImportError:
            print("⚠️  Audit Logger not available")
            self.audit_logger = None

    def _setup_routes(self) -> None:
        """ルート設定"""
        setup_routes(
            self.app,
            self.swing_trading_scheduler,
            self.version_module,
            self.audit_logger
        )

    def run(self, debug: bool = False) -> None:
        """
        Webサーバー実行
        
        Args:
            debug: デバッグモード
        """
        print(f"Starting Swing Trading Web UI on http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


def create_app(host: str = "127.0.0.1", port: int = 5001) -> SwingTradingWebUI:
    """
    Swing Trading Web UIアプリケーション作成
    
    Args:
        host: ホストアドレス
        port: ポート番号
        
    Returns:
        SwingTradingWebUI: アプリケーションインスタンス
    """
    return SwingTradingWebUI(host=host, port=port)