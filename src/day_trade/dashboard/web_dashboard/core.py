#!/usr/bin/env python3
"""
Webダッシュボード コアモジュール

WebDashboardクラスのメイン実装
"""

import os
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask
from flask_socketio import SocketIO

from ...utils.logging_config import get_context_logger
from ..dashboard_core import ProductionDashboard
from ..visualization_engine import DashboardVisualizationEngine

from .routes import setup_routes
from .websocket_handlers import setup_websocket_events
from .security import SecurityManager
from .template_generator import TemplateGenerator
from .static_generator import StaticFileGenerator

logger = get_context_logger(__name__)


class WebDashboard:
    """Webダッシュボード"""

    def __init__(self, port: int = 5000, debug: bool = False):
        """
        初期化

        Args:
            port: サーバーポート
            debug: デバッグモード
        """
        self.port = port
        self.debug = debug

        # Flask アプリケーション設定
        self.app = Flask(
            __name__,
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"),
        )
        
        # セキュリティ強化: 環境変数からSECRET_KEYを取得、なければランダム生成
        secret_key = os.environ.get("FLASK_SECRET_KEY")
        if not secret_key:
            secret_key = secrets.token_urlsafe(32)
            logger.warning(
                "FLASK_SECRET_KEY環境変数が未設定です。ランダムキーを生成しました。"
            )
            logger.info(
                "本番環境では環境変数を設定してください: "
                "export FLASK_SECRET_KEY='[32文字以上のランダム文字列]'"
            )

        self.app.config["SECRET_KEY"] = secret_key

        # WebSocket設定 - CORS制限
        cors_origins = os.environ.get(
            "DASHBOARD_CORS_ORIGINS", 
            "http://localhost:5000,http://127.0.0.1:5000"
        )
        allowed_origins = [origin.strip() for origin in cors_origins.split(",")]

        if debug:
            # デバッグモードでは開発用オリジンを許可
            allowed_origins.extend([
                "http://localhost:3000", 
                "http://127.0.0.1:3000"
            ])

        self.socketio = SocketIO(self.app, cors_allowed_origins=allowed_origins)
        logger.info(f"CORS許可オリジン: {allowed_origins}")

        # コアコンポーネント初期化
        self.dashboard_core = ProductionDashboard()
        self.visualization_engine = DashboardVisualizationEngine()
        self.analysis_app = None  # 遅延初期化

        # セキュリティマネージャー初期化
        self.security_manager = SecurityManager(self.app, debug)

        # 更新スレッド制御
        self.update_thread = None
        self.running = False

        # コンポーネント設定
        self._setup_components()

        logger.info(f"Webダッシュボード初期化完了 (ポート: {port})")

    def _setup_components(self):
        """各コンポーネントの設定"""
        # ルート設定
        setup_routes(
            self.app, 
            self.dashboard_core, 
            self.visualization_engine, 
            self.security_manager,
            self.socketio,
            self.debug
        )
        
        # WebSocketイベント設定
        setup_websocket_events(self.socketio, self.dashboard_core)
        
        # セキュリティヘッダー設定
        self.security_manager.setup_security_headers()

        # テンプレート・静的ファイル生成
        self._create_templates()
        self._create_static_files()

    def _create_templates(self):
        """HTMLテンプレート作成"""
        templates_dir = Path(self.app.template_folder)
        templates_dir.mkdir(exist_ok=True)

        template_generator = TemplateGenerator()
        template_generator.create_templates(templates_dir, self.security_manager)

    def _create_static_files(self):
        """静的ファイル作成"""
        static_dir = Path(self.app.static_folder)
        static_dir.mkdir(exist_ok=True)

        static_generator = StaticFileGenerator()
        static_generator.create_static_files(static_dir, self.security_manager)

    def start_monitoring(self):
        """監視開始"""
        logger.info("Webダッシュボード監視開始")
        self.dashboard_core.start_monitoring()
        self.running = True

        # リアルタイム更新スレッド開始
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def stop_monitoring(self):
        """監視停止"""
        logger.info("Webダッシュボード監視停止")
        self.running = False
        self.dashboard_core.stop_monitoring()

    def _update_loop(self):
        """リアルタイム更新ループ"""
        while self.running:
            try:
                # 現在のステータス取得
                status = self.dashboard_core.get_current_status()

                # WebSocket経由でクライアントに送信
                self.socketio.emit("dashboard_update", status)

            except Exception as e:
                logger.error(f"更新ループエラー: {e}")

            time.sleep(10)  # 10秒間隔で更新

    def run(self):
        """サーバー開始"""
        try:
            self.start_monitoring()
            logger.info(f"Webダッシュボードサーバー開始: http://localhost:{self.port}")
            self.socketio.run(
                self.app, host="0.0.0.0", port=self.port, debug=self.debug
            )
        except KeyboardInterrupt:
            logger.info("\nサーバー停止中...")
            self.stop_monitoring()
        except Exception as e:
            logger.error(f"サーバーエラー: {e}")
            self.stop_monitoring()


def main():
    """メイン実行"""
    logger.info("Webダッシュボード起動")
    logger.info("=" * 50)

    dashboard = WebDashboard(port=5000, debug=False)
    dashboard.run()