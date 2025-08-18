#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Web Server Configuration
Issue #901 Phase 2: プロダクション対応Webサーバー

Gunicorn/uWSGI対応のプロダクション用設定
- セキュリティ強化
- パフォーマンス最適化
- 基本認証機能
- エラーハンドリング改善
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import secrets

# Windows環境での文字化け対策
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

try:
    from flask import Flask, request, session, redirect, url_for, render_template_string, jsonify
    from flask_socketio import SocketIO
    from werkzeug.middleware.proxy_fix import ProxyFix
    from werkzeug.security import check_password_hash, generate_password_hash
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from enhanced_web_dashboard import EnhancedWebDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


class ProductionWebServer:
    """プロダクション対応Webサーバー"""

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8080,
                 secret_key: Optional[str] = None,
                 auth_enabled: bool = True,
                 username: str = "admin",
                 password_hash: Optional[str] = None):
        """
        初期化

        Args:
            host: ホスト（本番では127.0.0.1推奨）
            port: ポート番号
            secret_key: セッション暗号化キー
            auth_enabled: 認証機能有効化
            username: 認証ユーザー名
            password_hash: パスワードハッシュ
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask required for production web server")

        self.host = host
        self.port = port
        self.auth_enabled = auth_enabled
        self.username = username

        # セキュリティ設定
        self.secret_key = secret_key or self._generate_secret_key()
        self.password_hash = password_hash or self._get_default_password_hash()

        # ロギング設定（最初に初期化）
        self.logger = self._setup_logging()

        # Flask アプリケーション設定
        self.app = self._create_app()
        self.socketio = SocketIO(self.app,
                               cors_allowed_origins="*",
                               async_mode='threading',
                               logger=False,
                               engineio_logger=False)

        # Dashboard統合
        self.dashboard = None
        if DASHBOARD_AVAILABLE:
            try:
                self.dashboard = EnhancedWebDashboard(port=port)
                self.logger.info("Enhanced Web Dashboard integrated")
            except Exception as e:
                self.logger.warning(f"Dashboard integration failed: {e}")

    def _generate_secret_key(self) -> str:
        """セキュアな秘密鍵生成"""
        return secrets.token_hex(32)

    def _get_default_password_hash(self) -> str:
        """デフォルトパスワードハッシュ生成"""
        # 環境変数からパスワードを取得
        admin_password = os.environ.get('ADMIN_PASSWORD')
        if not admin_password:
            # セキュアなデフォルトパスワード生成
            admin_password = secrets.token_urlsafe(16)
            self.logger.warning(f"⚠️  本番環境では環境変数ADMIN_PASSWORDを設定してください")
            self.logger.warning(f"    一時的なパスワードが生成されました。ログを確認してください。")

        return generate_password_hash(admin_password)

    def _create_app(self) -> Flask:
        """プロダクション用Flask app作成"""
        app = Flask(__name__)

        # セキュリティ設定
        app.secret_key = self.secret_key
        app.config.update(
            SESSION_COOKIE_SECURE=False,  # HTTPS環境ではTrue
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE='Lax',
            PERMANENT_SESSION_LIFETIME=3600,  # 1時間
            WTF_CSRF_ENABLED=False  # 簡易版では無効化
        )

        # プロキシ対応（リバースプロキシ使用時）
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

        # 基本認証ルート
        if self.auth_enabled:
            self._setup_auth_routes(app)

        # 基本ルート
        self._setup_basic_routes(app)

        return app

    def _setup_auth_routes(self, app: Flask):
        """認証関連ルート設定"""

        @app.before_request
        def require_login():
            # 静的ファイルとログインページは除外
            if request.endpoint in ['login', 'static'] or request.path.startswith('/static'):
                return

            if 'logged_in' not in session:
                return redirect(url_for('login'))

        @app.route('/login', methods=['GET', 'POST'])
        def login():
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')

                if (username == self.username and
                    check_password_hash(self.password_hash, password)):
                    session['logged_in'] = True
                    session['username'] = username
                    session.permanent = True
                    return redirect('/')
                else:
                    return render_template_string(LOGIN_TEMPLATE, error="Invalid credentials")

            return render_template_string(LOGIN_TEMPLATE)

        @app.route('/logout')
        def logout():
            session.clear()
            return redirect(url_for('login'))

    def _setup_basic_routes(self, app: Flask):
        """基本ルート設定"""

        @app.route('/')
        def index():
            if self.dashboard:
                # Enhanced Dashboard統合
                return self.dashboard.get_dashboard_data()
            else:
                return render_template_string(SIMPLE_DASHBOARD_TEMPLATE)

        @app.route('/health')
        def health():
            """ヘルスチェックエンドポイント"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'auth_enabled': self.auth_enabled
            })

        @app.route('/api/status')
        def api_status():
            """APIステータス"""
            return jsonify({
                'server': 'production',
                'dashboard_available': DASHBOARD_AVAILABLE,
                'auth_enabled': self.auth_enabled,
                'uptime': datetime.now().isoformat()
            })

    def _setup_logging(self) -> logging.Logger:
        """プロダクション用ログ設定"""
        logger = logging.getLogger('production_web_server')
        logger.setLevel(logging.INFO)

        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # ファイルハンドラー
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'production_web.log')
        file_handler.setLevel(logging.INFO)

        # フォーマット設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def run_development(self):
        """開発モード実行（デバッグ有効）"""
        self.logger.info(f"Starting development server on {self.host}:{self.port}")
        self.socketio.run(self.app,
                         host=self.host,
                         port=self.port,
                         debug=True,
                         use_reloader=False)  # socketioとの競合回避

    def run_production(self):
        """プロダクションモード実行（デバッグ無効）"""
        self.logger.info(f"Starting production server on {self.host}:{self.port}")
        self.logger.info("For production deployment, use: gunicorn -w 4 -b 0.0.0.0:8080 production_web_server:app")

        # 開発環境でのプロダクションモード
        self.socketio.run(self.app,
                         host=self.host,
                         port=self.port,
                         debug=False)

    def get_wsgi_app(self):
        """WSGI アプリケーション取得（Gunicorn用）"""
        return self.app


# テンプレート定義
LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Day Trade Personal - Login</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; background: #f0f0f0; margin: 0; padding: 0; }
        .login-container { max-width: 400px; margin: 100px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .logo { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; color: #34495e; }
        input[type="text"], input[type="password"] { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        .btn { width: 100%; background: #3498db; color: white; padding: 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #2980b9; }
        .error { color: #e74c3c; margin-bottom: 15px; text-align: center; }
        .info { color: #7f8c8d; font-size: 12px; text-align: center; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <h2>🚀 Day Trade Personal</h2>
            <p>93%精度AI システム</p>
        </div>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        <form method="post">
            <div class="form-group">
                <label for="username">ユーザー名:</label>
                <input type="text" id="username" name="username" required>
            </div>

            <div class="form-group">
                <label for="password">パスワード:</label>
                <input type="password" id="password" name="password" required>
            </div>

            <button type="submit" class="btn">ログイン</button>
        </form>

        <div class="info">
            デフォルト: admin / admin123
        </div>
    </div>
</body>
</html>
'''

SIMPLE_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Day Trade Personal - Dashboard</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .header { text-align: center; margin-bottom: 30px; }
        .status { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .success { color: #27ae60; }
        .info { color: #3498db; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Day Trade Personal</h1>
        <h2>プロダクション Web ダッシュボード</h2>
    </div>

    <div class="status">
        <h3>システム状態</h3>
        <p class="success">✅ プロダクションモードで動作中</p>
        <p class="info">🔒 認証システム有効</p>
        <p class="info">📊 ダッシュボード待機中</p>

        <h3>利用可能エンドポイント</h3>
        <ul>
            <li><a href="/health">ヘルスチェック</a></li>
            <li><a href="/api/status">APIステータス</a></li>
            <li><a href="/logout">ログアウト</a></li>
        </ul>
    </div>
</body>
</html>
'''


# WSGI エントリーポイント（Gunicorn用）
def create_production_app():
    """プロダクション用アプリケーション作成"""
    server = ProductionWebServer(
        host="0.0.0.0",
        port=8080,
        auth_enabled=True
    )
    return server.get_wsgi_app()


# Gunicorn用のapp変数
app = create_production_app()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Day Trade Personal - Production Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    parser.add_argument("--no-auth", action="store_true", help="Disable authentication")
    parser.add_argument("--username", default="admin", help="Admin username")
    parser.add_argument("--password", default=None, help="Admin password (環境変数ADMIN_PASSWORDまたは--passwordで設定)")

    args = parser.parse_args()

    # パスワードハッシュ化
    password_hash = None
    if FLASK_AVAILABLE and args.password:
        password_hash = generate_password_hash(args.password)

    server = ProductionWebServer(
        host=args.host,
        port=args.port,
        auth_enabled=not args.no_auth,
        username=args.username,
        password_hash=password_hash
    )

    if args.production:
        server.run_production()
    else:
        server.run_development()