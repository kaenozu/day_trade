#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Web Server - プロダクション対応Webサーバー
Issue #939対応: Gunicorn対応のためのApplication Factoryパターン導入
"""

import os
import sys
import logging
import argparse
import redis
from pathlib import Path
from typing import Dict, Any
from flask import Flask, g
from datetime import datetime

from src.day_trade.utils.logging_config import setup_logging # NEW LINE

from celery_app import celery_app
from celery.result import AsyncResult
from web.routes.main_routes import setup_main_routes
from web.routes.api_routes import setup_api_routes
from web.services.recommendation_service import RecommendationService
from web.services.template_service import TemplateService

# バージョン情報
try:
    from version import get_version_info
    VERSION_INFO = get_version_info()
except ImportError:
    VERSION_INFO = {
        "version": "2.4.0",
        "version_extended": "2.4.0_production_ready",
        "release_name": "Production Ready",
        "build_date": "2025-08-19"
    }

def create_app() -> Flask:
    """Flaskアプリケーションを生成して設定 (Application Factory)"""
    app = Flask(__name__)
    # セキュアなsecret key設定
    import secrets
    secret_key = os.environ.get('FLASK_SECRET_KEY')
    if not secret_key:
        secret_key = secrets.token_urlsafe(32)
        print("WARNING:  本番環境では環境変数FLASK_SECRET_KEYを設定してください")
    app.secret_key = secret_key
    # --- バージョン情報をappコンテキストに保存 ---
    app.config['VERSION_INFO'] = VERSION_INFO

    # --- ログ設定 ---
    setup_logging()

    # --- サービスとクライアントの初期化 ---
    @app.before_request
    def before_request():
        if 'redis_client' not in g:
            try:
                g.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                g.redis_client.ping()
            except redis.exceptions.ConnectionError:
                g.redis_client = None

        if 'recommendation_service' not in g:
            g.recommendation_service = RecommendationService(redis_client=g.redis_client)

        if 'template_service' not in g:
            g.template_service = TemplateService()

    # --- 非同期タスクハンドラ ---
    def start_analysis_task(symbol: str) -> Dict[str, Any]:
        from celery_tasks import run_analysis
        task = run_analysis.delay(symbol)
        return {'task_id': task.id}

    def get_task_status(task_id: str) -> Dict[str, Any]:
        task_result = AsyncResult(task_id, app=celery_app)
        response = {
            'task_id': task_id,
            'status': task_result.status,
            'result': None
        }
        if task_result.successful():
            response['result'] = task_result.get()
        elif task_result.failed():
            response['result'] = task_result.get()
        return response

    # --- appコンテキストに関数を登録 ---
    app.start_analysis_task = start_analysis_task
    app.get_task_status = get_task_status

    # --- ルート設定 ---
    with app.app_context():
        setup_main_routes(app)
        setup_api_routes(app)
        print("Routes configured for DayTrade Web Server")

    # --- ログ設定 ---
    if not app.debug:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    # --- デバッグ: 登録ルートの表示 ---
    with app.app_context():
        print("--- Registered Routes ---")
        for rule in app.url_map.iter_rules():
            print(f"{rule.endpoint}: {rule.rule} Methods: {','.join(rule.methods)}")
        print("-----------------------")

    return app

# Gunicornがこの'app'インスタンスを使用する
app = create_app()

def main():
    """開発用サーバーを起動するメイン関数"""
    parser = argparse.ArgumentParser(description='Day Trade Web Server (Production Ready)')
    parser.add_argument('--port', '-p', type=int, default=8000, help='サーバーポート')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
    args = parser.parse_args()

    print(f"\nDay Trade Web Server (Production Ready) - Issue #939")
    print(f"Version: {app.config['VERSION_INFO']['version_extended']}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Architecture: Application Factory with Gunicorn & Celery")
    print(f"URL: http://localhost:{args.port}")
    print("To run in production with Gunicorn:")
    print("gunicorn --config gunicorn.conf.py daytrade_web:app")
    print("To run the Celery worker:")
    print("celery -A celery_app.celery_app worker --loglevel=info")
    print("=" * 50)

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug
    )

if __name__ == "__main__":
    main()