#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Web Server - プロダクション対応Webサーバー (リファクタリング後)
Issue #959対応: モジュール分割とアーキテクチャ改善
Issue #939対応: Redisキャッシュ & Celery非同期タスク追加
"""

import sys
import logging
import argparse
import redis
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask
from datetime import datetime

# 非同期タスクのインポート
from celery_app import celery_app
from celery_tasks import run_analysis
from celery.result import AsyncResult

# リファクタリング後のモジュールインポート
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
        "version": "2.3.0",
        "version_extended": "2.3.0_async_performance",
        "release_name": "Async Performance",
        "build_date": "2025-08-19"
    }

class DayTradeWebServer:
    """プロダクション対応Webサーバー (非同期タスク対応)"""
    
    def __init__(self, port: int = 8000, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = Flask(__name__)
        self.app.secret_key = 'day-trade-personal-2025-refactored-async'
        
        # Redisクライアント初期化 (キャッシュ用)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("✅ Redis connection successful.")
        except redis.exceptions.ConnectionError as e:
            print(f"❌ Redis connection failed: {e}")
            print("⚠️ Running without cache. Performance will be degraded.")
            self.redis_client = None

        # サービス初期化 (キャッシュ用Redisクライアントを渡す)
        self.recommendation_service = RecommendationService(redis_client=self.redis_client)
        self.template_service = TemplateService()
        self.version_info = VERSION_INFO
        
        # セッション管理
        self.session_id = f"web_refactored_async_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # ルート設定
        self._setup_routes()
        
        # ログ設定
        if not debug:
            logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    def _setup_routes(self):
        """ルート設定"""
        setup_main_routes(self.app, self)
        setup_api_routes(self.app, self)
        print(f"Routes configured for DayTrade Web Server")
    
    def _get_dashboard_template(self) -> str:
        """ダッシュボードテンプレート取得"""
        return self.template_service.get_dashboard_template()
    
    def _get_recommendations(self) -> list:
        """推奨銘柄取得 (キャッシュ利用)"""
        return self.recommendation_service.get_recommendations()
    
    def _start_analysis_task(self, symbol: str) -> Dict[str, Any]:
        """個別銘柄分析の非同期タスクを開始"""
        task = run_analysis.delay(symbol)
        return {'task_id': task.id}

    def _get_task_status(self, task_id: str) -> Dict[str, Any]:
        """非同期タスクのステータスと結果を取得"""
        task_result = AsyncResult(task_id, app=celery_app)
        
        response = {
            'task_id': task_id,
            'status': task_result.status,
            'result': None
        }
        
        if task_result.successful():
            response['result'] = task_result.get()
        elif task_result.failed():
            # エラー情報を取得
            result = task_result.get()
            response['result'] = result # resultにはエラー情報が含まれる
            
        return response

    def run(self) -> None:
        """サーバー起動"""
        print(f"\n🚀 Day Trade Web Server (Async) - Issue #939")
        print(f"Version: {self.version_info['version_extended']}")
        print(f"Port: {self.port}")
        print(f"Debug: {self.debug}")
        print(f"Architecture: Modular with Redis Caching & Celery Tasks")
        print(f"URL: http://localhost:{self.port}")
        print("To run the Celery worker:")
        print("celery -A celery_app.celery_app worker --loglevel=info")
        print("=" * 50)
        
        try:
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=self.debug,
                threaded=True,
                use_reloader=False
            )
        except KeyboardInterrupt:
            print("\n🛑 サーバーを停止しました")
        except Exception as e:
            print(f"❌ サーバーエラー: {e}")

def create_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサー作成"""
    parser = argparse.ArgumentParser(
        description='Day Trade Web Server (Async)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--port', '-p', type=int, default=8000, help='サーバーポート')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
    return parser

def main():
    """メイン実行関数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    server = DayTradeWebServer(port=args.port, debug=args.debug)
    server.run()

if __name__ == "__main__":
    main()
