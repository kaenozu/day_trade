#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Web Server - プロダクション対応Webサーバー (リファクタリング後)
Issue #959対応: モジュール分割とアーキテクチャ改善
Issue #939対応: Redisによるキャッシュ機能追加
"""

import sys
import logging
import argparse
import redis
import json
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask
import threading
from datetime import datetime

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
        "version": "2.2.0",
        "version_extended": "2.2.0_cached_performance",
        "release_name": "Cached Performance",
        "build_date": "2025-08-19"
    }

class DayTradeWebServer:
    """プロダクション対応Webサーバー (リファクタリング・キャッシュ対応後)"""
    
    def __init__(self, port: int = 8000, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = Flask(__name__)
        self.app.secret_key = 'day-trade-personal-2025-refactored-cached'
        
        # Redisクライアント初期化
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("✅ Redis connection successful.")
        except redis.exceptions.ConnectionError as e:
            print(f"❌ Redis connection failed: {e}")
            print("⚠️ Running without cache. Performance will be degraded.")
            self.redis_client = None

        # サービス初期化 (Redisクライアントを渡す)
        self.recommendation_service = RecommendationService(redis_client=self.redis_client)
        self.template_service = TemplateService()
        self.version_info = VERSION_INFO
        
        # セッション管理
        self.session_id = f"web_refactored_cached_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # ルート設定
        self._setup_routes()
        
        # ログ設定
        if not debug:
            logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    def _setup_routes(self):
        """リファクタリング後のルート設定"""
        # メインルート設定
        setup_main_routes(self.app, self)
        
        # APIルート設定
        setup_api_routes(self.app, self)
        
        print(f"Routes configured for refactored DayTrade Web Server")
    
    def _get_dashboard_template(self) -> str:
        """ダッシュボードテンプレート取得（リファクタリング後）"""
        return self.template_service.get_dashboard_template()
    
    def _get_recommendations(self) -> list:
        """推奨銘柄取得（リファクタリング・キャッシュ対応後）"""
        return self.recommendation_service.get_recommendations()
    
    def _analyze_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """個別銘柄分析（リファクタリング・キャッシュ対応後）"""
        return self.recommendation_service.analyze_single_symbol(symbol)
    
    def run(self) -> None:
        """サーバー起動（リファクタリング後）"""
        print(f"\n🚀 Day Trade Web Server (Refactored & Cached) - Issue #939")
        print(f"Version: {self.version_info['version_extended']}")
        print(f"Port: {self.port}")
        print(f"Debug: {self.debug}")
        print(f"Architecture: Modular with Redis Caching")
        print(f"URL: http://localhost:{self.port}")
        print("=" * 50)
        
        try:
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=self.debug,
                threaded=True,
                use_reloader=False  # リロードを無効化（本番対応）
            )
        except KeyboardInterrupt:
            print("\n🛑 サーバーを停止しました")
        except Exception as e:
            print(f"❌ サーバーエラー: {e}")

def create_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサー作成"""
    parser = argparse.ArgumentParser(
        description='Day Trade Web Server (Refactored & Cached)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='サーバーポート (デフォルト: 8000)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='デバッグモード'
    )
    
    return parser

def main():
    """メイン実行関数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Webサーバー起動
    server = DayTradeWebServer(port=args.port, debug=args.debug)
    server.run()

if __name__ == "__main__":
    main()