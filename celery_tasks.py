#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery Tasks - Issue #939対応
バックグラウンドで実行する非同期タスクを定義します。
"""

import redis
from celery_app import celery_app
from web.services.recommendation_service import RecommendationService

# Workerプロセスごとに単一のRecommendationServiceインスタンスを保持するためのグローバル変数
_recommendation_service = None

def get_recommendation_service():
    """RecommendationServiceのシングルトンインスタンスを取得"""
    global _recommendation_service
    if _recommendation_service is None:
        print("Initializing RecommendationService for Celery worker...")
        try:
            # CeleryワーカーはWebサーバーとは別のプロセスで実行されるため、
            # 独自のRedisクライアントを初期化する必要があります。
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping() # 接続確認
            _recommendation_service = RecommendationService(redis_client=redis_client)
            print("RecommendationService initialized successfully.")
        except redis.exceptions.ConnectionError as e:
            print(f"Celery worker failed to connect to Redis: {e}")
            # Redisに接続できない場合でも、キャッシュなしでサービスを初期化
            _recommendation_service = RecommendationService(redis_client=None)
    return _recommendation_service

@celery_app.task(name='celery_tasks.run_analysis')
def run_analysis(symbol: str):
    """指定された銘柄コードの分析を非同期で実行するタスク"""
    try:
        service = get_recommendation_service()
        print(f"Celery task 'run_analysis' started for symbol: {symbol}")
        result = service.analyze_single_symbol(symbol)
        print(f"Celery task 'run_analysis' finished for symbol: {symbol}")
        return {'status': 'SUCCESS', 'result': result}
    except Exception as e:
        print(f"Error in Celery task for symbol {symbol}: {e}")
        # エラーが発生した場合、トレースバック情報を含めて返す
        import traceback
        return {
            'status': 'FAILURE',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
