#!/usr/bin/env python3
"""
外部APIクライアント - モジュール定義

分割前のexternal_api_client.pyとの後方互換性を保持
"""

# 公開APIのインポート
from .api_client import ExternalAPIClient
from .auth_manager import AuthenticationManager
from .cache_manager import CacheManager  
from .data_normalizers import DataNormalizer
from .enums import APIProvider, DataType, RequestMethod
from .error_handlers import ErrorHandler
from .models import APIConfig, APIEndpoint, APIRequest, APIResponse, RateLimitState
from .rate_limiter import RateLimiter
from .request_executor import RequestExecutor
from .url_builder import URLBuilder

# テスト用関数（元ファイルから移行）
import asyncio
from typing import Optional


async def setup_api_client() -> ExternalAPIClient:
    """APIクライアントセットアップ"""
    config = APIConfig(
        max_concurrent_requests=5, cache_ttl_seconds=300, default_max_retries=2
    )

    client = ExternalAPIClient(config)
    await client.initialize()

    return client


async def test_stock_data_fetching():
    """株価データ取得テスト"""
    client = await setup_api_client()

    try:
        # 複数銘柄のデータ取得テスト
        test_symbols = ["7203", "8306", "9984"]  # トヨタ、三菱UFJ、SBG

        for symbol in test_symbols:
            print(f"\n{symbol} データ取得テスト:")

            response = await client.fetch_stock_data(symbol, APIProvider.MOCK_PROVIDER)

            if response and response.success:
                print(f"  ✅ 成功: {response.response_time_ms:.1f}ms")
                if response.normalized_data is not None:
                    print(f"  📊 データ: {len(response.normalized_data)} レコード")
                    print(f"  💰 価格: {response.normalized_data['close'].iloc[0]:.2f}")
            else:
                print(
                    f"  ❌ 失敗: {response.error_message if response else 'レスポンスなし'}"
                )

        # 統計情報表示
        stats = client.get_request_statistics()
        print("\n📈 統計情報:")
        print(f"  総リクエスト数: {stats['total_requests']}")
        print(f"  成功率: {stats['success_rate_percent']:.1f}%")
        print(f"  キャッシュヒット率: {stats['cache_hit_rate_percent']:.1f}%")

    finally:
        await client.cleanup()


# 公開API定義
__all__ = [
    # メインクラス
    "ExternalAPIClient",
    
    # データモデル
    "APIConfig",
    "APIEndpoint", 
    "APIRequest",
    "APIResponse",
    "RateLimitState",
    
    # 列挙型
    "APIProvider",
    "DataType", 
    "RequestMethod",
    
    # マネージャークラス
    "AuthenticationManager",
    "CacheManager",
    "DataNormalizer", 
    "ErrorHandler",
    "RateLimiter",
    "RequestExecutor",
    "URLBuilder",
    
    # ユーティリティ関数
    "setup_api_client",
    "test_stock_data_fetching",
]