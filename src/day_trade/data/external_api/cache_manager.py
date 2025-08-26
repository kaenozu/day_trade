"""
外部APIクライアント キャッシュ管理
レスポンスキャッシュとキー生成機能
"""

from datetime import datetime
from typing import Dict, Optional

from ...utils.logging_config import get_context_logger
from .models import APIRequest, APIResponse

logger = get_context_logger(__name__)


class CacheManager:
    """キャッシュ管理クラス"""

    def __init__(self, enable_caching: bool = True, cache_ttl_seconds: int = 300, 
                 cache_size_limit: int = 1000):
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_size_limit = cache_size_limit
        self.response_cache: Dict[str, APIResponse] = {}

    def generate_cache_key(self, request: APIRequest) -> str:
        """キャッシュキー生成"""
        key_parts = [
            request.endpoint.provider.value,
            request.endpoint.data_type.value,
            str(sorted(request.params.items())),
        ]
        
        try:
            from ...security.secure_hash_utils import replace_md5_hash
            return replace_md5_hash("|".join(key_parts))
        except ImportError:
            # フォールバック: 単純な結合
            return "|".join(key_parts)

    def get_cached_response(self, cache_key: str) -> Optional[APIResponse]:
        """キャッシュレスポンス取得"""
        if not self.enable_caching:
            return None

        if cache_key not in self.response_cache:
            return None

        cached_response = self.response_cache[cache_key]

        # TTLチェック
        age = (datetime.now() - cached_response.timestamp).total_seconds()
        if age > self.cache_ttl_seconds:
            del self.response_cache[cache_key]
            return None

        return cached_response

    def cache_response(self, cache_key: str, response: APIResponse) -> None:
        """レスポンスキャッシュ"""
        if not self.enable_caching:
            return

        # キャッシュサイズ制限
        if len(self.response_cache) >= self.cache_size_limit:
            # 最も古いエントリを削除（簡易LRU）
            oldest_key = min(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k].timestamp,
            )
            del self.response_cache[oldest_key]

        response.cache_key = cache_key
        self.response_cache[cache_key] = response

    async def clear_cache(self) -> None:
        """キャッシュクリア"""
        self.response_cache.clear()
        logger.info("APIレスポンスキャッシュをクリアしました")

    def get_cache_stats(self) -> Dict[str, any]:
        """キャッシュ統計取得"""
        return {
            "cache_enabled": self.enable_caching,
            "cache_entries": len(self.response_cache),
            "cache_size_limit": self.cache_size_limit,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }