#!/usr/bin/env python3
"""
外部APIクライアント - キャッシュ管理
"""

from datetime import datetime
from typing import Dict, Optional

from ...utils.logging_config import get_context_logger
from .models import APIConfig, APIRequest, APIResponse

logger = get_context_logger(__name__)


class CacheManager:
    """キャッシュ管理クラス"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.response_cache: Dict[str, APIResponse] = {}

    def generate_cache_key(self, request: APIRequest) -> str:
        """キャッシュキー生成"""
        key_parts = [
            request.endpoint.provider.value,
            request.endpoint.data_type.value,
            str(sorted(request.params.items())),
        ]
        
        # セキュリティ強化: MD5ハッシュの代替を使用
        try:
            from ...security.secure_hash_utils import replace_md5_hash
            return replace_md5_hash("|".join(key_parts))
        except ImportError:
            # フォールバック: 単純な文字列結合
            import hashlib
            return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]

    def get_cached_response(self, cache_key: str) -> Optional[APIResponse]:
        """キャッシュレスポンス取得"""
        if not self.config.enable_response_caching:
            return None

        if cache_key not in self.response_cache:
            return None

        cached_response = self.response_cache[cache_key]

        # TTLチェック
        age = (datetime.now() - cached_response.timestamp).total_seconds()
        if age > self.config.cache_ttl_seconds:
            del self.response_cache[cache_key]
            logger.debug(f"期限切れキャッシュエントリを削除: {cache_key[:16]}...")
            return None

        logger.debug(f"キャッシュヒット: {cache_key[:16]}... (年齢: {age:.1f}秒)")
        return cached_response

    def cache_response(self, cache_key: str, response: APIResponse) -> None:
        """レスポンスキャッシュ"""
        if not self.config.enable_response_caching:
            return

        # キャッシュサイズ制限
        if len(self.response_cache) >= self.config.cache_size_limit:
            # 最も古いエントリを削除（簡易LRU）
            oldest_key = min(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k].timestamp,
            )
            del self.response_cache[oldest_key]
            logger.debug(f"キャッシュサイズ制限により削除: {oldest_key[:16]}...")

        response.cache_key = cache_key
        self.response_cache[cache_key] = response
        logger.debug(
            f"レスポンスをキャッシュ: {cache_key[:16]}... "
            f"(サイズ: {len(self.response_cache)}/{self.config.cache_size_limit})"
        )

    def clear_cache(self) -> None:
        """キャッシュクリア"""
        cache_count = len(self.response_cache)
        self.response_cache.clear()
        logger.info(f"APIレスポンスキャッシュをクリア: {cache_count}エントリ削除")

    def get_cache_statistics(self) -> Dict[str, any]:
        """キャッシュ統計情報取得"""
        total_entries = len(self.response_cache)
        expired_count = 0
        current_time = datetime.now()

        for response in self.response_cache.values():
            age = (current_time - response.timestamp).total_seconds()
            if age > self.config.cache_ttl_seconds:
                expired_count += 1

        return {
            "total_entries": total_entries,
            "expired_entries": expired_count,
            "cache_size_limit": self.config.cache_size_limit,
            "cache_ttl_seconds": self.config.cache_ttl_seconds,
            "cache_enabled": self.config.enable_response_caching,
            "utilization_percent": (total_entries / self.config.cache_size_limit * 100)
            if self.config.cache_size_limit > 0
            else 0,
        }

    def cleanup_expired_entries(self) -> int:
        """期限切れエントリのクリーンアップ"""
        if not self.config.enable_response_caching:
            return 0

        expired_keys = []
        current_time = datetime.now()

        for cache_key, response in self.response_cache.items():
            age = (current_time - response.timestamp).total_seconds()
            if age > self.config.cache_ttl_seconds:
                expired_keys.append(cache_key)

        for key in expired_keys:
            del self.response_cache[key]

        if expired_keys:
            logger.info(f"期限切れキャッシュエントリを削除: {len(expired_keys)}件")

        return len(expired_keys)