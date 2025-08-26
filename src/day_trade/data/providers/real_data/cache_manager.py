#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Cache Manager

改善版キャッシュ管理モジュール
"""

import logging
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class ImprovedCacheManager:
    """改善版キャッシュ管理"""

    def __init__(
        self,
        cache_dir: Path = Path("data/cache"),
        use_redis: bool = False
    ):
        """初期化
        
        Args:
            cache_dir: キャッシュディレクトリのパス
            use_redis: Redisを使用するかどうか
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Redis設定
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.redis_client = None

        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host='localhost', port=6379, db=0, decode_responses=False
                )
                self.redis_client.ping()
                self.logger.info("Connected to Redis cache")
            except Exception as e:
                self.logger.warning(
                    f"Redis connection failed, using file cache: {e}"
                )
                self.use_redis = False

        # メモリキャッシュ
        self.memory_cache = {}
        self.cache_timestamps = {}
        self.max_memory_items = 1000

    def _get_cache_key(self, symbol: str, period: str, source: str) -> str:
        """キャッシュキー生成
        
        Args:
            symbol: 銘柄コード
            period: データ期間
            source: データソース
            
        Returns:
            ハッシュ化されたキャッシュキー
        """
        key_str = f"{symbol}_{period}_{source}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get_cached_data(
        self,
        symbol: str,
        period: str,
        source: str,
        ttl_seconds: int = 300
    ) -> Optional[pd.DataFrame]:
        """キャッシュからデータ取得
        
        Args:
            symbol: 銘柄コード
            period: データ期間
            source: データソース
            ttl_seconds: TTL（秒）
            
        Returns:
            キャッシュされたデータまたはNone
        """
        cache_key = self._get_cache_key(symbol, period, source)

        try:
            # 1. メモリキャッシュ確認
            if cache_key in self.memory_cache:
                cached_time = self.cache_timestamps.get(cache_key, 0)
                if time.time() - cached_time < ttl_seconds:
                    self.logger.debug(f"Memory cache hit: {symbol}")
                    return self.memory_cache[cache_key]
                else:
                    # 期限切れのため削除
                    del self.memory_cache[cache_key]
                    del self.cache_timestamps[cache_key]

            # 2. Redis確認
            if self.use_redis and self.redis_client:
                try:
                    cached_data = self.redis_client.get(cache_key)
                    if cached_data:
                        data = pickle.loads(cached_data)
                        self.logger.debug(f"Redis cache hit: {symbol}")
                        # メモリキャッシュにも保存
                        self._store_in_memory(cache_key, data)
                        return data
                except Exception as e:
                    self.logger.warning(f"Redis cache read error: {e}")

            # 3. ファイルキャッシュ確認
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < ttl_seconds:
                    try:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                        self.logger.debug(f"File cache hit: {symbol}")
                        # メモリキャッシュにも保存
                        self._store_in_memory(cache_key, data)
                        return data
                    except Exception as e:
                        self.logger.warning(f"File cache read error: {e}")
                        cache_file.unlink(missing_ok=True)

            return None

        except Exception as e:
            self.logger.error(f"Cache get error for {symbol}: {e}")
            return None

    async def store_cached_data(
        self,
        symbol: str,
        period: str,
        source: str,
        data: pd.DataFrame,
        ttl_seconds: int = 300
    ):
        """キャッシュにデータ保存
        
        Args:
            symbol: 銘柄コード
            period: データ期間
            source: データソース
            data: 保存するデータ
            ttl_seconds: TTL（秒）
        """
        cache_key = self._get_cache_key(symbol, period, source)

        try:
            # 1. メモリキャッシュに保存
            self._store_in_memory(cache_key, data)

            # 2. Redis に保存
            if self.use_redis and self.redis_client:
                try:
                    serialized_data = pickle.dumps(data)
                    self.redis_client.setex(
                        cache_key, ttl_seconds, serialized_data
                    )
                    self.logger.debug(f"Stored in Redis cache: {symbol}")
                except Exception as e:
                    self.logger.warning(f"Redis cache write error: {e}")

            # 3. ファイルキャッシュに保存
            try:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                self.logger.debug(f"Stored in file cache: {symbol}")
            except Exception as e:
                self.logger.warning(f"File cache write error: {e}")

        except Exception as e:
            self.logger.error(f"Cache store error for {symbol}: {e}")

    def _store_in_memory(self, cache_key: str, data: pd.DataFrame):
        """メモリキャッシュ保存
        
        Args:
            cache_key: キャッシュキー
            data: 保存するデータ
        """
        # メモリキャッシュサイズ制限
        if len(self.memory_cache) >= self.max_memory_items:
            # 最も古いエントリを削除
            oldest_key = min(
                self.cache_timestamps.keys(),
                key=lambda k: self.cache_timestamps[k]
            )
            del self.memory_cache[oldest_key]
            del self.cache_timestamps[oldest_key]

        self.memory_cache[cache_key] = data.copy()
        self.cache_timestamps[cache_key] = time.time()

    def clear_cache(self, symbol: Optional[str] = None):
        """キャッシュクリア
        
        Args:
            symbol: クリア対象の銘柄コード（Noneの場合は全クリア）
        """
        if symbol:
            # 特定銘柄のキャッシュクリア
            keys_to_remove = []
            for key in self.memory_cache.keys():
                if symbol in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]

            self.logger.info(f"Cleared cache for symbol: {symbol}")
        else:
            # 全キャッシュクリア
            self.memory_cache.clear()
            self.cache_timestamps.clear()

            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    self.logger.warning(f"Redis cache clear error: {e}")

            self.logger.info("Cleared all cache")

    def get_cache_stats(self) -> Dict[str, int]:
        """キャッシュ統計情報を取得
        
        Returns:
            キャッシュ統計情報の辞書
        """
        return {
            'memory_cache_size': len(self.memory_cache),
            'max_memory_items': self.max_memory_items,
            'redis_available': self.use_redis,
            'cache_dir_files': len(list(self.cache_dir.glob('*.pkl')))
        }