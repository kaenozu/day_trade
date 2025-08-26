#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider V2 - Cache Manager
リアルデータプロバイダー V2 - キャッシュ管理

データの効率的なキャッシングを行うモジュール
"""

import asyncio
import logging
import time
import pickle
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

# Redis依存関係の安全な読み込み
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class ImprovedCacheManager:
    """改善版キャッシュ管理"""

    def __init__(self, 
                 cache_dir: Path = Path("data/cache"), 
                 use_redis: bool = False,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0):
        """キャッシュマネージャー初期化
        
        Args:
            cache_dir: ファイルキャッシュディレクトリ
            use_redis: Redisキャッシュを使用するか
            redis_host: Redisホスト
            redis_port: Redisポート
            redis_db: Redis DB番号
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Redis設定
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db

        if self.use_redis:
            self._initialize_redis()

        # メモリキャッシュ
        self.memory_cache = {}
        self.cache_timestamps = {}
        self.max_memory_items = 1000

    def _initialize_redis(self):
        """Redis初期化"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.logger.info("Connected to Redis cache")
        except Exception as e:
            self.logger.warning(
                f"Redis connection failed, using file cache: {e}"
            )
            self.use_redis = False

    def _get_cache_key(self, symbol: str, period: str, source: str) -> str:
        """キャッシュキー生成"""
        key_str = f"{symbol}_{period}_{source}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get_cached_data(self, 
                              symbol: str, 
                              period: str, 
                              source: str,
                              ttl_seconds: int = 300) -> Optional[pd.DataFrame]:
        """キャッシュからデータ取得
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            source: データソース名
            ttl_seconds: キャッシュ有効期限（秒）
            
        Returns:
            キャッシュされたデータフレーム or None
        """
        cache_key = self._get_cache_key(symbol, period, source)

        try:
            # 1. メモリキャッシュ確認
            cached_data = await self._get_from_memory_cache(
                cache_key, ttl_seconds
            )
            if cached_data is not None:
                self.logger.debug(f"Memory cache hit: {symbol}")
                return cached_data

            # 2. Redis確認
            if self.use_redis and self.redis_client:
                cached_data = await self._get_from_redis_cache(cache_key)
                if cached_data is not None:
                    self.logger.debug(f"Redis cache hit: {symbol}")
                    # メモリキャッシュにも保存
                    self._store_in_memory(cache_key, cached_data)
                    return cached_data

            # 3. ファイルキャッシュ確認
            cached_data = await self._get_from_file_cache(
                cache_key, ttl_seconds
            )
            if cached_data is not None:
                self.logger.debug(f"File cache hit: {symbol}")
                # メモリキャッシュにも保存
                self._store_in_memory(cache_key, cached_data)
                return cached_data

            return None

        except Exception as e:
            self.logger.error(f"Cache get error for {symbol}: {e}")
            return None

    async def _get_from_memory_cache(self, 
                                   cache_key: str, 
                                   ttl_seconds: int) -> Optional[pd.DataFrame]:
        """メモリキャッシュからデータ取得"""
        if cache_key in self.memory_cache:
            cached_time = self.cache_timestamps.get(cache_key, 0)
            if time.time() - cached_time < ttl_seconds:
                return self.memory_cache[cache_key]
            else:
                # 期限切れのため削除
                del self.memory_cache[cache_key]
                del self.cache_timestamps[cache_key]
        return None

    async def _get_from_redis_cache(self, 
                                  cache_key: str) -> Optional[pd.DataFrame]:
        """Redisキャッシュからデータ取得"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"Redis cache read error: {e}")
        return None

    async def _get_from_file_cache(self, 
                                 cache_key: str, 
                                 ttl_seconds: int) -> Optional[pd.DataFrame]:
        """ファイルキャッシュからデータ取得"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < ttl_seconds:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"File cache read error: {e}")
                    cache_file.unlink(missing_ok=True)
        return None

    async def store_cached_data(self, 
                              symbol: str, 
                              period: str, 
                              source: str,
                              data: pd.DataFrame, 
                              ttl_seconds: int = 300):
        """キャッシュにデータ保存
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            source: データソース名
            data: 保存するデータフレーム
            ttl_seconds: キャッシュ有効期限（秒）
        """
        cache_key = self._get_cache_key(symbol, period, source)

        try:
            # 1. メモリキャッシュに保存
            self._store_in_memory(cache_key, data)

            # 2. Redis に保存
            if self.use_redis and self.redis_client:
                await self._store_to_redis(cache_key, data, ttl_seconds)

            # 3. ファイルキャッシュに保存
            await self._store_to_file(cache_key, data)

        except Exception as e:
            self.logger.error(f"Cache store error for {symbol}: {e}")

    def _store_in_memory(self, cache_key: str, data: pd.DataFrame):
        """メモリキャッシュ保存"""
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

    async def _store_to_redis(self, 
                            cache_key: str, 
                            data: pd.DataFrame, 
                            ttl_seconds: int):
        """Redisキャッシュ保存"""
        try:
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(cache_key, ttl_seconds, serialized_data)
            self.logger.debug("Stored in Redis cache")
        except Exception as e:
            self.logger.warning(f"Redis cache write error: {e}")

    async def _store_to_file(self, cache_key: str, data: pd.DataFrame):
        """ファイルキャッシュ保存"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug("Stored in file cache")
        except Exception as e:
            self.logger.warning(f"File cache write error: {e}")

    def clear_cache(self, symbol: Optional[str] = None):
        """キャッシュクリア
        
        Args:
            symbol: 特定銘柄のキャッシュクリア（None の場合は全クリア）
        """
        if symbol:
            self._clear_symbol_cache(symbol)
        else:
            self._clear_all_cache()

    def _clear_symbol_cache(self, symbol: str):
        """特定銘柄のキャッシュクリア"""
        # メモリキャッシュから削除
        keys_to_remove = []
        for key in self.memory_cache.keys():
            if symbol in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]

        # ファイルキャッシュから削除
        for cache_file in self.cache_dir.glob("*.pkl"):
            if symbol in cache_file.stem:
                cache_file.unlink(missing_ok=True)

        self.logger.info(f"Cleared cache for symbol: {symbol}")

    def _clear_all_cache(self):
        """全キャッシュクリア"""
        # メモリキャッシュクリア
        self.memory_cache.clear()
        self.cache_timestamps.clear()

        # Redisキャッシュクリア
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                self.logger.warning(f"Redis cache clear error: {e}")

        # ファイルキャッシュクリア
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)

        self.logger.info("Cleared all cache")

    def get_cache_stats(self) -> dict:
        """キャッシュ統計情報取得"""
        file_cache_count = len(list(self.cache_dir.glob("*.pkl")))
        
        stats = {
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_max': self.max_memory_items,
            'file_cache_size': file_cache_count,
            'redis_enabled': self.use_redis
        }

        if self.use_redis and self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats['redis_keys'] = redis_info.get('db0', {}).get('keys', 0)
                stats['redis_memory_usage'] = redis_info.get('used_memory_human', 'N/A')
            except Exception as e:
                self.logger.warning(f"Failed to get Redis stats: {e}")
                stats['redis_keys'] = 'Error'
                stats['redis_memory_usage'] = 'Error'

        return stats