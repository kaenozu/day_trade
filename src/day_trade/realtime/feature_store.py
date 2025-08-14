#!/usr/bin/env python3
"""
リアルタイム特徴量ストア
Real-Time Feature Store with Redis Backend

Issue #763: リアルタイム特徴量生成と予測パイプライン Phase 3
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import redis.asyncio as redis
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from .feature_engine import FeatureValue, MarketDataPoint

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class FeatureStoreConfig:
    """特徴量ストア設定"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_ttl: int = 3600  # 1時間
    key_prefix: str = "day_trade:feature"
    max_connections: int = 20
    connection_pool_size: int = 10


@dataclass
class FeatureStoreMetrics:
    """特徴量ストアメトリクス"""
    reads_total: int = 0
    writes_total: int = 0
    deletes_total: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors_total: int = 0
    avg_read_time_ms: float = 0.0
    avg_write_time_ms: float = 0.0
    total_features_stored: int = 0


class FeatureSerializer:
    """特徴量データシリアライザー"""

    @staticmethod
    def serialize_feature(feature: FeatureValue) -> str:
        """特徴量をJSON文字列にシリアライズ"""
        data = asdict(feature)
        # datetimeをISO形式文字列に変換
        data['timestamp'] = feature.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def deserialize_feature(data: str) -> FeatureValue:
        """JSON文字列から特徴量を復元"""
        feature_dict = json.loads(data)
        # ISO形式文字列をdatetimeに変換
        feature_dict['timestamp'] = datetime.fromisoformat(feature_dict['timestamp'])
        return FeatureValue(**feature_dict)

    @staticmethod
    def serialize_market_data(market_data: MarketDataPoint) -> str:
        """市場データをJSON文字列にシリアライズ"""
        data = asdict(market_data)
        data['timestamp'] = market_data.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def deserialize_market_data(data: str) -> MarketDataPoint:
        """JSON文字列から市場データを復元"""
        data_dict = json.loads(data)
        data_dict['timestamp'] = datetime.fromisoformat(data_dict['timestamp'])
        return MarketDataPoint(**data_dict)


class FeatureKeyGenerator:
    """特徴量キー生成器"""

    def __init__(self, prefix: str = "day_trade:feature"):
        self.prefix = prefix

    def feature_key(self, symbol: str, feature_name: str, timestamp: Optional[datetime] = None) -> str:
        """特徴量キー生成"""
        if timestamp:
            ts_str = timestamp.strftime('%Y%m%d%H%M%S')
            return f"{self.prefix}:{symbol}:{feature_name}:{ts_str}"
        else:
            return f"{self.prefix}:{symbol}:{feature_name}:latest"

    def symbol_features_pattern(self, symbol: str) -> str:
        """銘柄の全特徴量パターン"""
        return f"{self.prefix}:{symbol}:*"

    def feature_history_pattern(self, symbol: str, feature_name: str) -> str:
        """特徴量履歴パターン"""
        return f"{self.prefix}:{symbol}:{feature_name}:*"

    def symbol_list_key(self) -> str:
        """銘柄リストキー"""
        return f"{self.prefix}:symbols"

    def feature_names_key(self, symbol: str) -> str:
        """特徴量名リストキー"""
        return f"{self.prefix}:{symbol}:feature_names"

    def metadata_key(self, key_type: str, identifier: str) -> str:
        """メタデータキー"""
        return f"{self.prefix}:meta:{key_type}:{identifier}"


class RealTimeFeatureStore:
    """リアルタイム特徴量ストア"""

    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.key_generator = FeatureKeyGenerator(config.key_prefix)
        self.serializer = FeatureSerializer()
        self.metrics = FeatureStoreMetrics()

        # Redis接続プール
        self.redis_pool = None
        self.redis_client = None

        # 内部キャッシュ（高速アクセス用）
        self.local_cache: Dict[str, Tuple[FeatureValue, float]] = {}  # (value, expiry_time)
        self.cache_ttl = 60  # ローカルキャッシュTTL（秒）

        logger.info("RealTimeFeatureStore initialized")

    async def connect(self) -> None:
        """Redis接続初期化"""
        try:
            # 接続プール作成
            self.redis_pool = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                max_connections=self.config.max_connections,
                decode_responses=True
            )

            # Redisクライアント作成
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)

            # 接続テスト
            await self.redis_client.ping()

            logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Redis接続切断"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.redis_pool:
                await self.redis_pool.disconnect()

            logger.info("Disconnected from Redis")

        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")

    async def store_feature(self,
                          feature: FeatureValue,
                          ttl: Optional[int] = None,
                          store_history: bool = True) -> bool:
        """特徴量を保存"""
        start_time = time.time()

        try:
            if not self.redis_client:
                await self.connect()

            ttl = ttl or self.config.default_ttl

            # 特徴量をシリアライズ
            serialized_data = self.serializer.serialize_feature(feature)

            # 最新値キー
            latest_key = self.key_generator.feature_key(feature.symbol, feature.name)

            # 履歴キー（タイムスタンプ付き）
            history_key = self.key_generator.feature_key(feature.symbol, feature.name, feature.timestamp)

            # パイプライン実行で高速化
            pipe = self.redis_client.pipeline()

            # 最新値保存
            pipe.setex(latest_key, ttl, serialized_data)

            # 履歴保存（必要に応じて）
            if store_history:
                pipe.setex(history_key, ttl * 24, serialized_data)  # 履歴は長期保存

            # 銘柄リストに追加
            pipe.sadd(self.key_generator.symbol_list_key(), feature.symbol)

            # 特徴量名リストに追加
            pipe.sadd(self.key_generator.feature_names_key(feature.symbol), feature.name)

            # 実行
            await pipe.execute()

            # ローカルキャッシュ更新
            cache_key = f"{feature.symbol}:{feature.name}"
            self.local_cache[cache_key] = (feature, time.time() + self.cache_ttl)

            # メトリクス更新
            self.metrics.writes_total += 1
            self.metrics.total_features_stored += 1

            write_time = (time.time() - start_time) * 1000
            if self.metrics.avg_write_time_ms == 0:
                self.metrics.avg_write_time_ms = write_time
            else:
                self.metrics.avg_write_time_ms = (self.metrics.avg_write_time_ms * 0.9) + (write_time * 0.1)

            return True

        except Exception as e:
            logger.error(f"Error storing feature {feature.name} for {feature.symbol}: {e}")
            self.metrics.errors_total += 1
            return False

    async def get_feature(self,
                         symbol: str,
                         feature_name: str,
                         timestamp: Optional[datetime] = None) -> Optional[FeatureValue]:
        """特徴量を取得"""
        start_time = time.time()

        try:
            if not self.redis_client:
                await self.connect()

            # ローカルキャッシュチェック
            cache_key = f"{symbol}:{feature_name}"
            if timestamp is None and cache_key in self.local_cache:
                cached_feature, expiry = self.local_cache[cache_key]
                if time.time() < expiry:
                    self.metrics.cache_hits += 1
                    return cached_feature
                else:
                    # 期限切れキャッシュ削除
                    del self.local_cache[cache_key]

            # Redisから取得
            if timestamp:
                key = self.key_generator.feature_key(symbol, feature_name, timestamp)
            else:
                key = self.key_generator.feature_key(symbol, feature_name)

            data = await self.redis_client.get(key)

            if data:
                feature = self.serializer.deserialize_feature(data)

                # ローカルキャッシュ更新（最新値のみ）
                if timestamp is None:
                    self.local_cache[cache_key] = (feature, time.time() + self.cache_ttl)

                self.metrics.cache_hits += 1
                return feature
            else:
                self.metrics.cache_misses += 1
                return None

        except Exception as e:
            logger.error(f"Error getting feature {feature_name} for {symbol}: {e}")
            self.metrics.errors_total += 1
            return None

        finally:
            read_time = (time.time() - start_time) * 1000
            if self.metrics.avg_read_time_ms == 0:
                self.metrics.avg_read_time_ms = read_time
            else:
                self.metrics.avg_read_time_ms = (self.metrics.avg_read_time_ms * 0.9) + (read_time * 0.1)

            self.metrics.reads_total += 1

    async def get_latest_features(self, symbol: str, feature_names: Optional[List[str]] = None) -> Dict[str, FeatureValue]:
        """最新の特徴量を一括取得"""
        try:
            if not self.redis_client:
                await self.connect()

            # 特徴量名リスト取得
            if feature_names is None:
                feature_names_key = self.key_generator.feature_names_key(symbol)
                feature_names = await self.redis_client.smembers(feature_names_key)
                feature_names = list(feature_names)

            if not feature_names:
                return {}

            # 一括取得用のキー生成
            keys = [self.key_generator.feature_key(symbol, name) for name in feature_names]

            # パイプラインで一括取得
            pipe = self.redis_client.pipeline()
            for key in keys:
                pipe.get(key)

            results = await pipe.execute()

            # 結果を辞書形式で返す
            features = {}
            for i, (name, data) in enumerate(zip(feature_names, results)):
                if data:
                    try:
                        feature = self.serializer.deserialize_feature(data)
                        features[name] = feature
                    except Exception as e:
                        logger.error(f"Error deserializing feature {name}: {e}")

            return features

        except Exception as e:
            logger.error(f"Error getting latest features for {symbol}: {e}")
            self.metrics.errors_total += 1
            return {}

    async def get_feature_history(self,
                                symbol: str,
                                feature_name: str,
                                start_time: datetime,
                                end_time: datetime,
                                limit: int = 1000) -> List[FeatureValue]:
        """特徴量履歴を取得"""
        try:
            if not self.redis_client:
                await self.connect()

            # パターンマッチでキー一覧取得
            pattern = self.key_generator.feature_history_pattern(symbol, feature_name)
            keys = []

            async for key in self.redis_client.scan_iter(match=pattern):
                # キーからタイムスタンプ抽出
                timestamp_str = key.split(':')[-1]
                if timestamp_str != 'latest':
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                        if start_time <= timestamp <= end_time:
                            keys.append(key)
                    except ValueError:
                        continue

            # ソートして制限
            keys.sort()
            if limit:
                keys = keys[-limit:]

            # 一括取得
            if keys:
                pipe = self.redis_client.pipeline()
                for key in keys:
                    pipe.get(key)

                results = await pipe.execute()

                features = []
                for data in results:
                    if data:
                        try:
                            feature = self.serializer.deserialize_feature(data)
                            features.append(feature)
                        except Exception as e:
                            logger.error(f"Error deserializing historical feature: {e}")

                return sorted(features, key=lambda f: f.timestamp)

            return []

        except Exception as e:
            logger.error(f"Error getting feature history for {symbol}:{feature_name}: {e}")
            self.metrics.errors_total += 1
            return []

    async def delete_feature(self, symbol: str, feature_name: str, timestamp: Optional[datetime] = None) -> bool:
        """特徴量を削除"""
        try:
            if not self.redis_client:
                await self.connect()

            if timestamp:
                key = self.key_generator.feature_key(symbol, feature_name, timestamp)
                result = await self.redis_client.delete(key)
            else:
                # 最新値と履歴を全削除
                latest_key = self.key_generator.feature_key(symbol, feature_name)
                pattern = self.key_generator.feature_history_pattern(symbol, feature_name)

                keys_to_delete = [latest_key]

                async for key in self.redis_client.scan_iter(match=pattern):
                    keys_to_delete.append(key)

                if keys_to_delete:
                    result = await self.redis_client.delete(*keys_to_delete)
                else:
                    result = 0

                # ローカルキャッシュからも削除
                cache_key = f"{symbol}:{feature_name}"
                self.local_cache.pop(cache_key, None)

            self.metrics.deletes_total += 1
            return result > 0

        except Exception as e:
            logger.error(f"Error deleting feature {feature_name} for {symbol}: {e}")
            self.metrics.errors_total += 1
            return False

    async def get_symbols(self) -> List[str]:
        """全銘柄リストを取得"""
        try:
            if not self.redis_client:
                await self.connect()

            symbols_key = self.key_generator.symbol_list_key()
            symbols = await self.redis_client.smembers(symbols_key)
            return list(symbols)

        except Exception as e:
            logger.error(f"Error getting symbols list: {e}")
            self.metrics.errors_total += 1
            return []

    async def get_feature_names(self, symbol: str) -> List[str]:
        """銘柄の特徴量名リストを取得"""
        try:
            if not self.redis_client:
                await self.connect()

            feature_names_key = self.key_generator.feature_names_key(symbol)
            feature_names = await self.redis_client.smembers(feature_names_key)
            return list(feature_names)

        except Exception as e:
            logger.error(f"Error getting feature names for {symbol}: {e}")
            self.metrics.errors_total += 1
            return []

    async def clear_symbol_data(self, symbol: str) -> bool:
        """銘柄データを全削除"""
        try:
            if not self.redis_client:
                await self.connect()

            # 全特徴量パターン
            pattern = self.key_generator.symbol_features_pattern(symbol)
            keys_to_delete = []

            async for key in self.redis_client.scan_iter(match=pattern):
                keys_to_delete.append(key)

            # 特徴量名リストキーも削除
            feature_names_key = self.key_generator.feature_names_key(symbol)
            keys_to_delete.append(feature_names_key)

            if keys_to_delete:
                await self.redis_client.delete(*keys_to_delete)

            # 銘柄リストから削除
            symbols_key = self.key_generator.symbol_list_key()
            await self.redis_client.srem(symbols_key, symbol)

            # ローカルキャッシュクリア
            cache_keys_to_remove = [k for k in self.local_cache.keys() if k.startswith(f"{symbol}:")]
            for cache_key in cache_keys_to_remove:
                del self.local_cache[cache_key]

            logger.info(f"Cleared all data for symbol: {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error clearing data for symbol {symbol}: {e}")
            self.metrics.errors_total += 1
            return False

    def get_metrics(self) -> FeatureStoreMetrics:
        """メトリクス取得"""
        return self.metrics

    def clear_local_cache(self) -> None:
        """ローカルキャッシュクリア"""
        self.local_cache.clear()
        logger.info("Local cache cleared")


# 使用例とテスト
async def test_feature_store():
    """特徴量ストアのテスト"""

    # 設定
    config = FeatureStoreConfig(
        redis_host="localhost",
        redis_port=6379,
        redis_db=1,  # テスト用DB
        default_ttl=3600
    )

    # 特徴量ストア初期化
    store = RealTimeFeatureStore(config)

    try:
        await store.connect()
        print("Connected to Redis successfully")

        # テストデータ作成
        symbol = "7203"
        features = [
            FeatureValue(
                name="sma_20",
                value=2100.50,
                timestamp=datetime.now(),
                symbol=symbol,
                metadata={"period": 20}
            ),
            FeatureValue(
                name="rsi_14",
                value=45.67,
                timestamp=datetime.now(),
                symbol=symbol,
                metadata={"period": 14}
            ),
            FeatureValue(
                name="macd",
                value=12.34,
                timestamp=datetime.now(),
                symbol=symbol,
                metadata={"macd_line": 12.34, "signal_line": 10.5}
            )
        ]

        print(f"\nStoring {len(features)} features for {symbol}...")

        # 特徴量保存
        for feature in features:
            success = await store.store_feature(feature)
            print(f"Stored {feature.name}: {success}")

        # 個別取得テスト
        print(f"\nRetrieving individual features:")
        for feature in features:
            retrieved = await store.get_feature(symbol, feature.name)
            if retrieved:
                print(f"{feature.name}: {retrieved.value} (original: {feature.value})")
            else:
                print(f"{feature.name}: Not found")

        # 一括取得テスト
        print(f"\nRetrieving all features at once:")
        all_features = await store.get_latest_features(symbol)
        for name, feature in all_features.items():
            print(f"{name}: {feature.value}")

        # 銘柄・特徴量名リスト取得
        symbols = await store.get_symbols()
        feature_names = await store.get_feature_names(symbol)
        print(f"\nSymbols: {symbols}")
        print(f"Feature names for {symbol}: {feature_names}")

        # メトリクス表示
        metrics = store.get_metrics()
        print(f"\nMetrics:")
        print(f"  Reads: {metrics.reads_total}")
        print(f"  Writes: {metrics.writes_total}")
        print(f"  Cache hits: {metrics.cache_hits}")
        print(f"  Cache misses: {metrics.cache_misses}")
        print(f"  Avg read time: {metrics.avg_read_time_ms:.2f}ms")
        print(f"  Avg write time: {metrics.avg_write_time_ms:.2f}ms")
        print(f"  Total features stored: {metrics.total_features_stored}")

        # クリーンアップ
        await store.clear_symbol_data(symbol)
        print(f"\nCleaned up test data for {symbol}")

    except Exception as e:
        print(f"Test failed: {e}")

    finally:
        await store.disconnect()
        print("Disconnected from Redis")


if __name__ == "__main__":
    asyncio.run(test_feature_store())