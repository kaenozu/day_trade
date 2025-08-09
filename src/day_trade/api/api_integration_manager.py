#!/usr/bin/env python3
"""
API統合マネージャーシステム
Issue #331: API・外部統合システム - Phase 3

RESTful・WebSocket API統合管理・データ統一・キャッシュ・フォールバック
- マルチソースデータ統合
- 自動フェイルオーバー
- 統一データインターフェース
- インテリジェントキャッシュ管理
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..utils.logging_config import get_context_logger
    from .external_api_client import APIProvider as RestAPIProvider
    from .external_api_client import APIResponse, ExternalAPIClient
    from .websocket_streaming_client import (
        StreamMessage,
        StreamProvider,
        StreamType,
        WebSocketStreamingClient,
    )
except ImportError:
    # テスト環境や一部のモジュールが不足している場合のフォールバック
    ExternalAPIClient = None
    RestAPIProvider = None
    APIResponse = None
    WebSocketStreamingClient = None
    StreamProvider = None
    StreamMessage = None
    StreamType = None

    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DataSource(Enum):
    """データソース"""

    REST_API = "rest_api"
    WEBSOCKET_STREAM = "websocket_stream"
    CACHE = "cache"
    FALLBACK = "fallback"


class DataQuality(Enum):
    """データ品質"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNKNOWN = "unknown"


class DataFreshness(Enum):
    """データ鮮度"""

    REAL_TIME = "real_time"  # <5秒
    FRESH = "fresh"  # <1分
    RECENT = "recent"  # <5分
    STALE = "stale"  # <15分
    EXPIRED = "expired"  # >15分
    UNKNOWN = "unknown"  # 不明


@dataclass
class UnifiedMarketData:
    """統合市場データ"""

    symbol: str
    data_type: str
    timestamp: datetime

    # 価格データ
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: Optional[int] = None

    # メタデータ
    source: DataSource = DataSource.REST_API
    provider: str = "unknown"
    quality: DataQuality = DataQuality.UNKNOWN
    freshness: DataFreshness = DataFreshness.UNKNOWN
    confidence_score: float = 0.0

    # 拡張データ
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        return {
            "symbol": self.symbol,
            "data_type": self.data_type,
            "timestamp": self.timestamp.isoformat(),
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "close_price": self.close_price,
            "volume": self.volume,
            "source": self.source.value,
            "provider": self.provider,
            "quality": self.quality.value,
            "freshness": self.freshness.value,
            "confidence_score": self.confidence_score,
            **self.additional_data,
        }

    def to_pandas_row(self) -> Dict[str, Any]:
        """Pandas行形式変換"""
        return {
            "Symbol": self.symbol,
            "Timestamp": self.timestamp,
            "Open": self.open_price,
            "High": self.high_price,
            "Low": self.low_price,
            "Close": self.close_price,
            "Volume": self.volume,
            "Source": self.source.value,
            "Provider": self.provider,
            "Quality": self.quality.value,
            "Confidence": self.confidence_score,
        }


@dataclass
class DataSourceConfig:
    """データソース設定"""

    source_type: DataSource
    provider: str
    enabled: bool = True
    priority: int = 1  # 1=最高優先度

    # 品質設定
    min_confidence_score: float = 0.5
    max_staleness_minutes: int = 15

    # パフォーマンス設定
    timeout_seconds: int = 30
    retry_attempts: int = 3

    # フォールバック設定
    fallback_providers: List[str] = field(default_factory=list)


@dataclass
class CacheEntry:
    """キャッシュエントリ"""

    key: str
    data: UnifiedMarketData
    created_at: datetime
    ttl_seconds: int
    access_count: int = 0
    last_access: Optional[datetime] = None

    def is_expired(self) -> bool:
        """期限切れ判定"""
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def access(self) -> None:
        """アクセス記録"""
        self.access_count += 1
        self.last_access = datetime.now()


@dataclass
class IntegrationConfig:
    """統合設定"""

    # データソース優先度
    default_source_priority: List[DataSource] = field(
        default_factory=lambda: [
            DataSource.WEBSOCKET_STREAM,
            DataSource.REST_API,
            DataSource.CACHE,
            DataSource.FALLBACK,
        ]
    )

    # キャッシュ設定
    enable_intelligent_caching: bool = True
    cache_size_limit: int = 10000
    default_cache_ttl_seconds: int = 300  # 5分

    # 品質管理設定
    enable_data_quality_scoring: bool = True
    min_acceptable_quality: DataQuality = DataQuality.ACCEPTABLE
    enable_data_validation: bool = True

    # フェイルオーバー設定
    enable_automatic_fallback: bool = True
    max_fallback_attempts: int = 3
    fallback_delay_seconds: float = 1.0

    # 統合設定
    enable_data_fusion: bool = True  # 複数ソース統合
    fusion_confidence_threshold: float = 0.8
    max_concurrent_requests: int = 10


class APIIntegrationManager:
    """API統合マネージャー"""

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()

        # APIクライアント
        self.rest_client: Optional[ExternalAPIClient] = None
        self.websocket_client: Optional[WebSocketStreamingClient] = None

        # データソース設定
        self.data_sources: Dict[str, DataSourceConfig] = {}

        # キャッシュシステム
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "total_size": 0}

        # ストリーミングデータ
        self.streaming_subscriptions: Dict[str, str] = {}  # symbol -> subscription_id
        self.real_time_data: Dict[str, UnifiedMarketData] = {}

        # 統計情報
        self.integration_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0,
            "data_quality_issues": 0,
            "fusion_operations": 0,
        }

        # 制御フラグ
        self._initialized = False
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # デフォルト設定
        self._setup_default_data_sources()

    def _setup_default_data_sources(self) -> None:
        """デフォルトデータソース設定"""
        default_sources = [
            DataSourceConfig(
                source_type=DataSource.WEBSOCKET_STREAM,
                provider="mock_stream",
                priority=1,
                max_staleness_minutes=1,
                timeout_seconds=10,
            ),
            DataSourceConfig(
                source_type=DataSource.REST_API,
                provider="mock_provider",
                priority=2,
                max_staleness_minutes=5,
                timeout_seconds=30,
                fallback_providers=["yahoo_finance", "alpha_vantage"],
            ),
            DataSourceConfig(
                source_type=DataSource.CACHE,
                provider="intelligent_cache",
                priority=3,
                max_staleness_minutes=15,
            ),
        ]

        for source in default_sources:
            self.register_data_source(source)

    def register_data_source(self, source_config: DataSourceConfig) -> None:
        """データソース登録"""
        source_key = f"{source_config.source_type.value}_{source_config.provider}"
        self.data_sources[source_key] = source_config
        logger.info(f"データソース登録: {source_key}")

    async def initialize(self) -> None:
        """統合マネージャー初期化"""
        if self._initialized:
            return

        try:
            # RESTクライアント初期化
            self.rest_client = ExternalAPIClient()
            await self.rest_client.initialize()

            # WebSocketクライアント初期化
            self.websocket_client = WebSocketStreamingClient()
            await self.websocket_client.start_streaming()

            self._initialized = True
            logger.info("API統合マネージャー初期化完了")

        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            raise

    async def cleanup(self) -> None:
        """リソースクリーンアップ"""
        if self.rest_client:
            await self.rest_client.cleanup()

        if self.websocket_client:
            await self.websocket_client.stop_streaming()

        self._initialized = False
        logger.info("API統合マネージャークリーンアップ完了")

    async def get_market_data(
        self,
        symbol: str,
        data_type: str = "stock_price",
        prefer_real_time: bool = True,
        max_staleness_minutes: int = 5,
        **kwargs,
    ) -> Optional[UnifiedMarketData]:
        """統合市場データ取得"""
        if not self._initialized:
            await self.initialize()

        async with self.semaphore:
            self.integration_stats["total_requests"] += 1

            try:
                # データソース優先度決定
                sources = self._determine_data_sources(
                    symbol, data_type, prefer_real_time, max_staleness_minutes
                )

                # 優先度順にデータ取得試行
                for source_info in sources:
                    data = await self._fetch_from_source(
                        symbol, data_type, source_info, **kwargs
                    )

                    if data and self._validate_data_quality(data):
                        # 成功時の処理
                        self.integration_stats["successful_requests"] += 1

                        # キャッシュ更新
                        if self.config.enable_intelligent_caching:
                            await self._cache_data(data)

                        return data

                # 全ソース失敗時の処理
                self.integration_stats["failed_requests"] += 1
                logger.warning(f"全データソースから取得失敗: {symbol}")

                return None

            except Exception as e:
                self.integration_stats["failed_requests"] += 1
                logger.error(f"統合データ取得エラー {symbol}: {e}")
                return None

    def _determine_data_sources(
        self,
        symbol: str,
        data_type: str,
        prefer_real_time: bool,
        max_staleness_minutes: int,
    ) -> List[Tuple[DataSource, DataSourceConfig]]:
        """データソース優先度決定"""

        # リアルタイムデータ確認
        real_time_available = symbol in self.real_time_data

        sources = []

        if prefer_real_time and real_time_available:
            # リアルタイムデータ優先
            for source_key, config in self.data_sources.items():
                if (
                    config.source_type == DataSource.WEBSOCKET_STREAM
                    and config.enabled
                    and config.max_staleness_minutes <= max_staleness_minutes
                ):
                    sources.append((config.source_type, config))

        # 標準優先度順追加
        for source_type in self.config.default_source_priority:
            for source_key, config in self.data_sources.items():
                if (
                    config.source_type == source_type
                    and config.enabled
                    and config.max_staleness_minutes <= max_staleness_minutes
                    and (source_type, config) not in sources
                ):
                    sources.append((source_type, config))

        # 優先度でソート
        sources.sort(key=lambda x: x[1].priority)

        return sources

    async def _fetch_from_source(
        self,
        symbol: str,
        data_type: str,
        source_info: Tuple[DataSource, DataSourceConfig],
        **kwargs,
    ) -> Optional[UnifiedMarketData]:
        """データソースから取得"""
        source_type, config = source_info

        try:
            if source_type == DataSource.WEBSOCKET_STREAM:
                return await self._fetch_from_websocket(symbol, data_type, config)
            elif source_type == DataSource.REST_API:
                return await self._fetch_from_rest_api(
                    symbol, data_type, config, **kwargs
                )
            elif source_type == DataSource.CACHE:
                return await self._fetch_from_cache(symbol, data_type, config)
            else:
                return None

        except Exception as e:
            logger.warning(f"データソース取得エラー {source_type.value}: {e}")
            return None

    async def _fetch_from_websocket(
        self, symbol: str, data_type: str, config: DataSourceConfig
    ) -> Optional[UnifiedMarketData]:
        """WebSocketから取得"""
        if symbol in self.real_time_data:
            real_time = self.real_time_data[symbol]

            # 鮮度チェック
            age_seconds = (datetime.now() - real_time.timestamp).total_seconds()
            if age_seconds <= config.max_staleness_minutes * 60:
                # データ品質評価
                quality = self._evaluate_data_quality(real_time, age_seconds)
                real_time.quality = quality
                real_time.freshness = self._evaluate_data_freshness(age_seconds)
                real_time.source = DataSource.WEBSOCKET_STREAM

                return real_time

        return None

    async def _fetch_from_rest_api(
        self, symbol: str, data_type: str, config: DataSourceConfig, **kwargs
    ) -> Optional[UnifiedMarketData]:
        """REST APIから取得"""
        if not self.rest_client:
            return None

        try:
            # APIプロバイダー決定
            if config.provider == "mock_provider":
                api_provider = RestAPIProvider.MOCK_PROVIDER
            elif config.provider == "yahoo_finance":
                api_provider = RestAPIProvider.YAHOO_FINANCE
            elif config.provider == "alpha_vantage":
                api_provider = RestAPIProvider.ALPHA_VANTAGE
            else:
                api_provider = RestAPIProvider.MOCK_PROVIDER

            # API呼び出し
            response = await self.rest_client.fetch_stock_data(
                symbol, api_provider, **kwargs
            )

            if response and response.success and response.normalized_data is not None:
                # 統合データ変換
                unified_data = await self._convert_to_unified_data(
                    symbol, data_type, response, DataSource.REST_API, config.provider
                )

                return unified_data

        except Exception as e:
            logger.error(f"REST API取得エラー: {e}")

        return None

    async def _fetch_from_cache(
        self, symbol: str, data_type: str, config: DataSourceConfig
    ) -> Optional[UnifiedMarketData]:
        """キャッシュから取得"""
        cache_key = self._generate_cache_key(symbol, data_type)

        if cache_key in self.cache:
            entry = self.cache[cache_key]

            if not entry.is_expired():
                entry.access()
                self.cache_stats["hits"] += 1

                # キャッシュデータの鮮度・品質更新
                cached_data = entry.data
                age_seconds = (datetime.now() - cached_data.timestamp).total_seconds()

                cached_data.freshness = self._evaluate_data_freshness(age_seconds)
                cached_data.source = DataSource.CACHE

                return cached_data
            else:
                # 期限切れキャッシュ削除
                del self.cache[cache_key]

        self.cache_stats["misses"] += 1
        return None

    async def _convert_to_unified_data(
        self,
        symbol: str,
        data_type: str,
        response: APIResponse,
        source: DataSource,
        provider: str,
    ) -> UnifiedMarketData:
        """統合データ変換"""

        df = response.normalized_data

        if df is not None and not df.empty:
            # 最新データ取得
            latest_row = df.iloc[-1]

            unified_data = UnifiedMarketData(
                symbol=symbol,
                data_type=data_type,
                timestamp=latest_row.get("timestamp", datetime.now()),
                open_price=latest_row.get("open"),
                high_price=latest_row.get("high"),
                low_price=latest_row.get("low"),
                close_price=latest_row.get("close"),
                volume=latest_row.get("volume"),
                source=source,
                provider=provider,
                confidence_score=0.8,  # デフォルト信頼度
            )

            # データ品質評価
            age_seconds = (datetime.now() - unified_data.timestamp).total_seconds()
            unified_data.quality = self._evaluate_data_quality(
                unified_data, age_seconds
            )
            unified_data.freshness = self._evaluate_data_freshness(age_seconds)

            return unified_data

        # フォールバック: 基本データ
        return UnifiedMarketData(
            symbol=symbol,
            data_type=data_type,
            timestamp=datetime.now(),
            source=source,
            provider=provider,
            quality=DataQuality.POOR,
            freshness=DataFreshness.UNKNOWN,
        )

    def _evaluate_data_quality(
        self, data: UnifiedMarketData, age_seconds: float
    ) -> DataQuality:
        """データ品質評価"""
        if not self.config.enable_data_quality_scoring:
            return DataQuality.UNKNOWN

        quality_score = 0.0

        # 価格データ完全性チェック
        price_completeness = (
            sum(
                [
                    1 if data.open_price is not None else 0,
                    1 if data.high_price is not None else 0,
                    1 if data.low_price is not None else 0,
                    1 if data.close_price is not None else 0,
                ]
            )
            / 4.0
        )
        quality_score += price_completeness * 0.4

        # データ鮮度スコア
        if age_seconds < 60:  # 1分以内
            freshness_score = 1.0
        elif age_seconds < 300:  # 5分以内
            freshness_score = 0.8
        elif age_seconds < 900:  # 15分以内
            freshness_score = 0.6
        else:
            freshness_score = 0.3
        quality_score += freshness_score * 0.3

        # データソース信頼度
        if data.source == DataSource.WEBSOCKET_STREAM:
            source_score = 1.0
        elif data.source == DataSource.REST_API:
            source_score = 0.9
        elif data.source == DataSource.CACHE:
            source_score = 0.7
        else:
            source_score = 0.5
        quality_score += source_score * 0.3

        # 品質レベル判定
        if quality_score >= 0.9:
            return DataQuality.EXCELLENT
        elif quality_score >= 0.7:
            return DataQuality.GOOD
        elif quality_score >= 0.5:
            return DataQuality.ACCEPTABLE
        else:
            return DataQuality.POOR

    def _evaluate_data_freshness(self, age_seconds: float) -> DataFreshness:
        """データ鮮度評価"""
        if age_seconds < 5:
            return DataFreshness.REAL_TIME
        elif age_seconds < 60:
            return DataFreshness.FRESH
        elif age_seconds < 300:
            return DataFreshness.RECENT
        elif age_seconds < 900:
            return DataFreshness.STALE
        else:
            return DataFreshness.EXPIRED

    def _validate_data_quality(self, data: UnifiedMarketData) -> bool:
        """データ品質検証"""
        if not self.config.enable_data_validation:
            return True

        # 最低品質要件チェック
        if data.quality.value < self.config.min_acceptable_quality.value:
            self.integration_stats["data_quality_issues"] += 1
            return False

        # 基本データ検証
        if data.close_price is not None and data.close_price <= 0:
            return False

        if data.volume is not None and data.volume < 0:
            return False

        return True

    async def _cache_data(self, data: UnifiedMarketData) -> None:
        """データキャッシュ"""
        cache_key = self._generate_cache_key(data.symbol, data.data_type)

        # キャッシュサイズ制限
        if len(self.cache) >= self.config.cache_size_limit:
            await self._evict_cache_entries()

        # TTL決定（データ品質に基づく）
        if data.quality == DataQuality.EXCELLENT:
            ttl = self.config.default_cache_ttl_seconds
        elif data.quality == DataQuality.GOOD:
            ttl = self.config.default_cache_ttl_seconds // 2
        else:
            ttl = self.config.default_cache_ttl_seconds // 4

        # キャッシュエントリ作成
        entry = CacheEntry(
            key=cache_key, data=data, created_at=datetime.now(), ttl_seconds=ttl
        )

        self.cache[cache_key] = entry
        self.cache_stats["total_size"] = len(self.cache)

    async def _evict_cache_entries(self) -> None:
        """キャッシュエビクション"""
        if not self.cache:
            return

        # LRU方式でエビクション
        sorted_entries = sorted(
            self.cache.items(), key=lambda x: x[1].last_access or x[1].created_at
        )

        # 古いエントリを10%削除
        evict_count = max(1, len(self.cache) // 10)

        for i in range(evict_count):
            cache_key, entry = sorted_entries[i]
            del self.cache[cache_key]
            self.cache_stats["evictions"] += 1

        self.cache_stats["total_size"] = len(self.cache)

    def _generate_cache_key(self, symbol: str, data_type: str) -> str:
        """キャッシュキー生成"""
        key_string = f"{symbol}_{data_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    async def start_real_time_streaming(self, symbols: List[str]) -> Dict[str, str]:
        """リアルタイムストリーミング開始"""
        if not self.websocket_client:
            await self.initialize()

        subscription_ids = {}

        for symbol in symbols:
            try:
                # ストリーミングコールバック
                def create_callback(sym):
                    def callback(message: StreamMessage):
                        # UnifiedMarketDataに変換
                        unified_data = UnifiedMarketData(
                            symbol=sym,
                            data_type="real_time_price",
                            timestamp=message.timestamp,
                            close_price=message.data.get("price"),
                            volume=message.data.get("volume"),
                            source=DataSource.WEBSOCKET_STREAM,
                            provider=message.provider.value,
                            quality=DataQuality.GOOD,
                            freshness=DataFreshness.REAL_TIME,
                            confidence_score=0.9,
                        )

                        # リアルタイムデータ更新
                        self.real_time_data[sym] = unified_data

                    return callback

                # 購読開始
                subscription_id = await self.websocket_client.subscribe(
                    provider=StreamProvider.MOCK_STREAM,
                    stream_type=StreamType.REAL_TIME_QUOTES,
                    symbols=[symbol],
                    callback=create_callback(symbol),
                )

                subscription_ids[symbol] = subscription_id
                self.streaming_subscriptions[symbol] = subscription_id

                logger.info(f"リアルタイムストリーミング開始: {symbol}")

            except Exception as e:
                logger.error(f"ストリーミング開始エラー {symbol}: {e}")

        return subscription_ids

    async def stop_real_time_streaming(
        self, symbols: Optional[List[str]] = None
    ) -> None:
        """リアルタイムストリーミング停止"""
        if not self.websocket_client:
            return

        target_symbols = symbols or list(self.streaming_subscriptions.keys())

        for symbol in target_symbols:
            if symbol in self.streaming_subscriptions:
                subscription_id = self.streaming_subscriptions[symbol]

                try:
                    await self.websocket_client.unsubscribe(subscription_id)
                    del self.streaming_subscriptions[symbol]

                    if symbol in self.real_time_data:
                        del self.real_time_data[symbol]

                    logger.info(f"リアルタイムストリーミング停止: {symbol}")

                except Exception as e:
                    logger.error(f"ストリーミング停止エラー {symbol}: {e}")

    async def get_multiple_market_data(
        self,
        symbols: List[str],
        data_type: str = "stock_price",
        prefer_real_time: bool = True,
    ) -> Dict[str, UnifiedMarketData]:
        """複数銘柄データ取得"""

        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self.get_market_data(symbol, data_type, prefer_real_time)
            )
            tasks.append((symbol, task))

        results = {}
        for symbol, task in tasks:
            try:
                data = await task
                if data:
                    results[symbol] = data
            except Exception as e:
                logger.error(f"複数データ取得エラー {symbol}: {e}")

        return results

    def get_integration_statistics(self) -> Dict[str, Any]:
        """統合統計情報取得"""
        success_rate = (
            (
                self.integration_stats["successful_requests"]
                / self.integration_stats["total_requests"]
                * 100
            )
            if self.integration_stats["total_requests"] > 0
            else 0
        )

        cache_hit_rate = (
            (
                self.cache_stats["hits"]
                / (self.cache_stats["hits"] + self.cache_stats["misses"])
                * 100
            )
            if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0
            else 0
        )

        return {
            "integration_stats": {
                **self.integration_stats,
                "success_rate_percent": success_rate,
            },
            "cache_stats": {**self.cache_stats, "hit_rate_percent": cache_hit_rate},
            "streaming_stats": {
                "active_subscriptions": len(self.streaming_subscriptions),
                "real_time_symbols": len(self.real_time_data),
            },
            "system_status": {
                "initialized": self._initialized,
                "data_sources_registered": len(self.data_sources),
                "timestamp": datetime.now().isoformat(),
            },
        }

    def get_data_quality_report(self) -> Dict[str, Any]:
        """データ品質レポート"""
        quality_distribution = {quality.value: 0 for quality in DataQuality}
        freshness_distribution = {freshness.value: 0 for freshness in DataFreshness}

        # キャッシュデータ分析
        total_cached = len(self.cache)
        for entry in self.cache.values():
            quality_distribution[entry.data.quality.value] += 1
            freshness_distribution[entry.data.freshness.value] += 1

        # リアルタイムデータ分析
        total_real_time = len(self.real_time_data)
        for data in self.real_time_data.values():
            quality_distribution[data.quality.value] += 1
            freshness_distribution[data.freshness.value] += 1

        return {
            "total_data_points": total_cached + total_real_time,
            "cached_data_points": total_cached,
            "real_time_data_points": total_real_time,
            "quality_distribution": quality_distribution,
            "freshness_distribution": freshness_distribution,
            "data_quality_issues": self.integration_stats["data_quality_issues"],
            "timestamp": datetime.now().isoformat(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        rest_health = "unknown"
        websocket_health = "unknown"

        try:
            if self.rest_client:
                rest_status = await self.rest_client.health_check()
                rest_health = (
                    "healthy" if rest_status.get("session_active") else "unhealthy"
                )
        except Exception:
            rest_health = "unhealthy"

        try:
            if self.websocket_client:
                ws_status = await self.websocket_client.health_check()
                websocket_health = ws_status.get("status", "unknown")
        except Exception:
            websocket_health = "unhealthy"

        overall_status = (
            "healthy"
            if (rest_health == "healthy" and websocket_health == "healthy")
            else "degraded"
        )

        return {
            "overall_status": overall_status,
            "rest_api_status": rest_health,
            "websocket_status": websocket_health,
            "cache_size": len(self.cache),
            "streaming_subscriptions": len(self.streaming_subscriptions),
            "initialized": self._initialized,
            "timestamp": datetime.now().isoformat(),
        }


# 使用例・テスト関数


async def setup_integration_manager() -> APIIntegrationManager:
    """統合マネージャーセットアップ"""
    config = IntegrationConfig(
        enable_intelligent_caching=True,
        enable_data_quality_scoring=True,
        enable_automatic_fallback=True,
    )

    manager = APIIntegrationManager(config)
    await manager.initialize()

    return manager


async def test_integrated_market_data():
    """統合市場データテスト"""
    manager = await setup_integration_manager()

    try:
        # リアルタイムストリーミング開始
        symbols = ["7203", "8306", "9984"]  # トヨタ、三菱UFJ、SBG

        print("リアルタイムストリーミング開始...")
        await manager.start_real_time_streaming(symbols)

        # 少し待機してストリーミングデータを受信
        await asyncio.sleep(3)

        # 統合データ取得テスト
        print("\n統合市場データ取得テスト:")
        for symbol in symbols:
            data = await manager.get_market_data(symbol, prefer_real_time=True)

            if data:
                print(
                    f"  {symbol}: ¥{data.close_price:.2f} "
                    f"[{data.source.value}] {data.quality.value} {data.freshness.value}"
                )
            else:
                print(f"  {symbol}: データ取得失敗")

        # 複数銘柄一括取得テスト
        print("\n複数銘柄一括取得テスト:")
        multiple_data = await manager.get_multiple_market_data(symbols)

        for symbol, data in multiple_data.items():
            print(
                f"  {symbol}: ¥{data.close_price:.2f} 信頼度{data.confidence_score:.2f}"
            )

        # 統計情報表示
        stats = manager.get_integration_statistics()
        print("\n📊 統合統計:")
        print(f"  総リクエスト: {stats['integration_stats']['total_requests']}")
        print(f"  成功率: {stats['integration_stats']['success_rate_percent']:.1f}%")
        print(f"  キャッシュヒット率: {stats['cache_stats']['hit_rate_percent']:.1f}%")
        print(
            f"  アクティブストリーミング: {stats['streaming_stats']['active_subscriptions']}"
        )

        # データ品質レポート
        quality_report = manager.get_data_quality_report()
        print("\n📈 データ品質:")
        print(f"  総データポイント: {quality_report['total_data_points']}")
        print(f"  品質分布: {quality_report['quality_distribution']}")

    finally:
        await manager.stop_real_time_streaming()
        await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(test_integrated_market_data())
