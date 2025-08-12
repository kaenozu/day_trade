#!/usr/bin/env python3
"""
Next-Gen AI バッチデータパイプライン
高速リアルタイム市場データ取得・前処理システム

Apache Kafka統合・大規模並列処理・MLパイプライン対応
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# オプショナル高度ライブラリ（遅延インポート - メモリ効率化）
KAFKA_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("kafka") is not None:
        KAFKA_AVAILABLE = True
except ImportError:
    pass

REDIS_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("redis") is not None:
        REDIS_AVAILABLE = True
except ImportError:
    pass

CONFLUENT_KAFKA_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("confluent_kafka") is not None:
        CONFLUENT_KAFKA_AVAILABLE = True
except ImportError:
    pass

from ..utils.logging_config import get_context_logger
from .real_market_data import RealMarketDataManager

logger = get_context_logger(__name__)

# 統合テクニカル指標マネージャーをインポート
try:
    from ..analysis.technical_indicators_unified import TechnicalIndicatorsManager

    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False
    logger.warning("統合テクニカル指標マネージャーが利用できません")


@dataclass
class DataRequest:
    """データ取得リクエスト"""

    symbol: str
    period: str = "60d"
    interval: str = "1d"
    features: List[str] = None
    preprocessing: bool = True
    cache_ttl: int = 3600  # キャッシュ有効期間（秒）
    priority: int = 1  # 優先度（1:低 → 5:高）
    metadata: Dict[str, Any] = None


@dataclass
class DataResponse:
    """データ取得結果"""

    symbol: str
    data: Optional[pd.DataFrame]
    success: bool
    error_message: Optional[str] = None
    fetch_time: float = 0.0
    cache_hit: bool = False
    data_quality_score: float = 0.0
    record_count: int = 0
    feature_count: int = 0
    metadata: Dict[str, Any] = None


@dataclass
class PipelineStats:
    """パイプライン統計"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    avg_fetch_time: float = 0.0
    avg_data_quality: float = 0.0
    throughput_rps: float = 0.0
    timestamp: float = 0.0


class AdvancedBatchDataFetcher:
    """
    Next-Gen AI バッチデータフェッチャー

    高速並列処理・Kafka統合・リアルタイムストリーミング対応
    """

    def __init__(
        self,
        max_workers: int = 8,
        enable_kafka: bool = False,
        enable_redis: bool = False,
        kafka_config: Dict[str, Any] = None,
        redis_config: Dict[str, Any] = None,
    ):
        # CI環境でのメモリ最適化
        import os

        self.ci_mode = os.getenv("CI", "false").lower() == "true"

        self.data_manager = RealMarketDataManager()
        self.max_workers = (
            max_workers if not self.ci_mode else min(max_workers, 2)
        )  # CI時は2並列まで

        # Kafka設定（CI時は無効化）
        self.enable_kafka = enable_kafka and KAFKA_AVAILABLE and not self.ci_mode
        self.kafka_config = kafka_config or {
            "bootstrap_servers": "localhost:9092",
            "topic_prefix": "market_data",
        }
        self.kafka_producer = None

        # Redis設定（CI時は無効化）
        self.enable_redis = enable_redis and REDIS_AVAILABLE and not self.ci_mode
        self.redis_config = redis_config or {"host": "localhost", "port": 6379, "db": 0}
        self.redis_client = None

        # パフォーマンス統計
        self.stats = PipelineStats()
        self.request_history = []

        # 初期化
        self._initialize_connections()

        logger.info(
            f"Advanced Batch Data Fetcher 初期化完了: workers={max_workers}, kafka={self.enable_kafka}, redis={self.enable_redis}"
        )

    def _initialize_connections(self):
        """外部サービス接続初期化"""

        # Kafka初期化
        if self.enable_kafka:
            try:
                from kafka import KafkaProducer  # 遅延インポート

                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.kafka_config["bootstrap_servers"],
                    value_serializer=lambda v: json.dumps(v, default=str).encode(
                        "utf-8"
                    ),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                )
                logger.info("Kafka Producer初期化完了")
            except Exception as e:
                logger.error(f"Kafka初期化失敗: {e}")
                self.enable_kafka = False

        # Redis初期化
        if self.enable_redis:
            try:
                import redis  # 遅延インポート

                self.redis_client = redis.Redis(
                    **self.redis_config, decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Redis接続初期化完了")
            except Exception as e:
                logger.error(f"Redis初期化失敗: {e}")
                self.enable_redis = False

    def fetch_batch(
        self,
        requests: List[DataRequest],
        use_parallel: bool = True,
        priority_sort: bool = True,
    ) -> Dict[str, DataResponse]:
        """
        高度バッチデータ取得

        Args:
            requests: データリクエストリスト
            use_parallel: 並列処理使用
            priority_sort: 優先度ソート

        Returns:
            シンボル別データレスポンス辞書
        """

        start_time = time.time()
        logger.info(f"Advanced バッチ取得開始: {len(requests)} リクエスト")

        # 優先度ソート
        if priority_sort:
            requests.sort(key=lambda x: x.priority, reverse=True)

        # 並列/逐次選択
        if use_parallel and len(requests) > 1:
            results = self._fetch_parallel_advanced(requests)
        else:
            results = self._fetch_sequential_advanced(requests)

        # 統計更新
        elapsed_time = time.time() - start_time
        self._update_stats(requests, results, elapsed_time)

        # Kafka配信
        if self.enable_kafka:
            self._publish_to_kafka(results)

        logger.info(
            f"Advanced バッチ取得完了: {len(results)} レスポンス ({elapsed_time:.2f}秒)"
        )
        return results

    def _fetch_parallel_advanced(
        self, requests: List[DataRequest]
    ) -> Dict[str, DataResponse]:
        """高度並列取得"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 非同期タスク投入
            future_to_request = {
                executor.submit(self._process_single_request, req): req
                for req in requests
            }

            # 結果回収
            for future in as_completed(future_to_request, timeout=300):
                request = future_to_request[future]
                try:
                    response = future.result(timeout=60)
                    results[request.symbol] = response

                    if response.success:
                        logger.debug(
                            f"並列取得成功: {request.symbol} ({response.record_count}レコード)"
                        )
                    else:
                        logger.warning(
                            f"並列取得失敗: {request.symbol} - {response.error_message}"
                        )

                except Exception as e:
                    logger.error(f"並列取得エラー {request.symbol}: {e}")
                    results[request.symbol] = DataResponse(
                        symbol=request.symbol,
                        data=None,
                        success=False,
                        error_message=str(e),
                    )

        return results

    def _fetch_sequential_advanced(
        self, requests: List[DataRequest]
    ) -> Dict[str, DataResponse]:
        """高度逐次取得"""
        results = {}

        for request in requests:
            try:
                response = self._process_single_request(request)
                results[request.symbol] = response

                if response.success:
                    logger.debug(
                        f"逐次取得成功: {request.symbol} ({response.record_count}レコード)"
                    )
                else:
                    logger.warning(
                        f"逐次取得失敗: {request.symbol} - {response.error_message}"
                    )

            except Exception as e:
                logger.error(f"逐次取得エラー {request.symbol}: {e}")
                results[request.symbol] = DataResponse(
                    symbol=request.symbol,
                    data=None,
                    success=False,
                    error_message=str(e),
                )

        return results

    def _process_single_request(self, request: DataRequest) -> DataResponse:
        """単一リクエスト処理"""
        start_time = time.time()
        cache_hit = False

        try:
            # Redis キャッシュチェック
            if self.enable_redis:
                cached_data = self._get_from_redis_cache(request)
                if cached_data is not None:
                    cache_hit = True
                    data_quality = self._calculate_data_quality(cached_data)

                    return DataResponse(
                        symbol=request.symbol,
                        data=cached_data,
                        success=True,
                        fetch_time=time.time() - start_time,
                        cache_hit=True,
                        data_quality_score=data_quality,
                        record_count=len(cached_data),
                        feature_count=len(cached_data.columns),
                        metadata={"source": "redis_cache"},
                    )

            # データ取得
            raw_data = self.data_manager.get_stock_data(
                symbol=request.symbol, period=request.period
            )

            if raw_data is None or raw_data.empty:
                return DataResponse(
                    symbol=request.symbol,
                    data=None,
                    success=False,
                    error_message="データ取得失敗またはデータなし",
                    fetch_time=time.time() - start_time,
                )

            # 前処理実行
            processed_data = raw_data
            if request.preprocessing:
                processed_data = self._preprocess_data(raw_data, request)

            # データ品質スコア計算
            data_quality = self._calculate_data_quality(processed_data)

            # Redisキャッシュに保存
            if self.enable_redis and not cache_hit:
                self._save_to_redis_cache(request, processed_data)

            fetch_time = time.time() - start_time

            return DataResponse(
                symbol=request.symbol,
                data=processed_data,
                success=True,
                fetch_time=fetch_time,
                cache_hit=cache_hit,
                data_quality_score=data_quality,
                record_count=len(processed_data),
                feature_count=len(processed_data.columns),
                metadata={
                    "source": "market_api",
                    "preprocessing_applied": request.preprocessing,
                },
            )

        except Exception as e:
            return DataResponse(
                symbol=request.symbol,
                data=None,
                success=False,
                error_message=str(e),
                fetch_time=time.time() - start_time,
            )

    def _preprocess_data(
        self, data: pd.DataFrame, request: DataRequest
    ) -> pd.DataFrame:
        """高度データ前処理（統合テクニカル指標マネージャー使用）"""

        result = data.copy()

        try:
            # 統合テクニカル指標マネージャーを使用
            if INDICATORS_AVAILABLE:
                indicators_manager = TechnicalIndicatorsManager()

                # テクニカル指標を計算
                indicators_result = indicators_manager.calculate_all_indicators(
                    data=result,
                    indicators=["sma", "ema", "rsi", "bollinger_bands", "macd"],
                    periods={"sma": [5, 20, 50], "ema": [5, 20, 50], "rsi": 14},
                )

                # 指標データを結合
                result = pd.concat([result, indicators_result], axis=1)

                logger.debug(f"統合指標マネージャー使用: {request.symbol}")

            else:
                # フォールバック: 基本的な特徴量のみ
                if "終値" in result.columns:
                    result["returns"] = result["終値"].pct_change()
                    result["log_returns"] = np.log(
                        result["終値"] / result["終値"].shift(1)
                    )
                    result["volatility_5d"] = result["returns"].rolling(5).std()
                    result["SMA_20"] = result["終値"].rolling(20).mean()
                    result["EMA_20"] = result["終値"].ewm(span=20).mean()

                logger.debug(f"フォールバック処理: {request.symbol}")

            # 出来高特徴量（追加）
            if "出来高" in result.columns:
                result["volume_ma_20"] = result["出来高"].rolling(20).mean()
                result["volume_ratio"] = result["出来高"] / result["volume_ma_20"]
                result["volume_trend"] = (
                    result["出来高"].rolling(5).mean()
                    / result["出来高"].rolling(20).mean()
                )

            # カスタム特徴量（リクエストに基づく）- Issue #575対応
            if request.features:
                for feature in request.features:
                    # 特徴量固有パラメータをメタデータから取得
                    feature_params = request.metadata.get(f"{feature}_params", {}) if request.metadata else {}
                    result = self._add_custom_feature(result, feature, **feature_params)

            # 欠損値処理
            result = result.fillna(method="ffill").fillna(method="bfill")

            logger.debug(f"前処理完了: {request.symbol} - {len(result.columns)} 特徴量")

        except Exception as e:
            logger.error(f"前処理エラー {request.symbol}: {e}")
            # エラー時は元データを返す
            result = data

        return result

    def _add_custom_feature(
        self, data: pd.DataFrame, feature_name: str, **kwargs
    ) -> pd.DataFrame:
        """拡張可能なカスタム特徴量追加システム

        Issue #575対応: プラグイン式アーキテクチャによる拡張性とエラーハンドリング改善
        """

        try:
            # 拡張可能な特徴量エンジンシステムを使用
            from .feature_engines import calculate_custom_feature, FeatureEngineError

            original_columns = set(data.columns)
            result = calculate_custom_feature(data, feature_name, **kwargs)

            # 新しく追加された特徴量をログ出力
            new_columns = set(result.columns) - original_columns
            if new_columns:
                logger.debug(f"カスタム特徴量追加成功: {feature_name} -> {list(new_columns)[:3]}{'...' if len(new_columns) > 3 else ''}")
            else:
                logger.warning(f"カスタム特徴量 {feature_name}: 新しい特徴量が追加されませんでした")

            return result

        except FeatureEngineError as e:
            logger.error(f"特徴量エンジンエラー {feature_name}: {e}")
            return data  # 元データを返す

        except ImportError:
            logger.warning("feature_engines モジュールが利用できません。フォールバック処理を実行")
            return self._add_custom_feature_fallback(data, feature_name)

        except Exception as e:
            logger.error(f"カスタム特徴量追加で予期しないエラー {feature_name}: {e}", exc_info=True)
            return data  # エラー時は元データを返す

    def _add_custom_feature_fallback(
        self, data: pd.DataFrame, feature_name: str
    ) -> pd.DataFrame:
        """フォールバック特徴量追加（従来方式）"""

        try:
            if feature_name == "trend_strength" and "終値" in data.columns:
                # トレンド強度
                short_ma = data["終値"].rolling(10).mean()
                long_ma = data["終値"].rolling(30).mean()

                # ゼロ除算対策
                with np.errstate(divide='ignore', invalid='ignore'):
                    data["trend_strength"] = ((short_ma - long_ma) / long_ma * 100).replace([np.inf, -np.inf], np.nan)

            elif feature_name == "momentum" and "終値" in data.columns:
                # モメンタム指標（異常値対策追加）
                data["momentum_10"] = data["終値"].pct_change(10).clip(-0.5, 0.5)
                data["momentum_20"] = data["終値"].pct_change(20).clip(-0.5, 0.5)

            elif feature_name == "price_channel" and all(
                col in data.columns for col in ["高値", "安値", "終値"]
            ):
                # 価格チャネル（ゼロ除算対策追加）
                highest_high = data["高値"].rolling(20).max()
                lowest_low = data["安値"].rolling(20).min()
                channel_width = highest_high - lowest_low

                with np.errstate(divide='ignore', invalid='ignore'):
                    data["price_channel_position"] = (
                        (data["終値"] - lowest_low) / channel_width
                    ).replace([np.inf, -np.inf], np.nan)

            elif feature_name == "gap_analysis" and all(
                col in data.columns for col in ["始値", "終値"]
            ):
                # ギャップ分析
                prev_close = data["終値"].shift(1)
                gap_ratio = (data["始値"] - prev_close) / prev_close

                data["gap_up"] = (gap_ratio > 0.02).astype(int)
                data["gap_down"] = (gap_ratio < -0.02).astype(int)
                data["gap_size"] = gap_ratio

            else:
                logger.warning(f"未知のカスタム特徴量: {feature_name}")

        except Exception as e:
            logger.error(f"フォールバック特徴量処理エラー {feature_name}: {e}")

        return data

    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """データ品質スコア計算（0-100）"""

        if data is None or data.empty:
            return 0.0

        quality_score = 100.0

        try:
            # 欠損値チェック
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            quality_score -= missing_ratio * 30

            # データ完整性チェック
            if "終値" in data.columns:
                # 異常値チェック（極端な価格変動）
                returns = data["終値"].pct_change().abs()
                extreme_moves = (returns > 0.2).sum()  # 20%以上の変動
                quality_score -= min(extreme_moves * 5, 20)

                # ゼロ価格チェック
                zero_prices = (data["終値"] <= 0).sum()
                quality_score -= zero_prices * 10

            # データ量チェック
            if len(data) < 30:  # 30日未満
                quality_score -= 20
            elif len(data) < 10:  # 10日未満
                quality_score -= 40

            # 重複データチェック
            if data.index.duplicated().any():
                quality_score -= 15

            quality_score = max(0.0, min(100.0, quality_score))

        except Exception as e:
            logger.error(f"データ品質計算エラー: {e}")
            quality_score = 50.0  # デフォルト

        return quality_score

    def _get_from_redis_cache(self, request: DataRequest) -> Optional[pd.DataFrame]:
        """Redisキャッシュからデータ取得"""

        if not self.redis_client:
            return None

        try:
            cache_key = (
                f"market_data:{request.symbol}:{request.period}:{request.interval}"
            )
            cached_json = self.redis_client.get(cache_key)

            if cached_json:
                data_dict = json.loads(cached_json)
                df = pd.DataFrame(data_dict["data"])
                df.index = pd.to_datetime(df.index)

                logger.debug(f"Redis キャッシュヒット: {request.symbol}")
                return df

        except Exception as e:
            logger.error(f"Redis取得エラー {request.symbol}: {e}")

        return None

    def _save_to_redis_cache(self, request: DataRequest, data: pd.DataFrame):
        """Redisキャッシュにデータ保存"""

        if not self.redis_client or data is None:
            return

        try:
            cache_key = (
                f"market_data:{request.symbol}:{request.period}:{request.interval}"
            )

            # DataFrameをJSON形式で保存
            data_dict = {
                "data": data.to_dict("records"),
                "index": data.index.strftime("%Y-%m-%d").tolist(),
                "cached_at": time.time(),
            }

            self.redis_client.setex(
                cache_key, request.cache_ttl, json.dumps(data_dict, default=str)
            )

            logger.debug(f"Redis キャッシュ保存: {request.symbol}")

        except Exception as e:
            logger.error(f"Redis保存エラー {request.symbol}: {e}")

    def _publish_to_kafka(self, responses: Dict[str, DataResponse]):
        """Kafkaにデータ配信"""

        if not self.kafka_producer:
            return

        try:
            topic = f"{self.kafka_config['topic_prefix']}_processed"

            for symbol, response in responses.items():
                if response.success and response.data is not None:
                    # データをKafkaメッセージとして配信
                    kafka_message = {
                        "symbol": symbol,
                        "timestamp": time.time(),
                        "record_count": response.record_count,
                        "feature_count": response.feature_count,
                        "data_quality_score": response.data_quality_score,
                        "cache_hit": response.cache_hit,
                        "data_sample": response.data.tail(5).to_dict(
                            "records"
                        ),  # 最新5レコードのサンプル
                    }

                    self.kafka_producer.send(topic, key=symbol, value=kafka_message)

            self.kafka_producer.flush()
            logger.debug(f"Kafka配信完了: {len(responses)} メッセージ")

        except Exception as e:
            logger.error(f"Kafka配信エラー: {e}")

    def _update_stats(
        self,
        requests: List[DataRequest],
        responses: Dict[str, DataResponse],
        elapsed_time: float,
    ):
        """統計更新"""

        successful = len([r for r in responses.values() if r.success])
        cache_hits = len([r for r in responses.values() if r.cache_hit])

        self.stats.total_requests += len(requests)
        self.stats.successful_requests += successful
        self.stats.failed_requests += len(requests) - successful
        self.stats.cache_hits += cache_hits

        # 平均計算
        if self.stats.total_requests > 0:
            self.stats.avg_fetch_time = (
                self.stats.avg_fetch_time * (self.stats.total_requests - len(requests))
                + sum(r.fetch_time for r in responses.values())
            ) / self.stats.total_requests

            quality_scores = [
                r.data_quality_score for r in responses.values() if r.success
            ]
            if quality_scores:
                self.stats.avg_data_quality = (
                    self.stats.avg_data_quality
                    * (self.stats.successful_requests - len(quality_scores))
                    + sum(quality_scores)
                ) / self.stats.successful_requests

        self.stats.throughput_rps = (
            len(requests) / elapsed_time if elapsed_time > 0 else 0
        )
        self.stats.timestamp = time.time()

        # リクエスト履歴保存（最新1000件）
        self.request_history.extend(
            [
                {
                    "symbol": req.symbol,
                    "timestamp": time.time(),
                    "success": responses.get(
                        req.symbol, DataResponse(req.symbol, None, False)
                    ).success,
                }
                for req in requests
            ]
        )

        self.request_history = self.request_history[-1000:]  # 最新1000件のみ保持

    # ヘルパー関数（重複削除 - 統合テクニカル指標マネージャー使用）

    def get_pipeline_stats(self) -> PipelineStats:
        """パイプライン統計取得"""
        return self.stats

    def get_recent_requests(self, limit: int = 100) -> List[Dict[str, Any]]:
        """最近のリクエスト履歴取得"""
        return self.request_history[-limit:]

    def clear_cache(self) -> Dict[str, int]:
        """全キャッシュクリア"""
        cleared_counts = {}

        # Redisキャッシュクリア
        if self.redis_client:
            try:
                pattern = "market_data:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    cleared_counts["redis"] = self.redis_client.delete(*keys)
                else:
                    cleared_counts["redis"] = 0
            except Exception as e:
                logger.error(f"Redisキャッシュクリアエラー: {e}")
                cleared_counts["redis"] = -1

        # メモリキャッシュクリア
        cleared_counts["memory"] = len(self.data_manager.memory_cache)
        self.data_manager.memory_cache.clear()
        self.data_manager.cache_expiry.clear()

        logger.info(f"キャッシュクリア完了: {cleared_counts}")
        return cleared_counts

    def close(self):
        """リソース解放"""
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("Advanced Batch Data Fetcher リソース解放完了")


# 便利関数
def create_data_request(symbol: str, **kwargs) -> DataRequest:
    """データリクエスト作成ヘルパー"""
    return DataRequest(symbol=symbol, **kwargs)


def fetch_advanced_batch(
    symbols: List[str],
    period: str = "60d",
    preprocessing: bool = True,
    priority: int = 1,
    **kwargs,
) -> Dict[str, DataResponse]:
    """高度バッチ取得（簡易インターフェース）"""

    requests = [
        DataRequest(
            symbol=symbol, period=period, preprocessing=preprocessing, priority=priority
        )
        for symbol in symbols
    ]

    fetcher = AdvancedBatchDataFetcher(**kwargs)
    try:
        return fetcher.fetch_batch(requests)
    finally:
        fetcher.close()


def preload_advanced_cache(
    symbols: List[str], periods: List[str] = None, **kwargs
) -> Dict[str, int]:
    """高度キャッシュ事前読み込み"""

    if periods is None:
        periods = ["5d", "60d", "1y"]

    results = {}
    fetcher = AdvancedBatchDataFetcher(**kwargs)

    try:
        for period in periods:
            requests = [
                DataRequest(symbol=symbol, period=period, preprocessing=True)
                for symbol in symbols
            ]

            responses = fetcher.fetch_batch(requests)
            success_count = len([r for r in responses.values() if r.success])
            results[period] = success_count

            logger.info(f"期間 {period}: {success_count}/{len(symbols)} 銘柄キャッシュ")

    finally:
        fetcher.close()

    return results


if __name__ == "__main__":
    # 高度テスト実行
    print("=== Next-Gen AI バッチデータパイプライン テスト ===")

    test_symbols = ["7203", "8306", "9984", "6758", "4689", "2914", "6861", "8035"]

    # 高度データリクエスト作成
    requests = [
        DataRequest(
            symbol=symbol,
            period="60d",
            preprocessing=True,
            features=["trend_strength", "momentum", "price_channel"],
            priority=(
                5 if symbol in ["7203", "8306"] else 3
            ),  # トヨタ・三菱UFJは高優先度
            cache_ttl=1800,
        )
        for symbol in test_symbols
    ]

    # Advanced Batch Fetcher初期化
    fetcher = AdvancedBatchDataFetcher(
        max_workers=6,
        enable_kafka=False,  # テスト時はKafka無効
        enable_redis=False,  # テスト時はRedis無効
    )

    try:
        # バッチ取得実行
        print(f"\n高度バッチ取得開始: {len(requests)} リクエスト")
        responses = fetcher.fetch_batch(requests, use_parallel=True, priority_sort=True)

        # 結果分析
        successful = [r for r in responses.values() if r.success]
        failed = [r for r in responses.values() if not r.success]

        print(f"結果: 成功={len(successful)}, 失敗={len(failed)}")

        for symbol, response in responses.items():
            if response.success:
                print(
                    f"✓ {symbol}: {response.record_count}レコード, {response.feature_count}特徴量, "
                    f"品質={response.data_quality_score:.1f}, キャッシュ={'✓' if response.cache_hit else '✗'}"
                )
            else:
                print(f"✗ {symbol}: {response.error_message}")

        # パフォーマンス統計
        stats = fetcher.get_pipeline_stats()
        print("\nパフォーマンス統計:")
        print(f"  総リクエスト: {stats.total_requests}")
        print(
            f"  成功率: {stats.successful_requests / stats.total_requests * 100:.1f}%"
        )
        print(
            f"  キャッシュヒット率: {stats.cache_hits / stats.total_requests * 100:.1f}%"
        )
        print(f"  平均取得時間: {stats.avg_fetch_time:.3f}秒")
        print(f"  平均データ品質: {stats.avg_data_quality:.1f}")
        print(f"  スループット: {stats.throughput_rps:.2f} RPS")

    finally:
        fetcher.close()

    print("\n=== テスト完了 ===")
