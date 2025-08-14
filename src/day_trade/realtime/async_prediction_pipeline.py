#!/usr/bin/env python3
"""
非同期予測パイプライン
Async Prediction Pipeline for Real-Time Trading

Issue #763: リアルタイム特徴量生成と予測パイプライン Phase 4
"""

import asyncio
import logging
import time
import json
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from .feature_engine import RealTimeFeatureEngine, MarketDataPoint, FeatureValue
from .streaming_processor import StreamingDataProcessor, StreamConfig
from .feature_store import RealTimeFeatureStore, FeatureStoreConfig

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """予測結果"""
    symbol: str
    timestamp: datetime
    prediction_type: str  # "buy", "sell", "hold"
    confidence: float  # 0-1
    predicted_price: Optional[float] = None
    predicted_return: Optional[float] = None
    risk_score: Optional[float] = None
    features_used: Dict[str, float] = None
    model_version: str = "v1.0"
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class PipelineConfig:
    """パイプライン設定"""
    feature_store_config: FeatureStoreConfig
    stream_config: StreamConfig
    prediction_interval_ms: int = 100  # 100ms間隔で予測
    batch_size: int = 10
    max_concurrent_predictions: int = 5
    feature_cache_size: int = 1000
    prediction_cache_ttl: int = 300  # 5分
    enable_real_time_alerts: bool = True
    alert_thresholds: Dict[str, float] = None


@dataclass
class PipelineMetrics:
    """パイプラインメトリクス"""
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    avg_prediction_time_ms: float = 0.0
    avg_end_to_end_latency_ms: float = 0.0
    features_processed: int = 0
    alerts_generated: int = 0
    cache_hit_rate: float = 0.0
    throughput_predictions_per_second: float = 0.0


class PredictionModel(ABC):
    """予測モデルの基底クラス"""

    @abstractmethod
    async def predict(self, features: Dict[str, float], symbol: str) -> PredictionResult:
        """予測実行"""
        pass

    @abstractmethod
    def get_required_features(self) -> List[str]:
        """必要な特徴量リスト"""
        pass


class SimpleMovingAverageModel(PredictionModel):
    """単純移動平均ベースの予測モデル（デモ用）"""

    def __init__(self):
        self.required_features = ["sma_5", "sma_20", "sma_50", "rsi_14", "macd"]

    async def predict(self, features: Dict[str, float], symbol: str) -> PredictionResult:
        """移動平均ベースの簡単な予測"""
        start_time = time.time()

        try:
            # 特徴量チェック
            if not all(feature in features for feature in self.required_features):
                missing = [f for f in self.required_features if f not in features]
                raise ValueError(f"Missing required features: {missing}")

            sma_5 = features["sma_5"]
            sma_20 = features["sma_20"]
            sma_50 = features["sma_50"]
            rsi = features["rsi_14"]
            macd = features["macd"]

            # 簡単な予測ロジック
            prediction_type = "hold"
            confidence = 0.5
            risk_score = 0.5

            # トレンド判定
            if sma_5 > sma_20 > sma_50:
                # 上昇トレンド
                if rsi < 70 and macd > 0:
                    prediction_type = "buy"
                    confidence = min(0.8, 0.5 + (sma_5 - sma_20) / sma_20)
                    risk_score = max(0.2, rsi / 100)
            elif sma_5 < sma_20 < sma_50:
                # 下降トレンド
                if rsi > 30 and macd < 0:
                    prediction_type = "sell"
                    confidence = min(0.8, 0.5 + (sma_20 - sma_5) / sma_20)
                    risk_score = max(0.2, (100 - rsi) / 100)

            # 価格予測（簡易）
            price_change_rate = (sma_5 - sma_20) / sma_20 if sma_20 > 0 else 0
            predicted_return = price_change_rate * 0.1  # 保守的な予測

            processing_time = (time.time() - start_time) * 1000

            return PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction_type=prediction_type,
                confidence=confidence,
                predicted_return=predicted_return,
                risk_score=risk_score,
                features_used=features.copy(),
                model_version="simple_ma_v1.0",
                processing_time_ms=processing_time,
                metadata={
                    "trend_direction": "up" if sma_5 > sma_50 else "down",
                    "rsi_signal": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
                    "macd_signal": "positive" if macd > 0 else "negative"
                }
            )

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction_type="hold",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )

    def get_required_features(self) -> List[str]:
        return self.required_features.copy()


class EnsembleModelWrapper(PredictionModel):
    """EnsembleSystemのラッパー（Issue #487連携）"""

    def __init__(self):
        self.required_features = [
            "sma_5", "sma_20", "sma_50", "ema_12", "ema_26",
            "rsi_14", "macd", "bollinger_20"
        ]
        # 実際のEnsembleSystemを使用する場合はここで初期化
        self.ensemble_system = None

    async def predict(self, features: Dict[str, float], symbol: str) -> PredictionResult:
        """EnsembleSystemを使用した予測"""
        start_time = time.time()

        try:
            # 特徴量をDataFrame形式に変換
            feature_df = pd.DataFrame([features])

            # EnsembleSystemで予測（実装されている場合）
            if self.ensemble_system:
                prediction = await self.ensemble_system.predict_async(feature_df, symbol)

                return PredictionResult(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    prediction_type=prediction.get("action", "hold"),
                    confidence=prediction.get("confidence", 0.5),
                    predicted_return=prediction.get("expected_return"),
                    risk_score=prediction.get("risk_score"),
                    features_used=features.copy(),
                    model_version="ensemble_v487",
                    processing_time_ms=(time.time() - start_time) * 1000,
                    metadata=prediction.get("metadata", {})
                )
            else:
                # フォールバック：簡単な予測
                logger.warning("EnsembleSystem not available, using fallback prediction")
                return await self._fallback_prediction(features, symbol, start_time)

        except Exception as e:
            logger.error(f"Ensemble prediction error for {symbol}: {e}")
            return await self._fallback_prediction(features, symbol, start_time)

    async def _fallback_prediction(self, features: Dict[str, float], symbol: str, start_time: float) -> PredictionResult:
        """フォールバック予測"""
        # 基本的な移動平均クロス戦略
        sma_5 = features.get("sma_5", 0)
        sma_20 = features.get("sma_20", 0)
        rsi = features.get("rsi_14", 50)

        if sma_5 > sma_20 and rsi < 70:
            prediction_type = "buy"
            confidence = 0.6
        elif sma_5 < sma_20 and rsi > 30:
            prediction_type = "sell"
            confidence = 0.6
        else:
            prediction_type = "hold"
            confidence = 0.4

        return PredictionResult(
            symbol=symbol,
            timestamp=datetime.now(),
            prediction_type=prediction_type,
            confidence=confidence,
            features_used=features.copy(),
            model_version="fallback_v1.0",
            processing_time_ms=(time.time() - start_time) * 1000,
            metadata={"fallback": True}
        )

    def get_required_features(self) -> List[str]:
        return self.required_features.copy()


class AlertSystem:
    """リアルタイムアラートシステム"""

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds or {
            "high_confidence_buy": 0.8,
            "high_confidence_sell": 0.8,
            "high_risk": 0.9,
            "unusual_volume": 2.0  # 平均の2倍
        }
        self.alert_history: deque = deque(maxlen=1000)

    async def check_alerts(self, prediction: PredictionResult, market_data: MarketDataPoint) -> List[Dict[str, Any]]:
        """アラートチェック"""
        alerts = []

        # 高信頼度の買いシグナル
        if (prediction.prediction_type == "buy" and
            prediction.confidence >= self.thresholds["high_confidence_buy"]):
            alerts.append({
                "type": "high_confidence_buy",
                "symbol": prediction.symbol,
                "confidence": prediction.confidence,
                "timestamp": prediction.timestamp,
                "message": f"High confidence BUY signal for {prediction.symbol} (confidence: {prediction.confidence:.2f})"
            })

        # 高信頼度の売りシグナル
        if (prediction.prediction_type == "sell" and
            prediction.confidence >= self.thresholds["high_confidence_sell"]):
            alerts.append({
                "type": "high_confidence_sell",
                "symbol": prediction.symbol,
                "confidence": prediction.confidence,
                "timestamp": prediction.timestamp,
                "message": f"High confidence SELL signal for {prediction.symbol} (confidence: {prediction.confidence:.2f})"
            })

        # 高リスク警告
        if (prediction.risk_score and
            prediction.risk_score >= self.thresholds["high_risk"]):
            alerts.append({
                "type": "high_risk_warning",
                "symbol": prediction.symbol,
                "risk_score": prediction.risk_score,
                "timestamp": prediction.timestamp,
                "message": f"HIGH RISK warning for {prediction.symbol} (risk: {prediction.risk_score:.2f})"
            })

        # アラート履歴に追加
        for alert in alerts:
            self.alert_history.append(alert)

        return alerts


class AsyncPredictionPipeline:
    """非同期予測パイプライン"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = PipelineMetrics()

        # コンポーネント初期化
        self.feature_engine = RealTimeFeatureEngine()
        self.feature_store = RealTimeFeatureStore(config.feature_store_config)
        self.streaming_processor = StreamingDataProcessor(
            feature_engine=self.feature_engine,
            stream_config=config.stream_config
        )

        # 予測モデル
        self.prediction_models: Dict[str, PredictionModel] = {
            "simple_ma": SimpleMovingAverageModel(),
            "ensemble": EnsembleModelWrapper()
        }
        self.active_model = "simple_ma"

        # アラートシステム
        self.alert_system = AlertSystem(config.alert_thresholds)

        # 内部状態
        self.is_running = False
        self.prediction_queue: asyncio.Queue = asyncio.Queue(maxsize=config.batch_size * 10)
        self.result_cache: Dict[str, Tuple[PredictionResult, float]] = {}  # (result, expiry_time)

        # 並行処理制御
        self.prediction_semaphore = asyncio.Semaphore(config.max_concurrent_predictions)

        # パフォーマンス監視
        self.processing_times: deque = deque(maxlen=1000)
        self.latency_times: deque = deque(maxlen=1000)

        logger.info("AsyncPredictionPipeline initialized")

    async def start(self) -> None:
        """パイプライン開始"""
        try:
            self.is_running = True

            # 特徴量ストア接続
            await self.feature_store.connect()

            # バックグラウンドタスク開始
            tasks = [
                asyncio.create_task(self._feature_processing_loop()),
                asyncio.create_task(self._prediction_processing_loop()),
                asyncio.create_task(self._metrics_update_loop()),
            ]

            logger.info("AsyncPredictionPipeline started")

            # 全タスク完了まで待機
            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """パイプライン停止"""
        self.is_running = False

        try:
            # ストリーミング停止
            self.streaming_processor.stop()

            # 特徴量ストア切断
            await self.feature_store.disconnect()

            logger.info("AsyncPredictionPipeline stopped")

        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")

    async def _feature_processing_loop(self) -> None:
        """特徴量処理ループ"""
        while self.is_running:
            try:
                # 特徴量エンジンからのデータ待機（シミュレーション）
                await asyncio.sleep(self.config.prediction_interval_ms / 1000)

                # アクティブな銘柄の特徴量を取得
                symbols = await self.feature_store.get_symbols()

                for symbol in symbols:
                    # 最新特徴量取得
                    required_features = self.prediction_models[self.active_model].get_required_features()
                    features = await self.feature_store.get_latest_features(symbol, required_features)

                    if features and len(features) >= len(required_features) * 0.8:  # 80%以上の特徴量が揃っている
                        # 予測キューに追加
                        prediction_request = {
                            "symbol": symbol,
                            "features": {name: feature.value for name, feature in features.items()},
                            "timestamp": datetime.now()
                        }

                        try:
                            await asyncio.wait_for(
                                self.prediction_queue.put(prediction_request),
                                timeout=0.1
                            )
                        except asyncio.TimeoutError:
                            # キューが満杯の場合はスキップ
                            pass

            except Exception as e:
                logger.error(f"Error in feature processing loop: {e}")
                await asyncio.sleep(1)

    async def _prediction_processing_loop(self) -> None:
        """予測処理ループ"""
        while self.is_running:
            try:
                # 予測リクエスト取得
                request = await asyncio.wait_for(
                    self.prediction_queue.get(),
                    timeout=1.0
                )

                # 並行処理制御
                async with self.prediction_semaphore:
                    await self._process_prediction_request(request)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in prediction processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_prediction_request(self, request: Dict[str, Any]) -> None:
        """予測リクエスト処理"""
        start_time = time.time()

        try:
            symbol = request["symbol"]
            features = request["features"]
            request_timestamp = request["timestamp"]

            # キャッシュチェック
            cache_key = f"{symbol}:{hash(str(sorted(features.items())))}"
            if cache_key in self.result_cache:
                cached_result, expiry = self.result_cache[cache_key]
                if time.time() < expiry:
                    # キャッシュヒット
                    self.metrics.cache_hit_rate = (self.metrics.cache_hit_rate * 0.9) + (1.0 * 0.1)
                    return

            # 予測実行
            model = self.prediction_models[self.active_model]
            prediction_result = await model.predict(features, symbol)

            # エンドツーエンドレイテンシ計算
            end_to_end_latency = (time.time() - request_timestamp.timestamp()) * 1000
            self.latency_times.append(end_to_end_latency)

            # 結果をキャッシュ
            self.result_cache[cache_key] = (prediction_result, time.time() + self.config.prediction_cache_ttl)

            # アラートチェック
            if self.config.enable_real_time_alerts:
                # マーケットデータの作成（簡易）
                market_data = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=features.get("sma_5", 0),  # 仮の価格
                    volume=1000
                )

                alerts = await self.alert_system.check_alerts(prediction_result, market_data)

                for alert in alerts:
                    logger.warning(f"ALERT: {alert['message']}")
                    self.metrics.alerts_generated += 1

            # メトリクス更新
            self.metrics.total_predictions += 1
            self.metrics.successful_predictions += 1

            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)

            if self.metrics.avg_prediction_time_ms == 0:
                self.metrics.avg_prediction_time_ms = processing_time
            else:
                self.metrics.avg_prediction_time_ms = (self.metrics.avg_prediction_time_ms * 0.9) + (processing_time * 0.1)

            if self.metrics.avg_end_to_end_latency_ms == 0:
                self.metrics.avg_end_to_end_latency_ms = end_to_end_latency
            else:
                self.metrics.avg_end_to_end_latency_ms = (self.metrics.avg_end_to_end_latency_ms * 0.9) + (end_to_end_latency * 0.1)

            logger.debug(f"Prediction completed for {symbol}: {prediction_result.prediction_type} (confidence: {prediction_result.confidence:.2f})")

        except Exception as e:
            logger.error(f"Error processing prediction request: {e}")
            self.metrics.failed_predictions += 1

    async def _metrics_update_loop(self) -> None:
        """メトリクス更新ループ"""
        while self.is_running:
            try:
                # スループット計算
                if self.processing_times:
                    recent_predictions = len([t for t in self.processing_times if time.time() - t/1000 < 60])  # 過去1分
                    self.metrics.throughput_predictions_per_second = recent_predictions / 60

                # キャッシュヒット率計算
                cache_hits = sum(1 for _, expiry in self.result_cache.values() if time.time() < expiry)
                total_cache_entries = len(self.result_cache)
                if total_cache_entries > 0:
                    current_hit_rate = cache_hits / total_cache_entries
                    self.metrics.cache_hit_rate = (self.metrics.cache_hit_rate * 0.9) + (current_hit_rate * 0.1)

                await asyncio.sleep(10)  # 10秒ごと更新

            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(10)

    async def predict_single(self, symbol: str, features: Dict[str, float]) -> PredictionResult:
        """単体予測（同期的API）"""
        try:
            model = self.prediction_models[self.active_model]
            return await model.predict(features, symbol)
        except Exception as e:
            logger.error(f"Error in single prediction for {symbol}: {e}")
            return PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction_type="hold",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def get_metrics(self) -> PipelineMetrics:
        """メトリクス取得"""
        return self.metrics

    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """最近のアラート取得"""
        return list(self.alert_system.alert_history)[-count:]

    def switch_model(self, model_name: str) -> bool:
        """予測モデル切り替え"""
        if model_name in self.prediction_models:
            self.active_model = model_name
            logger.info(f"Switched to prediction model: {model_name}")
            return True
        else:
            logger.error(f"Unknown model: {model_name}")
            return False


# 使用例とテスト
async def test_prediction_pipeline():
    """予測パイプラインのテスト"""

    # 設定
    feature_store_config = FeatureStoreConfig(
        redis_host="localhost",
        redis_port=6379,
        redis_db=2,  # テスト用
        default_ttl=3600
    )

    stream_config = StreamConfig(
        url="ws://test",
        symbols=["7203", "8306", "9984"]
    )

    pipeline_config = PipelineConfig(
        feature_store_config=feature_store_config,
        stream_config=stream_config,
        prediction_interval_ms=500,  # 500ms間隔
        enable_real_time_alerts=True,
        alert_thresholds={
            "high_confidence_buy": 0.7,
            "high_confidence_sell": 0.7,
            "high_risk": 0.8
        }
    )

    # パイプライン初期化
    pipeline = AsyncPredictionPipeline(pipeline_config)

    try:
        print("Starting prediction pipeline test...")

        # テスト用特徴量データ作成
        test_features = {
            "sma_5": 2100.0,
            "sma_20": 2090.0,
            "sma_50": 2080.0,
            "ema_12": 2105.0,
            "ema_26": 2095.0,
            "rsi_14": 65.0,
            "macd": 5.2,
            "bollinger_20": 0.6
        }

        # 単体予測テスト
        print("\nTesting single prediction...")
        result = await pipeline.predict_single("7203", test_features)
        print(f"Prediction for 7203: {result.prediction_type} (confidence: {result.confidence:.2f})")
        print(f"Processing time: {result.processing_time_ms:.2f}ms")

        # モデル切り替えテスト
        print(f"\nSwitching to ensemble model...")
        pipeline.switch_model("ensemble")

        result2 = await pipeline.predict_single("7203", test_features)
        print(f"Ensemble prediction for 7203: {result2.prediction_type} (confidence: {result2.confidence:.2f})")

        # メトリクス表示
        metrics = pipeline.get_metrics()
        print(f"\nPipeline Metrics:")
        print(f"  Total predictions: {metrics.total_predictions}")
        print(f"  Successful predictions: {metrics.successful_predictions}")
        print(f"  Average prediction time: {metrics.avg_prediction_time_ms:.2f}ms")
        print(f"  Alerts generated: {metrics.alerts_generated}")

        print("\nPrediction pipeline test completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")

    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(test_prediction_pipeline())