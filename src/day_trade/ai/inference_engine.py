#!/usr/bin/env python3
"""
Real-Time ML Inference Engine
リアルタイムML推論エンジン
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Union, Callable
from enum import Enum
import uuid
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from ..functional.monads import Either, TradingResult
from ..reactive.streams import MarketDataStream

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """モデル状態"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOADING = "loading"
    ERROR = "error"
    DEPRECATED = "deprecated"

class InferenceMode(Enum):
    """推論モード"""
    SYNCHRONOUS = "sync"
    ASYNCHRONOUS = "async"
    STREAMING = "streaming"
    BATCH = "batch"

@dataclass
class ModelVersion:
    """モデルバージョン"""
    model_id: str
    version: str
    name: str
    description: str
    model_path: str
    status: ModelStatus = ModelStatus.INACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    
    @property
    def model_key(self) -> str:
        """モデルキー"""
        return f"{self.model_id}:{self.version}"

@dataclass
class InferenceRequest:
    """推論リクエスト"""
    request_id: str
    model_key: str
    features: Dict[str, Any]
    mode: InferenceMode = InferenceMode.SYNCHRONOUS
    timeout_ms: int = 5000
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class InferenceResult:
    """推論結果"""
    request_id: str
    model_key: str
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_version: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """高信頼度判定"""
        if not self.confidence_scores:
            return False
        return max(self.confidence_scores.values()) >= threshold


class ModelInterface(ABC):
    """モデルインターフェース"""
    
    @abstractmethod
    async def load(self, model_path: str) -> TradingResult[None]:
        """モデル読み込み"""
        pass
    
    @abstractmethod
    async def predict(self, features: Dict[str, Any]) -> TradingResult[Dict[str, Any]]:
        """推論実行"""
        pass
    
    @abstractmethod
    async def predict_batch(self, features_batch: List[Dict[str, Any]]) -> TradingResult[List[Dict[str, Any]]]:
        """バッチ推論"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """特徴量名取得"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        pass


class MockMLModel(ModelInterface):
    """モックMLモデル（デモ用）"""
    
    def __init__(self):
        self._loaded = False
        self._feature_names = ['price', 'volume', 'rsi', 'macd', 'volatility']
        self._model_info = {
            'type': 'RandomForest',
            'features': 5,
            'classes': 3,
            'accuracy': 0.85
        }
    
    async def load(self, model_path: str) -> TradingResult[None]:
        """モデル読み込みシミュレーション"""
        try:
            await asyncio.sleep(0.1)  # 読み込み時間シミュレーション
            self._loaded = True
            logger.info(f"Mock model loaded from {model_path}")
            return TradingResult.success(None)
        except Exception as e:
            return TradingResult.failure('LOAD_ERROR', str(e))
    
    async def predict(self, features: Dict[str, Any]) -> TradingResult[Dict[str, Any]]:
        """推論シミュレーション"""
        if not self._loaded:
            return TradingResult.failure('MODEL_NOT_LOADED', 'Model not loaded')
        
        try:
            # シミュレーション推論
            await asyncio.sleep(0.01)  # 推論時間シミュレーション
            
            # ランダム予測（実際の実装では学習済みモデルを使用）
            import random
            predictions = {
                'signal': random.choice(['BUY', 'SELL', 'HOLD']),
                'price_direction': random.choice(['UP', 'DOWN', 'FLAT']),
                'probability': random.uniform(0.6, 0.95)
            }
            
            return TradingResult.success(predictions)
            
        except Exception as e:
            return TradingResult.failure('PREDICTION_ERROR', str(e))
    
    async def predict_batch(self, features_batch: List[Dict[str, Any]]) -> TradingResult[List[Dict[str, Any]]]:
        """バッチ推論シミュレーション"""
        try:
            results = []
            for features in features_batch:
                pred_result = await self.predict(features)
                if pred_result.is_right():
                    results.append(pred_result.get_right())
                else:
                    results.append({'error': pred_result.get_left().message})
            
            return TradingResult.success(results)
            
        except Exception as e:
            return TradingResult.failure('BATCH_PREDICTION_ERROR', str(e))
    
    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        return self._model_info.copy()


class ModelRegistry:
    """モデルレジストリ"""
    
    def __init__(self):
        self._models: Dict[str, ModelVersion] = {}
        self._loaded_models: Dict[str, ModelInterface] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._default_models: Dict[str, str] = {}  # model_type -> model_key
        
    async def register_model(self, model_version: ModelVersion) -> TradingResult[None]:
        """モデル登録"""
        try:
            model_key = model_version.model_key
            self._models[model_key] = model_version
            self._model_locks[model_key] = threading.Lock()
            
            logger.info(f"Registered model: {model_key}")
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('REGISTRATION_ERROR', str(e))
    
    async def load_model(self, model_key: str) -> TradingResult[ModelInterface]:
        """モデル読み込み"""
        try:
            if model_key not in self._models:
                return TradingResult.failure('MODEL_NOT_FOUND', f'Model {model_key} not found')
            
            # 既に読み込み済みの場合
            if model_key in self._loaded_models:
                return TradingResult.success(self._loaded_models[model_key])
            
            # 同期的読み込み
            with self._model_locks[model_key]:
                if model_key in self._loaded_models:
                    return TradingResult.success(self._loaded_models[model_key])
                
                model_version = self._models[model_key]
                model_version.status = ModelStatus.LOADING
                
                # 実際の実装では適切なモデルローダーを使用
                model_interface = MockMLModel()
                load_result = await model_interface.load(model_version.model_path)
                
                if load_result.is_left():
                    model_version.status = ModelStatus.ERROR
                    return load_result
                
                model_version.status = ModelStatus.ACTIVE
                self._loaded_models[model_key] = model_interface
                
                logger.info(f"Loaded model: {model_key}")
                return TradingResult.success(model_interface)
                
        except Exception as e:
            return TradingResult.failure('LOAD_ERROR', str(e))
    
    async def unload_model(self, model_key: str) -> TradingResult[None]:
        """モデルアンロード"""
        try:
            if model_key in self._loaded_models:
                del self._loaded_models[model_key]
                if model_key in self._models:
                    self._models[model_key].status = ModelStatus.INACTIVE
                logger.info(f"Unloaded model: {model_key}")
            
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('UNLOAD_ERROR', str(e))
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """モデルバージョン一覧取得"""
        return [mv for mv in self._models.values() if mv.model_id == model_id]
    
    def get_active_models(self) -> List[ModelVersion]:
        """アクティブモデル一覧"""
        return [mv for mv in self._models.values() if mv.status == ModelStatus.ACTIVE]
    
    def set_default_model(self, model_type: str, model_key: str) -> TradingResult[None]:
        """デフォルトモデル設定"""
        if model_key not in self._models:
            return TradingResult.failure('MODEL_NOT_FOUND', f'Model {model_key} not found')
        
        self._default_models[model_type] = model_key
        return TradingResult.success(None)
    
    def get_default_model(self, model_type: str) -> Optional[str]:
        """デフォルトモデル取得"""
        return self._default_models.get(model_type)


class RealTimeInferenceEngine:
    """リアルタイム推論エンジン"""
    
    def __init__(self, model_registry: ModelRegistry, max_workers: int = 10):
        self.model_registry = model_registry
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        self._metrics: Dict[str, Any] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_latency': 0.0,
            'requests_per_second': 0.0
        }
        self._latencies: List[float] = []
        self._callbacks: Dict[str, List[Callable]] = {}
        
    async def start(self) -> None:
        """エンジン開始"""
        if self._processing:
            return
        
        self._processing = True
        logger.info("Starting real-time inference engine")
        
        # リクエスト処理ループ開始
        for _ in range(self.max_workers):
            asyncio.create_task(self._process_requests())
        
        # メトリクス更新ループ開始
        asyncio.create_task(self._update_metrics())
    
    async def stop(self) -> None:
        """エンジン停止"""
        self._processing = False
        logger.info("Stopped real-time inference engine")
    
    async def predict(self, request: InferenceRequest) -> TradingResult[InferenceResult]:
        """推論実行"""
        start_time = time.time()
        
        try:
            # モデル取得
            model_result = await self.model_registry.load_model(request.model_key)
            if model_result.is_left():
                return TradingResult.failure('MODEL_LOAD_ERROR', model_result.get_left().message)
            
            model = model_result.get_right()
            
            # 推論実行
            prediction_result = await model.predict(request.features)
            if prediction_result.is_left():
                return prediction_result
            
            predictions = prediction_result.get_right()
            
            # 信頼度スコア計算（簡略化）
            confidence_scores = self._calculate_confidence_scores(predictions)
            
            # 結果作成
            processing_time = (time.time() - start_time) * 1000
            result = InferenceResult(
                request_id=request.request_id,
                model_key=request.model_key,
                predictions=predictions,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time,
                model_version=request.model_key.split(':')[-1],
                metadata=request.metadata
            )
            
            # メトリクス更新
            self._update_request_metrics(processing_time, True)
            
            # コールバック実行
            await self._execute_callbacks('prediction_complete', result)
            
            return TradingResult.success(result)
            
        except Exception as e:
            self._update_request_metrics(time.time() - start_time, False)
            logger.error(f"Prediction failed: {e}")
            return TradingResult.failure('PREDICTION_ERROR', str(e))
    
    async def predict_async(self, request: InferenceRequest) -> TradingResult[str]:
        """非同期推論（リクエストID返却）"""
        try:
            await self._request_queue.put(request)
            return TradingResult.success(request.request_id)
        except Exception as e:
            return TradingResult.failure('QUEUE_ERROR', str(e))
    
    async def predict_streaming(self, model_key: str, 
                              feature_stream: MarketDataStream) -> TradingResult[None]:
        """ストリーミング推論"""
        try:
            async def process_stream_data(data):
                request = InferenceRequest(
                    request_id=str(uuid.uuid4()),
                    model_key=model_key,
                    features=data,
                    mode=InferenceMode.STREAMING
                )
                
                result = await self.predict(request)
                if result.is_right():
                    await self._execute_callbacks('streaming_prediction', result.get_right())
            
            # ストリーム購読
            feature_stream.subscribe(process_stream_data)
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('STREAMING_ERROR', str(e))
    
    def add_callback(self, event_type: str, callback: Callable) -> None:
        """コールバック追加"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """メトリクス取得"""
        return self._metrics.copy()
    
    async def _process_requests(self) -> None:
        """リクエスト処理ループ"""
        while self._processing:
            try:
                request = await asyncio.wait_for(
                    self._request_queue.get(), timeout=1.0
                )
                
                # 非同期推論実行
                result = await self.predict(request)
                
                # 結果コールバック
                if result.is_right():
                    await self._execute_callbacks('async_prediction_complete', result.get_right())
                else:
                    await self._execute_callbacks('async_prediction_failed', {
                        'request_id': request.request_id,
                        'error': result.get_left()
                    })
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Request processing error: {e}")
    
    async def _execute_callbacks(self, event_type: str, data: Any) -> None:
        """コールバック実行"""
        callbacks = self._callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    await asyncio.get_event_loop().run_in_executor(None, callback, data)
            except Exception as e:
                logger.error(f"Callback execution error: {e}")
    
    def _calculate_confidence_scores(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """信頼度スコア計算"""
        scores = {}
        
        # 簡略化された信頼度計算
        for key, value in predictions.items():
            if key == 'probability' and isinstance(value, (int, float)):
                scores[key] = float(value)
            elif isinstance(value, str) and key == 'signal':
                # シグナルの確度を簡易計算
                base_confidence = 0.7
                if 'BUY' in value or 'SELL' in value:
                    scores[key] = base_confidence + 0.1
                else:
                    scores[key] = base_confidence
        
        return scores
    
    def _update_request_metrics(self, latency_ms: float, success: bool) -> None:
        """リクエストメトリクス更新"""
        self._metrics['total_requests'] += 1
        
        if success:
            self._metrics['successful_requests'] += 1
        else:
            self._metrics['failed_requests'] += 1
        
        self._latencies.append(latency_ms)
        
        # 最新100件の平均レイテンシ
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]
        
        self._metrics['average_latency'] = sum(self._latencies) / len(self._latencies)
    
    async def _update_metrics(self) -> None:
        """メトリクス更新ループ"""
        last_request_count = 0
        
        while self._processing:
            try:
                await asyncio.sleep(1)
                
                # RPS計算
                current_requests = self._metrics['total_requests']
                rps = current_requests - last_request_count
                self._metrics['requests_per_second'] = rps
                last_request_count = current_requests
                
            except Exception as e:
                logger.error(f"Metrics update error: {e}")


class ModelEnsemble:
    """モデルアンサンブル"""
    
    def __init__(self, inference_engine: RealTimeInferenceEngine):
        self.inference_engine = inference_engine
        self._ensemble_configs: Dict[str, Dict[str, Any]] = {}
        
    def add_ensemble(self, ensemble_name: str, model_keys: List[str], 
                    weights: Optional[List[float]] = None,
                    voting_strategy: str = 'majority') -> None:
        """アンサンブル追加"""
        if weights and len(weights) != len(model_keys):
            raise ValueError("Weights count must match model count")
        
        self._ensemble_configs[ensemble_name] = {
            'model_keys': model_keys,
            'weights': weights or [1.0] * len(model_keys),
            'voting_strategy': voting_strategy
        }
        
        logger.info(f"Added ensemble: {ensemble_name} with {len(model_keys)} models")
    
    async def predict_ensemble(self, ensemble_name: str, 
                             features: Dict[str, Any]) -> TradingResult[InferenceResult]:
        """アンサンブル推論"""
        if ensemble_name not in self._ensemble_configs:
            return TradingResult.failure('ENSEMBLE_NOT_FOUND', f'Ensemble {ensemble_name} not found')
        
        config = self._ensemble_configs[ensemble_name]
        model_keys = config['model_keys']
        weights = config['weights']
        
        try:
            # 並列推論実行
            tasks = [
                self.inference_engine.predict(InferenceRequest(
                    request_id=str(uuid.uuid4()),
                    model_key=model_key,
                    features=features
                ))
                for model_key in model_keys
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 成功した結果のみ収集
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, TradingResult) and result.is_right():
                    successful_results.append((result.get_right(), weights[i]))
            
            if not successful_results:
                return TradingResult.failure('ALL_MODELS_FAILED', 'All ensemble models failed')
            
            # アンサンブル結果計算
            ensemble_result = self._combine_predictions(
                successful_results, config['voting_strategy']
            )
            
            return TradingResult.success(ensemble_result)
            
        except Exception as e:
            return TradingResult.failure('ENSEMBLE_ERROR', str(e))
    
    def _combine_predictions(self, results: List[Tuple[InferenceResult, float]], 
                           strategy: str) -> InferenceResult:
        """予測結果結合"""
        if strategy == 'majority':
            return self._majority_voting(results)
        elif strategy == 'weighted':
            return self._weighted_average(results)
        else:
            return results[0][0]  # フォールバック
    
    def _majority_voting(self, results: List[Tuple[InferenceResult, float]]) -> InferenceResult:
        """多数決投票"""
        signal_votes = {}
        total_confidence = 0.0
        
        for result, weight in results:
            predictions = result.predictions
            signal = predictions.get('signal', 'HOLD')
            
            if signal not in signal_votes:
                signal_votes[signal] = 0
            signal_votes[signal] += weight
            
            confidence = result.confidence_scores.get('signal', 0.5)
            total_confidence += confidence * weight
        
        # 最多票のシグナル選択
        winning_signal = max(signal_votes.items(), key=lambda x: x[1])[0]
        avg_confidence = total_confidence / sum(w for _, w in results)
        
        ensemble_result = InferenceResult(
            request_id=str(uuid.uuid4()),
            model_key=f"ensemble_{len(results)}_models",
            predictions={'signal': winning_signal},
            confidence_scores={'signal': avg_confidence},
            processing_time_ms=sum(r.processing_time_ms for r, _ in results),
            model_version="ensemble_v1.0"
        )
        
        return ensemble_result
    
    def _weighted_average(self, results: List[Tuple[InferenceResult, float]]) -> InferenceResult:
        """重み付き平均"""
        # 数値予測の場合の重み付き平均（簡略化）
        return self._majority_voting(results)  # この例では多数決と同じ