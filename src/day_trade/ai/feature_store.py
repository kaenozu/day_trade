#!/usr/bin/env python3
"""
Feature Store Implementation
特徴量ストア実装
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import uuid
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..functional.monads import Either, TradingResult

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """特徴量タイプ"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"

class FeatureGroup(Enum):
    """特徴量グループ"""
    MARKET_DATA = "market_data"
    TECHNICAL_INDICATORS = "technical_indicators"
    FUNDAMENTAL_DATA = "fundamental_data"
    NEWS_SENTIMENT = "news_sentiment"
    ALTERNATIVE_DATA = "alternative_data"

@dataclass
class FeatureDefinition:
    """特徴量定義"""
    name: str
    feature_type: FeatureType
    group: FeatureGroup
    description: str
    source_table: Optional[str] = None
    transformation: Optional[str] = None
    default_value: Optional[Any] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        return {
            'name': self.name,
            'type': self.feature_type.value,
            'group': self.group.value,
            'description': self.description,
            'source_table': self.source_table,
            'transformation': self.transformation,
            'default_value': self.default_value,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

@dataclass
class OnlineFeatures:
    """オンライン特徴量"""
    entity_id: str
    features: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"
    
    def get_feature(self, name: str, default: Any = None) -> Any:
        """特徴量取得"""
        return self.features.get(name, default)

@dataclass
class OfflineFeatures:
    """オフライン特徴量"""
    entity_ids: List[str]
    features_df: pd.DataFrame
    timestamp_column: str = "timestamp"
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        return {
            'entity_ids': self.entity_ids,
            'features': self.features_df.to_dict('records'),
            'timestamp_column': self.timestamp_column
        }

@dataclass
class FeatureQuery:
    """特徴量クエリ"""
    entity_ids: List[str]
    feature_names: List[str]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: Optional[int] = None
    filters: Dict[str, Any] = field(default_factory=dict)


class FeatureComputation(ABC):
    """特徴量計算基底クラス"""
    
    @abstractmethod
    async def compute(self, entity_id: str, context: Dict[str, Any]) -> TradingResult[Dict[str, Any]]:
        """特徴量計算"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """依存特徴量取得"""
        pass


class TechnicalIndicatorComputation(FeatureComputation):
    """テクニカル指標計算"""
    
    def __init__(self, indicator_name: str, window: int = 20):
        self.indicator_name = indicator_name
        self.window = window
    
    async def compute(self, entity_id: str, context: Dict[str, Any]) -> TradingResult[Dict[str, Any]]:
        """テクニカル指標計算"""
        try:
            prices = context.get('prices', [])
            if len(prices) < self.window:
                return TradingResult.failure('INSUFFICIENT_DATA', f'Need at least {self.window} price points')
            
            if self.indicator_name == 'sma':
                value = sum(prices[-self.window:]) / self.window
            elif self.indicator_name == 'rsi':
                value = self._calculate_rsi(prices)
            elif self.indicator_name == 'macd':
                value = self._calculate_macd(prices)
            else:
                return TradingResult.failure('UNKNOWN_INDICATOR', f'Unknown indicator: {self.indicator_name}')
            
            features = {f'{self.indicator_name}_{self.window}': value}
            return TradingResult.success(features)
            
        except Exception as e:
            return TradingResult.failure('COMPUTATION_ERROR', str(e))
    
    def get_dependencies(self) -> List[str]:
        return ['prices']
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """RSI計算"""
        if len(prices) < 2:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if not gains or not losses:
            return 50.0
        
        avg_gain = sum(gains[-self.window:]) / self.window
        avg_loss = sum(losses[-self.window:]) / self.window
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> float:
        """MACD計算"""
        if len(prices) < 26:
            return 0.0
        
        # EMA計算
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        return ema12 - ema26
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """EMA計算"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema


class FeatureStore:
    """特徴量ストア"""
    
    def __init__(self):
        self._feature_definitions: Dict[str, FeatureDefinition] = {}
        self._online_store: Dict[str, OnlineFeatures] = {}  # entity_id -> features
        self._offline_store: Dict[str, List[Dict[str, Any]]] = {}  # entity_id -> historical features
        self._computations: Dict[str, FeatureComputation] = {}
        self._feature_groups: Dict[FeatureGroup, List[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # キャッシュ
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self._default_cache_duration = timedelta(minutes=5)
    
    async def register_feature(self, feature_def: FeatureDefinition) -> TradingResult[None]:
        """特徴量登録"""
        try:
            self._feature_definitions[feature_def.name] = feature_def
            
            # グループ別インデックス更新
            if feature_def.group not in self._feature_groups:
                self._feature_groups[feature_def.group] = []
            
            if feature_def.name not in self._feature_groups[feature_def.group]:
                self._feature_groups[feature_def.group].append(feature_def.name)
            
            logger.info(f"Registered feature: {feature_def.name}")
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('REGISTRATION_ERROR', str(e))
    
    async def register_computation(self, feature_name: str, 
                                 computation: FeatureComputation) -> TradingResult[None]:
        """特徴量計算登録"""
        try:
            self._computations[feature_name] = computation
            logger.info(f"Registered computation for feature: {feature_name}")
            return TradingResult.success(None)
        except Exception as e:
            return TradingResult.failure('COMPUTATION_REGISTRATION_ERROR', str(e))
    
    async def get_online_features(self, entity_id: str, 
                                feature_names: Optional[List[str]] = None) -> TradingResult[OnlineFeatures]:
        """オンライン特徴量取得"""
        try:
            # キャッシュ確認
            cache_key = f"online_{entity_id}_{hash(tuple(feature_names or []))}"
            if self._is_cache_valid(cache_key):
                return TradingResult.success(self._cache[cache_key])
            
            # 保存済み特徴量取得
            stored_features = self._online_store.get(entity_id)
            if not stored_features:
                # 特徴量計算
                computed_result = await self._compute_features(entity_id, feature_names or [])
                if computed_result.is_left():
                    return computed_result
                
                features = computed_result.get_right()
                stored_features = OnlineFeatures(entity_id=entity_id, features=features)
                self._online_store[entity_id] = stored_features
            
            # 必要な特徴量のみフィルタ
            if feature_names:
                filtered_features = {
                    name: stored_features.features.get(name)
                    for name in feature_names
                    if name in stored_features.features
                }
                result_features = OnlineFeatures(
                    entity_id=entity_id,
                    features=filtered_features,
                    timestamp=stored_features.timestamp,
                    version=stored_features.version
                )
            else:
                result_features = stored_features
            
            # キャッシュ保存
            self._cache[cache_key] = result_features
            self._cache_ttl[cache_key] = datetime.now(timezone.utc) + self._default_cache_duration
            
            return TradingResult.success(result_features)
            
        except Exception as e:
            return TradingResult.failure('ONLINE_FEATURES_ERROR', str(e))
    
    async def get_offline_features(self, query: FeatureQuery) -> TradingResult[OfflineFeatures]:
        """オフライン特徴量取得"""
        try:
            all_features = []
            
            for entity_id in query.entity_ids:
                historical_features = self._offline_store.get(entity_id, [])
                
                # 時間範囲フィルタ
                if query.start_time or query.end_time:
                    filtered_features = []
                    for feature_record in historical_features:
                        timestamp = datetime.fromisoformat(feature_record.get('timestamp', ''))
                        
                        if query.start_time and timestamp < query.start_time:
                            continue
                        if query.end_time and timestamp > query.end_time:
                            continue
                        
                        filtered_features.append(feature_record)
                    
                    historical_features = filtered_features
                
                # 特徴量名フィルタ
                if query.feature_names:
                    for record in historical_features:
                        filtered_record = {
                            'entity_id': entity_id,
                            'timestamp': record.get('timestamp')
                        }
                        for name in query.feature_names:
                            if name in record:
                                filtered_record[name] = record[name]
                        all_features.append(filtered_record)
                else:
                    for record in historical_features:
                        record['entity_id'] = entity_id
                        all_features.append(record)
            
            # リミット適用
            if query.limit and len(all_features) > query.limit:
                all_features = all_features[:query.limit]
            
            df = pd.DataFrame(all_features) if all_features else pd.DataFrame()
            result = OfflineFeatures(
                entity_ids=query.entity_ids,
                features_df=df
            )
            
            return TradingResult.success(result)
            
        except Exception as e:
            return TradingResult.failure('OFFLINE_FEATURES_ERROR', str(e))
    
    async def write_online_features(self, features: OnlineFeatures) -> TradingResult[None]:
        """オンライン特徴量書き込み"""
        try:
            self._online_store[features.entity_id] = features
            
            # 履歴保存
            if features.entity_id not in self._offline_store:
                self._offline_store[features.entity_id] = []
            
            historical_record = {
                'timestamp': features.timestamp.isoformat(),
                **features.features
            }
            self._offline_store[features.entity_id].append(historical_record)
            
            # 古いレコード削除（最新1000件のみ保持）
            if len(self._offline_store[features.entity_id]) > 1000:
                self._offline_store[features.entity_id] = self._offline_store[features.entity_id][-1000:]
            
            # キャッシュ無効化
            self._invalidate_cache_for_entity(features.entity_id)
            
            logger.debug(f"Wrote online features for entity: {features.entity_id}")
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('WRITE_ERROR', str(e))
    
    async def compute_features_batch(self, entity_ids: List[str], 
                                   feature_names: List[str]) -> TradingResult[Dict[str, OnlineFeatures]]:
        """バッチ特徴量計算"""
        try:
            tasks = [
                self._compute_features(entity_id, feature_names)
                for entity_id in entity_ids
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            batch_results = {}
            for i, result in enumerate(results):
                entity_id = entity_ids[i]
                
                if isinstance(result, TradingResult) and result.is_right():
                    features = result.get_right()
                    batch_results[entity_id] = OnlineFeatures(
                        entity_id=entity_id,
                        features=features
                    )
                else:
                    logger.error(f"Failed to compute features for {entity_id}: {result}")
            
            return TradingResult.success(batch_results)
            
        except Exception as e:
            return TradingResult.failure('BATCH_COMPUTATION_ERROR', str(e))
    
    async def get_feature_statistics(self, feature_name: str, 
                                   entity_ids: Optional[List[str]] = None) -> TradingResult[Dict[str, Any]]:
        """特徴量統計取得"""
        try:
            if feature_name not in self._feature_definitions:
                return TradingResult.failure('FEATURE_NOT_FOUND', f'Feature {feature_name} not found')
            
            values = []
            
            # データ収集
            for entity_id, online_features in self._online_store.items():
                if entity_ids and entity_id not in entity_ids:
                    continue
                
                value = online_features.get_feature(feature_name)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)
            
            if not values:
                return TradingResult.success({'count': 0, 'message': 'No data available'})
            
            # 統計計算
            statistics = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'p25': np.percentile(values, 25),
                'p75': np.percentile(values, 75)
            }
            
            return TradingResult.success(statistics)
            
        except Exception as e:
            return TradingResult.failure('STATISTICS_ERROR', str(e))
    
    def list_features(self, group: Optional[FeatureGroup] = None) -> List[FeatureDefinition]:
        """特徴量一覧"""
        if group:
            feature_names = self._feature_groups.get(group, [])
            return [self._feature_definitions[name] for name in feature_names]
        else:
            return list(self._feature_definitions.values())
    
    def get_feature_definition(self, feature_name: str) -> Optional[FeatureDefinition]:
        """特徴量定義取得"""
        return self._feature_definitions.get(feature_name)
    
    async def _compute_features(self, entity_id: str, feature_names: List[str]) -> TradingResult[Dict[str, Any]]:
        """特徴量計算"""
        try:
            computed_features = {}
            
            for feature_name in feature_names:
                if feature_name in self._computations:
                    computation = self._computations[feature_name]
                    
                    # 依存特徴量準備
                    dependencies = computation.get_dependencies()
                    context = await self._prepare_context(entity_id, dependencies)
                    
                    # 計算実行
                    result = await computation.compute(entity_id, context)
                    if result.is_right():
                        computed_features.update(result.get_right())
                    else:
                        logger.warning(f"Failed to compute {feature_name}: {result.get_left().message}")
                else:
                    # デフォルト値使用
                    feature_def = self._feature_definitions.get(feature_name)
                    if feature_def and feature_def.default_value is not None:
                        computed_features[feature_name] = feature_def.default_value
            
            return TradingResult.success(computed_features)
            
        except Exception as e:
            return TradingResult.failure('COMPUTATION_ERROR', str(e))
    
    async def _prepare_context(self, entity_id: str, dependencies: List[str]) -> Dict[str, Any]:
        """計算コンテキスト準備"""
        context = {}
        
        # 既存の特徴量から依存関係取得
        if entity_id in self._online_store:
            online_features = self._online_store[entity_id]
            for dep in dependencies:
                value = online_features.get_feature(dep)
                if value is not None:
                    context[dep] = value
        
        # 履歴データから時系列データ取得
        if entity_id in self._offline_store:
            historical = self._offline_store[entity_id]
            
            # 価格データ抽出例
            if 'prices' in dependencies:
                prices = [record.get('price') for record in historical if record.get('price') is not None]
                if prices:
                    context['prices'] = prices[-100:]  # 最新100件
        
        return context
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュ有効性確認"""
        if cache_key not in self._cache_ttl:
            return False
        return datetime.now(timezone.utc) < self._cache_ttl[cache_key]
    
    def _invalidate_cache_for_entity(self, entity_id: str) -> None:
        """エンティティキャッシュ無効化"""
        keys_to_remove = [key for key in self._cache.keys() if entity_id in key]
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._cache_ttl.pop(key, None)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        return {
            'cache_size': len(self._cache),
            'ttl_entries': len(self._cache_ttl),
            'online_entities': len(self._online_store),
            'offline_entities': len(self._offline_store)
        }