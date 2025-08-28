#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Data Processing System - 高度データ処理システム

データ処理の速度と品質を大幅に向上させる包括的な最適化システム
- 高速並列データ処理
- インテリジェント キャッシング
- リアルタイム データ品質監視
- 自動データクリーニング
- スマート データ変換
- 分散処理対応
"""

import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import logging
import warnings
import gc
import psutil
from collections import defaultdict, deque
import hashlib
import pickle
import joblib

# データ処理ライブラリ
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
import numba
from numba import jit, prange

# 高速処理ライブラリ
try:
    import dask.dataframe as dd
    import dask.bag as db
    from dask.distributed import Client
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False

warnings.filterwarnings('ignore')

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessingStrategy(Enum):
    """処理戦略"""
    PANDAS = "pandas"
    DASK = "dask"
    POLARS = "polars"
    VAEX = "vaex"
    NUMBA = "numba"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"

class DataQuality(Enum):
    """データ品質レベル"""
    EXCELLENT = "excellent"    # 95%以上
    GOOD = "good"             # 85-95%
    FAIR = "fair"             # 75-85%
    POOR = "poor"             # 60-75%
    CRITICAL = "critical"     # 60%未満

class CacheStrategy(Enum):
    """キャッシュ戦略"""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"

@dataclass
class ProcessingMetrics:
    """処理メトリクス"""
    processing_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float  # rows per second
    cache_hit_rate: float
    error_rate: float
    data_quality_score: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DataQualityReport:
    """データ品質レポート"""
    completeness: float  # 欠損値なし割合
    consistency: float   # データ一貫性
    accuracy: float     # データ精度
    validity: float     # データ妥当性
    uniqueness: float   # 重複なし割合
    overall_score: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CacheStats:
    """キャッシュ統計"""
    hit_count: int = 0
    miss_count: int = 0
    total_size: int = 0
    memory_usage: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

class IntelligentCache:
    """インテリジェント キャッシュシステム"""
    
    def __init__(self, max_memory_mb: int = 1024, ttl_seconds: int = 3600):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.memory_cache = {}
        self.cache_metadata = {}
        self.access_times = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """キャッシュ取得"""
        with self.lock:
            if key not in self.memory_cache:
                self.stats.miss_count += 1
                return None
            
            # TTL チェック
            metadata = self.cache_metadata[key]
            if time.time() - metadata['created_at'] > self.ttl_seconds:
                self._evict_key(key)
                self.stats.miss_count += 1
                return None
            
            # アクセス時刻更新（LRU用）
            self.access_times[key] = time.time()
            metadata['access_count'] += 1
            
            self.stats.hit_count += 1
            return self.memory_cache[key]
    
    def put(self, key: str, value: Any, priority: float = 1.0) -> bool:
        """キャッシュ格納"""
        with self.lock:
            try:
                # データサイズ計算
                value_size = self._calculate_size(value)
                
                # メモリ制限チェック
                if value_size > self.max_memory_bytes:
                    logger.warning(f"データサイズ ({value_size} bytes) が制限を超過")
                    return False
                
                # メモリ不足時の退避処理
                while (self.stats.memory_usage + value_size > self.max_memory_bytes and
                       len(self.memory_cache) > 0):
                    self._evict_lru()
                
                # キャッシュ格納
                self.memory_cache[key] = value
                self.cache_metadata[key] = {
                    'created_at': time.time(),
                    'size': value_size,
                    'priority': priority,
                    'access_count': 0
                }
                self.access_times[key] = time.time()
                
                self.stats.memory_usage += value_size
                self.stats.total_size += 1
                
                return True
                
            except Exception as e:
                logger.error(f"キャッシュ格納エラー: {e}")
                return False
    
    def invalidate(self, key: str) -> bool:
        """キャッシュ無効化"""
        with self.lock:
            if key in self.memory_cache:
                self._evict_key(key)
                return True
            return False
    
    def clear(self):
        """キャッシュクリア"""
        with self.lock:
            self.memory_cache.clear()
            self.cache_metadata.clear()
            self.access_times.clear()
            self.stats = CacheStats()
    
    def _evict_key(self, key: str):
        """キー退避"""
        if key in self.memory_cache:
            size = self.cache_metadata[key]['size']
            del self.memory_cache[key]
            del self.cache_metadata[key]
            del self.access_times[key]
            
            self.stats.memory_usage -= size
            self.stats.total_size -= 1
    
    def _evict_lru(self):
        """LRU退避"""
        if not self.access_times:
            return
        
        # 最も古いアクセスのキーを特定
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict_key(lru_key)
    
    def _calculate_size(self, obj: Any) -> int:
        """オブジェクトサイズ計算"""
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, pd.Series):
                return obj.memory_usage(deep=True)
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                return len(pickle.dumps(obj))
        except:
            return 1024  # デフォルトサイズ

class DataQualityMonitor:
    """データ品質監視システム"""
    
    def __init__(self):
        self.quality_history = deque(maxlen=1000)
        self.thresholds = {
            'completeness': 0.95,
            'consistency': 0.90,
            'accuracy': 0.85,
            'validity': 0.90,
            'uniqueness': 0.95
        }
    
    def assess_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """データ品質評価"""
        logger.info("データ品質評価を開始")
        
        try:
            # 基本統計
            total_cells = df.size
            total_rows = len(df)
            total_columns = len(df.columns)
            
            # 完全性評価（欠損値なし割合）
            missing_cells = df.isnull().sum().sum()
            completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
            
            # 一意性評価（重複なし割合）
            duplicate_rows = df.duplicated().sum()
            uniqueness = 1.0 - (duplicate_rows / total_rows) if total_rows > 0 else 0.0
            
            # 妥当性評価（データ型と範囲）
            validity_score = self._evaluate_validity(df)
            
            # 一貫性評価（データ形式の統一性）
            consistency_score = self._evaluate_consistency(df)
            
            # 精度評価（データの正確性）
            accuracy_score = self._evaluate_accuracy(df)
            
            # 総合スコア計算
            overall_score = (
                completeness * 0.25 +
                uniqueness * 0.20 +
                validity_score * 0.20 +
                consistency_score * 0.20 +
                accuracy_score * 0.15
            )
            
            # 問題点の特定
            issues = []
            recommendations = []
            
            if completeness < self.thresholds['completeness']:
                issues.append(f"欠損値が多い: {missing_cells}個 ({(1-completeness)*100:.1f}%)")
                recommendations.append("欠損値の補完または行削除を検討")
            
            if uniqueness < self.thresholds['uniqueness']:
                issues.append(f"重複行が多い: {duplicate_rows}行")
                recommendations.append("重複行の削除を検討")
            
            if validity_score < self.thresholds['validity']:
                issues.append("データ妥当性に問題あり")
                recommendations.append("データ範囲とデータ型の見直しを実施")
            
            if consistency_score < self.thresholds['consistency']:
                issues.append("データ一貫性に問題あり")
                recommendations.append("データ形式の統一を実施")
            
            report = DataQualityReport(
                completeness=completeness,
                consistency=consistency_score,
                accuracy=accuracy_score,
                validity=validity_score,
                uniqueness=uniqueness,
                overall_score=overall_score,
                issues=issues,
                recommendations=recommendations
            )
            
            # 履歴保存
            self.quality_history.append(report)
            
            quality_level = self._determine_quality_level(overall_score)
            logger.info(f"データ品質評価完了: {quality_level.value} (スコア: {overall_score:.3f})")
            
            return report
            
        except Exception as e:
            logger.error(f"データ品質評価エラー: {e}")
            return DataQualityReport(
                completeness=0.0, consistency=0.0, accuracy=0.0, 
                validity=0.0, uniqueness=0.0, overall_score=0.0,
                issues=[f"品質評価エラー: {e}"],
                recommendations=["データの基本検証を実施してください"]
            )
    
    def _evaluate_validity(self, df: pd.DataFrame) -> float:
        """妥当性評価"""
        validity_scores = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # 文字列データの妥当性
                valid_ratio = 1.0  # 基本的に有効とする
            elif np.issubdtype(df[column].dtype, np.number):
                # 数値データの妥当性（無限大・NaN以外）
                invalid_count = np.sum(np.isinf(df[column]) | np.isnan(df[column]))
                valid_ratio = 1.0 - (invalid_count / len(df[column]))
            else:
                valid_ratio = 1.0
            
            validity_scores.append(valid_ratio)
        
        return np.mean(validity_scores) if validity_scores else 0.0
    
    def _evaluate_consistency(self, df: pd.DataFrame) -> float:
        """一貫性評価"""
        consistency_scores = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # 文字列の形式一貫性（長さのばらつき）
                str_lengths = df[column].dropna().astype(str).str.len()
                if len(str_lengths) > 1:
                    cv = str_lengths.std() / str_lengths.mean() if str_lengths.mean() > 0 else 0
                    consistency = max(0, 1.0 - cv)
                else:
                    consistency = 1.0
            else:
                # 数値データの一貫性（外れ値の検出）
                numeric_data = df[column].dropna()
                if len(numeric_data) > 3:
                    z_scores = np.abs(stats.zscore(numeric_data))
                    outlier_ratio = np.sum(z_scores > 3) / len(numeric_data)
                    consistency = 1.0 - outlier_ratio
                else:
                    consistency = 1.0
            
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _evaluate_accuracy(self, df: pd.DataFrame) -> float:
        """精度評価"""
        # 基本的な精度指標（完全性に基づく簡易評価）
        non_null_ratio = 1.0 - (df.isnull().sum().sum() / df.size)
        return non_null_ratio
    
    def _determine_quality_level(self, score: float) -> DataQuality:
        """品質レベル決定"""
        if score >= 0.95:
            return DataQuality.EXCELLENT
        elif score >= 0.85:
            return DataQuality.GOOD
        elif score >= 0.75:
            return DataQuality.FAIR
        elif score >= 0.60:
            return DataQuality.POOR
        else:
            return DataQuality.CRITICAL

class ParallelDataProcessor:
    """並列データ処理システム"""
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or min(8, os.cpu_count() or 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.n_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, self.n_workers))
        
    @jit(nopython=True)
    def _numba_apply(self, data: np.ndarray, func_id: int) -> np.ndarray:
        """Numba並列処理"""
        result = np.empty_like(data)
        
        for i in prange(len(data)):
            if func_id == 1:  # square
                result[i] = data[i] ** 2
            elif func_id == 2:  # log
                result[i] = np.log(data[i] + 1e-10)
            elif func_id == 3:  # normalize
                result[i] = (data[i] - np.mean(data)) / (np.std(data) + 1e-10)
            else:
                result[i] = data[i]
        
        return result
    
    async def process_chunks(self, df: pd.DataFrame, 
                           processing_func: Callable[[pd.DataFrame], pd.DataFrame],
                           chunk_size: int = 10000,
                           strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL) -> pd.DataFrame:
        """チャンク並列処理"""
        logger.info(f"並列処理開始: 戦略={strategy.value}, チャンクサイズ={chunk_size}")
        
        try:
            if strategy == ProcessingStrategy.DASK and DASK_AVAILABLE:
                return await self._process_with_dask(df, processing_func, chunk_size)
            
            elif strategy == ProcessingStrategy.POLARS and POLARS_AVAILABLE:
                return await self._process_with_polars(df, processing_func)
            
            elif strategy == ProcessingStrategy.NUMBA:
                return await self._process_with_numba(df, processing_func)
            
            else:
                return await self._process_with_threading(df, processing_func, chunk_size)
                
        except Exception as e:
            logger.error(f"並列処理エラー: {e}")
            return df  # 失敗時は元のデータを返す
    
    async def _process_with_threading(self, df: pd.DataFrame, 
                                    processing_func: Callable, 
                                    chunk_size: int) -> pd.DataFrame:
        """スレッド並列処理"""
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        loop = asyncio.get_event_loop()
        futures = []
        
        for chunk in chunks:
            future = loop.run_in_executor(self.thread_pool, processing_func, chunk)
            futures.append(future)
        
        processed_chunks = await asyncio.gather(*futures)
        return pd.concat(processed_chunks, ignore_index=True)
    
    async def _process_with_dask(self, df: pd.DataFrame, 
                               processing_func: Callable,
                               chunk_size: int) -> pd.DataFrame:
        """Dask並列処理"""
        ddf = dd.from_pandas(df, npartitions=max(1, len(df) // chunk_size))
        
        # 非同期でDask処理実行
        loop = asyncio.get_event_loop()
        
        def dask_compute():
            return ddf.apply(processing_func, axis=1).compute()
        
        result = await loop.run_in_executor(self.process_pool, dask_compute)
        return result
    
    async def _process_with_polars(self, df: pd.DataFrame, 
                                 processing_func: Callable) -> pd.DataFrame:
        """Polars処理"""
        # Pandas → Polars 変換
        pl_df = pl.from_pandas(df)
        
        # 処理実行（Polarsの高速処理）
        loop = asyncio.get_event_loop()
        
        def polars_process():
            # 簡単な処理例（実際の処理関数によって調整が必要）
            return pl_df.to_pandas()
        
        result = await loop.run_in_executor(self.thread_pool, polars_process)
        return result
    
    async def _process_with_numba(self, df: pd.DataFrame, 
                                processing_func: Callable) -> pd.DataFrame:
        """Numba加速処理"""
        result_df = df.copy()
        
        # 数値列に対してNumba処理適用
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if not df[col].empty:
                # Numba関数適用
                result_df[col] = self._numba_apply(df[col].values, 1)  # square function
        
        return result_df
    
    def cleanup(self):
        """リソース クリーンアップ"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class SmartDataTransformer:
    """スマート データ変換システム"""
    
    def __init__(self):
        self.transformers = {}
        self.transformation_history = []
        
    def auto_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """自動データ変換"""
        logger.info("自動データ変換を開始")
        
        transformation_log = {
            'applied_transformations': [],
            'skipped_columns': [],
            'warnings': []
        }
        
        result_df = df.copy()
        
        try:
            # 1. 欠損値処理
            result_df, missing_log = self._handle_missing_values(result_df)
            transformation_log['applied_transformations'].append(missing_log)
            
            # 2. データ型最適化
            result_df, dtype_log = self._optimize_dtypes(result_df)
            transformation_log['applied_transformations'].append(dtype_log)
            
            # 3. 外れ値処理
            result_df, outlier_log = self._handle_outliers(result_df)
            transformation_log['applied_transformations'].append(outlier_log)
            
            # 4. 特徴量正規化
            result_df, scaling_log = self._apply_scaling(result_df)
            transformation_log['applied_transformations'].append(scaling_log)
            
            # 5. カテゴリカル変数エンコーディング
            result_df, encoding_log = self._encode_categorical(result_df)
            transformation_log['applied_transformations'].append(encoding_log)
            
            logger.info(f"自動変換完了: {len(transformation_log['applied_transformations'])}個の変換を適用")
            
        except Exception as e:
            logger.error(f"自動変換エラー: {e}")
            transformation_log['warnings'].append(f"変換エラー: {e}")
            
        # 履歴保存
        self.transformation_history.append({
            'timestamp': datetime.now(),
            'input_shape': df.shape,
            'output_shape': result_df.shape,
            'transformations': transformation_log
        })
        
        return result_df, transformation_log
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """欠損値処理"""
        log = {'transformation': 'missing_values', 'details': {}}
        result_df = df.copy()
        
        for column in df.columns:
            missing_ratio = df[column].isnull().mean()
            
            if missing_ratio > 0:
                if missing_ratio > 0.5:
                    # 50%以上欠損している列は削除
                    result_df = result_df.drop(columns=[column])
                    log['details'][column] = 'dropped_high_missing'
                    
                elif df[column].dtype == 'object':
                    # カテゴリカル データは最頻値で補完
                    mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'unknown'
                    result_df[column] = result_df[column].fillna(mode_value)
                    log['details'][column] = f'filled_with_mode: {mode_value}'
                    
                else:
                    # 数値データは中央値で補完
                    median_value = df[column].median()
                    result_df[column] = result_df[column].fillna(median_value)
                    log['details'][column] = f'filled_with_median: {median_value}'
        
        return result_df, log
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """データ型最適化"""
        log = {'transformation': 'dtype_optimization', 'details': {}}
        result_df = df.copy()
        
        for column in df.columns:
            original_dtype = str(df[column].dtype)
            
            if df[column].dtype == 'object':
                # 文字列を数値に変換可能か試行
                try:
                    numeric_series = pd.to_numeric(df[column], errors='coerce')
                    if not numeric_series.isnull().all():
                        result_df[column] = numeric_series
                        log['details'][column] = f'{original_dtype} -> numeric'
                except:
                    pass
            
            elif np.issubdtype(df[column].dtype, np.integer):
                # 整数型の最適化
                col_min = df[column].min()
                col_max = df[column].max()
                
                if col_min >= -128 and col_max <= 127:
                    result_df[column] = df[column].astype(np.int8)
                    log['details'][column] = f'{original_dtype} -> int8'
                elif col_min >= -32768 and col_max <= 32767:
                    result_df[column] = df[column].astype(np.int16)
                    log['details'][column] = f'{original_dtype} -> int16'
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    result_df[column] = df[column].astype(np.int32)
                    log['details'][column] = f'{original_dtype} -> int32'
            
            elif np.issubdtype(df[column].dtype, np.floating):
                # 浮動小数点型の最適化
                result_df[column] = df[column].astype(np.float32)
                if original_dtype != 'float32':
                    log['details'][column] = f'{original_dtype} -> float32'
        
        return result_df, log
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """外れ値処理"""
        log = {'transformation': 'outlier_handling', 'details': {}}
        result_df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if len(df[column].dropna()) > 10:  # 十分なデータがある場合のみ
                # IQR法で外れ値検出
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                
                if outliers_count > 0:
                    # 外れ値をキャップ
                    result_df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                    log['details'][column] = f'clipped_{outliers_count}_outliers'
        
        return result_df, log
    
    def _apply_scaling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """スケーリング適用"""
        log = {'transformation': 'scaling', 'details': {}}
        result_df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if len(df[column].dropna()) > 1:
                # 標準化適用
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(df[column].values.reshape(-1, 1)).flatten()
                result_df[column] = scaled_values
                log['details'][column] = 'standard_scaled'
                
                # 変換器保存
                self.transformers[f'{column}_scaler'] = scaler
        
        return result_df, log
    
    def _encode_categorical(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """カテゴリカル エンコーディング"""
        log = {'transformation': 'categorical_encoding', 'details': {}}
        result_df = df.copy()
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            unique_count = df[column].nunique()
            
            if unique_count <= 10:
                # One-Hot エンコーディング
                dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                result_df = result_df.drop(columns=[column])
                result_df = pd.concat([result_df, dummies], axis=1)
                log['details'][column] = f'one_hot_encoded_{unique_count}_categories'
                
            else:
                # Label エンコーディング
                encoder = LabelEncoder()
                result_df[column] = encoder.fit_transform(df[column].fillna('unknown'))
                log['details'][column] = f'label_encoded_{unique_count}_categories'
                
                # エンコーダー保存
                self.transformers[f'{column}_encoder'] = encoder
        
        return result_df, log

class AdvancedDataProcessingSystem:
    """高度データ処理システム（メインクラス）"""
    
    def __init__(self, cache_size_mb: int = 1024, n_workers: int = None):
        self.intelligent_cache = IntelligentCache(max_memory_mb=cache_size_mb)
        self.quality_monitor = DataQualityMonitor()
        self.parallel_processor = ParallelDataProcessor(n_workers=n_workers)
        self.data_transformer = SmartDataTransformer()
        
        # 統計情報
        self.processing_stats = {
            'datasets_processed': 0,
            'total_processing_time': 0.0,
            'cache_operations': 0,
            'transformations_applied': 0,
            'quality_assessments': 0
        }
        
        # パフォーマンス履歴
        self.performance_history = deque(maxlen=1000)
        
        logger.info("高度データ処理システムを初期化しました")
    
    async def process_dataset(self, data: Union[pd.DataFrame, str, Path],
                            processing_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """データセット包括処理"""
        start_time = time.time()
        
        # デフォルト設定
        config = {
            'strategy': ProcessingStrategy.PARALLEL,
            'cache_enabled': True,
            'quality_check': True,
            'auto_transform': True,
            'chunk_size': 10000
        }
        config.update(processing_config or {})
        
        logger.info("データセット処理を開始")
        
        try:
            # 1. データ読み込み
            if isinstance(data, (str, Path)):
                df = await self._load_data(data)
            else:
                df = data.copy()
            
            # データハッシュ生成（キャッシュキー用）
            data_hash = self._generate_data_hash(df)
            
            # 2. キャッシュチェック
            cached_result = None
            if config['cache_enabled']:
                cache_key = f"processed_{data_hash}_{hash(str(config))}"
                cached_result = self.intelligent_cache.get(cache_key)
                
                if cached_result:
                    logger.info("キャッシュからデータを取得")
                    self.processing_stats['cache_operations'] += 1
                    return cached_result
            
            # 3. データ品質評価
            quality_report = None
            if config['quality_check']:
                quality_report = self.quality_monitor.assess_quality(df)
                self.processing_stats['quality_assessments'] += 1
            
            # 4. 自動データ変換
            transformed_df = df
            transformation_log = {}
            
            if config['auto_transform']:
                transformed_df, transformation_log = self.data_transformer.auto_transform(df)
                self.processing_stats['transformations_applied'] += 1
            
            # 5. 並列処理
            def identity_transform(chunk):
                return chunk  # 基本的な処理（実際には具体的な処理を指定）
            
            processed_df = await self.parallel_processor.process_chunks(
                transformed_df, 
                identity_transform,
                config['chunk_size'],
                config['strategy']
            )
            
            processing_time = time.time() - start_time
            
            # 6. 結果まとめ
            result = {
                'processed_data': processed_df,
                'original_shape': df.shape,
                'processed_shape': processed_df.shape,
                'quality_report': quality_report,
                'transformation_log': transformation_log,
                'processing_metrics': ProcessingMetrics(
                    processing_time=processing_time,
                    memory_usage=self._calculate_memory_usage(processed_df),
                    cpu_usage=psutil.cpu_percent(),
                    throughput=len(processed_df) / processing_time if processing_time > 0 else 0,
                    cache_hit_rate=self.intelligent_cache.stats.hit_rate,
                    error_rate=0.0,
                    data_quality_score=quality_report.overall_score if quality_report else 0.5
                ),
                'config': config
            }
            
            # 7. キャッシュ保存
            if config['cache_enabled'] and not cached_result:
                cache_success = self.intelligent_cache.put(cache_key, result, priority=1.0)
                if cache_success:
                    self.processing_stats['cache_operations'] += 1
            
            # 統計更新
            self.processing_stats['datasets_processed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            self.performance_history.append(result['processing_metrics'])
            
            logger.info(f"データセット処理完了: {processing_time:.2f}秒, "
                       f"品質スコア={result['processing_metrics'].data_quality_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"データセット処理エラー: {e}")
            raise
    
    async def batch_process_datasets(self, datasets: List[Union[pd.DataFrame, str, Path]],
                                   processing_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """バッチデータセット処理"""
        logger.info(f"バッチ処理開始: {len(datasets)}件")
        
        # 並列処理でバッチ実行
        tasks = [
            self.process_dataset(dataset, processing_config)
            for dataset in datasets
        ]
        
        results = []
        completed_count = 0
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
                completed_count += 1
                
                if completed_count % 10 == 0:
                    logger.info(f"バッチ処理進捗: {completed_count}/{len(datasets)}")
                    
            except Exception as e:
                logger.error(f"バッチ処理エラー: {e}")
                results.append({'error': str(e)})
        
        logger.info(f"バッチ処理完了: {len(results)}件")
        return results
    
    async def _load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """データ読み込み"""
        path = Path(data_path)
        
        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif path.suffix == '.json':
            return pd.read_json(path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(path)
        else:
            raise ValueError(f"サポートされていないファイル形式: {path.suffix}")
    
    def _generate_data_hash(self, df: pd.DataFrame) -> str:
        """データハッシュ生成"""
        try:
            # データの形状とサンプル値からハッシュ生成
            shape_str = f"{df.shape[0]}x{df.shape[1]}"
            columns_str = ",".join(df.columns.tolist())
            sample_str = str(df.head().values.tobytes()) if not df.empty else ""
            
            combined = f"{shape_str}_{columns_str}_{sample_str}"
            return hashlib.md5(combined.encode()).hexdigest()[:16]
        except:
            return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
    
    def _calculate_memory_usage(self, df: pd.DataFrame) -> float:
        """メモリ使用量計算（MB）"""
        try:
            return df.memory_usage(deep=True).sum() / (1024 * 1024)
        except:
            return 0.0
    
    def get_system_report(self) -> Dict[str, Any]:
        """システム レポート"""
        try:
            recent_performance = list(self.performance_history)[-10:] if self.performance_history else []
            
            return {
                'processing_statistics': self.processing_stats,
                'cache_statistics': {
                    'hit_rate': self.intelligent_cache.stats.hit_rate,
                    'hit_count': self.intelligent_cache.stats.hit_count,
                    'miss_count': self.intelligent_cache.stats.miss_count,
                    'total_size': self.intelligent_cache.stats.total_size,
                    'memory_usage_mb': self.intelligent_cache.stats.memory_usage / (1024 * 1024)
                },
                'recent_performance': [
                    {
                        'processing_time': p.processing_time,
                        'throughput': p.throughput,
                        'quality_score': p.data_quality_score,
                        'timestamp': p.timestamp.isoformat()
                    }
                    for p in recent_performance
                ],
                'quality_assessments': len(self.quality_monitor.quality_history),
                'transformation_history': len(self.data_transformer.transformation_history),
                'available_strategies': [strategy.value for strategy in ProcessingStrategy],
                'system_capabilities': {
                    'dask_available': DASK_AVAILABLE,
                    'polars_available': POLARS_AVAILABLE,
                    'vaex_available': VAEX_AVAILABLE,
                    'parallel_workers': self.parallel_processor.n_workers
                }
            }
            
        except Exception as e:
            logger.error(f"システム レポート生成エラー: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """システムクリーンアップ"""
        self.intelligent_cache.clear()
        self.parallel_processor.cleanup()
        gc.collect()
        logger.info("データ処理システムをクリーンアップしました")

# 使用例とデモ
async def demo_data_processing_system():
    """データ処理システム デモ"""
    try:
        # システム初期化
        system = AdvancedDataProcessingSystem(cache_size_mb=512, n_workers=4)
        
        # サンプルデータ作成
        n_rows = 100000
        n_cols = 20
        
        # 現実的なサンプルデータ
        sample_data = pd.DataFrame({
            f'numeric_{i}': np.random.randn(n_rows) * 100 + np.random.randint(0, 1000)
            for i in range(n_cols // 2)
        })
        
        # カテゴリカルデータ追加
        for i in range(n_cols // 2):
            categories = [f'category_{j}' for j in range(5)]
            sample_data[f'categorical_{i}'] = np.random.choice(categories, n_rows)
        
        # 欠損値を意図的に追加
        for col in sample_data.columns[:5]:
            missing_indices = np.random.choice(n_rows, size=int(n_rows * 0.1), replace=False)
            sample_data.loc[missing_indices, col] = np.nan
        
        print(f"サンプルデータ作成完了: {sample_data.shape}")
        
        # 処理設定
        config = {
            'strategy': ProcessingStrategy.PARALLEL,
            'cache_enabled': True,
            'quality_check': True,
            'auto_transform': True,
            'chunk_size': 20000
        }
        
        # データセット処理
        result = await system.process_dataset(sample_data, config)
        
        print("処理結果:")
        print(f"  元データ形状: {result['original_shape']}")
        print(f"  処理後形状: {result['processed_shape']}")
        print(f"  処理時間: {result['processing_metrics'].processing_time:.2f}秒")
        print(f"  スループット: {result['processing_metrics'].throughput:.0f} rows/sec")
        print(f"  データ品質スコア: {result['processing_metrics'].data_quality_score:.3f}")
        print(f"  キャッシュ ヒット率: {result['processing_metrics'].cache_hit_rate:.3f}")
        
        # 品質レポート
        if result['quality_report']:
            quality = result['quality_report']
            print(f"品質評価:")
            print(f"  完全性: {quality.completeness:.3f}")
            print(f"  一意性: {quality.uniqueness:.3f}")
            print(f"  妥当性: {quality.validity:.3f}")
            print(f"  総合スコア: {quality.overall_score:.3f}")
        
        # システム レポート
        system_report = system.get_system_report()
        print("システム統計:")
        print(f"  処理済みデータセット: {system_report['processing_statistics']['datasets_processed']}")
        print(f"  キャッシュヒット率: {system_report['cache_statistics']['hit_rate']:.3f}")
        
        # クリーンアップ
        system.cleanup()
        
        return system
        
    except Exception as e:
        logger.error(f"デモ実行エラー: {e}")
        raise

if __name__ == "__main__":
    # デモ実行
    asyncio.run(demo_data_processing_system())