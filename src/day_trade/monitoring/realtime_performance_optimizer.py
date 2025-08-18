#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime Performance Optimizer - リアルタイム性能最適化システム

Issue #802実装：リアルタイム性と性能の最適化
高速データ処理、キャッシュ、並行処理による性能向上
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import psutil

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

class CacheLevel(Enum):
    """キャッシュレベル"""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"

@dataclass
class PerformanceMetrics:
    """性能メトリクス"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    response_times: Dict[str, float]
    cache_hit_rates: Dict[str, float]
    throughput: Dict[str, int]
    error_rates: Dict[str, float]

@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    expiry: Optional[datetime] = None
    size_bytes: int = 0

class HighPerformanceCache:
    """高性能キャッシュシステム"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_queue = deque()  # LRU追跡用
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }

        # バックグラウンドクリーンアップ
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def get(self, key: str) -> Optional[Any]:
        """キャッシュから値を取得"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]

                # 有効期限チェック
                if entry.expiry and datetime.now() > entry.expiry:
                    self._remove_entry(key)
                    self.stats['misses'] += 1
                    return None

                # アクセス情報更新
                entry.last_accessed = datetime.now()
                entry.access_count += 1

                # LRUキューに移動
                if key in self.access_queue:
                    self.access_queue.remove(key)
                self.access_queue.append(key)

                self.stats['hits'] += 1
                return entry.value
            else:
                self.stats['misses'] += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """キャッシュに値を設定"""
        with self.lock:
            now = datetime.now()
            expiry = now + timedelta(seconds=ttl or self.ttl_seconds)

            # サイズ計算（簡易版）
            try:
                size_bytes = sys.getsizeof(value)
            except:
                size_bytes = 1024  # デフォルトサイズ

            # 容量チェックと古いエントリの削除
            while len(self.cache) >= self.max_size:
                if self.access_queue:
                    old_key = self.access_queue.popleft()
                    if old_key in self.cache:
                        self._remove_entry(old_key)
                        self.stats['evictions'] += 1
                else:
                    break

            # 新しいエントリ作成
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                expiry=expiry,
                size_bytes=size_bytes
            )

            self.cache[key] = entry
            self.access_queue.append(key)
            self.stats['size'] = len(self.cache)

            return True

    def _remove_entry(self, key: str):
        """エントリ削除"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_queue:
            self.access_queue.remove(key)
        self.stats['size'] = len(self.cache)

    def _cleanup_loop(self):
        """期限切れエントリのクリーンアップ"""
        while True:
            try:
                with self.lock:
                    now = datetime.now()
                    expired_keys = [
                        key for key, entry in self.cache.items()
                        if entry.expiry and now > entry.expiry
                    ]

                    for key in expired_keys:
                        self._remove_entry(key)

                time.sleep(60)  # 1分毎にクリーンアップ
            except Exception as e:
                logging.error(f"キャッシュクリーンアップエラー: {e}")
                time.sleep(300)  # エラー時は5分待機

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        with self.lock:
            hit_rate = (self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'total_requests': self.stats['hits'] + self.stats['misses']
            }

class AsyncDataProcessor:
    """非同期データ処理エンジン"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = asyncio.Queue()
        self.result_cache = HighPerformanceCache()

    async def process_batch(self, data_batch: List[Dict[str, Any]], processor_func) -> List[Any]:
        """バッチデータ処理"""

        start_time = time.time()
        results = []

        # 並列処理でバッチを分割
        chunk_size = max(1, len(data_batch) // self.max_workers)
        chunks = [data_batch[i:i + chunk_size] for i in range(0, len(data_batch), chunk_size)]

        # 各チャンクを並行処理
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._process_chunk, chunk, processor_func)
            futures.append(future)

        # 結果収集
        for future in as_completed(futures):
            try:
                chunk_results = future.result(timeout=30)
                results.extend(chunk_results)
            except Exception as e:
                logging.error(f"チャンク処理エラー: {e}")
                continue

        processing_time = time.time() - start_time
        logging.info(f"バッチ処理完了: {len(data_batch)}件 -> {len(results)}件 ({processing_time:.2f}秒)")

        return results

    def _process_chunk(self, chunk: List[Dict[str, Any]], processor_func) -> List[Any]:
        """チャンク処理"""
        results = []
        for item in chunk:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                logging.warning(f"アイテム処理エラー: {e}")
                continue
        return results

class RealtimePerformanceOptimizer:
    """リアルタイム性能最適化システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # キャッシュシステム
        self.data_cache = HighPerformanceCache(max_size=2000, ttl_seconds=300)
        self.prediction_cache = HighPerformanceCache(max_size=1000, ttl_seconds=600)
        self.analysis_cache = HighPerformanceCache(max_size=500, ttl_seconds=1800)

        # 非同期処理エンジン
        self.async_processor = AsyncDataProcessor(max_workers=4)

        # パフォーマンスメトリクス追跡
        self.metrics_history = deque(maxlen=1440)  # 24時間分（1分毎）
        self.performance_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'response_time': 2.0,
            'cache_hit_rate': 0.8
        }

        # データベース設定
        self.db_path = Path("performance_data/optimization_metrics.db")
        self.db_path.parent.mkdir(exist_ok=True)

        self._init_database()

        # メトリクス収集開始
        self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        self.metrics_thread.start()

        self.logger.info("Realtime performance optimizer initialized")

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # パフォーマンスメトリクステーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        cpu_usage REAL,
                        memory_usage REAL,
                        disk_usage REAL,
                        network_io TEXT,
                        response_times TEXT,
                        cache_hit_rates TEXT,
                        throughput TEXT,
                        error_rates TEXT
                    )
                ''')

                # 最適化イベントテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        description TEXT,
                        impact_score REAL,
                        success INTEGER DEFAULT 1
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    async def optimize_data_retrieval(self, symbol: str, period: str = "5d",
                                   force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """データ取得最適化"""

        cache_key = f"data_{symbol}_{period}"

        # キャッシュチェック
        if not force_refresh:
            cached_data = self.data_cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"キャッシュヒット: {cache_key}")
                return cached_data

        # データ取得
        start_time = time.time()
        try:
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            response_time = time.time() - start_time

            # キャッシュに保存
            if data is not None:
                self.data_cache.set(cache_key, data, ttl=300)  # 5分間キャッシュ

            # 性能メトリクス記録
            self._record_response_time("data_retrieval", response_time)

            return data

        except Exception as e:
            self.logger.error(f"データ取得エラー {symbol}: {e}")
            return None

    async def optimize_prediction(self, symbol: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """予測最適化"""

        cache_key = f"prediction_{symbol}"

        # キャッシュチェック
        if not force_refresh:
            cached_prediction = self.prediction_cache.get(cache_key)
            if cached_prediction is not None:
                self.logger.debug(f"予測キャッシュヒット: {cache_key}")
                return cached_prediction

        # 予測実行
        start_time = time.time()
        try:
            from optimized_prediction_system import optimized_prediction_system
            prediction = await optimized_prediction_system.predict_with_optimized_models(symbol)

            response_time = time.time() - start_time

            if prediction:
                # 予測結果を辞書に変換
                prediction_dict = {
                    'symbol': prediction.symbol,
                    'prediction': prediction.prediction,
                    'confidence': prediction.confidence,
                    'model_consensus': prediction.model_consensus,
                    'timestamp': prediction.timestamp.isoformat()
                }

                # キャッシュに保存（10分間）
                self.prediction_cache.set(cache_key, prediction_dict, ttl=600)

                # 性能メトリクス記録
                self._record_response_time("prediction", response_time)

                return prediction_dict

            return None

        except Exception as e:
            self.logger.error(f"予測エラー {symbol}: {e}")
            return None

    async def optimize_batch_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """バッチ分析最適化"""

        start_time = time.time()
        results = {}

        # 並行処理でシンボルを処理
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._analyze_symbol_optimized(symbol))
            tasks.append((symbol, task))

        # 結果収集
        for symbol, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=10)
                if result:
                    results[symbol] = result
            except asyncio.TimeoutError:
                self.logger.warning(f"分析タイムアウト: {symbol}")
            except Exception as e:
                self.logger.error(f"分析エラー {symbol}: {e}")

        processing_time = time.time() - start_time
        self._record_response_time("batch_analysis", processing_time)

        return {
            'results': results,
            'processing_time': processing_time,
            'symbols_processed': len(results),
            'success_rate': len(results) / len(symbols) if symbols else 0
        }

    async def _analyze_symbol_optimized(self, symbol: str) -> Optional[Dict[str, Any]]:
        """シンボル分析最適化"""

        cache_key = f"analysis_{symbol}"

        # キャッシュチェック
        cached_analysis = self.analysis_cache.get(cache_key)
        if cached_analysis is not None:
            return cached_analysis

        try:
            # データ取得（キャッシュ使用）
            data = await self.optimize_data_retrieval(symbol, "5d")
            if data is None or len(data) < 5:
                return None

            # 基本分析
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].pct_change().iloc[-1]
            volume_change = data['Volume'].pct_change().iloc[-1]

            # ボラティリティ
            volatility = data['Close'].pct_change().rolling(5).std().iloc[-1]

            # トレンド
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            trend = (current_price - sma_5) / sma_5

            analysis_result = {
                'symbol': symbol,
                'current_price': current_price,
                'price_change': price_change,
                'volume_change': volume_change,
                'volatility': volatility,
                'trend': trend,
                'timestamp': datetime.now().isoformat()
            }

            # キャッシュに保存（30分間）
            self.analysis_cache.set(cache_key, analysis_result, ttl=1800)

            return analysis_result

        except Exception as e:
            self.logger.error(f"シンボル分析エラー {symbol}: {e}")
            return None

    def _record_response_time(self, operation: str, response_time: float):
        """応答時間記録"""

        if not hasattr(self, '_response_times'):
            self._response_times = {}

        if operation not in self._response_times:
            self._response_times[operation] = deque(maxlen=100)

        self._response_times[operation].append(response_time)

    def _metrics_collection_loop(self):
        """メトリクス収集ループ"""

        while True:
            try:
                # システムメトリクス収集
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()

                # 応答時間統計
                response_times = {}
                if hasattr(self, '_response_times'):
                    for operation, times in self._response_times.items():
                        if times:
                            response_times[operation] = np.mean(list(times))

                # キャッシュヒット率
                cache_hit_rates = {
                    'data_cache': self.data_cache.get_stats()['hit_rate'],
                    'prediction_cache': self.prediction_cache.get_stats()['hit_rate'],
                    'analysis_cache': self.analysis_cache.get_stats()['hit_rate']
                }

                # メトリクス作成
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    cpu_usage=cpu_usage,
                    memory_usage=memory.percent,
                    disk_usage=disk.percent,
                    network_io={
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv
                    },
                    response_times=response_times,
                    cache_hit_rates=cache_hit_rates,
                    throughput={},
                    error_rates={}
                )

                # 履歴に追加
                self.metrics_history.append(metrics)

                # データベースに保存（5分毎）
                if len(self.metrics_history) % 5 == 0:
                    asyncio.create_task(self._save_metrics_to_db(metrics))

                # パフォーマンス警告チェック
                self._check_performance_thresholds(metrics)

                time.sleep(60)  # 1分間隔

            except Exception as e:
                self.logger.error(f"メトリクス収集エラー: {e}")
                time.sleep(300)  # エラー時は5分待機

    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """性能閾値チェック"""

        warnings = []

        if metrics.cpu_usage > self.performance_thresholds['cpu_usage']:
            warnings.append(f"高CPU使用率: {metrics.cpu_usage:.1f}%")

        if metrics.memory_usage > self.performance_thresholds['memory_usage']:
            warnings.append(f"高メモリ使用率: {metrics.memory_usage:.1f}%")

        # 応答時間チェック
        for operation, time_ms in metrics.response_times.items():
            if time_ms > self.performance_thresholds['response_time']:
                warnings.append(f"遅い応答時間 {operation}: {time_ms:.2f}秒")

        # キャッシュヒット率チェック
        for cache_name, hit_rate in metrics.cache_hit_rates.items():
            if hit_rate < self.performance_thresholds['cache_hit_rate']:
                warnings.append(f"低キャッシュヒット率 {cache_name}: {hit_rate:.1%}")

        if warnings:
            self.logger.warning(f"性能警告: {', '.join(warnings)}")

    async def _save_metrics_to_db(self, metrics: PerformanceMetrics):
        """メトリクスをデータベースに保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO performance_metrics
                    (timestamp, cpu_usage, memory_usage, disk_usage, network_io,
                     response_times, cache_hit_rates, throughput, error_rates)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp.isoformat(),
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    metrics.disk_usage,
                    json.dumps(metrics.network_io),
                    json.dumps(metrics.response_times),
                    json.dumps(metrics.cache_hit_rates),
                    json.dumps(metrics.throughput),
                    json.dumps(metrics.error_rates)
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"メトリクス保存エラー: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """性能レポート取得"""

        if not self.metrics_history:
            return {"error": "メトリクス履歴なし"}

        recent_metrics = list(self.metrics_history)[-60:]  # 過去1時間

        # 統計計算
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])

        # 応答時間統計
        response_time_stats = {}
        if hasattr(self, '_response_times'):
            for operation, times in self._response_times.items():
                if times:
                    response_time_stats[operation] = {
                        'avg': np.mean(list(times)),
                        'min': np.min(list(times)),
                        'max': np.max(list(times)),
                        'p95': np.percentile(list(times), 95)
                    }

        # キャッシュ統計
        cache_stats = {
            'data_cache': self.data_cache.get_stats(),
            'prediction_cache': self.prediction_cache.get_stats(),
            'analysis_cache': self.analysis_cache.get_stats()
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory,
                'metrics_count': len(recent_metrics)
            },
            'response_times': response_time_stats,
            'cache_stats': cache_stats,
            'performance_status': self._get_performance_status(avg_cpu, avg_memory, cache_stats)
        }

    def _get_performance_status(self, cpu: float, memory: float, cache_stats: Dict[str, Any]) -> str:
        """性能状態判定"""

        issues = 0

        if cpu > 80:
            issues += 1
        if memory > 85:
            issues += 1

        # キャッシュヒット率チェック
        for cache_name, stats in cache_stats.items():
            if stats['hit_rate'] < 0.7:
                issues += 1

        if issues == 0:
            return "EXCELLENT"
        elif issues <= 1:
            return "GOOD"
        elif issues <= 2:
            return "WARNING"
        else:
            return "CRITICAL"

# グローバルインスタンス
realtime_performance_optimizer = RealtimePerformanceOptimizer()

# テスト実行
async def run_performance_optimization_test():
    """性能最適化テスト実行"""

    print("=== 🚀 リアルタイム性能最適化テスト ===")

    test_symbols = ["7203", "8306", "4751"]

    # 個別最適化テスト
    print(f"\n📊 個別データ取得最適化テスト")
    for symbol in test_symbols:
        start_time = time.time()
        data = await realtime_performance_optimizer.optimize_data_retrieval(symbol)
        response_time = time.time() - start_time

        status = "✅" if data is not None else "❌"
        print(f"  {status} {symbol}: {response_time:.3f}秒")

    # バッチ分析最適化テスト
    print(f"\n⚡ バッチ分析最適化テスト")
    batch_result = await realtime_performance_optimizer.optimize_batch_analysis(test_symbols)

    print(f"  処理時間: {batch_result['processing_time']:.3f}秒")
    print(f"  成功率: {batch_result['success_rate']:.1%}")
    print(f"  処理銘柄数: {batch_result['symbols_processed']}")

    # キャッシュ効果テスト
    print(f"\n💾 キャッシュ効果テスト")
    symbol = "7203"

    # 1回目（キャッシュミス）
    start_time = time.time()
    await realtime_performance_optimizer.optimize_data_retrieval(symbol)
    first_time = time.time() - start_time

    # 2回目（キャッシュヒット）
    start_time = time.time()
    await realtime_performance_optimizer.optimize_data_retrieval(symbol)
    second_time = time.time() - start_time

    speedup = first_time / second_time if second_time > 0 else 1
    print(f"  1回目（キャッシュミス）: {first_time:.3f}秒")
    print(f"  2回目（キャッシュヒット）: {second_time:.3f}秒")
    print(f"  高速化倍率: {speedup:.1f}x")

    # 性能レポート
    print(f"\n📈 性能レポート")
    report = realtime_performance_optimizer.get_performance_report()

    print(f"  システム状態: {report['performance_status']}")
    print(f"  平均CPU使用率: {report['system_metrics']['avg_cpu_usage']:.1f}%")
    print(f"  平均メモリ使用率: {report['system_metrics']['avg_memory_usage']:.1f}%")

    # キャッシュ統計
    cache_stats = report['cache_stats']
    print(f"  データキャッシュヒット率: {cache_stats['data_cache']['hit_rate']:.1%}")
    print(f"  予測キャッシュヒット率: {cache_stats['prediction_cache']['hit_rate']:.1%}")
    print(f"  分析キャッシュヒット率: {cache_stats['analysis_cache']['hit_rate']:.1%}")

    print(f"\n✅ リアルタイム性能最適化システム稼働中")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_performance_optimization_test())