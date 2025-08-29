"""
高速処理エンジン

並列処理・メモリ最適化・キャッシュ戦略を統合した
高性能データ処理エンジン
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import concurrent.futures
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
import psutil
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import sqlite3
import pickle
import hashlib
import weakref
from functools import wraps, lru_cache
import gc
import sys


@dataclass
class ProcessingTask:
    """処理タスク"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 1
    timeout_seconds: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingResult:
    """処理結果"""
    task_id: str
    result: Any
    success: bool
    execution_time_ms: float
    memory_usage_mb: float
    error_message: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """性能メトリクス"""
    timestamp: datetime
    tasks_completed: int
    avg_execution_time_ms: float
    peak_memory_mb: float
    cpu_utilization: float
    cache_hit_rate: float
    throughput_tasks_per_sec: float
    queue_size: int


class IntelligentCache:
    """インテリジェントキャッシュシステム"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        # 自動クリーンアップ
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_active = True
        self._cleanup_thread.start()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """キー生成"""
        key_data = (func_name, args, tuple(sorted(kwargs.items())))
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """値取得"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # TTLチェック
                if datetime.now() - timestamp <= timedelta(seconds=self.ttl_seconds):
                    self.access_times[key] = datetime.now()
                    self.hit_count += 1
                    return value
                else:
                    # 期限切れ削除
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any):
        """値保存"""
        with self.lock:
            current_time = datetime.now()
            
            # サイズ制限チェック
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = (value, current_time)
            self.access_times[key] = current_time
    
    def _evict_lru(self):
        """LRU削除"""
        if not self.access_times:
            return
        
        # 最も古いアクセスの項目を削除
        lru_key = min(self.access_times, key=self.access_times.get)
        
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def _cleanup_loop(self):
        """定期クリーンアップ"""
        while self._cleanup_active:
            try:
                time.sleep(300)  # 5分間隔
                self._cleanup_expired()
            except Exception:
                pass
    
    def _cleanup_expired(self):
        """期限切れ項目削除"""
        with self.lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, (_, timestamp) in self.cache.items():
                if current_time - timestamp > timedelta(seconds=self.ttl_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self.cache:
                    del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """統計取得"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }
    
    def clear(self):
        """キャッシュクリア"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def __del__(self):
        self._cleanup_active = False


def cached(ttl_seconds: int = 3600):
    """キャッシュデコレータ"""
    def decorator(func: Callable) -> Callable:
        cache = IntelligentCache(ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # キー生成
            key = cache._generate_key(func.__name__, args, kwargs)
            
            # キャッシュから取得試行
            result = cache.get(key)
            if result is not None:
                return result
            
            # 関数実行
            result = func(*args, **kwargs)
            
            # キャッシュに保存
            cache.put(key, result)
            
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


class MemoryManager:
    """メモリ管理システム"""
    
    def __init__(self, memory_limit_mb: int = None):
        self.memory_limit_mb = memory_limit_mb or (psutil.virtual_memory().total // 1024 // 1024 // 2)
        self.memory_usage_history = deque(maxlen=100)
        self.gc_threshold_mb = self.memory_limit_mb * 0.8
        self.emergency_threshold_mb = self.memory_limit_mb * 0.9
        
    def get_current_memory_usage(self) -> float:
        """現在のメモリ使用量取得（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_pressure(self) -> str:
        """メモリプレッシャーチェック"""
        current_usage = self.get_current_memory_usage()
        self.memory_usage_history.append(current_usage)
        
        if current_usage > self.emergency_threshold_mb:
            return 'emergency'
        elif current_usage > self.gc_threshold_mb:
            return 'high'
        elif current_usage > self.memory_limit_mb * 0.5:
            return 'medium'
        else:
            return 'low'
    
    def optimize_memory(self, pressure_level: str = None):
        """メモリ最適化"""
        if pressure_level is None:
            pressure_level = self.check_memory_pressure()
        
        if pressure_level in ['emergency', 'high']:
            # 積極的ガベージコレクション
            collected = gc.collect()
            
            # 世代別ガベージコレクション
            for generation in range(3):
                gc.collect(generation)
            
            # 緊急時追加処理
            if pressure_level == 'emergency':
                # 大きなオブジェクトのクリーンアップ
                self._cleanup_large_objects()
        
        elif pressure_level == 'medium':
            # 軽量ガベージコレクション
            gc.collect(0)  # 世代0のみ
    
    def _cleanup_large_objects(self):
        """大きなオブジェクトのクリーンアップ"""
        # 参照追跡可能な大きなオブジェクトを特定・削除
        import gc
        
        large_objects = []
        for obj in gc.get_objects():
            if hasattr(obj, '__sizeof__'):
                size = sys.getsizeof(obj)
                if size > 1024 * 1024:  # 1MB以上
                    large_objects.append((obj, size))
        
        # 弱参照でない大きなオブジェクトの削除を試行
        for obj, size in sorted(large_objects, key=lambda x: x[1], reverse=True)[:10]:
            try:
                if isinstance(obj, (list, dict, set)):
                    if hasattr(obj, 'clear'):
                        obj.clear()
            except:
                pass


class AdaptiveTaskScheduler:
    """適応的タスクスケジューラ"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        
        # 複数の実行プール
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(self.max_workers, mp.cpu_count()))
        
        # タスクキュー
        self.high_priority_queue = queue.PriorityQueue()
        self.normal_priority_queue = queue.PriorityQueue()
        self.low_priority_queue = queue.PriorityQueue()
        
        # 実行中タスク追跡
        self.running_tasks: Dict[str, concurrent.futures.Future] = {}
        self.task_history: List[ProcessingResult] = []
        
        # 性能追跡
        self.performance_history = deque(maxlen=1000)
        
        # スケジューラ制御
        self.scheduler_active = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
    def start_scheduler(self):
        """スケジューラ開始"""
        if self.scheduler_active:
            return
        
        self.scheduler_active = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """スケジューラ停止"""
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
    
    def submit_task(self, task: ProcessingTask) -> str:
        """タスク投入"""
        # 優先度に基づくキュー選択
        if task.priority >= 5:
            self.high_priority_queue.put((task.priority, task.created_at, task))
        elif task.priority >= 3:
            self.normal_priority_queue.put((task.priority, task.created_at, task))
        else:
            self.low_priority_queue.put((task.priority, task.created_at, task))
        
        return task.task_id
    
    def _scheduler_loop(self):
        """スケジューラメインループ"""
        while self.scheduler_active:
            try:
                # タスク取得（優先度順）
                task = self._get_next_task()
                
                if task:
                    # 実行プール選択
                    executor = self._select_executor(task)
                    
                    # タスク実行
                    future = executor.submit(self._execute_task, task)
                    self.running_tasks[task.task_id] = future
                    
                    # 完了時コールバック設定
                    future.add_done_callback(
                        lambda f, tid=task.task_id: self._task_completed(tid, f)
                    )
                
                # 短時間待機
                time.sleep(0.01)
                
            except Exception as e:
                logging.error(f"スケジューラエラー: {e}")
                time.sleep(0.1)
    
    def _get_next_task(self) -> Optional[ProcessingTask]:
        """次のタスク取得"""
        # 優先度順にキューをチェック
        for task_queue in [self.high_priority_queue, self.normal_priority_queue, self.low_priority_queue]:
            try:
                _, _, task = task_queue.get_nowait()
                return task
            except queue.Empty:
                continue
        
        return None
    
    def _select_executor(self, task: ProcessingTask) -> concurrent.futures.Executor:
        """実行プール選択"""
        # CPU集約的タスクはプロセスプール、I/O集約的はスレッドプール
        function_name = getattr(task.function, '__name__', 'unknown')
        
        # ヒューリスティック判定
        if 'compute' in function_name.lower() or 'calculate' in function_name.lower():
            return self.process_executor
        else:
            return self.thread_executor
    
    def _execute_task(self, task: ProcessingTask) -> ProcessingResult:
        """タスク実行"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # タイムアウト設定
            if task.timeout_seconds:
                # シンプルタイムアウト実装
                result = task.function(*task.args, **task.kwargs)
            else:
                result = task.function(*task.args, **task.kwargs)
            
            execution_time = (time.time() - start_time) * 1000
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return ProcessingResult(
                task_id=task.task_id,
                result=result,
                success=True,
                execution_time_ms=execution_time,
                memory_usage_mb=memory_usage
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                task_id=task.task_id,
                result=None,
                success=False,
                execution_time_ms=execution_time,
                memory_usage_mb=0,
                error_message=str(e)
            )
    
    def _task_completed(self, task_id: str, future: concurrent.futures.Future):
        """タスク完了処理"""
        try:
            result = future.result()
            self.task_history.append(result)
            
            # 性能メトリクス更新
            self._update_performance_metrics()
            
        except Exception as e:
            logging.error(f"タスク完了処理エラー: {e}")
        
        # 実行中タスクから削除
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
    
    def _update_performance_metrics(self):
        """性能メトリクス更新"""
        if not self.task_history:
            return
        
        recent_tasks = self.task_history[-100:]  # 直近100タスク
        
        avg_execution_time = np.mean([t.execution_time_ms for t in recent_tasks])
        peak_memory = max([t.memory_usage_mb for t in recent_tasks])
        
        # CPUとメモリ使用率
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # スループット計算
        current_time = datetime.now()
        recent_time = current_time - timedelta(seconds=60)
        recent_completed = len([t for t in recent_tasks if t.completed_at > recent_time])
        throughput = recent_completed / 60.0
        
        metrics = PerformanceMetrics(
            timestamp=current_time,
            tasks_completed=len(recent_tasks),
            avg_execution_time_ms=avg_execution_time,
            peak_memory_mb=peak_memory,
            cpu_utilization=cpu_usage,
            cache_hit_rate=0.0,  # 後で更新
            throughput_tasks_per_sec=throughput,
            queue_size=self._get_total_queue_size()
        )
        
        self.performance_history.append(metrics)
    
    def _get_total_queue_size(self) -> int:
        """総キューサイズ取得"""
        return (self.high_priority_queue.qsize() + 
                self.normal_priority_queue.qsize() + 
                self.low_priority_queue.qsize())
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """性能サマリー取得"""
        if not self.performance_history:
            return {'error': 'データが不足しています'}
        
        recent_metrics = list(self.performance_history)[-10:]  # 直近10データ
        
        return {
            'current_queue_size': self._get_total_queue_size(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.task_history),
            'average_execution_time_ms': np.mean([m.avg_execution_time_ms for m in recent_metrics]),
            'peak_memory_mb': max([m.peak_memory_mb for m in recent_metrics]),
            'current_cpu_utilization': recent_metrics[-1].cpu_utilization if recent_metrics else 0,
            'throughput_tasks_per_sec': np.mean([m.throughput_tasks_per_sec for m in recent_metrics])
        }


class HighPerformanceEngine:
    """高性能処理エンジン"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # コンポーネント
        self.cache = IntelligentCache(
            max_size=self.config.get('cache_size', 5000),
            ttl_seconds=self.config.get('cache_ttl', 3600)
        )
        
        self.memory_manager = MemoryManager(
            memory_limit_mb=self.config.get('memory_limit_mb')
        )
        
        self.scheduler = AdaptiveTaskScheduler(
            max_workers=self.config.get('max_workers')
        )
        
        # 性能監視
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # 統計
        self.engine_stats = {
            'start_time': datetime.now(),
            'total_tasks_processed': 0,
            'total_cache_hits': 0,
            'total_memory_optimizations': 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def start_engine(self):
        """エンジン開始"""
        self.logger.info("高性能処理エンジン開始")
        
        # スケジューラ開始
        self.scheduler.start_scheduler()
        
        # 性能監視開始
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("高性能処理エンジン開始完了")
    
    def stop_engine(self):
        """エンジン停止"""
        self.logger.info("高性能処理エンジン停止中")
        
        # 監視停止
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # スケジューラ停止
        self.scheduler.stop_scheduler()
        
        self.logger.info("高性能処理エンジン停止完了")
    
    def _monitoring_loop(self):
        """性能監視ループ"""
        while self.monitoring_active:
            try:
                # メモリプレッシャーチェック
                pressure = self.memory_manager.check_memory_pressure()
                
                if pressure in ['high', 'emergency']:
                    self.logger.warning(f"高メモリプレッシャー検出: {pressure}")
                    self.memory_manager.optimize_memory(pressure)
                    self.engine_stats['total_memory_optimizations'] += 1
                
                # 統計更新
                cache_stats = self.cache.get_stats()
                self.engine_stats['total_cache_hits'] = cache_stats['hit_count']
                
                time.sleep(10)  # 10秒間隔
                
            except Exception as e:
                self.logger.error(f"性能監視エラー: {e}")
                time.sleep(30)
    
    async def submit_high_performance_task(self, 
                                         function: Callable, 
                                         *args, 
                                         priority: int = 3,
                                         use_cache: bool = True,
                                         **kwargs) -> str:
        """高性能タスク投入"""
        
        # キャッシュチェック
        if use_cache:
            cache_key = self.cache._generate_key(function.__name__, args, kwargs)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                self.logger.debug(f"キャッシュヒット: {function.__name__}")
                return cached_result
        
        # タスク作成
        task_id = f"task_{int(time.time() * 1000)}_{hash((function.__name__, args, tuple(kwargs.items())))}"
        
        task = ProcessingTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        # スケジューラに投入
        submitted_id = self.scheduler.submit_task(task)
        self.engine_stats['total_tasks_processed'] += 1
        
        return submitted_id
    
    @cached(ttl_seconds=1800)
    def process_dataframe_optimized(self, df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """データフレーム最適化処理"""
        result = df.copy()
        
        for operation in operations:
            if operation == 'technical_indicators':
                # 基本的なテクニカル指標
                result['sma_5'] = result['close'].rolling(5).mean()
                result['sma_20'] = result['close'].rolling(20).mean()
                result['rsi'] = self._calculate_rsi(result['close'])
                
            elif operation == 'volume_analysis':
                # ボリューム分析
                result['volume_ma'] = result['volume'].rolling(20).mean()
                result['volume_ratio'] = result['volume'] / result['volume_ma']
                
            elif operation == 'price_patterns':
                # 価格パターン
                result['higher_high'] = (result['high'] > result['high'].shift(1)).astype(int)
                result['lower_low'] = (result['low'] < result['low'].shift(1)).astype(int)
        
        return result.fillna(0)
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算（最適化版）"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def batch_process_data(self, datasets: List[pd.DataFrame], 
                          operations: List[str],
                          batch_size: int = 10) -> List[pd.DataFrame]:
        """バッチデータ処理"""
        results = []
        
        for i in range(0, len(datasets), batch_size):
            batch = datasets[i:i+batch_size]
            
            # 並列処理
            futures = []
            for df in batch:
                task_id = asyncio.create_task(
                    self.submit_high_performance_task(
                        self.process_dataframe_optimized,
                        df,
                        operations,
                        priority=4
                    )
                )
                futures.append(task_id)
            
            # 結果収集
            batch_results = []
            for future in futures:
                try:
                    result = asyncio.run(future)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"バッチ処理エラー: {e}")
                    batch_results.append(pd.DataFrame())
            
            results.extend(batch_results)
        
        return results
    
    def get_engine_status(self) -> Dict[str, Any]:
        """エンジン状態取得"""
        # 稼働時間
        uptime = (datetime.now() - self.engine_stats['start_time']).seconds
        
        # スケジューラ統計
        scheduler_stats = self.scheduler.get_performance_summary()
        
        # キャッシュ統計
        cache_stats = self.cache.get_stats()
        
        # メモリ使用量
        current_memory = self.memory_manager.get_current_memory_usage()
        memory_pressure = self.memory_manager.check_memory_pressure()
        
        return {
            'uptime_seconds': uptime,
            'engine_active': self.monitoring_active,
            'total_tasks_processed': self.engine_stats['total_tasks_processed'],
            'memory_usage_mb': current_memory,
            'memory_pressure': memory_pressure,
            'cache_stats': cache_stats,
            'scheduler_stats': scheduler_stats,
            'memory_optimizations': self.engine_stats['total_memory_optimizations']
        }


async def demo_high_performance_engine():
    """高性能エンジンデモ"""
    print("=== 高速処理エンジン デモ ===")
    
    # エンジン初期化
    engine = HighPerformanceEngine(config={
        'cache_size': 1000,
        'cache_ttl': 1800,
        'max_workers': 8
    })
    
    try:
        # エンジン開始
        await engine.start_engine()
        print("高速処理エンジン開始")
        
        # テストデータ作成
        print("\n1. テストデータ作成中...")
        test_datasets = []
        
        for i in range(5):
            np.random.seed(42 + i)
            size = 1000
            prices = np.random.uniform(100, 200, size)
            volumes = np.random.randint(1000, 10000, size)
            
            df = pd.DataFrame({
                'close': prices,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'volume': volumes
            })
            test_datasets.append(df)
        
        # 高性能処理テスト
        print("2. 高性能処理テスト実行中...")
        
        start_time = time.time()
        
        # 並列バッチ処理
        processed_data = engine.batch_process_data(
            test_datasets,
            ['technical_indicators', 'volume_analysis', 'price_patterns'],
            batch_size=3
        )
        
        processing_time = time.time() - start_time
        
        print(f"処理時間: {processing_time:.2f}秒")
        print(f"処理データセット数: {len(processed_data)}")
        
        if processed_data and len(processed_data[0]) > 0:
            print(f"出力特徴量数: {len(processed_data[0].columns)}")
        
        # エンジン状態確認
        print("\n3. エンジン状態確認...")
        status = engine.get_engine_status()
        
        print(f"=== エンジン状態 ===")
        print(f"稼働時間: {status['uptime_seconds']}秒")
        print(f"処理タスク数: {status['total_tasks_processed']}")
        print(f"メモリ使用量: {status['memory_usage_mb']:.1f}MB")
        print(f"メモリプレッシャー: {status['memory_pressure']}")
        
        cache_stats = status['cache_stats']
        print(f"キャッシュヒット率: {cache_stats['hit_rate']:.3f}")
        print(f"キャッシュサイズ: {cache_stats['cache_size']}/{cache_stats['max_size']}")
        
        if 'error' not in status['scheduler_stats']:
            sched_stats = status['scheduler_stats']
            print(f"実行中タスク数: {sched_stats['running_tasks']}")
            print(f"スループット: {sched_stats['throughput_tasks_per_sec']:.1f} tasks/sec")
        
        print(f"メモリ最適化回数: {status['memory_optimizations']}")
        
        # 性能テスト
        print("\n4. 性能ベンチマーク...")
        
        # 単一処理時間
        single_start = time.time()
        single_result = engine.process_dataframe_optimized(
            test_datasets[0], 
            ['technical_indicators', 'volume_analysis']
        )
        single_time = time.time() - single_start
        
        print(f"単一処理時間: {single_time:.3f}秒")
        print(f"処理速度比: {single_time / (processing_time / len(test_datasets)):.1f}x")
        
        print(f"✅ 高速処理エンジン完了")
        
        return status
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    finally:
        engine.stop_engine()


if __name__ == "__main__":
    asyncio.run(demo_high_performance_engine())