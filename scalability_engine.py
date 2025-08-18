#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalability Engine - システムスケーラビリティ強化
Issue #937対応: 分散処理 + 負荷分散 + クラスター管理
"""

import asyncio
import json
import time
import threading
import multiprocessing
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import hashlib
import socket

# Redis サポート（オプショナル）
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None

# 統合モジュール
try:
    from advanced_ai_engine import advanced_ai_engine, MarketSignal
    HAS_AI_ENGINE = True
except ImportError:
    HAS_AI_ENGINE = False

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False

try:
    from data_persistence import data_persistence
    HAS_DATA_PERSISTENCE = True
except ImportError:
    HAS_DATA_PERSISTENCE = False


@dataclass
class WorkerNode:
    """ワーカーノード情報"""
    node_id: str
    host: str
    port: int
    status: str  # 'active', 'idle', 'busy', 'error'
    last_heartbeat: datetime
    cpu_usage: float
    memory_usage: float
    active_tasks: int
    total_processed: int
    worker_type: str  # 'analysis', 'streaming', 'data'


@dataclass
class Task:
    """タスク情報"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int
    created_at: datetime
    assigned_to: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class LoadBalancer:
    """負荷分散器"""
    
    def __init__(self):
        self.workers: Dict[str, WorkerNode] = {}
        self.task_queue = asyncio.Queue()
        self.completed_tasks: Dict[str, Task] = {}
        self.balancing_strategy = 'round_robin'  # 'round_robin', 'least_connections', 'cpu_based'
        self.worker_rotation = 0
    
    def register_worker(self, worker: WorkerNode):
        """ワーカー登録"""
        self.workers[worker.node_id] = worker
        logging.info(f"Worker registered: {worker.node_id} ({worker.host}:{worker.port})")
    
    def unregister_worker(self, node_id: str):
        """ワーカー登録解除"""
        if node_id in self.workers:
            del self.workers[node_id]
            logging.info(f"Worker unregistered: {node_id}")
    
    def update_worker_status(self, node_id: str, status_update: Dict[str, Any]):
        """ワーカーステータス更新"""
        if node_id in self.workers:
            worker = self.workers[node_id]
            worker.last_heartbeat = datetime.now()
            worker.cpu_usage = status_update.get('cpu_usage', worker.cpu_usage)
            worker.memory_usage = status_update.get('memory_usage', worker.memory_usage)
            worker.active_tasks = status_update.get('active_tasks', worker.active_tasks)
            worker.status = status_update.get('status', worker.status)
    
    def select_worker(self, task_type: str = None) -> Optional[WorkerNode]:
        """ワーカー選択"""
        available_workers = [
            worker for worker in self.workers.values()
            if worker.status in ['active', 'idle'] and
            (not task_type or worker.worker_type == task_type)
        ]
        
        if not available_workers:
            return None
        
        if self.balancing_strategy == 'round_robin':
            worker = available_workers[self.worker_rotation % len(available_workers)]
            self.worker_rotation += 1
            return worker
        
        elif self.balancing_strategy == 'least_connections':
            return min(available_workers, key=lambda w: w.active_tasks)
        
        elif self.balancing_strategy == 'cpu_based':
            return min(available_workers, key=lambda w: w.cpu_usage)
        
        return available_workers[0]
    
    async def submit_task(self, task: Task) -> str:
        """タスク投入"""
        await self.task_queue.put(task)
        return task.task_id
    
    async def get_task_result(self, task_id: str, timeout: float = 30.0) -> Optional[Task]:
        """タスク結果取得"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            await asyncio.sleep(0.1)
        
        return None
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """負荷統計取得"""
        if not self.workers:
            return {
                'total_workers': 0,
                'active_workers': 0,
                'total_tasks_queued': self.task_queue.qsize(),
                'average_cpu_usage': 0.0,
                'average_memory_usage': 0.0
            }
        
        active_workers = [w for w in self.workers.values() if w.status == 'active']
        
        return {
            'total_workers': len(self.workers),
            'active_workers': len(active_workers),
            'total_tasks_queued': self.task_queue.qsize(),
            'average_cpu_usage': sum(w.cpu_usage for w in active_workers) / len(active_workers) if active_workers else 0.0,
            'average_memory_usage': sum(w.memory_usage for w in active_workers) / len(active_workers) if active_workers else 0.0,
            'workers_by_type': self._get_workers_by_type(),
            'task_completion_rate': len(self.completed_tasks) / max(1, len(self.completed_tasks) + self.task_queue.qsize())
        }
    
    def _get_workers_by_type(self) -> Dict[str, int]:
        """タイプ別ワーカー数"""
        type_counts = defaultdict(int)
        for worker in self.workers.values():
            type_counts[worker.worker_type] += 1
        return dict(type_counts)


class CacheManager:
    """分散キャッシュマネージャー"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = None
        self.local_cache: Dict[str, Any] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        
        if HAS_REDIS:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, 
                    port=redis_port, 
                    decode_responses=True,
                    socket_connect_timeout=1
                )
                # 接続テスト
                self.redis_client.ping()
                logging.info("Redis cache connected")
            except Exception as e:
                logging.warning(f"Redis connection failed, using local cache: {e}")
                self.redis_client = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """キー取得"""
        try:
            # Redis から取得
            if self.redis_client:
                value = self.redis_client.get(key)
                if value is not None:
                    self.cache_stats['hits'] += 1
                    return json.loads(value)
            
            # ローカルキャッシュから取得
            if key in self.local_cache:
                self.cache_stats['hits'] += 1
                return self.local_cache[key]
            
            self.cache_stats['misses'] += 1
            return default
            
        except Exception as e:
            logging.error(f"Cache get error: {e}")
            return default
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """キー設定"""
        try:
            # Redis に設定
            if self.redis_client:
                json_value = json.dumps(value, default=str)
                if expire:
                    self.redis_client.setex(key, expire, json_value)
                else:
                    self.redis_client.set(key, json_value)
                    
                self.cache_stats['sets'] += 1
                return True
            
            # ローカルキャッシュに設定
            self.local_cache[key] = value
            self.cache_stats['sets'] += 1
            
            # ローカルキャッシュサイズ制限
            if len(self.local_cache) > 1000:
                # 古いエントリを削除
                keys_to_remove = list(self.local_cache.keys())[:100]
                for k in keys_to_remove:
                    del self.local_cache[k]
            
            return True
            
        except Exception as e:
            logging.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """キー削除"""
        try:
            deleted = False
            
            # Redis から削除
            if self.redis_client:
                deleted = bool(self.redis_client.delete(key))
            
            # ローカルキャッシュから削除
            if key in self.local_cache:
                del self.local_cache[key]
                deleted = True
            
            if deleted:
                self.cache_stats['deletes'] += 1
            
            return deleted
            
        except Exception as e:
            logging.error(f"Cache delete error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            **self.cache_stats,
            'hit_rate_percent': hit_rate,
            'redis_available': self.redis_client is not None,
            'local_cache_size': len(self.local_cache)
        }


class ProcessPool:
    """プロセスプール管理"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        
        self.active_processes = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """CPU集約タスク投入"""
        self.active_processes += 1
        future = self.process_executor.submit(func, *args, **kwargs)
        
        def on_complete(f):
            self.active_processes -= 1
            if f.exception():
                self.failed_tasks += 1
            else:
                self.completed_tasks += 1
        
        future.add_done_callback(on_complete)
        return future
    
    def submit_io_task(self, func: Callable, *args, **kwargs):
        """IO集約タスク投入"""
        future = self.thread_executor.submit(func, *args, **kwargs)
        
        def on_complete(f):
            if f.exception():
                self.failed_tasks += 1
            else:
                self.completed_tasks += 1
        
        future.add_done_callback(on_complete)
        return future
    
    async def batch_process(self, tasks: List[Callable], task_type: str = 'cpu') -> List[Any]:
        """バッチ処理"""
        if task_type == 'cpu':
            executor = self.process_executor
        else:
            executor = self.thread_executor
        
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(executor, task) for task in tasks]
        return await asyncio.gather(*futures)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """プール統計"""
        return {
            'max_workers': self.max_workers,
            'active_processes': self.active_processes,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': (self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks)) * 100
        }
    
    def shutdown(self, wait: bool = True):
        """プール停止"""
        self.process_executor.shutdown(wait=wait)
        self.thread_executor.shutdown(wait=wait)


class ClusterManager:
    """クラスター管理"""
    
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or self._generate_node_id()
        self.cluster_nodes: Dict[str, Dict[str, Any]] = {}
        self.heartbeat_interval = 10.0  # 10秒間隔
        self.node_timeout = 30.0  # 30秒でタイムアウト
        
        self.is_leader = False
        self.leader_node_id: Optional[str] = None
        
        self._heartbeat_task = None
        self._cleanup_task = None
        
    def _generate_node_id(self) -> str:
        """ノードID生成"""
        hostname = socket.gethostname()
        timestamp = str(int(time.time()))
        return hashlib.md5(f"{hostname}_{timestamp}".encode()).hexdigest()[:8]
    
    async def join_cluster(self, bootstrap_nodes: List[str] = None):
        """クラスター参加"""
        self.cluster_nodes[self.node_id] = {
            'hostname': socket.gethostname(),
            'last_seen': datetime.now(),
            'status': 'active',
            'services': ['analysis', 'streaming']
        }
        
        # リーダー選出
        await self._elect_leader()
        
        # ハートビート開始
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logging.info(f"Node {self.node_id} joined cluster (Leader: {self.is_leader})")
    
    async def leave_cluster(self):
        """クラスター離脱"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        if self.node_id in self.cluster_nodes:
            del self.cluster_nodes[self.node_id]
        
        logging.info(f"Node {self.node_id} left cluster")
    
    async def _elect_leader(self):
        """リーダー選出"""
        if not self.cluster_nodes:
            self.is_leader = True
            self.leader_node_id = self.node_id
            return
        
        # 最古のノードをリーダーに選出
        oldest_node = min(
            self.cluster_nodes.items(),
            key=lambda x: x[1]['last_seen']
        )
        
        self.leader_node_id = oldest_node[0]
        self.is_leader = (self.node_id == self.leader_node_id)
    
    async def _heartbeat_loop(self):
        """ハートビートループ"""
        while True:
            try:
                # 自分のハートビート更新
                if self.node_id in self.cluster_nodes:
                    self.cluster_nodes[self.node_id]['last_seen'] = datetime.now()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Heartbeat error: {e}")
    
    async def _cleanup_loop(self):
        """クリーンアップループ"""
        while True:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.node_timeout)
                
                # タイムアウトしたノードを削除
                timeout_nodes = [
                    node_id for node_id, info in self.cluster_nodes.items()
                    if info['last_seen'] < timeout_threshold
                ]
                
                for node_id in timeout_nodes:
                    del self.cluster_nodes[node_id]
                    logging.warning(f"Node {node_id} removed due to timeout")
                
                # リーダーが削除された場合は再選出
                if self.leader_node_id not in self.cluster_nodes:
                    await self._elect_leader()
                    logging.info(f"New leader elected: {self.leader_node_id}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Cleanup error: {e}")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """クラスター情報取得"""
        return {
            'node_id': self.node_id,
            'is_leader': self.is_leader,
            'leader_node_id': self.leader_node_id,
            'cluster_size': len(self.cluster_nodes),
            'cluster_nodes': {
                node_id: {
                    **info,
                    'last_seen': info['last_seen'].isoformat()
                }
                for node_id, info in self.cluster_nodes.items()
            }
        }


class ScalabilityEngine:
    """スケーラビリティエンジン統合管理"""
    
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.cache_manager = CacheManager()
        self.process_pool = ProcessPool()
        self.cluster_manager = ClusterManager()
        
        self.running = False
        self.start_time = datetime.now()
        
        # 統計情報
        self.total_requests = 0
        self.avg_response_time = 0.0
        self.error_count = 0
    
    async def start(self):
        """エンジン開始"""
        self.running = True
        await self.cluster_manager.join_cluster()
        
        # タスク処理ループ開始
        asyncio.create_task(self._task_processing_loop())
        
        logging.info("Scalability Engine started")
    
    async def stop(self):
        """エンジン停止"""
        self.running = False
        await self.cluster_manager.leave_cluster()
        self.process_pool.shutdown()
        
        logging.info("Scalability Engine stopped")
    
    async def _task_processing_loop(self):
        """タスク処理ループ"""
        while self.running:
            try:
                # タスクキューからタスクを取得
                task = await asyncio.wait_for(
                    self.load_balancer.task_queue.get(), 
                    timeout=1.0
                )
                
                # ワーカー選択
                worker = self.load_balancer.select_worker(task.task_type)
                
                if worker:
                    # タスク実行
                    await self._execute_task(task, worker)
                else:
                    # ワーカーが利用できない場合は再キュー
                    await asyncio.sleep(0.1)
                    await self.load_balancer.task_queue.put(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Task processing error: {e}")
                self.error_count += 1
    
    async def _execute_task(self, task: Task, worker: WorkerNode):
        """タスク実行"""
        start_time = time.time()
        
        try:
            task.assigned_to = worker.node_id
            task.started_at = datetime.now()
            
            # キャッシュチェック
            cache_key = f"task:{task.task_type}:{hash(str(task.payload))}"
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                task.result = cached_result
                task.completed_at = datetime.now()
            else:
                # 実際のタスク処理
                if task.task_type == 'analysis':
                    result = await self._handle_analysis_task(task)
                elif task.task_type == 'data_processing':
                    result = await self._handle_data_processing_task(task)
                else:
                    result = {'error': f'Unknown task type: {task.task_type}'}
                
                task.result = result
                task.completed_at = datetime.now()
                
                # 結果をキャッシュ
                self.cache_manager.set(cache_key, result, expire=300)  # 5分
            
            # 完了タスクに追加
            self.load_balancer.completed_tasks[task.task_id] = task
            
            # 統計更新
            execution_time = time.time() - start_time
            self.total_requests += 1
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + execution_time) /
                self.total_requests
            )
            
        except Exception as e:
            task.error = str(e)
            task.completed_at = datetime.now()
            self.load_balancer.completed_tasks[task.task_id] = task
            self.error_count += 1
            logging.error(f"Task execution error: {e}")
    
    async def _handle_analysis_task(self, task: Task) -> Dict[str, Any]:
        """分析タスク処理"""
        symbol = task.payload.get('symbol', '')
        
        if HAS_AI_ENGINE and symbol:
            # プロセスプールで分析実行
            future = self.process_pool.submit_cpu_task(
                self._perform_analysis, symbol
            )
            
            # 結果待ち（非同期）
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, future.result, 30)  # 30秒タイムアウト
            
            return result
        else:
            return {'error': 'AI engine not available or missing symbol'}
    
    async def _handle_data_processing_task(self, task: Task) -> Dict[str, Any]:
        """データ処理タスク処理"""
        data = task.payload.get('data', [])
        operation = task.payload.get('operation', 'process')
        
        # データ処理をプロセスプールで実行
        future = self.process_pool.submit_cpu_task(
            self._process_data, data, operation
        )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, future.result, 60)  # 60秒タイムアウト
        
        return result
    
    @staticmethod
    def _perform_analysis(symbol: str) -> Dict[str, Any]:
        """分析実行（プロセス内）"""
        try:
            # 簡易分析（実際の実装では advanced_ai_engine を使用）
            import random
            
            confidence = random.uniform(0.6, 0.95)
            recommendation = random.choice(['BUY', 'SELL', 'HOLD'])
            
            return {
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'scalable_analysis'
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def _process_data(data: List[Any], operation: str) -> Dict[str, Any]:
        """データ処理実行（プロセス内）"""
        try:
            processed_count = len(data)
            
            if operation == 'aggregate':
                result = {'sum': sum(data) if all(isinstance(x, (int, float)) for x in data) else 0}
            elif operation == 'filter':
                result = {'filtered': [x for x in data if x is not None]}
            else:
                result = {'processed_count': processed_count}
            
            return {
                'operation': operation,
                'input_count': processed_count,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def submit_analysis(self, symbol: str, priority: int = 1) -> str:
        """分析投入"""
        task = Task(
            task_id=f"analysis_{symbol}_{int(time.time())}_{hash(symbol) % 1000}",
            task_type='analysis',
            payload={'symbol': symbol},
            priority=priority,
            created_at=datetime.now()
        )
        
        return await self.load_balancer.submit_task(task)
    
    async def submit_data_processing(self, data: List[Any], operation: str = 'process') -> str:
        """データ処理投入"""
        task = Task(
            task_id=f"data_{operation}_{int(time.time())}_{hash(str(data)) % 1000}",
            task_type='data_processing',
            payload={'data': data, 'operation': operation},
            priority=2,
            created_at=datetime.now()
        )
        
        return await self.load_balancer.submit_task(task)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """システムメトリクス取得"""
        uptime = datetime.now() - self.start_time
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_requests': self.total_requests,
            'avg_response_time_ms': self.avg_response_time * 1000,
            'error_count': self.error_count,
            'error_rate': (self.error_count / max(1, self.total_requests)) * 100,
            
            # 負荷分散統計
            'load_balancer': self.load_balancer.get_load_statistics(),
            
            # キャッシュ統計
            'cache': self.cache_manager.get_stats(),
            
            # プロセスプール統計
            'process_pool': self.process_pool.get_pool_stats(),
            
            # クラスター情報
            'cluster': self.cluster_manager.get_cluster_info(),
            
            # システムリソース
            'system_resources': self._get_system_resources()
        }
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """システムリソース取得"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024
            }
        except ImportError:
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'message': 'psutil not available'
            }


# グローバルインスタンス
scalability_engine = ScalabilityEngine()


async def test_scalability_engine():
    """スケーラビリティエンジンテスト"""
    print("=== Scalability Engine Test ===")
    
    # エンジン開始
    await scalability_engine.start()
    
    # 複数の分析タスクを投入
    symbols = ['7203', '8306', '9984', '6758', '4689']
    task_ids = []
    
    print(f"Submitting {len(symbols)} analysis tasks...")
    for symbol in symbols:
        task_id = await scalability_engine.submit_analysis(symbol)
        task_ids.append(task_id)
    
    # データ処理タスクを投入
    print("Submitting data processing task...")
    data_task_id = await scalability_engine.submit_data_processing(
        [1, 2, 3, 4, 5], 'aggregate'
    )
    task_ids.append(data_task_id)
    
    # 結果待ち
    print("Waiting for results...")
    for task_id in task_ids:
        result = await scalability_engine.load_balancer.get_task_result(task_id)
        if result:
            print(f"Task {task_id}: {result.result}")
        else:
            print(f"Task {task_id}: Timeout")
    
    # システムメトリクス表示
    print("\n=== System Metrics ===")
    metrics = scalability_engine.get_system_metrics()
    print(json.dumps(metrics, indent=2, default=str, ensure_ascii=False))
    
    # エンジン停止
    await scalability_engine.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_scalability_engine())