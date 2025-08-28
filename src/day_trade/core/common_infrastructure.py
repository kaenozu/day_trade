#!/usr/bin/env python3
"""
共通基盤インフラストラクチャクラス

全システムに共通する基盤機能を提供します。
"""

import asyncio
import json
import logging
import sqlite3
import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Generic, TypeVar
import psutil

from .base import BaseComponent, BaseConfig, HealthStatus, SystemStatus
from .unified_system_error import UnifiedSystemError, ErrorSeverity

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class ProcessingMode(Enum):
    """処理モード"""
    SYNC = "synchronous"
    ASYNC = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"


class Priority(Enum):
    """優先度レベル"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class SystemMetrics:
    """システムメトリクス"""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_threads: int
    active_processes: int
    queue_size: int = 0
    processing_rate: float = 0.0
    error_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0


@dataclass
class TaskConfig:
    """タスク設定"""
    task_id: str
    priority: Priority = Priority.NORMAL
    timeout_seconds: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    max_memory_mb: float = 100.0
    processing_mode: ProcessingMode = ProcessingMode.ASYNC
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """タスク実行結果"""
    task_id: str
    success: bool
    start_time: float
    end_time: float
    processing_time_ms: float
    result_data: Any = None
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseStorage(ABC, Generic[K, V]):
    """ストレージ基盤クラス"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._lock = threading.RLock()
        self._is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """ストレージ初期化"""
        pass
    
    @abstractmethod
    async def get(self, key: K) -> Tuple[bool, Optional[V]]:
        """データ取得"""
        pass
    
    @abstractmethod
    async def set(self, key: K, value: V, ttl: Optional[float] = None) -> bool:
        """データ設定"""
        pass
    
    @abstractmethod
    async def delete(self, key: K) -> bool:
        """データ削除"""
        pass
    
    @abstractmethod
    async def exists(self, key: K) -> bool:
        """存在確認"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """全削除"""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """健全性チェック"""
        pass


class InMemoryStorage(BaseStorage[str, Any]):
    """インメモリストレージ実装"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._access_count: Dict[str, int] = {}
        self._last_access: Dict[str, float] = {}
        self.max_size = config.get('max_size', 1000)
        self._cleanup_interval = config.get('cleanup_interval', 60)
        self._cleanup_task = None
    
    async def initialize(self) -> bool:
        """初期化"""
        try:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired())
            self._is_initialized = True
            return True
        except Exception as e:
            logging.error(f"InMemoryStorage initialization failed: {e}")
            return False
    
    async def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        """データ取得"""
        with self._lock:
            if key not in self._data:
                return False, None
            
            # TTL チェック
            if key in self._expiry and time.time() > self._expiry[key]:
                self._remove_key(key)
                return False, None
            
            # アクセス統計更新
            self._access_count[key] = self._access_count.get(key, 0) + 1
            self._last_access[key] = time.time()
            
            return True, self._data[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """データ設定"""
        with self._lock:
            try:
                # サイズ制限チェック
                if len(self._data) >= self.max_size and key not in self._data:
                    self._evict_lru()
                
                self._data[key] = value
                self._last_access[key] = time.time()
                
                if ttl is not None:
                    self._expiry[key] = time.time() + ttl
                elif key in self._expiry:
                    del self._expiry[key]
                
                return True
            except Exception as e:
                logging.error(f"InMemoryStorage set failed for key {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """データ削除"""
        with self._lock:
            if key in self._data:
                self._remove_key(key)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """存在確認"""
        found, _ = await self.get(key)
        return found
    
    async def clear(self) -> bool:
        """全削除"""
        with self._lock:
            self._data.clear()
            self._expiry.clear()
            self._access_count.clear()
            self._last_access.clear()
            return True
    
    async def health_check(self) -> HealthStatus:
        """健全性チェック"""
        try:
            with self._lock:
                size = len(self._data)
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                
                if size >= self.max_size * 0.9:
                    return HealthStatus.DEGRADED
                if memory_usage > self.config.get('max_memory_mb', 100):
                    return HealthStatus.DEGRADED
                
                return HealthStatus.HEALTHY
        except Exception:
            return HealthStatus.UNHEALTHY
    
    def _remove_key(self, key: str):
        """キー削除"""
        self._data.pop(key, None)
        self._expiry.pop(key, None)
        self._access_count.pop(key, None)
        self._last_access.pop(key, None)
    
    def _evict_lru(self):
        """LRU削除"""
        if not self._last_access:
            return
        
        lru_key = min(self._last_access, key=self._last_access.get)
        self._remove_key(lru_key)
    
    async def _cleanup_expired(self):
        """期限切れデータクリーンアップ"""
        while self._is_initialized:
            try:
                with self._lock:
                    current_time = time.time()
                    expired_keys = [
                        key for key, expiry in self._expiry.items()
                        if current_time > expiry
                    ]
                    
                    for key in expired_keys:
                        self._remove_key(key)
                
                await asyncio.sleep(self._cleanup_interval)
            except Exception as e:
                logging.error(f"Cleanup task error: {e}")
                await asyncio.sleep(self._cleanup_interval)


class BaseTaskProcessor(BaseComponent):
    """タスク処理基盤クラス"""
    
    def __init__(self, name: str, config: BaseConfig):
        super().__init__(name, config)
        self.task_queue = asyncio.Queue(maxsize=config.queue_size)
        self.result_storage = InMemoryStorage(f"{name}_results", {
            'max_size': config.max_results,
            'cleanup_interval': 300
        })
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=config.max_processes)
        self._metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage_percent=0.0,
            memory_usage_percent=0.0,
            disk_usage_percent=0.0,
            network_bytes_sent=0,
            network_bytes_recv=0,
            active_threads=0,
            active_processes=0
        )
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._metrics_task = None
    
    async def start(self) -> bool:
        """コンポーネント開始"""
        try:
            await self.result_storage.initialize()
            self._metrics_task = asyncio.create_task(self._collect_metrics())
            self.status = SystemStatus.RUNNING
            await self._emit_event('started', {'component': self.name})
            return True
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="start",
                component=self.name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    async def stop(self) -> bool:
        """コンポーネント停止"""
        try:
            self.status = SystemStatus.STOPPING
            
            # 処理中タスク完了待機
            if self._processing_tasks:
                await asyncio.gather(*self._processing_tasks.values(), return_exceptions=True)
            
            # リソースクリーンアップ
            self.executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            if self._metrics_task:
                self._metrics_task.cancel()
            
            self.status = SystemStatus.STOPPED
            await self._emit_event('stopped', {'component': self.name})
            return True
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="stop",
                component=self.name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    async def submit_task(self, task_config: TaskConfig, task_data: Any) -> str:
        """タスク投入"""
        try:
            task_item = {
                'config': task_config,
                'data': task_data,
                'submitted_at': time.time()
            }
            
            await self.task_queue.put(task_item)
            
            # 高優先度タスクは即座に処理開始
            if task_config.priority.value >= Priority.HIGH.value:
                processing_task = asyncio.create_task(
                    self._process_single_task(task_item)
                )
                self._processing_tasks[task_config.task_id] = processing_task
            
            return task_config.task_id
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="submit_task",
                task_id=task_config.task_id,
                severity=ErrorSeverity.MEDIUM
            ))
            raise
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """タスク結果取得"""
        found, result = await self.result_storage.get(task_id)
        return result if found else None
    
    @abstractmethod
    async def process_task(self, task_config: TaskConfig, task_data: Any) -> Any:
        """タスク処理（サブクラスで実装）"""
        pass
    
    async def _process_single_task(self, task_item: Dict[str, Any]) -> TaskResult:
        """単一タスク処理"""
        config = task_item['config']
        data = task_item['data']
        start_time = time.time()
        
        try:
            # タイムアウト設定
            result_data = await asyncio.wait_for(
                self.process_task(config, data),
                timeout=config.timeout_seconds
            )
            
            end_time = time.time()
            result = TaskResult(
                task_id=config.task_id,
                success=True,
                start_time=start_time,
                end_time=end_time,
                processing_time_ms=(end_time - start_time) * 1000,
                result_data=result_data
            )
            
        except asyncio.TimeoutError:
            end_time = time.time()
            result = TaskResult(
                task_id=config.task_id,
                success=False,
                start_time=start_time,
                end_time=end_time,
                processing_time_ms=(end_time - start_time) * 1000,
                error_message="Task timeout"
            )
            
        except Exception as e:
            end_time = time.time()
            result = TaskResult(
                task_id=config.task_id,
                success=False,
                start_time=start_time,
                end_time=end_time,
                processing_time_ms=(end_time - start_time) * 1000,
                error_message=str(e)
            )
            
            await self._handle_error(e, ErrorContext(
                operation="process_task",
                task_id=config.task_id,
                severity=ErrorSeverity.MEDIUM
            ))
        
        # 結果保存
        await self.result_storage.set(config.task_id, result, ttl=3600)
        
        # タスク完了イベント
        await self._emit_event('task_completed', {
            'task_id': config.task_id,
            'success': result.success,
            'processing_time_ms': result.processing_time_ms
        })
        
        return result
    
    async def _collect_metrics(self):
        """メトリクス収集"""
        while self.status == SystemStatus.RUNNING:
            try:
                # システムメトリクス取得
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                self._metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_usage_percent=cpu_percent,
                    memory_usage_percent=memory.percent,
                    disk_usage_percent=(disk.used / disk.total) * 100,
                    network_bytes_sent=network.bytes_sent,
                    network_bytes_recv=network.bytes_recv,
                    active_threads=threading.active_count(),
                    active_processes=len(psutil.pids()),
                    queue_size=self.task_queue.qsize(),
                    processing_rate=len(self._processing_tasks)
                )
                
                await asyncio.sleep(10)
            except Exception as e:
                await self._handle_error(e, ErrorContext(
                    operation="collect_metrics",
                    component=self.name,
                    severity=ErrorSeverity.LOW
                ))
                await asyncio.sleep(10)
    
    async def health_check(self) -> HealthStatus:
        """健全性チェック"""
        try:
            # 基本チェック
            if self.status != SystemStatus.RUNNING:
                return HealthStatus.UNHEALTHY
            
            # リソース使用量チェック
            if self._metrics.cpu_usage_percent > 90:
                return HealthStatus.DEGRADED
            if self._metrics.memory_usage_percent > 90:
                return HealthStatus.DEGRADED
            if self.task_queue.qsize() > self.config.queue_size * 0.8:
                return HealthStatus.DEGRADED
            
            # ストレージ健全性チェック
            storage_health = await self.result_storage.health_check()
            if storage_health != HealthStatus.HEALTHY:
                return storage_health
            
            return HealthStatus.HEALTHY
            
        except Exception:
            return HealthStatus.UNHEALTHY
    
    def get_metrics(self) -> SystemMetrics:
        """メトリクス取得"""
        return self._metrics


class BaseDataProcessor(BaseTaskProcessor):
    """データ処理基盤クラス"""
    
    def __init__(self, name: str, config: BaseConfig):
        super().__init__(name, config)
        self.batch_size = config.batch_size
        self.processing_timeout = config.processing_timeout
        self._data_buffer = deque(maxlen=config.buffer_size)
        self._buffer_lock = threading.Lock()
    
    async def process_batch(self, data_batch: List[Any]) -> List[Any]:
        """バッチ処理（サブクラスで実装）"""
        raise NotImplementedError("Subclass must implement process_batch")
    
    async def process_stream_item(self, data_item: Any) -> Any:
        """ストリーム項目処理（サブクラスで実装）"""
        raise NotImplementedError("Subclass must implement process_stream_item")
    
    async def add_to_buffer(self, data: Any) -> bool:
        """バッファ追加"""
        try:
            with self._buffer_lock:
                self._data_buffer.append({
                    'data': data,
                    'timestamp': time.time()
                })
            return True
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="add_to_buffer",
                component=self.name,
                severity=ErrorSeverity.LOW
            ))
            return False
    
    async def process_buffer(self) -> Optional[List[Any]]:
        """バッファ処理"""
        try:
            with self._buffer_lock:
                if len(self._data_buffer) < self.batch_size:
                    return None
                
                batch = list(self._data_buffer)
                self._data_buffer.clear()
            
            # バッチ処理実行
            batch_data = [item['data'] for item in batch]
            results = await self.process_batch(batch_data)
            
            return results
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="process_buffer",
                component=self.name,
                severity=ErrorSeverity.MEDIUM
            ))
            return None


# エクスポート関数
def create_storage(storage_type: str, name: str, config: Dict[str, Any]) -> BaseStorage:
    """ストレージファクトリ"""
    if storage_type == "memory":
        return InMemoryStorage(name, config)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


def create_task_processor(processor_type: str, name: str, config: BaseConfig) -> BaseTaskProcessor:
    """タスクプロセッサファクトリ"""
    # 具体的な実装は各システムで提供
    raise NotImplementedError(f"Processor type {processor_type} not implemented")


def create_data_processor(processor_type: str, name: str, config: BaseConfig) -> BaseDataProcessor:
    """データプロセッサファクトリ"""
    # 具体的な実装は各システムで提供
    raise NotImplementedError(f"Data processor type {processor_type} not implemented")