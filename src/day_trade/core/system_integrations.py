#!/usr/bin/env python3
"""
システム統合基盤

各高度システム間の統合パターンと共通機能を提供します。
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
import threading
from pathlib import Path

from .base import BaseComponent, BaseConfig, HealthStatus, SystemStatus
from .common_infrastructure import (
    BaseTaskProcessor, BaseDataProcessor, BaseStorage, 
    TaskConfig, TaskResult, SystemMetrics, Priority, ProcessingMode,
    InMemoryStorage
)
from .unified_system_error import UnifiedSystemError, ErrorSeverity

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """統合設定"""
    system_name: str
    enabled: bool = True
    priority: Priority = Priority.NORMAL
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    health_check_interval: float = 60.0
    metrics_collection: bool = True
    dependencies: List[str] = field(default_factory=list)
    integration_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """システム状態"""
    system_name: str
    status: SystemStatus
    health: HealthStatus
    last_update: float
    metrics: Optional[SystemMetrics] = None
    error_count: int = 0
    warning_count: int = 0
    uptime_seconds: float = 0.0
    last_restart: Optional[float] = None
    version: str = "1.0.0"


@dataclass
class IntegrationEvent:
    """統合イベント"""
    event_id: str
    event_type: str
    source_system: str
    target_system: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.INFO
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    handled: bool = False


class SystemOrchestrator(BaseComponent):
    """システムオーケストレーター"""
    
    def __init__(self, name: str, config: BaseConfig):
        super().__init__(name, config)
        self.systems: Dict[str, BaseComponent] = {}
        self.system_configs: Dict[str, IntegrationConfig] = {}
        self.system_states: Dict[str, SystemState] = {}
        self.event_bus = EventBus("system_events")
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self._monitoring_task = None
        self._coordination_lock = asyncio.Lock()
    
    async def start(self) -> bool:
        """オーケストレーター開始"""
        try:
            await self.event_bus.start()
            self._monitoring_task = asyncio.create_task(self._monitor_systems())
            
            # 依存関係に基づく起動順序計算
            self._calculate_startup_order()
            
            # システム順次起動
            for system_name in self.startup_order:
                if system_name in self.systems:
                    await self._start_system(system_name)
            
            self.status = SystemStatus.RUNNING
            await self._emit_event('orchestrator_started', {
                'managed_systems': list(self.systems.keys())
            })
            return True
            
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="orchestrator_start",
                component=self.name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    async def stop(self) -> bool:
        """オーケストレーター停止"""
        try:
            self.status = SystemStatus.STOPPING
            
            # システム逆順停止
            for system_name in reversed(self.shutdown_order):
                if system_name in self.systems:
                    await self._stop_system(system_name)
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
            
            await self.event_bus.stop()
            self.status = SystemStatus.STOPPED
            return True
            
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="orchestrator_stop",
                component=self.name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    def register_system(self, system: BaseComponent, config: IntegrationConfig) -> bool:
        """システム登録"""
        try:
            self.systems[system.name] = system
            self.system_configs[system.name] = config
            self.system_states[system.name] = SystemState(
                system_name=system.name,
                status=SystemStatus.STOPPED,
                health=HealthStatus.UNKNOWN,
                last_update=time.time()
            )
            
            # 依存関係グラフ更新
            self.dependency_graph[system.name] = set(config.dependencies)
            
            logger.info(f"System registered: {system.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register system {system.name}: {e}")
            return False
    
    async def restart_system(self, system_name: str) -> bool:
        """システム再起動"""
        async with self._coordination_lock:
            try:
                if system_name not in self.systems:
                    return False
                
                logger.info(f"Restarting system: {system_name}")
                
                # 依存システム停止
                dependents = self._get_dependent_systems(system_name)
                for dep_system in dependents:
                    await self._stop_system(dep_system)
                
                # 対象システム再起動
                await self._stop_system(system_name)
                await asyncio.sleep(1)  # クールダウン
                await self._start_system(system_name)
                
                # 依存システム再開
                for dep_system in dependents:
                    await self._start_system(dep_system)
                
                self.system_states[system_name].last_restart = time.time()
                return True
                
            except Exception as e:
                await self._handle_error(e, ErrorContext(
                    operation="restart_system",
                    system=system_name,
                    severity=ErrorSeverity.HIGH
                ))
                return False
    
    async def _start_system(self, system_name: str) -> bool:
        """単一システム起動"""
        try:
            system = self.systems[system_name]
            config = self.system_configs[system_name]
            
            if not config.enabled:
                logger.info(f"System {system_name} is disabled, skipping")
                return True
            
            # 依存関係チェック
            for dependency in config.dependencies:
                if dependency in self.system_states:
                    dep_state = self.system_states[dependency]
                    if dep_state.status != SystemStatus.RUNNING:
                        logger.warning(f"Dependency {dependency} not running for {system_name}")
                        return False
            
            logger.info(f"Starting system: {system_name}")
            success = await system.start()
            
            if success:
                self.system_states[system_name].status = SystemStatus.RUNNING
                self.system_states[system_name].last_update = time.time()
                
                await self.event_bus.publish(IntegrationEvent(
                    event_id=f"start_{system_name}_{int(time.time())}",
                    event_type="system_started",
                    source_system=system_name,
                    message=f"System {system_name} started successfully"
                ))
            
            return success
            
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="start_system",
                system=system_name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    async def _stop_system(self, system_name: str) -> bool:
        """単一システム停止"""
        try:
            system = self.systems[system_name]
            
            logger.info(f"Stopping system: {system_name}")
            success = await system.stop()
            
            if success:
                self.system_states[system_name].status = SystemStatus.STOPPED
                self.system_states[system_name].last_update = time.time()
                
                await self.event_bus.publish(IntegrationEvent(
                    event_id=f"stop_{system_name}_{int(time.time())}",
                    event_type="system_stopped",
                    source_system=system_name,
                    message=f"System {system_name} stopped successfully"
                ))
            
            return success
            
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="stop_system",
                system=system_name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    def _calculate_startup_order(self):
        """起動順序計算（トポロジカルソート）"""
        # 依存関係グラフからトポロジカルソート
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node: str):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            
            for dependency in self.dependency_graph.get(node, set()):
                if dependency in self.dependency_graph:
                    visit(dependency)
            
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        for system_name in self.systems:
            if system_name not in visited:
                visit(system_name)
        
        self.startup_order = result
        self.shutdown_order = list(reversed(result))
        
        logger.info(f"Calculated startup order: {self.startup_order}")
    
    def _get_dependent_systems(self, system_name: str) -> List[str]:
        """依存システム取得"""
        dependents = []
        for sys_name, dependencies in self.dependency_graph.items():
            if system_name in dependencies:
                dependents.append(sys_name)
        return dependents
    
    async def _monitor_systems(self):
        """システム監視"""
        while self.status == SystemStatus.RUNNING:
            try:
                for system_name, system in self.systems.items():
                    state = self.system_states[system_name]
                    
                    # 健全性チェック
                    if hasattr(system, 'health_check'):
                        health = await system.health_check()
                        state.health = health
                        state.last_update = time.time()
                        
                        # 異常検出時の対処
                        if health == HealthStatus.UNHEALTHY:
                            state.error_count += 1
                            await self._handle_unhealthy_system(system_name)
                        elif health == HealthStatus.DEGRADED:
                            state.warning_count += 1
                            await self._handle_degraded_system(system_name)
                        else:
                            # 正常時はカウンターリセット
                            if state.error_count > 0:
                                state.error_count = max(0, state.error_count - 1)
                    
                    # メトリクス収集
                    if hasattr(system, 'get_metrics'):
                        state.metrics = system.get_metrics()
                
                await asyncio.sleep(30)  # 監視間隔
                
            except Exception as e:
                await self._handle_error(e, ErrorContext(
                    operation="monitor_systems",
                    component=self.name,
                    severity=ErrorSeverity.MEDIUM
                ))
                await asyncio.sleep(30)
    
    async def _handle_unhealthy_system(self, system_name: str):
        """異常システム対処"""
        state = self.system_states[system_name]
        config = self.system_configs[system_name]
        
        if state.error_count >= config.circuit_breaker_threshold:
            logger.error(f"System {system_name} exceeded error threshold, initiating restart")
            await self.restart_system(system_name)
    
    async def _handle_degraded_system(self, system_name: str):
        """劣化システム対処"""
        logger.warning(f"System {system_name} is in degraded state")
        
        await self.event_bus.publish(IntegrationEvent(
            event_id=f"degraded_{system_name}_{int(time.time())}",
            event_type="system_degraded",
            source_system=system_name,
            severity=ErrorSeverity.WARNING,
            message=f"System {system_name} is running in degraded mode"
        ))
    
    async def health_check(self) -> HealthStatus:
        """統合健全性チェック"""
        try:
            unhealthy_count = 0
            degraded_count = 0
            total_systems = len(self.systems)
            
            if total_systems == 0:
                return HealthStatus.HEALTHY
            
            for state in self.system_states.values():
                if state.health == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1
                elif state.health == HealthStatus.DEGRADED:
                    degraded_count += 1
            
            # 異常システム比率が高い場合
            if unhealthy_count > total_systems * 0.3:
                return HealthStatus.UNHEALTHY
            if (unhealthy_count + degraded_count) > total_systems * 0.5:
                return HealthStatus.DEGRADED
            
            return HealthStatus.HEALTHY
            
        except Exception:
            return HealthStatus.UNHEALTHY
    
    def get_system_status(self) -> Dict[str, SystemState]:
        """システム状態取得"""
        return self.system_states.copy()


class EventBus(BaseComponent):
    """イベントバス"""
    
    def __init__(self, name: str):
        super().__init__(name, BaseConfig(name=name))
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history = InMemoryStorage(f"{name}_history", {
            'max_size': 1000,
            'cleanup_interval': 300
        })
        self._processing_tasks: Set[asyncio.Task] = set()
    
    async def start(self) -> bool:
        """イベントバス開始"""
        try:
            await self.event_history.initialize()
            self.status = SystemStatus.RUNNING
            return True
        except Exception as e:
            logger.error(f"EventBus start failed: {e}")
            return False
    
    async def stop(self) -> bool:
        """イベントバス停止"""
        try:
            self.status = SystemStatus.STOPPING
            
            # 処理中タスク完了待機
            if self._processing_tasks:
                await asyncio.gather(*self._processing_tasks, return_exceptions=True)
            
            self.status = SystemStatus.STOPPED
            return True
        except Exception as e:
            logger.error(f"EventBus stop failed: {e}")
            return False
    
    def subscribe(self, event_type: str, handler: Callable[[IntegrationEvent], None]):
        """イベント購読"""
        self.subscribers[event_type].append(handler)
        logger.info(f"Handler subscribed to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """購読解除"""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(handler)
                logger.info(f"Handler unsubscribed from event type: {event_type}")
            except ValueError:
                pass
    
    async def publish(self, event: IntegrationEvent):
        """イベント発行"""
        try:
            # イベント履歴保存
            await self.event_history.set(event.event_id, event, ttl=3600)
            
            # 購読者通知
            handlers = self.subscribers.get(event.event_type, [])
            handlers.extend(self.subscribers.get('*', []))  # 全イベント購読者
            
            for handler in handlers:
                task = asyncio.create_task(self._handle_event(handler, event))
                self._processing_tasks.add(task)
                task.add_done_callback(self._processing_tasks.discard)
            
            logger.debug(f"Event published: {event.event_type} from {event.source_system}")
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
    
    async def _handle_event(self, handler: Callable, event: IntegrationEvent):
        """イベント処理"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
            event.handled = True
        except Exception as e:
            logger.error(f"Event handler failed for {event.event_id}: {e}")
    
    async def get_event_history(self, event_type: Optional[str] = None, 
                               limit: int = 100) -> List[IntegrationEvent]:
        """イベント履歴取得"""
        # 実装簡略化のため、全履歴から検索
        # 実際の実装ではより効率的な検索が必要
        events = []
        # ストレージから履歴取得のロジックを実装
        return events[:limit]
    
    async def health_check(self) -> HealthStatus:
        """健全性チェック"""
        try:
            if self.status != SystemStatus.RUNNING:
                return HealthStatus.UNHEALTHY
            
            # アクティブタスク数チェック
            if len(self._processing_tasks) > 1000:
                return HealthStatus.DEGRADED
            
            # ストレージ健全性チェック
            storage_health = await self.event_history.health_check()
            return storage_health
            
        except Exception:
            return HealthStatus.UNHEALTHY


class CrossSystemDataBridge(BaseDataProcessor):
    """システム間データブリッジ"""
    
    def __init__(self, name: str, config: BaseConfig):
        super().__init__(name, config)
        self.data_transformers: Dict[str, Callable] = {}
        self.routing_rules: Dict[str, List[str]] = {}
        self.data_cache = InMemoryStorage(f"{name}_cache", {
            'max_size': config.cache_size,
            'cleanup_interval': 60
        })
    
    async def start(self) -> bool:
        """ブリッジ開始"""
        success = await super().start()
        if success:
            await self.data_cache.initialize()
        return success
    
    def register_transformer(self, data_type: str, transformer: Callable):
        """データ変換器登録"""
        self.data_transformers[data_type] = transformer
        logger.info(f"Transformer registered for data type: {data_type}")
    
    def set_routing_rule(self, source_system: str, target_systems: List[str]):
        """ルーティングルール設定"""
        self.routing_rules[source_system] = target_systems
        logger.info(f"Routing rule set: {source_system} -> {target_systems}")
    
    async def process_task(self, task_config: TaskConfig, task_data: Any) -> Any:
        """データブリッジタスク処理"""
        try:
            source_system = task_config.metadata.get('source_system')
            data_type = task_config.metadata.get('data_type')
            
            # データ変換
            if data_type in self.data_transformers:
                transformer = self.data_transformers[data_type]
                transformed_data = transformer(task_data)
            else:
                transformed_data = task_data
            
            # ルーティング
            target_systems = self.routing_rules.get(source_system, [])
            results = {}
            
            for target_system in target_systems:
                # データ配信（実際の実装では各システムのAPIを呼び出し）
                results[target_system] = await self._deliver_data(
                    target_system, transformed_data
                )
            
            # 結果キャッシュ
            cache_key = f"{source_system}_{data_type}_{int(time.time())}"
            await self.data_cache.set(cache_key, results, ttl=300)
            
            return results
            
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="process_bridge_task",
                task_id=task_config.task_id,
                severity=ErrorSeverity.MEDIUM
            ))
            raise
    
    async def _deliver_data(self, target_system: str, data: Any) -> Dict[str, Any]:
        """データ配信"""
        # 実際の実装では、対象システムのAPIエンドポイントに送信
        logger.info(f"Delivering data to {target_system}")
        return {
            'status': 'delivered',
            'timestamp': time.time(),
            'target_system': target_system,
            'data_size': len(str(data))
        }
    
    async def process_batch(self, data_batch: List[Any]) -> List[Any]:
        """バッチデータ処理"""
        results = []
        for data_item in data_batch:
            # 各データ項目を個別処理
            task_config = TaskConfig(
                task_id=f"batch_item_{int(time.time())}",
                metadata=data_item.get('metadata', {})
            )
            result = await self.process_task(task_config, data_item['data'])
            results.append(result)
        return results
    
    async def process_stream_item(self, data_item: Any) -> Any:
        """ストリームデータ処理"""
        task_config = TaskConfig(
            task_id=f"stream_item_{int(time.time())}",
            processing_mode=ProcessingMode.STREAMING,
            metadata=data_item.get('metadata', {})
        )
        return await self.process_task(task_config, data_item['data'])


# ファクトリ関数
def create_system_orchestrator(config: BaseConfig) -> SystemOrchestrator:
    """システムオーケストレーターファクトリ"""
    return SystemOrchestrator("system_orchestrator", config)


def create_event_bus() -> EventBus:
    """イベントバスファクトリ"""
    return EventBus("system_event_bus")


def create_data_bridge(config: BaseConfig) -> CrossSystemDataBridge:
    """データブリッジファクトリ"""
    return CrossSystemDataBridge("cross_system_bridge", config)