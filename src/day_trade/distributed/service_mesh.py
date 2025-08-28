#!/usr/bin/env python3
"""
Service Mesh Implementation
サービスメッシュ実装
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from enum import Enum
import uuid
import logging
import random
from concurrent.futures import ThreadPoolExecutor

from ..functional.monads import Either, TradingResult

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """サービス状態"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"

class CircuitState(Enum):
    """サーキットブレーカー状態"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class ServiceInstance:
    """サービスインスタンス"""
    service_id: str
    service_name: str
    host: str
    port: int
    version: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    weight: int = 100
    
    @property
    def address(self) -> str:
        """アドレス取得"""
        return f"{self.host}:{self.port}"
    
    def is_healthy(self, timeout_seconds: int = 30) -> bool:
        """健全性確認"""
        if self.status != ServiceStatus.HEALTHY:
            return False
        
        time_since_heartbeat = datetime.now(timezone.utc) - self.last_heartbeat
        return time_since_heartbeat.seconds < timeout_seconds

@dataclass
class LoadBalancingRule:
    """負荷分散ルール"""
    algorithm: str  # round_robin, weighted, least_connections
    health_check_enabled: bool = True
    sticky_sessions: bool = False
    session_affinity_key: Optional[str] = None

@dataclass
class RoutingRule:
    """ルーティングルール"""
    service_name: str
    path_pattern: str
    weight_distribution: Dict[str, int] = field(default_factory=dict)  # version -> weight
    header_routing: Dict[str, str] = field(default_factory=dict)
    canary_enabled: bool = False
    canary_percentage: int = 0

@dataclass
class CircuitBreakerConfig:
    """サーキットブレーカー設定"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    request_volume_threshold: int = 20
    success_threshold: int = 3
    timeout_duration: float = 30.0


class ServiceRegistry:
    """サービスレジストリ"""
    
    def __init__(self):
        self._services: Dict[str, List[ServiceInstance]] = {}
        self._service_watchers: List[Callable] = []
        self._health_checker_active = False
        
    async def register_service(self, instance: ServiceInstance) -> TradingResult[None]:
        """サービス登録"""
        try:
            if instance.service_name not in self._services:
                self._services[instance.service_name] = []
            
            # 既存インスタンス更新または新規追加
            existing_index = -1
            for i, existing in enumerate(self._services[instance.service_name]):
                if existing.service_id == instance.service_id:
                    existing_index = i
                    break
            
            if existing_index >= 0:
                self._services[instance.service_name][existing_index] = instance
                logger.info(f"Updated service instance: {instance.service_id}")
            else:
                self._services[instance.service_name].append(instance)
                logger.info(f"Registered new service instance: {instance.service_id}")
            
            await self._notify_watchers(instance.service_name)
            return TradingResult.success(None)
            
        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return TradingResult.failure('REGISTRATION_ERROR', str(e))
    
    async def deregister_service(self, service_id: str) -> TradingResult[None]:
        """サービス登録解除"""
        try:
            for service_name, instances in self._services.items():
                original_count = len(instances)
                self._services[service_name] = [
                    inst for inst in instances if inst.service_id != service_id
                ]
                
                if len(self._services[service_name]) != original_count:
                    logger.info(f"Deregistered service instance: {service_id}")
                    await self._notify_watchers(service_name)
                    return TradingResult.success(None)
            
            return TradingResult.failure('NOT_FOUND', f'Service {service_id} not found')
            
        except Exception as e:
            logger.error(f"Failed to deregister service: {e}")
            return TradingResult.failure('DEREGISTRATION_ERROR', str(e))
    
    async def get_service_instances(self, service_name: str, 
                                  healthy_only: bool = True) -> TradingResult[List[ServiceInstance]]:
        """サービスインスタンス取得"""
        try:
            instances = self._services.get(service_name, [])
            
            if healthy_only:
                instances = [inst for inst in instances if inst.is_healthy()]
            
            return TradingResult.success(instances)
            
        except Exception as e:
            return TradingResult.failure('QUERY_ERROR', str(e))
    
    async def update_health_status(self, service_id: str, status: ServiceStatus) -> TradingResult[None]:
        """健全性状態更新"""
        try:
            for instances in self._services.values():
                for instance in instances:
                    if instance.service_id == service_id:
                        instance.status = status
                        instance.last_heartbeat = datetime.now(timezone.utc)
                        logger.debug(f"Updated health status for {service_id}: {status}")
                        return TradingResult.success(None)
            
            return TradingResult.failure('NOT_FOUND', f'Service {service_id} not found')
            
        except Exception as e:
            return TradingResult.failure('HEALTH_UPDATE_ERROR', str(e))
    
    def add_watcher(self, callback: Callable[[str], None]) -> None:
        """サービス変更監視者追加"""
        self._service_watchers.append(callback)
    
    async def start_health_checker(self, check_interval: int = 10) -> None:
        """健全性チェッカー開始"""
        if self._health_checker_active:
            return
        
        self._health_checker_active = True
        logger.info("Starting service health checker")
        asyncio.create_task(self._health_check_loop(check_interval))
    
    async def stop_health_checker(self) -> None:
        """健全性チェッカー停止"""
        self._health_checker_active = False
        logger.info("Stopped service health checker")
    
    async def _notify_watchers(self, service_name: str) -> None:
        """監視者通知"""
        for watcher in self._service_watchers:
            try:
                await asyncio.get_event_loop().run_in_executor(None, watcher, service_name)
            except Exception as e:
                logger.error(f"Service watcher notification failed: {e}")
    
    async def _health_check_loop(self, check_interval: int) -> None:
        """健全性チェックループ"""
        while self._health_checker_active:
            try:
                for service_name, instances in self._services.items():
                    for instance in instances:
                        await self._check_instance_health(instance)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(1)
    
    async def _check_instance_health(self, instance: ServiceInstance) -> None:
        """インスタンス健全性チェック"""
        try:
            # 実際の実装では HTTP/gRPC ヘルスチェック
            # ここでは簡単なシミュレーション
            await asyncio.sleep(0.001)
            
            # 90%の確率で健全と判定
            is_healthy = random.random() > 0.1
            new_status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
            
            if instance.status != new_status:
                instance.status = new_status
                logger.debug(f"Health status changed for {instance.service_id}: {new_status}")
            
            instance.last_heartbeat = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Health check failed for {instance.service_id}: {e}")
            instance.status = ServiceStatus.UNHEALTHY


class LoadBalancer:
    """負荷分散器"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self._round_robin_counters: Dict[str, int] = {}
        self._connection_counts: Dict[str, int] = {}
    
    async def select_instance(self, service_name: str, 
                            rule: LoadBalancingRule) -> TradingResult[Optional[ServiceInstance]]:
        """インスタンス選択"""
        try:
            instances_result = await self.service_registry.get_service_instances(
                service_name, rule.health_check_enabled
            )
            
            if instances_result.is_left():
                return instances_result
            
            instances = instances_result.get_right()
            if not instances:
                return TradingResult.success(None)
            
            if rule.algorithm == "round_robin":
                selected = self._round_robin_select(service_name, instances)
            elif rule.algorithm == "weighted":
                selected = self._weighted_select(instances)
            elif rule.algorithm == "least_connections":
                selected = self._least_connections_select(instances)
            else:
                selected = random.choice(instances)
            
            return TradingResult.success(selected)
            
        except Exception as e:
            return TradingResult.failure('LOAD_BALANCING_ERROR', str(e))
    
    def _round_robin_select(self, service_name: str, 
                          instances: List[ServiceInstance]) -> ServiceInstance:
        """ラウンドロビン選択"""
        if service_name not in self._round_robin_counters:
            self._round_robin_counters[service_name] = 0
        
        index = self._round_robin_counters[service_name] % len(instances)
        self._round_robin_counters[service_name] += 1
        
        return instances[index]
    
    def _weighted_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """重み付き選択"""
        total_weight = sum(inst.weight for inst in instances)
        if total_weight == 0:
            return random.choice(instances)
        
        random_value = random.randint(1, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if random_value <= current_weight:
                return instance
        
        return instances[-1]
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最小接続数選択"""
        min_connections = float('inf')
        selected_instance = None
        
        for instance in instances:
            connections = self._connection_counts.get(instance.service_id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance
        
        return selected_instance or instances[0]
    
    def increment_connections(self, service_id: str) -> None:
        """接続数増加"""
        self._connection_counts[service_id] = self._connection_counts.get(service_id, 0) + 1
    
    def decrement_connections(self, service_id: str) -> None:
        """接続数減少"""
        if service_id in self._connection_counts:
            self._connection_counts[service_id] = max(0, self._connection_counts[service_id] - 1)


class CircuitBreaker:
    """サーキットブレーカー"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._request_count = 0
        self._metrics: Dict[str, Any] = {}
    
    async def call(self, func: Callable, *args, **kwargs) -> TradingResult[Any]:
        """サーキットブレーカー経由呼び出し"""
        try:
            if not await self._can_execute():
                return TradingResult.failure('CIRCUIT_OPEN', 'Circuit breaker is open')
            
            start_time = time.time()
            
            try:
                # タイムアウト付き実行
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_duration
                )
                
                await self._on_success()
                
                # メトリクス更新
                self._metrics['last_success_time'] = datetime.now(timezone.utc)
                self._metrics['avg_response_time'] = time.time() - start_time
                
                return TradingResult.success(result)
                
            except asyncio.TimeoutError:
                await self._on_failure()
                return TradingResult.failure('TIMEOUT', 'Request timed out')
            except Exception as e:
                await self._on_failure()
                return TradingResult.failure('EXECUTION_ERROR', str(e))
                
        except Exception as e:
            logger.error(f"Circuit breaker error: {e}")
            return TradingResult.failure('CIRCUIT_ERROR', str(e))
    
    async def _can_execute(self) -> bool:
        """実行可能確認"""
        if self._state == CircuitState.CLOSED:
            return True
        
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        
        if self._state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """リセット試行判定"""
        if not self._last_failure_time:
            return True
        
        time_since_failure = datetime.now(timezone.utc) - self._last_failure_time
        return time_since_failure.seconds >= self.config.recovery_timeout
    
    async def _on_success(self) -> None:
        """成功時処理"""
        self._request_count += 1
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker transitioned to CLOSED")
        
        if self._state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)
    
    async def _on_failure(self) -> None:
        """失敗時処理"""
        self._request_count += 1
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        
        if (self._state == CircuitState.CLOSED and 
            self._failure_count >= self.config.failure_threshold and
            self._request_count >= self.config.request_volume_threshold):
            
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker transitioned to OPEN")
        
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker returned to OPEN from HALF_OPEN")
    
    @property
    def state(self) -> CircuitState:
        """状態取得"""
        return self._state
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """メトリクス取得"""
        return {
            **self._metrics,
            'state': self._state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'request_count': self._request_count,
            'failure_rate': self._failure_count / max(self._request_count, 1)
        }


class ServiceDiscovery:
    """サービス発見"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self._dns_cache: Dict[str, List[ServiceInstance]] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=5)
    
    async def discover_service(self, service_name: str, 
                             use_cache: bool = True) -> TradingResult[List[ServiceInstance]]:
        """サービス発見"""
        try:
            # キャッシュ確認
            if use_cache and self._is_cache_valid(service_name):
                return TradingResult.success(self._dns_cache[service_name])
            
            # レジストリから取得
            instances_result = await self.service_registry.get_service_instances(service_name)
            if instances_result.is_left():
                return instances_result
            
            instances = instances_result.get_right()
            
            # キャッシュ更新
            self._dns_cache[service_name] = instances
            self._cache_ttl[service_name] = datetime.now(timezone.utc) + self._cache_duration
            
            return TradingResult.success(instances)
            
        except Exception as e:
            return TradingResult.failure('DISCOVERY_ERROR', str(e))
    
    async def resolve_service_address(self, service_name: str) -> TradingResult[Optional[str]]:
        """サービスアドレス解決"""
        instances_result = await self.discover_service(service_name)
        if instances_result.is_left():
            return instances_result
        
        instances = instances_result.get_right()
        if not instances:
            return TradingResult.success(None)
        
        # 最初の健全なインスタンスを選択
        for instance in instances:
            if instance.is_healthy():
                return TradingResult.success(instance.address)
        
        # 健全でなくても最初のインスタンスを返す
        return TradingResult.success(instances[0].address)
    
    def _is_cache_valid(self, service_name: str) -> bool:
        """キャッシュ有効性確認"""
        if service_name not in self._cache_ttl:
            return False
        return datetime.now(timezone.utc) < self._cache_ttl[service_name]
    
    def invalidate_cache(self, service_name: Optional[str] = None) -> None:
        """キャッシュ無効化"""
        if service_name:
            self._dns_cache.pop(service_name, None)
            self._cache_ttl.pop(service_name, None)
        else:
            self._dns_cache.clear()
            self._cache_ttl.clear()


class TrafficRouting:
    """トラフィックルーティング"""
    
    def __init__(self, service_discovery: ServiceDiscovery):
        self.service_discovery = service_discovery
        self._routing_rules: Dict[str, RoutingRule] = {}
        self._canary_sessions: Set[str] = set()
    
    def add_routing_rule(self, rule: RoutingRule) -> None:
        """ルーティングルール追加"""
        self._routing_rules[rule.service_name] = rule
        logger.info(f"Added routing rule for {rule.service_name}")
    
    async def route_request(self, service_name: str, 
                          request_context: Dict[str, Any]) -> TradingResult[Optional[ServiceInstance]]:
        """リクエストルーティング"""
        try:
            rule = self._routing_rules.get(service_name)
            if not rule:
                # デフォルトルーティング
                return await self._default_routing(service_name)
            
            # ヘッダーベースルーティング
            if rule.header_routing:
                version = self._extract_version_from_headers(request_context, rule)
                if version:
                    return await self._route_to_version(service_name, version)
            
            # カナリアデプロイメント
            if rule.canary_enabled:
                session_id = request_context.get('session_id', '')
                if self._should_route_to_canary(session_id, rule.canary_percentage):
                    return await self._route_to_canary(service_name)
            
            # 重み付きルーティング
            if rule.weight_distribution:
                version = self._select_version_by_weight(rule.weight_distribution)
                return await self._route_to_version(service_name, version)
            
            # デフォルトルーティング
            return await self._default_routing(service_name)
            
        except Exception as e:
            return TradingResult.failure('ROUTING_ERROR', str(e))
    
    async def _default_routing(self, service_name: str) -> TradingResult[Optional[ServiceInstance]]:
        """デフォルトルーティング"""
        instances_result = await self.service_discovery.discover_service(service_name)
        if instances_result.is_left():
            return instances_result
        
        instances = instances_result.get_right()
        if not instances:
            return TradingResult.success(None)
        
        # 健全なインスタンスをランダム選択
        healthy_instances = [inst for inst in instances if inst.is_healthy()]
        if healthy_instances:
            return TradingResult.success(random.choice(healthy_instances))
        
        return TradingResult.success(random.choice(instances))
    
    async def _route_to_version(self, service_name: str, version: str) -> TradingResult[Optional[ServiceInstance]]:
        """バージョン指定ルーティング"""
        instances_result = await self.service_discovery.discover_service(service_name)
        if instances_result.is_left():
            return instances_result
        
        instances = instances_result.get_right()
        version_instances = [inst for inst in instances if inst.version == version]
        
        if not version_instances:
            return await self._default_routing(service_name)
        
        healthy_instances = [inst for inst in version_instances if inst.is_healthy()]
        if healthy_instances:
            return TradingResult.success(random.choice(healthy_instances))
        
        return TradingResult.success(random.choice(version_instances))
    
    async def _route_to_canary(self, service_name: str) -> TradingResult[Optional[ServiceInstance]]:
        """カナリアルーティング"""
        instances_result = await self.service_discovery.discover_service(service_name)
        if instances_result.is_left():
            return instances_result
        
        instances = instances_result.get_right()
        canary_instances = [
            inst for inst in instances 
            if inst.tags.get('deployment_type') == 'canary'
        ]
        
        if not canary_instances:
            return await self._default_routing(service_name)
        
        return TradingResult.success(random.choice(canary_instances))
    
    def _extract_version_from_headers(self, context: Dict[str, Any], 
                                    rule: RoutingRule) -> Optional[str]:
        """ヘッダーからバージョン抽出"""
        headers = context.get('headers', {})
        for header_key, version_value in rule.header_routing.items():
            if headers.get(header_key) == version_value:
                return version_value
        return None
    
    def _should_route_to_canary(self, session_id: str, canary_percentage: int) -> bool:
        """カナリアルーティング判定"""
        if session_id in self._canary_sessions:
            return True
        
        # セッション ID のハッシュ値で決定
        hash_value = hash(session_id) % 100
        should_route = hash_value < canary_percentage
        
        if should_route:
            self._canary_sessions.add(session_id)
        
        return should_route
    
    def _select_version_by_weight(self, weight_distribution: Dict[str, int]) -> str:
        """重み付きバージョン選択"""
        total_weight = sum(weight_distribution.values())
        if total_weight == 0:
            return list(weight_distribution.keys())[0]
        
        random_value = random.randint(1, total_weight)
        current_weight = 0
        
        for version, weight in weight_distribution.items():
            current_weight += weight
            if random_value <= current_weight:
                return version
        
        return list(weight_distribution.keys())[-1]