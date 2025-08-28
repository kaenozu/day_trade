#!/usr/bin/env python3
"""
Distributed Event Sourcing Implementation
分散イベントソーシング実装
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, AsyncIterator
from enum import Enum
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

from ..architecture.domain.events import DomainEvent
from ..functional.monads import Either, TradingResult

logger = logging.getLogger(__name__)

T = TypeVar('T')

class EventState(Enum):
    """イベント状態"""
    PENDING = "pending"
    COMMITTED = "committed"
    REPLICATED = "replicated"
    FAILED = "failed"

@dataclass(frozen=True)
class StoredEvent:
    """保存イベント"""
    event_id: str
    aggregate_id: str
    aggregate_type: str
    event_type: str
    event_data: Dict[str, Any]
    version: int
    timestamp: datetime
    state: EventState = EventState.PENDING
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.checksum is None:
            checksum = self._calculate_checksum()
            object.__setattr__(self, 'checksum', checksum)
    
    def _calculate_checksum(self) -> str:
        """イベントチェックサム計算"""
        data = f"{self.event_id}{self.aggregate_id}{self.version}{json.dumps(self.event_data, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """整合性確認"""
        return self.checksum == self._calculate_checksum()


class EventStore(ABC):
    """イベントストア抽象基底クラス"""
    
    @abstractmethod
    async def append_events(self, aggregate_id: str, expected_version: int, 
                          events: List[DomainEvent]) -> TradingResult[List[StoredEvent]]:
        """イベント追加"""
        pass
    
    @abstractmethod
    async def get_events(self, aggregate_id: str, 
                        from_version: int = 0) -> TradingResult[List[StoredEvent]]:
        """イベント取得"""
        pass
    
    @abstractmethod
    async def get_all_events(self, from_timestamp: Optional[datetime] = None) -> AsyncIterator[StoredEvent]:
        """全イベント取得"""
        pass


class DistributedEventStore(EventStore):
    """分散イベントストア"""
    
    def __init__(self, nodes: List[str], replication_factor: int = 3):
        self.nodes = nodes
        self.replication_factor = min(replication_factor, len(nodes))
        self._events: Dict[str, List[StoredEvent]] = {}
        self._global_events: List[StoredEvent] = []
        self._version_cache: Dict[str, int] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    async def append_events(self, aggregate_id: str, expected_version: int, 
                          events: List[DomainEvent]) -> TradingResult[List[StoredEvent]]:
        """分散イベント追加"""
        try:
            # バージョン確認
            current_version = await self._get_current_version(aggregate_id)
            if current_version != expected_version:
                return TradingResult.failure(
                    'CONCURRENCY_ERROR', 
                    f'Version conflict: expected {expected_version}, got {current_version}'
                )
            
            # イベント変換
            stored_events = []
            for i, event in enumerate(events):
                stored_event = StoredEvent(
                    event_id=str(uuid.uuid4()),
                    aggregate_id=aggregate_id,
                    aggregate_type=event.__class__.__module__.split('.')[-2],
                    event_type=event.__class__.__name__,
                    event_data=asdict(event),
                    version=expected_version + i + 1,
                    timestamp=datetime.now(timezone.utc)
                )
                stored_events.append(stored_event)
            
            # 分散レプリケーション
            replication_result = await self._replicate_events(stored_events)
            if replication_result.is_left():
                return replication_result
            
            # ローカル保存
            if aggregate_id not in self._events:
                self._events[aggregate_id] = []
            
            self._events[aggregate_id].extend(stored_events)
            self._global_events.extend(stored_events)
            self._version_cache[aggregate_id] = stored_events[-1].version
            
            logger.info(f"Events appended for aggregate {aggregate_id}, new version: {self._version_cache[aggregate_id]}")
            return TradingResult.success(stored_events)
            
        except Exception as e:
            logger.error(f"Failed to append events: {e}")
            return TradingResult.failure('APPEND_ERROR', str(e))
    
    async def get_events(self, aggregate_id: str, 
                        from_version: int = 0) -> TradingResult[List[StoredEvent]]:
        """イベント取得"""
        try:
            events = self._events.get(aggregate_id, [])
            filtered_events = [e for e in events if e.version > from_version]
            
            # 整合性確認
            for event in filtered_events:
                if not event.verify_integrity():
                    return TradingResult.failure('INTEGRITY_ERROR', f'Event {event.event_id} integrity check failed')
            
            return TradingResult.success(filtered_events)
            
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return TradingResult.failure('GET_ERROR', str(e))
    
    async def get_all_events(self, from_timestamp: Optional[datetime] = None) -> AsyncIterator[StoredEvent]:
        """全イベント取得"""
        for event in self._global_events:
            if from_timestamp is None or event.timestamp >= from_timestamp:
                yield event
    
    async def _get_current_version(self, aggregate_id: str) -> int:
        """現在バージョン取得"""
        if aggregate_id in self._version_cache:
            return self._version_cache[aggregate_id]
        
        events = self._events.get(aggregate_id, [])
        if not events:
            return 0
        
        version = max(e.version for e in events)
        self._version_cache[aggregate_id] = version
        return version
    
    async def _replicate_events(self, events: List[StoredEvent]) -> TradingResult[None]:
        """イベントレプリケーション"""
        try:
            # レプリケーション先ノード選択
            target_nodes = self._select_replication_nodes()
            
            # 並列レプリケーション
            tasks = [self._replicate_to_node(node, events) for node in target_nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 成功率確認
            successful_replications = sum(1 for r in results if isinstance(r, bool) and r)
            if successful_replications < self.replication_factor // 2 + 1:
                return TradingResult.failure('REPLICATION_ERROR', 'Insufficient replications')
            
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('REPLICATION_ERROR', str(e))
    
    def _select_replication_nodes(self) -> List[str]:
        """レプリケーションノード選択"""
        return self.nodes[:self.replication_factor]
    
    async def _replicate_to_node(self, node: str, events: List[StoredEvent]) -> bool:
        """ノードへレプリケーション"""
        try:
            # シミュレーション（実際の実装では HTTP/gRPC 通信）
            await asyncio.sleep(0.001)  # ネットワーク遅延シミュレーション
            logger.debug(f"Replicated {len(events)} events to node {node}")
            return True
        except Exception as e:
            logger.error(f"Replication to node {node} failed: {e}")
            return False


class EventProjection(ABC, Generic[T]):
    """イベントプロジェクション基底クラス"""
    
    @abstractmethod
    def can_handle(self, event: StoredEvent) -> bool:
        """処理可能確認"""
        pass
    
    @abstractmethod
    async def handle(self, event: StoredEvent) -> TradingResult[T]:
        """イベント処理"""
        pass
    
    @abstractmethod
    async def get_state(self, aggregate_id: str) -> TradingResult[Optional[T]]:
        """状態取得"""
        pass


class EventProjectionManager:
    """イベントプロジェクション管理"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self._projections: List[EventProjection] = []
        self._is_running = False
    
    def register_projection(self, projection: EventProjection) -> None:
        """プロジェクション登録"""
        self._projections.append(projection)
    
    async def start(self) -> None:
        """プロジェクション開始"""
        if self._is_running:
            return
        
        self._is_running = True
        logger.info("Starting event projection manager")
        
        # 過去のイベント処理
        await self._process_historical_events()
        
        # リアルタイムイベント処理
        asyncio.create_task(self._process_realtime_events())
    
    async def stop(self) -> None:
        """プロジェクション停止"""
        self._is_running = False
        logger.info("Stopped event projection manager")
    
    async def _process_historical_events(self) -> None:
        """過去イベント処理"""
        try:
            async for event in self.event_store.get_all_events():
                await self._process_event(event)
        except Exception as e:
            logger.error(f"Failed to process historical events: {e}")
    
    async def _process_realtime_events(self) -> None:
        """リアルタイムイベント処理"""
        last_processed = datetime.now(timezone.utc)
        
        while self._is_running:
            try:
                async for event in self.event_store.get_all_events(last_processed):
                    await self._process_event(event)
                    last_processed = max(last_processed, event.timestamp)
                
                await asyncio.sleep(0.1)  # ポーリング間隔
                
            except Exception as e:
                logger.error(f"Failed to process realtime events: {e}")
                await asyncio.sleep(1)
    
    async def _process_event(self, event: StoredEvent) -> None:
        """イベント処理"""
        for projection in self._projections:
            if projection.can_handle(event):
                try:
                    await projection.handle(event)
                except Exception as e:
                    logger.error(f"Projection {projection.__class__.__name__} failed to handle event {event.event_id}: {e}")


class SnapshotManager:
    """スナップショット管理"""
    
    def __init__(self, event_store: EventStore, snapshot_frequency: int = 100):
        self.event_store = event_store
        self.snapshot_frequency = snapshot_frequency
        self._snapshots: Dict[str, Dict[str, Any]] = {}
    
    async def create_snapshot(self, aggregate_id: str, state: Dict[str, Any], version: int) -> TradingResult[None]:
        """スナップショット作成"""
        try:
            snapshot = {
                'aggregate_id': aggregate_id,
                'state': state,
                'version': version,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'checksum': self._calculate_state_checksum(state)
            }
            
            self._snapshots[aggregate_id] = snapshot
            logger.info(f"Snapshot created for aggregate {aggregate_id} at version {version}")
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('SNAPSHOT_ERROR', str(e))
    
    async def get_snapshot(self, aggregate_id: str) -> TradingResult[Optional[Dict[str, Any]]]:
        """スナップショット取得"""
        try:
            snapshot = self._snapshots.get(aggregate_id)
            if snapshot and self._verify_snapshot_integrity(snapshot):
                return TradingResult.success(snapshot)
            return TradingResult.success(None)
        except Exception as e:
            return TradingResult.failure('SNAPSHOT_ERROR', str(e))
    
    def _calculate_state_checksum(self, state: Dict[str, Any]) -> str:
        """状態チェックサム計算"""
        data = json.dumps(state, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _verify_snapshot_integrity(self, snapshot: Dict[str, Any]) -> bool:
        """スナップショット整合性確認"""
        stored_checksum = snapshot.get('checksum')
        calculated_checksum = self._calculate_state_checksum(snapshot['state'])
        return stored_checksum == calculated_checksum


class EventReplication:
    """イベントレプリケーション"""
    
    def __init__(self, primary_store: EventStore, replica_stores: List[EventStore]):
        self.primary_store = primary_store
        self.replica_stores = replica_stores
        self._replication_lag: Dict[str, int] = {}
    
    async def sync_replicas(self) -> TradingResult[None]:
        """レプリカ同期"""
        try:
            tasks = [self._sync_replica(replica) for replica in self.replica_stores]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            failed_syncs = [r for r in results if isinstance(r, Exception)]
            if failed_syncs:
                logger.error(f"Replica sync failures: {failed_syncs}")
                return TradingResult.failure('SYNC_ERROR', f'{len(failed_syncs)} replicas failed to sync')
            
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('SYNC_ERROR', str(e))
    
    async def _sync_replica(self, replica: EventStore) -> None:
        """個別レプリカ同期"""
        try:
            last_sync = datetime.now(timezone.utc) - asyncio.get_event_loop().time()
            
            async for event in self.primary_store.get_all_events(last_sync):
                # レプリカに書き込み（実装は簡略化）
                logger.debug(f"Syncing event {event.event_id} to replica")
                
        except Exception as e:
            logger.error(f"Replica sync failed: {e}")
            raise