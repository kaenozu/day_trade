#!/usr/bin/env python3
"""
Distributed Consensus Implementation
分散合意アルゴリズム実装
"""

import asyncio
import json
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import uuid
import logging
import random

from ..functional.monads import Either, TradingResult

logger = logging.getLogger(__name__)

class NodeState(Enum):
    """ノード状態"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate" 
    LEADER = "leader"

class LogEntryType(Enum):
    """ログエントリタイプ"""
    COMMAND = "command"
    CONFIG = "config"
    NOOP = "noop"

@dataclass
class LogEntry:
    """ログエントリ"""
    term: int
    index: int
    entry_type: LogEntryType
    command: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = field(init=False)
    
    def __post_init__(self):
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """チェックサム計算"""
        data = f"{self.term}{self.index}{self.entry_type.value}{json.dumps(self.command, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """整合性確認"""
        return self.checksum == self._calculate_checksum()

@dataclass
class RaftNode:
    """Raftノード"""
    node_id: str
    address: str
    state: NodeState = NodeState.FOLLOWER
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[LogEntry] = field(default_factory=list)
    commit_index: int = 0
    last_applied: int = 0
    
    # リーダー専用
    next_index: Dict[str, int] = field(default_factory=dict)
    match_index: Dict[str, int] = field(default_factory=dict)
    
    # タイムスタンプ
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    election_timeout: float = field(default_factory=lambda: random.uniform(150, 300))
    heartbeat_interval: float = 50.0


class RaftMessage(ABC):
    """Raftメッセージ基底クラス"""
    pass

@dataclass
class VoteRequest(RaftMessage):
    """投票要求"""
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int

@dataclass
class VoteResponse(RaftMessage):
    """投票応答"""
    term: int
    vote_granted: bool
    voter_id: str

@dataclass
class AppendEntriesRequest(RaftMessage):
    """エントリ追加要求"""
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int

@dataclass
class AppendEntriesResponse(RaftMessage):
    """エントリ追加応答"""
    term: int
    success: bool
    follower_id: str
    match_index: int = 0


class RaftConsensus:
    """Raft合意アルゴリズム"""
    
    def __init__(self, node: RaftNode, cluster_nodes: List[str]):
        self.node = node
        self.cluster_nodes = cluster_nodes
        self.cluster_size = len(cluster_nodes)
        self.majority = self.cluster_size // 2 + 1
        
        self._running = False
        self._state_machine: Dict[str, Any] = {}
        self._message_handlers: Dict[type, Callable] = {
            VoteRequest: self._handle_vote_request,
            VoteResponse: self._handle_vote_response,
            AppendEntriesRequest: self._handle_append_entries_request,
            AppendEntriesResponse: self._handle_append_entries_response,
        }
        
        # 選挙関連
        self._votes_received: Set[str] = set()
        self._election_timer: Optional[asyncio.Task] = None
        self._heartbeat_timer: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """ノード開始"""
        if self._running:
            return
        
        self._running = True
        self.node.state = NodeState.FOLLOWER
        
        logger.info(f"Starting Raft node {self.node.node_id}")
        
        # 選挙タイマー開始
        await self._start_election_timer()
        
        # メッセージ処理ループ開始
        asyncio.create_task(self._message_processing_loop())
    
    async def stop(self) -> None:
        """ノード停止"""
        self._running = False
        
        if self._election_timer:
            self._election_timer.cancel()
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()
        
        logger.info(f"Stopped Raft node {self.node.node_id}")
    
    async def append_command(self, command: Dict[str, Any]) -> TradingResult[bool]:
        """コマンド追加"""
        if self.node.state != NodeState.LEADER:
            return TradingResult.failure('NOT_LEADER', 'Only leader can append commands')
        
        try:
            # ログエントリ作成
            entry = LogEntry(
                term=self.node.current_term,
                index=len(self.node.log) + 1,
                entry_type=LogEntryType.COMMAND,
                command=command
            )
            
            # ローカルログに追加
            self.node.log.append(entry)
            logger.info(f"Appended command to log: index={entry.index}, term={entry.term}")
            
            # フォロワーに複製
            success = await self._replicate_to_followers()
            return TradingResult.success(success)
            
        except Exception as e:
            return TradingResult.failure('APPEND_ERROR', str(e))
    
    async def get_state(self) -> Dict[str, Any]:
        """状態取得"""
        return {
            'node_id': self.node.node_id,
            'state': self.node.state.value,
            'current_term': self.node.current_term,
            'voted_for': self.node.voted_for,
            'log_length': len(self.node.log),
            'commit_index': self.node.commit_index,
            'last_applied': self.node.last_applied,
            'is_leader': self.node.state == NodeState.LEADER,
            'state_machine': self._state_machine
        }
    
    async def handle_message(self, message: RaftMessage) -> Optional[RaftMessage]:
        """メッセージ処理"""
        handler = self._message_handlers.get(type(message))
        if not handler:
            logger.warning(f"No handler for message type: {type(message)}")
            return None
        
        try:
            return await handler(message)
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            return None
    
    async def _start_election_timer(self) -> None:
        """選挙タイマー開始"""
        if self._election_timer:
            self._election_timer.cancel()
        
        async def election_timeout():
            await asyncio.sleep(self.node.election_timeout / 1000)  # ミリ秒を秒に変換
            if self._running and self.node.state != NodeState.LEADER:
                await self._start_election()
        
        self._election_timer = asyncio.create_task(election_timeout())
    
    async def _start_election(self) -> None:
        """選挙開始"""
        self.node.state = NodeState.CANDIDATE
        self.node.current_term += 1
        self.node.voted_for = self.node.node_id
        self._votes_received = {self.node.node_id}
        
        logger.info(f"Starting election for term {self.node.current_term}")
        
        # 自分に投票してリセット
        await self._start_election_timer()
        
        # 他のノードに投票要求送信
        last_log_index = len(self.node.log)
        last_log_term = self.node.log[-1].term if self.node.log else 0
        
        vote_request = VoteRequest(
            term=self.node.current_term,
            candidate_id=self.node.node_id,
            last_log_index=last_log_index,
            last_log_term=last_log_term
        )
        
        # 並列投票要求送信
        tasks = [self._send_vote_request(node_id, vote_request) 
                for node_id in self.cluster_nodes if node_id != self.node.node_id]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_vote_request(self, node_id: str, request: VoteRequest) -> None:
        """投票要求送信"""
        try:
            # 実際の実装では gRPC/HTTP で送信
            await asyncio.sleep(0.01)  # ネットワーク遅延シミュレーション
            
            # シミュレーション応答
            vote_granted = random.random() > 0.3  # 70%の確率で投票
            response = VoteResponse(
                term=self.node.current_term,
                vote_granted=vote_granted,
                voter_id=node_id
            )
            
            await self._handle_vote_response(response)
            
        except Exception as e:
            logger.error(f"Failed to send vote request to {node_id}: {e}")
    
    async def _handle_vote_request(self, request: VoteRequest) -> VoteResponse:
        """投票要求処理"""
        vote_granted = False
        
        # 用語チェック
        if request.term > self.node.current_term:
            self.node.current_term = request.term
            self.node.voted_for = None
            self.node.state = NodeState.FOLLOWER
        
        # 投票可否判定
        if (request.term == self.node.current_term and
            (self.node.voted_for is None or self.node.voted_for == request.candidate_id) and
            self._is_log_up_to_date(request.last_log_index, request.last_log_term)):
            
            vote_granted = True
            self.node.voted_for = request.candidate_id
            await self._start_election_timer()
        
        return VoteResponse(
            term=self.node.current_term,
            vote_granted=vote_granted,
            voter_id=self.node.node_id
        )
    
    async def _handle_vote_response(self, response: VoteResponse) -> None:
        """投票応答処理"""
        if response.term > self.node.current_term:
            self.node.current_term = response.term
            self.node.voted_for = None
            self.node.state = NodeState.FOLLOWER
            await self._start_election_timer()
            return
        
        if (self.node.state == NodeState.CANDIDATE and 
            response.term == self.node.current_term and 
            response.vote_granted):
            
            self._votes_received.add(response.voter_id)
            
            # 過半数の票を獲得
            if len(self._votes_received) >= self.majority:
                await self._become_leader()
    
    async def _become_leader(self) -> None:
        """リーダーになる"""
        self.node.state = NodeState.LEADER
        logger.info(f"Node {self.node.node_id} became leader for term {self.node.current_term}")
        
        # 選挙タイマー停止
        if self._election_timer:
            self._election_timer.cancel()
        
        # リーダー状態初期化
        for node_id in self.cluster_nodes:
            if node_id != self.node.node_id:
                self.node.next_index[node_id] = len(self.node.log) + 1
                self.node.match_index[node_id] = 0
        
        # ハートビート開始
        await self._start_heartbeat()
        
        # No-op エントリ追加（リーダーシップ確立）
        noop_command = LogEntry(
            term=self.node.current_term,
            index=len(self.node.log) + 1,
            entry_type=LogEntryType.NOOP,
            command={}
        )
        self.node.log.append(noop_command)
    
    async def _start_heartbeat(self) -> None:
        """ハートビート開始"""
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()
        
        async def send_heartbeats():
            while self._running and self.node.state == NodeState.LEADER:
                await self._send_heartbeats()
                await asyncio.sleep(self.node.heartbeat_interval / 1000)
        
        self._heartbeat_timer = asyncio.create_task(send_heartbeats())
    
    async def _send_heartbeats(self) -> None:
        """ハートビート送信"""
        for node_id in self.cluster_nodes:
            if node_id != self.node.node_id:
                await self._send_append_entries(node_id)
    
    async def _send_append_entries(self, node_id: str, entries: List[LogEntry] = None) -> None:
        """エントリ追加送信"""
        try:
            next_index = self.node.next_index.get(node_id, 1)
            prev_log_index = next_index - 1
            prev_log_term = 0
            
            if prev_log_index > 0 and prev_log_index <= len(self.node.log):
                prev_log_term = self.node.log[prev_log_index - 1].term
            
            request = AppendEntriesRequest(
                term=self.node.current_term,
                leader_id=self.node.node_id,
                prev_log_index=prev_log_index,
                prev_log_term=prev_log_term,
                entries=entries or [],
                leader_commit=self.node.commit_index
            )
            
            # 実際の実装では gRPC/HTTP で送信
            await asyncio.sleep(0.01)  # ネットワーク遅延シミュレーション
            
            # シミュレーション応答
            success = random.random() > 0.1  # 90%の確率で成功
            response = AppendEntriesResponse(
                term=self.node.current_term,
                success=success,
                follower_id=node_id,
                match_index=next_index - 1 if success else 0
            )
            
            await self._handle_append_entries_response(response)
            
        except Exception as e:
            logger.error(f"Failed to send append entries to {node_id}: {e}")
    
    async def _handle_append_entries_request(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        """エントリ追加要求処理"""
        success = False
        
        # 用語チェック
        if request.term > self.node.current_term:
            self.node.current_term = request.term
            self.node.voted_for = None
            self.node.state = NodeState.FOLLOWER
        
        if request.term == self.node.current_term:
            self.node.state = NodeState.FOLLOWER
            await self._start_election_timer()
            
            # ログ整合性チェック
            if self._log_consistency_check(request.prev_log_index, request.prev_log_term):
                success = True
                
                # エントリ追加
                if request.entries:
                    # 競合するエントリ削除
                    if request.prev_log_index + 1 <= len(self.node.log):
                        self.node.log = self.node.log[:request.prev_log_index]
                    
                    # 新しいエントリ追加
                    self.node.log.extend(request.entries)
                
                # コミットインデックス更新
                if request.leader_commit > self.node.commit_index:
                    self.node.commit_index = min(request.leader_commit, len(self.node.log))
                    await self._apply_committed_entries()
        
        return AppendEntriesResponse(
            term=self.node.current_term,
            success=success,
            follower_id=self.node.node_id,
            match_index=len(self.node.log) if success else 0
        )
    
    async def _handle_append_entries_response(self, response: AppendEntriesResponse) -> None:
        """エントリ追加応答処理"""
        if response.term > self.node.current_term:
            self.node.current_term = response.term
            self.node.voted_for = None
            self.node.state = NodeState.FOLLOWER
            await self._start_election_timer()
            return
        
        if (self.node.state == NodeState.LEADER and 
            response.term == self.node.current_term):
            
            if response.success:
                self.node.match_index[response.follower_id] = response.match_index
                self.node.next_index[response.follower_id] = response.match_index + 1
                
                # コミット可能性チェック
                await self._check_commit_advancement()
            else:
                # 失敗時は next_index を減らす
                if response.follower_id in self.node.next_index:
                    self.node.next_index[response.follower_id] = max(
                        1, self.node.next_index[response.follower_id] - 1
                    )
    
    async def _replicate_to_followers(self) -> bool:
        """フォロワーへの複製"""
        if self.node.state != NodeState.LEADER:
            return False
        
        # 最新エントリ取得
        latest_entries = [self.node.log[-1]] if self.node.log else []
        
        # 並列送信
        tasks = [self._send_append_entries(node_id, latest_entries)
                for node_id in self.cluster_nodes if node_id != self.node.node_id]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        return True
    
    async def _check_commit_advancement(self) -> None:
        """コミット進行チェック"""
        for index in range(self.node.commit_index + 1, len(self.node.log) + 1):
            replica_count = 1  # リーダー自身
            
            for node_id in self.cluster_nodes:
                if (node_id != self.node.node_id and 
                    self.node.match_index.get(node_id, 0) >= index):
                    replica_count += 1
            
            # 過半数に複製された場合コミット
            if replica_count >= self.majority and self.node.log[index - 1].term == self.node.current_term:
                self.node.commit_index = index
                await self._apply_committed_entries()
    
    async def _apply_committed_entries(self) -> None:
        """コミット済みエントリ適用"""
        while self.node.last_applied < self.node.commit_index:
            self.node.last_applied += 1
            entry = self.node.log[self.node.last_applied - 1]
            
            if entry.entry_type == LogEntryType.COMMAND:
                await self._apply_command_to_state_machine(entry.command)
    
    async def _apply_command_to_state_machine(self, command: Dict[str, Any]) -> None:
        """コマンドを状態マシンに適用"""
        try:
            command_type = command.get('type')
            
            if command_type == 'set':
                key = command.get('key')
                value = command.get('value')
                if key and value is not None:
                    self._state_machine[key] = value
                    logger.debug(f"Applied SET command: {key} = {value}")
            
            elif command_type == 'delete':
                key = command.get('key')
                if key and key in self._state_machine:
                    del self._state_machine[key]
                    logger.debug(f"Applied DELETE command: {key}")
            
        except Exception as e:
            logger.error(f"Failed to apply command to state machine: {e}")
    
    def _is_log_up_to_date(self, last_log_index: int, last_log_term: int) -> bool:
        """ログが最新か確認"""
        if not self.node.log:
            return True
        
        my_last_term = self.node.log[-1].term
        my_last_index = len(self.node.log)
        
        if last_log_term > my_last_term:
            return True
        elif last_log_term == my_last_term:
            return last_log_index >= my_last_index
        else:
            return False
    
    def _log_consistency_check(self, prev_log_index: int, prev_log_term: int) -> bool:
        """ログ整合性チェック"""
        if prev_log_index == 0:
            return True
        
        if prev_log_index > len(self.node.log):
            return False
        
        return self.node.log[prev_log_index - 1].term == prev_log_term
    
    async def _message_processing_loop(self) -> None:
        """メッセージ処理ループ"""
        while self._running:
            try:
                # 実際の実装では外部からのメッセージを処理
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1)


class DistributedLock:
    """分散ロック"""
    
    def __init__(self, consensus: RaftConsensus, lock_name: str, ttl_seconds: int = 30):
        self.consensus = consensus
        self.lock_name = lock_name
        self.ttl_seconds = ttl_seconds
        self.lock_id = str(uuid.uuid4())
        self._acquired = False
    
    async def acquire(self, timeout_seconds: int = 10) -> TradingResult[bool]:
        """ロック取得"""
        try:
            command = {
                'type': 'acquire_lock',
                'lock_name': self.lock_name,
                'lock_id': self.lock_id,
                'ttl': self.ttl_seconds,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            result = await self.consensus.append_command(command)
            if result.is_right() and result.get_right():
                self._acquired = True
                
                # TTL タイマー設定
                asyncio.create_task(self._ttl_timer())
                
            return result
            
        except Exception as e:
            return TradingResult.failure('LOCK_ERROR', str(e))
    
    async def release(self) -> TradingResult[bool]:
        """ロック解放"""
        if not self._acquired:
            return TradingResult.success(True)
        
        try:
            command = {
                'type': 'release_lock',
                'lock_name': self.lock_name,
                'lock_id': self.lock_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            result = await self.consensus.append_command(command)
            if result.is_right():
                self._acquired = False
            
            return result
            
        except Exception as e:
            return TradingResult.failure('LOCK_ERROR', str(e))
    
    async def _ttl_timer(self) -> None:
        """TTL タイマー"""
        await asyncio.sleep(self.ttl_seconds)
        if self._acquired:
            await self.release()


class LeaderElection:
    """リーダー選出"""
    
    def __init__(self, consensus: RaftConsensus):
        self.consensus = consensus
        self._leadership_callbacks: List[Callable] = []
    
    def add_leadership_callback(self, callback: Callable[[bool], None]) -> None:
        """リーダーシップ変更コールバック追加"""
        self._leadership_callbacks.append(callback)
    
    async def is_leader(self) -> bool:
        """リーダー確認"""
        state = await self.consensus.get_state()
        return state['is_leader']
    
    async def wait_for_leadership(self, timeout_seconds: int = 60) -> TradingResult[bool]:
        """リーダーシップ待機"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            if await self.is_leader():
                return TradingResult.success(True)
            await asyncio.sleep(0.1)
        
        return TradingResult.failure('TIMEOUT', 'Leadership timeout')
    
    async def step_down(self) -> TradingResult[None]:
        """リーダー辞任"""
        try:
            if await self.is_leader():
                # 実際の実装では明示的な辞任メッセージ
                await self.consensus.stop()
                await asyncio.sleep(0.1)
                await self.consensus.start()
            
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('STEP_DOWN_ERROR', str(e))


class ConsistentHashing:
    """一貫性ハッシュ"""
    
    def __init__(self, nodes: List[str], virtual_nodes: int = 100):
        self.nodes = set(nodes)
        self.virtual_nodes = virtual_nodes
        self._ring: Dict[int, str] = {}
        self._build_ring()
    
    def add_node(self, node: str) -> None:
        """ノード追加"""
        if node in self.nodes:
            return
        
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            virtual_key = self._hash(f"{node}:{i}")
            self._ring[virtual_key] = node
        
        logger.info(f"Added node {node} to consistent hash ring")
    
    def remove_node(self, node: str) -> None:
        """ノード削除"""
        if node not in self.nodes:
            return
        
        self.nodes.remove(node)
        keys_to_remove = [key for key, value in self._ring.items() if value == node]
        for key in keys_to_remove:
            del self._ring[key]
        
        logger.info(f"Removed node {node} from consistent hash ring")
    
    def get_node(self, key: str) -> Optional[str]:
        """キーに対応するノード取得"""
        if not self._ring:
            return None
        
        hash_key = self._hash(key)
        
        # リング上で最初に見つかるノード
        sorted_keys = sorted(self._ring.keys())
        for ring_key in sorted_keys:
            if hash_key <= ring_key:
                return self._ring[ring_key]
        
        # ラップアラウンド
        return self._ring[sorted_keys[0]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """キーに対応する複数ノード取得"""
        if not self._ring or count <= 0:
            return []
        
        hash_key = self._hash(key)
        sorted_keys = sorted(self._ring.keys())
        
        # 開始位置検索
        start_index = 0
        for i, ring_key in enumerate(sorted_keys):
            if hash_key <= ring_key:
                start_index = i
                break
        
        # ユニークノード収集
        unique_nodes = []
        visited_keys = set()
        
        for i in range(len(sorted_keys)):
            ring_key = sorted_keys[(start_index + i) % len(sorted_keys)]
            if ring_key in visited_keys:
                continue
            
            node = self._ring[ring_key]
            if node not in unique_nodes:
                unique_nodes.append(node)
                if len(unique_nodes) >= count:
                    break
            
            visited_keys.add(ring_key)
        
        return unique_nodes
    
    def _build_ring(self) -> None:
        """リング構築"""
        self._ring.clear()
        for node in self.nodes:
            for i in range(self.virtual_nodes):
                virtual_key = self._hash(f"{node}:{i}")
                self._ring[virtual_key] = node
    
    def _hash(self, key: str) -> int:
        """ハッシュ関数"""
        return hash(key) % (2**32)