#!/usr/bin/env python3
"""
Edge Computing Orchestrator
エッジコンピューティング・オーケストレーター
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import uuid
import logging
import psutil
from concurrent.futures import ThreadPoolExecutor

from ..functional.monads import Either, TradingResult

logger = logging.getLogger(__name__)

class EdgeNodeType(Enum):
    """エッジノードタイプ"""
    GATEWAY = "gateway"
    COMPUTE = "compute" 
    STORAGE = "storage"
    INFERENCE = "inference"
    HYBRID = "hybrid"

class NodeStatus(Enum):
    """ノード状態"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"

class WorkloadType(Enum):
    """ワークロードタイプ"""
    MARKET_DATA_PROCESSING = "market_data_processing"
    TRADING_SIGNAL_GENERATION = "trading_signal_generation"
    RISK_CALCULATION = "risk_calculation"
    ORDER_EXECUTION = "order_execution"
    ML_INFERENCE = "ml_inference"
    DATA_ANALYTICS = "data_analytics"

@dataclass
class ResourceMetrics:
    """リソースメトリクス"""
    cpu_usage: float  # 0.0-1.0
    memory_usage: float  # 0.0-1.0
    disk_usage: float  # 0.0-1.0
    network_bandwidth: float  # Mbps
    latency: float  # milliseconds
    throughput: float  # operations/second
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def overall_utilization(self) -> float:
        """総合使用率"""
        return (self.cpu_usage + self.memory_usage + self.disk_usage) / 3.0
    
    def is_overloaded(self, threshold: float = 0.8) -> bool:
        """過負荷判定"""
        return self.overall_utilization > threshold

@dataclass
class EdgeNode:
    """エッジノード"""
    node_id: str
    node_type: EdgeNodeType
    location: str  # 地理的位置
    endpoint: str  # 接続エンドポイント
    capabilities: Set[str] = field(default_factory=set)
    status: NodeStatus = NodeStatus.OFFLINE
    resources: Optional[ResourceMetrics] = None
    workloads: List[str] = field(default_factory=list)  # 実行中ワークロードID
    last_heartbeat: Optional[datetime] = None
    
    # ハードウェア仕様
    cpu_cores: int = 4
    memory_gb: int = 8
    storage_gb: int = 100
    gpu_available: bool = False
    
    # ネットワーク情報
    max_bandwidth: float = 1000.0  # Mbps
    average_latency: float = 10.0  # ms
    
    def update_status(self, new_status: NodeStatus) -> None:
        """ステータス更新"""
        if self.status != new_status:
            logger.info(f"Node {self.node_id} status changed: {self.status} -> {new_status}")
            self.status = new_status
    
    def is_healthy(self) -> bool:
        """健全性確認"""
        return (self.status == NodeStatus.ONLINE and 
                self.last_heartbeat and 
                datetime.now(timezone.utc) - self.last_heartbeat < timedelta(minutes=5))
    
    def can_handle_workload(self, workload_type: WorkloadType) -> bool:
        """ワークロード処理可能性確認"""
        required_capabilities = {
            WorkloadType.ML_INFERENCE: {"gpu", "ml_runtime"},
            WorkloadType.MARKET_DATA_PROCESSING: {"high_bandwidth", "low_latency"},
            WorkloadType.ORDER_EXECUTION: {"ultra_low_latency", "high_reliability"},
            WorkloadType.RISK_CALCULATION: {"high_compute", "memory"},
            WorkloadType.DATA_ANALYTICS: {"storage", "compute"},
            WorkloadType.TRADING_SIGNAL_GENERATION: {"low_latency", "compute"}
        }
        
        required = required_capabilities.get(workload_type, set())
        return required.issubset(self.capabilities)

@dataclass
class Workload:
    """ワークロード"""
    workload_id: str
    workload_type: WorkloadType
    priority: int  # 1=最高, 5=最低
    resource_requirements: Dict[str, float]  # cpu, memory, storage, bandwidth
    latency_requirement: float  # ms
    data_locality_preference: Optional[str] = None  # 優先データ位置
    affinity_rules: List[str] = field(default_factory=list)  # ノード親和性ルール
    anti_affinity_rules: List[str] = field(default_factory=list)  # ノード反親和性ルール
    
    # 実行情報
    assigned_node: Optional[str] = None
    status: str = "pending"
    start_time: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None

@dataclass
class EdgeCluster:
    """エッジクラスター"""
    cluster_id: str
    name: str
    location: str
    nodes: List[EdgeNode] = field(default_factory=list)
    load_balancer_endpoint: Optional[str] = None
    
    def get_total_resources(self) -> ResourceMetrics:
        """総リソース計算"""
        if not self.nodes:
            return ResourceMetrics(0, 0, 0, 0, float('inf'), 0)
        
        online_nodes = [n for n in self.nodes if n.status == NodeStatus.ONLINE]
        if not online_nodes:
            return ResourceMetrics(0, 0, 0, 0, float('inf'), 0)
        
        total_cpu = sum(n.resources.cpu_usage if n.resources else 0 for n in online_nodes) / len(online_nodes)
        total_memory = sum(n.resources.memory_usage if n.resources else 0 for n in online_nodes) / len(online_nodes)
        total_disk = sum(n.resources.disk_usage if n.resources else 0 for n in online_nodes) / len(online_nodes)
        total_bandwidth = sum(n.max_bandwidth for n in online_nodes)
        avg_latency = sum(n.average_latency for n in online_nodes) / len(online_nodes)
        total_throughput = sum(n.resources.throughput if n.resources else 0 for n in online_nodes)
        
        return ResourceMetrics(
            cpu_usage=total_cpu,
            memory_usage=total_memory, 
            disk_usage=total_disk,
            network_bandwidth=total_bandwidth,
            latency=avg_latency,
            throughput=total_throughput
        )
    
    def get_healthy_nodes(self) -> List[EdgeNode]:
        """健全ノード取得"""
        return [node for node in self.nodes if node.is_healthy()]


class WorkloadScheduler(ABC):
    """ワークロードスケジューラー基底クラス"""
    
    @abstractmethod
    async def schedule(self, workload: Workload, 
                      available_nodes: List[EdgeNode]) -> TradingResult[EdgeNode]:
        """ワークロードスケジューリング"""
        pass

class LatencyAwareScheduler(WorkloadScheduler):
    """レイテンシ重視スケジューラー"""
    
    async def schedule(self, workload: Workload, 
                      available_nodes: List[EdgeNode]) -> TradingResult[EdgeNode]:
        """レイテンシ最小化スケジューリング"""
        try:
            # 要件満たすノードフィルタ
            suitable_nodes = [
                node for node in available_nodes
                if (node.can_handle_workload(workload.workload_type) and
                    node.average_latency <= workload.latency_requirement and
                    not (node.resources and node.resources.is_overloaded()))
            ]
            
            if not suitable_nodes:
                return TradingResult.failure('NO_SUITABLE_NODES', 'No nodes meet requirements')
            
            # レイテンシ最小のノード選択
            best_node = min(suitable_nodes, key=lambda n: n.average_latency)
            
            return TradingResult.success(best_node)
            
        except Exception as e:
            return TradingResult.failure('SCHEDULING_ERROR', str(e))

class LoadAwareScheduler(WorkloadScheduler):
    """負荷分散スケジューラー"""
    
    async def schedule(self, workload: Workload,
                      available_nodes: List[EdgeNode]) -> TradingResult[EdgeNode]:
        """負荷分散スケジューリング"""
        try:
            suitable_nodes = [
                node for node in available_nodes
                if node.can_handle_workload(workload.workload_type)
            ]
            
            if not suitable_nodes:
                return TradingResult.failure('NO_SUITABLE_NODES', 'No nodes meet requirements')
            
            # 負荷最小のノード選択
            def node_load_score(node: EdgeNode) -> float:
                if not node.resources:
                    return 0.0
                
                # リソース要件を考慮した負荷スコア
                cpu_req = workload.resource_requirements.get('cpu', 0.1)
                mem_req = workload.resource_requirements.get('memory', 0.1)
                
                cpu_score = (node.resources.cpu_usage + cpu_req) * 0.4
                mem_score = (node.resources.memory_usage + mem_req) * 0.4
                workload_score = len(node.workloads) * 0.2
                
                return cpu_score + mem_score + workload_score
            
            best_node = min(suitable_nodes, key=node_load_score)
            
            return TradingResult.success(best_node)
            
        except Exception as e:
            return TradingResult.failure('SCHEDULING_ERROR', str(e))

class WorkloadDistributor:
    """ワークロード分散器"""
    
    def __init__(self):
        self._active_workloads: Dict[str, Workload] = {}
        self._workload_history: List[Dict[str, Any]] = []
        self._scheduler = LatencyAwareScheduler()  # デフォルトスケジューラー
    
    def set_scheduler(self, scheduler: WorkloadScheduler) -> None:
        """スケジューラー設定"""
        self._scheduler = scheduler
    
    async def submit_workload(self, workload: Workload, 
                            available_nodes: List[EdgeNode]) -> TradingResult[str]:
        """ワークロード投入"""
        try:
            # スケジューリング実行
            schedule_result = await self._scheduler.schedule(workload, available_nodes)
            
            if schedule_result.is_left():
                return schedule_result
            
            assigned_node = schedule_result.get_right()
            
            # ワークロード割り当て
            workload.assigned_node = assigned_node.node_id
            workload.status = "scheduled"
            workload.start_time = datetime.now(timezone.utc)
            
            # ノードにワークロード追加
            assigned_node.workloads.append(workload.workload_id)
            
            # 履歴記録
            self._active_workloads[workload.workload_id] = workload
            self._record_workload_event(workload, "scheduled", assigned_node.node_id)
            
            logger.info(f"Workload {workload.workload_id} scheduled to node {assigned_node.node_id}")
            
            return TradingResult.success(workload.workload_id)
            
        except Exception as e:
            return TradingResult.failure('WORKLOAD_SUBMISSION_ERROR', str(e))
    
    async def complete_workload(self, workload_id: str) -> TradingResult[None]:
        """ワークロード完了"""
        try:
            if workload_id not in self._active_workloads:
                return TradingResult.failure('WORKLOAD_NOT_FOUND', f'Workload {workload_id} not found')
            
            workload = self._active_workloads[workload_id]
            workload.status = "completed"
            
            # ノードからワークロード削除
            if workload.assigned_node:
                # ノード検索は実際の実装では別途管理
                pass
            
            # 履歴記録
            self._record_workload_event(workload, "completed")
            
            # アクティブリストから削除
            del self._active_workloads[workload_id]
            
            logger.info(f"Workload {workload_id} completed")
            
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('WORKLOAD_COMPLETION_ERROR', str(e))
    
    def get_workload_status(self, workload_id: str) -> Optional[Workload]:
        """ワークロード状態取得"""
        return self._active_workloads.get(workload_id)
    
    def get_active_workloads(self) -> List[Workload]:
        """アクティブワークロード一覧"""
        return list(self._active_workloads.values())
    
    def _record_workload_event(self, workload: Workload, event: str, node_id: str = None) -> None:
        """ワークロードイベント記録"""
        event_record = {
            'workload_id': workload.workload_id,
            'event': event,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'node_id': node_id or workload.assigned_node,
            'workload_type': workload.workload_type.value,
            'priority': workload.priority
        }
        
        self._workload_history.append(event_record)
        
        # 履歴サイズ制限（最新1000件）
        if len(self._workload_history) > 1000:
            self._workload_history = self._workload_history[-1000:]


class EdgeOrchestrator:
    """エッジオーケストレーター"""
    
    def __init__(self):
        self._clusters: Dict[str, EdgeCluster] = {}
        self._nodes: Dict[str, EdgeNode] = {}
        self._workload_distributor = WorkloadDistributor()
        self._monitoring_active = False
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    async def register_node(self, node: EdgeNode) -> TradingResult[None]:
        """ノード登録"""
        try:
            self._nodes[node.node_id] = node
            
            # 初期リソースメトリクス設定
            if not node.resources:
                node.resources = await self._collect_node_metrics(node)
            
            node.last_heartbeat = datetime.now(timezone.utc)
            node.update_status(NodeStatus.ONLINE)
            
            logger.info(f"Registered edge node: {node.node_id} at {node.location}")
            
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('NODE_REGISTRATION_ERROR', str(e))
    
    async def create_cluster(self, cluster_id: str, name: str, location: str,
                           node_ids: List[str]) -> TradingResult[EdgeCluster]:
        """クラスター作成"""
        try:
            # ノード存在確認
            cluster_nodes = []
            for node_id in node_ids:
                if node_id not in self._nodes:
                    return TradingResult.failure('NODE_NOT_FOUND', f'Node {node_id} not found')
                cluster_nodes.append(self._nodes[node_id])
            
            cluster = EdgeCluster(
                cluster_id=cluster_id,
                name=name,
                location=location,
                nodes=cluster_nodes
            )
            
            self._clusters[cluster_id] = cluster
            
            logger.info(f"Created edge cluster: {cluster_id} with {len(cluster_nodes)} nodes")
            
            return TradingResult.success(cluster)
            
        except Exception as e:
            return TradingResult.failure('CLUSTER_CREATION_ERROR', str(e))
    
    async def deploy_workload(self, workload: Workload, 
                            target_cluster: Optional[str] = None) -> TradingResult[str]:
        """ワークロードデプロイ"""
        try:
            # 利用可能ノード決定
            if target_cluster:
                if target_cluster not in self._clusters:
                    return TradingResult.failure('CLUSTER_NOT_FOUND', f'Cluster {target_cluster} not found')
                available_nodes = self._clusters[target_cluster].get_healthy_nodes()
            else:
                available_nodes = [node for node in self._nodes.values() if node.is_healthy()]
            
            if not available_nodes:
                return TradingResult.failure('NO_AVAILABLE_NODES', 'No healthy nodes available')
            
            # ワークロード分散実行
            result = await self._workload_distributor.submit_workload(workload, available_nodes)
            
            return result
            
        except Exception as e:
            return TradingResult.failure('WORKLOAD_DEPLOYMENT_ERROR', str(e))
    
    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """モニタリング開始"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        logger.info("Starting edge monitoring")
        
        asyncio.create_task(self._monitoring_loop(interval_seconds))
    
    async def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self._monitoring_active = False
        logger.info("Stopped edge monitoring")
    
    async def _monitoring_loop(self, interval: int) -> None:
        """モニタリングループ"""
        while self._monitoring_active:
            try:
                # 全ノードの健全性チェック
                tasks = [self._check_node_health(node) for node in self._nodes.values()]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # クラスターヘルスチェック
                await self._check_cluster_health()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1)
    
    async def _check_node_health(self, node: EdgeNode) -> None:
        """ノード健全性チェック"""
        try:
            # ハートビート確認
            if node.last_heartbeat:
                time_since_heartbeat = datetime.now(timezone.utc) - node.last_heartbeat
                if time_since_heartbeat > timedelta(minutes=5):
                    node.update_status(NodeStatus.OFFLINE)
                    logger.warning(f"Node {node.node_id} offline (no heartbeat)")
                    return
            
            # リソースメトリクス更新
            node.resources = await self._collect_node_metrics(node)
            
            # 負荷状態チェック
            if node.resources and node.resources.is_overloaded():
                node.update_status(NodeStatus.OVERLOADED)
            elif node.status != NodeStatus.ONLINE:
                node.update_status(NodeStatus.ONLINE)
                
        except Exception as e:
            logger.error(f"Health check failed for node {node.node_id}: {e}")
            node.update_status(NodeStatus.DEGRADED)
    
    async def _collect_node_metrics(self, node: EdgeNode) -> ResourceMetrics:
        """ノードメトリクス収集"""
        try:
            # 実際の実装では、ノードからメトリクスAPIで取得
            # ここではローカルシステムのメトリクスを使用
            
            cpu_percent = psutil.cpu_percent(interval=1) / 100.0
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # ネットワーク統計（簡略化）
            network_stats = psutil.net_io_counters()
            bandwidth = (network_stats.bytes_sent + network_stats.bytes_recv) / (1024 * 1024)  # MB
            
            return ResourceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent / 100.0,
                disk_usage=disk.percent / 100.0,
                network_bandwidth=min(bandwidth, node.max_bandwidth),
                latency=node.average_latency,  # 実際にはpingで測定
                throughput=len(node.workloads) * 10.0  # 簡略化
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for node {node.node_id}: {e}")
            return ResourceMetrics(0, 0, 0, 0, float('inf'), 0)
    
    async def _check_cluster_health(self) -> None:
        """クラスター健全性チェック"""
        for cluster in self._clusters.values():
            healthy_nodes = cluster.get_healthy_nodes()
            total_nodes = len(cluster.nodes)
            
            if len(healthy_nodes) < total_nodes * 0.5:  # 50%未満が健全
                logger.warning(f"Cluster {cluster.cluster_id} degraded: "
                             f"{len(healthy_nodes)}/{total_nodes} nodes healthy")
    
    def get_cluster_status(self) -> Dict[str, Dict[str, Any]]:
        """クラスター状態取得"""
        status = {}
        
        for cluster_id, cluster in self._clusters.items():
            healthy_nodes = cluster.get_healthy_nodes()
            total_resources = cluster.get_total_resources()
            
            status[cluster_id] = {
                'name': cluster.name,
                'location': cluster.location,
                'total_nodes': len(cluster.nodes),
                'healthy_nodes': len(healthy_nodes),
                'resources': {
                    'cpu_usage': total_resources.cpu_usage,
                    'memory_usage': total_resources.memory_usage,
                    'latency': total_resources.latency
                },
                'active_workloads': sum(len(node.workloads) for node in healthy_nodes)
            }
        
        return status
    
    def get_node_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """ノード詳細取得"""
        if node_id not in self._nodes:
            return None
        
        node = self._nodes[node_id]
        
        return {
            'node_id': node.node_id,
            'type': node.node_type.value,
            'location': node.location,
            'status': node.status.value,
            'capabilities': list(node.capabilities),
            'hardware': {
                'cpu_cores': node.cpu_cores,
                'memory_gb': node.memory_gb,
                'storage_gb': node.storage_gb,
                'gpu_available': node.gpu_available
            },
            'network': {
                'max_bandwidth': node.max_bandwidth,
                'average_latency': node.average_latency
            },
            'resources': {
                'cpu_usage': node.resources.cpu_usage if node.resources else 0,
                'memory_usage': node.resources.memory_usage if node.resources else 0,
                'disk_usage': node.resources.disk_usage if node.resources else 0
            } if node.resources else None,
            'active_workloads': len(node.workloads),
            'last_heartbeat': node.last_heartbeat.isoformat() if node.last_heartbeat else None
        }