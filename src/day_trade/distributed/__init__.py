#!/usr/bin/env python3
"""
Distributed Systems Module
分散システム統合モジュール
"""

# Legacy Dask/Ray distributed computing
from .dask_data_processor import (
    DaskBatchProcessor,
    DaskDataProcessor,
    DaskStockAnalyzer,
    create_dask_data_processor,
)
from .distributed_computing_manager import (
    ComputingBackend,
    DistributedComputingManager,
    DistributedResult,
    DistributedTask,
    TaskDistributionStrategy,
)

# Next-Gen Distributed Architecture
from .event_sourcing import (
    DistributedEventStore, 
    EventProjectionManager,
    SnapshotManager,
    EventReplication
)
from .cqrs import (
    DistributedCommandBus,
    DistributedQueryBus,
    CommandHandler,
    QueryHandler,
    EventualConsistency
)
from .service_mesh import (
    ServiceRegistry,
    LoadBalancer,
    CircuitBreaker,
    ServiceDiscovery,
    TrafficRouting
)
from .consensus import (
    RaftConsensus,
    DistributedLock,
    LeaderElection,
    ConsistentHashing
)

__all__ = [
    # Legacy Distributed Computing
    "DaskDataProcessor",
    "DaskStockAnalyzer",
    "DaskBatchProcessor",
    "create_dask_data_processor",
    "DistributedComputingManager",
    "ComputingBackend",
    "TaskDistributionStrategy",
    "DistributedTask",
    "DistributedResult",
    
    # Next-Gen Distributed Systems
    # Event Sourcing
    'DistributedEventStore',
    'EventProjectionManager',
    'SnapshotManager', 
    'EventReplication',
    
    # CQRS
    'DistributedCommandBus',
    'DistributedQueryBus',
    'CommandHandler',
    'QueryHandler',
    'EventualConsistency',
    
    # Service Mesh
    'ServiceRegistry',
    'LoadBalancer',
    'CircuitBreaker',
    'ServiceDiscovery',
    'TrafficRouting',
    
    # Consensus
    'RaftConsensus',
    'DistributedLock',
    'LeaderElection',
    'ConsistentHashing'
]
