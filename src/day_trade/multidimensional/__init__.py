#!/usr/bin/env python3
"""
Multidimensional Parallel Processing System
多次元並列処理システム

This module implements advanced multidimensional parallel processing
for trading systems operating across multiple dimensions simultaneously.
"""

from .parallel_processing_engine import (
    MultidimensionalOrchestrator,
    WorkspaceManager,
    ParallelExecutionEngine,
    ResourcePool,
    DimensionalProcessor,
    TemporalProcessor,
    SpatialProcessor,
    InstrumentProcessor,
    ProcessingDimension,
    ComputationTask,
    ProcessingResult,
    ResourceAllocation,
    DimensionalWorkspace,
    TaskScheduler,
    PerformanceMonitor,
    DimensionType,
    ProcessingLevel,
    SynchronizationMode,
    ComputationPattern,
    initialize_multidimensional_system,
    create_trading_workflow,
    execute_workflow,
    get_system_status,
    shutdown_system
)

__all__ = [
    # Core Orchestration
    'MultidimensionalOrchestrator',
    'WorkspaceManager',
    'ParallelExecutionEngine',
    'ResourcePool',
    
    # Dimensional Processors
    'DimensionalProcessor',
    'TemporalProcessor',
    'SpatialProcessor', 
    'InstrumentProcessor',
    
    # Data Structures
    'ProcessingDimension',
    'ComputationTask',
    'ProcessingResult',
    'ResourceAllocation',
    'DimensionalWorkspace',
    
    # Supporting Components
    'TaskScheduler',
    'PerformanceMonitor',
    
    # Enums
    'DimensionType',
    'ProcessingLevel',
    'SynchronizationMode',
    'ComputationPattern',
    
    # Global Functions
    'initialize_multidimensional_system',
    'create_trading_workflow',
    'execute_workflow',
    'get_system_status',
    'shutdown_system'
]