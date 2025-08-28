#!/usr/bin/env python3
"""
バッチ処理エンジンパッケージ
Issue #376: バッチ処理の強化

統一バッチ処理エンジンの分割されたモジュール群
後方互換性のためのエクスポート
"""

# コア型定義
from .core_types import BatchJob, JobResult, PipelineStatus, StageResult, WorkflowStage

# プロセッサー類
from .data_fetch_processor import DataFetchProcessor
from .data_store_processor import DataStoreProcessor  
from .data_transform_processor import DataTransformProcessor
from .data_validate_processor import DataValidateProcessor
from .stage_processor import StageProcessor

# メインエンジン
from .engine import BatchProcessingEngine

# ユーティリティ
from .utils import execute_stock_batch_pipeline

# 後方互換性のため、すべてのクラスをエクスポート
__all__ = [
    # Core types
    "WorkflowStage",
    "PipelineStatus", 
    "BatchJob",
    "StageResult",
    "JobResult",
    
    # Processors
    "StageProcessor",
    "DataFetchProcessor", 
    "DataTransformProcessor",
    "DataValidateProcessor",
    "DataStoreProcessor",
    
    # Main engine
    "BatchProcessingEngine",
    
    # Utilities
    "execute_stock_batch_pipeline",
]