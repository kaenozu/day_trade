#!/usr/bin/env python3
"""
バッチ処理エンジンのコア型定義
Issue #376: バッチ処理の強化

ワークフローステージ、パイプライン状態、ジョブ・結果データクラス
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkflowStage(Enum):
    """ワークフローステージ"""

    DATA_FETCH = "data_fetch"
    DATA_TRANSFORM = "data_transform"
    DATA_VALIDATE = "data_validate"
    DATA_STORE = "data_store"
    DATA_ANALYZE = "data_analyze"


class PipelineStatus(Enum):
    """パイプライン状態"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """バッチジョブ定義"""

    job_id: str
    job_name: str
    workflow_stages: List[WorkflowStage]
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout_seconds: int = 3600
    retry_count: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: PipelineStatus = PipelineStatus.PENDING

    def is_expired(self) -> bool:
        """有効期限チェック"""
        return time.time() - self.created_at > self.timeout_seconds


@dataclass
class StageResult:
    """ステージ実行結果"""

    stage: WorkflowStage
    success: bool
    data: Any = None
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """ジョブ実行結果"""

    job_id: str
    job_name: str
    success: bool
    total_processing_time_ms: float
    stage_results: List[StageResult] = field(default_factory=list)
    final_data: Any = None
    error_message: Optional[str] = None
    completed_at: float = field(default_factory=time.time)