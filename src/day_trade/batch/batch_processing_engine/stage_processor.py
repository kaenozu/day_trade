#!/usr/bin/env python3
"""
ステージプロセッサー基底クラス
Issue #376: バッチ処理の強化

すべてのステージプロセッサーの基底クラス
"""

from typing import Any, Dict

from .core_types import StageResult, WorkflowStage


class StageProcessor:
    """ステージプロセッサー基底クラス"""

    def __init__(self, stage: WorkflowStage):
        self.stage = stage

    async def process(self, data: Any, parameters: Dict[str, Any]) -> StageResult:
        """ステージ処理実行"""
        raise NotImplementedError