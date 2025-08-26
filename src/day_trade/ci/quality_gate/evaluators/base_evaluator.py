#!/usr/bin/env python3
"""
品質ゲートシステム - ベース評価器

全ての品質メトリクス評価器の基底クラス。
共通の評価ロジックとヘルパーメソッドを提供する。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from ..enums import QualityLevel
from ..types import QualityGate, QualityResult


class BaseEvaluator(ABC):
    """品質評価器の基底クラス
    
    全ての具体的な評価器が継承するベースクラス。
    共通の評価ロジックとインターフェースを定義する。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = Path(project_root)

    @abstractmethod
    async def evaluate(self, gate: QualityGate) -> QualityResult:
        """品質ゲートを評価
        
        サブクラスで具体的な評価ロジックを実装する。
        
        Args:
            gate: 評価対象の品質ゲート
            
        Returns:
            品質評価結果
        """
        pass

    def _determine_quality_level(
        self, value: float, gate: QualityGate, inverse: bool = False
    ) -> QualityLevel:
        """品質レベルを判定
        
        評価値と閾値を比較して品質レベルを決定する。
        
        Args:
            value: 評価値
            gate: 品質ゲート定義
            inverse: True=低い値ほど良い（複雑度など）
            
        Returns:
            品質レベル
        """
        if not inverse:
            # 高い値ほど良い（カバレッジ、保守性など）
            if value >= gate.threshold_excellent:
                return QualityLevel.EXCELLENT
            elif value >= gate.threshold_good:
                return QualityLevel.GOOD
            elif value >= gate.threshold_acceptable:
                return QualityLevel.ACCEPTABLE
            else:
                return QualityLevel.CRITICAL
        else:
            # 低い値ほど良い（複雑度など）
            if value <= gate.threshold_excellent:
                return QualityLevel.EXCELLENT
            elif value <= gate.threshold_good:
                return QualityLevel.GOOD
            elif value <= gate.threshold_acceptable:
                return QualityLevel.ACCEPTABLE
            else:
                return QualityLevel.CRITICAL

    def _create_error_result(self, gate: QualityGate, error: Exception) -> QualityResult:
        """エラー結果を作成
        
        評価中にエラーが発生した場合のQualityResultを作成する。
        
        Args:
            gate: 品質ゲート定義
            error: 発生したエラー
            
        Returns:
            エラー状態のQualityResult
        """
        return QualityResult(
            gate_id=gate.id,
            metric_type=gate.metric_type,
            value=0.0,
            level=QualityLevel.CRITICAL,
            passed=False,
            message=f"評価エラー: {str(error)}",
            details={"error": str(error), "error_type": type(error).__name__},
            recommendations=["評価エラーが発生しました。ログを確認してください。"],
        )