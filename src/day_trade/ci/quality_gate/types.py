#!/usr/bin/env python3
"""
品質ゲートシステム - データ型定義

品質ゲートシステムで使用される主要なデータ構造を定義するモジュール。
各データクラスは品質評価の各段階で情報を保持し、
システム間でのデータ交換を効率化する。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .enums import QualityLevel, QualityMetric


@dataclass
class QualityGate:
    """品質ゲート定義
    
    個別の品質チェック項目を定義するデータクラス。
    各ゲートは独立して評価され、閾値に基づいて
    パス/フェイルを判定する。
    
    Attributes:
        id: ゲートの一意識別子
        name: ゲートの表示名
        description: ゲートの詳細説明
        metric_type: 評価対象のメトリクス種類
        threshold_excellent: 優秀レベルの閾値
        threshold_good: 良好レベルの閾値
        threshold_acceptable: 許容レベルの閾値
        weight: 総合評価での重み係数
        mandatory: 必須ゲートかどうか
        enabled: ゲートが有効かどうか
    """

    id: str
    name: str
    description: str
    metric_type: QualityMetric
    threshold_excellent: float
    threshold_good: float
    threshold_acceptable: float
    weight: float = 1.0
    mandatory: bool = True
    enabled: bool = True


@dataclass
class QualityResult:
    """品質評価結果
    
    個別品質ゲートの評価結果を保持するデータクラス。
    評価値とレベル判定、推奨事項などの詳細情報を含む。
    
    Attributes:
        gate_id: 評価対象ゲートのID
        metric_type: メトリクス種類
        value: 実測値
        level: 判定された品質レベル
        passed: パス/フェイルの判定
        message: 評価結果のメッセージ
        details: 詳細な評価データ
        recommendations: 改善推奨事項
    """

    gate_id: str
    metric_type: QualityMetric
    value: float
    level: QualityLevel
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FileQualityReport:
    """ファイル品質レポート
    
    個別ファイルの品質評価結果を保持するデータクラス。
    ファイルレベルでの詳細な品質指標を提供。
    
    Attributes:
        file_path: ファイルパス
        lines_of_code: コード行数
        complexity_score: 複雑度スコア
        maintainability_index: 保守性指標
        type_coverage: 型アノテーションカバレッジ
        documentation_coverage: ドキュメントカバレッジ
        security_issues: セキュリティ問題リスト
        quality_level: 総合品質レベル
    """

    file_path: str
    lines_of_code: int
    complexity_score: float
    maintainability_index: float
    type_coverage: float
    documentation_coverage: float
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    quality_level: QualityLevel = QualityLevel.ACCEPTABLE