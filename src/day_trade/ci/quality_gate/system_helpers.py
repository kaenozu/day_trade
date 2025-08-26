#!/usr/bin/env python3
"""
品質ゲートシステム - システムヘルパー

品質評価システムのヘルパー機能を提供するモジュール。
品質ゲート定義、スコア計算、推奨事項生成などを行う。
"""

from typing import Any, Dict, List

from .enums import QualityLevel, QualityMetric
from .types import QualityGate, QualityResult


def define_quality_gates() -> List[QualityGate]:
    """標準品質ゲートを定義
    
    プロジェクトで使用する品質ゲートとその閾値を定義する。
    各ゲートは独立して評価され、総合判定に寄与する。
    
    Returns:
        品質ゲートのリスト
    """
    return [
        # テストカバレッジ
        QualityGate(
            id="test_coverage",
            name="テストカバレッジ",
            description="単体テストカバレッジ率",
            metric_type=QualityMetric.COVERAGE,
            threshold_excellent=80.0,
            threshold_good=60.0,
            threshold_acceptable=30.0,
            weight=2.0,
            mandatory=True,
        ),
        # コード複雑度
        QualityGate(
            id="code_complexity",
            name="コード複雑度",
            description="McCabe循環的複雑度の平均",
            metric_type=QualityMetric.COMPLEXITY,
            threshold_excellent=5.0,
            threshold_good=10.0,
            threshold_acceptable=15.0,
            weight=1.5,
            mandatory=True,
        ),
        # 保守性指標
        QualityGate(
            id="maintainability_index",
            name="保守性指標",
            description="コードの保守性指標",
            metric_type=QualityMetric.MAINTAINABILITY,
            threshold_excellent=85.0,
            threshold_good=70.0,
            threshold_acceptable=50.0,
            weight=1.5,
            mandatory=True,
        ),
        # 依存関係健全性
        QualityGate(
            id="dependency_health",
            name="依存関係健全性",
            description="依存関係の脆弱性とライセンス適合性",
            metric_type=QualityMetric.DEPENDENCIES,
            threshold_excellent=95.0,
            threshold_good=85.0,
            threshold_acceptable=70.0,
            weight=2.0,
            mandatory=True,
        ),
        # セキュリティスコア
        QualityGate(
            id="security_score",
            name="セキュリティスコア",
            description="静的セキュリティ分析結果",
            metric_type=QualityMetric.SECURITY,
            threshold_excellent=95.0,
            threshold_good=85.0,
            threshold_acceptable=70.0,
            weight=2.5,
            mandatory=True,
        ),
        # 型チェック適合性
        QualityGate(
            id="type_checking",
            name="型チェック適合性",
            description="MyPy型チェック通過率",
            metric_type=QualityMetric.TYPING,
            threshold_excellent=95.0,
            threshold_good=85.0,
            threshold_acceptable=70.0,
            weight=1.0,
            mandatory=False,
        ),
    ]


def calculate_overall_score(results: List[QualityResult], quality_gates: List[QualityGate]) -> float:
    """総合スコアを計算
    
    各品質ゲートの結果を重み付き平均して総合スコアを算出。
    
    Args:
        results: 品質評価結果のリスト
        quality_gates: 品質ゲート定義のリスト
        
    Returns:
        0-100の総合スコア
    """
    if not results:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    gate_dict = {gate.id: gate for gate in quality_gates}

    for result in results:
        gate = gate_dict.get(result.gate_id)
        if gate:
            weight = gate.weight
            weighted_sum += result.value * weight
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def determine_overall_level(
    overall_score: float, results: List[QualityResult], quality_gates: List[QualityGate]
) -> QualityLevel:
    """総合品質レベルを判定
    
    総合スコアと必須ゲートの状況を考慮して総合品質レベルを決定。
    
    Args:
        overall_score: 総合スコア
        results: 品質評価結果のリスト
        quality_gates: 品質ゲート定義のリスト
        
    Returns:
        総合品質レベル
    """
    # 必須ゲートで失敗があるかチェック
    gate_dict = {gate.id: gate for gate in quality_gates}

    for result in results:
        gate = gate_dict.get(result.gate_id)
        if gate and gate.mandatory and not result.passed:
            return QualityLevel.CRITICAL

    # スコアベースの判定
    if overall_score >= 85:
        return QualityLevel.EXCELLENT
    elif overall_score >= 70:
        return QualityLevel.GOOD
    elif overall_score >= 50:
        return QualityLevel.ACCEPTABLE
    else:
        return QualityLevel.CRITICAL


def generate_recommendations(
    results: List[QualityResult], quality_gates: List[QualityGate]
) -> List[str]:
    """総合推奨事項を生成
    
    各品質ゲートの結果から改善推奨事項を生成する。
    
    Args:
        results: 品質評価結果のリスト
        quality_gates: 品質ゲート定義のリスト
        
    Returns:
        推奨事項のリスト
    """
    all_recommendations = []

    # 各ゲートの推奨事項を収集
    for result in results:
        all_recommendations.extend(result.recommendations)

    # 全体的な推奨事項を追加
    failed_mandatory = [
        r for r in results if not r.passed and is_mandatory_gate(r.gate_id, quality_gates)
    ]

    if failed_mandatory:
        all_recommendations.insert(
            0, "必須品質ゲートで失敗があります。最優先で修正してください。"
        )

    # 重複を除去し、優先順位でソート
    unique_recommendations = list(dict.fromkeys(all_recommendations))

    return unique_recommendations[:10]  # 上位10件に制限


def is_mandatory_gate(gate_id: str, quality_gates: List[QualityGate]) -> bool:
    """必須ゲートかどうかを判定
    
    Args:
        gate_id: ゲートID
        quality_gates: 品質ゲート定義のリスト
        
    Returns:
        必須ゲートかどうか
    """
    gate = next((g for g in quality_gates if g.id == gate_id), None)
    return gate.mandatory if gate else False


def create_quality_report_summary(gate_results: List[QualityResult]) -> Dict[str, int]:
    """品質レポートのサマリーを作成
    
    Args:
        gate_results: 品質評価結果のリスト
        
    Returns:
        品質レベル別のサマリー
    """
    return {
        "excellent_gates": len(
            [r for r in gate_results if r.level == QualityLevel.EXCELLENT]
        ),
        "good_gates": len(
            [r for r in gate_results if r.level == QualityLevel.GOOD]
        ),
        "acceptable_gates": len(
            [r for r in gate_results if r.level == QualityLevel.ACCEPTABLE]
        ),
        "critical_gates": len(
            [r for r in gate_results if r.level == QualityLevel.CRITICAL]
        ),
        "needs_improvement_gates": len(
            [r for r in gate_results if r.level == QualityLevel.NEEDS_IMPROVEMENT]
        ),
    }


def validate_quality_gates(quality_gates: List[QualityGate]) -> List[str]:
    """品質ゲート設定を検証
    
    品質ゲート設定の妥当性をチェックし、問題があれば警告を返す。
    
    Args:
        quality_gates: 品質ゲート定義のリスト
        
    Returns:
        検証エラーメッセージのリスト
    """
    errors = []
    
    for gate in quality_gates:
        # 閾値の妥当性チェック
        if gate.threshold_excellent < gate.threshold_good:
            errors.append(f"{gate.id}: 優秀閾値が良好閾値を下回っています")
        
        if gate.threshold_good < gate.threshold_acceptable:
            errors.append(f"{gate.id}: 良好閾値が許容閾値を下回っています")
        
        # 重みの妥当性チェック
        if gate.weight <= 0:
            errors.append(f"{gate.id}: 重みが0以下です")
        
        if gate.weight > 10:
            errors.append(f"{gate.id}: 重みが異常に大きいです（10超）")
    
    # ID重複チェック
    gate_ids = [gate.id for gate in quality_gates]
    duplicates = [gate_id for gate_id in gate_ids if gate_ids.count(gate_id) > 1]
    if duplicates:
        errors.append(f"重複したゲートID: {set(duplicates)}")
    
    return errors