#!/usr/bin/env python3
"""
品質ゲートシステム - テストカバレッジ評価器

テストカバレッジメトリクスの評価を行うモジュール。
pytest-covでカバレッジデータを取得・分析し、
品質判定を実施する。
"""

from typing import List

from ..coverage_analyzer import TestCoverageAnalyzer
from ..types import QualityGate, QualityResult
from .base_evaluator import BaseEvaluator


class CoverageEvaluator(BaseEvaluator):
    """テストカバレッジ評価器
    
    テストカバレッジを評価し、品質ゲートの
    パス/フェイル判定を行う。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        super().__init__(project_root)
        self.coverage_analyzer = TestCoverageAnalyzer()

    async def evaluate(self, gate: QualityGate) -> QualityResult:
        """テストカバレッジゲートを評価
        
        テストカバレッジを分析し、閾値との比較で
        品質レベルとパス/フェイル判定を実行する。
        
        Args:
            gate: 評価対象のカバレッジゲート
            
        Returns:
            カバレッジ評価結果
        """
        try:
            # カバレッジデータを取得
            coverage_data = self.coverage_analyzer.analyze_coverage()
            coverage_percentage = coverage_data.get("overall_coverage", 0)

            # 品質レベル判定
            level = self._determine_quality_level(coverage_percentage, gate)
            passed = coverage_percentage >= gate.threshold_acceptable

            # 推奨事項生成
            recommendations = self._generate_coverage_recommendations(
                coverage_percentage, coverage_data, gate
            )

            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=coverage_percentage,
                level=level,
                passed=passed,
                message=f"テストカバレッジ: {coverage_percentage:.1f}%",
                details=coverage_data,
                recommendations=recommendations,
            )

        except Exception as e:
            return self._create_error_result(gate, e)

    def _generate_coverage_recommendations(
        self, coverage_percentage: float, coverage_data: dict, gate: QualityGate
    ) -> List[str]:
        """カバレッジ改善推奨事項を生成
        
        カバレッジ状況に応じた具体的な改善提案を生成する。
        
        Args:
            coverage_percentage: 全体カバレッジ率
            coverage_data: カバレッジ詳細データ
            gate: 品質ゲート定義
            
        Returns:
            推奨事項のリスト
        """
        recommendations = []

        # カバレッジが不足している場合
        if coverage_percentage < gate.threshold_acceptable:
            recommendations.extend([
                "テストカバレッジが不足しています。追加のテストを作成してください。",
                f"目標カバレッジ: {gate.threshold_acceptable}%",
                f"現在のカバレッジ: {coverage_percentage:.1f}%",
                f"追加テストが必要な割合: {gate.threshold_acceptable - coverage_percentage:.1f}%",
            ])

        # 低カバレッジファイルの情報
        low_coverage_files = coverage_data.get("low_coverage_files", [])
        if low_coverage_files:
            recommendations.append(f"低カバレッジファイル数: {len(low_coverage_files)}個")
            recommendations.append("特に重要な機能のテストを優先してください。")

        # モジュール別カバレッジの分析
        module_coverage = coverage_data.get("module_coverage", {})
        if module_coverage:
            low_modules = [
                module for module, coverage in module_coverage.items()
                if coverage < 50
            ]
            if low_modules:
                recommendations.append(f"カバレッジが低いモジュール: {', '.join(low_modules[:3])}")

        # カバレッジ分布の分析
        coverage_dist = coverage_data.get("coverage_distribution", {})
        if coverage_dist:
            low_coverage_count = coverage_dist.get("0-20", 0) + coverage_dist.get("21-40", 0)
            if low_coverage_count > 0:
                recommendations.append(f"カバレッジ40%未満のファイルが{low_coverage_count}個あります。")

        # 一般的な改善提案
        if coverage_percentage < gate.threshold_good:
            recommendations.extend([
                "単体テストの拡充を検討してください。",
                "エッジケースのテストを追加してください。",
                "統合テストの導入も有効です。",
            ])

        return recommendations[:8]  # 最大8個の推奨事項に制限