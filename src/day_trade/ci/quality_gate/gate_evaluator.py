#!/usr/bin/env python3
"""
品質ゲートシステム - ゲート評価エンジン（モジュラー版）

個別の品質ゲートを評価し、判定結果を生成するメインエンジン。
分割された評価器を統合して使用する。
"""

from pathlib import Path

from .enums import QualityLevel, QualityMetric
from .evaluators import (
    ComplexityEvaluator,
    CoverageEvaluator,
    DependencyEvaluator,
    SecurityEvaluator,
    TypingEvaluator,
)
from .types import QualityGate, QualityResult


class QualityGateEvaluator:
    """品質ゲート評価エンジン
    
    各種品質メトリクスを評価し、品質ゲートの
    パス/フェイル判定を行うメインエンジンクラス。
    分割された評価器を統合して使用する。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = Path(project_root)
        
        # 各評価器を初期化
        self.coverage_evaluator = CoverageEvaluator(project_root)
        self.complexity_evaluator = ComplexityEvaluator(project_root)
        self.dependency_evaluator = DependencyEvaluator(project_root)
        self.security_evaluator = SecurityEvaluator(project_root)
        self.typing_evaluator = TypingEvaluator(project_root)

    async def evaluate_gate(self, gate: QualityGate) -> QualityResult:
        """品質ゲートを評価
        
        指定された品質ゲートを適切な評価器に委譲し、判定結果を返す。
        
        Args:
            gate: 評価対象の品質ゲート
            
        Returns:
            品質評価結果
        """
        try:
            # メトリクス種類に応じて適切な評価器を選択
            if gate.metric_type == QualityMetric.COVERAGE:
                return await self.coverage_evaluator.evaluate(gate)
            elif gate.metric_type == QualityMetric.COMPLEXITY:
                return await self.complexity_evaluator.evaluate(gate)
            elif gate.metric_type == QualityMetric.MAINTAINABILITY:
                return await self._evaluate_maintainability_gate(gate)
            elif gate.metric_type == QualityMetric.DEPENDENCIES:
                return await self.dependency_evaluator.evaluate(gate)
            elif gate.metric_type == QualityMetric.SECURITY:
                return await self.security_evaluator.evaluate(gate)
            elif gate.metric_type == QualityMetric.TYPING:
                return await self.typing_evaluator.evaluate(gate)
            else:
                return QualityResult(
                    gate_id=gate.id,
                    metric_type=gate.metric_type,
                    value=0.0,
                    level=QualityLevel.CRITICAL,
                    passed=False,
                    message="未対応のメトリクス種類",
                )

        except Exception as e:
            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=0.0,
                level=QualityLevel.CRITICAL,
                passed=False,
                message=f"評価エラー: {str(e)}",
            )

    async def _evaluate_maintainability_gate(self, gate: QualityGate) -> QualityResult:
        """保守性ゲートを評価
        
        複雑度評価器を使用して保守性指標を評価する。
        保守性はComplexityAnalyzerに含まれる。
        
        Args:
            gate: 保守性品質ゲート
            
        Returns:
            保守性評価結果
        """
        try:
            python_files = list(self.project_root.glob("src/**/*.py"))
            maintainability_scores = []
            low_maintainability_files = []

            for file_path in python_files:
                if self._should_skip_file(str(file_path)):
                    continue

                complexity_data = self.complexity_evaluator.complexity_analyzer.analyze_file(str(file_path))

                if "error" not in complexity_data:
                    mi = complexity_data["maintainability_index"]
                    maintainability_scores.append(mi)

                    if mi < 50:  # 低保守性ファイル
                        low_maintainability_files.append(
                            {"file": str(file_path), "maintainability_index": mi}
                        )

            average_maintainability = (
                sum(maintainability_scores) / len(maintainability_scores)
                if maintainability_scores
                else 0
            )

            level = self._determine_quality_level(average_maintainability, gate)
            passed = average_maintainability >= gate.threshold_acceptable

            recommendations = self._generate_maintainability_recommendations(
                average_maintainability, low_maintainability_files, gate
            )

            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=average_maintainability,
                level=level,
                passed=passed,
                message=f"平均保守性指標: {average_maintainability:.1f}",
                details={
                    "total_files": len(maintainability_scores),
                    "average_maintainability": average_maintainability,
                    "low_maintainability_files": low_maintainability_files[:5],
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=0.0,
                level=QualityLevel.CRITICAL,
                passed=False,
                message=f"保守性評価エラー: {str(e)}",
            )

    def _should_skip_file(self, file_path: str) -> bool:
        """ファイルを分析対象からスキップするかを判定
        
        Args:
            file_path: ファイルパス
            
        Returns:
            スキップするかどうか
        """
        file_path_lower = file_path.lower()
        return "test" in file_path_lower or "example" in file_path_lower

    def _determine_quality_level(
        self, value: float, gate: QualityGate, inverse: bool = False
    ) -> QualityLevel:
        """品質レベルを判定
        
        Args:
            value: 評価値
            gate: 品質ゲート定義
            inverse: True=低い値ほど良い（複雑度など）
            
        Returns:
            品質レベル
        """
        if not inverse:
            if value >= gate.threshold_excellent:
                return QualityLevel.EXCELLENT
            elif value >= gate.threshold_good:
                return QualityLevel.GOOD
            elif value >= gate.threshold_acceptable:
                return QualityLevel.ACCEPTABLE
            else:
                return QualityLevel.CRITICAL
        else:
            if value <= gate.threshold_excellent:
                return QualityLevel.EXCELLENT
            elif value <= gate.threshold_good:
                return QualityLevel.GOOD
            elif value <= gate.threshold_acceptable:
                return QualityLevel.ACCEPTABLE
            else:
                return QualityLevel.CRITICAL

    def _generate_maintainability_recommendations(
        self, average_maintainability: float, low_maintainability_files: list, gate: QualityGate
    ) -> list:
        """保守性改善推奨事項を生成
        
        Args:
            average_maintainability: 平均保守性指標
            low_maintainability_files: 低保守性ファイルリスト
            gate: 品質ゲート定義
            
        Returns:
            推奨事項のリスト
        """
        recommendations = []

        if average_maintainability < gate.threshold_acceptable:
            recommendations.extend([
                f"保守性指標 {average_maintainability:.1f} が目標値 {gate.threshold_acceptable} を下回っています。",
                "長い関数の分割、複雑なロジックの簡素化を検討してください。",
                "適切なコメントとドキュメントを追加してください。",
            ])

        if low_maintainability_files:
            recommendations.append(f"低保守性ファイル数: {len(low_maintainability_files)}個")
            recommendations.append("特に保守性の低いファイルから優先的に改善してください。")

        return recommendations[:6]  # 最大6個の推奨事項

    # プロパティで下位互換性を保つ
    @property
    def complexity_analyzer(self):
        """複雑度アナライザーへのアクセス（互換性維持）"""
        return self.complexity_evaluator.complexity_analyzer