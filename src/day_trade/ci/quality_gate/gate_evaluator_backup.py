#!/usr/bin/env python3
"""
品質ゲートシステム - ゲート評価エンジン

個別の品質ゲートを評価し、判定結果を生成するモジュール。
各種メトリクス（カバレッジ、複雑度、セキュリティ等）の
評価ロジックを提供する。
"""

import ast
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .complexity_analyzer import CodeComplexityAnalyzer
from .coverage_analyzer import TestCoverageAnalyzer
from .dependency_checker import DependencyHealthChecker
from .enums import QualityLevel, QualityMetric
from .types import QualityGate, QualityResult


class QualityGateEvaluator:
    """品質ゲート評価エンジン
    
    各種品質メトリクスを評価し、品質ゲートの
    パス/フェイル判定を行うエンジンクラス。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = Path(project_root)
        self.complexity_analyzer = CodeComplexityAnalyzer()
        self.coverage_analyzer = TestCoverageAnalyzer()
        self.dependency_checker = DependencyHealthChecker()

    async def evaluate_gate(self, gate: QualityGate) -> QualityResult:
        """品質ゲートを評価
        
        指定された品質ゲートを評価し、判定結果を返す。
        
        Args:
            gate: 評価対象の品質ゲート
            
        Returns:
            品質評価結果
        """
        try:
            if gate.metric_type == QualityMetric.COVERAGE:
                return await self._evaluate_coverage_gate(gate)
            elif gate.metric_type == QualityMetric.COMPLEXITY:
                return await self._evaluate_complexity_gate(gate)
            elif gate.metric_type == QualityMetric.MAINTAINABILITY:
                return await self._evaluate_maintainability_gate(gate)
            elif gate.metric_type == QualityMetric.DEPENDENCIES:
                return await self._evaluate_dependency_gate(gate)
            elif gate.metric_type == QualityMetric.SECURITY:
                return await self._evaluate_security_gate(gate)
            elif gate.metric_type == QualityMetric.TYPING:
                return await self._evaluate_typing_gate(gate)
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

    async def _evaluate_coverage_gate(self, gate: QualityGate) -> QualityResult:
        """テストカバレッジゲートを評価"""
        coverage_data = self.coverage_analyzer.analyze_coverage()
        coverage_percentage = coverage_data.get("overall_coverage", 0)

        level = self._determine_quality_level(coverage_percentage, gate)
        passed = coverage_percentage >= gate.threshold_acceptable

        recommendations = []
        if coverage_percentage < gate.threshold_acceptable:
            recommendations.extend(
                [
                    "テストカバレッジが不足しています。追加のテストを作成してください。",
                    f"低カバレッジファイル数: {len(coverage_data.get('low_coverage_files', []))}",
                    "特に重要な機能のテストを優先してください。",
                ]
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

    async def _evaluate_complexity_gate(self, gate: QualityGate) -> QualityResult:
        """複雑度ゲートを評価"""
        python_files = list(self.project_root.glob("src/**/*.py"))
        
        total_complexity = 0
        file_count = 0
        high_complexity_files = []

        for file_path in python_files:
            if "test" in str(file_path).lower() or "example" in str(file_path).lower():
                continue

            complexity_data = self.complexity_analyzer.analyze_file(str(file_path))

            if "error" not in complexity_data:
                complexity = complexity_data["mccabe_complexity"]
                total_complexity += complexity
                file_count += 1

                if complexity > 20:  # 高複雑度ファイル
                    high_complexity_files.append(
                        {"file": str(file_path), "complexity": complexity}
                    )

        average_complexity = total_complexity / file_count if file_count > 0 else 0

        level = self._determine_quality_level(
            average_complexity, gate, inverse=True
        )  # 複雑度は低い方が良い
        passed = average_complexity <= gate.threshold_acceptable

        recommendations = []
        if average_complexity > gate.threshold_acceptable:
            recommendations.extend(
                [
                    f"平均複雑度 {average_complexity:.1f} が閾値を超えています。",
                    "複雑な関数やクラスのリファクタリングを検討してください。",
                    f"高複雑度ファイル数: {len(high_complexity_files)}",
                ]
            )

        return QualityResult(
            gate_id=gate.id,
            metric_type=gate.metric_type,
            value=average_complexity,
            level=level,
            passed=passed,
            message=f"平均複雑度: {average_complexity:.1f}",
            details={
                "total_files": file_count,
                "average_complexity": average_complexity,
                "high_complexity_files": high_complexity_files[:5],
            },
            recommendations=recommendations,
        )

    async def _evaluate_maintainability_gate(self, gate: QualityGate) -> QualityResult:
        """保守性ゲートを評価"""
        python_files = list(self.project_root.glob("src/**/*.py"))

        maintainability_scores = []
        low_maintainability_files = []

        for file_path in python_files:
            if "test" in str(file_path).lower() or "example" in str(file_path).lower():
                continue

            complexity_data = self.complexity_analyzer.analyze_file(str(file_path))

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

        recommendations = []
        if average_maintainability < gate.threshold_acceptable:
            recommendations.extend(
                [
                    f"保守性指標 {average_maintainability:.1f} が低下しています。",
                    "長い関数の分割、複雑なロジックの簡素化を検討してください。",
                    "適切なコメントとドキュメントを追加してください。",
                ]
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

    async def _evaluate_dependency_gate(self, gate: QualityGate) -> QualityResult:
        """依存関係ゲートを評価"""
        dependency_data = self.dependency_checker.check_dependencies()
        health_score = dependency_data.get("health_score", 50.0)

        level = self._determine_quality_level(health_score, gate)
        passed = health_score >= gate.threshold_acceptable

        vulnerabilities = dependency_data.get("vulnerabilities", [])
        outdated_count = len(dependency_data.get("outdated_packages", []))

        recommendations = []
        if health_score < gate.threshold_acceptable:
            recommendations.extend(
                [
                    f"依存関係健全性スコア {health_score:.1f} が低下しています。",
                    f"脆弱性のあるパッケージ: {len(vulnerabilities)}個",
                    f"更新が必要なパッケージ: {outdated_count}個",
                    "依存関係の更新とセキュリティパッチ適用を実施してください。",
                ]
            )

        return QualityResult(
            gate_id=gate.id,
            metric_type=gate.metric_type,
            value=health_score,
            level=level,
            passed=passed,
            message=f"依存関係健全性: {health_score:.1f}%",
            details=dependency_data,
            recommendations=recommendations,
        )

    async def _evaluate_security_gate(self, gate: QualityGate) -> QualityResult:
        """セキュリティゲートを評価"""
        try:
            # Bandit セキュリティスキャンを実行
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json", "-q"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                security_score = 100.0  # 問題なし
                issues = []
            else:
                try:
                    bandit_output = json.loads(result.stdout)
                    issues = bandit_output.get("results", [])

                    # 重要度別にスコア計算
                    high_issues = len(
                        [i for i in issues if i.get("issue_severity") == "HIGH"]
                    )
                    medium_issues = len(
                        [i for i in issues if i.get("issue_severity") == "MEDIUM"]
                    )
                    low_issues = len(
                        [i for i in issues if i.get("issue_severity") == "LOW"]
                    )

                    security_score = 100 - (
                        high_issues * 20 + medium_issues * 10 + low_issues * 2
                    )
                    security_score = max(0, security_score)

                except json.JSONDecodeError:
                    security_score = 50.0
                    issues = []

            level = self._determine_quality_level(security_score, gate)
            passed = security_score >= gate.threshold_acceptable

            recommendations = []
            if security_score < gate.threshold_acceptable:
                recommendations.extend(
                    [
                        f"セキュリティスコア {security_score:.1f} が低下しています。",
                        f"検出されたセキュリティ問題: {len(issues)}個",
                        "セキュリティ脆弱性の修正を優先してください。",
                    ]
                )

            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=security_score,
                level=level,
                passed=passed,
                message=f"セキュリティスコア: {security_score:.1f}%",
                details={"issues": issues[:10], "total_issues": len(issues)},
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=0.0,
                level=QualityLevel.CRITICAL,
                passed=False,
                message=f"セキュリティチェックエラー: {str(e)}",
            )

    async def _evaluate_typing_gate(self, gate: QualityGate) -> QualityResult:
        """型チェックゲートを評価"""
        try:
            # MyPy型チェックを実行
            result = subprocess.run(
                ["mypy", "src/", "--ignore-missing-imports", "--follow-imports=silent"],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                typing_score = 100.0  # エラーなし
                error_count = 0
            else:
                # エラー数をカウント
                error_lines = result.stdout.count(": error:")
                warning_lines = result.stdout.count(": warning:")
                error_count = error_lines + warning_lines

                # スコア計算（エラーが多いほど低下）
                typing_score = max(0, 100 - (error_lines * 5 + warning_lines * 2))

            level = self._determine_quality_level(typing_score, gate)
            passed = typing_score >= gate.threshold_acceptable

            recommendations = []
            if typing_score < gate.threshold_acceptable:
                recommendations.extend(
                    [
                        f"型チェックスコア {typing_score:.1f} が低下しています。",
                        f"型エラー数: {error_count}個",
                        "型注釈を追加して型安全性を向上させてください。",
                    ]
                )

            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=typing_score,
                level=level,
                passed=passed,
                message=f"型チェックスコア: {typing_score:.1f}%",
                details={
                    "error_count": error_count,
                    "mypy_output": result.stdout[:1000],
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
                message=f"型チェックエラー: {str(e)}",
            )

    def _determine_quality_level(
        self, value: float, gate: QualityGate, inverse: bool = False
    ) -> QualityLevel:
        """品質レベルを判定
        
        評価値と閾値を比較して品質レベルを判定する。
        
        Args:
            value: 評価値
            gate: 品質ゲート定義
            inverse: 逆評価フラグ（複雑度など低い方が良い場合）
            
        Returns:
            判定された品質レベル
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