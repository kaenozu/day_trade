#!/usr/bin/env python3
"""
品質ゲートシステム - 依存関係評価器

依存関係の健全性評価を行うモジュール。
脆弱性チェック、ライセンス適合性、
更新状況を分析し、品質判定を実施する。
"""

from typing import List

from ..dependency_checker import DependencyHealthChecker
from ..types import QualityGate, QualityResult
from .base_evaluator import BaseEvaluator


class DependencyEvaluator(BaseEvaluator):
    """依存関係評価器
    
    依存関係の健全性を評価し、品質ゲートの
    パス/フェイル判定を行う。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        super().__init__(project_root)
        self.dependency_checker = DependencyHealthChecker()

    async def evaluate(self, gate: QualityGate) -> QualityResult:
        """依存関係ゲートを評価
        
        依存関係の脆弱性、ライセンス、更新状況を分析し、
        健全性スコアによる品質判定を実行する。
        
        Args:
            gate: 評価対象の依存関係ゲート
            
        Returns:
            依存関係評価結果
        """
        try:
            # 依存関係健全性データを取得
            dependency_data = self.dependency_checker.check_dependencies()
            health_score = dependency_data.get("health_score", 50.0)

            # 品質レベル判定
            level = self._determine_quality_level(health_score, gate)
            passed = health_score >= gate.threshold_acceptable

            # 推奨事項生成
            recommendations = self._generate_dependency_recommendations(
                dependency_data, gate
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

        except Exception as e:
            return self._create_error_result(gate, e)

    def _generate_dependency_recommendations(
        self, dependency_data: dict, gate: QualityGate
    ) -> List[str]:
        """依存関係改善推奨事項を生成
        
        Args:
            dependency_data: 依存関係データ
            gate: 品質ゲート定義
            
        Returns:
            推奨事項のリスト
        """
        recommendations = []
        health_score = dependency_data.get("health_score", 50.0)
        vulnerabilities = dependency_data.get("vulnerabilities", [])
        outdated_packages = dependency_data.get("outdated_packages", [])
        license_results = dependency_data.get("licenses", {})

        # 健全性スコアが低い場合
        if health_score < gate.threshold_acceptable:
            recommendations.extend([
                f"依存関係健全性スコア {health_score:.1f}% が目標値 {gate.threshold_acceptable}% を下回っています。",
                "依存関係の更新とセキュリティパッチ適用を実施してください。",
            ])

        # 脆弱性関連の推奨事項
        if vulnerabilities:
            critical_vulns = [v for v in vulnerabilities if v.get("severity") == "critical"]
            high_vulns = [v for v in vulnerabilities if v.get("severity") == "high"]
            medium_vulns = [v for v in vulnerabilities if v.get("severity") == "medium"]

            if critical_vulns:
                recommendations.extend([
                    f"緊急: 重大な脆弱性が{len(critical_vulns)}個検出されました。",
                    "最優先で対応してください。",
                ])

            if high_vulns:
                recommendations.append(f"高リスクの脆弱性が{len(high_vulns)}個あります。")

            if medium_vulns:
                recommendations.append(f"中リスクの脆弱性が{len(medium_vulns)}個あります。")

            # 具体的な対応策
            recommendations.extend([
                "以下のコマンドで脆弱性を確認してください:",
                "  pip-audit --fix",
                "  pip install --upgrade [パッケージ名]",
            ])

        # 古くなったパッケージの推奨事項
        if outdated_packages:
            recommendations.append(f"更新が必要なパッケージ: {len(outdated_packages)}個")
            
            if len(outdated_packages) > 10:
                recommendations.append("多数のパッケージが古くなっています。")
                recommendations.append("定期的な依存関係更新スケジュールを検討してください。")
            
            recommendations.extend([
                "以下のコマンドで更新できます:",
                "  pip list --outdated",
                "  pip install --upgrade [パッケージ名]",
            ])

        # ライセンス問題の推奨事項
        if license_results:
            compliance_score = license_results.get("compliance_score", 100)
            license_issues = license_results.get("license_issues", [])
            
            if compliance_score < 90:
                recommendations.append(f"ライセンス適合性スコア: {compliance_score}%")
            
            if license_issues:
                recommendations.extend([
                    f"ライセンス問題のあるパッケージ: {len(license_issues)}個",
                    "法務チームと相談してください。",
                ])
                
                # 具体的な問題パッケージ
                for issue in license_issues[:3]:  # 最初の3個のみ表示
                    pkg_name = issue.get("package", "不明")
                    license_name = issue.get("license", "不明")
                    recommendations.append(f"  {pkg_name}: {license_name}")

        # 依存関係グラフ関連の推奨事項
        dependency_graph = dependency_data.get("dependency_graph", {})
        if dependency_graph:
            max_depth = dependency_graph.get("max_depth", 0)
            circular_deps = dependency_graph.get("circular_dependencies", [])
            
            if max_depth > 8:
                recommendations.append(f"依存関係が深すぎます（深度: {max_depth}）。")
                recommendations.append("依存関係の整理を検討してください。")
            
            if circular_deps:
                recommendations.extend([
                    f"循環依存が{len(circular_deps)}個検出されました。",
                    "循環依存の解消が必要です。",
                ])

        # 一般的な改善策
        if health_score < gate.threshold_good:
            recommendations.extend([
                "依存関係管理のベストプラクティス:",
                "・定期的な依存関係の棚卸し",
                "・セキュリティスキャンの自動化",
                "・requirements.txtの適切な管理",
                "・開発/本番環境の依存関係分離",
            ])

        return recommendations[:10]  # 最大10個の推奨事項に制限