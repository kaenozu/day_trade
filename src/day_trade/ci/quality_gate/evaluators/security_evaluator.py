#!/usr/bin/env python3
"""
品質ゲートシステム - セキュリティ評価器

静的セキュリティ分析の評価を行うモジュール。
Banditを使用してセキュリティ脆弱性を検出し、
品質判定を実施する。
"""

import json
import subprocess
from typing import List

from ..types import QualityGate, QualityResult
from .base_evaluator import BaseEvaluator


class SecurityEvaluator(BaseEvaluator):
    """セキュリティ評価器
    
    静的セキュリティ分析を実行し、品質ゲートの
    パス/フェイル判定を行う。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        super().__init__(project_root)

    async def evaluate(self, gate: QualityGate) -> QualityResult:
        """セキュリティゲートを評価
        
        Banditを使用してセキュリティスキャンを実行し、
        セキュリティスコアによる品質判定を実行する。
        
        Args:
            gate: 評価対象のセキュリティゲート
            
        Returns:
            セキュリティ評価結果
        """
        try:
            # Bandit セキュリティスキャンを実行
            security_results = self._run_bandit_scan()
            security_score = security_results["security_score"]
            issues = security_results["issues"]

            # 品質レベル判定
            level = self._determine_quality_level(security_score, gate)
            passed = security_score >= gate.threshold_acceptable

            # 推奨事項生成
            recommendations = self._generate_security_recommendations(
                security_results, gate
            )

            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=security_score,
                level=level,
                passed=passed,
                message=f"セキュリティスコア: {security_score:.1f}%",
                details={
                    "security_score": security_score,
                    "issues": issues[:10],  # 最初の10件のみ詳細に含める
                    "total_issues": len(issues),
                    "issue_summary": self._summarize_issues(issues),
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return self._create_error_result(gate, e)

    def _run_bandit_scan(self) -> dict:
        """Banditセキュリティスキャンを実行
        
        Returns:
            セキュリティスキャン結果
        """
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json", "-q"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                # 問題なし
                return {
                    "security_score": 100.0,
                    "issues": [],
                    "bandit_output": "No security issues found",
                }
            else:
                # セキュリティ問題を検出
                try:
                    bandit_output = json.loads(result.stdout)
                    issues = bandit_output.get("results", [])
                    
                    # セキュリティスコア計算
                    security_score = self._calculate_security_score(issues)
                    
                    return {
                        "security_score": security_score,
                        "issues": issues,
                        "bandit_output": result.stdout,
                    }
                except json.JSONDecodeError:
                    # JSONパースエラー
                    return {
                        "security_score": 50.0,
                        "issues": [],
                        "bandit_output": result.stdout,
                        "parse_error": "Bandit出力の解析に失敗しました",
                    }

        except subprocess.TimeoutExpired:
            return {
                "security_score": 0.0,
                "issues": [],
                "error": "Banditスキャンがタイムアウトしました",
            }
        except FileNotFoundError:
            return {
                "security_score": 50.0,
                "issues": [],
                "error": "Banditが見つかりません。pip install bandit で インストールしてください。",
            }

    def _calculate_security_score(self, issues: List[dict]) -> float:
        """セキュリティスコアを計算
        
        Args:
            issues: セキュリティ問題のリスト
            
        Returns:
            0-100のセキュリティスコア
        """
        if not issues:
            return 100.0

        # 重要度別にイシューをカウント
        high_issues = len([i for i in issues if i.get("issue_severity") == "HIGH"])
        medium_issues = len([i for i in issues if i.get("issue_severity") == "MEDIUM"])
        low_issues = len([i for i in issues if i.get("issue_severity") == "LOW"])

        # スコア計算（重要度に応じて減点）
        score_reduction = (high_issues * 25) + (medium_issues * 10) + (low_issues * 2)
        
        security_score = max(0, 100 - score_reduction)
        return security_score

    def _summarize_issues(self, issues: List[dict]) -> dict:
        """セキュリティ問題の概要を作成
        
        Args:
            issues: セキュリティ問題のリスト
            
        Returns:
            問題の概要
        """
        summary = {
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "issue_types": {},
            "affected_files": set(),
        }

        for issue in issues:
            severity = issue.get("issue_severity", "LOW")
            test_id = issue.get("test_id", "Unknown")
            filename = issue.get("filename", "Unknown")

            # 重要度カウント
            if severity == "HIGH":
                summary["high_severity"] += 1
            elif severity == "MEDIUM":
                summary["medium_severity"] += 1
            else:
                summary["low_severity"] += 1

            # 問題タイプカウント
            if test_id in summary["issue_types"]:
                summary["issue_types"][test_id] += 1
            else:
                summary["issue_types"][test_id] = 1

            # 影響ファイル
            summary["affected_files"].add(filename)

        # setをlistに変換（JSON化のため）
        summary["affected_files"] = list(summary["affected_files"])
        summary["total_affected_files"] = len(summary["affected_files"])

        return summary

    def _generate_security_recommendations(
        self, security_results: dict, gate: QualityGate
    ) -> List[str]:
        """セキュリティ改善推奨事項を生成
        
        Args:
            security_results: セキュリティスキャン結果
            gate: 品質ゲート定義
            
        Returns:
            推奨事項のリスト
        """
        recommendations = []
        security_score = security_results["security_score"]
        issues = security_results["issues"]

        # エラーがある場合
        if "error" in security_results:
            recommendations.append(f"スキャンエラー: {security_results['error']}")
            recommendations.append("セキュリティスキャンツールの設定を確認してください。")
            return recommendations

        # セキュリティスコアが低い場合
        if security_score < gate.threshold_acceptable:
            recommendations.extend([
                f"セキュリティスコア {security_score:.1f}% が目標値 {gate.threshold_acceptable}% を下回っています。",
                f"検出されたセキュリティ問題: {len(issues)}個",
                "セキュリティ脆弱性の修正を優先してください。",
            ])

        if issues:
            issue_summary = self._summarize_issues(issues)
            
            # 重要度別の推奨事項
            if issue_summary["high_severity"] > 0:
                recommendations.extend([
                    f"🚨 高リスク問題: {issue_summary['high_severity']}個",
                    "即座に対応が必要です。",
                ])

            if issue_summary["medium_severity"] > 0:
                recommendations.append(f"⚠️ 中リスク問題: {issue_summary['medium_severity']}個")

            if issue_summary["low_severity"] > 0:
                recommendations.append(f"ℹ️ 低リスク問題: {issue_summary['low_severity']}個")

            # よくある問題タイプへの対応策
            issue_types = issue_summary.get("issue_types", {})
            common_fixes = {
                "B101": "assert文の使用を避け、適切な例外処理を実装してください。",
                "B102": "exec()の使用を避け、より安全な方法を検討してください。",
                "B103": "ファイルパーミッションを適切に設定してください。",
                "B105": "ハードコードされた文字列をパスワードとして使用しないでください。",
                "B106": "ハードコードされたパスワードを削除してください。",
                "B107": "ハードコードされたパスワードを環境変数に移してください。",
                "B201": "Flaskのdebug=Trueを本番環境で使用しないでください。",
                "B601": "shell=Trueの使用を避けてください。",
                "B602": "subprocess.Popenでshell=Trueを使用しないでください。",
            }

            for issue_type, count in list(issue_types.items())[:3]:  # 上位3種類
                if issue_type in common_fixes:
                    recommendations.append(f"{issue_type} ({count}個): {common_fixes[issue_type]}")

            # 一般的なセキュリティ対策
            if security_score < gate.threshold_good:
                recommendations.extend([
                    "セキュリティ強化の推奨事項:",
                    "・入力値の適切な検証とサニタイゼーション",
                    "・機密情報の環境変数への移行",
                    "・セキュアなライブラリの使用",
                    "・定期的なセキュリティスキャンの実施",
                ])

        # 影響ファイル数が多い場合
        if len(security_results.get("issues", [])) > 0:
            affected_files = len(set([i.get("filename", "") for i in issues]))
            if affected_files > 10:
                recommendations.append(f"影響ファイル数: {affected_files}個")
                recommendations.append("段階的な修正計画を立てることを推奨します。")

        return recommendations[:10]  # 最大10個の推奨事項に制限