#!/usr/bin/env python3
"""
品質ゲートシステム - メインシステム

高度品質ゲートシステムのメインクラス。
各種コンポーネントを統合し、包括的な品質評価を実行する。
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .database import QualityDatabase
from .enums import QualityLevel, QualityMetric
from .gate_evaluator import QualityGateEvaluator
from .report_generator import QualityReportGenerator
from .types import QualityGate, QualityResult


class AdvancedQualityGateSystem:
    """高度品質ゲートシステム
    
    エンタープライズレベルのコード品質管理システム。
    包括的品質分析、CI/CD統合、レポート生成機能を提供する。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = Path(project_root)

        # コンポーネント初期化
        self.gate_evaluator = QualityGateEvaluator(project_root)
        self.report_generator = QualityReportGenerator()
        self.database = QualityDatabase(project_root)

        # 品質ゲート定義
        self.quality_gates = self._define_quality_gates()

    def _define_quality_gates(self) -> List[QualityGate]:
        """品質ゲートを定義
        
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

    async def run_comprehensive_quality_check(self) -> Dict[str, Any]:
        """包括的品質チェックを実行
        
        全ての品質ゲートを順次評価し、総合的な品質レポートを生成する。
        
        Returns:
            包括的品質評価結果
        """
        report_id = f"quality_check_{int(time.time())}"
        timestamp = datetime.now(timezone.utc)

        print("[検査] 包括的品質チェックを開始します...")

        # 各品質ゲートをチェック
        gate_results = []
        file_reports = []

        for gate in self.quality_gates:
            if not gate.enabled:
                continue

            print(f"  [チェック] {gate.name}をチェック中...")

            try:
                result = await self.gate_evaluator.evaluate_gate(gate)
                gate_results.append(result)

                # ファイル別レポートの収集（複雑度チェック時のみ）
                if gate.metric_type == QualityMetric.COMPLEXITY:
                    file_reports = self.report_generator.collect_file_quality_reports(
                        self.gate_evaluator.complexity_analyzer, self.project_root
                    )

            except Exception as e:
                print(f"  [エラー] {gate.name}の評価中にエラー: {e}")
                gate_results.append(
                    QualityResult(
                        gate_id=gate.id,
                        metric_type=gate.metric_type,
                        value=0.0,
                        level=QualityLevel.CRITICAL,
                        passed=False,
                        message=f"評価エラー: {str(e)}",
                    )
                )

        # 総合評価計算
        overall_score = self._calculate_overall_score(gate_results)
        overall_level = self._determine_overall_level(overall_score, gate_results)

        # 推奨事項生成
        recommendations = self._generate_recommendations(gate_results)

        # レポート作成
        report = {
            "report_id": report_id,
            "timestamp": timestamp.isoformat(),
            "overall_score": overall_score,
            "overall_level": overall_level.value,
            "gates_passed": len([r for r in gate_results if r.passed]),
            "gates_total": len(gate_results),
            "gate_results": gate_results,
            "file_reports": file_reports,
            "recommendations": recommendations,
            "summary": {
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
            },
        }

        # データベースに保存
        await self.database.save_quality_report(report)

        print(
            f"[完了] 品質チェック完了 - 総合スコア: {overall_score:.1f} ({overall_level.value})"
        )

        return report

    def _calculate_overall_score(self, results: List[QualityResult]) -> float:
        """総合スコアを計算
        
        各品質ゲートの結果を重み付き平均して総合スコアを算出。
        
        Args:
            results: 品質評価結果のリスト
            
        Returns:
            0-100の総合スコア
        """
        if not results:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        gate_dict = {gate.id: gate for gate in self.quality_gates}

        for result in results:
            gate = gate_dict.get(result.gate_id)
            if gate:
                weight = gate.weight
                weighted_sum += result.value * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_overall_level(
        self, overall_score: float, results: List[QualityResult]
    ) -> QualityLevel:
        """総合品質レベルを判定
        
        総合スコアと必須ゲートの状況を考慮して総合品質レベルを決定。
        
        Args:
            overall_score: 総合スコア
            results: 品質評価結果のリスト
            
        Returns:
            総合品質レベル
        """
        # 必須ゲートで失敗があるかチェック
        gate_dict = {gate.id: gate for gate in self.quality_gates}

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

    def _generate_recommendations(self, results: List[QualityResult]) -> List[str]:
        """総合推奨事項を生成
        
        各品質ゲートの結果から改善推奨事項を生成する。
        
        Args:
            results: 品質評価結果のリスト
            
        Returns:
            推奨事項のリスト
        """
        all_recommendations = []

        # 各ゲートの推奨事項を収集
        for result in results:
            all_recommendations.extend(result.recommendations)

        # 全体的な推奨事項を追加
        failed_mandatory = [
            r for r in results if not r.passed and self._is_mandatory_gate(r.gate_id)
        ]

        if failed_mandatory:
            all_recommendations.insert(
                0, "必須品質ゲートで失敗があります。最優先で修正してください。"
            )

        # 重複を除去し、優先順位でソート
        unique_recommendations = list(dict.fromkeys(all_recommendations))

        return unique_recommendations[:10]  # 上位10件に制限

    def _is_mandatory_gate(self, gate_id: str) -> bool:
        """必須ゲートかどうかを判定
        
        Args:
            gate_id: ゲートID
            
        Returns:
            必須ゲートかどうか
        """
        gate = next((g for g in self.quality_gates if g.id == gate_id), None)
        return gate.mandatory if gate else False

    def get_quality_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """品質履歴を取得
        
        Args:
            days: 取得する日数
            
        Returns:
            品質履歴のリスト
        """
        return self.database.get_quality_history(days)

    def get_metric_trend(self, metric_type: str, days: int = 90) -> List[Dict[str, Any]]:
        """メトリクス推移を取得
        
        Args:
            metric_type: メトリクス種類
            days: 取得する日数
            
        Returns:
            メトリクス推移データ
        """
        return self.database.get_metric_trend(metric_type, days)

    def generate_ci_report(self, report: Dict[str, Any]) -> str:
        """CI/CD用レポートを生成
        
        Args:
            report: 品質評価レポート
            
        Returns:
            CI/CD用マークダウンレポート
        """
        return self.report_generator.generate_ci_report(report)

    def generate_detailed_report(self, report: Dict[str, Any]) -> str:
        """詳細レポートを生成
        
        Args:
            report: 品質評価レポート
            
        Returns:
            詳細マークダウンレポート
        """
        return self.report_generator.generate_detailed_report(report)

    def cleanup_old_data(self, keep_days: int = 90):
        """古いデータをクリーンアップ
        
        Args:
            keep_days: 保持する日数
        """
        self.database.cleanup_old_reports(keep_days)