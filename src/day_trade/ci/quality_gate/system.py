#!/usr/bin/env python3
"""
品質ゲートシステム - メインシステム（簡略版）

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
from .system_helpers import (
    calculate_overall_score,
    create_quality_report_summary,
    define_quality_gates,
    determine_overall_level,
    generate_recommendations,
)
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
        self.quality_gates = define_quality_gates()

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

        # 総合評価計算（ヘルパー使用）
        overall_score = calculate_overall_score(gate_results, self.quality_gates)
        overall_level = determine_overall_level(overall_score, gate_results, self.quality_gates)

        # 推奨事項生成（ヘルパー使用）
        recommendations = generate_recommendations(gate_results, self.quality_gates)

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
            "summary": create_quality_report_summary(gate_results),
        }

        # データベースに保存
        await self.database.save_quality_report(report)

        print(
            f"[完了] 品質チェック完了 - 総合スコア: {overall_score:.1f} ({overall_level.value})"
        )

        return report

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

    def get_latest_report(self) -> Dict[str, Any]:
        """最新の品質レポートを取得
        
        Returns:
            最新の品質レポート
        """
        return self.database.get_latest_report()

    def get_quality_statistics(self) -> Dict[str, Any]:
        """品質統計情報を取得
        
        Returns:
            品質統計データ
        """
        return self.database.get_statistics()

    def optimize_database(self):
        """データベースを最適化"""
        self.database.optimize_database()

    def backup_database(self, backup_path: str):
        """データベースをバックアップ
        
        Args:
            backup_path: バックアップファイルのパス
        """
        self.database.backup_database(backup_path)

    def validate_configuration(self) -> List[str]:
        """設定の妥当性を検証
        
        Returns:
            検証エラーメッセージのリスト
        """
        from .system_helpers import validate_quality_gates
        return validate_quality_gates(self.quality_gates)

    def update_quality_gates(self, new_gates: List[QualityGate]):
        """品質ゲート設定を更新
        
        Args:
            new_gates: 新しい品質ゲート設定
        """
        validation_errors = self.validate_configuration()
        if validation_errors:
            raise ValueError(f"品質ゲート設定が無効です: {validation_errors}")
        
        self.quality_gates = new_gates

    def get_quality_gates_info(self) -> List[Dict[str, Any]]:
        """品質ゲート情報を取得
        
        Returns:
            品質ゲート情報のリスト
        """
        return [
            {
                "id": gate.id,
                "name": gate.name,
                "description": gate.description,
                "metric_type": gate.metric_type.value,
                "thresholds": {
                    "excellent": gate.threshold_excellent,
                    "good": gate.threshold_good,
                    "acceptable": gate.threshold_acceptable,
                },
                "weight": gate.weight,
                "mandatory": gate.mandatory,
                "enabled": gate.enabled,
            }
            for gate in self.quality_gates
        ]