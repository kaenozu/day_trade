#!/usr/bin/env python3
"""
品質ゲートシステム - コード複雑度評価器

コード複雑度メトリクスの評価を行うモジュール。
McCabe複雑度やハルステッドメトリクスを分析し、
品質判定を実施する。
"""

from typing import List

from ..complexity_analyzer import CodeComplexityAnalyzer
from ..types import QualityGate, QualityResult
from .base_evaluator import BaseEvaluator


class ComplexityEvaluator(BaseEvaluator):
    """コード複雑度評価器
    
    コード複雑度を評価し、品質ゲートの
    パス/フェイル判定を行う。
    """

    def __init__(self, project_root: str = "."):
        """初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        super().__init__(project_root)
        self.complexity_analyzer = CodeComplexityAnalyzer()

    async def evaluate(self, gate: QualityGate) -> QualityResult:
        """複雑度ゲートを評価
        
        プロジェクト内のPythonファイルの複雑度を分析し、
        平均複雑度による品質判定を実行する。
        
        Args:
            gate: 評価対象の複雑度ゲート
            
        Returns:
            複雑度評価結果
        """
        try:
            # Python ファイルを収集
            python_files = list(self.project_root.glob("src/**/*.py"))
            
            complexity_stats = self._analyze_project_complexity(python_files)
            
            average_complexity = complexity_stats["average_complexity"]
            
            # 品質レベル判定（複雑度は低い方が良い）
            level = self._determine_quality_level(
                average_complexity, gate, inverse=True
            )
            passed = average_complexity <= gate.threshold_acceptable

            # 推奨事項生成
            recommendations = self._generate_complexity_recommendations(
                complexity_stats, gate
            )

            return QualityResult(
                gate_id=gate.id,
                metric_type=gate.metric_type,
                value=average_complexity,
                level=level,
                passed=passed,
                message=f"平均複雑度: {average_complexity:.1f}",
                details=complexity_stats,
                recommendations=recommendations,
            )

        except Exception as e:
            return self._create_error_result(gate, e)

    def _analyze_project_complexity(self, python_files: List) -> dict:
        """プロジェクト全体の複雑度を分析
        
        Args:
            python_files: 分析対象のPythonファイルリスト
            
        Returns:
            複雑度統計データ
        """
        total_complexity = 0
        file_count = 0
        high_complexity_files = []
        complexity_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for file_path in python_files:
            # テストファイルや例サンプルファイルをスキップ
            if self._should_skip_file(str(file_path)):
                continue

            complexity_data = self.complexity_analyzer.analyze_file(str(file_path))

            if "error" not in complexity_data:
                complexity = complexity_data["mccabe_complexity"]
                total_complexity += complexity
                file_count += 1

                # 複雑度分類
                if complexity > 30:
                    complexity_distribution["critical"] += 1
                    high_complexity_files.append({
                        "file": str(file_path),
                        "complexity": complexity,
                        "category": "critical"
                    })
                elif complexity > 20:
                    complexity_distribution["high"] += 1
                    high_complexity_files.append({
                        "file": str(file_path),
                        "complexity": complexity,
                        "category": "high"
                    })
                elif complexity > 10:
                    complexity_distribution["medium"] += 1
                else:
                    complexity_distribution["low"] += 1

        average_complexity = total_complexity / file_count if file_count > 0 else 0

        return {
            "total_files": file_count,
            "average_complexity": average_complexity,
            "high_complexity_files": sorted(
                high_complexity_files, key=lambda x: x["complexity"], reverse=True
            )[:10],  # 上位10件
            "complexity_distribution": complexity_distribution,
            "total_complexity": total_complexity,
        }

    def _should_skip_file(self, file_path: str) -> bool:
        """ファイルを分析対象からスキップするかを判定
        
        Args:
            file_path: ファイルパス
            
        Returns:
            スキップするかどうか
        """
        file_path_lower = file_path.lower()
        skip_patterns = [
            "test", "example", "sample", "__pycache__",
            ".pyc", "conftest.py", "setup.py"
        ]
        
        return any(pattern in file_path_lower for pattern in skip_patterns)

    def _generate_complexity_recommendations(
        self, complexity_stats: dict, gate: QualityGate
    ) -> List[str]:
        """複雑度改善推奨事項を生成
        
        Args:
            complexity_stats: 複雑度統計データ
            gate: 品質ゲート定義
            
        Returns:
            推奨事項のリスト
        """
        recommendations = []
        average_complexity = complexity_stats["average_complexity"]
        high_complexity_files = complexity_stats["high_complexity_files"]
        distribution = complexity_stats["complexity_distribution"]

        # 平均複雑度が高い場合
        if average_complexity > gate.threshold_acceptable:
            recommendations.extend([
                f"平均複雑度 {average_complexity:.1f} が目標値 {gate.threshold_acceptable} を超えています。",
                "複雑な関数やクラスのリファクタリングを検討してください。",
            ])

        # 高複雑度ファイルの詳細情報
        if high_complexity_files:
            critical_files = [f for f in high_complexity_files if f.get("category") == "critical"]
            high_files = [f for f in high_complexity_files if f.get("category") == "high"]
            
            if critical_files:
                recommendations.append(f"危険レベル（30+）の複雑度ファイル: {len(critical_files)}個")
                recommendations.append("最優先でリファクタリングが必要です。")
            
            if high_files:
                recommendations.append(f"高複雑度（20-30）ファイル: {len(high_files)}個")
                recommendations.append("リファクタリングを検討してください。")

        # 具体的な改善手法
        if average_complexity > gate.threshold_good:
            recommendations.extend([
                "以下の手法で複雑度を削減できます:",
                "・長い関数を小さな関数に分割",
                "・ネストしたif文の削減",
                "・早期リターンパターンの使用",
                "・戦略パターンの適用",
            ])

        # 分布に基づく推奨事項
        total_files = sum(distribution.values())
        if total_files > 0:
            high_ratio = (distribution["high"] + distribution["critical"]) / total_files
            if high_ratio > 0.2:  # 20%以上が高複雑度
                recommendations.append(f"高複雑度ファイルの割合が{high_ratio*100:.1f}%と高めです。")

        return recommendations[:8]  # 最大8個の推奨事項に制限