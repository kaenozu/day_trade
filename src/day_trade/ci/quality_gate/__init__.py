#!/usr/bin/env python3
"""
品質ゲートシステム - パッケージ初期化

高度品質ゲートシステムのメインパッケージ。
バックワード互換性を維持しながら、モジュール化された
各コンポーネントへのアクセスを提供する。
"""

# バックワード互換性のための主要クラスのインポート
from .complexity_analyzer import CodeComplexityAnalyzer, ComplexityVisitor, HalsteadVisitor
from .coverage_analyzer import TestCoverageAnalyzer
from .database import QualityDatabase
from .dependency_checker import DependencyHealthChecker
from .enums import QualityLevel, QualityMetric
from .gate_evaluator import QualityGateEvaluator
from .report_generator import QualityReportGenerator
from .system import AdvancedQualityGateSystem
from .types import FileQualityReport, QualityGate, QualityResult

# 元ファイルとの互換性のための別名
AdvancedQualityGateSystem = AdvancedQualityGateSystem

# パッケージメタデータ
__version__ = "1.0.0"
__author__ = "Day Trade System"
__description__ = "Enterprise-level code quality gate system"

# 公開API
__all__ = [
    # メインシステム
    "AdvancedQualityGateSystem",
    
    # コアコンポーネント
    "QualityGateEvaluator",
    "QualityReportGenerator", 
    "QualityDatabase",
    
    # 解析器
    "CodeComplexityAnalyzer",
    "TestCoverageAnalyzer", 
    "DependencyHealthChecker",
    
    # データ型
    "QualityGate",
    "QualityResult",
    "FileQualityReport",
    
    # Enum
    "QualityLevel",
    "QualityMetric",
    
    # AST訪問者
    "ComplexityVisitor",
    "HalsteadVisitor",
]

# ファクトリー関数（便利関数）
def create_quality_system(project_root: str = ".") -> AdvancedQualityGateSystem:
    """品質ゲートシステムのインスタンスを作成
    
    Args:
        project_root: プロジェクトルートディレクトリ
        
    Returns:
        設定済みの品質ゲートシステムインスタンス
    """
    return AdvancedQualityGateSystem(project_root)


def get_default_quality_gates() -> list:
    """デフォルトの品質ゲート設定を取得
    
    Returns:
        デフォルト品質ゲートのリスト
    """
    system = AdvancedQualityGateSystem()
    return system.quality_gates


# 便利関数：元のファイルとの互換性
def main():
    """メイン実行関数（互換性維持）
    
    元のadvanced_quality_gate_system.pyのmain()関数との互換性を保つ。
    """
    import argparse
    import asyncio
    import json
    import sys

    parser = argparse.ArgumentParser(description="Advanced Quality Gate System")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", help="Output report file path")
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown", 
        help="Output format",
    )

    args = parser.parse_args()

    async def run_check():
        # 品質チェック実行
        quality_system = AdvancedQualityGateSystem(args.project_root)
        report = await quality_system.run_comprehensive_quality_check()

        # レポート出力
        if args.format == "json":
            output = json.dumps(report, default=str, indent=2, ensure_ascii=False)
        else:
            output = quality_system.generate_ci_report(report)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"📝 Report saved to: {args.output}")
        else:
            print(output)

        # 終了コード設定
        if report["overall_level"] in ["critical", "needs_improvement"]:
            sys.exit(1)
        else:
            sys.exit(0)

    asyncio.run(run_check())


# ドキュメント用のモジュール使用例
def example_usage():
    """
    使用例：
    
    ```python
    import asyncio
    from day_trade.ci.quality_gate import AdvancedQualityGateSystem
    
    async def check_quality():
        system = AdvancedQualityGateSystem()
        report = await system.run_comprehensive_quality_check()
        
        print(f"総合スコア: {report['overall_score']}")
        print(f"品質レベル: {report['overall_level']}")
        
        # CI用レポート生成
        ci_report = system.generate_ci_report(report)
        print(ci_report)
    
    # 実行
    asyncio.run(check_quality())
    ```
    """
    pass