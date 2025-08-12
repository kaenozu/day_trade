#!/usr/bin/env python3
"""
DataFrame使用状況・パフォーマンス分析ツール
Issue #378: データI/O・データ処理最適化

既存システムのDataFrame使用量・データ型・メモリ効率を包括分析
- メモリ使用量プロファイリング
- データ型最適化可能性分析
- 非効率操作検出・改善提案
- パフォーマンスボトルネック特定
"""

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil

# ログ設定（フォールバック対応）
try:
    from .logging_config import get_context_logger

    logger = get_context_logger(__name__)
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class DataFrameUsageStats:
    """DataFrame使用統計"""

    file_path: str
    function_name: str
    line_number: int

    # DataFrame基本情報
    creation_type: str  # "pd.DataFrame", "read_csv", "read_sql", etc.
    estimated_columns: int = 0
    estimated_rows: int = 0

    # データ型情報
    data_types: Dict[str, str] = field(default_factory=dict)
    memory_usage_mb: float = 0.0

    # 使用パターン
    operations: List[str] = field(default_factory=list)  # apply, groupby, merge, etc.
    optimization_potential: float = 0.0  # 0-1スコア

    # 最適化提案
    recommended_optimizations: List[str] = field(default_factory=list)
    expected_memory_reduction: float = 0.0  # %
    expected_speed_improvement: float = 0.0  # 倍率


@dataclass
class DataFrameAnalysisReport:
    """DataFrame分析レポート"""

    analyzed_files: int = 0
    total_dataframe_usage: int = 0

    # メモリ使用量統計
    total_estimated_memory_mb: float = 0.0
    potential_memory_savings_mb: float = 0.0
    memory_optimization_ratio: float = 0.0

    # パフォーマンス統計
    inefficient_operations_count: int = 0
    vectorization_opportunities: int = 0

    # 最適化機会
    dtype_optimization_files: List[str] = field(default_factory=list)
    apply_to_vectorization_files: List[str] = field(default_factory=list)
    memory_copy_optimization_files: List[str] = field(default_factory=list)

    # 詳細統計
    usage_stats: List[DataFrameUsageStats] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """レポートを辞書形式に変換"""
        return {
            "analyzed_files": self.analyzed_files,
            "total_dataframe_usage": self.total_dataframe_usage,
            "total_estimated_memory_mb": self.total_estimated_memory_mb,
            "potential_memory_savings_mb": self.potential_memory_savings_mb,
            "memory_optimization_ratio": self.memory_optimization_ratio,
            "inefficient_operations_count": self.inefficient_operations_count,
            "vectorization_opportunities": self.vectorization_opportunities,
            "dtype_optimization_files": self.dtype_optimization_files,
            "apply_to_vectorization_files": self.apply_to_vectorization_files,
            "memory_copy_optimization_files": self.memory_copy_optimization_files,
            "usage_stats": [
                {
                    "file_path": stat.file_path,
                    "function_name": stat.function_name,
                    "line_number": stat.line_number,
                    "creation_type": stat.creation_type,
                    "estimated_memory_mb": stat.memory_usage_mb,
                    "optimization_potential": stat.optimization_potential,
                    "recommended_optimizations": stat.recommended_optimizations,
                    "expected_memory_reduction": stat.expected_memory_reduction,
                    "expected_speed_improvement": stat.expected_speed_improvement,
                }
                for stat in self.usage_stats
            ],
        }


class DataFrameCodeAnalyzer:
    """DataFrame使用コード静的解析器"""

    def __init__(self):
        # DataFrame操作パターン
        self.inefficient_patterns = {
            r"\.apply\(": {"type": "apply_usage", "severity": "medium"},
            r"for\s+.*\s+in\s+.*\.iterrows\(\)": {
                "type": "iterrows_loop",
                "severity": "high",
            },
            r"for\s+.*\s+in\s+.*\.itertuples\(\)": {
                "type": "itertuples_loop",
                "severity": "medium",
            },
            r"pd\.concat\(.*for.*\)": {"type": "concat_in_loop", "severity": "high"},
            r"\.append\(.*\)": {"type": "append_usage", "severity": "medium"},
            r"df\[.*\]\s*=\s*df\[.*\]\.apply\(": {
                "type": "column_apply",
                "severity": "medium",
            },
        }

        self.vectorizable_patterns = {
            r"\.apply\(lambda\s+x:\s*x\s*[+\-*/]\s*\d+": "arithmetic_operation",
            r"\.apply\(lambda\s+x:\s*np\.(abs|sqrt|log|exp)\(x\)\)": "numpy_function",
            r"\.apply\(lambda\s+x:\s*x\s*>\s*\d+": "comparison_operation",
            r"\.apply\(lambda\s+x:\s*x\.str\.": "string_operation",
        }

        self.dtype_optimization_indicators = {
            "pd.read_csv": ["dtype", "parse_dates", "category"],
            "pd.DataFrame": ["dtype"],
            "astype": ["float64", "int64", "object"],
        }

    def analyze_file(self, file_path: Path) -> List[DataFrameUsageStats]:
        """単一ファイル分析"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # AST解析
            tree = ast.parse(content)
            usage_stats = []

            # 行別解析も実行
            lines = content.split("\n")
            usage_stats.extend(self._analyze_lines(file_path, lines))

            # AST訪問
            visitor = DataFrameASTVisitor(file_path, usage_stats)
            visitor.visit(tree)

            return usage_stats

        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"ファイル解析エラー {file_path}: {e}")
            return []

    def _analyze_lines(self, file_path: Path, lines: List[str]) -> List[DataFrameUsageStats]:
        """行別パターン解析"""
        usage_stats = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # DataFrame作成パターン
            df_creation_patterns = [
                (r"pd\.DataFrame\(", "pd.DataFrame"),
                (r"pd\.read_csv\(", "read_csv"),
                (r"pd\.read_sql\(", "read_sql"),
                (r"pd\.read_json\(", "read_json"),
                (r"pd\.read_excel\(", "read_excel"),
            ]

            for pattern, creation_type in df_creation_patterns:
                if re.search(pattern, line):
                    stat = DataFrameUsageStats(
                        file_path=str(file_path),
                        function_name="<module>",  # AST訪問で更新
                        line_number=line_num,
                        creation_type=creation_type,
                    )

                    # 最適化機会検出
                    self._detect_optimization_opportunities(stat, line)
                    usage_stats.append(stat)

            # 非効率操作検出
            for pattern, info in self.inefficient_patterns.items():
                if re.search(pattern, line):
                    # 既存の統計に操作追加、または新規作成
                    if usage_stats:
                        usage_stats[-1].operations.append(info["type"])
                        if info["severity"] == "high":
                            usage_stats[-1].optimization_potential += 0.3
                        elif info["severity"] == "medium":
                            usage_stats[-1].optimization_potential += 0.2

        return usage_stats

    def _detect_optimization_opportunities(self, stat: DataFrameUsageStats, line: str):
        """最適化機会検出"""
        recommendations = []
        memory_reduction = 0.0
        speed_improvement = 1.0

        # データ型最適化
        if "read_csv" in stat.creation_type and "dtype=" not in line:
            recommendations.append("dtype指定でメモリ使用量削減")
            memory_reduction += 30.0  # 推定30%削減
            speed_improvement += 0.5

        if "read_csv" in stat.creation_type and "parse_dates=" not in line:
            recommendations.append("parse_dates指定で日時処理高速化")
            speed_improvement += 0.3

        # float64 → float32最適化
        if "float64" in line or (
            "dtype" not in line and stat.creation_type in ["pd.DataFrame", "read_csv"]
        ):
            recommendations.append("float64→float32でメモリ50%削減")
            memory_reduction += 50.0
            speed_improvement += 0.8

        # category型最適化
        if stat.creation_type in ["read_csv", "pd.DataFrame"]:
            recommendations.append("category型でメモリ大幅削減")
            memory_reduction += 40.0

        stat.recommended_optimizations = recommendations
        stat.expected_memory_reduction = memory_reduction
        stat.expected_speed_improvement = speed_improvement


class DataFrameASTVisitor(ast.NodeVisitor):
    """DataFrame使用AST訪問者"""

    def __init__(self, file_path: Path, usage_stats: List[DataFrameUsageStats]):
        self.file_path = file_path
        self.usage_stats = usage_stats
        self.current_function = "<module>"

    def visit_FunctionDef(self, node):
        """関数定義訪問"""
        old_function = self.current_function
        self.current_function = node.name

        # 関数内の統計に関数名設定
        for stat in self.usage_stats:
            if stat.line_number >= node.lineno and stat.function_name == "<module>":
                stat.function_name = node.name

        self.generic_visit(node)
        self.current_function = old_function

    def visit_Call(self, node):
        """関数呼び出し訪問"""
        # DataFrame操作メソッド検出
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr

            dataframe_methods = [
                "apply",
                "map",
                "applymap",
                "groupby",
                "merge",
                "join",
                "concat",
                "pivot",
                "pivot_table",
                "melt",
                "stack",
                "unstack",
                "sort_values",
                "sort_index",
                "drop_duplicates",
                "fillna",
                "dropna",
                "replace",
                "astype",
            ]

            if method_name in dataframe_methods:
                # 対応する統計項目に操作追加
                for stat in self.usage_stats:
                    if (
                        stat.function_name == self.current_function
                        and abs(stat.line_number - node.lineno) <= 3
                    ):  # 近い行の統計
                        stat.operations.append(method_name)
                        break

        self.generic_visit(node)


class DataFrameRuntimeProfiler:
    """DataFrame実行時プロファイラー"""

    def __init__(self):
        self.profiling_enabled = False
        self.memory_snapshots = []
        self.operation_times = []

    def profile_dataframe_operation(self, operation_name: str = ""):
        """DataFrame操作プロファイリングデコレータ"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return func(*args, **kwargs)

                # メモリ使用量記録
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()

                # メモリ使用量記録
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                operation_info = {
                    "operation": operation_name or func.__name__,
                    "execution_time_ms": (end_time - start_time) * 1000,
                    "memory_before_mb": memory_before,
                    "memory_after_mb": memory_after,
                    "memory_delta_mb": memory_after - memory_before,
                    "timestamp": time.time(),
                }

                self.operation_times.append(operation_info)
                logger.debug(f"DataFrame操作プロファイル: {operation_info}")

                return result

            return wrapper

        return decorator

    def start_profiling(self):
        """プロファイリング開始"""
        self.profiling_enabled = True
        self.memory_snapshots.clear()
        self.operation_times.clear()
        logger.info("DataFrame実行時プロファイリング開始")

    def stop_profiling(self) -> Dict[str, Any]:
        """プロファイリング停止・結果取得"""
        self.profiling_enabled = False

        if not self.operation_times:
            return {"message": "プロファイリングデータなし"}

        # 統計計算
        total_time = sum(op["execution_time_ms"] for op in self.operation_times)
        total_memory_delta = sum(op["memory_delta_mb"] for op in self.operation_times)

        # 最もメモリ使用量が多い操作
        max_memory_op = max(self.operation_times, key=lambda x: x["memory_delta_mb"])

        # 最も時間がかかった操作
        slowest_op = max(self.operation_times, key=lambda x: x["execution_time_ms"])

        results = {
            "profiling_summary": {
                "total_operations": len(self.operation_times),
                "total_execution_time_ms": total_time,
                "average_execution_time_ms": total_time / len(self.operation_times),
                "total_memory_delta_mb": total_memory_delta,
                "max_memory_operation": max_memory_op,
                "slowest_operation": slowest_op,
            },
            "operation_details": self.operation_times,
        }

        logger.info(f"DataFrame実行時プロファイリング完了: {len(self.operation_times)}操作")
        return results


class DataFrameAnalyzer:
    """DataFrame総合分析システム"""

    def __init__(self):
        self.code_analyzer = DataFrameCodeAnalyzer()
        self.runtime_profiler = DataFrameRuntimeProfiler()

    def analyze_project(self, project_root: str = "src") -> DataFrameAnalysisReport:
        """プロジェクト全体のDataFrame使用状況分析"""
        logger.info(f"DataFrame使用状況分析開始: {project_root}")

        project_path = Path(project_root)
        python_files = list(project_path.rglob("*.py"))

        report = DataFrameAnalysisReport()
        report.analyzed_files = len(python_files)

        all_stats = []

        for py_file in python_files:
            try:
                file_stats = self.code_analyzer.analyze_file(py_file)
                all_stats.extend(file_stats)

                # ファイル別最適化機会分類
                if any(stat.expected_memory_reduction > 20 for stat in file_stats):
                    report.dtype_optimization_files.append(str(py_file))

                if any("apply" in stat.operations for stat in file_stats):
                    report.apply_to_vectorization_files.append(str(py_file))

                if any(stat.optimization_potential > 0.5 for stat in file_stats):
                    report.memory_copy_optimization_files.append(str(py_file))

            except Exception as e:
                logger.error(f"ファイル分析エラー {py_file}: {e}")

        report.usage_stats = all_stats
        report.total_dataframe_usage = len(all_stats)

        # 統計計算
        report.total_estimated_memory_mb = sum(stat.memory_usage_mb for stat in all_stats)

        report.potential_memory_savings_mb = sum(
            stat.memory_usage_mb * (stat.expected_memory_reduction / 100) for stat in all_stats
        )

        if report.total_estimated_memory_mb > 0:
            report.memory_optimization_ratio = (
                report.potential_memory_savings_mb / report.total_estimated_memory_mb
            )

        report.inefficient_operations_count = sum(
            len(
                [
                    op
                    for op in stat.operations
                    if op in ["apply_usage", "iterrows_loop", "concat_in_loop"]
                ]
            )
            for stat in all_stats
        )

        report.vectorization_opportunities = len(
            [
                stat
                for stat in all_stats
                if any(op in stat.operations for op in ["apply_usage", "column_apply"])
            ]
        )

        logger.info(
            f"DataFrame分析完了: {report.analyzed_files}ファイル, "
            f"{report.total_dataframe_usage}使用箇所, "
            f"メモリ削減可能性 {report.memory_optimization_ratio:.1%}"
        )

        return report

    def benchmark_dataframe_operations(self, sample_data_rows: int = 100000) -> Dict[str, Any]:
        """DataFrame操作ベンチマーク"""
        logger.info(f"DataFrame操作ベンチマーク開始: {sample_data_rows}行")

        # サンプルデータ生成
        np.random.seed(42)
        sample_data = pd.DataFrame(
            {
                "float_col": np.random.randn(sample_data_rows).astype(np.float64),
                "int_col": np.random.randint(0, 1000, sample_data_rows).astype(np.int64),
                "category_col": np.random.choice(["A", "B", "C", "D", "E"], sample_data_rows),
                "string_col": [f"item_{i % 1000}" for i in range(sample_data_rows)],
            }
        )

        results = {}

        # データ型最適化ベンチマーク
        logger.info("データ型最適化ベンチマーク実行")

        # 元のメモリ使用量
        original_memory = sample_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

        # 最適化: float64 → float32
        start_time = time.perf_counter()
        optimized_float = sample_data.copy()
        optimized_float["float_col"] = optimized_float["float_col"].astype(np.float32)
        float32_time = time.perf_counter() - start_time
        float32_memory = optimized_float.memory_usage(deep=True).sum() / 1024 / 1024

        # 最適化: string → category
        start_time = time.perf_counter()
        optimized_category = sample_data.copy()
        optimized_category["category_col"] = optimized_category["category_col"].astype("category")
        optimized_category["string_col"] = optimized_category["string_col"].astype("category")
        category_time = time.perf_counter() - start_time
        category_memory = optimized_category.memory_usage(deep=True).sum() / 1024 / 1024

        # 完全最適化
        start_time = time.perf_counter()
        fully_optimized = sample_data.copy()
        fully_optimized["float_col"] = fully_optimized["float_col"].astype(np.float32)
        fully_optimized["int_col"] = pd.to_numeric(fully_optimized["int_col"], downcast="integer")
        fully_optimized["category_col"] = fully_optimized["category_col"].astype("category")
        fully_optimized["string_col"] = fully_optimized["string_col"].astype("category")
        full_optimization_time = time.perf_counter() - start_time
        fully_optimized_memory = fully_optimized.memory_usage(deep=True).sum() / 1024 / 1024

        results["dtype_optimization"] = {
            "original_memory_mb": original_memory,
            "float32_optimization": {
                "time_seconds": float32_time,
                "memory_mb": float32_memory,
                "memory_reduction_percent": (1 - float32_memory / original_memory) * 100,
            },
            "category_optimization": {
                "time_seconds": category_time,
                "memory_mb": category_memory,
                "memory_reduction_percent": (1 - category_memory / original_memory) * 100,
            },
            "full_optimization": {
                "time_seconds": full_optimization_time,
                "memory_mb": fully_optimized_memory,
                "memory_reduction_percent": (1 - fully_optimized_memory / original_memory) * 100,
            },
        }

        # ベクトル化操作ベンチマーク
        logger.info("ベクトル化操作ベンチマーク実行")

        # apply vs vectorized 比較
        test_df = sample_data.head(10000)  # 小さなサンプルでテスト

        # apply使用
        start_time = time.perf_counter()
        apply_result = test_df["float_col"].apply(lambda x: x * 2 + 1)
        apply_time = time.perf_counter() - start_time

        # ベクトル化
        start_time = time.perf_counter()
        vectorized_result = test_df["float_col"] * 2 + 1
        vectorized_time = time.perf_counter() - start_time

        results["vectorization_benchmark"] = {
            "apply_time_seconds": apply_time,
            "vectorized_time_seconds": vectorized_time,
            "speedup_ratio": apply_time / vectorized_time if vectorized_time > 0 else 0,
            "results_equal": np.allclose(apply_result, vectorized_result),
        }

        logger.info(
            f"DataFrame操作ベンチマーク完了: "
            f"メモリ削減 {results['dtype_optimization']['full_optimization']['memory_reduction_percent']:.1f}%, "
            f"ベクトル化速度向上 {results['vectorization_benchmark']['speedup_ratio']:.1f}x"
        )

        return results

    def generate_optimization_report(self, report: DataFrameAnalysisReport) -> str:
        """最適化レポート生成"""
        lines = [
            "=" * 80,
            "DataFrame使用状況・最適化分析レポート",
            "Issue #378: データI/O・データ処理最適化",
            "=" * 80,
            "",
            "[分析サマリー]",
            f"  分析ファイル数: {report.analyzed_files}",
            f"  DataFrame使用箇所: {report.total_dataframe_usage}",
            f"  推定メモリ使用量: {report.total_estimated_memory_mb:.1f}MB",
            f"  メモリ削減可能性: {report.potential_memory_savings_mb:.1f}MB ({report.memory_optimization_ratio:.1%})",
            "",
            "[最適化機会]",
            f"  非効率操作: {report.inefficient_operations_count}箇所",
            f"  ベクトル化機会: {report.vectorization_opportunities}箇所",
            f"  データ型最適化対象: {len(report.dtype_optimization_files)}ファイル",
            f"  apply→ベクトル化対象: {len(report.apply_to_vectorization_files)}ファイル",
            f"  メモリコピー最適化対象: {len(report.memory_copy_optimization_files)}ファイル",
            "",
            "[高優先度最適化対象ファイル]",
            "-" * 50,
        ]

        # 最適化対象ファイルの詳細
        high_priority_stats = sorted(
            [stat for stat in report.usage_stats if stat.optimization_potential > 0.3],
            key=lambda x: x.optimization_potential,
            reverse=True,
        )

        for stat in high_priority_stats[:10]:  # 上位10件表示
            lines.extend(
                [
                    "",
                    f"File: {stat.file_path}:{stat.line_number} ({stat.function_name})",
                    f"  作成方法: {stat.creation_type}",
                    f"  最適化可能性: {stat.optimization_potential:.2f}",
                    f"  推奨最適化: {', '.join(stat.recommended_optimizations)}",
                    f"  期待メモリ削減: {stat.expected_memory_reduction:.1f}%",
                    f"  期待速度向上: {stat.expected_speed_improvement:.1f}x",
                ]
            )

        lines.extend(["", "=" * 80, "分析完了", "=" * 80])

        return "\n".join(lines)


# エクスポート用ファクトリ関数
def analyze_dataframe_usage(project_root: str = "src") -> DataFrameAnalysisReport:
    """DataFrame使用状況分析実行"""
    analyzer = DataFrameAnalyzer()
    return analyzer.analyze_project(project_root)


def run_dataframe_benchmark(sample_rows: int = 100000) -> Dict[str, Any]:
    """DataFrame操作ベンチマーク実行"""
    analyzer = DataFrameAnalyzer()
    return analyzer.benchmark_dataframe_operations(sample_rows)


if __name__ == "__main__":
    # テスト実行
    print("=== DataFrame使用状況・最適化分析 ===")

    # プロジェクト分析
    report = analyze_dataframe_usage()

    # ベンチマーク実行
    benchmark_results = run_dataframe_benchmark(50000)

    # レポート生成
    analyzer = DataFrameAnalyzer()
    summary_report = analyzer.generate_optimization_report(report)

    print(summary_report)
    print("\n✅ DataFrame分析完了")
