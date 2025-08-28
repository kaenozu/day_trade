#!/usr/bin/env python3
"""
品質ゲートシステム - テストカバレッジ解析

テストカバレッジの測定と分析を行うモジュール。
ファイル別・モジュール別のカバレッジ分析、
カバレッジ分布の計算等を提供する。
"""

import json
import os
import subprocess
from typing import Any, Dict, List


class TestCoverageAnalyzer:
    """テストカバレッジ解析システム
    
    Pythonプロジェクトのテストカバレッジを詳細に分析し、
    品質評価に必要な各種メトリクスを算出するクラス。
    """

    def analyze_coverage(self, coverage_file: str = "coverage.json") -> Dict[str, Any]:
        """テストカバレッジを解析
        
        既存のカバレッジファイルを分析するか、
        新たにカバレッジ解析を実行してテストカバレッジを評価する。
        
        Args:
            coverage_file: カバレッジデータファイルのパス
            
        Returns:
            カバレッジ分析結果を含む辞書
        """
        try:
            if os.path.exists(coverage_file):
                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                return self._analyze_coverage_data(coverage_data)
            else:
                # coverage.pyを実行してカバレッジを取得
                return self._run_coverage_analysis()

        except Exception as e:
            return {"error": str(e), "coverage_percentage": 0}

    def _analyze_coverage_data(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """カバレッジデータを分析
        
        coverage.pyが生成したJSONデータを解析し、
        詳細なカバレッジレポートを生成する。
        
        Args:
            coverage_data: coverage.pyからのJSONデータ
            
        Returns:
            詳細なカバレッジ分析結果
        """
        totals = coverage_data.get("totals", {})
        files = coverage_data.get("files", {})

        # ファイル別カバレッジ分析
        file_coverage = []
        low_coverage_files = []

        for file_path, file_data in files.items():
            coverage_pct = file_data.get("summary", {}).get("percent_covered", 0)
            file_coverage.append(
                {
                    "file": file_path,
                    "coverage": coverage_pct,
                    "lines_covered": file_data.get("summary", {}).get(
                        "covered_lines", 0
                    ),
                    "lines_total": file_data.get("summary", {}).get(
                        "num_statements", 0
                    ),
                }
            )

            if coverage_pct < 50:  # 50%未満は低カバレッジ
                low_coverage_files.append(file_path)

        # モジュール別カバレッジ
        module_coverage = self._calculate_module_coverage(file_coverage)

        return {
            "overall_coverage": totals.get("percent_covered", 0),
            "lines_covered": totals.get("covered_lines", 0),
            "lines_total": totals.get("num_statements", 0),
            "file_coverage": sorted(file_coverage, key=lambda x: x["coverage"]),
            "low_coverage_files": low_coverage_files,
            "module_coverage": module_coverage,
            "coverage_distribution": self._calculate_coverage_distribution(
                file_coverage
            ),
        }

    def _run_coverage_analysis(self) -> Dict[str, Any]:
        """カバレッジ解析を実行
        
        pytest と coverage.py を使用してテストカバレッジを測定する。
        
        Returns:
            カバレッジ分析結果を含む辞書
        """
        try:
            # pytest with coverage
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "--cov=src",
                    "--cov-report=json:coverage.json",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if os.path.exists("coverage.json"):
                with open("coverage.json") as f:
                    coverage_data = json.load(f)
                return self._analyze_coverage_data(coverage_data)
            else:
                return {
                    "error": "Coverage report not generated",
                    "coverage_percentage": 0,
                }

        except Exception as e:
            return {"error": str(e), "coverage_percentage": 0}

    def _calculate_module_coverage(
        self, file_coverage: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """モジュール別カバレッジを計算
        
        ファイル別カバレッジデータからモジュール単位の
        カバレッジを集計計算する。
        
        Args:
            file_coverage: ファイル別カバレッジデータのリスト
            
        Returns:
            モジュール名をキーとしたカバレッジパーセンテージの辞書
        """
        module_stats = {}

        for file_info in file_coverage:
            file_path = file_info["file"]

            # モジュール名を抽出 (例: src/day_trade/core/file.py -> core)
            path_parts = file_path.replace("\\", "/").split("/")
            if (
                len(path_parts) >= 3
                and path_parts[0] == "src"
                and path_parts[1] == "day_trade"
            ):
                module = path_parts[2]
            else:
                module = "other"

            if module not in module_stats:
                module_stats[module] = {"covered": 0, "total": 0}

            module_stats[module]["covered"] += file_info["lines_covered"]
            module_stats[module]["total"] += file_info["lines_total"]

        # パーセンテージに変換
        module_coverage = {}
        for module, stats in module_stats.items():
            if stats["total"] > 0:
                module_coverage[module] = (stats["covered"] / stats["total"]) * 100
            else:
                module_coverage[module] = 0

        return module_coverage

    def _calculate_coverage_distribution(
        self, file_coverage: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """カバレッジ分布を計算
        
        ファイルのカバレッジを範囲別に分類し、
        カバレッジ分布のヒストグラムを作成する。
        
        Args:
            file_coverage: ファイル別カバレッジデータのリスト
            
        Returns:
            カバレッジ範囲をキーとしたファイル数の辞書
        """
        distribution = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}

        for file_info in file_coverage:
            coverage = file_info["coverage"]

            if coverage <= 20:
                distribution["0-20"] += 1
            elif coverage <= 40:
                distribution["21-40"] += 1
            elif coverage <= 60:
                distribution["41-60"] += 1
            elif coverage <= 80:
                distribution["61-80"] += 1
            else:
                distribution["81-100"] += 1

        return distribution