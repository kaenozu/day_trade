#!/usr/bin/env python3
"""
高度品質ゲートシステム
Issue #415: CI/CDパイプラインの強化と品質ゲートの厳格化

エンタープライズレベルのコード品質管理・監視システム:
- 包括的コード品質分析
- 依存関係健全性チェック
- セキュリティ脆弱性監視
- テストカバレッジ分析
- 複雑度・保守性評価
- CI/CD統合レポート生成
"""

import ast
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    import requests

    NETWORK_AVAILABLE = True
except ImportError:
    NETWORK_AVAILABLE = False


class QualityMetric(Enum):
    """品質メトリクス種類"""

    COMPLEXITY = "complexity"
    COVERAGE = "coverage"
    TYPING = "typing"
    SECURITY = "security"
    DEPENDENCIES = "dependencies"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"


class QualityLevel(Enum):
    """品質レベル"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    CRITICAL = "critical"


@dataclass
class QualityGate:
    """品質ゲート定義"""

    id: str
    name: str
    description: str
    metric_type: QualityMetric
    threshold_excellent: float
    threshold_good: float
    threshold_acceptable: float
    weight: float = 1.0
    mandatory: bool = True
    enabled: bool = True


@dataclass
class QualityResult:
    """品質評価結果"""

    gate_id: str
    metric_type: QualityMetric
    value: float
    level: QualityLevel
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FileQualityReport:
    """ファイル品質レポート"""

    file_path: str
    lines_of_code: int
    complexity_score: float
    maintainability_index: float
    type_coverage: float
    documentation_coverage: float
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    quality_level: QualityLevel = QualityLevel.ACCEPTABLE


class CodeComplexityAnalyzer:
    """コード複雑度解析システム"""

    def __init__(self):
        self.complexity_cache = {}

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """ファイルの複雑度を解析"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            analyzer = ComplexityVisitor()
            analyzer.visit(tree)

            # McCabe複雑度計算
            mccabe_complexity = analyzer.complexity

            # ハルステッド複雑度計算
            halstead_metrics = self._calculate_halstead_metrics(content)

            # 保守性指標計算
            maintainability_index = self._calculate_maintainability_index(
                mccabe_complexity, halstead_metrics, len(content.splitlines())
            )

            return {
                "file_path": file_path,
                "lines_of_code": len(
                    [
                        line
                        for line in content.splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ]
                ),
                "mccabe_complexity": mccabe_complexity,
                "halstead_metrics": halstead_metrics,
                "maintainability_index": maintainability_index,
                "functions": analyzer.functions,
                "classes": analyzer.classes,
            }

        except Exception as e:
            return {
                "file_path": file_path,
                "error": str(e),
                "lines_of_code": 0,
                "mccabe_complexity": 0,
                "maintainability_index": 0,
            }

    def _calculate_halstead_metrics(self, content: str) -> Dict[str, float]:
        """ハルステッド複雑度メトリクスを計算"""
        try:
            tree = ast.parse(content)
            visitor = HalsteadVisitor()
            visitor.visit(tree)

            # ハルステッド指標
            n1 = len(visitor.operators)  # ユニークオペレータ数
            n2 = len(visitor.operands)  # ユニークオペランド数
            N1 = visitor.operator_count  # 総オペレータ数
            N2 = visitor.operand_count  # 総オペランド数

            if n1 == 0 or n2 == 0:
                return {
                    "vocabulary": 0,
                    "length": 0,
                    "volume": 0,
                    "difficulty": 0,
                    "effort": 0,
                }

            vocabulary = n1 + n2
            length = N1 + N2
            volume = length * (vocabulary.bit_length() if vocabulary > 0 else 0)
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume

            return {
                "vocabulary": vocabulary,
                "length": length,
                "volume": volume,
                "difficulty": difficulty,
                "effort": effort,
            }

        except Exception:
            return {
                "vocabulary": 0,
                "length": 0,
                "volume": 0,
                "difficulty": 0,
                "effort": 0,
            }

    def _calculate_maintainability_index(
        self, complexity: float, halstead: Dict[str, float], loc: int
    ) -> float:
        """保守性指標を計算"""
        try:
            import math

            # 標準的な保守性指標計算
            # MI = 171 - 5.2 * ln(HalsteadVolume) - 0.23 * CyclomaticComplexity - 16.2 * ln(LinesOfCode)

            volume = max(halstead.get("volume", 1), 1)
            complexity = max(complexity, 1)
            loc = max(loc, 1)

            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(loc)

            # 0-100にスケール調整
            return max(0, min(100, mi))

        except Exception:
            return 50.0  # デフォルト値


class ComplexityVisitor(ast.NodeVisitor):
    """McCabe複雑度計算Visitor"""

    def __init__(self):
        self.complexity = 1  # 基本複雑度
        self.functions = []
        self.classes = []
        self.current_function = None
        self.current_class = None

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        function_complexity = 1

        for child in ast.walk(node):
            if isinstance(
                child,
                (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith),
            ):
                function_complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                function_complexity += 1
            elif isinstance(child, ast.BoolOp):
                function_complexity += len(child.values) - 1

        self.functions.append(
            {
                "name": node.name,
                "complexity": function_complexity,
                "line_number": node.lineno,
            }
        )

        self.complexity += function_complexity - 1
        self.generic_visit(node)
        self.current_function = None

    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.classes.append(
            {"name": node.name, "line_number": node.lineno, "methods": []}
        )
        self.generic_visit(node)
        self.current_class = None

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


class HalsteadVisitor(ast.NodeVisitor):
    """ハルステッドメトリクス計算Visitor"""

    def __init__(self):
        self.operators = set()
        self.operands = set()
        self.operator_count = 0
        self.operand_count = 0

    def visit_BinOp(self, node):
        self.operators.add(type(node.op).__name__)
        self.operator_count += 1
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self.operators.add(type(node.op).__name__)
        self.operator_count += 1
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            self.operators.add(type(op).__name__)
            self.operator_count += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        self.operands.add(node.id)
        self.operand_count += 1
        self.generic_visit(node)

    def visit_Constant(self, node):
        self.operands.add(repr(node.value))
        self.operand_count += 1
        self.generic_visit(node)


class DependencyHealthChecker:
    """依存関係健全性チェックシステム"""

    def __init__(self):
        self.vulnerability_cache = {}
        self.license_cache = {}

    def check_dependencies(
        self, requirements_file: str = "requirements.txt"
    ) -> Dict[str, Any]:
        """依存関係の健全性をチェック"""
        try:
            # pip-audit で脆弱性チェック
            audit_results = self._run_pip_audit()

            # ライセンスチェック
            license_results = self._check_licenses()

            # 依存関係グラフ分析
            dependency_graph = self._analyze_dependency_graph()

            # 更新可能性チェック
            outdated_packages = self._check_outdated_packages()

            return {
                "vulnerabilities": audit_results,
                "licenses": license_results,
                "dependency_graph": dependency_graph,
                "outdated_packages": outdated_packages,
                "health_score": self._calculate_dependency_health_score(
                    audit_results, license_results, outdated_packages
                ),
            }

        except Exception as e:
            return {"error": str(e), "health_score": 50.0}

    def _run_pip_audit(self) -> List[Dict[str, Any]]:
        """pip-auditを実行して脆弱性をチェック"""
        try:
            result = subprocess.run(
                ["pip-audit", "--format=json"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                return []  # 脆弱性なし
            else:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    return vulnerabilities.get("vulnerabilities", [])
                except json.JSONDecodeError:
                    return []

        except Exception:
            return []

    def _check_licenses(self) -> Dict[str, Any]:
        """ライセンス適合性をチェック"""
        try:
            result = subprocess.run(
                ["pip-licenses", "--format=json"],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                licenses = json.loads(result.stdout)

                # 問題のあるライセンスをチェック
                problematic_licenses = {"GPL", "LGPL", "AGPL", "SSPL"}
                issues = []

                for package in licenses:
                    license_name = package.get("License", "").upper()
                    if any(prob in license_name for prob in problematic_licenses):
                        issues.append(
                            {
                                "package": package.get("Name"),
                                "license": license_name,
                                "issue": "Potentially restrictive license",
                            }
                        )

                return {
                    "total_packages": len(licenses),
                    "license_issues": issues,
                    "compliance_score": max(0, 100 - len(issues) * 10),
                }
            else:
                return {"error": "Failed to check licenses", "compliance_score": 50}

        except Exception as e:
            return {"error": str(e), "compliance_score": 50}

    def _analyze_dependency_graph(self) -> Dict[str, Any]:
        """依存関係グラフを分析"""
        try:
            result = subprocess.run(
                ["pipdeptree", "--json-tree"],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                dependency_tree = json.loads(result.stdout)

                # 循環依存をチェック
                circular_deps = self._detect_circular_dependencies(dependency_tree)

                # 深い依存関係をチェック
                deep_deps = self._find_deep_dependencies(dependency_tree)

                return {
                    "total_packages": len(dependency_tree),
                    "circular_dependencies": circular_deps,
                    "deep_dependencies": deep_deps,
                    "max_depth": max(
                        [self._calculate_depth(pkg) for pkg in dependency_tree],
                        default=0,
                    ),
                }
            else:
                return {"error": "Failed to analyze dependency graph"}

        except Exception as e:
            return {"error": str(e)}

    def _detect_circular_dependencies(self, dependency_tree: List[Dict]) -> List[str]:
        """循環依存を検出"""
        # 簡易実装 - より高度なアルゴリズムが必要な場合は拡張
        return []

    def _find_deep_dependencies(
        self, dependency_tree: List[Dict], max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """深い依存関係を検出"""
        deep_deps = []

        for package in dependency_tree:
            depth = self._calculate_depth(package)
            if depth > max_depth:
                deep_deps.append(
                    {
                        "package": package.get("package", {}).get("package_name"),
                        "depth": depth,
                    }
                )

        return deep_deps

    def _calculate_depth(self, package: Dict, visited: Set[str] = None) -> int:
        """依存関係の深さを計算"""
        if visited is None:
            visited = set()

        package_name = package.get("package", {}).get("package_name")
        if package_name in visited:
            return 0  # 循環を避ける

        visited.add(package_name)
        dependencies = package.get("dependencies", [])

        if not dependencies:
            return 1

        max_child_depth = max(
            [self._calculate_depth(dep, visited.copy()) for dep in dependencies],
            default=0,
        )
        return max_child_depth + 1

    def _check_outdated_packages(self) -> List[Dict[str, Any]]:
        """古くなったパッケージをチェック"""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return []

        except Exception:
            return []

    def _calculate_dependency_health_score(
        self,
        vulnerabilities: List[Dict],
        license_results: Dict[str, Any],
        outdated_packages: List[Dict],
    ) -> float:
        """依存関係健全性スコアを計算"""
        score = 100.0

        # 脆弱性による減点
        critical_vulns = len(
            [v for v in vulnerabilities if v.get("severity") == "critical"]
        )
        high_vulns = len([v for v in vulnerabilities if v.get("severity") == "high"])
        medium_vulns = len(
            [v for v in vulnerabilities if v.get("severity") == "medium"]
        )

        score -= critical_vulns * 25
        score -= high_vulns * 15
        score -= medium_vulns * 5

        # ライセンス問題による減点
        license_score = license_results.get("compliance_score", 100)
        score = (score + license_score) / 2

        # 古いパッケージによる減点
        score -= min(len(outdated_packages) * 2, 20)

        return max(0.0, min(100.0, score))


class TestCoverageAnalyzer:
    """テストカバレッジ解析システム"""

    def analyze_coverage(self, coverage_file: str = "coverage.json") -> Dict[str, Any]:
        """テストカバレッジを解析"""
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
        """カバレッジデータを分析"""
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
        """カバレッジ解析を実行"""
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
        """モジュール別カバレッジを計算"""
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
        """カバレッジ分布を計算"""
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


class AdvancedQualityGateSystem:
    """高度品質ゲートシステム"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.db_path = self.project_root / "quality_metrics.db"

        # コンポーネント初期化
        self.complexity_analyzer = CodeComplexityAnalyzer()
        self.dependency_checker = DependencyHealthChecker()
        self.coverage_analyzer = TestCoverageAnalyzer()

        # 品質ゲート定義
        self.quality_gates = self._define_quality_gates()

        # データベース初期化
        self._initialize_database()

    def _define_quality_gates(self) -> List[QualityGate]:
        """品質ゲートを定義"""
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

    def _initialize_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_reports (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    overall_score REAL NOT NULL,
                    overall_level TEXT NOT NULL,
                    gates_passed INTEGER NOT NULL,
                    gates_total INTEGER NOT NULL,
                    details TEXT NOT NULL,
                    recommendations TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT NOT NULL,
                    gate_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    level TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES quality_reports (id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    lines_of_code INTEGER NOT NULL,
                    complexity_score REAL NOT NULL,
                    maintainability_index REAL NOT NULL,
                    quality_level TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES quality_reports (id)
                )
            """
            )

            conn.commit()

    async def run_comprehensive_quality_check(self) -> Dict[str, Any]:
        """包括的品質チェックを実行"""
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
                result = await self._evaluate_quality_gate(gate)
                gate_results.append(result)

                # ファイル別レポートの収集
                if gate.metric_type == QualityMetric.COMPLEXITY:
                    file_reports.extend(await self._collect_file_quality_reports())

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
        await self._save_quality_report(report)

        print(
            f"[完了] 品質チェック完了 - 総合スコア: {overall_score:.1f} ({overall_level.value})"
        )

        return report

    async def _evaluate_quality_gate(self, gate: QualityGate) -> QualityResult:
        """個別品質ゲートを評価"""
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
        # Python ファイルを収集
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
                "high_complexity_files": high_complexity_files[:5],  # 上位5件
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
                        "セキュアコーディングプラクティスを強化してください。",
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
                note_lines = result.stdout.count(": note:")

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
                        "MyPyエラーを段階的に修正してください。",
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
        """品質レベルを判定"""
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

    def _calculate_overall_score(self, results: List[QualityResult]) -> float:
        """総合スコアを計算"""
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
        """総合品質レベルを判定"""
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
        """総合推奨事項を生成"""
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
        """必須ゲートかどうかを判定"""
        gate = next((g for g in self.quality_gates if g.id == gate_id), None)
        return gate.mandatory if gate else False

    async def _collect_file_quality_reports(self) -> List[FileQualityReport]:
        """ファイル別品質レポートを収集"""
        file_reports = []
        python_files = list(self.project_root.glob("src/**/*.py"))

        for file_path in python_files:
            if "test" in str(file_path).lower() or "example" in str(file_path).lower():
                continue

            complexity_data = self.complexity_analyzer.analyze_file(str(file_path))

            if "error" not in complexity_data:
                quality_level = QualityLevel.GOOD  # デフォルト

                # 品質レベル判定
                complexity = complexity_data["mccabe_complexity"]
                maintainability = complexity_data["maintainability_index"]

                if complexity > 20 or maintainability < 30:
                    quality_level = QualityLevel.CRITICAL
                elif complexity > 15 or maintainability < 50:
                    quality_level = QualityLevel.NEEDS_IMPROVEMENT
                elif complexity > 10 or maintainability < 70:
                    quality_level = QualityLevel.ACCEPTABLE

                file_report = FileQualityReport(
                    file_path=str(file_path),
                    lines_of_code=complexity_data["lines_of_code"],
                    complexity_score=complexity,
                    maintainability_index=maintainability,
                    type_coverage=self._calculate_type_coverage(file_path),  # 本番実装
                    documentation_coverage=self._calculate_doc_coverage(file_path),  # 本番実装
                    quality_level=quality_level,
                )

                file_reports.append(file_report)

        return file_reports

    async def _save_quality_report(self, report: Dict[str, Any]):
        """品質レポートをデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # メインレポート保存
                conn.execute(
                    """
                    INSERT OR REPLACE INTO quality_reports
                    (id, timestamp, overall_score, overall_level, gates_passed, gates_total, details, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        report["report_id"],
                        report["timestamp"],
                        report["overall_score"],
                        report["overall_level"],
                        report["gates_passed"],
                        report["gates_total"],
                        json.dumps(report, default=str, ensure_ascii=False),
                        json.dumps(report["recommendations"], ensure_ascii=False),
                    ),
                )

                # メトリクス詳細保存
                for result in report["gate_results"]:
                    conn.execute(
                        """
                        INSERT INTO quality_metrics
                        (report_id, gate_id, metric_type, value, level, passed, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            report["report_id"],
                            result.gate_id,
                            result.metric_type.value,
                            result.value,
                            result.level.value,
                            result.passed,
                            report["timestamp"],
                        ),
                    )

                # ファイル品質データ保存
                for file_report in report["file_reports"]:
                    conn.execute(
                        """
                        INSERT INTO file_quality
                        (report_id, file_path, lines_of_code, complexity_score,
                         maintainability_index, quality_level, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            report["report_id"],
                            file_report.file_path,
                            file_report.lines_of_code,
                            file_report.complexity_score,
                            file_report.maintainability_index,
                            file_report.quality_level.value,
                            report["timestamp"],
                        ),
                    )

                conn.commit()
        except Exception as e:
            print(f"レポート保存エラー: {e}")

    def generate_ci_report(self, report: Dict[str, Any]) -> str:
        """CI/CD用レポートを生成"""
        lines = [
            "# [品質] Quality Gate Report",
            "",
            f"**Overall Score**: {report['overall_score']:.1f}/100 ({report['overall_level'].upper()})",
            f"**Gates Passed**: {report['gates_passed']}/{report['gates_total']}",
            f"**Timestamp**: {report['timestamp']}",
            "",
            "## [状況] Quality Gates Status",
            "",
        ]

        for result in report["gate_results"]:
            status_mark = "[PASS]" if result.passed else "[FAIL]"
            level_mark = {
                QualityLevel.EXCELLENT: "[優秀]",
                QualityLevel.GOOD: "[良好]",
                QualityLevel.ACCEPTABLE: "[許容]",
                QualityLevel.NEEDS_IMPROVEMENT: "[要改善]",
                QualityLevel.CRITICAL: "[危険]",
            }.get(result.level, "[不明]")

            lines.extend(
                [
                    f"### {status_mark} {result.gate_id.replace('_', ' ').title()}",
                    f"- **Value**: {result.value:.1f}",
                    f"- **Level**: {level_mark} {result.level.value}",
                    f"- **Status**: {'PASS' if result.passed else 'FAIL'}",
                    f"- **Message**: {result.message}",
                    "",
                ]
            )

        if report["recommendations"]:
            lines.extend(["## [推奨] Recommendations", ""])
            for i, rec in enumerate(report["recommendations"], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.extend(
            [
                "## [概要] Summary",
                f"- Excellent Gates: {report['summary']['excellent_gates']}",
                f"- Good Gates: {report['summary']['good_gates']}",
                f"- Acceptable Gates: {report['summary']['acceptable_gates']}",
                f"- Critical Gates: {report['summary']['critical_gates']}",
                "",
                "---",
                "*Generated by Advanced Quality Gate System*",
            ]
        )

        return "\n".join(lines)

    def _calculate_type_coverage(self, file_path: str) -> float:
        """型アノテーションカバレッジ計算（本番実装）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            import ast

            # 関数定義数をカウント
            tree = ast.parse(content)
            total_functions = 0
            typed_functions = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_functions += 1

                    # 戻り値の型アノテーションチェック
                    has_return_annotation = node.returns is not None

                    # 引数の型アノテーションチェック
                    has_arg_annotations = all(
                        arg.annotation is not None for arg in node.args.args
                        if arg.arg != 'self'  # selfは除外
                    )

                    if has_return_annotation and has_arg_annotations:
                        typed_functions += 1

            # カバレッジ計算
            if total_functions == 0:
                return 100.0  # 関数がない場合は100%

            coverage = (typed_functions / total_functions) * 100
            return round(coverage, 1)

        except Exception:
            return 0.0

    def _calculate_doc_coverage(self, file_path: str) -> float:
        """ドキュメントカバレッジ計算（本番実装）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            import ast

            tree = ast.parse(content)
            total_items = 0
            documented_items = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    total_items += 1

                    # docstringの存在チェック
                    if (node.body and
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                        documented_items += 1

            # カバレッジ計算
            if total_items == 0:
                return 100.0  # アイテムがない場合は100%

            coverage = (documented_items / total_items) * 100
            return round(coverage, 1)

        except Exception:
            return 0.0


# CLI インターフェース
async def main():
    """メイン実行関数"""
    import argparse

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


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
