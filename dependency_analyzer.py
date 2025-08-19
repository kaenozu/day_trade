#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依存関係分析・最適化ツール - Issue #961対応
プロジェクトの依存関係を分析し、最適化提案を生成
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util

@dataclass
class DependencyInfo:
    """依存関係情報"""
    name: str
    version: str
    location: str  # requirements.txt, pyproject.toml等
    is_used: bool = False
    is_dev_only: bool = False
    is_transitive: bool = False
    usage_count: int = 0
    files_using: List[str] = None

    def __post_init__(self):
        if self.files_using is None:
            self.files_using = []

@dataclass
class SecurityIssue:
    """セキュリティ問題"""
    package: str
    vulnerability: str
    severity: str
    current_version: str
    fixed_version: Optional[str] = None

@dataclass
class AnalysisResult:
    """分析結果"""
    total_dependencies: int
    used_dependencies: int
    unused_dependencies: int
    dev_dependencies: int
    security_issues: List[SecurityIssue]
    version_conflicts: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    size_analysis: Dict[str, Any]

class DependencyAnalyzer:
    """依存関係分析クラス"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.import_usage: Dict[str, Set[str]] = defaultdict(set)
        self.logger = self._setup_logging()

        # パッケージ名とインポート名のマッピング
        self.package_import_mapping = {
            "scikit-learn": "sklearn",
            "pillow": "PIL",
            "beautifulsoup4": "bs4",
            "python-dateutil": "dateutil",
            "python-dotenv": "dotenv",
            "pyyaml": "yaml",
            "requests-oauthlib": "requests_oauthlib",
            "flask-sqlalchemy": "flask_sqlalchemy",
            "flask-migrate": "flask_migrate",
            "flask-login": "flask_login",
            "flask-wtf": "flask_wtf",
            "sqlalchemy": "sqlalchemy",
            "pandas-datareader": "pandas_datareader",
            "pandas-ta": "pandas_ta",
            "psycopg2-binary": "psycopg2"
        }

    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('dependency_analyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def analyze_dependencies(self) -> AnalysisResult:
        """依存関係の完全分析"""
        self.logger.info("Dependency Analysis - Issue #961")
        self.logger.info("=" * 50)

        # Step 1: 依存関係ファイルの読み込み
        self._load_dependencies()

        # Step 2: 実際の使用状況分析
        self._analyze_import_usage()

        # Step 3: セキュリティ問題のチェック
        security_issues = self._check_security_issues()

        # Step 4: バージョン競合のチェック
        version_conflicts = self._check_version_conflicts()

        # Step 5: パッケージサイズ分析
        size_analysis = self._analyze_package_sizes()

        # Step 6: 最適化推奨事項の生成
        recommendations = self._generate_recommendations()

        # 統計計算
        total_deps = len(self.dependencies)
        used_deps = len([dep for dep in self.dependencies.values() if dep.is_used])
        unused_deps = total_deps - used_deps
        dev_deps = len([dep for dep in self.dependencies.values() if dep.is_dev_only])

        result = AnalysisResult(
            total_dependencies=total_deps,
            used_dependencies=used_deps,
            unused_dependencies=unused_deps,
            dev_dependencies=dev_deps,
            security_issues=security_issues,
            version_conflicts=version_conflicts,
            recommendations=recommendations,
            size_analysis=size_analysis
        )

        self.logger.info(f"Analysis completed: {total_deps} dependencies, {unused_deps} unused")
        return result

    def _load_dependencies(self) -> None:
        """依存関係ファイルの読み込み"""
        # requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            self._parse_requirements_file(requirements_file, "requirements.txt", False)

        # requirements-dev.txt
        dev_requirements_file = self.project_root / "requirements-dev.txt"
        if dev_requirements_file.exists():
            self._parse_requirements_file(dev_requirements_file, "requirements-dev.txt", True)

        # pyproject.toml
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            self._parse_pyproject_toml(pyproject_file)

        self.logger.info(f"Loaded {len(self.dependencies)} dependencies")

    def _parse_requirements_file(self, file_path: Path, location: str, is_dev: bool) -> None:
        """requirements.txtファイルの解析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep_info = self._parse_dependency_line(line, location, is_dev)
                        if dep_info:
                            self.dependencies[dep_info.name] = dep_info
        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path}: {e}")

    def _parse_pyproject_toml(self, file_path: Path) -> None:
        """pyproject.tomlファイルの解析"""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                self.logger.warning("toml library not available, skipping pyproject.toml")
                return

        try:
            with open(file_path, 'rb') as f:
                data = tomllib.load(f)

            # プロダクション依存関係
            if 'project' in data and 'dependencies' in data['project']:
                for dep_line in data['project']['dependencies']:
                    dep_info = self._parse_dependency_line(dep_line, "pyproject.toml", False)
                    if dep_info and dep_info.name not in self.dependencies:
                        self.dependencies[dep_info.name] = dep_info

            # オプショナル依存関係
            if 'project' in data and 'optional-dependencies' in data['project']:
                optional_deps = data['project']['optional-dependencies']
                for group_name, deps in optional_deps.items():
                    is_dev = group_name in ['dev', 'test', 'docs']
                    for dep_line in deps:
                        dep_info = self._parse_dependency_line(dep_line, f"pyproject.toml[{group_name}]", is_dev)
                        if dep_info and dep_info.name not in self.dependencies:
                            self.dependencies[dep_info.name] = dep_info

        except Exception as e:
            self.logger.warning(f"Failed to parse {file_path}: {e}")

    def _parse_dependency_line(self, line: str, location: str, is_dev: bool) -> Optional[DependencyInfo]:
        """依存関係行の解析"""
        # コメントを除去
        line = line.split('#')[0].strip()
        if not line:
            return None

        # バージョン指定の解析
        version_patterns = [
            r'^([a-zA-Z0-9_-]+)\[([^\]]+)\]==([0-9.]+)',  # package[extra]==version
            r'^([a-zA-Z0-9_-]+)==([0-9.]+)',  # package==version
            r'^([a-zA-Z0-9_-]+)>=([0-9.]+)',  # package>=version
            r'^([a-zA-Z0-9_-]+)<=([0-9.]+)',  # package<=version
            r'^([a-zA-Z0-9_-]+)~=([0-9.]+)',  # package~=version
            r'^([a-zA-Z0-9_-]+)>([0-9.]+)',   # package>version
            r'^([a-zA-Z0-9_-]+)<([0-9.]+)',   # package<version
            r'^([a-zA-Z0-9_-]+)',             # package only
        ]

        for pattern in version_patterns:
            match = re.match(pattern, line)
            if match:
                name = match.group(1).lower()
                version = match.group(2) if len(match.groups()) >= 2 else "latest"

                return DependencyInfo(
                    name=name,
                    version=version,
                    location=location,
                    is_dev_only=is_dev
                )

        return None

    def _analyze_import_usage(self) -> None:
        """インポート使用状況の分析"""
        python_files = list(self.project_root.glob("**/*.py"))

        # 除外パターン
        exclude_patterns = [
            "venv", ".venv", "env", ".env",
            "__pycache__", ".git", "node_modules",
            "build", "dist", ".eggs",
            "config_migration_backup"
        ]

        for py_file in python_files:
            # 除外パスのチェック
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            try:
                self._analyze_file_imports(py_file)
            except Exception as e:
                self.logger.debug(f"Failed to analyze {py_file}: {e}")

        # 使用状況の更新
        self._update_usage_status()

        self.logger.info(f"Analyzed {len(python_files)} Python files")

    def _analyze_file_imports(self, file_path: Path) -> None:
        """ファイルのインポート分析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            try:
                tree = ast.parse(content)
            except SyntaxError:
                return

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.import_usage[alias.name].add(str(file_path))

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.import_usage[node.module].add(str(file_path))

        except Exception as e:
            self.logger.debug(f"Error parsing {file_path}: {e}")

    def _update_usage_status(self) -> None:
        """使用状況の更新"""
        for dep_name, dep_info in self.dependencies.items():
            # インポート名の正規化
            import_names = [dep_name]

            # パッケージ名からインポート名への変換
            if dep_name in self.package_import_mapping:
                import_names.append(self.package_import_mapping[dep_name])

            # 一般的な変換パターン
            import_names.append(dep_name.replace('-', '_'))
            import_names.append(dep_name.replace('_', '-'))

            # 使用チェック
            for import_name in import_names:
                if import_name in self.import_usage:
                    dep_info.is_used = True
                    dep_info.usage_count = len(self.import_usage[import_name])
                    dep_info.files_using = list(self.import_usage[import_name])
                    break

                # 部分マッチもチェック
                for used_import in self.import_usage.keys():
                    if used_import.startswith(import_name + '.'):
                        dep_info.is_used = True
                        dep_info.usage_count += len(self.import_usage[used_import])
                        dep_info.files_using.extend(self.import_usage[used_import])

    def _check_security_issues(self) -> List[SecurityIssue]:
        """セキュリティ問題のチェック"""
        security_issues = []

        # 既知の脆弱性がある古いバージョン
        vulnerable_packages = {
            "flask": {"<2.2.0": "Cross-site scripting vulnerability"},
            "requests": {"<2.25.0": "Server-side request forgery"},
            "sqlalchemy": {"<1.4.0": "SQL injection vulnerability"},
            "pillow": {"<8.3.2": "Buffer overflow vulnerability"},
            "pyyaml": {"<5.4.0": "Code execution vulnerability"},
            "jinja2": {"<2.11.3": "Template injection vulnerability"}
        }

        for dep_name, dep_info in self.dependencies.items():
            if dep_name in vulnerable_packages:
                for version_constraint, vulnerability in vulnerable_packages[dep_name].items():
                    if self._version_matches_constraint(dep_info.version, version_constraint):
                        security_issues.append(SecurityIssue(
                            package=dep_name,
                            vulnerability=vulnerability,
                            severity="HIGH",
                            current_version=dep_info.version,
                            fixed_version="latest"
                        ))

        return security_issues

    def _version_matches_constraint(self, version: str, constraint: str) -> bool:
        """バージョン制約のマッチング"""
        if constraint.startswith('<'):
            constraint_version = constraint[1:]
            return version < constraint_version
        elif constraint.startswith('>'):
            constraint_version = constraint[1:]
            return version > constraint_version
        elif constraint.startswith('=='):
            constraint_version = constraint[2:]
            return version == constraint_version
        return False

    def _check_version_conflicts(self) -> List[Dict[str, Any]]:
        """バージョン競合のチェック"""
        conflicts = []

        # 同じパッケージの異なるバージョン指定
        package_versions = defaultdict(list)
        for dep_info in self.dependencies.values():
            package_versions[dep_info.name].append(dep_info)

        for package_name, deps in package_versions.items():
            if len(deps) > 1:
                versions = [dep.version for dep in deps]
                locations = [dep.location for dep in deps]

                conflicts.append({
                    "package": package_name,
                    "versions": versions,
                    "locations": locations,
                    "type": "multiple_definitions"
                })

        return conflicts

    def _analyze_package_sizes(self) -> Dict[str, Any]:
        """パッケージサイズ分析"""
        # サイズ推定（一般的なパッケージサイズ）
        estimated_sizes = {
            "numpy": 50.0,
            "pandas": 80.0,
            "scikit-learn": 120.0,
            "tensorflow": 500.0,
            "torch": 800.0,
            "matplotlib": 40.0,
            "scipy": 60.0,
            "flask": 5.0,
            "django": 30.0,
            "requests": 2.0,
            "pillow": 15.0,
            "opencv-python": 200.0
        }

        total_size = 0
        largest_packages = []

        for dep_name, dep_info in self.dependencies.items():
            if not dep_info.is_dev_only:
                size = estimated_sizes.get(dep_name, 5.0)  # デフォルト5MB
                total_size += size
                largest_packages.append({"name": dep_name, "size": size, "used": dep_info.is_used})

        largest_packages.sort(key=lambda x: x["size"], reverse=True)

        return {
            "total_estimated_size_mb": total_size,
            "largest_packages": largest_packages[:10],
            "unused_package_size_mb": sum(
                estimated_sizes.get(dep.name, 5.0)
                for dep in self.dependencies.values()
                if not dep.is_used and not dep.is_dev_only
            )
        }

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """最適化推奨事項の生成"""
        recommendations = []

        # 未使用依存関係の除去
        unused_deps = [dep for dep in self.dependencies.values() if not dep.is_used and not dep.is_dev_only]
        if unused_deps:
            recommendations.append({
                "priority": "HIGH",
                "category": "cleanup",
                "title": "未使用依存関係の除去",
                "description": f"{len(unused_deps)}個の未使用パッケージを除去してください",
                "packages": [dep.name for dep in unused_deps],
                "estimated_reduction_mb": sum(self._estimate_package_size(dep.name) for dep in unused_deps)
            })

        # セキュリティアップデート
        security_packages = set()
        for dep in self.dependencies.values():
            if dep.name in ["flask", "requests", "sqlalchemy"] and dep.is_used:
                security_packages.add(dep.name)

        if security_packages:
            recommendations.append({
                "priority": "CRITICAL",
                "category": "security",
                "title": "セキュリティアップデート",
                "description": "セキュリティに関わるパッケージを最新版にアップデートしてください",
                "packages": list(security_packages)
            })

        # バージョン統一
        dev_prod_overlap = []
        for dep_name, dep_info in self.dependencies.items():
            if dep_info.is_dev_only and dep_name in [d.name for d in self.dependencies.values() if not d.is_dev_only]:
                dev_prod_overlap.append(dep_name)

        if dev_prod_overlap:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "optimization",
                "title": "依存関係の統一",
                "description": "開発用と本番用で重複している依存関係を統一してください",
                "packages": dev_prod_overlap
            })

        # 軽量代替の提案
        lightweight_alternatives = {
            "requests": "httpx (非同期対応)",
            "flask": "fastapi (高性能)",
            "pandas": "polars (高速処理)"
        }

        heavy_packages = []
        for dep_name, dep_info in self.dependencies.items():
            if dep_info.is_used and dep_name in lightweight_alternatives:
                heavy_packages.append({
                    "current": dep_name,
                    "alternative": lightweight_alternatives[dep_name]
                })

        if heavy_packages:
            recommendations.append({
                "priority": "LOW",
                "category": "performance",
                "title": "軽量代替パッケージの検討",
                "description": "より軽量・高性能な代替パッケージを検討してください",
                "alternatives": heavy_packages
            })

        return recommendations

    def _estimate_package_size(self, package_name: str) -> float:
        """パッケージサイズの推定"""
        size_estimates = {
            "numpy": 50.0, "pandas": 80.0, "scikit-learn": 120.0,
            "matplotlib": 40.0, "scipy": 60.0, "pillow": 15.0
        }
        return size_estimates.get(package_name, 5.0)

    def generate_optimized_requirements(self) -> Dict[str, str]:
        """最適化された依存関係ファイルの生成"""
        optimized_files = {}

        # 使用中のプロダクション依存関係
        prod_deps = []
        for dep in self.dependencies.values():
            if dep.is_used and not dep.is_dev_only:
                version_spec = f">={dep.version}" if dep.version != "latest" else ""
                prod_deps.append(f"{dep.name}{version_spec}")

        optimized_files["requirements.txt"] = "\n".join(sorted(prod_deps))

        # 開発用依存関係
        dev_deps = []
        for dep in self.dependencies.values():
            if dep.is_dev_only:
                version_spec = f">={dep.version}" if dep.version != "latest" else ""
                dev_deps.append(f"{dep.name}{version_spec}")

        optimized_files["requirements-dev.txt"] = "\n".join(sorted(dev_deps))

        return optimized_files

    def export_analysis(self, result: AnalysisResult, output_file: str = "dependency_analysis_report.json") -> None:
        """分析結果のエクスポート"""
        export_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": asdict(result),
            "detailed_dependencies": {
                name: {
                    "version": dep.version,
                    "location": dep.location,
                    "is_used": dep.is_used,
                    "is_dev_only": dep.is_dev_only,
                    "usage_count": dep.usage_count,
                    "files_using_count": len(dep.files_using)
                }
                for name, dep in self.dependencies.items()
            },
            "optimization_plan": self._create_optimization_plan(result)
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Analysis exported to {output_file}")

    def _create_optimization_plan(self, result: AnalysisResult) -> Dict[str, Any]:
        """最適化計画の作成"""
        return {
            "immediate_actions": [
                "未使用依存関係の除去",
                "セキュリティアップデートの適用"
            ],
            "medium_term_actions": [
                "バージョン制約の統一",
                "開発用・本番用依存関係の分離"
            ],
            "long_term_actions": [
                "軽量代替パッケージへの移行検討",
                "モノレポ構成での依存関係管理"
            ],
            "estimated_benefits": {
                "size_reduction_mb": result.size_analysis.get("unused_package_size_mb", 0),
                "security_improvements": len(result.security_issues),
                "maintenance_simplification": "HIGH"
            }
        }

def main():
    """メイン実行"""
    analyzer = DependencyAnalyzer()
    result = analyzer.analyze_dependencies()

    print("\nDEPENDENCY ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total dependencies: {result.total_dependencies}")
    print(f"Used dependencies: {result.used_dependencies}")
    print(f"Unused dependencies: {result.unused_dependencies}")
    print(f"Development dependencies: {result.dev_dependencies}")
    print(f"Security issues: {len(result.security_issues)}")
    print(f"Version conflicts: {len(result.version_conflicts)}")
    print(f"Estimated total size: {result.size_analysis['total_estimated_size_mb']:.1f} MB")
    print(f"Unused package size: {result.size_analysis['unused_package_size_mb']:.1f} MB")

    # 推奨事項表示
    print(f"\nRECOMMENDATIONS ({len(result.recommendations)}):")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i}. [{rec['priority']}] {rec['title']}")
        print(f"   {rec['description']}")

    # 分析結果エクスポート
    analyzer.export_analysis(result)

    # 最適化ファイル生成
    optimized_files = analyzer.generate_optimized_requirements()
    print(f"\nOptimized requirements files generated:")
    for filename, content in optimized_files.items():
        print(f"- {filename}: {len(content.splitlines())} dependencies")

if __name__ == "__main__":
    main()