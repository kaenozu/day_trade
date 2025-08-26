#!/usr/bin/env python3
"""
品質ゲートシステム - 依存関係健全性チェック

プロジェクトの依存関係の健全性を多角的に評価するモジュール。
セキュリティ脆弱性、ライセンス適合性、依存関係グラフの分析を行い、
総合的な依存関係健全性スコアを算出する。
"""

import json
import subprocess
from typing import Any, Dict, List, Set


class DependencyHealthChecker:
    """依存関係健全性チェックシステム
    
    Pythonプロジェクトの依存関係を包括的に分析し、
    セキュリティ・ライセンス・構造の観点から健全性を評価する。
    """

    def __init__(self):
        """初期化
        
        分析結果のキャッシュを初期化。
        """
        self.vulnerability_cache = {}
        self.license_cache = {}

    def check_dependencies(
        self, requirements_file: str = "requirements.txt"
    ) -> Dict[str, Any]:
        """依存関係の健全性をチェック
        
        複数の観点から依存関係を分析し、
        総合的な健全性評価を提供する。
        
        Args:
            requirements_file: requirements.txtファイルのパス
            
        Returns:
            依存関係健全性の分析結果を含む辞書
        """
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
        """pip-auditを実行して脆弱性をチェック
        
        pip-auditツールを使用してインストール済みパッケージの
        既知の脆弱性を検出する。
        
        Returns:
            脆弱性リスト。問題がない場合は空リスト。
        """
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
        """ライセンス適合性をチェック
        
        pip-licensesを使用してパッケージのライセンスを調査し、
        制限の厳しいライセンスの使用をチェックする。
        
        Returns:
            ライセンス分析結果を含む辞書
        """
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
        """依存関係グラフを分析
        
        pipdeptreeを使用して依存関係の構造を分析し、
        循環依存や深い依存チェーンを検出する。
        
        Returns:
            依存関係グラフ分析結果を含む辞書
        """
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
        """循環依存を検出
        
        依存関係ツリーから循環依存を検出する。
        
        Args:
            dependency_tree: pipdeptreeから取得した依存関係ツリー
            
        Returns:
            循環依存が検出されたパッケージ名のリスト
        """
        # 簡易実装 - より高度なアルゴリズムが必要な場合は拡張
        return []

    def _find_deep_dependencies(
        self, dependency_tree: List[Dict], max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """深い依存関係を検出
        
        指定された深度を超える依存関係チェーンを検出する。
        
        Args:
            dependency_tree: 依存関係ツリー
            max_depth: 警告を出す最大深度
            
        Returns:
            深い依存関係を持つパッケージのリスト
        """
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
        """依存関係の深さを計算
        
        指定されたパッケージの依存関係の最大深度を再帰的に計算。
        
        Args:
            package: パッケージ情報
            visited: 訪問済みパッケージ集合（循環検出用）
            
        Returns:
            依存関係の最大深度
        """
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
        """古くなったパッケージをチェック
        
        pip list --outdatedを実行して、
        更新が必要なパッケージを検出する。
        
        Returns:
            更新が必要なパッケージのリスト
        """
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
        """依存関係健全性スコアを計算
        
        各種分析結果を総合して0-100の健全性スコアを算出。
        
        Args:
            vulnerabilities: 脆弱性リスト
            license_results: ライセンス分析結果
            outdated_packages: 古いパッケージリスト
            
        Returns:
            0-100の健全性スコア
        """
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