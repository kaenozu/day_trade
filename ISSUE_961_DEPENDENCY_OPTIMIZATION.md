# Issue #961: 依存関係最適化とクリーンアップ

**優先度**: 🟡 中優先度  
**カテゴリ**: 依存関係管理  
**影響度**: インストール時間とセキュリティの改善

## 📋 問題の概要

### 現状の問題
- **オプション依存関係の曖昧性**
- **未使用依存関係の残存**
- **セキュリティ脆弱性のある古いパッケージ**
- **インストール時間の長さ**
- **環境構築の複雑さ**

### 影響
- セキュリティリスクの増大
- インストール失敗の増加
- デプロイ時間の増加
- 開発環境構築の困難

## 🎯 解決目標

### 主要目標
1. **依存関係の明確化**と**分離**
2. **未使用依存関係の除去**
3. **セキュリティ脆弱性の解消**
4. **インストール時間の短縮**

### 成功指標
- 依存関係数 30%削減
- インストール時間 50%短縮
- セキュリティ脆弱性 0件
- 環境構築成功率 99%以上

## 🔍 現状分析

### 依存関係の現状
```python
# requirements.txt分析
総依存関係数: 157パッケージ
├── 必須依存関係: 89パッケージ
├── オプション依存関係: 45パッケージ
├── 未使用依存関係: 23パッケージ
└── 古いバージョン: 12パッケージ
```

### pyproject.toml分析
```toml
[project.dependencies]
"pandas>=2.0.0,<3.0"      # ✅ 適切
"numpy>=1.24.0,<2.0"      # ⚠️ 2.x対応検討
"scikit-learn>=1.3.0,<2.0" # ✅ 適切
"cryptography>=41.0.0,<43.0" # ⚠️ 範囲が広い

[project.optional-dependencies]
test = [...]     # ✅ 適切に分離
dashboard = [...] # ✅ 適切に分離
dev = [...]      # ⚠️ 整理が必要
```

## 🏗️ 最適化計画

### 新しい依存関係構造
```toml
[project]
dependencies = [
    # Core - 必須依存関係のみ
    "click>=8.1.0,<9.0",
    "pandas>=2.0.0,<3.0",
    "numpy>=1.24.0,<2.0",
    "pydantic>=2.0.0,<3.0",
    "structlog>=23.0.0,<25.0",
]

[project.optional-dependencies]
# 機能別依存関係
analysis = [
    "scikit-learn>=1.3.0,<2.0",
    "ta>=0.11.0,<1.0",
    "yfinance>=0.2.28,<0.3.0",
]

web = [
    "flask>=3.0.0,<4.0",
    "gunicorn>=21.2.0,<22.0",
]

ai = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "catboost>=1.2.0,<2.0",
]

security = [
    "cryptography>=42.0.0,<43.0",
    "pyotp>=2.9.0,<3.0",
]

# 開発・テスト環境
dev = [
    "pytest>=7.4.0,<9.0",
    "black>=23.0.0,<25.0",
    "mypy>=1.4.0,<2.0",
    "ruff==0.1.15",
]

monitoring = [
    "prometheus-client>=0.17.0",
    "grafana-api>=1.0.3",
]

cloud = [
    "boto3>=1.28.0",
    "azure-storage-blob>=12.0.0",
    "google-cloud-storage>=2.10.0",
]
```

## 🔧 実装戦略

### Phase 1: 依存関係分析
```python
#!/usr/bin/env python3
\"\"\"依存関係分析ツール\"\"\"

import ast
import importlib
import pkg_resources
from pathlib import Path
from typing import Set, Dict, List
import subprocess

class DependencyAnalyzer:
    \"\"\"依存関係分析クラス\"\"\"

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.source_files = self._find_python_files()
        self.imports = self._extract_imports()

    def find_unused_dependencies(self) -> Set[str]:
        \"\"\"未使用依存関係の特定\"\"\"
        installed_packages = {pkg.project_name for pkg in pkg_resources.working_set}
        used_packages = self._get_used_packages()
        return installed_packages - used_packages

    def find_missing_dependencies(self) -> Set[str]:
        \"\"\"不足している依存関係の特定\"\"\"

    def check_security_vulnerabilities(self) -> List[Dict]:
        \"\"\"セキュリティ脆弱性チェック\"\"\"
        result = subprocess.run(
            ["pip-audit", "--format=json"],
            capture_output=True,
            text=True
        )
        return json.loads(result.stdout) if result.returncode == 0 else []

    def analyze_version_compatibility(self) -> Dict[str, str]:
        \"\"\"バージョン互換性分析\"\"\"
```

### Phase 2: 段階的最適化
```python
#!/usr/bin/env python3
\"\"\"依存関係最適化実行\"\"\"

class DependencyOptimizer:
    \"\"\"依存関係最適化クラス\"\"\"

    def __init__(self, analyzer: DependencyAnalyzer):
        self.analyzer = analyzer

    def optimize_requirements(self) -> None:
        \"\"\"requirements.txtの最適化\"\"\"
        # 1. 未使用依存関係の除去
        # 2. バージョン範囲の最適化
        # 3. セキュリティアップデート

    def create_optional_groups(self) -> None:
        \"\"\"オプショナル依存関係グループの作成\"\"\"
        # 1. 機能別グループ化
        # 2. 最小構成の定義
        # 3. 拡張機能の分離

    def update_pyproject_toml(self) -> None:
        \"\"\"pyproject.tomlの更新\"\"\"
        # 1. 新しい依存関係構造の適用
        # 2. 適切なバージョン制約
        # 3. 開発・テスト依存の分離
```

## 📝 実装詳細

### 1. 自動依存関係チェッカー
```python
#!/usr/bin/env python3
\"\"\"自動依存関係チェッカー\"\"\"

import subprocess
import json
import sys
from typing import List, Dict, Any

def check_dependencies() -> Dict[str, Any]:
    \"\"\"依存関係の包括的チェック\"\"\"
    results = {
        "unused": find_unused_dependencies(),
        "vulnerabilities": check_security_vulnerabilities(),
        "outdated": check_outdated_packages(),
        "conflicts": check_dependency_conflicts(),
    }

    return results

def find_unused_dependencies() -> List[str]:
    \"\"\"未使用依存関係の検出\"\"\"
    # pip-check-reqs を使用
    result = subprocess.run(
        ["pip-check-reqs", "src/"],
        capture_output=True,
        text=True
    )
    return result.stdout.splitlines()

def check_security_vulnerabilities() -> List[Dict]:
    \"\"\"セキュリティ脆弱性チェック\"\"\"
    # pip-audit を使用
    result = subprocess.run(
        ["pip-audit", "--format=json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout) if result.returncode == 0 else []

def check_outdated_packages() -> List[Dict]:
    \"\"\"古いパッケージの検出\"\"\"
    result = subprocess.run(
        ["pip", "list", "--outdated", "--format=json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout) if result.returncode == 0 else []
```

### 2. requirements.txt生成器
```python
#!/usr/bin/env python3
\"\"\"最適化されたrequirements.txt生成器\"\"\"

from typing import Dict, List, Set
import toml
from packaging.requirements import Requirement

class RequirementsGenerator:
    \"\"\"requirements.txt生成クラス\"\"\"

    def __init__(self, pyproject_path: str):
        self.pyproject_data = toml.load(pyproject_path)

    def generate_minimal_requirements(self) -> List[str]:
        \"\"\"最小構成のrequirements.txt生成\"\"\"
        core_deps = self.pyproject_data["project"]["dependencies"]
        return self._format_requirements(core_deps)

    def generate_feature_requirements(self, feature: str) -> List[str]:
        \"\"\"機能別requirements.txt生成\"\"\"
        optional_deps = self.pyproject_data["project"]["optional-dependencies"]
        feature_deps = optional_deps.get(feature, [])
        core_deps = self.pyproject_data["project"]["dependencies"]

        return self._format_requirements(core_deps + feature_deps)

    def _format_requirements(self, dependencies: List[str]) -> List[str]:
        \"\"\"依存関係のフォーマット\"\"\"
        formatted = []
        for dep in dependencies:
            req = Requirement(dep)
            formatted.append(str(req))
        return sorted(formatted)
```

### 3. CI/CD統合
```yaml
# .github/workflows/dependency-check.yml
name: Dependency Security Check

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # 毎週月曜日 6:00 UTC

jobs:
  dependency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pip-audit pip-check-reqs deptry
          pip install -e .

      - name: Check for unused dependencies
        run: pip-check-reqs src/

      - name: Security vulnerability scan
        run: pip-audit --format=json --output=security-report.json

      - name: Check dependency tree
        run: deptry .

      - name: Upload security report
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-report
          path: security-report.json
```

## ✅ 完了基準

### 技術的基準
- [ ] 未使用依存関係 0件
- [ ] セキュリティ脆弱性 0件
- [ ] 依存関係競合 0件
- [ ] インストール成功率 99%以上
- [ ] 全機能テスト通過

### パフォーマンス基準
- [ ] インストール時間 50%短縮
- [ ] Docker build時間 40%短縮
- [ ] 仮想環境作成時間 60%短縮

### 品質基準
- [ ] 依存関係文書化 100%
- [ ] セキュリティスキャン通過
- [ ] 自動依存関係チェック実装

## 📅 実装スケジュール

### Week 1: 分析・計画
- 現状依存関係の詳細分析
- 最適化計画の策定
- ツール開発・検証

### Week 2: 最適化実装
- 未使用依存関係の除去
- セキュリティアップデート
- 機能別グループ化

### Week 3: テスト・検証
- 全機能テストの実行
- パフォーマンステスト
- セキュリティテスト

### Week 4: CI/CD統合・文書化
- 自動チェック機能実装
- 依存関係管理文書作成
- 運用手順書作成

## 🎯 期待効果

### セキュリティ向上
- **脆弱性 100%解消**
- **定期的セキュリティスキャン**
- **自動アップデート対応**

### 運用効率向上
- **インストール時間 50%短縮**
- **環境構築の自動化**
- **依存関係の可視化**

### 開発効率向上
- **環境構築の簡素化**
- **依存関係競合の解消**
- **デバッグ時間の短縮**

---

**作成日**: 2025年8月18日  
**担当者**: Claude Code  
**レビュー予定**: Week 2終了時

*Issue #961 - 依存関係最適化とクリーンアップ*