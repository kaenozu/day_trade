#!/usr/bin/env python3
"""
依存関係管理ツール

このスクリプトは、プロジェクトの依存関係を管理・最適化します。
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DependencyManager:
    """依存関係管理クラス"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.requirements_path = self.project_root / "requirements.txt"
        self.requirements_dev_path = self.project_root / "requirements-dev.txt"
        self.reports_dir = self.project_root / "reports" / "dependencies"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def check_outdated_packages(self) -> List[Dict[str, str]]:
        """古いパッケージをチェック"""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"古いパッケージのチェックに失敗: {e}")
            return []

    def check_security_vulnerabilities(self) -> Optional[str]:
        """セキュリティ脆弱性をチェック"""
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                check=False,  # safety checkは脆弱性があっても0以外を返すことがある
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"セキュリティチェックに失敗: {e}")
            return None

    def validate_pyproject_dependencies(self) -> Dict[str, List[str]]:
        """pyproject.tomlの依存関係を検証"""
        issues = {"errors": [], "warnings": [], "suggestions": []}

        if not self.pyproject_path.exists():
            issues["errors"].append("pyproject.tomlが見つかりません")
            return issues

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                issues["errors"].append("tomlライブラリが見つかりません")
                return issues

        try:
            with open(self.pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)

            # 基本構造チェック
            if "project" not in pyproject_data:
                issues["errors"].append("projectセクションが見つかりません")
                return issues

            project = pyproject_data["project"]

            # 依存関係チェック
            if "dependencies" not in project:
                issues["warnings"].append("メイン依存関係が定義されていません")

            # オプション依存関係チェック
            if "optional-dependencies" not in project:
                issues["warnings"].append("オプション依存関係が定義されていません")
            else:
                opt_deps = project["optional-dependencies"]
                if "dev" not in opt_deps:
                    issues["warnings"].append("開発用依存関係が定義されていません")

            # バージョン制約チェック
            dependencies = project.get("dependencies", [])
            for dep in dependencies:
                if ">=" not in dep and "==" not in dep and "~=" not in dep:
                    issues["suggestions"].append(f"バージョン制約の追加を推奨: {dep}")

        except Exception as e:
            issues["errors"].append(f"pyproject.tomlの解析に失敗: {e}")

        return issues

    def generate_dependency_report(self) -> str:
        """依存関係レポートを生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"dependency_report_{timestamp}.md"

        outdated_packages = self.check_outdated_packages()
        security_report = self.check_security_vulnerabilities()
        validation_issues = self.validate_pyproject_dependencies()

        report_content = f"""# 依存関係管理レポート

**生成日時**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## 📊 概要

### プロジェクト構成
- **プロジェクトルート**: {self.project_root}
- **pyproject.toml**: {"✅ 存在" if self.pyproject_path.exists() else "❌ 不存在"}
- **requirements.txt**: {"✅ 存在" if self.requirements_path.exists() else "❌ 不存在"}
- **requirements-dev.txt**: {"✅ 存在" if self.requirements_dev_path.exists() else "❌ 不存在"}

## 🔄 古いパッケージ

"""

        if outdated_packages:
            report_content += (
                f"**{len(outdated_packages)}個の古いパッケージが見つかりました:**\n\n"
            )
            report_content += "| パッケージ | 現在のバージョン | 最新バージョン |\n"
            report_content += "|------------|------------------|----------------|\n"
            for pkg in outdated_packages:
                report_content += (
                    f"| {pkg['name']} | {pkg['version']} | {pkg['latest_version']} |\n"
                )
        else:
            report_content += "✅ すべてのパッケージが最新です。\n"

        report_content += "\n## 🔒 セキュリティチェック\n\n"

        if security_report:
            try:
                security_data = json.loads(security_report)
                if security_data:
                    report_content += f"⚠️ **{len(security_data)}件のセキュリティ問題が見つかりました**\n\n"
                    for issue in security_data:
                        report_content += f"- **{issue.get('package', 'Unknown')}**: {issue.get('vulnerability', 'Unknown vulnerability')}\n"
                else:
                    report_content += "✅ セキュリティ脆弱性は見つかりませんでした。\n"
            except json.JSONDecodeError:
                report_content += "⚠️ セキュリティレポートの解析に失敗しました。\n"
        else:
            report_content += "⚠️ セキュリティチェックを実行できませんでした。\n"

        report_content += "\n## ⚙️ 構成検証\n\n"

        if validation_issues["errors"]:
            report_content += "### ❌ エラー\n"
            for error in validation_issues["errors"]:
                report_content += f"- {error}\n"
            report_content += "\n"

        if validation_issues["warnings"]:
            report_content += "### ⚠️ 警告\n"
            for warning in validation_issues["warnings"]:
                report_content += f"- {warning}\n"
            report_content += "\n"

        if validation_issues["suggestions"]:
            report_content += "### 💡 提案\n"
            for suggestion in validation_issues["suggestions"]:
                report_content += f"- {suggestion}\n"
            report_content += "\n"

        if not any(validation_issues.values()):
            report_content += "✅ 構成に問題はありません。\n\n"

        report_content += """
## 🛠️ 推奨アクション

### 依存関係更新
```bash
# 古いパッケージを更新
pip install --upgrade package_name

# 開発用依存関係のインストール
pip install -e ".[dev]"

# 全ての依存関係を最新に更新 (注意して実行)
pip install --upgrade -r requirements.txt
```

### セキュリティ対策
```bash
# セキュリティ脆弱性チェック
safety check

# セキュリティ脆弱性の修正
pip install --upgrade vulnerable_package
```

### 依存関係管理のベストプラクティス
1. **バージョン制約の使用**: 具体的なバージョン範囲を指定
2. **定期的な更新**: 月1回程度の頻度で依存関係を更新
3. **セキュリティチェック**: CI/CDパイプラインにセキュリティチェックを組み込み
4. **開発用依存関係の分離**: 本番環境には不要なパッケージを分離

---
*このレポートは dependency_manager.py により自動生成されました。*
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return str(report_path)

    def update_packages(
        self, packages: Optional[List[str]] = None, dry_run: bool = False
    ) -> bool:
        """パッケージを更新"""
        if packages is None:
            # 古いパッケージを取得
            outdated = self.check_outdated_packages()
            packages = [pkg["name"] for pkg in outdated]

        if not packages:
            print("更新するパッケージがありません。")
            return True

        print(f"{'[DRY RUN] ' if dry_run else ''}以下のパッケージを更新します:")
        for pkg in packages:
            print(f"  - {pkg}")

        if dry_run:
            return True

        try:
            cmd = ["pip", "install", "--upgrade"] + packages
            subprocess.run(cmd, check=True)
            print("パッケージの更新が完了しました。")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ パッケージの更新に失敗: {e}")
            return False

    def sync_requirements_files(self) -> bool:
        """requirements.txtとpyproject.tomlを同期"""
        if not self.pyproject_path.exists():
            print("pyproject.tomlが見つかりません。")
            return False

        try:
            # pip-toolsを使用してrequirements.txtを生成
            subprocess.run(
                ["pip-compile", "--generate-hashes", str(self.pyproject_path)],
                check=True,
            )
            print("requirements.txtを更新しました。")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ requirements.txtの同期に失敗: {e}")
            return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="依存関係管理ツール")
    parser.add_argument(
        "--project-root", type=Path, help="プロジェクトルートディレクトリ"
    )

    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")

    # reportコマンド
    subparsers.add_parser("report", help="依存関係レポートを生成")

    # checkコマンド
    subparsers.add_parser("check", help="古いパッケージをチェック")

    # updateコマンド
    update_parser = subparsers.add_parser("update", help="パッケージを更新")
    update_parser.add_argument(
        "--packages",
        nargs="*",
        help="更新するパッケージ名（省略時は全ての古いパッケージ）",
    )
    update_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際には更新せず、更新予定のパッケージを表示",
    )

    # syncコマンド
    subparsers.add_parser("sync", help="requirements.txtを同期")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = DependencyManager(args.project_root)

    if args.command == "report":
        report_path = manager.generate_dependency_report()
        print(f"依存関係レポートを生成しました: {report_path}")

    elif args.command == "check":
        outdated = manager.check_outdated_packages()
        if outdated:
            print(f"📦 {len(outdated)}個の古いパッケージが見つかりました:")
            for pkg in outdated:
                print(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
        else:
            print("すべてのパッケージが最新です。")

    elif args.command == "update":
        success = manager.update_packages(args.packages, args.dry_run)
        sys.exit(0 if success else 1)

    elif args.command == "sync":
        success = manager.sync_requirements_files()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
