#!/usr/bin/env python3
"""
セキュアコーディングプラクティスチェッカー
Issue #419対応 - セキュアコーディング規約の徹底
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class SecurityRiskLevel(Enum):
    """セキュリティリスクレベル"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIssue:
    """セキュリティ問題"""

    file_path: str
    line_number: int
    risk_level: SecurityRiskLevel
    issue_type: str
    description: str
    recommendation: str
    code_snippet: str


class SecureCodingChecker:
    """セキュアコーディングチェッカー"""

    def __init__(self, project_root: Path = Path(".")):
        self.project_root = project_root
        self.issues: List[SecurityIssue] = []

        # 機密情報パターン
        self.sensitive_patterns = {
            "api_key": [
                r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
                r'apikey\s*=\s*["\'][^"\']{10,}["\']',
            ],
            "password": [
                r'password\s*=\s*["\'][^"\']{3,}["\']',
                r'passwd\s*=\s*["\'][^"\']{3,}["\']',
                r'pwd\s*=\s*["\'][^"\']{3,}["\']',
            ],
            "secret": [
                r'secret\s*=\s*["\'][^"\']{10,}["\']',
                r'secret[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
            ],
            "token": [
                r'token\s*=\s*["\'][^"\']{20,}["\']',
                r'access[_-]?token\s*=\s*["\'][^"\']{20,}["\']',
            ],
            "database_url": [
                r'database[_-]?url\s*=\s*["\'][^"\']{10,}["\']',
                r'db[_-]?url\s*=\s*["\'][^"\']{10,}["\']',
            ],
        }

        # 危険な関数呼び出しパターン
        self.dangerous_functions = {
            "eval": {
                "patterns": [r"\beval\s*\("],
                "risk": SecurityRiskLevel.CRITICAL,
                "description": "eval()の使用はコード注入脆弱性のリスク",
                "recommendation": "ast.literal_eval()や安全な代替手段を使用",
            },
            "exec": {
                "patterns": [r"\bexec\s*\("],
                "risk": SecurityRiskLevel.CRITICAL,
                "description": "exec()の使用はコード実行脆弱性のリスク",
                "recommendation": "動的コード実行を避け、静的な実装に変更",
            },
            "pickle_loads": {
                "patterns": [r"pickle\.loads?\s*\("],
                "risk": SecurityRiskLevel.HIGH,
                "description": "pickleは信頼できないデータで安全でない",
                "recommendation": "JSON等の安全なシリアライゼーション形式を使用",
            },
            "shell_injection": {
                "patterns": [
                    r"subprocess\.[^(]*\([^)]*shell\s*=\s*True",
                    r"os\.system\s*\(",
                    r"os\.popen\s*\(",
                ],
                "risk": SecurityRiskLevel.HIGH,
                "description": "シェルインジェクション脆弱性のリスク",
                "recommendation": "shell=Falseを使用し、引数を適切にエスケープ",
            },
            "sql_injection": {
                "patterns": [
                    r'\.execute\s*\(\s*["\'][^"\']*%[sd][^"\']*["\']',
                    r"\.execute\s*\([^)]*\+[^)]*\)",
                ],
                "risk": SecurityRiskLevel.HIGH,
                "description": "SQLインジェクション脆弱性のリスク",
                "recommendation": "パラメータ化クエリまたはORMを使用",
            },
        }

    def scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """単一ファイルのスキャン"""
        issues = []

        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith("#"):
                    continue

                # 機密情報チェック
                issues.extend(self._check_sensitive_data(file_path, line_num, line))

                # 危険な関数チェック
                issues.extend(self._check_dangerous_functions(file_path, line_num, line))

                # その他のセキュリティチェック
                issues.extend(self._check_other_security_issues(file_path, line_num, line))

        except (UnicodeDecodeError, PermissionError) as e:
            logger.warning(f"ファイル読み込みエラー {file_path}: {e}")

        return issues

    def _check_sensitive_data(
        self, file_path: Path, line_num: int, line: str
    ) -> List[SecurityIssue]:
        """機密データチェック"""
        issues = []

        for data_type, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issue = SecurityIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        risk_level=SecurityRiskLevel.HIGH,
                        issue_type=f"sensitive_data_{data_type}",
                        description=f"機密情報({data_type})がハードコーディングされている可能性",
                        recommendation="環境変数や専用の秘密管理システムを使用",
                        code_snippet=line.strip(),
                    )
                    issues.append(issue)

        return issues

    def _check_dangerous_functions(
        self, file_path: Path, line_num: int, line: str
    ) -> List[SecurityIssue]:
        """危険な関数チェック"""
        issues = []

        for func_name, func_data in self.dangerous_functions.items():
            for pattern in func_data["patterns"]:
                if re.search(pattern, line):
                    issue = SecurityIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        risk_level=func_data["risk"],
                        issue_type=f"dangerous_function_{func_name}",
                        description=func_data["description"],
                        recommendation=func_data["recommendation"],
                        code_snippet=line.strip(),
                    )
                    issues.append(issue)

        return issues

    def _check_other_security_issues(
        self, file_path: Path, line_num: int, line: str
    ) -> List[SecurityIssue]:
        """その他のセキュリティ問題チェック"""
        issues = []

        # HTTPSでない通信
        http_patterns = [
            r'http://[^/]*api[^"\']*',
            r'http://[^/]*auth[^"\']*',
        ]
        for pattern in http_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issue = SecurityIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    risk_level=SecurityRiskLevel.MEDIUM,
                    issue_type="insecure_http",
                    description="HTTPSではなくHTTPを使用している可能性",
                    recommendation="HTTPSを使用してデータ転送を暗号化",
                    code_snippet=line.strip(),
                )
                issues.append(issue)

        # デバッグ情報の残存
        debug_patterns = [
            r"print\s*\([^)]*password[^)]*\)",
            r"print\s*\([^)]*secret[^)]*\)",
            r"print\s*\([^)]*token[^)]*\)",
            r"console\.log\([^)]*password[^)]*\)",
        ]
        for pattern in debug_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issue = SecurityIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    risk_level=SecurityRiskLevel.MEDIUM,
                    issue_type="debug_sensitive_info",
                    description="機密情報をログ出力している可能性",
                    recommendation="機密情報をログに出力しないよう修正",
                    code_snippet=line.strip(),
                )
                issues.append(issue)

        return issues

    def scan_project(
        self, include_patterns: List[str] = None, exclude_patterns: List[str] = None
    ) -> List[SecurityIssue]:
        """プロジェクト全体のスキャン"""
        if include_patterns is None:
            include_patterns = [
                "**/*.py",
                "**/*.json",
                "**/*.yaml",
                "**/*.yml",
                "**/*.env",
            ]

        if exclude_patterns is None:
            exclude_patterns = [
                "**/venv/**",
                "**/env/**",
                "**/__pycache__/**",
                "**/node_modules/**",
                "**/build/**",
                "**/dist/**",
                "**/tests/**",
                "**/test_**",
                "**/*_test.py",
            ]

        all_issues = []
        scanned_files = 0

        for pattern in include_patterns:
            for file_path in self.project_root.glob(pattern):
                # 除外パターンチェック
                should_exclude = False
                for exclude_pattern in exclude_patterns:
                    if file_path.match(exclude_pattern):
                        should_exclude = True
                        break

                if should_exclude or not file_path.is_file():
                    continue

                logger.info(f"スキャン中: {file_path}")
                file_issues = self.scan_file(file_path)
                all_issues.extend(file_issues)
                scanned_files += 1

        logger.info(f"スキャン完了: {scanned_files}ファイル, {len(all_issues)}問題検出")
        return all_issues

    def generate_report(self, issues: List[SecurityIssue]) -> str:
        """セキュリティレポート生成"""
        if not issues:
            return "セキュリティ問題は検出されませんでした。"

        # リスクレベル別集計
        risk_counts = {level: 0 for level in SecurityRiskLevel}
        for issue in issues:
            risk_counts[issue.risk_level] += 1

        report = ["=" * 80]
        report.append("セキュアコーディングチェック レポート")
        report.append("=" * 80)
        report.append(f"総問題数: {len(issues)}")
        report.append("")

        # リスクレベル別サマリー
        report.append("【リスクレベル別サマリー】")
        for level, count in risk_counts.items():
            if count > 0:
                report.append(f"  {level.value.upper()}: {count}件")
        report.append("")

        # 問題種別別グループ化
        issues_by_type = {}
        for issue in issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = []
            issues_by_type[issue.issue_type].append(issue)

        # リスクレベル順でソート
        risk_order = [
            SecurityRiskLevel.CRITICAL,
            SecurityRiskLevel.HIGH,
            SecurityRiskLevel.MEDIUM,
            SecurityRiskLevel.LOW,
            SecurityRiskLevel.INFO,
        ]

        for risk_level in risk_order:
            level_issues = [i for i in issues if i.risk_level == risk_level]
            if not level_issues:
                continue

            report.append(f"【{risk_level.value.upper()}リスク問題】")

            for issue in level_issues:
                report.append(f"  ファイル: {issue.file_path}:{issue.line_number}")
                report.append(f"  問題: {issue.description}")
                report.append(f"  推奨: {issue.recommendation}")
                report.append(f"  コード: {issue.code_snippet}")
                report.append("")

        # 修正優先度の推奨
        report.append("【修正優先度】")
        if risk_counts[SecurityRiskLevel.CRITICAL] > 0:
            report.append("1. CRITICAL問題の即座の修正")
        if risk_counts[SecurityRiskLevel.HIGH] > 0:
            report.append("2. HIGH問題の優先的修正")
        if risk_counts[SecurityRiskLevel.MEDIUM] > 0:
            report.append("3. MEDIUM問題の計画的修正")

        report.append("=" * 80)
        return "\n".join(report)

    def save_report(self, issues: List[SecurityIssue], output_path: Path) -> None:
        """レポートファイル保存"""
        report = self.generate_report(issues)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"レポート保存: {output_path}")

    def save_json_report(self, issues: List[SecurityIssue], output_path: Path) -> None:
        """JSON形式レポート保存"""
        issues_data = []
        for issue in issues:
            issues_data.append(
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "risk_level": issue.risk_level.value,
                    "issue_type": issue.issue_type,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "code_snippet": issue.code_snippet,
                }
            )

        report_data = {
            "scan_summary": {
                "total_issues": len(issues),
                "risk_counts": {
                    level.value: len([i for i in issues if i.risk_level == level])
                    for level in SecurityRiskLevel
                },
            },
            "issues": issues_data,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSONレポート保存: {output_path}")


def main():
    """メイン実行"""
    logging.basicConfig(level=logging.INFO)

    checker = SecureCodingChecker()

    # プロジェクトスキャン
    issues = checker.scan_project()

    # レポート生成・表示
    report = checker.generate_report(issues)
    print(report)

    # レポート保存
    output_dir = Path("security_reports")
    output_dir.mkdir(exist_ok=True)

    checker.save_report(issues, output_dir / "secure_coding_report.txt")
    checker.save_json_report(issues, output_dir / "secure_coding_report.json")


if __name__ == "__main__":
    main()
