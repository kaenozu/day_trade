#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Audit Tool for Day Trade Personal
Issue #901 Phase 3: セキュリティ監査・強化

包括的なセキュリティ脆弱性スキャンと改善提案
"""

import os
import re
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


@dataclass
class SecurityIssue:
    """セキュリティ問題の詳細"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str
    cve_reference: Optional[str] = None


@dataclass
class SecurityReport:
    """セキュリティ監査レポート"""
    scan_date: datetime
    total_files_scanned: int
    issues_found: List[SecurityIssue] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class SecurityAuditor:
    """セキュリティ監査ツール"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = self._setup_logging()

        # セキュリティパターン定義
        self.security_patterns = self._load_security_patterns()

        # 除外パターン
        self.exclude_patterns = [
            r'\.git/',
            r'__pycache__/',
            r'\.pyc$',
            r'node_modules/',
            r'venv/',
            r'\.venv/',
            r'logs/',
            r'cache/',
            r'test.*mock',
            r'\.test\.',
            r'conftest\.py'
        ]

    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('security_auditor')
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _load_security_patterns(self) -> Dict[str, Dict]:
        """セキュリティパターン定義読み込み"""
        return {
            "hardcoded_secrets": {
                "patterns": [
                    r"(?i)(password|passwd|pwd)\s*=\s*['\"][^'\"]{3,}['\"]",
                    r"(?i)(secret|secret_key)\s*=\s*['\"][^'\"]{8,}['\"]",
                    r"(?i)(api_key|apikey)\s*=\s*['\"][^'\"]{8,}['\"]",
                    r"(?i)(token|access_token)\s*=\s*['\"][^'\"]{8,}['\"]",
                    r"['\"][A-Za-z0-9]{32,}['\"]",  # Long hex strings
                    r"sk-[A-Za-z0-9]{32,}",  # API keys
                    r"ghp_[A-Za-z0-9]{36}",  # GitHub tokens
                ],
                "severity": "CRITICAL",
                "category": "Credential Management"
            },
            "sql_injection": {
                "patterns": [
                    r"\.execute\s*\(\s*['\"].*%.*['\"]",
                    r"\.format\s*\(.*\).*execute",
                    r"f['\"].*{.*}.*['\"].*execute",
                    r"cursor\.execute.*\+.*",
                    r"query\s*=.*\+.*",
                ],
                "severity": "HIGH",
                "category": "Injection Vulnerability"
            },
            "debug_exposure": {
                "patterns": [
                    r"debug\s*=\s*True",
                    r"DEBUG\s*=\s*True",
                    r"app\.run\(.*debug.*True",
                    r"\.run\(.*debug.*True",
                    r"print\s*\(.*password",
                    r"print\s*\(.*secret",
                    r"logging\.(debug|info).*password",
                ],
                "severity": "MEDIUM",
                "category": "Information Disclosure"
            },
            "weak_crypto": {
                "patterns": [
                    r"md5\(",
                    r"sha1\(",
                    r"DES\(",
                    r"RC4\(",
                    r"random\.random\(\)",
                    r"random\.randint\(",
                    r"secrets.*32\)",  # Weak secret length
                ],
                "severity": "MEDIUM",
                "category": "Cryptographic Issues"
            },
            "unsafe_deserialization": {
                "patterns": [
                    r"pickle\.loads?",
                    r"cPickle\.loads?",
                    r"yaml\.load\(",
                    r"eval\(",
                    r"exec\(",
                ],
                "severity": "HIGH",
                "category": "Unsafe Deserialization"
            },
            "path_traversal": {
                "patterns": [
                    r"open\s*\(.*\+.*",
                    r"Path\(.*\+.*",
                    r"\.\.\/",
                    r"\.\.\\",
                    r"join\(.*input",
                ],
                "severity": "HIGH",
                "category": "Path Traversal"
            },
            "xss_vulnerability": {
                "patterns": [
                    r"render_template_string\(.*\+",
                    r"Markup\(.*\+",
                    r"escape=False",
                    r"safe\s*\|",
                ],
                "severity": "MEDIUM",
                "category": "Cross-Site Scripting"
            }
        }

    def scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """単一ファイルのセキュリティスキャン"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                for pattern_name, pattern_config in self.security_patterns.items():
                    for pattern in pattern_config["patterns"]:
                        if re.search(pattern, line):
                            # 除外パターンチェック
                            if self._should_exclude(str(file_path), line):
                                continue

                            issue = SecurityIssue(
                                severity=pattern_config["severity"],
                                category=pattern_config["category"],
                                title=f"{pattern_name.replace('_', ' ').title()} Detected",
                                description=f"Potential {pattern_name.replace('_', ' ')} vulnerability found",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                code_snippet=line.strip(),
                                recommendation=self._get_recommendation(pattern_name)
                            )
                            issues.append(issue)

        except Exception as e:
            self.logger.warning(f"Failed to scan {file_path}: {e}")

        return issues

    def _should_exclude(self, file_path: str, code_line: str) -> bool:
        """除外すべきかどうかチェック"""
        # ファイルパス除外
        for pattern in self.exclude_patterns:
            if re.search(pattern, file_path):
                return True

        # コメント行除外
        stripped = code_line.strip()
        if stripped.startswith('#') or stripped.startswith('//'):
            return True

        # テスト・モック関連除外
        if any(keyword in file_path.lower() for keyword in ['test', 'mock', 'fixture', 'conftest']):
            return True

        # ドキュメント内の例除外
        if any(keyword in code_line.lower() for keyword in ['example', 'demo', 'test_', 'mock_']):
            return True

        return False

    def _get_recommendation(self, pattern_name: str) -> str:
        """改善提案を取得"""
        recommendations = {
            "hardcoded_secrets": "環境変数またはセキュアなキー管理サービスを使用してください。",
            "sql_injection": "パラメータ化クエリまたはORMを使用してください。",
            "debug_exposure": "本番環境ではdebug=Falseに設定してください。",
            "weak_crypto": "強力な暗号化アルゴリズム（AES, SHA-256以上）を使用してください。",
            "unsafe_deserialization": "安全なシリアライゼーション形式（JSON）を使用してください。",
            "path_traversal": "ユーザー入力の検証とsecure_filename()を使用してください。",
            "xss_vulnerability": "適切なエスケープ処理を実装してください。"
        }
        return recommendations.get(pattern_name, "セキュリティのベストプラクティスに従ってください。")

    def scan_dependencies(self) -> List[SecurityIssue]:
        """依存関係の脆弱性スキャン"""
        issues = []

        # requirements.txt チェック
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                # safety チェック（もし利用可能なら）
                result = subprocess.run(
                    ["python", "-m", "pip", "check"],
                    capture_output=True, text=True, timeout=30
                )

                if result.returncode != 0 and result.stdout:
                    issue = SecurityIssue(
                        severity="MEDIUM",
                        category="Dependency Management",
                        title="Dependency Conflict Detected",
                        description="Pip dependency conflicts found",
                        file_path="requirements.txt",
                        line_number=0,
                        code_snippet=result.stdout,
                        recommendation="依存関係の競合を解決してください。"
                    )
                    issues.append(issue)

            except Exception as e:
                self.logger.warning(f"Dependency check failed: {e}")

        return issues

    def check_file_permissions(self) -> List[SecurityIssue]:
        """ファイル権限チェック"""
        issues = []

        sensitive_files = [
            "*.key", "*.pem", "*.crt", "*.p12",
            "config/*.json", "config/*.yaml", "config/*.yml",
            ".env", ".env.*", "secrets.*"
        ]

        for pattern in sensitive_files:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    try:
                        # Windows環境では権限チェック簡略化
                        if sys.platform != 'win32':
                            stat_info = file_path.stat()
                            permissions = oct(stat_info.st_mode)[-3:]

                            if permissions != '600':  # 所有者読み書きのみ
                                issue = SecurityIssue(
                                    severity="MEDIUM",
                                    category="File Permissions",
                                    title="Insecure File Permissions",
                                    description=f"Sensitive file has permissive permissions: {permissions}",
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=0,
                                    code_snippet=f"Permissions: {permissions}",
                                    recommendation="chmod 600でファイル権限を制限してください。"
                                )
                                issues.append(issue)
                    except Exception as e:
                        self.logger.warning(f"Permission check failed for {file_path}: {e}")

        return issues

    def scan_project(self) -> SecurityReport:
        """プロジェクト全体のセキュリティスキャン"""
        self.logger.info("🔍 Starting comprehensive security audit...")

        report = SecurityReport(
            scan_date=datetime.now(),
            total_files_scanned=0
        )

        # Python ファイルスキャン
        python_files = list(self.project_root.rglob("*.py"))
        total_files = len(python_files)

        self.logger.info(f"📂 Scanning {total_files} Python files...")

        for i, file_path in enumerate(python_files):
            if i % 50 == 0:
                self.logger.info(f"Progress: {i}/{total_files} files")

            file_issues = self.scan_file(file_path)
            report.issues_found.extend(file_issues)

        report.total_files_scanned = total_files

        # 設定ファイルスキャン
        config_files = list(self.project_root.rglob("*.json")) + \
                      list(self.project_root.rglob("*.yaml")) + \
                      list(self.project_root.rglob("*.yml"))

        for file_path in config_files:
            file_issues = self.scan_file(file_path)
            report.issues_found.extend(file_issues)

        # 依存関係チェック
        self.logger.info("📦 Checking dependencies...")
        dependency_issues = self.scan_dependencies()
        report.issues_found.extend(dependency_issues)

        # ファイル権限チェック
        self.logger.info("🔒 Checking file permissions...")
        permission_issues = self.check_file_permissions()
        report.issues_found.extend(permission_issues)

        # 統計作成
        report.summary = self._create_summary(report.issues_found)
        report.recommendations = self._create_recommendations(report.issues_found)

        self.logger.info(f"✅ Security audit completed. Found {len(report.issues_found)} issues.")

        return report

    def _create_summary(self, issues: List[SecurityIssue]) -> Dict[str, int]:
        """サマリー統計作成"""
        summary = {
            "total_issues": len(issues),
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }

        categories = {}

        for issue in issues:
            # 重要度別
            summary[issue.severity.lower()] += 1

            # カテゴリ別
            if issue.category not in categories:
                categories[issue.category] = 0
            categories[issue.category] += 1

        summary["categories"] = categories
        return summary

    def _create_recommendations(self, issues: List[SecurityIssue]) -> List[str]:
        """改善提案作成"""
        recommendations = []

        # 重要度順の推奨事項
        if any(issue.severity == "CRITICAL" for issue in issues):
            recommendations.append("🚨 CRITICAL: 秘匿情報のハードコーディングを即座に修正してください")

        if any(issue.severity == "HIGH" for issue in issues):
            recommendations.append("⚠️ HIGH: SQLインジェクションやパストラバーサル脆弱性を優先的に修正してください")

        # カテゴリ別推奨事項
        categories = {}
        for issue in issues:
            if issue.category not in categories:
                categories[issue.category] = []
            categories[issue.category].append(issue)

        if "Credential Management" in categories:
            recommendations.append("🔑 環境変数またはキー管理サービス（AWS Secrets Manager等）を導入してください")

        if "Information Disclosure" in categories:
            recommendations.append("🔍 本番環境でのデバッグ情報露出を防止してください")

        if "Injection Vulnerability" in categories:
            recommendations.append("💉 パラメータ化クエリとORMを使用してください")

        # 全般的な推奨事項
        recommendations.extend([
            "🛡️ セキュリティテストの自動化を検討してください",
            "📋 定期的なセキュリティ監査を実施してください",
            "🔒 アクセス制御とログ監視を強化してください"
        ])

        return recommendations

    def generate_report(self, report: SecurityReport, output_file: Optional[Path] = None) -> str:
        """レポート生成"""
        if output_file is None:
            output_file = self.project_root / f"security_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # JSON レポート
        report_data = {
            "scan_date": report.scan_date.isoformat(),
            "total_files_scanned": report.total_files_scanned,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "title": issue.title,
                    "description": issue.description,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "code_snippet": issue.code_snippet,
                    "recommendation": issue.recommendation
                }
                for issue in report.issues_found
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # コンソール レポート
        console_report = self._generate_console_report(report)

        return console_report

    def _generate_console_report(self, report: SecurityReport) -> str:
        """コンソール用レポート生成"""
        lines = []
        lines.append("=" * 80)
        lines.append("🛡️ Day Trade Personal - Security Audit Report")
        lines.append("=" * 80)
        lines.append(f"📅 Scan Date: {report.scan_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"📂 Files Scanned: {report.total_files_scanned}")
        lines.append(f"🔍 Total Issues: {report.summary['total_issues']}")
        lines.append("")

        # サマリー
        lines.append("📊 Issue Summary:")
        lines.append(f"  🚨 CRITICAL: {report.summary['critical']}")
        lines.append(f"  ⚠️  HIGH:     {report.summary['high']}")
        lines.append(f"  🔶 MEDIUM:   {report.summary['medium']}")
        lines.append(f"  ℹ️  LOW:      {report.summary['low']}")
        lines.append("")

        # カテゴリ別
        if 'categories' in report.summary:
            lines.append("📂 Issues by Category:")
            for category, count in report.summary['categories'].items():
                lines.append(f"  • {category}: {count}")
            lines.append("")

        # 重要な問題のリスト
        critical_issues = [i for i in report.issues_found if i.severity == "CRITICAL"]
        if critical_issues:
            lines.append("🚨 CRITICAL Issues (immediate action required):")
            for issue in critical_issues[:5]:  # Top 5
                lines.append(f"  • {issue.file_path}:{issue.line_number} - {issue.title}")
            lines.append("")

        high_issues = [i for i in report.issues_found if i.severity == "HIGH"]
        if high_issues:
            lines.append("⚠️ HIGH Priority Issues:")
            for issue in high_issues[:5]:  # Top 5
                lines.append(f"  • {issue.file_path}:{issue.line_number} - {issue.title}")
            lines.append("")

        # 推奨事項
        lines.append("💡 Recommendations:")
        for rec in report.recommendations:
            lines.append(f"  {rec}")
        lines.append("")

        lines.append("=" * 80)
        lines.append("📋 For detailed report, see: security_audit_report_*.json")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Day Trade Personal - Security Audit Tool")
    parser.add_argument("--output", "-o", help="Output report file", type=Path)
    parser.add_argument("--format", choices=["console", "json", "both"],
                       default="both", help="Report format")

    args = parser.parse_args()

    project_root = Path(__file__).parent
    auditor = SecurityAuditor(project_root)

    # セキュリティスキャン実行
    report = auditor.scan_project()

    # レポート生成
    console_report = auditor.generate_report(report, args.output)

    if args.format in ["console", "both"]:
        print(console_report)

    if args.format in ["json", "both"]:
        print(f"\n📋 Detailed JSON report saved to: {args.output or 'security_audit_report_*.json'}")


if __name__ == "__main__":
    main()