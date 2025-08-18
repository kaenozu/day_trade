#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
セキュリティ監査システム

システムのセキュリティ状態を監視・評価・改善提案を行う
"""

import os
import re
import hashlib
import subprocess
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json


class SecurityLevel(Enum):
    """セキュリティレベル"""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """脆弱性タイプ"""
    HARDCODED_SECRET = "hardcoded_secret"
    WEAK_ENCRYPTION = "weak_encryption"
    INJECTION_RISK = "injection_risk"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXPOSURE = "data_exposure"
    INSECURE_DEPENDENCIES = "insecure_dependencies"
    CONFIGURATION_ERROR = "configuration_error"
    ACCESS_CONTROL = "access_control"


@dataclass
class SecurityFinding:
    """セキュリティ発見事項"""
    id: str
    timestamp: datetime
    file_path: str
    line_number: int
    vulnerability_type: VulnerabilityType
    security_level: SecurityLevel
    title: str
    description: str
    recommendation: str
    code_snippet: Optional[str] = None
    fixed: bool = False


@dataclass
class SecurityReport:
    """セキュリティレポート"""
    timestamp: datetime
    total_files_scanned: int
    findings: List[SecurityFinding]
    security_score: float
    summary: Dict[str, int]
    recommendations: List[str]


class CodeSecurityScanner:
    """コードセキュリティスキャナー"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 危険なパターンの定義
        self.security_patterns = {
            VulnerabilityType.HARDCODED_SECRET: [
                (r'password\s*=\s*["\'][^"\']{8,}["\']', SecurityLevel.HIGH_RISK),
                (r'api_key\s*=\s*["\'][^"\']{20,}["\']', SecurityLevel.HIGH_RISK),
                (r'secret\s*=\s*["\'][^"\']{16,}["\']', SecurityLevel.HIGH_RISK),
                (r'token\s*=\s*["\'][^"\']{32,}["\']', SecurityLevel.HIGH_RISK),
                (r'["\'][A-Za-z0-9]{32,}["\']', SecurityLevel.MEDIUM_RISK),
            ],
            VulnerabilityType.WEAK_ENCRYPTION: [
                (r'md5\(', SecurityLevel.MEDIUM_RISK),
                (r'sha1\(', SecurityLevel.MEDIUM_RISK),
                (r'\.md5\(\)', SecurityLevel.MEDIUM_RISK),
                (r'hashlib\.md5', SecurityLevel.MEDIUM_RISK),
                (r'base64\.b64encode\(.*password', SecurityLevel.LOW_RISK),
            ],
            VulnerabilityType.INJECTION_RISK: [
                (r'execute\(.*%.*\)', SecurityLevel.HIGH_RISK),
                (r'os\.system\(.*input\(', SecurityLevel.CRITICAL),
                (r'subprocess\..*shell=True', SecurityLevel.MEDIUM_RISK),
                (r'eval\(.*input\(', SecurityLevel.CRITICAL),
                (r'exec\(.*input\(', SecurityLevel.CRITICAL),
            ],
            VulnerabilityType.DATA_EXPOSURE: [
                (r'print\(.*password', SecurityLevel.MEDIUM_RISK),
                (r'print\(.*secret', SecurityLevel.MEDIUM_RISK),
                (r'logger\.info\(.*password', SecurityLevel.MEDIUM_RISK),
                (r'debug.*=.*True', SecurityLevel.LOW_RISK),
            ],
            VulnerabilityType.INSECURE_DEPENDENCIES: [
                (r'requests\.get\(.*verify=False', SecurityLevel.HIGH_RISK),
                (r'ssl\..*CERT_NONE', SecurityLevel.HIGH_RISK),
                (r'urllib3\.disable_warnings', SecurityLevel.MEDIUM_RISK),
            ]
        }
        
        # 安全なファイル拡張子
        self.scannable_extensions = {'.py', '.json', '.yaml', '.yml', '.ini', '.cfg', '.conf'}
        
        # 除外パターン
        self.exclude_patterns = [
            r'__pycache__',
            r'\.git',
            r'\.pytest_cache',
            r'venv',
            r'env',
            r'node_modules',
            r'test_.*\.py',  # テストファイルは一部制限を緩和
        ]
    
    def scan_directory(self, directory: Path) -> List[SecurityFinding]:
        """ディレクトリをスキャン"""
        findings = []
        
        for file_path in directory.rglob('*'):
            if self._should_scan_file(file_path):
                file_findings = self._scan_file(file_path)
                findings.extend(file_findings)
        
        return findings
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """ファイルをスキャンすべきかチェック"""
        # ディレクトリは除外
        if file_path.is_dir():
            return False
        
        # 拡張子チェック
        if file_path.suffix not in self.scannable_extensions:
            return False
        
        # 除外パターンチェック
        path_str = str(file_path)
        for pattern in self.exclude_patterns:
            if re.search(pattern, path_str):
                return False
        
        return True
    
    def _scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """ファイルをスキャン"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line_findings = self._scan_line(
                    file_path, line_num, line.strip()
                )
                findings.extend(line_findings)
                
        except Exception as e:
            self.logger.warning(f"ファイルスキャンエラー {file_path}: {e}")
        
        return findings
    
    def _scan_line(self, file_path: Path, line_num: int, line: str) -> List[SecurityFinding]:
        """行をスキャン"""
        findings = []
        
        for vuln_type, patterns in self.security_patterns.items():
            for pattern, security_level in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    finding = self._create_finding(
                        file_path, line_num, line, vuln_type, 
                        security_level, pattern
                    )
                    findings.append(finding)
        
        return findings
    
    def _create_finding(self, 
                       file_path: Path, 
                       line_num: int, 
                       line: str,
                       vuln_type: VulnerabilityType,
                       security_level: SecurityLevel,
                       pattern: str) -> SecurityFinding:
        """セキュリティ発見事項を作成"""
        
        finding_id = hashlib.md5(
            f"{file_path}:{line_num}:{pattern}".encode()
        ).hexdigest()[:16]
        
        # 脆弱性タイプ別の説明とレコメンデーション
        descriptions = {
            VulnerabilityType.HARDCODED_SECRET: {
                "title": "ハードコードされたシークレット",
                "description": "ソースコードに直接埋め込まれたパスワードやAPIキーが検出されました",
                "recommendation": "環境変数や設定ファイルを使用してシークレットを外部化してください"
            },
            VulnerabilityType.WEAK_ENCRYPTION: {
                "title": "弱い暗号化",
                "description": "安全でない暗号化アルゴリズムの使用が検出されました",
                "recommendation": "SHA-256以上の強力なハッシュアルゴリズムを使用してください"
            },
            VulnerabilityType.INJECTION_RISK: {
                "title": "インジェクション脆弱性",
                "description": "コードインジェクション攻撃の可能性がある実装が検出されました",
                "recommendation": "入力値の検証とサニタイゼーションを実装してください"
            },
            VulnerabilityType.DATA_EXPOSURE: {
                "title": "データ露出リスク",
                "description": "機密データがログやコンソールに出力される可能性があります",
                "recommendation": "機密データのログ出力を避け、必要な場合はマスキングしてください"
            },
            VulnerabilityType.INSECURE_DEPENDENCIES: {
                "title": "安全でない依存関係設定",
                "description": "セキュリティ機能を無効化する設定が検出されました",
                "recommendation": "SSL証明書検証やセキュリティ警告を無効化しないでください"
            }
        }
        
        info = descriptions.get(vuln_type, {
            "title": "セキュリティ問題",
            "description": "潜在的なセキュリティリスクが検出されました",
            "recommendation": "コードを見直してセキュリティを強化してください"
        })
        
        return SecurityFinding(
            id=finding_id,
            timestamp=datetime.now(),
            file_path=str(file_path),
            line_number=line_num,
            vulnerability_type=vuln_type,
            security_level=security_level,
            title=info["title"],
            description=info["description"],
            recommendation=info["recommendation"],
            code_snippet=line[:200]  # 最初の200文字のみ
        )


class DependencyScanner:
    """依存関係スキャナー"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.known_vulnerabilities = {
            # 既知の脆弱な依存関係（例）
            "requests": {"<2.20.0": SecurityLevel.MEDIUM_RISK},
            "urllib3": {"<1.24.2": SecurityLevel.HIGH_RISK},
            "pyyaml": {"<5.1": SecurityLevel.HIGH_RISK},
        }
    
    def scan_requirements(self, requirements_file: Path) -> List[SecurityFinding]:
        """requirements.txtをスキャン"""
        findings = []
        
        if not requirements_file.exists():
            return findings
        
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    finding = self._check_dependency(requirements_file, line_num, line)
                    if finding:
                        findings.append(finding)
                        
        except Exception as e:
            self.logger.warning(f"依存関係スキャンエラー: {e}")
        
        return findings
    
    def _check_dependency(self, file_path: Path, line_num: int, line: str) -> Optional[SecurityFinding]:
        """依存関係をチェック"""
        # パッケージ名とバージョンを抽出
        match = re.match(r'([a-zA-Z0-9_-]+)([>=<]+)([0-9.]+)', line)
        if not match:
            return None
        
        package_name, operator, version = match.groups()
        
        # 既知の脆弱性をチェック
        if package_name.lower() in self.known_vulnerabilities:
            vuln_info = self.known_vulnerabilities[package_name.lower()]
            for vuln_version, security_level in vuln_info.items():
                if self._version_matches_vulnerability(version, vuln_version):
                    return SecurityFinding(
                        id=f"dep_{package_name}_{version}",
                        timestamp=datetime.now(),
                        file_path=str(file_path),
                        line_number=line_num,
                        vulnerability_type=VulnerabilityType.INSECURE_DEPENDENCIES,
                        security_level=security_level,
                        title=f"脆弱な依存関係: {package_name}",
                        description=f"{package_name} {version} には既知の脆弱性があります",
                        recommendation=f"{package_name}を最新バージョンにアップデートしてください",
                        code_snippet=line
                    )
        
        return None
    
    def _version_matches_vulnerability(self, current_version: str, vuln_pattern: str) -> bool:
        """バージョンが脆弱性パターンにマッチするかチェック"""
        # 簡単な実装（実際にはより複雑なバージョン比較が必要）
        if vuln_pattern.startswith('<'):
            target_version = vuln_pattern[1:]
            return current_version < target_version
        return False


class SecurityAuditSystem:
    """セキュリティ監査システム"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        self.code_scanner = CodeSecurityScanner()
        self.dependency_scanner = DependencyScanner()
        
        # 監査履歴
        self.audit_history: List[SecurityReport] = []
        self.max_history_size = 50
        
        self.logger.info("セキュリティ監査システム初期化完了")
    
    def run_full_audit(self) -> SecurityReport:
        """完全なセキュリティ監査を実行"""
        self.logger.info("セキュリティ監査開始")
        start_time = time.time()
        
        # コードスキャン
        code_findings = self.code_scanner.scan_directory(self.project_root)
        
        # 依存関係スキャン
        requirements_file = self.project_root / "requirements.txt"
        dependency_findings = self.dependency_scanner.scan_requirements(requirements_file)
        
        # 設定ファイルスキャン
        config_findings = self._scan_configurations()
        
        # 全ての発見事項を統合
        all_findings = code_findings + dependency_findings + config_findings
        
        # ファイル数カウント
        total_files = len([
            f for f in self.project_root.rglob('*') 
            if f.is_file() and self.code_scanner._should_scan_file(f)
        ])
        
        # セキュリティスコア計算
        security_score = self._calculate_security_score(all_findings, total_files)
        
        # サマリー作成
        summary = self._create_summary(all_findings)
        
        # レコメンデーション生成
        recommendations = self._generate_recommendations(all_findings)
        
        # レポート作成
        report = SecurityReport(
            timestamp=datetime.now(),
            total_files_scanned=total_files,
            findings=all_findings,
            security_score=security_score,
            summary=summary,
            recommendations=recommendations
        )
        
        # 履歴に追加
        self.audit_history.append(report)
        if len(self.audit_history) > self.max_history_size:
            self.audit_history = self.audit_history[-self.max_history_size:]
        
        duration = time.time() - start_time
        self.logger.info(f"セキュリティ監査完了: {duration:.2f}秒, {len(all_findings)}件の発見事項")
        
        return report
    
    def _scan_configurations(self) -> List[SecurityFinding]:
        """設定ファイルをスキャン"""
        findings = []
        
        # よくある設定ファイルをチェック
        config_files = [
            "config/environments/production.json",
            "config/environments/production_enhanced.json",
            ".env",
            "docker-compose.yml",
            "Dockerfile"
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                file_findings = self._scan_config_file(file_path)
                findings.extend(file_findings)
        
        return findings
    
    def _scan_config_file(self, file_path: Path) -> List[SecurityFinding]:
        """設定ファイルをスキャン"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # 危険な設定パターンをチェック
                if re.search(r'"debug"\s*:\s*true', line, re.IGNORECASE):
                    findings.append(SecurityFinding(
                        id=f"config_debug_{file_path.name}_{line_num}",
                        timestamp=datetime.now(),
                        file_path=str(file_path),
                        line_number=line_num,
                        vulnerability_type=VulnerabilityType.CONFIGURATION_ERROR,
                        security_level=SecurityLevel.MEDIUM_RISK,
                        title="デバッグモードが有効",
                        description="本番環境でデバッグモードが有効になっています",
                        recommendation="本番環境ではデバッグモードを無効にしてください",
                        code_snippet=line
                    ))
                
                if re.search(r'"paper_trading"\s*:\s*false', line, re.IGNORECASE):
                    findings.append(SecurityFinding(
                        id=f"config_trading_{file_path.name}_{line_num}",
                        timestamp=datetime.now(),
                        file_path=str(file_path),
                        line_number=line_num,
                        vulnerability_type=VulnerabilityType.CONFIGURATION_ERROR,
                        security_level=SecurityLevel.HIGH_RISK,
                        title="実取引モードが有効",
                        description="実際の取引が実行される可能性があります",
                        recommendation="個人利用版では paper_trading を true に設定してください",
                        code_snippet=line
                    ))
                        
        except Exception as e:
            self.logger.warning(f"設定ファイルスキャンエラー {file_path}: {e}")
        
        return findings
    
    def _calculate_security_score(self, findings: List[SecurityFinding], total_files: int) -> float:
        """セキュリティスコアを計算（0-100）"""
        if not findings:
            return 100.0
        
        # 重要度別の重み
        severity_weights = {
            SecurityLevel.SAFE: 0,
            SecurityLevel.LOW_RISK: 1,
            SecurityLevel.MEDIUM_RISK: 3,
            SecurityLevel.HIGH_RISK: 7,
            SecurityLevel.CRITICAL: 15
        }
        
        # 総減点を計算
        total_deduction = sum(
            severity_weights.get(finding.security_level, 1)
            for finding in findings
        )
        
        # ファイル数で正規化
        if total_files > 0:
            normalized_deduction = (total_deduction / total_files) * 10
        else:
            normalized_deduction = total_deduction
        
        # スコア計算（最低0点）
        score = max(0, 100 - normalized_deduction)
        
        return round(score, 1)
    
    def _create_summary(self, findings: List[SecurityFinding]) -> Dict[str, int]:
        """サマリーを作成"""
        summary = {
            "total": len(findings),
            "by_severity": {},
            "by_type": {}
        }
        
        # 重要度別
        for severity in SecurityLevel:
            count = len([f for f in findings if f.security_level == severity])
            summary["by_severity"][severity.value] = count
        
        # タイプ別
        for vuln_type in VulnerabilityType:
            count = len([f for f in findings if f.vulnerability_type == vuln_type])
            summary["by_type"][vuln_type.value] = count
        
        return summary
    
    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """レコメンデーションを生成"""
        recommendations = set()
        
        # 重要度の高い問題を優先
        critical_findings = [f for f in findings if f.security_level == SecurityLevel.CRITICAL]
        high_risk_findings = [f for f in findings if f.security_level == SecurityLevel.HIGH_RISK]
        
        if critical_findings:
            recommendations.add("🚨 CRITICAL: 緊急対応が必要な重大な脆弱性があります")
        
        if high_risk_findings:
            recommendations.add("⚠️ 高リスクの脆弱性を早急に修正してください")
        
        # タイプ別のレコメンデーション
        type_counts = {}
        for finding in findings:
            type_counts[finding.vulnerability_type] = type_counts.get(finding.vulnerability_type, 0) + 1
        
        if type_counts.get(VulnerabilityType.HARDCODED_SECRET, 0) > 0:
            recommendations.add("🔐 シークレット管理: 環境変数や設定ファイルを使用してください")
        
        if type_counts.get(VulnerabilityType.WEAK_ENCRYPTION, 0) > 0:
            recommendations.add("🔒 暗号化強化: SHA-256以上の強力なアルゴリズムを使用してください")
        
        if type_counts.get(VulnerabilityType.INJECTION_RISK, 0) > 0:
            recommendations.add("🛡️ 入力検証: すべての外部入力を適切に検証してください")
        
        if type_counts.get(VulnerabilityType.INSECURE_DEPENDENCIES, 0) > 0:
            recommendations.add("📦 依存関係更新: 脆弱な依存関係を最新版に更新してください")
        
        # 一般的なレコメンデーション
        recommendations.add("📋 定期監査: セキュリティ監査を定期的に実行してください")
        recommendations.add("🎓 セキュリティ教育: 開発チームのセキュリティ意識を向上させてください")
        
        return sorted(list(recommendations))
    
    def export_report(self, report: SecurityReport, format_type: str = "json") -> str:
        """レポートをエクスポート"""
        if format_type == "json":
            # JSON形式でエクスポート
            export_data = {
                "audit_timestamp": report.timestamp.isoformat(),
                "security_score": report.security_score,
                "total_files_scanned": report.total_files_scanned,
                "summary": report.summary,
                "recommendations": report.recommendations,
                "findings": [
                    {
                        "id": f.id,
                        "timestamp": f.timestamp.isoformat(),
                        "file_path": f.file_path,
                        "line_number": f.line_number,
                        "vulnerability_type": f.vulnerability_type.value,
                        "security_level": f.security_level.value,
                        "title": f.title,
                        "description": f.description,
                        "recommendation": f.recommendation,
                        "code_snippet": f.code_snippet,
                        "fixed": f.fixed
                    }
                    for f in report.findings
                ]
            }
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        
        return f"Unsupported format: {format_type}"


# 使用例とユーティリティ関数
def run_security_audit(project_root: str = ".") -> SecurityReport:
    """セキュリティ監査を実行"""
    audit_system = SecurityAuditSystem(Path(project_root))
    return audit_system.run_full_audit()


def save_security_report(report: SecurityReport, output_file: str):
    """セキュリティレポートを保存"""
    audit_system = SecurityAuditSystem(Path("."))
    report_content = audit_system.export_report(report)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"セキュリティレポートを保存: {output_file}")


if __name__ == "__main__":
    # 単体実行時のテスト
    report = run_security_audit()
    print(f"セキュリティスコア: {report.security_score}/100")
    print(f"発見事項: {len(report.findings)}件")
    
    if report.findings:
        print("\n主要な発見事項:")
        for finding in report.findings[:5]:  # 最初の5件
            print(f"- {finding.title} ({finding.security_level.value})")
    
    save_security_report(report, "security_audit_report.json")