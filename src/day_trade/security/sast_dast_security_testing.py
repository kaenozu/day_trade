"""
SAST/DASTセキュリティテストシステム
Issue #419: セキュリティ対策の強化と脆弱性管理プロセスの確立

静的アプリケーションセキュリティテスト（SAST）と
動的アプリケーションセキュリティテスト（DAST）の統合実行システム。
"""

import asyncio
import json
import logging
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class TestType(Enum):
    """セキュリティテストタイプ"""

    SAST = "sast"
    DAST = "dast"
    IAST = "iast"
    SCA = "sca"  # Software Composition Analysis


class SeverityLevel(Enum):
    """脆弱性重要度"""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestStatus(Enum):
    """テスト状態"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SecurityTestResult:
    """セキュリティテスト結果"""

    result_id: str
    test_type: TestType
    test_name: str
    target: str
    severity: SeverityLevel
    title: str
    description: str
    location: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[int] = None
    owasp_category: Optional[str] = None
    confidence: str = "medium"  # low, medium, high
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "open"


@dataclass
class TestSession:
    """テストセッション"""

    session_id: str
    test_type: TestType
    target: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: TestStatus = TestStatus.PENDING
    results_count: int = 0
    results: List[SecurityTestResult] = field(default_factory=list)
    error_message: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)


class SASTScanner:
    """静的セキュリティテストスキャナー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def run_semgrep_scan(self, target_path: str) -> TestSession:
        """Semgrepによる静的解析"""
        session_id = f"semgrep_{int(datetime.utcnow().timestamp())}"
        session = TestSession(
            session_id=session_id,
            test_type=TestType.SAST,
            target=target_path,
            started_at=datetime.utcnow(),
        )

        try:
            session.status = TestStatus.RUNNING

            # Semgrepコマンドを実行
            result = subprocess.run(
                [
                    "semgrep",
                    "--config=auto",
                    "--json",
                    "--quiet",
                    "--no-git-ignore",
                    target_path,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            session.completed_at = datetime.utcnow()

            if result.returncode == 0:
                # 結果をパース
                try:
                    output = json.loads(result.stdout)
                    results = []

                    for finding in output.get("results", []):
                        test_result = SecurityTestResult(
                            result_id=f"semgrep_{finding.get('check_id', 'unknown')}_{hash(finding.get('path', ''))}",
                            test_type=TestType.SAST,
                            test_name="semgrep",
                            target=finding.get("path", ""),
                            severity=self._parse_semgrep_severity(
                                finding.get("extra", {}).get("severity", "INFO")
                            ),
                            title=finding.get("extra", {}).get(
                                "message", "Security issue detected"
                            ),
                            description=f"Rule: {finding.get('check_id', 'unknown')} - {finding.get('extra', {}).get('message', '')}",
                            location=finding.get("path", ""),
                            line_number=finding.get("start", {}).get("line", 0),
                            owasp_category=self._get_owasp_category(
                                finding.get("check_id", "")
                            ),
                            confidence="high",
                            remediation=finding.get("extra", {}).get("fix", ""),
                            references=[f"Rule ID: {finding.get('check_id', '')}"],
                        )
                        results.append(test_result)

                    session.results = results
                    session.results_count = len(results)
                    session.status = TestStatus.COMPLETED

                except json.JSONDecodeError:
                    session.status = TestStatus.FAILED
                    session.error_message = "Failed to parse Semgrep output"

            else:
                session.status = TestStatus.FAILED
                session.error_message = result.stderr

        except subprocess.TimeoutExpired:
            session.status = TestStatus.FAILED
            session.error_message = "Semgrep scan timeout"
        except Exception as e:
            session.status = TestStatus.FAILED
            session.error_message = str(e)

        return session

    async def run_codeql_scan(self, target_path: str) -> TestSession:
        """CodeQLによる静的解析"""
        session_id = f"codeql_{int(datetime.utcnow().timestamp())}"
        session = TestSession(
            session_id=session_id,
            test_type=TestType.SAST,
            target=target_path,
            started_at=datetime.utcnow(),
        )

        try:
            session.status = TestStatus.RUNNING

            # CodeQLデータベース作成
            db_path = f"/tmp/codeql_db_{session_id}"

            create_result = subprocess.run(
                [
                    "codeql",
                    "database",
                    "create",
                    db_path,
                    f"--source-root={target_path}",
                    "--language=python",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if create_result.returncode == 0:
                # CodeQL解析実行
                analyze_result = subprocess.run(
                    [
                        "codeql",
                        "database",
                        "analyze",
                        db_path,
                        "python-security-and-quality",
                        "--format=json",
                        f"--output=/tmp/codeql_results_{session_id}.json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                session.completed_at = datetime.utcnow()

                if analyze_result.returncode == 0:
                    # 結果を読み込み
                    results_file = f"/tmp/codeql_results_{session_id}.json"
                    with open(results_file) as f:
                        output = json.load(f)

                    results = []
                    for finding in output.get("runs", [{}])[0].get("results", []):
                        rule_id = finding.get("ruleId", "unknown")
                        message = finding.get("message", {}).get("text", "")

                        # 位置情報の取得
                        location_info = finding.get("locations", [{}])[0]
                        file_path = (
                            location_info.get("physicalLocation", {})
                            .get("artifactLocation", {})
                            .get("uri", "")
                        )
                        line_number = (
                            location_info.get("physicalLocation", {})
                            .get("region", {})
                            .get("startLine", 0)
                        )

                        test_result = SecurityTestResult(
                            result_id=f"codeql_{rule_id}_{hash(file_path + str(line_number))}",
                            test_type=TestType.SAST,
                            test_name="codeql",
                            target=file_path,
                            severity=self._parse_codeql_severity(
                                finding.get("level", "note")
                            ),
                            title=rule_id,
                            description=message,
                            location=file_path,
                            line_number=line_number,
                            confidence="high",
                            references=[f"CodeQL Rule: {rule_id}"],
                        )
                        results.append(test_result)

                    session.results = results
                    session.results_count = len(results)
                    session.status = TestStatus.COMPLETED

                else:
                    session.status = TestStatus.FAILED
                    session.error_message = analyze_result.stderr
            else:
                session.status = TestStatus.FAILED
                session.error_message = create_result.stderr

        except subprocess.TimeoutExpired:
            session.status = TestStatus.FAILED
            session.error_message = "CodeQL scan timeout"
        except Exception as e:
            session.status = TestStatus.FAILED
            session.error_message = str(e)

        return session

    def _parse_semgrep_severity(self, severity: str) -> SeverityLevel:
        """Semgrep重要度のパース"""
        severity_mapping = {
            "ERROR": SeverityLevel.HIGH,
            "WARNING": SeverityLevel.MEDIUM,
            "INFO": SeverityLevel.LOW,
        }
        return severity_mapping.get(severity.upper(), SeverityLevel.LOW)

    def _parse_codeql_severity(self, level: str) -> SeverityLevel:
        """CodeQL重要度のパース"""
        level_mapping = {
            "error": SeverityLevel.HIGH,
            "warning": SeverityLevel.MEDIUM,
            "note": SeverityLevel.LOW,
        }
        return level_mapping.get(level.lower(), SeverityLevel.LOW)

    def _get_owasp_category(self, rule_id: str) -> str:
        """ルールIDからOWASPカテゴリを推定"""
        owasp_mapping = {
            "sql-injection": "A03:2021 – Injection",
            "xss": "A03:2021 – Injection",
            "path-traversal": "A01:2021 – Broken Access Control",
            "command-injection": "A03:2021 – Injection",
            "crypto": "A02:2021 – Cryptographic Failures",
            "hardcoded": "A07:2021 – Identification and Authentication Failures",
        }

        rule_lower = rule_id.lower()
        for pattern, category in owasp_mapping.items():
            if pattern in rule_lower:
                return category

        return "A10:2021 – Server-Side Request Forgery"


class DASTScanner:
    """動的セキュリティテストスキャナー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def run_zap_scan(
        self, target_url: str, scan_type: str = "baseline"
    ) -> TestSession:
        """OWASP ZAPによる動的解析"""
        session_id = f"zap_{scan_type}_{int(datetime.utcnow().timestamp())}"
        session = TestSession(
            session_id=session_id,
            test_type=TestType.DAST,
            target=target_url,
            started_at=datetime.utcnow(),
            configuration={"scan_type": scan_type},
        )

        try:
            session.status = TestStatus.RUNNING

            # ZAPスキャンタイプ別のコマンド構築
            if scan_type == "baseline":
                zap_cmd = [
                    "zap-baseline.py",
                    "-t",
                    target_url,
                    "-J",
                    f"/tmp/zap_baseline_{session_id}.json",
                    "-x",
                    f"/tmp/zap_baseline_{session_id}.xml",
                ]
            elif scan_type == "full":
                zap_cmd = [
                    "zap-full-scan.py",
                    "-t",
                    target_url,
                    "-J",
                    f"/tmp/zap_full_{session_id}.json",
                    "-x",
                    f"/tmp/zap_full_{session_id}.xml",
                ]
            else:
                raise ValueError(f"Unsupported scan type: {scan_type}")

            # ZAPスキャン実行
            result = subprocess.run(
                zap_cmd, capture_output=True, text=True, timeout=1800
            )  # 30分タイムアウト

            session.completed_at = datetime.utcnow()

            # ZAPは脆弱性が見つかった場合に0以外のリターンコードを返すため、
            # ファイルの存在をチェック
            json_file = f"/tmp/zap_{scan_type}_{session_id}.json"

            if Path(json_file).exists():
                with open(json_file) as f:
                    output = json.load(f)

                results = []
                for site in output.get("site", []):
                    for alert in site.get("alerts", []):
                        for instance in alert.get("instances", []):
                            test_result = SecurityTestResult(
                                result_id=f"zap_{alert.get('pluginid', 'unknown')}_{hash(instance.get('uri', ''))}",
                                test_type=TestType.DAST,
                                test_name=f"zap_{scan_type}",
                                target=instance.get("uri", ""),
                                severity=self._parse_zap_risk(
                                    alert.get("riskdesc", "Low")
                                ),
                                title=alert.get("name", "Security Alert"),
                                description=alert.get("desc", ""),
                                cwe_id=int(alert.get("cweid", 0))
                                if alert.get("cweid") and alert.get("cweid").isdigit()
                                else None,
                                confidence=alert.get("confidence", "Medium").lower(),
                                remediation=alert.get("solution", ""),
                                references=[
                                    ref.strip()
                                    for ref in alert.get("reference", "").split("\n")
                                    if ref.strip()
                                ],
                            )
                            results.append(test_result)

                session.results = results
                session.results_count = len(results)
                session.status = TestStatus.COMPLETED

            else:
                session.status = TestStatus.FAILED
                session.error_message = (
                    result.stderr or "ZAP scan produced no output file"
                )

        except subprocess.TimeoutExpired:
            session.status = TestStatus.FAILED
            session.error_message = "ZAP scan timeout"
        except Exception as e:
            session.status = TestStatus.FAILED
            session.error_message = str(e)

        return session

    async def run_nikto_scan(self, target_url: str) -> TestSession:
        """Niktoによる脆弱性スキャン"""
        session_id = f"nikto_{int(datetime.utcnow().timestamp())}"
        session = TestSession(
            session_id=session_id,
            test_type=TestType.DAST,
            target=target_url,
            started_at=datetime.utcnow(),
        )

        try:
            session.status = TestStatus.RUNNING

            # Niktoスキャン実行
            result = subprocess.run(
                [
                    "nikto",
                    "-h",
                    target_url,
                    "-Format",
                    "json",
                    "-output",
                    f"/tmp/nikto_{session_id}.json",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            session.completed_at = datetime.utcnow()

            json_file = f"/tmp/nikto_{session_id}.json"
            if Path(json_file).exists():
                with open(json_file) as f:
                    output = json.load(f)

                results = []
                for vuln in output.get("vulnerabilities", []):
                    test_result = SecurityTestResult(
                        result_id=f"nikto_{vuln.get('id', 'unknown')}_{hash(target_url)}",
                        test_type=TestType.DAST,
                        test_name="nikto",
                        target=target_url,
                        severity=SeverityLevel.MEDIUM,  # Niktoは通常MEDIUMレベル
                        title=vuln.get("msg", "Web Server Vulnerability"),
                        description=vuln.get("msg", ""),
                        confidence="medium",
                        references=[vuln.get("OSVDB", "")],
                    )
                    results.append(test_result)

                session.results = results
                session.results_count = len(results)
                session.status = TestStatus.COMPLETED

            else:
                session.status = TestStatus.FAILED
                session.error_message = result.stderr or "Nikto scan produced no output"

        except subprocess.TimeoutExpired:
            session.status = TestStatus.FAILED
            session.error_message = "Nikto scan timeout"
        except Exception as e:
            session.status = TestStatus.FAILED
            session.error_message = str(e)

        return session

    def _parse_zap_risk(self, risk_desc: str) -> SeverityLevel:
        """ZAPリスクレベルのパース"""
        if "High" in risk_desc:
            return SeverityLevel.HIGH
        elif "Medium" in risk_desc:
            return SeverityLevel.MEDIUM
        elif "Low" in risk_desc:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO


class SecurityTestOrchestrator:
    """セキュリティテスト統合オーケストレーター"""

    def __init__(self, db_path: str = "security_test_results.db"):
        self.db_path = db_path
        self.sast_scanner = SASTScanner()
        self.dast_scanner = DASTScanner()
        self._initialize_database()
        self.logger = logging.getLogger(__name__)

    def _initialize_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_sessions (
                    session_id TEXT PRIMARY KEY,
                    test_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    started_at DATETIME NOT NULL,
                    completed_at DATETIME,
                    status TEXT NOT NULL,
                    results_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    configuration TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_results (
                    result_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    target TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    location TEXT,
                    line_number INTEGER,
                    cwe_id INTEGER,
                    owasp_category TEXT,
                    confidence TEXT,
                    remediation TEXT,
                    references TEXT,
                    detected_at DATETIME NOT NULL,
                    status TEXT DEFAULT 'open',
                    FOREIGN KEY (session_id) REFERENCES test_sessions (session_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    campaign_name TEXT NOT NULL,
                    description TEXT,
                    targets TEXT,
                    test_types TEXT,
                    started_at DATETIME NOT NULL,
                    completed_at DATETIME,
                    total_sessions INTEGER DEFAULT 0,
                    total_results INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_results_severity ON test_results(severity)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_results_session ON test_results(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_sessions_type ON test_sessions(test_type)"
            )

            conn.commit()

    async def run_comprehensive_security_test(
        self,
        sast_targets: List[str] = None,
        dast_targets: List[str] = None,
        campaign_name: str = "Comprehensive Security Test",
    ) -> Dict[str, Any]:
        """包括的セキュリティテストの実行"""
        campaign_id = f"campaign_{int(datetime.utcnow().timestamp())}"
        started_at = datetime.utcnow()

        if sast_targets is None:
            sast_targets = ["src/"]
        if dast_targets is None:
            dast_targets = []

        # キャンペーン開始を記録
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO test_campaigns
                (campaign_id, campaign_name, description, targets, test_types, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    campaign_id,
                    campaign_name,
                    "Comprehensive SAST and DAST security testing",
                    json.dumps({"sast": sast_targets, "dast": dast_targets}),
                    json.dumps(["sast", "dast"]),
                    started_at.isoformat(),
                ),
            )
            conn.commit()

        all_sessions = []
        all_results = []

        try:
            # SASTテスト実行
            sast_tasks = []
            for target in sast_targets:
                # Semgrep
                sast_tasks.append(self.sast_scanner.run_semgrep_scan(target))
                # CodeQL（利用可能な場合）
                sast_tasks.append(self.sast_scanner.run_codeql_scan(target))

            # DASTテスト実行
            dast_tasks = []
            for target in dast_targets:
                # ZAP baseline
                dast_tasks.append(self.dast_scanner.run_zap_scan(target, "baseline"))
                # Nikto
                dast_tasks.append(self.dast_scanner.run_nikto_scan(target))

            # 全テストを並列実行
            all_tasks = sast_tasks + dast_tasks
            if all_tasks:
                sessions = await asyncio.gather(*all_tasks, return_exceptions=True)

                for session in sessions:
                    if isinstance(session, TestSession):
                        all_sessions.append(session)
                        all_results.extend(session.results)

                        # セッションをデータベースに保存
                        self._save_session(session)

            completed_at = datetime.utcnow()

            # キャンペーン完了を記録
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE test_campaigns
                    SET completed_at = ?, total_sessions = ?, total_results = ?, status = 'completed'
                    WHERE campaign_id = ?
                """,
                    (
                        completed_at.isoformat(),
                        len(all_sessions),
                        len(all_results),
                        campaign_id,
                    ),
                )
                conn.commit()

            return {
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "total_sessions": len(all_sessions),
                "total_results": len(all_results),
                "sessions": all_sessions,
                "summary": self._generate_test_summary(all_results),
                "status": "completed",
            }

        except Exception as e:
            # エラーの場合
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE test_campaigns
                    SET completed_at = ?, status = 'failed'
                    WHERE campaign_id = ?
                """,
                    (datetime.utcnow().isoformat(), campaign_id),
                )
                conn.commit()

            return {
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "started_at": started_at.isoformat(),
                "total_sessions": len(all_sessions),
                "total_results": len(all_results),
                "status": "failed",
                "error": str(e),
            }

    def _save_session(self, session: TestSession):
        """テストセッションをデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO test_sessions
                (session_id, test_type, target, started_at, completed_at, status,
                 results_count, error_message, configuration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    session.test_type.value,
                    session.target,
                    session.started_at.isoformat(),
                    session.completed_at.isoformat() if session.completed_at else None,
                    session.status.value,
                    session.results_count,
                    session.error_message,
                    json.dumps(session.configuration),
                ),
            )

            # テスト結果を保存
            for result in session.results:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO test_results
                    (result_id, session_id, test_type, test_name, target, severity,
                     title, description, location, line_number, cwe_id, owasp_category,
                     confidence, remediation, references, detected_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result.result_id,
                        session.session_id,
                        result.test_type.value,
                        result.test_name,
                        result.target,
                        result.severity.value,
                        result.title,
                        result.description,
                        result.location,
                        result.line_number,
                        result.cwe_id,
                        result.owasp_category,
                        result.confidence,
                        result.remediation,
                        json.dumps(result.references),
                        result.detected_at.isoformat(),
                        result.status,
                    ),
                )

            conn.commit()

    def _generate_test_summary(
        self, results: List[SecurityTestResult]
    ) -> Dict[str, Any]:
        """テスト結果サマリーを生成"""
        severity_counts = {level.value: 0 for level in SeverityLevel}
        test_type_counts = {test_type.value: 0 for test_type in TestType}
        owasp_categories = {}
        cwe_counts = {}

        for result in results:
            severity_counts[result.severity.value] += 1
            test_type_counts[result.test_type.value] += 1

            if result.owasp_category:
                owasp_categories[result.owasp_category] = (
                    owasp_categories.get(result.owasp_category, 0) + 1
                )

            if result.cwe_id:
                cwe_counts[result.cwe_id] = cwe_counts.get(result.cwe_id, 0) + 1

        return {
            "total_findings": len(results),
            "severity_breakdown": severity_counts,
            "test_type_breakdown": test_type_counts,
            "top_owasp_categories": sorted(
                owasp_categories.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "top_cwe_ids": sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True)[
                :10
            ],
            "risk_score": self._calculate_risk_score(results),
        }

    def _calculate_risk_score(self, results: List[SecurityTestResult]) -> float:
        """リスクスコアを計算"""
        severity_weights = {
            SeverityLevel.CRITICAL: 10,
            SeverityLevel.HIGH: 7,
            SeverityLevel.MEDIUM: 4,
            SeverityLevel.LOW: 2,
            SeverityLevel.INFO: 1,
        }

        total_score = sum(
            severity_weights.get(result.severity, 1) for result in results
        )
        max_possible = len(results) * 10  # 全てCRITICALの場合

        return (total_score / max_possible) * 100 if max_possible > 0 else 0

    def get_test_results(
        self,
        severity: Optional[SeverityLevel] = None,
        test_type: Optional[TestType] = None,
        status: str = "open",
        limit: int = 100,
    ) -> List[SecurityTestResult]:
        """テスト結果を取得"""
        query = "SELECT * FROM test_results WHERE 1=1"
        params = []

        if severity:
            query += " AND severity = ?"
            params.append(severity.value)

        if test_type:
            query += " AND test_type = ?"
            params.append(test_type.value)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        results = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)

            for row in cursor.fetchall():
                result = SecurityTestResult(
                    result_id=row[0],
                    test_type=TestType(row[2]),
                    test_name=row[3],
                    target=row[4],
                    severity=SeverityLevel(row[5]),
                    title=row[6],
                    description=row[7] or "",
                    location=row[8],
                    line_number=row[9],
                    cwe_id=row[10],
                    owasp_category=row[11],
                    confidence=row[12] or "medium",
                    remediation=row[13],
                    references=json.loads(row[14]) if row[14] else [],
                    detected_at=datetime.fromisoformat(row[15]),
                    status=row[16],
                )
                results.append(result)

        return results

    def get_security_test_dashboard(self) -> Dict[str, Any]:
        """セキュリティテストダッシュボード用データを取得"""
        with sqlite3.connect(self.db_path) as conn:
            # 重要度別統計
            cursor = conn.execute(
                """
                SELECT severity, COUNT(*)
                FROM test_results
                WHERE status = 'open'
                GROUP BY severity
            """
            )
            severity_stats = dict(cursor.fetchall())

            # テストタイプ別統計
            cursor = conn.execute(
                """
                SELECT test_type, COUNT(*)
                FROM test_results
                WHERE status = 'open'
                GROUP BY test_type
            """
            )
            type_stats = dict(cursor.fetchall())

            # 最新キャンペーン情報
            cursor = conn.execute(
                """
                SELECT MAX(completed_at) as latest_campaign,
                       SUM(total_sessions) as total_sessions,
                       SUM(total_results) as total_results
                FROM test_campaigns
                WHERE status = 'completed'
            """
            )
            campaign_info = cursor.fetchone()

            return {
                "severity_summary": {
                    "critical": severity_stats.get("critical", 0),
                    "high": severity_stats.get("high", 0),
                    "medium": severity_stats.get("medium", 0),
                    "low": severity_stats.get("low", 0),
                    "info": severity_stats.get("info", 0),
                },
                "test_type_summary": type_stats,
                "campaign_info": {
                    "latest_campaign": campaign_info[0] if campaign_info[0] else None,
                    "total_sessions": campaign_info[1] or 0,
                    "total_results": campaign_info[2] or 0,
                },
                "total_open_findings": sum(severity_stats.values()),
                "last_updated": datetime.utcnow().isoformat(),
            }


# グローバルインスタンス
_security_test_orchestrator = None


def get_security_test_orchestrator() -> SecurityTestOrchestrator:
    """グローバルセキュリティテストオーケストレーターを取得"""
    global _security_test_orchestrator
    if _security_test_orchestrator is None:
        _security_test_orchestrator = SecurityTestOrchestrator()
    return _security_test_orchestrator
