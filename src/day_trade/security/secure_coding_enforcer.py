"""
セキュアコーディング規約チェックシステム
Issue #419: セキュリティ対策の強化と脆弱性管理プロセスの確立

OWASP Top 10等のセキュリティガイドラインに基づく
セキュアコーディングプラクティスの自動チェック・強制システム。
"""

import ast
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SecurityRuleCategory(Enum):
    """セキュリティルールカテゴリ"""

    INJECTION = "injection"
    AUTHENTICATION = "authentication"
    SENSITIVE_DATA = "sensitive_data"
    CRYPTOGRAPHY = "cryptography"
    INPUT_VALIDATION = "input_validation"
    ERROR_HANDLING = "error_handling"
    LOGGING = "logging"
    ACCESS_CONTROL = "access_control"


class ViolationSeverity(Enum):
    """違反重要度"""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityRule:
    """セキュリティルール定義"""

    id: str
    name: str
    description: str
    category: SecurityRuleCategory
    severity: ViolationSeverity
    pattern: str  # 正規表現パターン
    remediation: str
    owasp_references: List[str] = field(default_factory=list)
    cwe_ids: List[int] = field(default_factory=list)
    enabled: bool = True


@dataclass
class SecurityViolation:
    """セキュリティ違反"""

    id: str
    rule_id: str
    rule_name: str
    file_path: str
    line_number: int
    column_number: int
    code_snippet: str
    message: str
    severity: ViolationSeverity
    category: SecurityRuleCategory
    detected_at: datetime
    status: str = "open"  # open, acknowledged, fixed, false_positive
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None


class SecureCodingRules:
    """セキュアコーディングルール定義"""

    @staticmethod
    def get_default_rules() -> List[SecurityRule]:
        """デフォルトセキュリティルール一覧"""
        return [
            # 1. インジェクション攻撃対策
            SecurityRule(
                id="sql_injection_check",
                name="SQLインジェクション対策",
                description="動的SQL構築の検出",
                category=SecurityRuleCategory.INJECTION,
                severity=ViolationSeverity.CRITICAL,
                pattern=r'(execute|executemany|query)\s*\(\s*["\'].*%.*["\']|f["\'].*{.*}.*["\'].*\+.*["\']',
                remediation="パラメータ化クエリまたはORM使用を推奨",
                owasp_references=["A01:2021 – Broken Access Control"],
                cwe_ids=[89, 564],
            ),
            SecurityRule(
                id="command_injection_check",
                name="コマンドインジェクション対策",
                description="危険なコマンド実行パターンの検出",
                category=SecurityRuleCategory.INJECTION,
                severity=ViolationSeverity.HIGH,
                pattern=r"(os\.system|subprocess\.call|subprocess\.run|subprocess\.Popen).*shell=True",
                remediation="shell=Falseを使用し、入力値をサニタイズ",
                owasp_references=["A01:2021 – Broken Access Control"],
                cwe_ids=[78],
            ),
            # 2. 認証・認可
            SecurityRule(
                id="hardcoded_password_check",
                name="ハードコードパスワード検出",
                description="ソースコード内のハードコードされたパスワード",
                category=SecurityRuleCategory.AUTHENTICATION,
                severity=ViolationSeverity.HIGH,
                pattern=r'password\s*=\s*["\'][^"\']{6,}["\']|passwd\s*=\s*["\'][^"\']{6,}["\']',
                remediation="環境変数または秘密管理システムを使用",
                owasp_references=[
                    "A07:2021 – Identification and Authentication Failures"
                ],
                cwe_ids=[259, 798],
            ),
            SecurityRule(
                id="weak_crypto_check",
                name="弱い暗号化アルゴリズム",
                description="脆弱な暗号化アルゴリズムの使用",
                category=SecurityRuleCategory.CRYPTOGRAPHY,
                severity=ViolationSeverity.MEDIUM,
                pattern=r"(MD5|SHA1|DES|3DES|RC4)\s*\(",
                remediation="SHA-256以上またはAES暗号化を使用",
                owasp_references=["A02:2021 – Cryptographic Failures"],
                cwe_ids=[326, 327],
            ),
            # 3. 機密データ保護
            SecurityRule(
                id="sensitive_data_logging",
                name="機密データログ出力",
                description="機密データのログ出力検出",
                category=SecurityRuleCategory.SENSITIVE_DATA,
                severity=ViolationSeverity.MEDIUM,
                pattern=r"(log|print|logger)\s*.*\b(password|passwd|secret|token|key|ssn|credit_card)\b",
                remediation="機密データをログに出力せず、マスキングを実装",
                owasp_references=[
                    "A09:2021 – Security Logging and Monitoring Failures"
                ],
                cwe_ids=[532, 200],
            ),
            SecurityRule(
                id="api_key_exposure",
                name="APIキー露出",
                description="ソースコード内のAPIキー露出",
                category=SecurityRuleCategory.SENSITIVE_DATA,
                severity=ViolationSeverity.HIGH,
                pattern=r'(api_key|apikey|access_token|secret_key)\s*=\s*["\'][A-Za-z0-9+/]{20,}["\']',
                remediation="環境変数または秘密管理システムを使用",
                owasp_references=["A02:2021 – Cryptographic Failures"],
                cwe_ids=[798],
            ),
            # 4. 入力値検証
            SecurityRule(
                id="input_validation_check",
                name="入力値検証不備",
                description="直接的なユーザー入力の使用",
                category=SecurityRuleCategory.INPUT_VALIDATION,
                severity=ViolationSeverity.MEDIUM,
                pattern=r'request\.(args|form|json)\[["\'][^"\']+["\']\](?!\s*\.|.*validate)',
                remediation="入力値の検証・サニタイズを実装",
                owasp_references=["A03:2021 – Injection"],
                cwe_ids=[20, 79],
            ),
            # 5. エラーハンドリング
            SecurityRule(
                id="information_disclosure",
                name="情報漏洩リスク",
                description="詳細なエラー情報の露出",
                category=SecurityRuleCategory.ERROR_HANDLING,
                severity=ViolationSeverity.LOW,
                pattern=r"except.*:\s*\n\s*(print|return|raise).*traceback|except.*as.*:\s*\n\s*return.*str\(.*\)",
                remediation="一般的なエラーメッセージを使用し、詳細はログのみに出力",
                owasp_references=[
                    "A09:2021 – Security Logging and Monitoring Failures"
                ],
                cwe_ids=[209, 497],
            ),
            # 6. アクセス制御
            SecurityRule(
                id="privilege_escalation",
                name="権限昇格リスク",
                description="管理者権限の不適切な使用",
                category=SecurityRuleCategory.ACCESS_CONTROL,
                severity=ViolationSeverity.HIGH,
                pattern=r"(sudo|root|administrator|admin).*=.*True|is_admin\s*=\s*True",
                remediation="最小権限の原則に従い、必要時のみ権限を付与",
                owasp_references=["A01:2021 – Broken Access Control"],
                cwe_ids=[269, 284],
            ),
            # 7. ロギング・監査
            SecurityRule(
                id="audit_logging_missing",
                name="監査ログ不備",
                description="重要な操作の監査ログが不足",
                category=SecurityRuleCategory.LOGGING,
                severity=ViolationSeverity.MEDIUM,
                pattern=r"def\s+(login|logout|delete|admin|create_user|change_password)\s*\([^)]*\):\s*(?!.*log)",
                remediation="重要な操作には監査ログを実装",
                owasp_references=[
                    "A09:2021 – Security Logging and Monitoring Failures"
                ],
                cwe_ids=[778],
            ),
        ]


class PythonASTAnalyzer(ast.NodeVisitor):
    """Python AST（抽象構文木）解析によるセキュリティチェック"""

    def __init__(self, file_path: str, source_code: str):
        self.file_path = file_path
        self.source_code = source_code
        self.lines = source_code.split("\n")
        self.violations = []
        self.current_line = 1

    def analyze(self) -> List[SecurityViolation]:
        """AST解析を実行"""
        try:
            tree = ast.parse(self.source_code)
            self.visit(tree)
        except SyntaxError:
            # 構文エラーがある場合はスキップ
            pass

        return self.violations

    def visit_Call(self, node: ast.Call):
        """関数呼び出しの検査"""
        # 危険な関数呼び出しパターンをチェック
        if isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in ["os", "subprocess"]
                and node.func.attr in ["system", "call", "run", "Popen"]
            ):
                # shell=Trueをチェック
                for keyword in node.keywords:
                    if (
                        keyword.arg == "shell"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        self._add_violation(
                            rule_id="command_injection_check",
                            node=node,
                            message="危険なshell=Trueの使用が検出されました",
                        )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """変数代入の検査"""
        # ハードコードされたパスワードのチェック
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id.lower()
                if (
                    "password" in var_name
                    or "passwd" in var_name
                    or "secret" in var_name
                ):
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str
                    ):
                        if len(node.value.value) > 5:  # 5文字以上の文字列
                            self._add_violation(
                                rule_id="hardcoded_password_check",
                                node=node,
                                message=f"ハードコードされた機密情報: {var_name}",
                            )

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """関数定義の検査"""
        func_name = node.name.lower()

        # 重要な操作での監査ログチェック
        if any(
            keyword in func_name
            for keyword in ["login", "logout", "delete", "admin", "create_user"]
        ):
            # 関数内にログ出力があるかチェック
            has_logging = False
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr
                    in ["info", "warning", "error", "debug", "critical"]
                ):
                    has_logging = True
                    break

            if not has_logging:
                self._add_violation(
                    rule_id="audit_logging_missing",
                    node=node,
                    message=f"重要な操作 '{func_name}' に監査ログが不足",
                )

        self.generic_visit(node)

    def _add_violation(self, rule_id: str, node: ast.AST, message: str):
        """違反を追加"""
        # ルール情報を取得
        rules = {rule.id: rule for rule in SecureCodingRules.get_default_rules()}
        rule = rules.get(rule_id)

        if not rule:
            return

        line_number = getattr(node, "lineno", 1)
        column_number = getattr(node, "col_offset", 0)

        # コードスニペットを取得
        code_snippet = ""
        if 1 <= line_number <= len(self.lines):
            code_snippet = self.lines[line_number - 1].strip()

        violation = SecurityViolation(
            id=f"{rule_id}_{hash(self.file_path + str(line_number))}",
            rule_id=rule_id,
            rule_name=rule.name,
            file_path=self.file_path,
            line_number=line_number,
            column_number=column_number,
            code_snippet=code_snippet,
            message=message,
            severity=rule.severity,
            category=rule.category,
            detected_at=datetime.utcnow(),
        )

        self.violations.append(violation)


class RegexSecurityChecker:
    """正規表現ベースのセキュリティチェッカー"""

    def __init__(self):
        self.rules = {rule.id: rule for rule in SecureCodingRules.get_default_rules()}

    def check_file(self, file_path: str, content: str) -> List[SecurityViolation]:
        """ファイルのセキュリティチェック"""
        violations = []
        lines = content.split("\n")

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            pattern = re.compile(rule.pattern, re.IGNORECASE | re.MULTILINE)

            for line_num, line in enumerate(lines, 1):
                matches = pattern.finditer(line)

                for match in matches:
                    violation = SecurityViolation(
                        id=f"{rule.id}_{hash(file_path + str(line_num) + str(match.start()))}",
                        rule_id=rule.id,
                        rule_name=rule.name,
                        file_path=file_path,
                        line_number=line_num,
                        column_number=match.start(),
                        code_snippet=line.strip(),
                        message=f"{rule.description}: {match.group(0)}",
                        severity=rule.severity,
                        category=rule.category,
                        detected_at=datetime.utcnow(),
                    )
                    violations.append(violation)

        return violations


class SecureCodingEnforcer:
    """セキュアコーディング規約チェック・強制システム"""

    def __init__(self, db_path: str = "secure_coding_violations.db"):
        self.db_path = db_path
        self.regex_checker = RegexSecurityChecker()
        self.rules = {rule.id: rule for rule in SecureCodingRules.get_default_rules()}
        self._initialize_database()

    def _initialize_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_violations (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    column_number INTEGER,
                    code_snippet TEXT,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    detected_at DATETIME NOT NULL,
                    status TEXT DEFAULT 'open',
                    assigned_to TEXT,
                    resolution_notes TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at DATETIME NOT NULL,
                    completed_at DATETIME,
                    files_scanned INTEGER DEFAULT 0,
                    violations_found INTEGER DEFAULT 0,
                    scan_successful BOOLEAN DEFAULT FALSE
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_severity ON security_violations(severity)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_status ON security_violations(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_file ON security_violations(file_path)"
            )

            conn.commit()

    def scan_directory(
        self, directory_path: str, extensions: List[str] = None
    ) -> Dict[str, Any]:
        """ディレクトリ全体のセキュリティスキャン"""
        if extensions is None:
            extensions = [".py"]

        session_id = f"scan_{int(datetime.utcnow().timestamp())}"
        started_at = datetime.utcnow()

        # セッション開始を記録
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO scan_sessions (session_id, started_at)
                VALUES (?, ?)
            """,
                (session_id, started_at.isoformat()),
            )
            conn.commit()

        try:
            directory = Path(directory_path)
            all_violations = []
            files_scanned = 0

            for ext in extensions:
                for file_path in directory.rglob(f"*{ext}"):
                    if self._should_skip_file(file_path):
                        continue

                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()

                        # 正規表現チェック
                        regex_violations = self.regex_checker.check_file(
                            str(file_path), content
                        )

                        # ASTチェック（Pythonファイルのみ）
                        if ext == ".py":
                            ast_analyzer = PythonASTAnalyzer(str(file_path), content)
                            ast_violations = ast_analyzer.analyze()
                            regex_violations.extend(ast_violations)

                        # 違反をデータベースに保存
                        for violation in regex_violations:
                            self._save_violation(violation)
                            all_violations.append(violation)

                        files_scanned += 1

                    except Exception as e:
                        print(f"ファイルスキャンエラー {file_path}: {e}")
                        continue

            completed_at = datetime.utcnow()

            # セッション完了を記録
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE scan_sessions
                    SET completed_at = ?, files_scanned = ?, violations_found = ?, scan_successful = TRUE
                    WHERE session_id = ?
                """,
                    (
                        completed_at.isoformat(),
                        files_scanned,
                        len(all_violations),
                        session_id,
                    ),
                )
                conn.commit()

            return {
                "session_id": session_id,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "files_scanned": files_scanned,
                "violations_found": len(all_violations),
                "violations": all_violations,
                "scan_successful": True,
            }

        except Exception as e:
            # エラーの場合
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE scan_sessions
                    SET completed_at = ?, scan_successful = FALSE
                    WHERE session_id = ?
                """,
                    (datetime.utcnow().isoformat(), session_id),
                )
                conn.commit()

            return {
                "session_id": session_id,
                "started_at": started_at.isoformat(),
                "files_scanned": 0,
                "violations_found": 0,
                "violations": [],
                "scan_successful": False,
                "error": str(e),
            }

    def _should_skip_file(self, file_path: Path) -> bool:
        """ファイルをスキップするかチェック"""
        skip_patterns = [
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            "test_",
            "_test.py",
            "conftest.py",
            "example",
            "demo",
        ]

        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)

    def _save_violation(self, violation: SecurityViolation):
        """違反をデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO security_violations
                (id, rule_id, rule_name, file_path, line_number, column_number,
                 code_snippet, message, severity, category, detected_at, status,
                 assigned_to, resolution_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    violation.id,
                    violation.rule_id,
                    violation.rule_name,
                    violation.file_path,
                    violation.line_number,
                    violation.column_number,
                    violation.code_snippet,
                    violation.message,
                    violation.severity.value,
                    violation.category.value,
                    violation.detected_at.isoformat(),
                    violation.status,
                    violation.assigned_to,
                    violation.resolution_notes,
                ),
            )
            conn.commit()

    def get_violations(
        self,
        severity: Optional[ViolationSeverity] = None,
        category: Optional[SecurityRuleCategory] = None,
        status: str = "open",
        limit: int = 100,
    ) -> List[SecurityViolation]:
        """違反一覧を取得"""
        query = "SELECT * FROM security_violations WHERE 1=1"
        params = []

        if severity:
            query += " AND severity = ?"
            params.append(severity.value)

        if category:
            query += " AND category = ?"
            params.append(category.value)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        violations = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)

            for row in cursor.fetchall():
                violation = SecurityViolation(
                    id=row[0],
                    rule_id=row[1],
                    rule_name=row[2],
                    file_path=row[3],
                    line_number=row[4],
                    column_number=row[5] or 0,
                    code_snippet=row[6] or "",
                    message=row[7],
                    severity=ViolationSeverity(row[8]),
                    category=SecurityRuleCategory(row[9]),
                    detected_at=datetime.fromisoformat(row[10]),
                    status=row[11],
                    assigned_to=row[12],
                    resolution_notes=row[13],
                )
                violations.append(violation)

        return violations

    def get_security_summary(self) -> Dict[str, Any]:
        """セキュリティ概要を取得"""
        with sqlite3.connect(self.db_path) as conn:
            # 重要度別統計
            cursor = conn.execute(
                """
                SELECT severity, COUNT(*)
                FROM security_violations
                WHERE status = 'open'
                GROUP BY severity
            """
            )
            severity_stats = dict(cursor.fetchall())

            # カテゴリ別統計
            cursor = conn.execute(
                """
                SELECT category, COUNT(*)
                FROM security_violations
                WHERE status = 'open'
                GROUP BY category
            """
            )
            category_stats = dict(cursor.fetchall())

            # 最新スキャン情報
            cursor = conn.execute(
                """
                SELECT MAX(completed_at) as latest_scan,
                       SUM(files_scanned) as total_files,
                       SUM(violations_found) as total_violations
                FROM scan_sessions
                WHERE scan_successful = 1
            """
            )
            scan_info = cursor.fetchone()

            return {
                "severity_summary": {
                    "critical": severity_stats.get("critical", 0),
                    "high": severity_stats.get("high", 0),
                    "medium": severity_stats.get("medium", 0),
                    "low": severity_stats.get("low", 0),
                    "info": severity_stats.get("info", 0),
                },
                "category_summary": category_stats,
                "scan_info": {
                    "latest_scan": scan_info[0] if scan_info[0] else None,
                    "total_files_scanned": scan_info[1] or 0,
                    "total_violations_found": scan_info[2] or 0,
                },
                "total_open_violations": sum(severity_stats.values()),
                "last_updated": datetime.utcnow().isoformat(),
            }


# グローバルインスタンス
_secure_coding_enforcer = None


def get_secure_coding_enforcer() -> SecureCodingEnforcer:
    """グローバルセキュアコーディング規約チェッカーを取得"""
    global _secure_coding_enforcer
    if _secure_coding_enforcer is None:
        _secure_coding_enforcer = SecureCodingEnforcer()
    return _secure_coding_enforcer
