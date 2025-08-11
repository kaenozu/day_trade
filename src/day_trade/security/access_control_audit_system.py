"""
アクセス制御定期監査システム
Issue #419: セキュリティ対策の強化と脆弱性管理プロセスの確立

ロールベースアクセス制御（RBAC）の定期監査、権限昇格検出、
最小権限の原則に基づいたアクセス制御監査システム。
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AccessLevel(Enum):
    """アクセスレベル"""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class AuditRiskLevel(Enum):
    """監査リスクレベル"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PermissionType(Enum):
    """権限タイプ"""

    SYSTEM = "system"
    DATA = "data"
    USER_MANAGEMENT = "user_management"
    FINANCIAL = "financial"
    SECURITY = "security"
    ADMIN = "admin"


@dataclass
class UserRole:
    """ユーザーロール定義"""

    role_id: str
    role_name: str
    description: str
    permissions: List[str] = field(default_factory=list)
    access_level: AccessLevel = AccessLevel.READ
    permission_type: PermissionType = PermissionType.DATA
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class UserAccess:
    """ユーザーアクセス情報"""

    user_id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)
    direct_permissions: List[str] = field(default_factory=list)
    last_login: Optional[datetime] = None
    login_count: int = 0
    failed_login_attempts: int = 0
    account_locked: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class AccessAuditFinding:
    """アクセス監査所見"""

    finding_id: str
    user_id: str
    username: str
    finding_type: str
    risk_level: AuditRiskLevel
    title: str
    description: str
    evidence: Dict[str, Any]
    recommendations: List[str]
    status: str = "open"  # open, acknowledged, remediated, dismissed
    detected_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    remediation_notes: Optional[str] = None


@dataclass
class AccessPattern:
    """アクセスパターン分析結果"""

    user_id: str
    pattern_type: str
    frequency: int
    last_occurrence: datetime
    risk_score: float
    anomaly_indicators: List[str] = field(default_factory=list)


class AccessControlAuditRules:
    """アクセス制御監査ルール定義"""

    @staticmethod
    def get_audit_rules() -> List[Dict[str, Any]]:
        """監査ルール一覧を取得"""
        return [
            {
                "rule_id": "excessive_permissions",
                "name": "過剰権限検出",
                "description": "ユーザーの役割に不必要な権限が付与されている",
                "risk_level": AuditRiskLevel.HIGH,
                "check_function": "_check_excessive_permissions",
            },
            {
                "rule_id": "dormant_accounts",
                "name": "休眠アカウント検出",
                "description": "長期間使用されていないアカウント",
                "risk_level": AuditRiskLevel.MEDIUM,
                "check_function": "_check_dormant_accounts",
            },
            {
                "rule_id": "privilege_escalation",
                "name": "権限昇格検出",
                "description": "短期間での権限昇格",
                "risk_level": AuditRiskLevel.CRITICAL,
                "check_function": "_check_privilege_escalation",
            },
            {
                "rule_id": "shared_accounts",
                "name": "共有アカウント検出",
                "description": "複数ユーザーで使用されている疑いのあるアカウント",
                "risk_level": AuditRiskLevel.HIGH,
                "check_function": "_check_shared_accounts",
            },
            {
                "rule_id": "admin_without_mfa",
                "name": "MFA未設定管理者",
                "description": "多要素認証が設定されていない管理者アカウント",
                "risk_level": AuditRiskLevel.CRITICAL,
                "check_function": "_check_admin_without_mfa",
            },
            {
                "rule_id": "temporal_access_violation",
                "name": "時間外アクセス",
                "description": "営業時間外の異常なアクセス",
                "risk_level": AuditRiskLevel.MEDIUM,
                "check_function": "_check_temporal_violations",
            },
            {
                "rule_id": "role_segregation",
                "name": "職務分離違反",
                "description": "相反する役割の同時保有",
                "risk_level": AuditRiskLevel.HIGH,
                "check_function": "_check_role_segregation",
            },
            {
                "rule_id": "expired_access",
                "name": "期限切れアクセス",
                "description": "有効期限が切れた権限の継続使用",
                "risk_level": AuditRiskLevel.HIGH,
                "check_function": "_check_expired_access",
            },
        ]


class AccessControlAuditor:
    """アクセス制御監査システム"""

    def __init__(self, db_path: str = "access_control_audit.db"):
        self.db_path = db_path
        self.audit_rules = AccessControlAuditRules.get_audit_rules()
        self._initialize_database()
        self.logger = logging.getLogger(__name__)

    def _initialize_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_roles (
                    role_id TEXT PRIMARY KEY,
                    role_name TEXT NOT NULL,
                    description TEXT,
                    permissions TEXT,
                    access_level TEXT NOT NULL,
                    permission_type TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_access (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    email TEXT NOT NULL,
                    roles TEXT,
                    direct_permissions TEXT,
                    last_login DATETIME,
                    login_count INTEGER DEFAULT 0,
                    failed_login_attempts INTEGER DEFAULT 0,
                    account_locked BOOLEAN DEFAULT FALSE,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_findings (
                    finding_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    finding_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    evidence TEXT,
                    recommendations TEXT,
                    status TEXT DEFAULT 'open',
                    detected_at DATETIME NOT NULL,
                    acknowledged_at DATETIME,
                    acknowledged_by TEXT,
                    remediation_notes TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS access_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    frequency INTEGER,
                    last_occurrence DATETIME,
                    risk_score REAL,
                    anomaly_indicators TEXT,
                    analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at DATETIME NOT NULL,
                    completed_at DATETIME,
                    users_audited INTEGER DEFAULT 0,
                    findings_detected INTEGER DEFAULT 0,
                    audit_successful BOOLEAN DEFAULT FALSE
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_findings_risk ON audit_findings(risk_level)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_findings_status ON audit_findings(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_access_active ON user_access(is_active)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_access_patterns_user ON access_patterns(user_id)"
            )

            conn.commit()

    def add_user(self, user_access: UserAccess) -> str:
        """ユーザーを追加"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_access
                (user_id, username, email, roles, direct_permissions, last_login,
                 login_count, failed_login_attempts, account_locked, created_at,
                 updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_access.user_id,
                    user_access.username,
                    user_access.email,
                    json.dumps(user_access.roles),
                    json.dumps(user_access.direct_permissions),
                    user_access.last_login.isoformat()
                    if user_access.last_login
                    else None,
                    user_access.login_count,
                    user_access.failed_login_attempts,
                    user_access.account_locked,
                    user_access.created_at.isoformat(),
                    user_access.updated_at.isoformat(),
                    user_access.is_active,
                ),
            )
            conn.commit()
        return user_access.user_id

    def add_role(self, role: UserRole) -> str:
        """ロールを追加"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_roles
                (role_id, role_name, description, permissions, access_level,
                 permission_type, created_at, updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    role.role_id,
                    role.role_name,
                    role.description,
                    json.dumps(role.permissions),
                    role.access_level.value,
                    role.permission_type.value,
                    role.created_at.isoformat(),
                    role.updated_at.isoformat(),
                    role.is_active,
                ),
            )
            conn.commit()
        return role.role_id

    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """包括的アクセス制御監査を実行"""
        session_id = f"audit_{int(datetime.utcnow().timestamp())}"
        started_at = datetime.utcnow()

        # セッション開始を記録
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_sessions (session_id, started_at)
                VALUES (?, ?)
            """,
                (session_id, started_at.isoformat()),
            )
            conn.commit()

        all_findings = []
        users_audited = 0

        try:
            # 全ユーザーを取得
            users = self._get_all_users()

            for user in users:
                user_findings = self._audit_user(user)
                all_findings.extend(user_findings)
                users_audited += 1

            # パターン分析
            patterns = self._analyze_access_patterns()

            completed_at = datetime.utcnow()

            # セッション完了を記録
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE audit_sessions
                    SET completed_at = ?, users_audited = ?, findings_detected = ?, audit_successful = TRUE
                    WHERE session_id = ?
                """,
                    (
                        completed_at.isoformat(),
                        users_audited,
                        len(all_findings),
                        session_id,
                    ),
                )
                conn.commit()

            return {
                "session_id": session_id,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "users_audited": users_audited,
                "findings_detected": len(all_findings),
                "findings": all_findings,
                "access_patterns": patterns,
                "audit_successful": True,
            }

        except Exception as e:
            # エラーの場合
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE audit_sessions
                    SET completed_at = ?, audit_successful = FALSE
                    WHERE session_id = ?
                """,
                    (datetime.utcnow().isoformat(), session_id),
                )
                conn.commit()

            return {
                "session_id": session_id,
                "started_at": started_at.isoformat(),
                "users_audited": users_audited,
                "findings_detected": 0,
                "findings": [],
                "audit_successful": False,
                "error": str(e),
            }

    def _get_all_users(self) -> List[UserAccess]:
        """全ユーザーを取得"""
        users = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT user_id, username, email, roles, direct_permissions,
                       last_login, login_count, failed_login_attempts, account_locked,
                       created_at, updated_at, is_active
                FROM user_access
                WHERE is_active = TRUE
            """
            )

            for row in cursor.fetchall():
                user = UserAccess(
                    user_id=row[0],
                    username=row[1],
                    email=row[2],
                    roles=json.loads(row[3]) if row[3] else [],
                    direct_permissions=json.loads(row[4]) if row[4] else [],
                    last_login=datetime.fromisoformat(row[5]) if row[5] else None,
                    login_count=row[6],
                    failed_login_attempts=row[7],
                    account_locked=bool(row[8]),
                    created_at=datetime.fromisoformat(row[9]),
                    updated_at=datetime.fromisoformat(row[10]),
                    is_active=bool(row[11]),
                )
                users.append(user)

        return users

    def _audit_user(self, user: UserAccess) -> List[AccessAuditFinding]:
        """個別ユーザーの監査"""
        findings = []

        for rule in self.audit_rules:
            try:
                check_function = getattr(self, rule["check_function"])
                rule_findings = check_function(user)
                findings.extend(rule_findings)
            except Exception as e:
                self.logger.error(f"監査ルール {rule['rule_id']} の実行に失敗: {e}")

        # 所見をデータベースに保存
        for finding in findings:
            self._save_finding(finding)

        return findings

    def _check_excessive_permissions(
        self, user: UserAccess
    ) -> List[AccessAuditFinding]:
        """過剰権限チェック"""
        findings = []

        # ユーザーの全権限を取得
        all_permissions = set(user.direct_permissions)

        # ロールからの権限も追加
        for role_id in user.roles:
            role = self._get_role(role_id)
            if role:
                all_permissions.update(role.permissions)

        # 過剰権限の判定（簡単な例：10個以上の権限）
        if len(all_permissions) > 10:
            finding = AccessAuditFinding(
                finding_id=f"excessive_perms_{user.user_id}_{int(datetime.utcnow().timestamp())}",
                user_id=user.user_id,
                username=user.username,
                finding_type="excessive_permissions",
                risk_level=AuditRiskLevel.HIGH,
                title="過剰権限の検出",
                description=f"ユーザー {user.username} に {len(all_permissions)} 個の権限が付与されています",
                evidence={
                    "total_permissions": len(all_permissions),
                    "permissions": list(all_permissions),
                    "roles": user.roles,
                },
                recommendations=[
                    "最小権限の原則に基づき、不要な権限を削除",
                    "役割ベースの権限管理を見直し",
                    "定期的な権限レビューの実施",
                ],
            )
            findings.append(finding)

        return findings

    def _check_dormant_accounts(self, user: UserAccess) -> List[AccessAuditFinding]:
        """休眠アカウントチェック"""
        findings = []

        if user.last_login:
            days_inactive = (datetime.utcnow() - user.last_login).days
            if days_inactive > 90:  # 90日以上ログインなし
                finding = AccessAuditFinding(
                    finding_id=f"dormant_{user.user_id}_{int(datetime.utcnow().timestamp())}",
                    user_id=user.user_id,
                    username=user.username,
                    finding_type="dormant_accounts",
                    risk_level=AuditRiskLevel.MEDIUM,
                    title="休眠アカウントの検出",
                    description=f"ユーザー {user.username} は {days_inactive} 日間ログインしていません",
                    evidence={
                        "last_login": user.last_login.isoformat(),
                        "days_inactive": days_inactive,
                        "login_count": user.login_count,
                    },
                    recommendations=[
                        "アカウントの無効化を検討",
                        "ユーザーの現在の在籍状況を確認",
                        "権限の一時停止",
                    ],
                )
                findings.append(finding)
        else:
            # 一度もログインしていない
            days_since_creation = (datetime.utcnow() - user.created_at).days
            if days_since_creation > 30:
                finding = AccessAuditFinding(
                    finding_id=f"never_login_{user.user_id}_{int(datetime.utcnow().timestamp())}",
                    user_id=user.user_id,
                    username=user.username,
                    finding_type="dormant_accounts",
                    risk_level=AuditRiskLevel.MEDIUM,
                    title="未使用アカウントの検出",
                    description=f"ユーザー {user.username} は作成後一度もログインしていません",
                    evidence={
                        "created_at": user.created_at.isoformat(),
                        "days_since_creation": days_since_creation,
                        "login_count": user.login_count,
                    },
                    recommendations=[
                        "アカウントの削除を検討",
                        "ユーザー登録プロセスの見直し",
                    ],
                )
                findings.append(finding)

        return findings

    def _check_privilege_escalation(self, user: UserAccess) -> List[AccessAuditFinding]:
        """権限昇格チェック"""
        findings = []

        # 管理者権限を持つユーザーをチェック
        admin_roles = ["admin", "super_admin", "security_admin"]
        has_admin_role = any(role in admin_roles for role in user.roles)

        if has_admin_role:
            account_age = (datetime.utcnow() - user.created_at).days
            if account_age < 30:  # 30日未満で管理者権限
                finding = AccessAuditFinding(
                    finding_id=f"quick_escalation_{user.user_id}_{int(datetime.utcnow().timestamp())}",
                    user_id=user.user_id,
                    username=user.username,
                    finding_type="privilege_escalation",
                    risk_level=AuditRiskLevel.CRITICAL,
                    title="短期間での権限昇格",
                    description=f"ユーザー {user.username} はアカウント作成後 {account_age} 日で管理者権限を取得",
                    evidence={
                        "created_at": user.created_at.isoformat(),
                        "account_age_days": account_age,
                        "admin_roles": [
                            role for role in user.roles if role in admin_roles
                        ],
                    },
                    recommendations=[
                        "権限付与プロセスの見直し",
                        "管理者権限付与の承認フローの強化",
                        "一時的な権限付与の検討",
                    ],
                )
                findings.append(finding)

        return findings

    def _check_shared_accounts(self, user: UserAccess) -> List[AccessAuditFinding]:
        """共有アカウントチェック"""
        findings = []

        # 共通のパターン（service, shared, generic等）をチェック
        shared_patterns = ["service", "shared", "generic", "common", "test", "demo"]

        if any(pattern in user.username.lower() for pattern in shared_patterns):
            finding = AccessAuditFinding(
                finding_id=f"shared_account_{user.user_id}_{int(datetime.utcnow().timestamp())}",
                user_id=user.user_id,
                username=user.username,
                finding_type="shared_accounts",
                risk_level=AuditRiskLevel.HIGH,
                title="共有アカウントの検出",
                description=f"ユーザー名 {user.username} は共有アカウントの可能性があります",
                evidence={
                    "username": user.username,
                    "login_count": user.login_count,
                    "matched_patterns": [
                        p for p in shared_patterns if p in user.username.lower()
                    ],
                },
                recommendations=[
                    "個別アカウントへの移行",
                    "共有アカウントの使用停止",
                    "監査ログの強化",
                ],
            )
            findings.append(finding)

        return findings

    def _check_admin_without_mfa(self, user: UserAccess) -> List[AccessAuditFinding]:
        """MFA未設定管理者チェック"""
        findings = []

        admin_roles = ["admin", "super_admin", "security_admin"]
        has_admin_role = any(role in admin_roles for role in user.roles)

        if has_admin_role:
            # 実際のMFA設定状態は外部システムとの連携が必要
            # ここでは簡単な例として実装
            finding = AccessAuditFinding(
                finding_id=f"admin_no_mfa_{user.user_id}_{int(datetime.utcnow().timestamp())}",
                user_id=user.user_id,
                username=user.username,
                finding_type="admin_without_mfa",
                risk_level=AuditRiskLevel.CRITICAL,
                title="MFA未設定の管理者アカウント",
                description=f"管理者 {user.username} に多要素認証が設定されていません",
                evidence={
                    "admin_roles": [role for role in user.roles if role in admin_roles],
                    "mfa_enabled": False,  # 実装時は実際の状態を確認
                },
                recommendations=[
                    "多要素認証の即座の設定",
                    "管理者アカウントのMFA必須化ポリシー",
                    "アクセス権限の一時停止",
                ],
            )
            findings.append(finding)

        return findings

    def _check_temporal_violations(self, user: UserAccess) -> List[AccessAuditFinding]:
        """時間外アクセスチェック"""
        findings = []

        if user.last_login:
            # 営業時間外（夜間・週末）のログインをチェック
            last_login_hour = user.last_login.hour
            last_login_weekday = user.last_login.weekday()

            # 深夜（22時〜6時）または週末のアクセス
            if (last_login_hour < 6 or last_login_hour > 22) or last_login_weekday >= 5:
                finding = AccessAuditFinding(
                    finding_id=f"temporal_violation_{user.user_id}_{int(datetime.utcnow().timestamp())}",
                    user_id=user.user_id,
                    username=user.username,
                    finding_type="temporal_access_violation",
                    risk_level=AuditRiskLevel.MEDIUM,
                    title="営業時間外アクセス",
                    description=f"ユーザー {user.username} が営業時間外にアクセスしました",
                    evidence={
                        "last_login": user.last_login.isoformat(),
                        "login_hour": last_login_hour,
                        "login_weekday": last_login_weekday,
                    },
                    recommendations=[
                        "業務外アクセスの正当性確認",
                        "時間ベースのアクセス制御の検討",
                        "監視アラートの設定",
                    ],
                )
                findings.append(finding)

        return findings

    def _check_role_segregation(self, user: UserAccess) -> List[AccessAuditFinding]:
        """職務分離違反チェック"""
        findings = []

        # 相反する役割の組み合わせ
        conflicting_roles = [
            (["financial_admin", "audit_admin"], "財務管理と監査"),
            (["developer", "production_admin"], "開発者と本番管理"),
            (["security_admin", "user_admin"], "セキュリティ管理とユーザー管理"),
        ]

        for role_pair, description in conflicting_roles:
            if all(role in user.roles for role in role_pair):
                finding = AccessAuditFinding(
                    finding_id=f"role_conflict_{user.user_id}_{hash(''.join(role_pair))}",
                    user_id=user.user_id,
                    username=user.username,
                    finding_type="role_segregation",
                    risk_level=AuditRiskLevel.HIGH,
                    title="職務分離違反",
                    description=f"ユーザー {user.username} が相反する役割を保有: {description}",
                    evidence={"conflicting_roles": role_pair, "all_roles": user.roles},
                    recommendations=[
                        "役割の分離",
                        "職務分離ポリシーの見直し",
                        "代替承認フローの実装",
                    ],
                )
                findings.append(finding)

        return findings

    def _check_expired_access(self, user: UserAccess) -> List[AccessAuditFinding]:
        """期限切れアクセスチェック"""
        findings = []

        # アカウント作成から長期間経過している場合
        account_age = (datetime.utcnow() - user.created_at).days
        if account_age > 365:  # 1年以上
            finding = AccessAuditFinding(
                finding_id=f"expired_access_{user.user_id}_{int(datetime.utcnow().timestamp())}",
                user_id=user.user_id,
                username=user.username,
                finding_type="expired_access",
                risk_level=AuditRiskLevel.HIGH,
                title="長期間継続するアクセス権",
                description=f"ユーザー {user.username} のアカウントが {account_age} 日間継続しています",
                evidence={
                    "created_at": user.created_at.isoformat(),
                    "account_age_days": account_age,
                },
                recommendations=[
                    "アクセス権の定期更新",
                    "ユーザーの現在の職責確認",
                    "権限の再認証",
                ],
            )
            findings.append(finding)

        return findings

    def _analyze_access_patterns(self) -> List[AccessPattern]:
        """アクセスパターン分析"""
        patterns = []

        # 簡単なパターン分析の例
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT user_id, COUNT(*) as login_frequency,
                       MAX(last_login) as latest_login,
                       failed_login_attempts
                FROM user_access
                WHERE is_active = TRUE AND last_login IS NOT NULL
                GROUP BY user_id
            """
            )

            for row in cursor.fetchall():
                user_id, login_freq, latest_login, failed_attempts = row

                # リスクスコア計算（簡単な例）
                risk_score = 0.0
                anomaly_indicators = []

                if failed_attempts > 5:
                    risk_score += 0.3
                    anomaly_indicators.append("多数の失敗ログイン試行")

                if login_freq > 100:
                    risk_score += 0.2
                    anomaly_indicators.append("異常に高いログイン頻度")

                pattern = AccessPattern(
                    user_id=user_id,
                    pattern_type="login_behavior",
                    frequency=login_freq,
                    last_occurrence=datetime.fromisoformat(latest_login),
                    risk_score=risk_score,
                    anomaly_indicators=anomaly_indicators,
                )
                patterns.append(pattern)

                # パターンをデータベースに保存
                conn.execute(
                    """
                    INSERT INTO access_patterns
                    (user_id, pattern_type, frequency, last_occurrence, risk_score, anomaly_indicators)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern.user_id,
                        pattern.pattern_type,
                        pattern.frequency,
                        pattern.last_occurrence.isoformat(),
                        pattern.risk_score,
                        json.dumps(pattern.anomaly_indicators),
                    ),
                )

            conn.commit()

        return patterns

    def _get_role(self, role_id: str) -> Optional[UserRole]:
        """ロール情報を取得"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT role_id, role_name, description, permissions, access_level,
                       permission_type, created_at, updated_at, is_active
                FROM user_roles
                WHERE role_id = ? AND is_active = TRUE
            """,
                (role_id,),
            )

            row = cursor.fetchone()
            if row:
                return UserRole(
                    role_id=row[0],
                    role_name=row[1],
                    description=row[2] or "",
                    permissions=json.loads(row[3]) if row[3] else [],
                    access_level=AccessLevel(row[4]),
                    permission_type=PermissionType(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    is_active=bool(row[8]),
                )
        return None

    def _save_finding(self, finding: AccessAuditFinding):
        """監査所見をデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_findings
                (finding_id, user_id, username, finding_type, risk_level, title,
                 description, evidence, recommendations, status, detected_at,
                 acknowledged_at, acknowledged_by, remediation_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    finding.finding_id,
                    finding.user_id,
                    finding.username,
                    finding.finding_type,
                    finding.risk_level.value,
                    finding.title,
                    finding.description,
                    json.dumps(finding.evidence),
                    json.dumps(finding.recommendations),
                    finding.status,
                    finding.detected_at.isoformat(),
                    finding.acknowledged_at.isoformat()
                    if finding.acknowledged_at
                    else None,
                    finding.acknowledged_by,
                    finding.remediation_notes,
                ),
            )
            conn.commit()

    def get_findings(
        self,
        risk_level: Optional[AuditRiskLevel] = None,
        status: str = "open",
        limit: int = 100,
    ) -> List[AccessAuditFinding]:
        """監査所見を取得"""
        query = "SELECT * FROM audit_findings WHERE 1=1"
        params = []

        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level.value)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        findings = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)

            for row in cursor.fetchall():
                finding = AccessAuditFinding(
                    finding_id=row[0],
                    user_id=row[1],
                    username=row[2],
                    finding_type=row[3],
                    risk_level=AuditRiskLevel(row[4]),
                    title=row[5],
                    description=row[6] or "",
                    evidence=json.loads(row[7]) if row[7] else {},
                    recommendations=json.loads(row[8]) if row[8] else [],
                    status=row[9],
                    detected_at=datetime.fromisoformat(row[10]),
                    acknowledged_at=datetime.fromisoformat(row[11])
                    if row[11]
                    else None,
                    acknowledged_by=row[12],
                    remediation_notes=row[13],
                )
                findings.append(finding)

        return findings

    def acknowledge_finding(
        self, finding_id: str, acknowledged_by: str, notes: str = ""
    ):
        """監査所見を確認済みとしてマーク"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE audit_findings
                SET status = 'acknowledged', acknowledged_at = ?, acknowledged_by = ?, remediation_notes = ?
                WHERE finding_id = ?
            """,
                (datetime.utcnow().isoformat(), acknowledged_by, notes, finding_id),
            )
            conn.commit()

    def get_audit_summary(self) -> Dict[str, Any]:
        """監査サマリーを取得"""
        with sqlite3.connect(self.db_path) as conn:
            # リスクレベル別統計
            cursor = conn.execute(
                """
                SELECT risk_level, COUNT(*)
                FROM audit_findings
                WHERE status = 'open'
                GROUP BY risk_level
            """
            )
            risk_stats = dict(cursor.fetchall())

            # 所見タイプ別統計
            cursor = conn.execute(
                """
                SELECT finding_type, COUNT(*)
                FROM audit_findings
                WHERE status = 'open'
                GROUP BY finding_type
            """
            )
            type_stats = dict(cursor.fetchall())

            # 最新監査情報
            cursor = conn.execute(
                """
                SELECT MAX(completed_at) as latest_audit,
                       SUM(users_audited) as total_users,
                       SUM(findings_detected) as total_findings
                FROM audit_sessions
                WHERE audit_successful = 1
            """
            )
            audit_info = cursor.fetchone()

            return {
                "risk_summary": {
                    "critical": risk_stats.get("critical", 0),
                    "high": risk_stats.get("high", 0),
                    "medium": risk_stats.get("medium", 0),
                    "low": risk_stats.get("low", 0),
                },
                "finding_types": type_stats,
                "audit_info": {
                    "latest_audit": audit_info[0] if audit_info[0] else None,
                    "total_users_audited": audit_info[1] or 0,
                    "total_findings": audit_info[2] or 0,
                },
                "total_open_findings": sum(risk_stats.values()),
                "last_updated": datetime.utcnow().isoformat(),
            }


# グローバルインスタンス
_access_control_auditor = None


def get_access_control_auditor() -> AccessControlAuditor:
    """グローバルアクセス制御監査システムを取得"""
    global _access_control_auditor
    if _access_control_auditor is None:
        _access_control_auditor = AccessControlAuditor()
    return _access_control_auditor
