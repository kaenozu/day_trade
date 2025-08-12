#!/usr/bin/env python3
"""
ゼロトラストセキュリティマネージャー
Issue #435: 本番環境セキュリティ最終監査 - エンタープライズレベル保証

ゼロトラストアーキテクチャによる包括的アクセス制御・認証・認可システム
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import jwt  # type: ignore
    import pyotp  # type: ignore
    from cryptography.fernet import Fernet  # type: ignore
    from cryptography.hazmat.primitives import hashes  # type: ignore
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # type: ignore

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class TrustLevel(Enum):
    """トラストレベル"""

    UNKNOWN = "unknown"
    DENIED = "denied"
    LIMITED = "limited"
    CONDITIONAL = "conditional"
    TRUSTED = "trusted"
    HIGH_TRUST = "high_trust"


class AccessDecision(Enum):
    """アクセス決定"""

    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"
    MONITOR = "monitor"


class RiskLevel(Enum):
    """リスクレベル"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ZeroTrustConfig:
    """ゼロトラスト設定"""

    # セキュリティレベル
    minimum_trust_level: TrustLevel = TrustLevel.CONDITIONAL
    require_mfa: bool = True
    session_timeout_minutes: int = 60

    # リスク評価
    enable_behavioral_analysis: bool = True
    enable_device_fingerprinting: bool = True
    enable_geolocation_checks: bool = True

    # 暗号化設定
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None

    # 監視設定
    log_all_requests: bool = True
    alert_on_anomalies: bool = True

    def __post_init__(self):
        if self.encryption_key is None:
            self.encryption_key = (
                Fernet.generate_key().decode() if CRYPTO_AVAILABLE else None
            )
        if self.jwt_secret is None:
            self.jwt_secret = secrets.token_urlsafe(32)


@dataclass
class UserContext:
    """ユーザーコンテキスト"""

    user_id: str
    username: str
    roles: List[str]
    groups: List[str]
    attributes: Dict[str, Any]

    # セッション情報
    session_id: str
    created_at: float
    last_activity: float

    # 認証情報
    authentication_methods: List[str]
    mfa_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceContext:
    """デバイスコンテキスト"""

    device_id: str
    device_type: str  # desktop, mobile, server, etc.
    os_info: str
    browser_info: str
    ip_address: str
    geolocation: Optional[Dict[str, Any]]

    # セキュリティ属性
    trusted_device: bool = False
    device_fingerprint: str = ""
    last_seen: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceContext:
    """リソースコンテキスト"""

    resource_id: str
    resource_type: str  # api, data, service, etc.
    classification: str  # public, internal, confidential, restricted
    required_permissions: List[str]
    data_sensitivity: str  # low, medium, high, critical

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccessRequest:
    """アクセス要求"""

    request_id: str
    user_context: UserContext
    device_context: DeviceContext
    resource_context: ResourceContext
    requested_action: str
    timestamp: float

    # リクエスト詳細
    request_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyRule:
    """ポリシールール"""

    rule_id: str
    name: str
    description: str

    # 条件
    user_conditions: Dict[str, Any]
    device_conditions: Dict[str, Any]
    resource_conditions: Dict[str, Any]
    contextual_conditions: Dict[str, Any]

    # アクション
    decision: AccessDecision
    required_trust_level: TrustLevel
    additional_controls: List[str]

    # メタデータ
    priority: int = 100
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RiskAssessment:
    """リスク評価エンジン"""

    def __init__(self):
        self.risk_factors = {
            "unknown_device": {"weight": 0.3, "base_risk": 0.4},
            "unusual_location": {"weight": 0.25, "base_risk": 0.3},
            "off_hours_access": {"weight": 0.15, "base_risk": 0.2},
            "privileged_access": {"weight": 0.2, "base_risk": 0.3},
            "sensitive_resource": {"weight": 0.35, "base_risk": 0.4},
            "failed_authentications": {"weight": 0.4, "base_risk": 0.5},
            "behavioral_anomaly": {"weight": 0.3, "base_risk": 0.4},
        }

    def assess_risk(
        self, access_request: AccessRequest
    ) -> Tuple[RiskLevel, float, Dict[str, Any]]:
        """リスク評価実行"""
        risk_score = 0.0
        risk_factors_detected = {}

        # デバイスリスク
        if not access_request.device_context.trusted_device:
            factor = self.risk_factors["unknown_device"]
            risk_score += factor["weight"] * factor["base_risk"]
            risk_factors_detected["unknown_device"] = factor["base_risk"]

        # 地理的位置リスク
        if self._is_unusual_location(access_request):
            factor = self.risk_factors["unusual_location"]
            risk_score += factor["weight"] * factor["base_risk"]
            risk_factors_detected["unusual_location"] = factor["base_risk"]

        # 時間ベースリスク
        if self._is_off_hours_access(access_request):
            factor = self.risk_factors["off_hours_access"]
            risk_score += factor["weight"] * factor["base_risk"]
            risk_factors_detected["off_hours_access"] = factor["base_risk"]

        # 権限レベルリスク
        if self._is_privileged_access(access_request):
            factor = self.risk_factors["privileged_access"]
            risk_score += factor["weight"] * factor["base_risk"]
            risk_factors_detected["privileged_access"] = factor["base_risk"]

        # リソース機密性リスク
        if access_request.resource_context.data_sensitivity in ["high", "critical"]:
            factor = self.risk_factors["sensitive_resource"]
            risk_score += factor["weight"] * factor["base_risk"]
            risk_factors_detected["sensitive_resource"] = factor["base_risk"]

        # 認証失敗履歴リスク
        if self._has_recent_auth_failures(access_request):
            factor = self.risk_factors["failed_authentications"]
            risk_score += factor["weight"] * factor["base_risk"]
            risk_factors_detected["failed_authentications"] = factor["base_risk"]

        # 正規化
        risk_score = min(1.0, max(0.0, risk_score))

        # リスクレベル決定
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return risk_level, risk_score, risk_factors_detected

    def _is_unusual_location(self, request: AccessRequest) -> bool:
        """異常な位置からのアクセスかチェック"""
        geo = request.device_context.geolocation
        if not geo:
            return True  # 位置不明は異常とみなす

        # 簡略実装 - 実際は過去のアクセス履歴と比較
        country = geo.get("country", "unknown")
        allowed_countries = ["JP", "US", "GB", "CA", "AU"]  # 許可国を拡張
        return country not in allowed_countries

    def _is_off_hours_access(self, request: AccessRequest) -> bool:
        """営業時間外アクセスかチェック"""
        access_time = datetime.fromtimestamp(request.timestamp)
        hour = access_time.hour

        # 営業時間を9-18時とする（設定可能にする）
        business_start = 9
        business_end = 18
        return hour < business_start or hour > business_end

    def _is_privileged_access(self, request: AccessRequest) -> bool:
        """特権アクセスかチェック"""
        privileged_roles = ["admin", "root", "superuser", "operator"]
        return any(role in privileged_roles for role in request.user_context.roles)

    def _has_recent_auth_failures(self, request: AccessRequest) -> bool:
        """最近の認証失敗があるかチェック"""
        # 簡略実装 - 実際は認証失敗履歴を参照
        # TODO: データベースから認証失敗履歴を取得して判定
        user_id = request.user_context.user_id
        current_time = time.time()
        failure_window = 3600  # 1時間以内の失敗をチェック

        # プレースホルダー: 実際の実装では永続化されたデータを使用
        return False


class PolicyEngine:
    """ポリシーエンジン"""

    def __init__(self):
        self.policies = {}
        self.default_policies = self._create_default_policies()
        self.policies.update(self.default_policies)

    def _create_default_policies(self) -> Dict[str, PolicyRule]:
        """デフォルトポリシー作成"""
        policies = {}

        # デフォルト拒否ポリシー
        policies["default_deny"] = PolicyRule(
            rule_id="default_deny",
            name="Default Deny",
            description="Deny access by default",
            user_conditions={},
            device_conditions={},
            resource_conditions={},
            contextual_conditions={},
            decision=AccessDecision.DENY,
            required_trust_level=TrustLevel.TRUSTED,
            additional_controls=[],
            priority=1000,
            enabled=True,
        )

        # 信頼済みデバイスポリシー
        policies["trusted_device_allow"] = PolicyRule(
            rule_id="trusted_device_allow",
            name="Trusted Device Access",
            description="Allow access from trusted devices",
            user_conditions={"mfa_verified": True},
            device_conditions={"trusted_device": True},
            resource_conditions={"classification": ["public", "internal"]},
            contextual_conditions={},
            decision=AccessDecision.ALLOW,
            required_trust_level=TrustLevel.TRUSTED,
            additional_controls=["session_monitoring"],
            priority=10,
            enabled=True,
        )

        # 高機密リソースポリシー
        policies["confidential_resource_strict"] = PolicyRule(
            rule_id="confidential_resource_strict",
            name="Confidential Resource Access",
            description="Strict access control for confidential resources",
            user_conditions={"roles": ["admin", "security_officer"]},
            device_conditions={"trusted_device": True},
            resource_conditions={"classification": ["confidential", "restricted"]},
            contextual_conditions={"business_hours": True},
            decision=AccessDecision.CHALLENGE,
            required_trust_level=TrustLevel.HIGH_TRUST,
            additional_controls=["additional_mfa", "continuous_monitoring"],
            priority=5,
            enabled=True,
        )

        # 異常検知チャレンジポリシー
        policies["anomaly_challenge"] = PolicyRule(
            rule_id="anomaly_challenge",
            name="Anomaly Challenge",
            description="Challenge suspicious activities",
            user_conditions={},
            device_conditions={},
            resource_conditions={},
            contextual_conditions={"risk_level": ["high", "critical"]},
            decision=AccessDecision.CHALLENGE,
            required_trust_level=TrustLevel.CONDITIONAL,
            additional_controls=["step_up_auth", "manager_approval"],
            priority=20,
            enabled=True,
        )

        return policies

    def evaluate_policies(
        self,
        access_request: AccessRequest,
        risk_assessment: Tuple[RiskLevel, float, Dict[str, Any]],
    ) -> Tuple[AccessDecision, TrustLevel, List[str]]:
        """ポリシー評価実行"""

        risk_level, risk_score, risk_factors = risk_assessment

        # ポリシーを優先度順でソート
        sorted_policies = sorted(self.policies.values(), key=lambda p: p.priority)

        for policy in sorted_policies:
            if not policy.enabled:
                continue

            if self._matches_policy(access_request, policy, risk_level):
                logger.info(f"ポリシーマッチ: {policy.name} -> {policy.decision.value}")

                return (
                    policy.decision,
                    policy.required_trust_level,
                    policy.additional_controls,
                )

        # デフォルトは拒否
        return AccessDecision.DENY, TrustLevel.DENIED, []

    def _matches_policy(
        self, request: AccessRequest, policy: PolicyRule, risk_level: RiskLevel
    ) -> bool:
        """ポリシーマッチング"""

        # ユーザー条件チェック
        if not self._matches_user_conditions(
            request.user_context, policy.user_conditions
        ):
            return False

        # デバイス条件チェック
        if not self._matches_device_conditions(
            request.device_context, policy.device_conditions
        ):
            return False

        # リソース条件チェック
        if not self._matches_resource_conditions(
            request.resource_context, policy.resource_conditions
        ):
            return False

        # コンテキスト条件チェック
        return self._matches_contextual_conditions(
            request, policy.contextual_conditions, risk_level
        )

    def _matches_user_conditions(
        self, user: UserContext, conditions: Dict[str, Any]
    ) -> bool:
        """ユーザー条件マッチング"""
        if not conditions:
            return True

        for key, value in conditions.items():
            if key == "roles" and isinstance(value, list):
                if not any(role in user.roles for role in value):
                    return False
            elif key == "mfa_verified" and isinstance(value, bool):
                if user.mfa_verified != value:
                    return False
            elif key == "groups" and isinstance(value, list):
                if not any(group in user.groups for group in value):
                    return False

        return True

    def _matches_device_conditions(
        self, device: DeviceContext, conditions: Dict[str, Any]
    ) -> bool:
        """デバイス条件マッチング"""
        if not conditions:
            return True

        for key, value in conditions.items():
            if key == "trusted_device" and isinstance(value, bool):
                if device.trusted_device != value:
                    return False
            elif key == "device_type" and isinstance(value, list):
                if device.device_type not in value:
                    return False

        return True

    def _matches_resource_conditions(
        self, resource: ResourceContext, conditions: Dict[str, Any]
    ) -> bool:
        """リソース条件マッチング"""
        if not conditions:
            return True

        for key, value in conditions.items():
            if key == "classification" and isinstance(value, list):
                if resource.classification not in value:
                    return False
            elif key == "data_sensitivity" and isinstance(value, list):
                if resource.data_sensitivity not in value:
                    return False

        return True

    def _matches_contextual_conditions(
        self, request: AccessRequest, conditions: Dict[str, Any], risk_level: RiskLevel
    ) -> bool:
        """コンテキスト条件マッチング"""
        if not conditions:
            return True

        for key, value in conditions.items():
            if key == "risk_level" and isinstance(value, list):
                if risk_level.value not in value:
                    return False
            elif key == "business_hours" and isinstance(value, bool):
                is_business_hours = self._is_business_hours(request.timestamp)
                if is_business_hours != value:
                    return False

        return True

    def _is_business_hours(self, timestamp: float) -> bool:
        """営業時間判定"""
        dt = datetime.fromtimestamp(timestamp)

        # 平日の9-18時を営業時間とする
        if dt.weekday() >= 5:  # 土日
            return False

        business_start = 9
        business_end = 18
        return business_start <= dt.hour <= business_end


class MultiFactorAuthenticator:
    """多要素認証"""

    def __init__(self, config: ZeroTrustConfig):
        self.config = config
        self.totp_secrets = {}  # 実際の実装では永続化が必要

    def setup_totp(self, user_id: str) -> Tuple[str, str]:
        """TOTP設定"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("TOTP requires pyotp library")

        secret = pyotp.random_base32()
        self.totp_secrets[user_id] = secret

        # QRコード用URI生成
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            user_id, issuer_name="DayTrade Security"
        )

        return secret, totp_uri

    def verify_totp(self, user_id: str, token: str) -> bool:
        """TOTP検証"""
        if not CRYPTO_AVAILABLE:
            return False

        secret = self.totp_secrets.get(user_id)
        if not secret:
            return False

        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)  # 30秒の猶予時間

    def generate_backup_codes(self, user_id: str, count: int = 8) -> List[str]:
        """バックアップコード生成"""
        if count <= 0 or count > 20:
            raise ValueError("バックアップコード数は1-20の範囲で指定してください")

        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()  # 8文字のコード
            codes.append(code)

        # 実際の実装では暗号化して保存
        logger.info(f"バックアップコード生成: ユーザー={user_id}, 数={count}")
        return codes


class SessionManager:
    """セッション管理"""

    def __init__(self, config: ZeroTrustConfig):
        self.config = config
        self.active_sessions = {}
        self.session_cipher = None

        if CRYPTO_AVAILABLE and config.encryption_key:
            self.session_cipher = Fernet(config.encryption_key.encode())

    def create_session(
        self, user_context: UserContext, device_context: DeviceContext
    ) -> str:
        """セッション作成"""
        session_token = secrets.token_urlsafe(32)

        session_data = {
            "user_context": user_context.to_dict(),
            "device_context": device_context.to_dict(),
            "created_at": time.time(),
            "last_activity": time.time(),
            "trust_level": TrustLevel.CONDITIONAL.value,
        }

        # セッションデータ暗号化
        if self.session_cipher:
            encrypted_data = self.session_cipher.encrypt(
                json.dumps(session_data).encode()
            )
            self.active_sessions[session_token] = encrypted_data
        else:
            self.active_sessions[session_token] = session_data

        logger.info(f"新規セッション作成: {user_context.user_id}")
        return session_token

    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """セッション検証"""
        if session_token not in self.active_sessions:
            return None

        try:
            # セッションデータ復号化
            if self.session_cipher:
                encrypted_data = self.active_sessions[session_token]
                decrypted_data = self.session_cipher.decrypt(encrypted_data)
                session_data = json.loads(decrypted_data.decode())
            else:
                session_data = self.active_sessions[session_token]

            # セッションタイムアウトチェック
            last_activity = session_data["last_activity"]
            timeout_threshold = time.time() - (self.config.session_timeout_minutes * 60)

            if last_activity < timeout_threshold:
                self.revoke_session(session_token)
                return None

            # 最終活動時刻更新
            session_data["last_activity"] = time.time()

            if self.session_cipher:
                encrypted_data = self.session_cipher.encrypt(
                    json.dumps(session_data).encode()
                )
                self.active_sessions[session_token] = encrypted_data
            else:
                self.active_sessions[session_token] = session_data

            return session_data

        except Exception as e:
            logger.error(f"セッション検証エラー: {e}")
            self.revoke_session(session_token)
            return None

    def revoke_session(self, session_token: str) -> bool:
        """セッション無効化"""
        if session_token in self.active_sessions:
            del self.active_sessions[session_token]
            logger.info(f"セッション無効化: {session_token[:8]}...")
            return True
        return False

    def cleanup_expired_sessions(self):
        """期限切れセッションクリーンアップ"""
        current_time = time.time()
        timeout_threshold = current_time - (self.config.session_timeout_minutes * 60)

        expired_sessions = []

        for token, data in self.active_sessions.items():
            try:
                if self.session_cipher:
                    decrypted_data = self.session_cipher.decrypt(data)
                    session_data = json.loads(decrypted_data.decode())
                else:
                    session_data = data

                if session_data["last_activity"] < timeout_threshold:
                    expired_sessions.append(token)

            except Exception:
                expired_sessions.append(token)  # 破損セッション

        for token in expired_sessions:
            self.revoke_session(token)

        logger.info(f"期限切れセッションクリーンアップ: {len(expired_sessions)}件")


class ZeroTrustManager:
    """ゼロトラストセキュリティマネージャー"""

    def __init__(self, config: Optional[ZeroTrustConfig] = None):
        self.config = config or ZeroTrustConfig()
        self.risk_assessor = RiskAssessment()
        self.policy_engine = PolicyEngine()
        self.mfa_authenticator = MultiFactorAuthenticator(self.config)
        self.session_manager = SessionManager(self.config)

        # 統計情報
        self.access_stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "denied_requests": 0,
            "challenged_requests": 0,
            "high_risk_requests": 0,
            "blocked_ips": set(),
            "failed_logins_by_user": {},
            "suspicious_activities": 0,
        }

        # レート制限
        self.rate_limiter = {}
        self.blocked_ips = set()
        self.failed_login_attempts = {}

        # 定期クリーンアップタスク初期化
        self._last_cleanup = time.time()

        logger.info("ゼロトラストマネージャー初期化完了")

    async def evaluate_access_request(
        self, access_request: AccessRequest
    ) -> Dict[str, Any]:
        """アクセス要求評価"""
        logger.info(f"アクセス要求評価開始: {access_request.request_id}")

        # 統計更新
        self.access_stats["total_requests"] += 1

        # レート制限チェック
        rate_limit_ok, rate_limit_reason = self.check_rate_limit(
            access_request.device_context.ip_address,
            access_request.user_context.user_id,
        )

        if not rate_limit_ok:
            logger.warning(
                f"レート制限違反: {access_request.user_context.user_id} - {rate_limit_reason}"
            )
            return {
                "request_id": access_request.request_id,
                "decision": AccessDecision.DENY.value,
                "trust_level": TrustLevel.DENIED.value,
                "reason": rate_limit_reason,
                "evaluated_at": time.time(),
            }

        try:
            # 1. リスク評価
            risk_level, risk_score, risk_factors = self.risk_assessor.assess_risk(
                access_request
            )

            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.access_stats["high_risk_requests"] += 1

            # 2. ポリシー評価
            decision, trust_level, additional_controls = (
                self.policy_engine.evaluate_policies(
                    access_request, (risk_level, risk_score, risk_factors)
                )
            )

            # 3. 統計更新
            if decision == AccessDecision.ALLOW:
                self.access_stats["allowed_requests"] += 1
            elif decision == AccessDecision.DENY:
                self.access_stats["denied_requests"] += 1
            elif decision == AccessDecision.CHALLENGE:
                self.access_stats["challenged_requests"] += 1

            # 4. 評価結果
            evaluation_result = {
                "request_id": access_request.request_id,
                "decision": decision.value,
                "trust_level": trust_level.value,
                "risk_assessment": {
                    "risk_level": risk_level.value,
                    "risk_score": risk_score,
                    "risk_factors": risk_factors,
                },
                "additional_controls": additional_controls,
                "evaluated_at": time.time(),
                "session_token": None,
            }

            # 5. セッション管理
            if decision == AccessDecision.ALLOW:
                session_token = self.session_manager.create_session(
                    access_request.user_context, access_request.device_context
                )
                evaluation_result["session_token"] = session_token

            logger.info(
                f"アクセス評価完了: {decision.value} (リスク: {risk_level.value})"
            )

            return evaluation_result

        except Exception as e:
            logger.error(f"アクセス評価エラー: {e}")
            return {
                "request_id": access_request.request_id,
                "decision": AccessDecision.DENY.value,
                "trust_level": TrustLevel.DENIED.value,
                "error": str(e),
                "evaluated_at": time.time(),
            }

    def generate_device_fingerprint(self, device_context: DeviceContext) -> str:
        """デバイスフィンガープリント生成"""
        fingerprint_data = f"{device_context.device_type}:{device_context.os_info}:{device_context.browser_info}"

        fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
        device_context.device_fingerprint = fingerprint

        return fingerprint

    def add_trusted_device(self, device_context: DeviceContext, user_id: str) -> bool:
        """信頼済みデバイス追加"""
        try:
            device_context.trusted_device = True
            device_context.last_seen = time.time()

            # 実際の実装では永続化が必要
            logger.info(
                f"信頼済みデバイス登録: {device_context.device_id} (ユーザー: {user_id})"
            )

            return True

        except Exception as e:
            logger.error(f"信頼済みデバイス登録エラー: {e}")
            return False

    def create_access_request(
        self,
        user_id: str,
        username: str,
        roles: List[str],
        device_info: Dict[str, Any],
        resource_info: Dict[str, Any],
        action: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AccessRequest:
        """アクセス要求作成"""

        # ユーザーコンテキスト
        user_context = UserContext(
            user_id=user_id,
            username=username,
            roles=roles,
            groups=metadata.get("groups", []) if metadata else [],
            attributes=metadata.get("user_attributes", {}) if metadata else {},
            session_id=str(uuid.uuid4()),
            created_at=time.time(),
            last_activity=time.time(),
            authentication_methods=(
                metadata.get("auth_methods", ["password"]) if metadata else ["password"]
            ),
            mfa_verified=metadata.get("mfa_verified", False) if metadata else False,
        )

        # デバイスコンテキスト
        device_context = DeviceContext(
            device_id=device_info.get("device_id", str(uuid.uuid4())),
            device_type=device_info.get("device_type", "unknown"),
            os_info=device_info.get("os_info", "unknown"),
            browser_info=device_info.get("browser_info", "unknown"),
            ip_address=device_info.get("ip_address", "unknown"),
            geolocation=device_info.get("geolocation"),
            trusted_device=device_info.get("trusted_device", False),
        )

        # デバイスフィンガープリント生成
        self.generate_device_fingerprint(device_context)

        # リソースコンテキスト
        resource_context = ResourceContext(
            resource_id=resource_info.get("resource_id", "unknown"),
            resource_type=resource_info.get("resource_type", "api"),
            classification=resource_info.get("classification", "internal"),
            required_permissions=resource_info.get("required_permissions", []),
            data_sensitivity=resource_info.get("data_sensitivity", "medium"),
        )

        # アクセス要求
        access_request = AccessRequest(
            request_id=str(uuid.uuid4()),
            user_context=user_context,
            device_context=device_context,
            resource_context=resource_context,
            requested_action=action,
            timestamp=time.time(),
            request_metadata=metadata if metadata is not None else {},
        )

        return access_request

    def check_rate_limit(self, ip_address: str, user_id: str) -> Tuple[bool, str]:
        """レート制限チェック"""
        current_time = time.time()
        window_size = 3600  # 1時間
        max_requests_per_hour = 100  # 1時間あたり最大100リクエスト

        # IPベースのレート制限
        if ip_address in self.blocked_ips:
            return False, "ブロックされた IP アドレス"

        # リクエスト数チェック
        key = f"{ip_address}:{user_id}"
        if key not in self.rate_limiter:
            self.rate_limiter[key] = []

        # 古いリクエストをクリーンアップ
        self.rate_limiter[key] = [
            timestamp
            for timestamp in self.rate_limiter[key]
            if current_time - timestamp < window_size
        ]

        if len(self.rate_limiter[key]) >= max_requests_per_hour:
            return False, "レート制限超過"

        self.rate_limiter[key].append(current_time)
        return True, "OK"

    def record_failed_login(self, ip_address: str, user_id: str) -> None:
        """ログイン失敗記録"""
        current_time = time.time()
        max_failures = 5  # 最大失敗回数
        failure_window = 3600  # 1時間

        # 失敗記録初期化
        if user_id not in self.failed_login_attempts:
            self.failed_login_attempts[user_id] = []

        # 古い失敗記録をクリーンアップ
        self.failed_login_attempts[user_id] = [
            timestamp
            for timestamp in self.failed_login_attempts[user_id]
            if current_time - timestamp < failure_window
        ]

        # 新しい失敗を記録
        self.failed_login_attempts[user_id].append(current_time)

        # 統計更新
        if user_id not in self.access_stats["failed_logins_by_user"]:
            self.access_stats["failed_logins_by_user"][user_id] = 0
        self.access_stats["failed_logins_by_user"][user_id] += 1

        # 失敗回数が上限を超えた場合IPをブロック
        if len(self.failed_login_attempts[user_id]) >= max_failures:
            self.blocked_ips.add(ip_address)
            self.access_stats["blocked_ips"].add(ip_address)
            self.access_stats["suspicious_activities"] += 1
            logger.warning(
                f"IPブロック: {ip_address} (ユーザー: {user_id}, 失敗回数: {len(self.failed_login_attempts[user_id])})"
            )

    def unblock_ip(self, ip_address: str) -> bool:
        """手動IPブロック解除"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            if ip_address in self.access_stats["blocked_ips"]:
                self.access_stats["blocked_ips"].remove(ip_address)
            logger.info(f"IPブロック解除: {ip_address}")
            return True
        return False

    def get_security_dashboard(self) -> Dict[str, Any]:
        """セキュリティダッシュボード情報"""
        total_requests = self.access_stats["total_requests"]

        if total_requests == 0:
            return {
                "status": "初期状態",
                "access_stats": self.access_stats,
                "active_sessions": 0,
                "trust_distribution": {},
                "risk_distribution": {},
            }

        return {
            "status": "運用中",
            "access_stats": self.access_stats,
            "success_rate": (self.access_stats["allowed_requests"] / total_requests)
            * 100,
            "challenge_rate": (
                self.access_stats["challenged_requests"] / total_requests
            )
            * 100,
            "high_risk_rate": (self.access_stats["high_risk_requests"] / total_requests)
            * 100,
            "active_sessions": len(self.session_manager.active_sessions),
            "blocked_ips_count": len(self.access_stats["blocked_ips"]),
            "suspicious_activities": self.access_stats["suspicious_activities"],
            "top_failed_users": sorted(
                self.access_stats["failed_logins_by_user"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "config": {
                "minimum_trust_level": self.config.minimum_trust_level.value,
                "require_mfa": self.config.require_mfa,
                "session_timeout": self.config.session_timeout_minutes,
            },
        }

    async def cleanup_resources(self):
        """リソースクリーンアップ"""
        current_time = time.time()

        # セッションクリーンアップ
        self.session_manager.cleanup_expired_sessions()

        # レート制限データクリーンアップ
        window_size = 3600  # 1時間
        keys_to_remove = []

        for key in list(self.rate_limiter.keys()):
            self.rate_limiter[key] = [
                timestamp
                for timestamp in self.rate_limiter[key]
                if current_time - timestamp < window_size
            ]
            if not self.rate_limiter[key]:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.rate_limiter[key]

        # 失敗ログインデータクリーンアップ
        failure_window = 3600  # 1時間
        users_to_clean = []

        for user_id in list(self.failed_login_attempts.keys()):
            self.failed_login_attempts[user_id] = [
                timestamp
                for timestamp in self.failed_login_attempts[user_id]
                if current_time - timestamp < failure_window
            ]
            if not self.failed_login_attempts[user_id]:
                users_to_clean.append(user_id)

        for user_id in users_to_clean:
            del self.failed_login_attempts[user_id]

        self._last_cleanup = current_time
        logger.info("ゼロトラストマネージャーリソースクリーンアップ完了")


if __name__ == "__main__":
    # ゼロトラストデモ
    async def main():
        print("=== ゼロトラストセキュリティマネージャー ===")

        config = ZeroTrustConfig(
            minimum_trust_level=TrustLevel.CONDITIONAL,
            require_mfa=True,
            session_timeout_minutes=30,
        )

        zt_manager = ZeroTrustManager(config)

        print("ゼロトラストマネージャー初期化完了")

        # テストアクセス要求
        access_request = zt_manager.create_access_request(
            user_id="user_123",
            username="john.doe",
            roles=["user", "analyst"],
            device_info={
                "device_type": "desktop",
                "os_info": "Windows 11",
                "browser_info": "Chrome 120",
                "ip_address": "192.168.1.100",
                "trusted_device": False,
            },
            resource_info={
                "resource_id": "api_trading_data",
                "resource_type": "api",
                "classification": "confidential",
                "data_sensitivity": "high",
                "required_permissions": ["read_trading_data"],
            },
            action="read",
            metadata={"auth_methods": ["password"], "mfa_verified": False},
        )

        print(f"アクセス要求作成: {access_request.request_id}")

        # アクセス評価
        evaluation = await zt_manager.evaluate_access_request(access_request)

        print("\n=== アクセス評価結果 ===")
        print(f"決定: {evaluation['decision']}")
        print(f"トラストレベル: {evaluation['trust_level']}")
        print(f"リスクレベル: {evaluation['risk_assessment']['risk_level']}")
        print(f"リスクスコア: {evaluation['risk_assessment']['risk_score']:.2f}")

        if evaluation["risk_assessment"]["risk_factors"]:
            print(
                f"リスク要因: {list(evaluation['risk_assessment']['risk_factors'].keys())}"
            )

        if evaluation["additional_controls"]:
            print(f"追加制御: {evaluation['additional_controls']}")

        # 信頼済みデバイステスト
        print("\n=== 信頼済みデバイス追加 ===")
        trusted_result = zt_manager.add_trusted_device(
            access_request.device_context, access_request.user_context.user_id
        )
        print(f"信頼済みデバイス登録: {trusted_result}")

        # 再評価（信頼済みデバイスとして）
        access_request.device_context.trusted_device = True
        access_request.user_context.mfa_verified = True

        re_evaluation = await zt_manager.evaluate_access_request(access_request)
        print("\n=== 再評価結果（信頼済みデバイス）===")
        print(f"決定: {re_evaluation['decision']}")
        print(f"トラストレベル: {re_evaluation['trust_level']}")
        print(f"リスクレベル: {re_evaluation['risk_assessment']['risk_level']}")

        # ダッシュボード
        dashboard = zt_manager.get_security_dashboard()
        print("\n=== セキュリティダッシュボード ===")
        print(f"ステータス: {dashboard['status']}")
        print(f"総リクエスト: {dashboard['access_stats']['total_requests']}")
        print(f"許可率: {dashboard.get('success_rate', 0):.1f}%")
        print(f"チャレンジ率: {dashboard.get('challenge_rate', 0):.1f}%")
        print(f"高リスク率: {dashboard.get('high_risk_rate', 0):.1f}%")
        print(f"アクティブセッション: {dashboard['active_sessions']}")

        # クリーンアップ
        await zt_manager.cleanup_resources()
        print("\n=== ゼロトラストテスト完了 ===")

    # 実行
    asyncio.run(main())
