#!/usr/bin/env python3
"""
Zero Trust Identity Engine
ゼロトラスト身元認証エンジン
"""

import asyncio
import hashlib
import hmac
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import uuid
import logging
import json
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..functional.monads import Either, TradingResult

logger = logging.getLogger(__name__)

class IdentityType(Enum):
    """アイデンティティタイプ"""
    USER = "user"
    SERVICE = "service"
    DEVICE = "device"
    API_CLIENT = "api_client"

class AuthenticationFactor(Enum):
    """認証要素"""
    PASSWORD = "password"
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BIOMETRIC = "biometric"
    HARDWARE_TOKEN = "hardware_token"
    PUSH_NOTIFICATION = "push_notification"

class RiskLevel(Enum):
    """リスクレベル"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Identity:
    """アイデンティティ"""
    identity_id: str
    identity_type: IdentityType
    principal_name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified: Optional[datetime] = None
    trust_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    def update_trust_score(self, new_score: float) -> None:
        """信頼スコア更新"""
        self.trust_score = max(0.0, min(1.0, new_score))
        
        # リスクレベル更新
        if self.trust_score >= 0.8:
            self.risk_level = RiskLevel.LOW
        elif self.trust_score >= 0.6:
            self.risk_level = RiskLevel.MEDIUM
        elif self.trust_score >= 0.3:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.CRITICAL

@dataclass
class AuthenticationAttempt:
    """認証試行"""
    attempt_id: str
    identity_id: str
    factors_used: List[AuthenticationFactor]
    source_ip: str
    user_agent: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = False
    risk_indicators: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None
    device_fingerprint: Optional[str] = None

@dataclass
class BiometricTemplate:
    """バイオメトリックテンプレート"""
    template_id: str
    identity_id: str
    biometric_type: str  # fingerprint, iris, face, voice
    template_data: bytes  # 暗号化されたテンプレート
    quality_score: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    
    def is_expired(self, max_age_days: int = 90) -> bool:
        """期限切れ確認"""
        if not self.last_used:
            return False
        return datetime.now(timezone.utc) - self.last_used > timedelta(days=max_age_days)


class IdentityEngine:
    """アイデンティティエンジン"""
    
    def __init__(self):
        self._identities: Dict[str, Identity] = {}
        self._authentication_attempts: List[AuthenticationAttempt] = []
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._encryption_key = Fernet.generate_key()
        self._cipher = Fernet(self._encryption_key)
        
    async def create_identity(self, identity_type: IdentityType, 
                            principal_name: str,
                            attributes: Dict[str, Any] = None) -> TradingResult[Identity]:
        """アイデンティティ作成"""
        try:
            identity_id = str(uuid.uuid4())
            identity = Identity(
                identity_id=identity_id,
                identity_type=identity_type,
                principal_name=principal_name,
                attributes=attributes or {}
            )
            
            self._identities[identity_id] = identity
            logger.info(f"Created identity: {identity_id} ({principal_name})")
            
            return TradingResult.success(identity)
            
        except Exception as e:
            return TradingResult.failure('IDENTITY_CREATION_ERROR', str(e))
    
    async def authenticate_identity(self, principal_name: str,
                                  credentials: Dict[str, Any],
                                  context: Dict[str, Any]) -> TradingResult[Tuple[Identity, str]]:
        """アイデンティティ認証"""
        try:
            # アイデンティティ検索
            identity = self._find_identity_by_principal(principal_name)
            if not identity:
                await self._record_failed_attempt(principal_name, context)
                return TradingResult.failure('IDENTITY_NOT_FOUND', 'Identity not found')
            
            # 認証試行記録
            attempt = AuthenticationAttempt(
                attempt_id=str(uuid.uuid4()),
                identity_id=identity.identity_id,
                factors_used=[],
                source_ip=context.get('source_ip', ''),
                user_agent=context.get('user_agent', ''),
                location=context.get('location'),
                device_fingerprint=context.get('device_fingerprint')
            )
            
            # 多要素認証
            auth_result = await self._verify_multi_factor_authentication(
                identity, credentials, attempt
            )
            
            if auth_result.is_left():
                attempt.success = False
                self._authentication_attempts.append(attempt)
                return auth_result
            
            # 認証成功
            attempt.success = True
            self._authentication_attempts.append(attempt)
            
            # セッション作成
            session_token = await self._create_session(identity, context)
            
            # 信頼スコア更新
            await self._update_trust_score(identity, attempt)
            
            identity.last_verified = datetime.now(timezone.utc)
            
            return TradingResult.success((identity, session_token))
            
        except Exception as e:
            return TradingResult.failure('AUTHENTICATION_ERROR', str(e))
    
    async def validate_session(self, session_token: str) -> TradingResult[Identity]:
        """セッション検証"""
        try:
            if session_token not in self._active_sessions:
                return TradingResult.failure('INVALID_SESSION', 'Session not found')
            
            session = self._active_sessions[session_token]
            
            # セッション期限確認
            expires_at = datetime.fromisoformat(session['expires_at'])
            if datetime.now(timezone.utc) > expires_at:
                del self._active_sessions[session_token]
                return TradingResult.failure('SESSION_EXPIRED', 'Session has expired')
            
            identity_id = session['identity_id']
            identity = self._identities.get(identity_id)
            
            if not identity:
                del self._active_sessions[session_token]
                return TradingResult.failure('IDENTITY_NOT_FOUND', 'Identity not found')
            
            # セッション使用時刻更新
            session['last_used'] = datetime.now(timezone.utc).isoformat()
            
            return TradingResult.success(identity)
            
        except Exception as e:
            return TradingResult.failure('SESSION_VALIDATION_ERROR', str(e))
    
    async def revoke_session(self, session_token: str) -> TradingResult[None]:
        """セッション無効化"""
        try:
            if session_token in self._active_sessions:
                del self._active_sessions[session_token]
                logger.info(f"Revoked session: {session_token}")
            
            return TradingResult.success(None)
            
        except Exception as e:
            return TradingResult.failure('SESSION_REVOCATION_ERROR', str(e))
    
    def _find_identity_by_principal(self, principal_name: str) -> Optional[Identity]:
        """プリンシパル名でアイデンティティ検索"""
        for identity in self._identities.values():
            if identity.principal_name == principal_name:
                return identity
        return None
    
    async def _verify_multi_factor_authentication(self, identity: Identity,
                                                credentials: Dict[str, Any],
                                                attempt: AuthenticationAttempt) -> TradingResult[None]:
        """多要素認証検証"""
        required_factors = self._get_required_factors(identity)
        verified_factors = []
        
        # パスワード認証
        if AuthenticationFactor.PASSWORD in required_factors:
            password_result = await self._verify_password(identity, credentials.get('password'))
            if password_result.is_right():
                verified_factors.append(AuthenticationFactor.PASSWORD)
            elif AuthenticationFactor.PASSWORD in required_factors:
                return TradingResult.failure('PASSWORD_VERIFICATION_FAILED', 'Password verification failed')
        
        # TOTP認証
        if AuthenticationFactor.TOTP in required_factors:
            totp_result = await self._verify_totp(identity, credentials.get('totp_code'))
            if totp_result.is_right():
                verified_factors.append(AuthenticationFactor.TOTP)
            else:
                return TradingResult.failure('TOTP_VERIFICATION_FAILED', 'TOTP verification failed')
        
        # バイオメトリック認証
        if AuthenticationFactor.BIOMETRIC in required_factors:
            biometric_result = await self._verify_biometric(identity, credentials.get('biometric_data'))
            if biometric_result.is_right():
                verified_factors.append(AuthenticationFactor.BIOMETRIC)
            else:
                return TradingResult.failure('BIOMETRIC_VERIFICATION_FAILED', 'Biometric verification failed')
        
        attempt.factors_used = verified_factors
        
        # 必要な要素がすべて検証されたか確認
        if len(verified_factors) < len(required_factors):
            return TradingResult.failure('INSUFFICIENT_FACTORS', 'Insufficient authentication factors')
        
        return TradingResult.success(None)
    
    def _get_required_factors(self, identity: Identity) -> List[AuthenticationFactor]:
        """必要な認証要素取得"""
        # リスクレベルに基づいて必要な認証要素を決定
        base_factors = [AuthenticationFactor.PASSWORD]
        
        if identity.risk_level == RiskLevel.CRITICAL:
            return base_factors + [AuthenticationFactor.TOTP, AuthenticationFactor.BIOMETRIC]
        elif identity.risk_level == RiskLevel.HIGH:
            return base_factors + [AuthenticationFactor.TOTP]
        elif identity.risk_level == RiskLevel.MEDIUM:
            return base_factors + [AuthenticationFactor.TOTP]
        else:
            return base_factors
    
    async def _verify_password(self, identity: Identity, password: str) -> TradingResult[None]:
        """パスワード検証"""
        if not password:
            return TradingResult.failure('MISSING_PASSWORD', 'Password is required')
        
        # 実際の実装では、ハッシュ化されたパスワードと比較
        stored_hash = identity.attributes.get('password_hash', '')
        
        if not stored_hash:
            return TradingResult.failure('NO_PASSWORD_SET', 'No password set for identity')
        
        # パスワード検証（簡略化）
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash != stored_hash:
            return TradingResult.failure('INVALID_PASSWORD', 'Invalid password')
        
        return TradingResult.success(None)
    
    async def _verify_totp(self, identity: Identity, totp_code: str) -> TradingResult[None]:
        """TOTP検証"""
        if not totp_code:
            return TradingResult.failure('MISSING_TOTP', 'TOTP code is required')
        
        # TOTPシークレット取得
        totp_secret = identity.attributes.get('totp_secret')
        if not totp_secret:
            return TradingResult.failure('NO_TOTP_SECRET', 'TOTP not configured')
        
        # TOTP検証（簡略化実装）
        current_time = int(time.time())
        time_window = current_time // 30  # 30秒ウィンドウ
        
        expected_code = self._generate_totp(totp_secret, time_window)
        
        if totp_code != expected_code:
            # 前後のウィンドウも確認（時計のずれ対応）
            prev_code = self._generate_totp(totp_secret, time_window - 1)
            next_code = self._generate_totp(totp_secret, time_window + 1)
            
            if totp_code not in [prev_code, next_code]:
                return TradingResult.failure('INVALID_TOTP', 'Invalid TOTP code')
        
        return TradingResult.success(None)
    
    async def _verify_biometric(self, identity: Identity, biometric_data: bytes) -> TradingResult[None]:
        """バイオメトリック検証"""
        if not biometric_data:
            return TradingResult.failure('MISSING_BIOMETRIC', 'Biometric data is required')
        
        # バイオメトリックテンプレート取得
        templates = identity.attributes.get('biometric_templates', [])
        if not templates:
            return TradingResult.failure('NO_BIOMETRIC_ENROLLED', 'No biometric templates enrolled')
        
        # バイオメトリック照合（簡略化）
        # 実際の実装では、バイオメトリック照合エンジンを使用
        match_score = self._calculate_biometric_similarity(biometric_data, templates[0])
        
        if match_score < 0.8:  # 80%以上の一致率が必要
            return TradingResult.failure('BIOMETRIC_MISMATCH', 'Biometric verification failed')
        
        return TradingResult.success(None)
    
    def _generate_totp(self, secret: str, time_window: int) -> str:
        """TOTP生成"""
        # HMAC-SHA1ベースのTOTP生成
        key = base64.b32decode(secret.encode())
        message = time_window.to_bytes(8, byteorder='big')
        
        hmac_digest = hmac.new(key, message, hashlib.sha1).digest()
        
        # Dynamic truncation
        offset = hmac_digest[19] & 0x0f
        truncated = hmac_digest[offset:offset + 4]
        code = int.from_bytes(truncated, byteorder='big') & 0x7fffffff
        
        return f"{code % 1000000:06d}"
    
    def _calculate_biometric_similarity(self, sample: bytes, template: bytes) -> float:
        """バイオメトリック類似度計算（簡略化）"""
        # 実際の実装では、専用のバイオメトリック照合アルゴリズムを使用
        sample_hash = hashlib.sha256(sample).digest()
        template_hash = hashlib.sha256(template).digest()
        
        # ハミング距離による簡易照合
        distance = sum(a != b for a, b in zip(sample_hash, template_hash))
        similarity = 1.0 - (distance / len(sample_hash))
        
        return similarity
    
    async def _create_session(self, identity: Identity, context: Dict[str, Any]) -> str:
        """セッション作成"""
        session_token = str(uuid.uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(hours=8)
        
        session = {
            'session_token': session_token,
            'identity_id': identity.identity_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'expires_at': expires_at.isoformat(),
            'last_used': datetime.now(timezone.utc).isoformat(),
            'source_ip': context.get('source_ip'),
            'user_agent': context.get('user_agent'),
            'device_fingerprint': context.get('device_fingerprint')
        }
        
        self._active_sessions[session_token] = session
        
        return session_token
    
    async def _update_trust_score(self, identity: Identity, attempt: AuthenticationAttempt) -> None:
        """信頼スコア更新"""
        current_score = identity.trust_score
        
        # 成功認証による信頼スコア向上
        if attempt.success:
            score_increase = 0.1
            
            # 多要素認証使用による追加向上
            if len(attempt.factors_used) > 1:
                score_increase += 0.05
            
            # バイオメトリック使用による追加向上
            if AuthenticationFactor.BIOMETRIC in attempt.factors_used:
                score_increase += 0.05
            
            new_score = min(1.0, current_score + score_increase)
        else:
            # 失敗による信頼スコア低下
            new_score = max(0.0, current_score - 0.2)
        
        identity.update_trust_score(new_score)
        logger.debug(f"Updated trust score for {identity.identity_id}: {current_score} -> {new_score}")
    
    async def _record_failed_attempt(self, principal_name: str, context: Dict[str, Any]) -> None:
        """失敗試行記録"""
        attempt = AuthenticationAttempt(
            attempt_id=str(uuid.uuid4()),
            identity_id="unknown",
            factors_used=[],
            source_ip=context.get('source_ip', ''),
            user_agent=context.get('user_agent', ''),
            success=False,
            risk_indicators={'reason': 'identity_not_found'}
        )
        
        self._authentication_attempts.append(attempt)
        logger.warning(f"Failed authentication attempt for unknown principal: {principal_name}")


class MultiFactorAuthentication:
    """多要素認証管理"""
    
    def __init__(self, identity_engine: IdentityEngine):
        self.identity_engine = identity_engine
        self._factor_providers: Dict[AuthenticationFactor, Any] = {}
    
    def register_factor_provider(self, factor: AuthenticationFactor, provider: Any) -> None:
        """認証要素プロバイダ登録"""
        self._factor_providers[factor] = provider
        logger.info(f"Registered factor provider: {factor.value}")
    
    async def setup_totp(self, identity_id: str) -> TradingResult[str]:
        """TOTP設定"""
        try:
            # TOTP秘密鍵生成
            secret = base64.b32encode(uuid.uuid4().bytes).decode()[:16]
            
            identity = self.identity_engine._identities.get(identity_id)
            if not identity:
                return TradingResult.failure('IDENTITY_NOT_FOUND', 'Identity not found')
            
            identity.attributes['totp_secret'] = secret
            
            # QRコード用URL生成
            issuer = "DayTrade"
            account_name = identity.principal_name
            qr_url = f"otpauth://totp/{issuer}:{account_name}?secret={secret}&issuer={issuer}"
            
            return TradingResult.success(qr_url)
            
        except Exception as e:
            return TradingResult.failure('TOTP_SETUP_ERROR', str(e))
    
    async def enroll_biometric(self, identity_id: str, biometric_type: str,
                             template_data: bytes) -> TradingResult[str]:
        """バイオメトリック登録"""
        try:
            identity = self.identity_engine._identities.get(identity_id)
            if not identity:
                return TradingResult.failure('IDENTITY_NOT_FOUND', 'Identity not found')
            
            template_id = str(uuid.uuid4())
            
            # テンプレート暗号化
            encrypted_template = self.identity_engine._cipher.encrypt(template_data)
            
            template = BiometricTemplate(
                template_id=template_id,
                identity_id=identity_id,
                biometric_type=biometric_type,
                template_data=encrypted_template,
                quality_score=0.95  # 品質スコア（実装では実際の品質評価）
            )
            
            # アイデンティティにテンプレート追加
            if 'biometric_templates' not in identity.attributes:
                identity.attributes['biometric_templates'] = []
            
            identity.attributes['biometric_templates'].append(template.template_data)
            
            return TradingResult.success(template_id)
            
        except Exception as e:
            return TradingResult.failure('BIOMETRIC_ENROLLMENT_ERROR', str(e))


class BehavioralAnalytics:
    """行動分析"""
    
    def __init__(self):
        self._behavioral_profiles: Dict[str, Dict[str, Any]] = {}
        self._anomaly_threshold = 0.7
    
    async def analyze_behavior(self, identity_id: str, 
                             current_behavior: Dict[str, Any]) -> TradingResult[float]:
        """行動分析"""
        try:
            profile = self._behavioral_profiles.get(identity_id)
            
            if not profile:
                # 初回プロファイル作成
                self._behavioral_profiles[identity_id] = {
                    'login_times': [],
                    'locations': [],
                    'devices': [],
                    'trading_patterns': []
                }
                return TradingResult.success(0.5)  # 中立スコア
            
            anomaly_score = await self._calculate_anomaly_score(profile, current_behavior)
            
            # プロファイル更新
            await self._update_behavioral_profile(identity_id, current_behavior)
            
            return TradingResult.success(anomaly_score)
            
        except Exception as e:
            return TradingResult.failure('BEHAVIORAL_ANALYSIS_ERROR', str(e))
    
    async def _calculate_anomaly_score(self, profile: Dict[str, Any],
                                     current_behavior: Dict[str, Any]) -> float:
        """異常スコア計算"""
        anomalies = []
        
        # 時間帯の異常検知
        current_hour = datetime.now().hour
        typical_hours = profile.get('login_times', [])
        
        if typical_hours:
            hour_deviation = min(abs(current_hour - h) for h in typical_hours)
            if hour_deviation > 6:  # 6時間以上の差
                anomalies.append(0.3)
        
        # 位置の異常検知
        current_location = current_behavior.get('location')
        if current_location:
            typical_locations = profile.get('locations', [])
            if typical_locations and current_location not in typical_locations:
                anomalies.append(0.5)
        
        # デバイスの異常検知
        current_device = current_behavior.get('device_fingerprint')
        if current_device:
            typical_devices = profile.get('devices', [])
            if typical_devices and current_device not in typical_devices:
                anomalies.append(0.4)
        
        # 総合異常スコア
        if not anomalies:
            return 0.0
        
        return min(1.0, sum(anomalies) / len(anomalies))
    
    async def _update_behavioral_profile(self, identity_id: str,
                                       current_behavior: Dict[str, Any]) -> None:
        """行動プロファイル更新"""
        profile = self._behavioral_profiles[identity_id]
        
        # ログイン時間記録
        current_hour = datetime.now().hour
        login_times = profile['login_times']
        if current_hour not in login_times:
            login_times.append(current_hour)
            # 最新20件のみ保持
            profile['login_times'] = login_times[-20:]
        
        # 位置記録
        location = current_behavior.get('location')
        if location:
            locations = profile['locations']
            if location not in locations:
                locations.append(location)
                # 最新10件のみ保持
                profile['locations'] = locations[-10:]
        
        # デバイス記録
        device = current_behavior.get('device_fingerprint')
        if device:
            devices = profile['devices']
            if device not in devices:
                devices.append(device)
                # 最新5件のみ保持
                profile['devices'] = devices[-5:]


class BiometricAuthentication:
    """バイオメトリック認証"""
    
    def __init__(self):
        self._templates: Dict[str, List[BiometricTemplate]] = {}
        self._match_threshold = 0.8
    
    async def verify_biometric(self, identity_id: str, biometric_type: str,
                             sample_data: bytes) -> TradingResult[float]:
        """バイオメトリック検証"""
        try:
            templates = self._templates.get(identity_id, [])
            matching_templates = [t for t in templates if t.biometric_type == biometric_type]
            
            if not matching_templates:
                return TradingResult.failure('NO_TEMPLATES', 'No biometric templates found')
            
            best_match_score = 0.0
            
            for template in matching_templates:
                if template.is_expired():
                    continue
                
                match_score = await self._match_biometric(sample_data, template.template_data)
                best_match_score = max(best_match_score, match_score)
                
                # 使用時刻更新
                template.last_used = datetime.now(timezone.utc)
            
            if best_match_score >= self._match_threshold:
                return TradingResult.success(best_match_score)
            else:
                return TradingResult.failure('BIOMETRIC_MISMATCH', f'Match score {best_match_score} below threshold')
                
        except Exception as e:
            return TradingResult.failure('BIOMETRIC_VERIFICATION_ERROR', str(e))
    
    async def _match_biometric(self, sample: bytes, template: bytes) -> float:
        """バイオメトリック照合"""
        # 実際の実装では、専用のバイオメトリック照合エンジンを使用
        # ここでは簡略化したマッチング
        
        sample_features = self._extract_features(sample)
        template_features = self._extract_features(template)
        
        # コサイン類似度計算
        dot_product = np.dot(sample_features, template_features)
        norms = np.linalg.norm(sample_features) * np.linalg.norm(template_features)
        
        if norms == 0:
            return 0.0
        
        similarity = dot_product / norms
        return max(0.0, similarity)
    
    def _extract_features(self, data: bytes) -> np.ndarray:
        """特徴抽出（簡略化）"""
        # 実際の実装では、バイオメトリック特有の特徴抽出アルゴリズムを使用
        hash_digest = hashlib.sha256(data).digest()
        features = np.frombuffer(hash_digest, dtype=np.uint8).astype(np.float32)
        return features / 255.0  # 正規化