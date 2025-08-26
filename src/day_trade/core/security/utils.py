#!/usr/bin/env python3
"""
セキュリティユーティリティ関数
Issue #918 項目9対応: セキュリティ強化

セキュリティ関連のヘルパー関数とユーティリティ
"""

from datetime import datetime

from ..dependency_injection import get_container
from .interfaces import (
    IInputValidationService, IAuthenticationService, IAuthorizationService,
    IRateLimitService, ISecurityAuditService
)
from .types import SecurityEvent, ThreatLevel


def register_security_services():
    """セキュリティサービスを登録"""
    from .validation import InputValidationService
    from .authentication import AuthenticationService
    from .authorization import AuthorizationService
    from .rate_limiting import RateLimitService
    from .audit import SecurityAuditService

    container = get_container()

    # 入力検証サービス
    if not container.is_registered(IInputValidationService):
        container.register_singleton(IInputValidationService, InputValidationService)

    # 認証サービス
    if not container.is_registered(IAuthenticationService):
        container.register_singleton(IAuthenticationService, AuthenticationService)

    # 認可サービス
    if not container.is_registered(IAuthorizationService):
        container.register_singleton(IAuthorizationService, AuthorizationService)

    # レート制限サービス
    if not container.is_registered(IRateLimitService):
        container.register_singleton(IRateLimitService, RateLimitService)

    # セキュリティ監査サービス
    if not container.is_registered(ISecurityAuditService):
        container.register_singleton(ISecurityAuditService, SecurityAuditService)


def get_security_services():
    """セキュリティサービス取得"""
    container = get_container()
    return {
        'validation': container.resolve(IInputValidationService),
        'authentication': container.resolve(IAuthenticationService),
        'authorization': container.resolve(IAuthorizationService),
        'rate_limit': container.resolve(IRateLimitService),
        'audit': container.resolve(ISecurityAuditService)
    }


def create_security_event(event_type: str, threat_level: ThreatLevel, **kwargs) -> SecurityEvent:
    """セキュリティイベント作成ヘルパー"""
    return SecurityEvent(
        event_id="",  # サービスで自動生成
        timestamp=datetime.now(),
        event_type=event_type,
        threat_level=threat_level,
        source_ip=kwargs.get('source_ip'),
        user_id=kwargs.get('user_id'),
        action=kwargs.get('action'),
        resource=kwargs.get('resource'),
        details=kwargs.get('details', {}),
        blocked=kwargs.get('blocked', False)
    )