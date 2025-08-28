#!/usr/bin/env python3
"""
セキュリティモジュール統合パッケージ
Issue #918 項目9対応: セキュリティ強化

分割されたセキュリティモジュールの統合と後方互換性の提供
"""

# 型定義とEnum
from .types import (
    SecurityLevel, ThreatLevel, ActionType,
    SecurityEvent, ValidationResult, AuthenticationResult, RateLimitInfo
)

# インターフェース
from .interfaces import (
    IInputValidationService, IAuthenticationService, IAuthorizationService,
    IRateLimitService, ISecurityAuditService
)

# 実装クラス
from .validation import InputValidationService
from .authentication import AuthenticationService
from .authorization import AuthorizationService
from .rate_limiting import RateLimitService
from .audit import SecurityAuditService

# パターンマッチング
from .patterns import SecurityPatterns, security_patterns

# ユーティリティ関数
from .utils import (
    register_security_services, get_security_services, create_security_event
)

# 後方互換性のためのエクスポート
__all__ = [
    # 型定義とEnum
    'SecurityLevel', 'ThreatLevel', 'ActionType',
    'SecurityEvent', 'ValidationResult', 'AuthenticationResult', 'RateLimitInfo',
    
    # インターフェース
    'IInputValidationService', 'IAuthenticationService', 'IAuthorizationService',
    'IRateLimitService', 'ISecurityAuditService',
    
    # 実装クラス
    'InputValidationService', 'AuthenticationService', 'AuthorizationService',
    'RateLimitService', 'SecurityAuditService',
    
    # パターンマッチング
    'SecurityPatterns', 'security_patterns',
    
    # ユーティリティ関数
    'register_security_services', 'get_security_services', 'create_security_event'
]