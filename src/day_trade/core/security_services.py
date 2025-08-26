#!/usr/bin/env python3
"""
セキュリティサービス - 後方互換性レイヤー
Issue #918 項目9対応: セキュリティ強化

このファイルは後方互換性のために保持されています。
実装は security/ パッケージに移動されました。
"""

# 警告：このファイルは deprecated です
import warnings
warnings.warn(
    "security_services.py は非推奨です。代わりに day_trade.core.security パッケージを使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# 後方互換性のための re-export
from .security import (
    # 型定義とEnum
    SecurityLevel, ThreatLevel, ActionType,
    SecurityEvent, ValidationResult, AuthenticationResult, RateLimitInfo,
    
    # インターフェース
    IInputValidationService, IAuthenticationService, IAuthorizationService,
    IRateLimitService, ISecurityAuditService,
    
    # 実装クラス
    InputValidationService, AuthenticationService, AuthorizationService,
    RateLimitService, SecurityAuditService,
    
    # ユーティリティ関数
    register_security_services, get_security_services, create_security_event
)

# すべてのエクスポートを維持
__all__ = [
    'SecurityLevel', 'ThreatLevel', 'ActionType',
    'SecurityEvent', 'ValidationResult', 'AuthenticationResult', 'RateLimitInfo',
    'IInputValidationService', 'IAuthenticationService', 'IAuthorizationService',
    'IRateLimitService', 'ISecurityAuditService',
    'InputValidationService', 'AuthenticationService', 'AuthorizationService',
    'RateLimitService', 'SecurityAuditService',
    'register_security_services', 'get_security_services', 'create_security_event'
]