#!/usr/bin/env python3
"""
アクセス制御システム - パッケージ初期化

このモジュールは、分割されたアクセス制御システムの
すべてのクラスと関数を再エクスポートし、
既存のコードとの後方互換性を保ちます。
"""

# 列挙型とデータクラス
from .enums import (
    AccessLogEntry,
    AuthenticationMethod,
    Permission,
    Session,
    SessionStatus,
    User,
    UserRole,
)

# バリデーター
from .validators import PasswordValidator

# ロール権限管理
from .role_manager import RolePermissionManager

# MFA管理
from .mfa_manager import MFAManager

# メインのアクセス制御管理システム
from .access_control_manager import AccessControlManager

# 認証とレポート機能
from .authentication import AuthenticationMixin
from .reporting import ReportingMixin

# ミックスイン統合（自動的にAccessControlManagerに機能追加）
from .mixins import integrate_mixins

# ファクトリ関数
from .factory import create_access_control_manager, run_access_control_test

# すべてのエクスポート
__all__ = [
    # 列挙型
    "UserRole",
    "Permission", 
    "AuthenticationMethod",
    "SessionStatus",
    
    # データクラス
    "User",
    "Session", 
    "AccessLogEntry",
    
    # 管理クラス
    "PasswordValidator",
    "RolePermissionManager",
    "MFAManager",
    "AccessControlManager",
    "AuthenticationMixin",
    "ReportingMixin",
    
    # ファクトリ関数
    "create_access_control_manager",
    "run_access_control_test",
]

# 旧インターフェースとの完全互換性を確保
# 元のaccess_control.pyから直接インポートしていたコードが動作するよう
# すべてのクラスと関数をトップレベルで利用可能にします

# パッケージ情報
__version__ = "1.0.0"
__author__ = "Day Trade Security Team"
__description__ = "Comprehensive access control and authentication system"