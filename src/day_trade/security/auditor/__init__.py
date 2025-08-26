#!/usr/bin/env python3
"""
セキュリティ監査システム - 後方互換性インターフェース

このモジュールは分割されたsecurity_auditorモジュールの統合インターフェースを提供し、
既存のコードとの後方互換性を確保します。
"""

# 列挙型とデータクラス
from .enums import (
    AuditConfig,
    AuditFinding,
    AuditResult,
    AuditScope,
    ComplianceFramework,
    SecurityReport,
)

# 各分析器
from .code_analyzer import CodeSecurityAnalyzer
from .compliance_assessor import ComplianceAssessor
from .infrastructure_analyzer import InfrastructureSecurityAnalyzer

# メイン監査システム
from .main_auditor import SecurityAuditor

# CLIインターフェース
from .cli import main

__all__ = [
    # 列挙型
    "ComplianceFramework",
    "AuditScope",
    "AuditResult",
    # データクラス
    "AuditConfig",
    "AuditFinding",
    "SecurityReport",
    # 分析器クラス
    "CodeSecurityAnalyzer",
    "InfrastructureSecurityAnalyzer",
    "ComplianceAssessor",
    # メインクラス
    "SecurityAuditor",
    # CLI機能
    "main",
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Day Trade Security Team"
__description__ = "統合セキュリティ監査システム"