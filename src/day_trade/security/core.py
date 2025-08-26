#!/usr/bin/env python3
"""
セキュリティテストフレームワーク - Core Module
Issue #419: セキュリティ強化 - セキュリティテストフレームワークの導入

基底クラス、データ構造、Enums定義
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class TestCategory(Enum):
    """テストカテゴリ"""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_VALIDATION = "data_validation"
    ENCRYPTION = "encryption"
    NETWORK_SECURITY = "network_security"
    SESSION_MANAGEMENT = "session_management"
    INPUT_VALIDATION = "input_validation"
    ERROR_HANDLING = "error_handling"
    COMPLIANCE = "compliance"
    INFRASTRUCTURE = "infrastructure"


class TestSeverity(Enum):
    """テスト結果重要度"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestStatus(Enum):
    """テストステータス"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class SecurityTestResult:
    """セキュリティテスト結果"""

    test_id: str
    test_name: str
    category: TestCategory
    severity: TestSeverity
    status: TestStatus

    # 実行情報
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # 結果詳細
    description: str = ""
    expected: str = ""
    actual: str = ""
    error_message: Optional[str] = None

    # 修正ガイダンス
    remediation: str = ""
    references: List[str] = field(default_factory=list)

    # 証跡
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "description": self.description,
            "expected": self.expected,
            "actual": self.actual,
            "error_message": self.error_message,
            "remediation": self.remediation,
            "references": self.references,
            "evidence": self.evidence,
        }


class SecurityTest(ABC):
    """セキュリティテスト基底クラス"""

    def __init__(
        self,
        test_id: str,
        test_name: str,
        category: TestCategory,
        severity: TestSeverity = TestSeverity.MEDIUM,
    ):
        self.test_id = test_id
        self.test_name = test_name
        self.category = category
        self.severity = severity

    @abstractmethod
    async def execute(self, **kwargs) -> SecurityTestResult:
        """テスト実行"""
        pass

    def create_result(
        self,
        status: TestStatus,
        description: str = "",
        expected: str = "",
        actual: str = "",
        error_message: Optional[str] = None,
        remediation: str = "",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> SecurityTestResult:
        """テスト結果作成"""
        return SecurityTestResult(
            test_id=self.test_id,
            test_name=self.test_name,
            category=self.category,
            severity=self.severity,
            status=status,
            start_time=datetime.utcnow(),
            description=description,
            expected=expected,
            actual=actual,
            error_message=error_message,
            remediation=remediation,
            evidence=evidence or {},
        )