#!/usr/bin/env python3
"""
入力検証サービス実装
Issue #918 項目9対応: セキュリティ強化

入力データの検証とサニタイゼーション機能
"""

import re
import ipaddress
from pathlib import Path
from typing import Set, Union

from ..dependency_injection import ILoggingService, injectable, singleton
from .interfaces import IInputValidationService
from .types import ValidationResult, ThreatLevel
from .patterns import security_patterns


@singleton(IInputValidationService)
@injectable
class InputValidationService(IInputValidationService):
    """入力検証サービス実装"""

    def __init__(self, logging_service: ILoggingService):
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "InputValidationService")
        self.patterns = security_patterns

    def validate_string(self, value: str, max_length: int = 1000,
                       allow_special_chars: bool = False) -> ValidationResult:
        """文字列検証"""
        try:
            if not isinstance(value, str):
                return ValidationResult(
                    is_valid=False,
                    error_message="Value must be a string",
                    threat_level=ThreatLevel.LOW
                )

            # 長さチェック
            if len(value) > max_length:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"String length exceeds maximum of {max_length}",
                    threat_level=ThreatLevel.MEDIUM
                )

            # XSS攻撃パターンチェック
            if self.patterns.check_xss(value):
                matched_pattern = self.patterns.get_matched_pattern(value, 'xss')
                self.logger.warning(f"XSS pattern detected: {matched_pattern}")
                return ValidationResult(
                    is_valid=False,
                    error_message="Potentially malicious script detected",
                    threat_level=ThreatLevel.HIGH
                )

            # 特殊文字チェック
            if not allow_special_chars:
                if re.search(r'[<>&"\']', value):
                    sanitized = re.sub(r'[<>&"\']', '', value)
                    return ValidationResult(
                        is_valid=True,
                        sanitized_value=sanitized,
                        error_message="Special characters removed",
                        threat_level=ThreatLevel.LOW
                    )

            return ValidationResult(
                is_valid=True,
                sanitized_value=value,
                threat_level=ThreatLevel.INFO
            )

        except Exception as e:
            self.logger.error(f"String validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def validate_number(self, value: Union[str, int, float],
                       min_value: float = None, max_value: float = None) -> ValidationResult:
        """数値検証"""
        try:
            # 数値への変換試行
            if isinstance(value, str):
                # 危険な文字列パターンチェック
                if re.search(r'[^\d\.\-\+e]', value.lower()):
                    return ValidationResult(
                        is_valid=False,
                        error_message="Invalid characters in numeric string",
                        threat_level=ThreatLevel.MEDIUM
                    )
                numeric_value = float(value)
            elif isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message="Value must be numeric",
                    threat_level=ThreatLevel.LOW
                )

            # 範囲チェック
            if min_value is not None and numeric_value < min_value:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Value {numeric_value} is below minimum {min_value}",
                    threat_level=ThreatLevel.LOW
                )

            if max_value is not None and numeric_value > max_value:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Value {numeric_value} exceeds maximum {max_value}",
                    threat_level=ThreatLevel.LOW
                )

            return ValidationResult(
                is_valid=True,
                sanitized_value=numeric_value,
                threat_level=ThreatLevel.INFO
            )

        except ValueError as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid numeric value: {e}",
                threat_level=ThreatLevel.MEDIUM
            )
        except Exception as e:
            self.logger.error(f"Number validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def validate_email(self, email: str) -> ValidationResult:
        """メールアドレス検証"""
        try:
            # 基本的なフォーマット検証
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

            if not re.match(pattern, email):
                return ValidationResult(
                    is_valid=False,
                    error_message="Invalid email format",
                    threat_level=ThreatLevel.LOW
                )

            # 長さ制限
            if len(email) > 254:  # RFC 5321 制限
                return ValidationResult(
                    is_valid=False,
                    error_message="Email address too long",
                    threat_level=ThreatLevel.LOW
                )

            # 危険パターンチェック
            if self.patterns.check_xss(email):
                return ValidationResult(
                    is_valid=False,
                    error_message="Potentially malicious email format",
                    threat_level=ThreatLevel.HIGH
                )

            return ValidationResult(
                is_valid=True,
                sanitized_value=email.lower(),
                threat_level=ThreatLevel.INFO
            )

        except Exception as e:
            self.logger.error(f"Email validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def validate_ip_address(self, ip: str) -> ValidationResult:
        """IPアドレス検証"""
        try:
            # IPv4/IPv6アドレス検証
            ip_obj = ipaddress.ip_address(ip)

            return ValidationResult(
                is_valid=True,
                sanitized_value=str(ip_obj),
                threat_level=ThreatLevel.INFO
            )

        except ValueError:
            return ValidationResult(
                is_valid=False,
                error_message="Invalid IP address format",
                threat_level=ThreatLevel.MEDIUM
            )
        except Exception as e:
            self.logger.error(f"IP validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def sanitize_sql_input(self, value: str) -> ValidationResult:
        """SQL入力サニタイゼーション"""
        try:
            # SQLインジェクション攻撃パターンチェック
            threat_level = ThreatLevel.INFO
            sanitized = value
            pattern_found = False

            if self.patterns.check_sql_injection(value):
                matched_pattern = self.patterns.get_matched_pattern(value, 'sql')
                self.logger.warning(f"SQL injection pattern detected: {matched_pattern}")
                threat_level = ThreatLevel.HIGH
                pattern_found = True
                # 危険なパターンを削除
                for pattern in self.patterns.sql_injection_patterns:
                    sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

            # パストラバーサルパターンもチェック
            if self.patterns.check_path_traversal(value):
                threat_level = ThreatLevel.HIGH
                pattern_found = True
                for pattern in self.patterns.path_traversal_patterns:
                    sanitized = re.sub(pattern, '', sanitized)

            # 基本的なエスケープ処理
            sanitized = sanitized.replace("'", "''")  # シングルクオートのエスケープ
            sanitized = sanitized.replace('"', '""')  # ダブルクオートのエスケープ

            return ValidationResult(
                is_valid=True,
                sanitized_value=sanitized,
                threat_level=threat_level,
                error_message="SQL input sanitized" if pattern_found else None
            )

        except Exception as e:
            self.logger.error(f"SQL sanitization error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Sanitization error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )

    def validate_file_path(self, path: str, allowed_extensions: Set[str] = None) -> ValidationResult:
        """ファイルパス検証"""
        try:
            # パストラバーサル攻撃チェック
            if self.patterns.check_path_traversal(path):
                matched_pattern = self.patterns.get_matched_pattern(path, 'path')
                self.logger.warning(f"Path traversal pattern detected: {matched_pattern}")
                return ValidationResult(
                    is_valid=False,
                    error_message="Path traversal attack detected",
                    threat_level=ThreatLevel.HIGH
                )

            # パスの正規化
            normalized_path = Path(path).resolve()

            # 拡張子チェック
            if allowed_extensions is not None:
                file_extension = normalized_path.suffix.lower()
                if file_extension not in allowed_extensions:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"File extension {file_extension} not allowed",
                        threat_level=ThreatLevel.MEDIUM
                    )

            # 危険な文字チェック
            dangerous_chars = ['<', '>', '|', '&', ';']
            if any(char in str(normalized_path) for char in dangerous_chars):
                return ValidationResult(
                    is_valid=False,
                    error_message="Dangerous characters in file path",
                    threat_level=ThreatLevel.HIGH
                )

            return ValidationResult(
                is_valid=True,
                sanitized_value=str(normalized_path),
                threat_level=ThreatLevel.INFO
            )

        except Exception as e:
            self.logger.error(f"File path validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {e}",
                threat_level=ThreatLevel.MEDIUM
            )