#!/usr/bin/env python3
"""
セキュリティパターン定義
Issue #918 項目9対応: セキュリティ強化

各種攻撃パターンの定義とパターンマッチング機能
"""

import re
from typing import List


class SecurityPatterns:
    """セキュリティ攻撃パターン定義"""
    
    def __init__(self):
        # 危険なパターン定義
        self._sql_injection_patterns = [
            r'(\bunion\b|\bselect\b|\binsert\b|\bdelete\b|\bupdate\b|\bdrop\b)',
            r'(--|\*\/|\*)',
            r'(\bor\b|\band\b)\s+\d+\s*=\s*\d+',
            r'(\bor\b|\band\b)\s+[\'"].*[\'"]',
        ]

        self._xss_patterns = [
            r'<\s*script[^>]*>.*?</\s*script\s*>',
            r'<\s*iframe[^>]*>.*?</\s*iframe\s*>',
            r'javascript:',
            r'on\w+\s*=',
        ]

        self._path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'\.\./\.\.',
            r'\.\.\\\.\.',
        ]
    
    @property
    def sql_injection_patterns(self) -> List[str]:
        """SQLインジェクションパターン取得"""
        return self._sql_injection_patterns.copy()
    
    @property
    def xss_patterns(self) -> List[str]:
        """XSS攻撃パターン取得"""
        return self._xss_patterns.copy()
    
    @property
    def path_traversal_patterns(self) -> List[str]:
        """パストラバーサル攻撃パターン取得"""
        return self._path_traversal_patterns.copy()
    
    def check_sql_injection(self, value: str) -> bool:
        """SQLインジェクションパターンチェック"""
        for pattern in self._sql_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    def check_xss(self, value: str) -> bool:
        """XSS攻撃パターンチェック"""
        for pattern in self._xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    def check_path_traversal(self, value: str) -> bool:
        """パストラバーサル攻撃パターンチェック"""
        for pattern in self._path_traversal_patterns:
            if re.search(pattern, value):
                return True
        return False
    
    def get_matched_pattern(self, value: str, pattern_type: str) -> str:
        """マッチしたパターンを取得"""
        patterns = {
            'sql': self._sql_injection_patterns,
            'xss': self._xss_patterns,
            'path': self._path_traversal_patterns
        }
        
        target_patterns = patterns.get(pattern_type, [])
        flags = re.IGNORECASE if pattern_type in ['sql', 'xss'] else 0
        
        for pattern in target_patterns:
            if re.search(pattern, value, flags):
                return pattern
        return ""


# グローバルインスタンス
security_patterns = SecurityPatterns()