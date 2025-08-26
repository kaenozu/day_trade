#!/usr/bin/env python3
"""
アクセス制御システム - パスワード検証

このモジュールは、パスワード強度の検証機能を提供します。
セキュアなパスワードポリシーに基づいた検証を行います。
"""

from typing import List, Set, Tuple


class PasswordValidator:
    """
    パスワード強度バリデーター
    
    セキュアなパスワードポリシーに基づいてパスワードの強度を検証し、
    セキュリティリスクを軽減します。
    """

    def __init__(self):
        """
        パスワード検証ルールの初期化
        """
        self.min_length = 12
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_numbers = True
        self.require_symbols = True
        self.common_passwords = self._load_common_passwords()

    def _load_common_passwords(self) -> Set[str]:
        """
        一般的な脆弱パスワードリストの読み込み
        
        Returns:
            Set[str]: 脆弱パスワードのセット
        """
        return {
            "password",
            "123456",
            "password123",
            "admin",
            "qwerty",
            "letmein",
            "welcome",
            "monkey",
            "dragon",
            "master",
            "password1",
            "123456789",
            "12345678",
            "12345",
            "1234567890",
            "trading",
            "finance",
            "money",
            "profit",
            "investment",
            "daytrading",
            "stock",
            "market",
            "trader",
            "analyst",
        }

    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """
        パスワード強度チェック
        
        Args:
            password: 検証対象のパスワード
            
        Returns:
            Tuple[bool, List[str]]: (検証結果, エラーメッセージリスト)
        """
        errors = []

        if len(password) < self.min_length:
            errors.append(f"パスワードは{self.min_length}文字以上である必要があります")

        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("大文字を含む必要があります")

        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("小文字を含む必要があります")

        if self.require_numbers and not any(c.isdigit() for c in password):
            errors.append("数字を含む必要があります")

        if self.require_symbols and not any(
            c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password
        ):
            errors.append("記号を含む必要があります")

        if password.lower() in self.common_passwords:
            errors.append("一般的すぎるパスワードです")

        # 連続文字チェック
        if self._has_sequential_chars(password):
            errors.append("連続した文字の使用は避けてください")

        return len(errors) == 0, errors

    def _has_sequential_chars(self, password: str) -> bool:
        """
        連続文字チェック
        
        連続する3文字以上の文字列（abc, 123など）を検出します。
        
        Args:
            password: 検証対象のパスワード
            
        Returns:
            bool: 連続文字が含まれているかどうか
        """
        for i in range(len(password) - 2):
            if (
                ord(password[i + 1]) == ord(password[i]) + 1
                and ord(password[i + 2]) == ord(password[i]) + 2
            ):
                return True
        return False

    def get_password_strength_score(self, password: str) -> float:
        """
        パスワード強度スコアの計算
        
        Args:
            password: 評価対象のパスワード
            
        Returns:
            float: 強度スコア（0.0-1.0）
        """
        score = 0.0
        max_score = 6.0  # 最大スコア

        # 長さによるスコア
        if len(password) >= self.min_length:
            score += 1.0
        else:
            score += len(password) / self.min_length

        # 文字種によるスコア
        if any(c.isupper() for c in password):
            score += 1.0
        if any(c.islower() for c in password):
            score += 1.0
        if any(c.isdigit() for c in password):
            score += 1.0
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1.0

        # 複雑性によるスコア
        if not self._has_sequential_chars(password):
            score += 1.0

        # 辞書攻撃耐性
        if password.lower() not in self.common_passwords:
            score += 1.0
        else:
            score -= 0.5  # 減点

        return min(max(score / max_score, 0.0), 1.0)

    def generate_password_suggestions(self) -> List[str]:
        """
        安全なパスワードの作成提案
        
        Returns:
            List[str]: パスワード作成のガイドライン
        """
        return [
            f"最低{self.min_length}文字以上の長さにしてください",
            "大文字、小文字、数字、記号を組み合わせてください",
            "辞書に載っている単語や一般的なパスワードは避けてください",
            "連続した文字（abc、123など）は使用しないでください",
            "個人情報（名前、生年月日など）を含めないでください",
            "パスフレーズ（複数の単語の組み合わせ）を検討してください",
            "パスワードマネージャーの使用を推奨します",
        ]