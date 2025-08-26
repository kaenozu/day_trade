#!/usr/bin/env python3
"""
アクセス制御システム - 多要素認証管理

このモジュールは、多要素認証（MFA）機能を提供します。
TOTP（Time-based One-Time Password）、QRコード生成、
バックアップコード管理を含む包括的なMFA機能を実装しています。
"""

import base64
import io
import secrets
from typing import List, Tuple

import pyotp
import qrcode


class MFAManager:
    """
    多要素認証管理システム
    
    TOTP、QRコード生成、バックアップコード管理を統合した
    セキュアな多要素認証機能を提供します。
    """

    def __init__(self):
        """
        MFA管理システムの初期化
        """
        self.issuer_name = "DayTrade Security"

    def generate_totp_secret(self) -> str:
        """
        TOTP秘密鍵生成
        
        Returns:
            str: Base32エンコードされた秘密鍵
        """
        return pyotp.random_base32()

    def generate_qr_code(self, username: str, secret: str) -> str:
        """
        QRコード生成（Base64エンコード済み画像）
        
        認証アプリに登録するためのQRコードを生成します。
        
        Args:
            username: ユーザー名
            secret: TOTP秘密鍵
            
        Returns:
            str: Base64エンコードされたPNG画像データ
        """
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username, issuer_name=self.issuer_name
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        return base64.b64encode(img_buffer.getvalue()).decode()

    def verify_totp(self, secret: str, token: str) -> bool:
        """
        TOTP検証
        
        Args:
            secret: TOTP秘密鍵
            token: 検証するTOTPコード
            
        Returns:
            bool: 検証結果
        """
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)  # 30秒の許容範囲
        except Exception:
            return False

    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """
        バックアップコード生成
        
        TOTPが使用できない場合の予備認証手段として
        使い捨てのバックアップコードを生成します。
        
        Args:
            count: 生成するコード数（デフォルト: 10）
            
        Returns:
            List[str]: バックアップコードのリスト
        """
        return [secrets.token_hex(4).upper() for _ in range(count)]

    def verify_backup_code(self, backup_codes: List[str], code: str) -> Tuple[bool, List[str]]:
        """
        バックアップコードの検証
        
        Args:
            backup_codes: 有効なバックアップコードのリスト
            code: 検証するコード
            
        Returns:
            Tuple[bool, List[str]]: (検証結果, 更新後のバックアップコードリスト)
        """
        code_upper = code.upper().strip()
        
        if code_upper in backup_codes:
            # 使用済みコードを削除
            updated_codes = [c for c in backup_codes if c != code_upper]
            return True, updated_codes
        
        return False, backup_codes

    def get_totp_current_code(self, secret: str) -> str:
        """
        現在のTOTPコード取得（テスト用）
        
        Args:
            secret: TOTP秘密鍵
            
        Returns:
            str: 現在有効なTOTPコード
        """
        try:
            totp = pyotp.TOTP(secret)
            return totp.now()
        except Exception:
            return ""

    def get_totp_remaining_time(self, secret: str) -> int:
        """
        TOTPコードの残り有効時間（秒）
        
        Args:
            secret: TOTP秘密鍵
            
        Returns:
            int: 残り有効時間（秒）
        """
        try:
            import time
            totp = pyotp.TOTP(secret)
            return 30 - (int(time.time()) % 30)
        except Exception:
            return 0

    def validate_secret_format(self, secret: str) -> bool:
        """
        TOTP秘密鍵フォーマット検証
        
        Args:
            secret: 検証する秘密鍵
            
        Returns:
            bool: フォーマットが正しいかどうか
        """
        try:
            # Base32デコードを試行
            import base64
            base64.b32decode(secret)
            return len(secret) >= 16  # 最低長チェック
        except Exception:
            return False

    def get_provisioning_uri(self, username: str, secret: str) -> str:
        """
        プロビジョニングURI取得
        
        Args:
            username: ユーザー名
            secret: TOTP秘密鍵
            
        Returns:
            str: プロビジョニングURI
        """
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name=self.issuer_name)

    def generate_recovery_info(self, username: str, secret: str) -> dict:
        """
        回復情報生成
        
        MFA設定時にユーザーに提供する情報パッケージを生成します。
        
        Args:
            username: ユーザー名
            secret: TOTP秘密鍵
            
        Returns:
            dict: 回復情報（QRコード、秘密鍵、バックアップコードなど）
        """
        backup_codes = self.generate_backup_codes()
        qr_code = self.generate_qr_code(username, secret)
        
        return {
            "secret_key": secret,
            "qr_code_image": qr_code,
            "backup_codes": backup_codes,
            "provisioning_uri": self.get_provisioning_uri(username, secret),
            "issuer": self.issuer_name,
            "username": username,
            "instructions": [
                "1. 認証アプリ（Google Authenticator、Authy等）でQRコードをスキャンしてください",
                "2. 秘密鍵を安全な場所に保管してください",
                "3. バックアップコードを印刷して安全な場所に保管してください",
                "4. バックアップコードは一度のみ使用可能です",
                "5. 認証アプリを紛失した場合はバックアップコードを使用してください",
            ],
        }