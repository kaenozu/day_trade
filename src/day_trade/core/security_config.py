"""
セキュリティ強化された設定管理
機密情報の暗号化と環境変数の活用
"""

import base64
import json
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None
    hashes = None
    PBKDF2HMAC = None

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SecureConfigManager:
    """セキュリティ強化された設定管理クラス"""

    def __init__(
        self, config_path: Optional[Path] = None, encryption_key: Optional[str] = None
    ):
        """
        初期化

        Args:
            config_path: 設定ファイルのパス
            encryption_key: 暗号化キー（環境変数から取得推奨）
        """
        self.logger = get_context_logger(__name__, component="secure_config")

        # 設定ファイルパス
        if config_path is None:
            config_dir = Path.home() / ".daytrade"
            self.config_path = config_dir / "secure_config.json"
        else:
            self.config_path = Path(config_path)

        # 暗号化キーの設定
        self.encryption_key = encryption_key or os.environ.get(
            "DAYTRADE_ENCRYPTION_KEY"
        )
        if not CRYPTO_AVAILABLE:
            self.logger.warning(
                "cryptographyライブラリが利用できません。暗号化機能は無効です"
            )
            self._cipher = None
        elif not self.encryption_key:
            self.logger.warning(
                "暗号化キーが設定されていません。機密情報の暗号化は利用できません"
            )
            self._cipher = None
        else:
            # 既存のソルトがあるかチェック（復号化時に必要）
            existing_salt = self._load_salt()
            if existing_salt:
                # 既存のソルトを使用して暗号化オブジェクトを作成
                self._cipher = self._create_cipher_with_salt(self.encryption_key, existing_salt)
            else:
                # 新規作成
                self._cipher = self._create_cipher(self.encryption_key)

        # 機密情報として扱うキー（パターンマッチング）
        self.sensitive_keys = {
            "password",
            "pass",
            "secret",
            "key",
            "token",
            "api_key",
            "smtp_password",
            "db_password",
            "webhook_secret",
        }

    def _generate_salt(self) -> bytes:
        """
        セキュアなランダムソルトを生成

        Returns:
            32バイトのランダムソルト
        """
        return secrets.token_bytes(32)

    def _create_cipher_with_salt(self, password: str, salt: bytes):
        """
        既存のソルトを使用して暗号化オブジェクトを作成

        Args:
            password: パスワード
            salt: 既存のソルト

        Returns:
            FernetオブジェクトまたはNone
        """
        if not CRYPTO_AVAILABLE:
            return None

        try:
            password_bytes = password.encode("utf-8")

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )

            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            return Fernet(key)

        except Exception as e:
            self.logger.error(f"既存ソルトでの暗号化オブジェクト作成に失敗: {e}")
            return None

    def _create_cipher(self, password: str):
        """
        暗号化キーから暗号化オブジェクトを作成

        セキュリティ強化:
        - 動的ソルト生成
        - エラーハンドリング強化
        - 適切なKDF設定
        """
        if not CRYPTO_AVAILABLE:
            self.logger.error("cryptographyライブラリが利用できないため、暗号化を初期化できません")
            return None

        try:
            # パスワードベースの鍵導出
            password_bytes = password.encode("utf-8")

            # セキュアなランダムソルトを生成（固定ソルトは使用しない）
            salt = self._generate_salt()

            # ソルトをファイルに保存（後で復号化時に必要）
            salt_file = self.config_path.parent / "salt.key"
            try:
                with open(salt_file, "wb") as f:
                    f.write(salt)
                self._set_secure_file_permissions(salt_file)
            except Exception as e:
                self.logger.error(f"ソルトファイルの保存に失敗: {e}")
                # ソルトを保存できない場合は暗号化を無効化
                return None

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,  # セキュリティを高めるため反復回数を維持
            )

            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            return Fernet(key)

        except Exception as e:
            self.logger.error(f"暗号化オブジェクトの作成に失敗: {e}")
            return None

    def _load_salt(self) -> Optional[bytes]:
        """
        保存されたソルトを読み込み

        Returns:
            ソルト、またはNone（ファイルが存在しない場合）
        """
        salt_file = self.config_path.parent / "salt.key"
        try:
            if salt_file.exists():
                with open(salt_file, "rb") as f:
                    return f.read()
        except Exception as e:
            self.logger.error(f"ソルトファイルの読み込みに失敗: {e}")
        return None

    def _set_secure_file_permissions(self, file_path: Path):
        """
        クロスプラットフォームでのセキュアなファイル権限設定

        Args:
            file_path: 権限を設定するファイルのパス
        """
        try:
            if sys.platform == "win32":
                # Windows環境では警告を出力
                self.logger.warning(
                    f"Windows環境では自動的なファイル権限制限ができません。"
                    f"ファイル '{file_path}' へのアクセス権限を手動で制限してください。"
                )
            else:
                # Unix系OSでのみ権限制限を適用
                os.chmod(file_path, 0o600)  # 所有者のみ読み書き可能
                self.logger.debug(f"ファイル権限を制限: {file_path}")
        except Exception as e:
            self.logger.error(f"ファイル権限の設定に失敗: {file_path} - {e}")

    def _is_sensitive_key(self, key: str) -> bool:
        """キーが機密情報かどうかを判定"""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.sensitive_keys)

    def _encrypt_value(self, value: Any) -> str:
        """
        値を暗号化（厳格なエラーハンドリング）

        Args:
            value: 暗号化する値

        Returns:
            暗号化された値

        Raises:
            ValueError: 暗号化が必要だが利用できない場合
        """
        if self._cipher is None:
            # 暗号化が期待されるが利用できない場合は例外を発生
            raise ValueError(
                "機密情報の暗号化が必要ですが、暗号化システムが利用できません。"
                "暗号化キーの設定またはcryptographyライブラリのインストールを確認してください。"
            )

        try:
            value_str = json.dumps(value, ensure_ascii=False)
            encrypted_bytes = self._cipher.encrypt(value_str.encode("utf-8"))
            return base64.urlsafe_b64encode(encrypted_bytes).decode("utf-8")
        except Exception as e:
            self.logger.error(f"値の暗号化に失敗: {e}")
            raise ValueError(f"機密情報の暗号化に失敗しました: {e}")

    def _decrypt_value(self, encrypted_value: str) -> Any:
        """
        値を復号化（厳格なエラーハンドリング）

        Args:
            encrypted_value: 暗号化された値

        Returns:
            復号化された値

        Raises:
            ValueError: 復号化が必要だが失敗した場合
        """
        if self._cipher is None:
            raise ValueError(
                "暗号化された機密情報の復号化が必要ですが、暗号化システムが利用できません。"
                "暗号化キーの設定またはcryptographyライブラリのインストールを確認してください。"
            )

        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode("utf-8"))
            decrypted_str = self._cipher.decrypt(encrypted_bytes).decode("utf-8")
            return json.loads(decrypted_str)
        except Exception as e:
            self.logger.error(f"値の復号化に失敗: {e}")
            raise ValueError(f"機密情報の復号化に失敗しました: {e}")

    def _process_config_for_save(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """設定を保存用に処理（機密情報を暗号化）"""
        processed = {}

        for key, value in config.items():
            if isinstance(value, dict):
                # ネストした辞書を再帰的に処理
                processed[key] = self._process_config_for_save(value)
            else:
                if self._is_sensitive_key(key):
                    # 機密情報を暗号化
                    processed[f"encrypted_{key}"] = self._encrypt_value(value)
                    self.logger.debug(f"機密情報を暗号化: {key}")
                else:
                    processed[key] = value

        return processed

    def _process_config_for_load(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """設定を読み込み用に処理（暗号化された情報を復号化）"""
        processed = {}

        for key, value in config.items():
            if isinstance(value, dict):
                # ネストした辞書を再帰的に処理
                processed[key] = self._process_config_for_load(value)
            elif key.startswith("encrypted_"):
                # 暗号化された値を復号化
                original_key = key[10:]  # "encrypted_" を除去
                processed[original_key] = self._decrypt_value(value)
                self.logger.debug(f"機密情報を復号化: {original_key}")
            else:
                processed[key] = value

        return processed

    def save_config(self, config: Dict[str, Any]):
        """設定を安全に保存（クロスプラットフォーム対応）"""
        try:
            # ディレクトリ作成
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # 機密情報を暗号化
            processed_config = self._process_config_for_save(config)

            # ファイル保存（適切な権限設定）
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(processed_config, f, indent=2, ensure_ascii=False)

            # クロスプラットフォームでのセキュアなファイル権限設定
            self._set_secure_file_permissions(self.config_path)

            self.logger.info(f"セキュア設定を保存: {self.config_path}")

        except Exception as e:
            self.logger.error(f"セキュア設定の保存に失敗: {e}")
            raise

    def load_config(self) -> Dict[str, Any]:
        """設定を安全に読み込み"""
        try:
            if not self.config_path.exists():
                self.logger.info(f"設定ファイルが存在しません: {self.config_path}")
                return {}

            with open(self.config_path, encoding="utf-8") as f:
                encrypted_config = json.load(f)

            # 暗号化された情報を復号化
            config = self._process_config_for_load(encrypted_config)

            self.logger.info(f"セキュア設定を読み込み: {self.config_path}")
            return config

        except Exception as e:
            self.logger.error(f"セキュア設定の読み込みに失敗: {e}")
            return {}

    def get_from_env_or_config(
        self, key: str, config: Dict[str, Any], default: Any = None
    ) -> Any:
        """環境変数または設定ファイルから値を取得（環境変数を優先）"""
        # 環境変数名を生成（大文字＋アンダースコア）
        env_key = f"DAYTRADE_{key.upper().replace('.', '_')}"

        # 環境変数から取得を試行
        env_value = os.environ.get(env_key)
        if env_value is not None:
            self.logger.debug(f"環境変数から設定を取得: {env_key}")
            # 型変換の試行
            return self._parse_env_value(env_value)

        # 設定ファイルから取得
        return self._get_nested_value(config, key, default)

    def _parse_env_value(self, value: str) -> Any:
        """環境変数の値を適切な型に変換"""
        # ブール値
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # 数値
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # JSON形式
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # 文字列として返す
        return value

    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any) -> Any:
        """ドット記法でネストした値を取得"""
        keys = key.split(".")
        current = config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def _validate_password_strength(self, password: str) -> List[str]:
        """
        パスワード強度を検証

        Args:
            password: 検証するパスワード

        Returns:
            問題のリスト（空の場合は強度十分）
        """
        issues = []

        if len(password) < 8:
            issues.append("最低8文字が必要です")

        if len(password) < 12:
            issues.append("12文字以上を推奨します")

        if not re.search(r'[a-z]', password):
            issues.append("小文字を含める必要があります")

        if not re.search(r'[A-Z]', password):
            issues.append("大文字を含める必要があります")

        if not re.search(r'\d', password):
            issues.append("数字を含める必要があります")

        if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
            issues.append("記号を含めることを推奨します")

        # 一般的な弱いパスワードチェック
        weak_passwords = {
            'password', '12345678', 'qwerty123', 'admin123',
            'password123', '123456789', 'welcome123', 'letmein123'
        }

        if password.lower() in weak_passwords:
            issues.append("一般的すぎるパスワードです")

        # 同じ文字の連続チェック
        if re.search(r'(.)\1{2,}', password):
            issues.append("同じ文字の連続は避けてください")

        return issues

    def validate_config_security(self, config: Dict[str, Any]) -> List[str]:
        """
        設定のセキュリティ検証（強化版）

        Args:
            config: 検証する設定

        Returns:
            セキュリティ問題のリスト
        """
        warnings = []

        def check_dict(d: Dict[str, Any], path: str = ""):
            for key, value in d.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    check_dict(value, current_path)
                elif isinstance(value, str):
                    # 機密情報のプレーンテキスト保存をチェック
                    if self._is_sensitive_key(key) and not key.startswith("encrypted_"):
                        warnings.append(
                            f"機密情報がプレーンテキストで保存されています: {current_path}"
                        )

                    # パスワード強度の詳細チェック
                    if "password" in key.lower():
                        password_issues = self._validate_password_strength(value)
                        for issue in password_issues:
                            warnings.append(f"パスワード強度の問題 ({current_path}): {issue}")

                    # APIキーの形式チェック
                    if "api_key" in key.lower() or "token" in key.lower():
                        if len(value) < 16:
                            warnings.append(f"APIキー/トークンが短すぎます: {current_path}")

                    # データベースURLの機密情報チェック
                    if "database_url" in key.lower() or "db_url" in key.lower():
                        if "password" in value.lower() and "***" not in value:
                            warnings.append(f"データベースURLに平文パスワードが含まれています: {current_path}")

        check_dict(config)
        return warnings

    def suggest_security_improvements(self, config: Dict[str, Any]) -> List[str]:
        """
        セキュリティ改善提案

        Args:
            config: チェックする設定

        Returns:
            改善提案のリスト
        """
        suggestions = []

        # 暗号化設定のチェック
        if not self.encryption_key:
            suggestions.append("暗号化キーを環境変数 DAYTRADE_ENCRYPTION_KEY に設定することを強く推奨します")

        if not CRYPTO_AVAILABLE:
            suggestions.append("cryptographyライブラリのインストールを推奨します: pip install cryptography")

        # 明示的な機密情報指定の推奨
        sensitive_found = False
        def check_for_sensitive(d: Dict[str, Any]):
            nonlocal sensitive_found
            for key, value in d.items():
                if isinstance(value, dict):
                    check_for_sensitive(value)
                elif isinstance(value, str) and self._is_sensitive_key(key):
                    sensitive_found = True

        check_for_sensitive(config)

        if sensitive_found:
            suggestions.append("機密情報フィールドに '_sensitive' サフィックスを付けることで、明示的にマークできます")

        # バックアップの推奨
        suggestions.append("設定ファイルの定期的なバックアップを推奨します（暗号化したまま）")

        return suggestions


class EnvironmentConfigLoader:
    """環境変数による設定読み込み"""

    @staticmethod
    def load_database_config() -> Dict[str, Any]:
        """データベース設定を環境変数から読み込み"""
        return {
            "database_url": os.environ.get("DAYTRADE_DATABASE_URL"),
            "database_pool_size": int(
                os.environ.get("DAYTRADE_DATABASE_POOL_SIZE", "10")
            ),
            "database_max_overflow": int(
                os.environ.get("DAYTRADE_DATABASE_MAX_OVERFLOW", "20")
            ),
            "database_pool_timeout": int(
                os.environ.get("DAYTRADE_DATABASE_POOL_TIMEOUT", "30")
            ),
        }

    @staticmethod
    def load_api_config() -> Dict[str, Any]:
        """API設定を環境変数から読み込み"""
        return {
            "api_timeout": int(os.environ.get("DAYTRADE_API_TIMEOUT", "30")),
            "api_retry_count": int(os.environ.get("DAYTRADE_API_RETRY_COUNT", "3")),
            "api_cache_enabled": os.environ.get(
                "DAYTRADE_API_CACHE_ENABLED", "true"
            ).lower()
            == "true",
            "api_cache_size": int(os.environ.get("DAYTRADE_API_CACHE_SIZE", "128")),
        }

    @staticmethod
    def load_smtp_config() -> Dict[str, Any]:
        """SMTP設定を環境変数から読み込み"""
        return {
            "smtp_server": os.environ.get("DAYTRADE_SMTP_SERVER"),
            "smtp_port": int(os.environ.get("DAYTRADE_SMTP_PORT", "587")),
            "smtp_user": os.environ.get("DAYTRADE_SMTP_USER"),
            "smtp_password": os.environ.get("DAYTRADE_SMTP_PASSWORD"),
            "smtp_from_email": os.environ.get("DAYTRADE_SMTP_FROM_EMAIL"),
            "smtp_to_emails": os.environ.get("DAYTRADE_SMTP_TO_EMAILS", "").split(",")
            if os.environ.get("DAYTRADE_SMTP_TO_EMAILS")
            else [],
        }


def create_example_env_file(path: Path):
    """
    サンプル環境変数ファイルを作成（セキュリティ強化）

    Args:
        path: 作成するファイルのパス
    """
    env_content = """# Day Trade アプリケーション環境変数設定
# セキュリティのため、このファイルをバージョン管理にコミットしないでください
# 本番環境では、実際の値に置き換えてください

# 暗号化キー（必須・強力なランダム値に変更）
# 実際の使用時は以下のコマンドで生成してください:
# python -c "import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
DAYTRADE_ENCRYPTION_KEY=YOUR_SECURE_ENCRYPTION_KEY_HERE

# データベース設定
DAYTRADE_DATABASE_URL=sqlite:///day_trade.db
DAYTRADE_DATABASE_POOL_SIZE=10
DAYTRADE_DATABASE_MAX_OVERFLOW=20

# API設定
DAYTRADE_API_TIMEOUT=30
DAYTRADE_API_RETRY_COUNT=3
DAYTRADE_API_CACHE_ENABLED=true

# SMTP設定（メール通知・実際の値に置き換え）
DAYTRADE_SMTP_SERVER=smtp.gmail.com
DAYTRADE_SMTP_PORT=587
DAYTRADE_SMTP_USER=your_email@gmail.com
DAYTRADE_SMTP_PASSWORD=your_strong_app_password_here
DAYTRADE_SMTP_FROM_EMAIL=your_email@gmail.com
DAYTRADE_SMTP_TO_EMAILS=recipient1@example.com,recipient2@example.com

# Webhook設定（強力なランダム値に変更）
# 実際の使用時は以下のコマンドで生成してください:
# python -c "import secrets; print(secrets.token_urlsafe(32))"
DAYTRADE_WEBHOOK_SECRET=YOUR_SECURE_WEBHOOK_SECRET_HERE

# ログレベル
DAYTRADE_LOG_LEVEL=INFO

# セキュリティ注意事項:
# - パスワードは12文字以上、大文字・小文字・数字・記号を含む
# - APIキーは正規のプロバイダーから取得したものを使用
# - 本番環境では環境変数に直接設定することを推奨
"""

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(env_content)

        # クロスプラットフォーム対応のファイル権限設定
        if sys.platform == "win32":
            logger.warning(
                f"Windows環境: 作成されたファイル '{path}' への"
                f"アクセス権限を手動で制限してください。"
            )
        else:
            try:
                os.chmod(path, 0o600)  # 所有者のみ読み書き可能
                logger.debug(f"ファイル権限を制限: {path}")
            except Exception as e:
                logger.warning(f"ファイル権限の設定に失敗: {path} - {e}")

        logger.info(f"セキュリティ強化されたサンプル環境変数ファイルを作成: {path}")

    except Exception as e:
        logger.error(f"サンプル環境変数ファイルの作成に失敗: {e}")
        raise
