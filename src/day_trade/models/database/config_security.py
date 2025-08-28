"""
設定ファイルのセキュリティ管理モジュール
Alembic設定ファイルの安全な検索・検証機能

Issue #120: declarative_base()の定義場所の最適化対応
- セキュリティ強化（TOCTOU攻撃対策）
- ファイル検証の責務を明確化
"""

from pathlib import Path
from typing import Optional

from alembic.config import Config

from ...utils.exceptions import DatabaseError
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ConfigSecurityManager:
    """設定ファイルのセキュリティ管理クラス"""

    def __init__(self, database_url: str):
        """
        設定セキュリティ管理の初期化

        Args:
            database_url: データベース接続URL
        """
        self.database_url = database_url

    def get_secure_alembic_config(self, config_path: Optional[str] = None) -> Config:
        """
        セキュアなAlembic設定を取得（TOCTOU脆弱性対策・セキュリティ強化版）

        Args:
            config_path: alembic.iniファイルのパス（Noneの場合は自動検索）

        Returns:
            Alembic設定オブジェクト

        Raises:
            DatabaseError: 設定ファイルの読み込みに失敗した場合
        """
        if config_path is None:
            # 設定ファイルの自動検索（セキュリティ強化）
            config_path = self._find_secure_alembic_config()

        # TOCTOU脆弱性対策: 安全なファイルパス検証
        validated_config_path = self._validate_alembic_config_path(config_path)

        # 安全なAlembic設定作成
        try:
            alembic_cfg = Config(validated_config_path)
            alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)

            logger.debug(f"Alembic設定読み込み成功: {validated_config_path}")
            return alembic_cfg

        except Exception as e:
            logger.error(f"Alembic設定読み込みエラー: {validated_config_path} - {e}")
            raise DatabaseError(
                f"Failed to load Alembic config: {config_path}",
                error_code="ALEMBIC_CONFIG_LOAD_ERROR",
            ) from e

    def _find_secure_alembic_config(self) -> str:
        """
        安全なAlembic設定ファイル検索（TOCTOU対策）

        Returns:
            str: 検証済み安全な設定ファイルパス

        Raises:
            DatabaseError: 設定ファイルが見つからない場合
        """

        # 許可された検索ベースディレクトリ
        allowed_base_dirs = [
            Path.cwd(),  # 現在の作業ディレクトリ
            Path(__file__).parent.parent.parent.parent.parent,  # プロジェクトルート
        ]

        # 検索対象ファイル名パターン
        config_filenames = ["alembic.ini"]

        for base_dir in allowed_base_dirs:
            try:
                # ベースディレクトリの正規化と検証
                base_dir_resolved = base_dir.resolve()

                for filename in config_filenames:
                    config_path = base_dir_resolved / filename

                    # 原子的なファイル存在・読み取り可能チェック
                    if self._is_safe_readable_file(config_path):
                        logger.debug(f"Alembic設定ファイル発見: {config_path}")
                        return str(config_path)

            except Exception as e:
                logger.debug(f"ディレクトリ検索エラー: {base_dir} - {e}")
                continue

        # 設定ファイルが見つからない場合
        raise DatabaseError(
            "alembic.ini not found in any secure locations",
            error_code="ALEMBIC_CONFIG_NOT_FOUND",
        )

    def _validate_alembic_config_path(self, config_path: str) -> str:
        """
        Alembic設定ファイルパスの安全性検証（TOCTOU対策）

        Args:
            config_path: 検証対象のファイルパス

        Returns:
            str: 検証済み安全なファイルパス

        Raises:
            DatabaseError: 危険なパスまたはファイルアクセス不可の場合
        """

        try:
            # 1. パス正規化とセキュリティチェック
            path_obj = Path(config_path).resolve()

            # 2. 危険なパスパターンの検出
            path_str = str(path_obj).lower()
            dangerous_patterns = [
                "/etc/",
                "/usr/",
                "/var/",
                "/root/",
                "/boot/",  # Unix系システムディレクトリ
                "c:\\windows\\",
                "c:\\program files\\",  # Windowsシステムディレクトリ
                "\\\\",
                "/..",
                "\\..",  # UNCパス・パストラバーサル
            ]

            for pattern in dangerous_patterns:
                if pattern in path_str:
                    logger.warning(f"危険なAlembicパスパターン検出: {config_path}")
                    raise DatabaseError(
                        f"Dangerous path pattern detected: {config_path}",
                        error_code="ALEMBIC_DANGEROUS_PATH",
                    )

            # 3. 許可されたベースディレクトリ内かチェック
            allowed_base_dirs = [
                Path.cwd().resolve(),  # 現在の作業ディレクトリ
                Path(__file__).parent.parent.parent.parent.parent.resolve(),  # プロジェクトルート
            ]

            is_allowed = False
            for allowed_base in allowed_base_dirs:
                try:
                    # 許可されたベースディレクトリ内またはその配下かチェック
                    if path_obj == allowed_base or allowed_base in path_obj.parents:
                        is_allowed = True
                        break
                except Exception:
                    continue

            if not is_allowed:
                logger.warning(f"許可されていないAlembicパス: {path_obj}")
                raise DatabaseError(
                    f"Path outside allowed directories: {config_path}",
                    error_code="ALEMBIC_PATH_NOT_ALLOWED",
                )

            # 4. ファイルの存在と読み取り可能性を原子的にチェック
            if not self._is_safe_readable_file(path_obj):
                raise DatabaseError(
                    f"Alembic config file not accessible: {config_path}",
                    error_code="ALEMBIC_CONFIG_NOT_ACCESSIBLE",
                )

            logger.debug(f"Alembicパス検証完了: {path_obj}")
            return str(path_obj)

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Alembicパス検証エラー: {config_path} - {e}")
            raise DatabaseError(
                f"Path validation failed: {config_path}",
                error_code="ALEMBIC_PATH_VALIDATION_ERROR",
            ) from e

    def _is_safe_readable_file(self, file_path: Path) -> bool:
        """
        ファイルの安全な読み取り可能性チェック（TOCTOU対策）

        Args:
            file_path: チェック対象ファイルパス

        Returns:
            bool: 安全に読み取り可能な場合True
        """
        try:
            # 原子的操作: ファイル存在・タイプ・読み取り権限のチェック
            if not file_path.exists():
                return False

            if not file_path.is_file():
                logger.warning(f"通常ファイルではない: {file_path}")
                return False

            # シンボリックリンク攻撃対策
            if file_path.is_symlink():
                logger.warning(f"シンボリックリンクのためスキップ: {file_path}")
                return False

            # ファイルサイズ制限（設定ファイルが異常に大きい場合を検出）
            stat_info = file_path.stat()
            if stat_info.st_size > 10 * 1024 * 1024:  # 10MB制限
                logger.warning(
                    f"設定ファイルが大きすぎます: {file_path} ({stat_info.st_size} bytes)"
                )
                return False

            # 読み取り権限の確認
            with open(file_path, encoding="utf-8") as f:
                # ファイルの先頭を少し読んで読み取り可能性を確認
                f.read(1)

            return True

        except (PermissionError, OSError, FileNotFoundError):
            # ファイルアクセス不可の場合
            return False
        except UnicodeDecodeError:
            # 不正な文字エンコーディングの場合
            logger.warning(f"不正なエンコーディング: {file_path}")
            return False
        except Exception as e:
            logger.debug(f"ファイル読み取りチェックエラー: {file_path} - {e}")
            return False

    def validate_config_security(self, config_path: str) -> dict:
        """
        設定ファイルのセキュリティ状態を検証

        Args:
            config_path: 検証対象のファイルパス

        Returns:
            dict: セキュリティ検証結果
        """
        result = {
            "is_secure": False,
            "checks": {},
            "warnings": [],
            "errors": []
        }

        try:
            path_obj = Path(config_path).resolve()
            
            # 基本チェック
            result["checks"]["file_exists"] = path_obj.exists()
            result["checks"]["is_file"] = path_obj.is_file()
            result["checks"]["is_readable"] = self._is_safe_readable_file(path_obj)
            result["checks"]["is_symlink"] = path_obj.is_symlink()
            
            # サイズチェック
            if path_obj.exists():
                stat_info = path_obj.stat()
                result["checks"]["file_size"] = stat_info.st_size
                result["checks"]["size_ok"] = stat_info.st_size <= 10 * 1024 * 1024
            
            # パストラバーサルチェック
            path_str = str(path_obj).lower()
            dangerous_patterns = [
                "/etc/", "/usr/", "/var/", "/root/", "/boot/",
                "c:\\windows\\", "c:\\program files\\",
                "\\\\", "/..", "\\.."
            ]
            
            dangerous_found = [pattern for pattern in dangerous_patterns if pattern in path_str]
            result["checks"]["dangerous_patterns"] = dangerous_found
            result["checks"]["path_safe"] = len(dangerous_found) == 0
            
            # 全体評価
            result["is_secure"] = all([
                result["checks"]["file_exists"],
                result["checks"]["is_file"],
                result["checks"]["is_readable"],
                not result["checks"]["is_symlink"],
                result["checks"].get("size_ok", True),
                result["checks"]["path_safe"]
            ])
            
            # 警告とエラーの生成
            if result["checks"]["is_symlink"]:
                result["warnings"].append("ファイルがシンボリックリンクです")
            if not result["checks"].get("size_ok", True):
                result["warnings"].append("ファイルサイズが大きすぎます")
            if not result["checks"]["path_safe"]:
                result["errors"].append(f"危険なパスパターンが検出されました: {dangerous_found}")
                
        except Exception as e:
            result["errors"].append(f"検証エラー: {str(e)}")
        
        return result