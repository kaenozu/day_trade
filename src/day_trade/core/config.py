"""
設定ファイル管理モジュール
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..utils.logging_config import get_context_logger, log_error_with_context


class TradingConfig(BaseModel):
    """取引設定"""

    default_commission: float = Field(default=0.0, description="デフォルト手数料")
    tax_rate: float = Field(default=0.20315, description="譲渡益税率")


class DisplayConfig(BaseModel):
    """表示設定"""

    max_display_rows: int = Field(default=10, description="テーブル表示の最大行数")
    decimal_places: int = Field(default=0, description="価格表示の小数点以下桁数")
    color_enabled: bool = Field(default=True, description="カラー表示の有効/無効")


class APIConfig(BaseModel):
    """API設定"""

    cache_enabled: bool = Field(default=True, description="APIキャッシュの有効/無効")
    cache_size: int = Field(default=128, description="キャッシュサイズ")
    timeout: int = Field(default=30, description="APIタイムアウト（秒）")


class AppConfig(BaseModel):
    """アプリケーション設定"""

    trading: TradingConfig = Field(default_factory=TradingConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database_url: Optional[str] = Field(default=None, description="データベースURL")


class ConfigManager:
    """設定管理クラス"""

    DEFAULT_CONFIG_DIR = Path.home() / ".daytrade"
    DEFAULT_CONFIG_FILE = "config.json"

    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: 設定ファイルのパス（指定しない場合はデフォルト）
        """
        self.logger = get_context_logger(__name__, component="config_manager")

        if config_path is None:
            self.config_dir = self.DEFAULT_CONFIG_DIR
            self.config_path = self.config_dir / self.DEFAULT_CONFIG_FILE
        else:
            self.config_path = Path(config_path)
            self.config_dir = self.config_path.parent

        self.config = self.load_config()

    def load_config(self) -> AppConfig:
        """設定ファイルを読み込む"""
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    data = json.load(f)

                self.logger.info(
                    "設定ファイル読み込み成功",
                    config_path=str(self.config_path),
                    config_size=len(data),
                )
                return AppConfig(**data)
            except Exception as e:
                log_error_with_context(
                    e,
                    {
                        "operation": "load_config",
                        "config_path": str(self.config_path),
                        "error_type": type(e).__name__,
                    },
                )
                self.logger.warning(
                    "設定ファイル読み込みエラー、デフォルト設定を使用",
                    config_path=str(self.config_path),
                    error=str(e),
                )
                return AppConfig()
        else:
            self.logger.info(
                "設定ファイルが存在しません、デフォルト設定を使用",
                config_path=str(self.config_path),
            )
            return AppConfig()

    def save_config(self):
        """設定ファイルを保存"""
        try:
            # ディレクトリが存在しない場合は作成
            self.config_dir.mkdir(parents=True, exist_ok=True)

            # 設定を保存
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.model_dump(), f, indent=2, ensure_ascii=False)

            self.logger.info("設定ファイル保存成功", config_path=str(self.config_path))

        except PermissionError as e:
            self.logger.error(
                "設定ファイル保存失敗: 権限不足",
                config_path=str(self.config_path),
                error=str(e),
            )
            raise
        except OSError as e:
            self.logger.error(
                "設定ファイル保存失敗: ディスク容量不足またはIO エラー",
                config_path=str(self.config_path),
                error=str(e),
            )
            raise
        except Exception as e:
            self.logger.error(
                "設定ファイル保存失敗: 予期せぬエラー",
                config_path=str(self.config_path),
                error=str(e),
            )
            raise

    def update_config(self, updates: Dict[str, Any]):
        """設定を更新"""

        # ネストした辞書の更新に対応
        def update_nested(config_dict: dict, updates_dict: dict):
            for key, value in updates_dict.items():
                if isinstance(value, dict) and key in config_dict:
                    update_nested(config_dict[key], value)
                else:
                    config_dict[key] = value

        config_dict = self.config.model_dump()
        update_nested(config_dict, updates)
        self.config = AppConfig(**config_dict)
        self.save_config()

    def get(self, key: str, default: Any = None) -> Any:
        """ドット記法で設定値を取得"""
        keys = key.split(".")
        value = self.config.model_dump()

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """ドット記法で設定値を設定"""
        keys = key.split(".")
        updates = {}

        # ネストした辞書を作成
        current = updates
        for _i, k in enumerate(keys[:-1]):
            current[k] = {}
            current = current[k]
        current[keys[-1]] = value

        self.update_config(updates)

    def reset(self):
        """設定をデフォルトにリセット"""
        self.config = AppConfig()
        self.save_config()

    def export_config(self, path: Path):
        """設定をエクスポート"""
        try:
            # パスの親ディレクトリが存在することを確認
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.config.model_dump(), f, indent=2, ensure_ascii=False)

            self.logger.info("設定エクスポート成功", export_path=str(path))

        except PermissionError as e:
            self.logger.error(
                "設定エクスポート失敗: 権限不足", export_path=str(path), error=str(e)
            )
            raise
        except OSError as e:
            self.logger.error(
                "設定エクスポート失敗: ディスク容量不足またはIO エラー",
                export_path=str(path),
                error=str(e),
            )
            raise
        except Exception as e:
            self.logger.error(
                "設定エクスポート失敗: 予期せぬエラー",
                export_path=str(path),
                error=str(e),
            )
            raise

    def import_config(self, path: Path):
        """設定をインポート"""
        try:
            # ファイルの存在確認
            if not path.exists():
                raise FileNotFoundError(f"インポートファイルが見つかりません: {path}")

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # データの検証
            self.config = AppConfig(**data)
            self.save_config()

            self.logger.info("設定インポート成功", import_path=str(path))

        except FileNotFoundError as e:
            self.logger.error(
                "設定インポート失敗: ファイルが見つかりません",
                import_path=str(path),
                error=str(e),
            )
            raise
        except PermissionError as e:
            self.logger.error(
                "設定インポート失敗: 権限不足", import_path=str(path), error=str(e)
            )
            raise
        except json.JSONDecodeError as e:
            self.logger.error(
                "設定インポート失敗: JSONフォーマットエラー",
                import_path=str(path),
                error=str(e),
            )
            raise
        except Exception as e:
            self.logger.error(
                "設定インポート失敗: 予期せぬエラー",
                import_path=str(path),
                error=str(e),
            )
            raise


# グローバル設定インスタンス
config_manager = ConfigManager()
