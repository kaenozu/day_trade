"""
設定ファイル管理モジュール
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


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
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return AppConfig(**data)
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
                return AppConfig()
        else:
            # デフォルト設定を返す
            return AppConfig()

    def save_config(self):
        """設定ファイルを保存"""
        # ディレクトリが存在しない場合は作成
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 設定を保存
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=2, ensure_ascii=False)

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
        for i, k in enumerate(keys[:-1]):
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=2, ensure_ascii=False)

    def import_config(self, path: Path):
        """設定をインポート"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.config = AppConfig(**data)
        self.save_config()


# グローバル設定インスタンス
config_manager = ConfigManager()
