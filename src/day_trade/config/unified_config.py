"""
統合設定管理モジュール

各モジュールで分散していた設定クラスを統合し、
一元的で一貫性のある設定管理を提供します。

統合される設定:
- core.config からの AppConfig, TradingConfig, DisplayConfig, APIConfig
- analysis.screening_config からの ScreeningConfig
- utils.performance_config からのパフォーマンス設定
- utils.logging_config からのロギング設定
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="unified_config")


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


class ScreeningConfig(BaseModel):
    """スクリーニング設定（旧 analysis.screening_config から移行）"""

    default_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "volume_spike": 2.0,
            "strong_momentum": 0.05,
            "bollinger_squeeze": 0.02,
            "price_near_support": 0.03,
            "price_near_resistance": 0.03,
        },
        description="デフォルト閾値設定",
    )
    default_lookback_days: Dict[str, int] = Field(
        default_factory=lambda: {
            "rsi_oversold": 20,
            "rsi_overbought": 20,
            "macd_bullish": 20,
            "macd_bearish": 20,
            "golden_cross": 20,
            "dead_cross": 20,
            "volume_spike": 20,
            "strong_momentum": 20,
            "bollinger_breakout": 20,
            "bollinger_squeeze": 20,
            "price_near_support": 20,
            "price_near_resistance": 20,
            "reversal_pattern": 20,
        },
        description="デフォルト参照期間",
    )
    performance_settings: Dict[str, Union[int, str]] = Field(
        default_factory=lambda: {
            "max_workers": 5,
            "cache_size": 100,
            "data_period": "3mo",
            "min_data_points": 30,
        },
        description="パフォーマンス設定",
    )
    predefined_screeners: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="事前定義スクリーナー"
    )
    week_calculation: Dict[str, bool] = Field(
        default_factory=lambda: {
            "use_actual_52_weeks": True,
            "fallback_to_available_data": True,
        },
        description="52週間計算設定",
    )
    formatting: Dict[str, Union[bool, int]] = Field(
        default_factory=lambda: {
            "use_formatters": True,
            "currency_precision": 0,
            "percentage_precision": 2,
            "volume_compact": True,
        },
        description="フォーマット設定",
    )


class PerformanceConfig(BaseModel):
    """パフォーマンス設定"""

    enable_optimization: bool = Field(default=True, description="最適化を有効にする")
    parallel_processing: bool = Field(default=True, description="並列処理を有効にする")
    max_workers: int = Field(default=4, description="最大ワーカー数")
    cache_enabled: bool = Field(default=True, description="キャッシュを有効にする")
    batch_size: int = Field(default=100, description="バッチサイズ")


class LoggingConfig(BaseModel):
    """ロギング設定"""

    level: str = Field(default="INFO", description="ログレベル")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="ログフォーマット",
    )
    enable_file_logging: bool = Field(
        default=True, description="ファイルロギングを有効にする"
    )
    log_directory: str = Field(default="logs", description="ログディレクトリ")
    max_file_size_mb: int = Field(
        default=10, description="ログファイル最大サイズ（MB）"
    )
    backup_count: int = Field(default=5, description="ログファイルバックアップ数")


class SecurityConfig(BaseModel):
    """セキュリティ設定"""

    enable_encryption: bool = Field(default=True, description="暗号化を有効にする")
    api_key_length: int = Field(default=32, description="APIキーの長さ")
    session_timeout_minutes: int = Field(
        default=60, description="セッションタイムアウト（分）"
    )


class UnifiedAppConfig(BaseModel):
    """統合アプリケーション設定"""

    trading: TradingConfig = Field(default_factory=TradingConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database_url: Optional[str] = Field(default=None, description="データベースURL")


class UnifiedConfigManager:
    """統合設定管理クラス"""

    DEFAULT_CONFIG_DIR = Path.home() / ".daytrade"
    DEFAULT_CONFIG_FILE = "unified_config.json"

    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: 設定ファイルのパス（指定しない場合はデフォルト）
        """
        self.logger = get_context_logger(__name__, component="unified_config_manager")

        if config_path is None:
            self.config_dir = self.DEFAULT_CONFIG_DIR
            self.config_path = self.config_dir / self.DEFAULT_CONFIG_FILE
        else:
            self.config_path = Path(config_path)
            self.config_dir = self.config_path.parent

        self.config = self.load_config()

    def load_config(self) -> UnifiedAppConfig:
        """設定ファイルを読み込む"""
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    data = json.load(f)

                self.logger.info(
                    f"統合設定ファイル読み込み成功: {str(self.config_path)} (size: {len(data)})"
                )
                return UnifiedAppConfig(**data)
            except Exception as e:
                self.logger.error(
                    f"設定ファイル読み込みエラー、デフォルト設定を使用: {str(self.config_path)} - {str(e)}"
                )
                return UnifiedAppConfig()
        else:
            self.logger.info(
                f"設定ファイルが存在しません、デフォルト設定を使用: {str(self.config_path)}"
            )
            return UnifiedAppConfig()

    def save_config(self):
        """設定ファイルを保存"""
        try:
            # ディレクトリが存在しない場合は作成
            self.config_dir.mkdir(parents=True, exist_ok=True)

            # 設定を保存
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.model_dump(), f, indent=2, ensure_ascii=False)

            self.logger.info(f"統合設定ファイル保存成功: {str(self.config_path)}")

        except Exception as e:
            self.logger.error(
                f"設定ファイル保存失敗: {str(self.config_path)} - {str(e)}"
            )
            raise

    def get_trading_config(self) -> TradingConfig:
        """取引設定を取得"""
        return self.config.trading

    def get_display_config(self) -> DisplayConfig:
        """表示設定を取得"""
        return self.config.display

    def get_api_config(self) -> APIConfig:
        """API設定を取得"""
        return self.config.api

    def get_screening_config(self) -> ScreeningConfig:
        """スクリーニング設定を取得"""
        return self.config.screening

    def get_performance_config(self) -> PerformanceConfig:
        """パフォーマンス設定を取得"""
        return self.config.performance

    def get_logging_config(self) -> LoggingConfig:
        """ロギング設定を取得"""
        return self.config.logging

    def get_security_config(self) -> SecurityConfig:
        """セキュリティ設定を取得"""
        return self.config.security

    def update_config(self, section: str, updates: Dict[str, Any]):
        """設定セクションを更新"""
        if hasattr(self.config, section):
            section_config = getattr(self.config, section)
            if hasattr(section_config, "model_copy"):
                # Pydantic v2
                updated_section = section_config.model_copy(update=updates)
            else:
                # Pydantic v1
                updated_section = section_config.copy(update=updates)
            setattr(self.config, section, updated_section)
            self.save_config()
        else:
            raise ValueError(f"Unknown configuration section: {section}")

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
        if len(keys) < 2:
            raise ValueError("Key must contain at least one dot (section.property)")

        section = keys[0]
        property_keys = keys[1:]

        # ネストした辞書を作成
        updates = {}
        current = updates
        for k in property_keys[:-1]:
            current[k] = {}
            current = current[k]
        current[property_keys[-1]] = value

        self.update_config(section, updates)

    def reset(self):
        """設定をデフォルトにリセット"""
        self.config = UnifiedAppConfig()
        self.save_config()

    def export_config(self, path: Path):
        """設定をエクスポート"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.config.model_dump(), f, indent=2, ensure_ascii=False)
            self.logger.info(f"統合設定エクスポート成功: {str(path)}")
        except Exception as e:
            self.logger.error(f"設定エクスポート失敗: {str(path)} - {str(e)}")
            raise

    def import_config(self, path: Path):
        """設定をインポート"""
        try:
            if not path.exists():
                raise FileNotFoundError(f"インポートファイルが見つかりません: {path}")

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            self.config = UnifiedAppConfig(**data)
            self.save_config()

            self.logger.info(f"統合設定インポート成功: {str(path)}")

        except Exception as e:
            self.logger.error(f"設定インポート失敗: {str(path)} - {str(e)}")
            raise


# グローバル統合設定インスタンス
_unified_config_manager = None


def get_unified_config_manager() -> UnifiedConfigManager:
    """統合設定マネージャーを取得（シングルトン）"""
    global _unified_config_manager
    if _unified_config_manager is None:
        _unified_config_manager = UnifiedConfigManager()
    return _unified_config_manager


def set_unified_config_manager(config_manager: UnifiedConfigManager):
    """統合設定マネージャーを設定（テスト用）"""
    global _unified_config_manager
    _unified_config_manager = config_manager


# 後方互換性のための便利関数
def get_screening_config() -> ScreeningConfig:
    """スクリーニング設定を取得（後方互換性）"""
    return get_unified_config_manager().get_screening_config()


def get_performance_config() -> PerformanceConfig:
    """パフォーマンス設定を取得"""
    return get_unified_config_manager().get_performance_config()


def get_logging_config() -> LoggingConfig:
    """ロギング設定を取得"""
    return get_unified_config_manager().get_logging_config()
