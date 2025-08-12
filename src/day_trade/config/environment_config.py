#!/usr/bin/env python3
"""
環境設定管理システム - 環境別設定の統合管理

Issue #320: システム本番稼働準備・設定最適化
環境変数・設定ファイル・セキュリティの統合管理
"""

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


# 環境タイプ
class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class ConfigurationError(Exception):
    """設定エラー"""

    message: str
    config_path: Optional[str] = None
    environment: Optional[str] = None


@dataclass
class SecurityConfig:
    """セキュリティ設定"""

    api_keys_encrypted: bool = True
    access_control_enabled: bool = True
    data_encryption_enabled: bool = True
    allowed_ips: list = field(default_factory=list)
    rate_limit_rpm: int = 60


@dataclass
class SystemConfig:
    """システム設定"""

    performance_target_seconds: float = 3.6
    memory_limit_mb: int = 2048
    cpu_limit_percent: int = 80
    logging_level: str = "INFO"
    monitoring_enabled: bool = True


@dataclass
class TradingConfig:
    """取引設定"""

    max_position_percent: float = 5.0
    stop_loss_percent: float = 2.0
    profit_taking_percent: float = 3.0
    max_daily_trades: int = 10
    paper_trading: bool = False


class EnvironmentConfigManager:
    """環境設定管理マネージャー"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.current_environment = self._detect_environment()
        self.config_cache = {}
        self.logger = logging.getLogger(__name__)

        # 設定ディレクトリ作成
        self.config_dir.mkdir(exist_ok=True)

        self.logger.info(f"環境設定管理システム初期化完了 - 環境: {self.current_environment.value}")

    def _detect_environment(self) -> Environment:
        """環境自動検出"""
        # 環境変数から取得
        env_var = os.getenv("DAYTRADE_ENV", "development").lower()

        # 環境変数マッピング
        env_mapping = {
            "dev": Environment.DEVELOPMENT,
            "development": Environment.DEVELOPMENT,
            "staging": Environment.STAGING,
            "stage": Environment.STAGING,
            "prod": Environment.PRODUCTION,
            "production": Environment.PRODUCTION,
            "test": Environment.TEST,
            "testing": Environment.TEST,
        }

        return env_mapping.get(env_var, Environment.DEVELOPMENT)

    def load_config(self, environment: Optional[Environment] = None) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        env = environment or self.current_environment

        # キャッシュチェック
        cache_key = env.value
        if cache_key in self.config_cache:
            return copy.deepcopy(self.config_cache[cache_key])

        # 設定ファイルパス
        config_file = self.config_dir / f"{env.value}.json"

        if not config_file.exists():
            # フォールバック: デフォルト設定ファイル
            default_config = self.config_dir / "settings.json"
            if default_config.exists():
                config_file = default_config
            else:
                raise ConfigurationError(
                    f"設定ファイルが見つかりません: {config_file}",
                    str(config_file),
                    env.value,
                )

        try:
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)

            # 環境変数で設定を上書き
            config = self._apply_environment_overrides(config)

            # 設定検証
            self._validate_config(config, env)

            # キャッシュに保存
            self.config_cache[cache_key] = copy.deepcopy(config)

            self.logger.info(f"設定ファイル読み込み完了: {config_file}")
            return config

        except json.JSONDecodeError as e:
            raise ConfigurationError(f"設定ファイル形式エラー: {e}", str(config_file), env.value)
        except Exception as e:
            raise ConfigurationError(
                f"設定ファイル読み込みエラー: {e}", str(config_file), env.value
            )

    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """環境変数による設定上書き"""
        # 重要な設定項目の環境変数マッピング
        env_mappings = {
            "DAYTRADE_DB_URL": ["data", "storage", "database_url"],
            "DAYTRADE_LOG_LEVEL": ["system", "logging", "level"],
            "DAYTRADE_MAX_SYMBOLS": ["watchlist", "max_symbols"],
            "DAYTRADE_PAPER_TRADING": ["daytrading", "paper_trading"],
            "DAYTRADE_MAX_POSITION": [
                "daytrading",
                "risk_management",
                "max_position_per_stock",
            ],
            "DAYTRADE_MONITORING": ["system", "monitoring", "enabled"],
            "DAYTRADE_FAULT_TOLERANCE": ["system", "fault_tolerance", "enabled"],
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # 型変換
                if env_value.lower() in ["true", "false"]:
                    env_value = env_value.lower() == "true"
                elif env_value.isdigit():
                    env_value = int(env_value)
                elif self._is_float(env_value):
                    env_value = float(env_value)

                # ネストした設定に値を設定
                self._set_nested_config(config, config_path, env_value)
                self.logger.info(f"環境変数で設定を上書き: {env_var} = {env_value}")

        return config

    def _is_float(self, value: str) -> bool:
        """float値判定"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _set_nested_config(self, config: Dict, path: list, value: Any):
        """ネストした設定値の設定"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _validate_config(self, config: Dict[str, Any], environment: Environment):
        """設定検証"""
        required_sections = [
            "watchlist",
            "system",
            "daytrading",
            "analysis",
            "portfolio",
            "alerts",
            "data",
        ]

        # 必須セクション確認
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(
                    f"必須設定セクションが不足: {section}",
                    environment=environment.value,
                )

        # 本番環境の追加検証
        if environment == Environment.PRODUCTION:
            self._validate_production_config(config)

        self.logger.info(f"設定検証完了: {environment.value}")

    def _validate_production_config(self, config: Dict[str, Any]):
        """本番環境設定の追加検証"""
        prod_requirements = [
            (["system", "logging", "log_to_file"], True),
            (["system", "monitoring", "enabled"], True),
            (["system", "fault_tolerance", "enabled"], True),
            (["security", "data_protection", "encryption_at_rest"], True),
            (["data", "storage", "backup_enabled"], True),
            (
                ["daytrading", "risk_management", "max_position_per_stock"],
                lambda x: x <= 10.0,
            ),
        ]

        for config_path, expected_value in prod_requirements:
            actual_value = self._get_nested_config(config, config_path)

            if callable(expected_value):
                if not expected_value(actual_value):
                    raise ConfigurationError(
                        f"本番環境設定要件未満: {'.'.join(config_path)} = {actual_value}",
                        environment="production",
                    )
            elif actual_value != expected_value:
                raise ConfigurationError(
                    f"本番環境必須設定が無効: {'.'.join(config_path)} = {actual_value} (expected: {expected_value})",
                    environment="production",
                )

    def _get_nested_config(self, config: Dict, path: list) -> Any:
        """ネストした設定値の取得"""
        current = config
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def get_config(
        self,
        key_path: str,
        default: Any = None,
        environment: Optional[Environment] = None,
    ) -> Any:
        """設定値取得（ドット記法対応）"""
        config = self.load_config(environment)
        keys = key_path.split(".")

        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def get_system_config(self, environment: Optional[Environment] = None) -> SystemConfig:
        """システム設定取得"""
        config = self.load_config(environment)
        system_config = config.get("system", {})

        return SystemConfig(
            performance_target_seconds=system_config.get("performance", {}).get(
                "ml_analysis_target_seconds", 3.6
            ),
            memory_limit_mb=system_config.get("performance", {}).get("memory_limit_mb", 2048),
            cpu_limit_percent=system_config.get("performance", {}).get("cpu_limit_percent", 80),
            logging_level=system_config.get("logging", {}).get("level", "INFO"),
            monitoring_enabled=system_config.get("monitoring", {}).get("enabled", True),
        )

    def get_security_config(self, environment: Optional[Environment] = None) -> SecurityConfig:
        """セキュリティ設定取得"""
        config = self.load_config(environment)
        security_config = config.get("security", {})

        return SecurityConfig(
            api_keys_encrypted=security_config.get("api_keys", {}).get("encryption", True),
            access_control_enabled=security_config.get("access_control", {}).get("enabled", True),
            data_encryption_enabled=security_config.get("data_protection", {}).get(
                "encryption_at_rest", True
            ),
            allowed_ips=security_config.get("access_control", {}).get("allowed_ips", []),
            rate_limit_rpm=security_config.get("access_control", {})
            .get("rate_limiting", {})
            .get("requests_per_minute", 60),
        )

    def get_trading_config(self, environment: Optional[Environment] = None) -> TradingConfig:
        """取引設定取得"""
        config = self.load_config(environment)
        trading_config = config.get("daytrading", {})
        risk_config = trading_config.get("risk_management", {})

        return TradingConfig(
            max_position_percent=risk_config.get("max_position_per_stock", 5.0),
            stop_loss_percent=risk_config.get("stop_loss_percentage", 2.0),
            profit_taking_percent=risk_config.get("profit_taking_percentage", 3.0),
            max_daily_trades=risk_config.get("max_daily_trades", 10),
            paper_trading=trading_config.get("paper_trading", False),
        )

    def save_config(self, config: Dict[str, Any], environment: Optional[Environment] = None):
        """設定ファイル保存"""
        env = environment or self.current_environment
        config_file = self.config_dir / f"{env.value}.json"

        try:
            # バックアップ作成
            if config_file.exists():
                backup_file = config_file.with_suffix(".json.backup")
                config_file.replace(backup_file)

            # 設定保存
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # キャッシュ更新
            self.config_cache[env.value] = copy.deepcopy(config)

            self.logger.info(f"設定ファイル保存完了: {config_file}")

        except Exception as e:
            raise ConfigurationError(f"設定ファイル保存エラー: {e}", str(config_file), env.value)

    def clear_cache(self):
        """設定キャッシュクリア"""
        self.config_cache.clear()
        self.logger.info("設定キャッシュクリア完了")

    def get_environment_info(self) -> Dict[str, Any]:
        """環境情報取得"""
        return {
            "current_environment": self.current_environment.value,
            "config_directory": str(self.config_dir.absolute()),
            "available_configs": [
                f.stem
                for f in self.config_dir.glob("*.json")
                if f.stem not in ["settings"]  # デフォルト設定を除外
            ],
            "environment_variables": {
                key: value for key, value in os.environ.items() if key.startswith("DAYTRADE_")
            },
            "cache_status": {
                "cached_environments": list(self.config_cache.keys()),
                "cache_size": len(self.config_cache),
            },
        }


# グローバルインスタンス
_global_config_manager = None


def get_environment_config_manager() -> EnvironmentConfigManager:
    """グローバル環境設定管理取得"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = EnvironmentConfigManager()
    return _global_config_manager


def get_env_config(key_path: str, default: Any = None, environment: Optional[str] = None) -> Any:
    """環境設定値取得（便利関数）"""
    env = Environment(environment) if environment else None
    return get_environment_config_manager().get_config(key_path, default, env)


def get_current_environment() -> str:
    """現在の環境取得"""
    return get_environment_config_manager().current_environment.value


if __name__ == "__main__":
    # テスト実行
    print("Environment Configuration Manager Test")
    print("=" * 50)

    try:
        # 環境設定管理テスト
        config_mgr = get_environment_config_manager()

        # 環境情報表示
        env_info = config_mgr.get_environment_info()
        print(f"Current Environment: {env_info['current_environment']}")
        print(f"Config Directory: {env_info['config_directory']}")
        print(f"Available Configs: {env_info['available_configs']}")

        # 設定読み込みテスト
        config = config_mgr.load_config()
        print("\nConfiguration loaded successfully")
        print(f"Watchlist symbols: {len(config.get('watchlist', {}).get('symbols', []))}")

        # 構造化設定取得テスト
        system_config = config_mgr.get_system_config()
        print("\nSystem Config:")
        print(f"  Performance Target: {system_config.performance_target_seconds}s")
        print(f"  Memory Limit: {system_config.memory_limit_mb}MB")
        print(f"  Logging Level: {system_config.logging_level}")

        # セキュリティ設定取得
        security_config = config_mgr.get_security_config()
        print("\nSecurity Config:")
        print(f"  API Keys Encrypted: {security_config.api_keys_encrypted}")
        print(f"  Access Control: {security_config.access_control_enabled}")
        print(f"  Data Encryption: {security_config.data_encryption_enabled}")

        # 取引設定取得
        trading_config = config_mgr.get_trading_config()
        print("\nTrading Config:")
        print(f"  Max Position: {trading_config.max_position_percent}%")
        print(f"  Stop Loss: {trading_config.stop_loss_percent}%")
        print(f"  Paper Trading: {trading_config.paper_trading}")

        # 個別設定値取得テスト
        target_seconds = get_env_config("system.performance.ml_analysis_target_seconds", 5.0)
        print(f"\nML Analysis Target: {target_seconds}s")

        print("\nEnvironment Configuration Manager test completed successfully!")

    except Exception as e:
        print(f"Environment Configuration Manager test failed: {e}")
        import traceback

        traceback.print_exc()
