#!/usr/bin/env python3
"""
統合設定管理システム
Issue #870統合: 拡張予測システムと既存システムの設定統合管理

設定ファイルの読み込み、統合、検証機能を提供
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 設定関連インポート
try:
    from enhanced_prediction_core import PredictionConfig, PredictionMode
    ENHANCED_CONFIG_AVAILABLE = True
except ImportError:
    ENHANCED_CONFIG_AVAILABLE = False

try:
    from prediction_adapter import AdapterConfig, AdapterMode
    ADAPTER_CONFIG_AVAILABLE = True
except ImportError:
    ADAPTER_CONFIG_AVAILABLE = False


@dataclass
class ConfigurationError(Exception):
    """設定エラー"""
    message: str
    config_file: Optional[str] = None
    config_section: Optional[str] = None


@dataclass
class ConfigValidationResult:
    """設定検証結果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_sections: List[str] = field(default_factory=list)
    deprecated_settings: List[str] = field(default_factory=list)


class ConfigManager:
    """統合設定管理システム"""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "config"
        self.logger = logging.getLogger(__name__)

        # 設定キャッシュ
        self.config_cache = {}
        self.config_timestamps = {}

        # デフォルト設定ファイル
        self.default_configs = {
            "main": "settings.json",
            "enhanced_prediction": "enhanced_prediction.yaml",
            "system_integration": "system_integration.yaml",
            "ml": "ml.json",
            "symbols": "symbols.yaml",
            "optimization": "optimization_config.yaml"
        }

    def load_config(self, config_name: str,
                   reload: bool = False) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        config_file = self._get_config_file_path(config_name)

        # キャッシュチェック
        if not reload and config_name in self.config_cache:
            cached_time = self.config_timestamps.get(config_name, datetime.min)
            file_time = datetime.fromtimestamp(config_file.stat().st_mtime)

            if cached_time >= file_time:
                return self.config_cache[config_name]

        # ファイル読み込み
        try:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config = self._load_yaml_config(config_file)
            elif config_file.suffix.lower() == '.json':
                config = self._load_json_config(config_file)
            else:
                raise ConfigurationError(
                    f"対応していない設定ファイル形式: {config_file.suffix}",
                    config_file=str(config_file)
                )

            # キャッシュ更新
            self.config_cache[config_name] = config
            self.config_timestamps[config_name] = datetime.now()

            self.logger.info(f"設定ファイル読み込み完了: {config_file}")
            return config

        except Exception as e:
            self.logger.error(f"設定ファイル読み込み失敗: {config_file}, エラー: {e}")
            raise ConfigurationError(
                f"設定ファイル読み込み失敗: {e}",
                config_file=str(config_file)
            )

    def _get_config_file_path(self, config_name: str) -> Path:
        """設定ファイルパス取得"""
        if config_name in self.default_configs:
            filename = self.default_configs[config_name]
        else:
            filename = config_name

        config_file = self.config_dir / filename

        if not config_file.exists():
            raise ConfigurationError(
                f"設定ファイルが見つかりません: {config_file}",
                config_file=str(config_file)
            )

        return config_file

    def _load_yaml_config(self, config_file: Path) -> Dict[str, Any]:
        """YAML設定ファイル読み込み"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _load_json_config(self, config_file: Path) -> Dict[str, Any]:
        """JSON設定ファイル読み込み"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_integrated_config(self) -> Dict[str, Any]:
        """統合設定取得"""
        integrated_config = {}

        # 基本設定読み込み
        try:
            main_config = self.load_config("main")
            integrated_config.update(main_config)
        except ConfigurationError:
            self.logger.warning("メイン設定ファイルが見つかりません")

        # 拡張予測システム設定
        try:
            enhanced_config = self.load_config("enhanced_prediction")
            integrated_config["enhanced_prediction"] = enhanced_config
        except ConfigurationError:
            self.logger.warning("拡張予測設定ファイルが見つかりません")

        # システム統合設定
        try:
            integration_config = self.load_config("system_integration")
            integrated_config["system_integration"] = integration_config
        except ConfigurationError:
            self.logger.warning("システム統合設定ファイルが見つかりません")

        # ML設定
        try:
            ml_config = self.load_config("ml")
            integrated_config["ml"] = ml_config
        except ConfigurationError:
            self.logger.warning("ML設定ファイルが見つかりません")

        return integrated_config

    def create_enhanced_prediction_config(self,
                                        override_config: Optional[Dict[str, Any]] = None) -> Optional['PredictionConfig']:
        """拡張予測システム設定作成"""
        if not ENHANCED_CONFIG_AVAILABLE:
            self.logger.warning("拡張予測システム設定クラス未対応")
            return None

        try:
            # 設定ファイル読み込み
            config_data = self.load_config("enhanced_prediction")
            enhanced_section = config_data.get("enhanced_prediction", {})

            # オーバーライド適用
            if override_config:
                enhanced_section = self._merge_configs(enhanced_section, override_config)

            # PredictionConfig作成
            prediction_config = PredictionConfig(
                mode=PredictionMode(enhanced_section.get("system", {}).get("mode", "auto")),
                feature_selection_enabled=enhanced_section.get("feature_selection", {}).get("enabled", True),
                ensemble_enabled=enhanced_section.get("ensemble", {}).get("enabled", True),
                hybrid_timeseries_enabled=enhanced_section.get("hybrid_timeseries", {}).get("enabled", True),
                meta_learning_enabled=enhanced_section.get("meta_learning", {}).get("enabled", True),
                fallback_to_legacy=enhanced_section.get("system", {}).get("fallback_to_legacy", True),
                max_features=enhanced_section.get("feature_selection", {}).get("max_features", 50),
                cv_folds=enhanced_section.get("ensemble", {}).get("cv_folds", 5),
                sequence_length=enhanced_section.get("hybrid_timeseries", {}).get("lstm", {}).get("sequence_length", 20),
                lstm_units=enhanced_section.get("hybrid_timeseries", {}).get("lstm", {}).get("hidden_units", 50),
                repository_size=enhanced_section.get("meta_learning", {}).get("repository_size", 100)
            )

            return prediction_config

        except Exception as e:
            self.logger.error(f"拡張予測システム設定作成失敗: {e}")
            return None

    def create_adapter_config(self,
                            override_config: Optional[Dict[str, Any]] = None) -> Optional['AdapterConfig']:
        """アダプター設定作成"""
        if not ADAPTER_CONFIG_AVAILABLE:
            self.logger.warning("アダプター設定クラス未対応")
            return None

        try:
            # 設定ファイル読み込み
            config_data = self.load_config("enhanced_prediction")
            adapter_section = config_data.get("adapter", {})

            # オーバーライド適用
            if override_config:
                adapter_section = self._merge_configs(adapter_section, override_config)

            # AdapterConfig作成
            adapter_config = AdapterConfig(
                mode=AdapterMode(adapter_section.get("mode", "smart_fallback")),
                rollout_percentage=adapter_section.get("gradual_rollout", {}).get("initial_percentage", 0.1),
                ab_test_split=adapter_section.get("ab_test", {}).get("split_ratio", 0.5),
                performance_threshold=adapter_section.get("performance_comparison", {}).get("performance_threshold", 0.8),
                fallback_threshold=adapter_section.get("performance_comparison", {}).get("fallback_threshold", 0.5),
                comparison_window=adapter_section.get("performance_comparison", {}).get("comparison_window", 100),
                enable_logging=adapter_section.get("monitoring", {}).get("enable_logging", True),
                enable_metrics=adapter_section.get("performance_comparison", {}).get("enable_metrics", True)
            )

            return adapter_config

        except Exception as e:
            self.logger.error(f"アダプター設定作成失敗: {e}")
            return None

    def _merge_configs(self, base_config: Dict[str, Any],
                      override_config: Dict[str, Any]) -> Dict[str, Any]:
        """設定マージ"""
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def validate_config(self, config_name: str) -> ConfigValidationResult:
        """設定検証"""
        result = ConfigValidationResult(is_valid=True)

        try:
            config = self.load_config(config_name)

            # 設定タイプ別検証
            if config_name == "enhanced_prediction":
                result = self._validate_enhanced_prediction_config(config)
            elif config_name == "system_integration":
                result = self._validate_integration_config(config)
            elif config_name == "main":
                result = self._validate_main_config(config)
            else:
                result = self._validate_generic_config(config)

        except ConfigurationError as e:
            result.is_valid = False
            result.errors.append(str(e))

        return result

    def _validate_enhanced_prediction_config(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """拡張予測設定検証"""
        result = ConfigValidationResult(is_valid=True)

        # 必須セクション確認
        required_sections = ["enhanced_prediction", "adapter"]
        for section in required_sections:
            if section not in config:
                result.missing_sections.append(section)
                result.errors.append(f"必須セクション未定義: {section}")

        # 拡張予測設定詳細検証
        if "enhanced_prediction" in config:
            enhanced_config = config["enhanced_prediction"]

            # システム設定検証
            system_config = enhanced_config.get("system", {})
            mode = system_config.get("mode", "auto")
            valid_modes = ["auto", "enhanced", "legacy", "hybrid"]
            if mode not in valid_modes:
                result.errors.append(f"無効なシステムモード: {mode}")

            # 特徴量選択設定検証
            feature_config = enhanced_config.get("feature_selection", {})
            max_features = feature_config.get("max_features", 50)
            if not isinstance(max_features, int) or max_features <= 0:
                result.errors.append("max_featuresは正の整数である必要があります")

        # エラーがある場合は無効
        if result.errors:
            result.is_valid = False

        return result

    def _validate_integration_config(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """統合設定検証"""
        result = ConfigValidationResult(is_valid=True)

        # 必須セクション確認
        required_sections = ["system_integration"]
        for section in required_sections:
            if section not in config:
                result.missing_sections.append(section)
                result.errors.append(f"必須セクション未定義: {section}")

        # 統合モード検証
        if "system_integration" in config:
            integration_config = config["system_integration"]
            mode = integration_config.get("integration_mode", "hybrid")
            valid_modes = ["legacy", "enhanced", "hybrid", "auto"]
            if mode not in valid_modes:
                result.errors.append(f"無効な統合モード: {mode}")

        if result.errors:
            result.is_valid = False

        return result

    def _validate_main_config(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """メイン設定検証"""
        result = ConfigValidationResult(is_valid=True)

        # 基本検証のみ実行
        if not isinstance(config, dict):
            result.errors.append("設定はディクショナリ形式である必要があります")
            result.is_valid = False

        return result

    def _validate_generic_config(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """汎用設定検証"""
        result = ConfigValidationResult(is_valid=True)

        # 基本形式確認
        if not isinstance(config, dict):
            result.errors.append("設定はディクショナリ形式である必要があります")
            result.is_valid = False

        return result

    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """設定ファイル保存"""
        try:
            config_file = self._get_config_file_path(config_name)

            # バックアップ作成
            backup_file = config_file.with_suffix(f"{config_file.suffix}.backup")
            if config_file.exists():
                backup_file.write_bytes(config_file.read_bytes())

            # 設定保存
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(config_data, f, default_flow_style=False, allow_unicode=True)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)

            # キャッシュクリア
            if config_name in self.config_cache:
                del self.config_cache[config_name]
            if config_name in self.config_timestamps:
                del self.config_timestamps[config_name]

            self.logger.info(f"設定ファイル保存完了: {config_file}")
            return True

        except Exception as e:
            self.logger.error(f"設定ファイル保存失敗: {e}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """設定サマリー取得"""
        summary = {
            "timestamp": datetime.now(),
            "config_directory": str(self.config_dir),
            "available_configs": {},
            "validation_results": {}
        }

        # 利用可能設定ファイル確認
        for config_name, filename in self.default_configs.items():
            config_file = self.config_dir / filename
            summary["available_configs"][config_name] = {
                "filename": filename,
                "exists": config_file.exists(),
                "size_bytes": config_file.stat().st_size if config_file.exists() else 0,
                "modified": datetime.fromtimestamp(config_file.stat().st_mtime) if config_file.exists() else None
            }

            # 設定検証
            if config_file.exists():
                try:
                    validation_result = self.validate_config(config_name)
                    summary["validation_results"][config_name] = {
                        "is_valid": validation_result.is_valid,
                        "error_count": len(validation_result.errors),
                        "warning_count": len(validation_result.warnings)
                    }
                except Exception as e:
                    summary["validation_results"][config_name] = {
                        "is_valid": False,
                        "error": str(e)
                    }

        return summary


def create_config_manager(config_dir: Optional[Union[str, Path]] = None) -> ConfigManager:
    """設定管理システム作成"""
    return ConfigManager(config_dir)


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)

    # 設定管理システム作成
    config_manager = create_config_manager()

    try:
        # 設定サマリー表示
        summary = config_manager.get_config_summary()

        print("設定管理システムテスト")
        print("=" * 50)
        print(f"設定ディレクトリ: {summary['config_directory']}")
        print(f"タイムスタンプ: {summary['timestamp']}")

        print("\n利用可能設定ファイル:")
        for config_name, info in summary["available_configs"].items():
            status = "OK" if info["exists"] else "NG"
            print(f"  {config_name}: {status} ({info['filename']})")

        print("\n設定検証結果:")
        for config_name, result in summary["validation_results"].items():
            status = "OK" if result["is_valid"] else "NG"
            print(f"  {config_name}: {status}")
            if not result["is_valid"]:
                print(f"    エラー数: {result.get('error_count', 0)}")

        # 統合設定取得テスト
        try:
            integrated_config = config_manager.get_integrated_config()
            print(f"\n統合設定取得成功: {len(integrated_config)}セクション")
        except Exception as e:
            print(f"\n統合設定取得失敗: {e}")

        # 拡張予測設定作成テスト
        try:
            enhanced_config = config_manager.create_enhanced_prediction_config()
            if enhanced_config:
                print(f"拡張予測設定作成成功: {enhanced_config.mode}")
            else:
                print("拡張予測設定作成失敗")
        except Exception as e:
            print(f"拡張予測設定作成エラー: {e}")

        print("\n設定管理システムテスト完了")

    except Exception as e:
        print(f"テストエラー: {e}")
        exit(1)