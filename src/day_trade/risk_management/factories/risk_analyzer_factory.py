#!/usr/bin/env python3
"""
Risk Analyzer Factory
リスク分析器ファクトリー

動的リスク分析器生成とプラグインシステムの基盤
"""

import importlib
import inspect
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from ..exceptions.risk_exceptions import ConfigurationError, ValidationError
from ..interfaces.risk_interfaces import IRiskAnalyzer


class AnalyzerType(Enum):
    """分析器タイプ"""

    GENERATIVE_AI = "generative_ai"
    FRAUD_DETECTION = "fraud_detection"
    RULE_BASED = "rule_based"
    MARKET_ANALYSIS = "market_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PLUGIN = "plugin"
    ENSEMBLE = "ensemble"


class RiskAnalyzerFactory:
    """リスク分析器ファクトリー"""

    def __init__(self):
        self._analyzer_registry: Dict[AnalyzerType, Type[IRiskAnalyzer]] = {}
        self._plugin_registry: Dict[str, Type[IRiskAnalyzer]] = {}
        self._config_schemas: Dict[AnalyzerType, Dict[str, Any]] = {}
        self._instance_cache: Dict[str, IRiskAnalyzer] = {}

        # 組み込み分析器を登録
        self._register_builtin_analyzers()

    def _register_builtin_analyzers(self):
        """組み込み分析器登録"""
        try:
            # 遅延インポートで循環依存を回避

            # 生成AI分析器
            self.register_analyzer(
                AnalyzerType.GENERATIVE_AI,
                "src.day_trade.risk.generative_ai_engine",
                "GenerativeAIRiskEngine",
                {
                    "openai_api_key": {"type": str, "required": False},
                    "anthropic_api_key": {"type": str, "required": False},
                    "temperature": {"type": float, "default": 0.3},
                    "max_tokens": {"type": int, "default": 1000},
                },
            )

            # 不正検知分析器
            self.register_analyzer(
                AnalyzerType.FRAUD_DETECTION,
                "src.day_trade.risk.fraud_detection_engine",
                "FraudDetectionEngine",
                {
                    "model_path": {"type": str, "required": False},
                    "threshold": {"type": float, "default": 0.5},
                },
            )

            # ルールベース分析器
            self.register_analyzer(
                AnalyzerType.RULE_BASED,
                "src.day_trade.risk_management.analyzers.rule_based_analyzer",
                "RuleBasedRiskAnalyzer",
                {
                    "rules_file": {"type": str, "required": False},
                    "strict_mode": {"type": bool, "default": False},
                },
            )

        except Exception as e:
            # 組み込み分析器の登録失敗は警告レベル
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register some builtin analyzers: {e}")

    def register_analyzer(
        self,
        analyzer_type: AnalyzerType,
        module_path: str,
        class_name: str,
        config_schema: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """分析器タイプ登録"""
        try:
            # 動的インポート
            module = importlib.import_module(module_path)
            analyzer_class = getattr(module, class_name)

            # インターフェース実装チェック
            if not issubclass(analyzer_class, IRiskAnalyzer):
                raise ConfigurationError(
                    f"Analyzer class {class_name} must implement IRiskAnalyzer interface",
                    config_key=f"analyzer.{analyzer_type.value}",
                )

            self._analyzer_registry[analyzer_type] = analyzer_class

            if config_schema:
                self._config_schemas[analyzer_type] = config_schema

            return True

        except Exception as e:
            raise ConfigurationError(
                f"Failed to register analyzer type {analyzer_type.value}",
                config_key=f"analyzer.{analyzer_type.value}",
                cause=e,
            ) from e

    def register_plugin_analyzer(
        self,
        plugin_name: str,
        analyzer_class: Type[IRiskAnalyzer],
        config_schema: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """プラグイン分析器登録"""
        try:
            if not issubclass(analyzer_class, IRiskAnalyzer):
                raise ConfigurationError(
                    f"Plugin analyzer {plugin_name} must implement IRiskAnalyzer interface",
                    config_key=f"plugin.{plugin_name}",
                )

            self._plugin_registry[plugin_name] = analyzer_class

            if config_schema:
                self._config_schemas[AnalyzerType.PLUGIN] = self._config_schemas.get(
                    AnalyzerType.PLUGIN, {}
                )
                self._config_schemas[AnalyzerType.PLUGIN][plugin_name] = config_schema

            return True

        except Exception as e:
            raise ConfigurationError(
                f"Failed to register plugin analyzer {plugin_name}",
                config_key=f"plugin.{plugin_name}",
                cause=e,
            ) from e

    def create_analyzer(
        self,
        analyzer_type: AnalyzerType,
        config: Optional[Dict[str, Any]] = None,
        plugin_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> IRiskAnalyzer:
        """分析器インスタンス作成"""

        # キャッシュキー生成
        cache_key = self._generate_cache_key(analyzer_type, plugin_name, config)

        # キャッシュからインスタンス取得
        if use_cache and cache_key in self._instance_cache:
            return self._instance_cache[cache_key]

        # 設定検証
        validated_config = self._validate_config(analyzer_type, config, plugin_name)

        # 分析器クラス取得
        analyzer_class = self._get_analyzer_class(analyzer_type, plugin_name)

        try:
            # インスタンス作成
            instance = self._create_instance(analyzer_class, validated_config)

            # キャッシュに保存
            if use_cache:
                self._instance_cache[cache_key] = instance

            return instance

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create analyzer instance of type {analyzer_type.value}",
                config_key=f"analyzer.{analyzer_type.value}",
                cause=e,
            ) from e

    def create_ensemble_analyzer(
        self,
        analyzer_configs: List[Dict[str, Any]],
        ensemble_weights: Optional[Dict[str, float]] = None,
        ensemble_strategy: str = "weighted_average",
    ) -> IRiskAnalyzer:
        """アンサンブル分析器作成"""

        analyzers = []

        for config in analyzer_configs:
            analyzer_type = AnalyzerType(config["type"])
            plugin_name = config.get("plugin_name")
            analyzer_config = config.get("config", {})

            analyzer = self.create_analyzer(
                analyzer_type, analyzer_config, plugin_name, use_cache=True
            )
            analyzers.append(analyzer)

        # アンサンブル分析器作成（簡略化実装）
        from ..analyzers.ensemble_analyzer import EnsembleRiskAnalyzer

        return EnsembleRiskAnalyzer(
            analyzers=analyzers, weights=ensemble_weights, strategy=ensemble_strategy
        )

    def get_available_analyzers(self) -> Dict[str, Dict[str, Any]]:
        """利用可能分析器一覧取得"""

        available = {}

        # 組み込み分析器
        for analyzer_type in self._analyzer_registry:
            analyzer_class = self._analyzer_registry[analyzer_type]
            config_schema = self._config_schemas.get(analyzer_type, {})

            try:
                # メタデータ取得（可能な場合）
                temp_instance = analyzer_class({})
                if hasattr(temp_instance, "get_metadata"):
                    metadata = temp_instance.get_metadata()
                    available[analyzer_type.value] = {
                        "type": "builtin",
                        "name": metadata.name,
                        "version": metadata.version,
                        "description": metadata.description,
                        "config_schema": config_schema,
                    }
                else:
                    available[analyzer_type.value] = {
                        "type": "builtin",
                        "name": analyzer_type.value,
                        "config_schema": config_schema,
                    }
            except Exception:
                # メタデータ取得に失敗した場合は基本情報のみ
                available[analyzer_type.value] = {
                    "type": "builtin",
                    "name": analyzer_type.value,
                    "config_schema": config_schema,
                }

        # プラグイン分析器
        for plugin_name, analyzer_class in self._plugin_registry.items():
            try:
                temp_instance = analyzer_class({})
                if hasattr(temp_instance, "get_metadata"):
                    metadata = temp_instance.get_metadata()
                    available[plugin_name] = {
                        "type": "plugin",
                        "name": metadata.name,
                        "version": metadata.version,
                        "description": metadata.description,
                        "config_schema": self._config_schemas.get(
                            AnalyzerType.PLUGIN, {}
                        ).get(plugin_name, {}),
                    }
                else:
                    available[plugin_name] = {
                        "type": "plugin",
                        "name": plugin_name,
                        "config_schema": self._config_schemas.get(
                            AnalyzerType.PLUGIN, {}
                        ).get(plugin_name, {}),
                    }
            except Exception:
                available[plugin_name] = {"type": "plugin", "name": plugin_name}

        return available

    def clear_cache(self):
        """インスタンスキャッシュクリア"""
        self._instance_cache.clear()

    def _validate_config(
        self,
        analyzer_type: AnalyzerType,
        config: Optional[Dict[str, Any]],
        plugin_name: Optional[str],
    ) -> Dict[str, Any]:
        """設定検証"""

        if config is None:
            config = {}

        # スキーマ取得
        if analyzer_type == AnalyzerType.PLUGIN and plugin_name:
            schema = self._config_schemas.get(AnalyzerType.PLUGIN, {}).get(
                plugin_name, {}
            )
        else:
            schema = self._config_schemas.get(analyzer_type, {})

        validated_config = {}

        # 必須フィールドチェック
        for field_name, field_schema in schema.items():
            if field_schema.get("required", False) and field_name not in config:
                raise ValidationError(
                    f"Required configuration field '{field_name}' is missing",
                    field_name=field_name,
                    validation_rules=["required"],
                )

            # デフォルト値適用
            if field_name not in config and "default" in field_schema:
                validated_config[field_name] = field_schema["default"]
            elif field_name in config:
                # 型チェック
                expected_type = field_schema.get("type")
                if expected_type and not isinstance(config[field_name], expected_type):
                    raise ValidationError(
                        f"Configuration field '{field_name}' must be of type {expected_type.__name__}",
                        field_name=field_name,
                        invalid_value=config[field_name],
                        validation_rules=[f"type:{expected_type.__name__}"],
                    )
                validated_config[field_name] = config[field_name]

        # 追加設定フィールドも含める（スキーマに定義されていない場合）
        for field_name, field_value in config.items():
            if field_name not in validated_config:
                validated_config[field_name] = field_value

        return validated_config

    def _get_analyzer_class(
        self, analyzer_type: AnalyzerType, plugin_name: Optional[str]
    ) -> Type[IRiskAnalyzer]:
        """分析器クラス取得"""

        if analyzer_type == AnalyzerType.PLUGIN:
            if not plugin_name:
                raise ConfigurationError(
                    "Plugin name is required for plugin analyzer type",
                    config_key="plugin_name",
                )

            if plugin_name not in self._plugin_registry:
                raise ConfigurationError(
                    f"Plugin analyzer '{plugin_name}' is not registered",
                    config_key=f"plugin.{plugin_name}",
                )

            return self._plugin_registry[plugin_name]

        else:
            if analyzer_type not in self._analyzer_registry:
                raise ConfigurationError(
                    f"Analyzer type '{analyzer_type.value}' is not registered",
                    config_key=f"analyzer.{analyzer_type.value}",
                )

            return self._analyzer_registry[analyzer_type]

    def _create_instance(
        self, analyzer_class: Type[IRiskAnalyzer], config: Dict[str, Any]
    ) -> IRiskAnalyzer:
        """インスタンス作成"""

        # コンストラクターシグネチャーチェック
        sig = inspect.signature(analyzer_class.__init__)

        # 設定パラメーターを適切に渡す
        if len(sig.parameters) > 1:  # self 以外のパラメーターがある場合
            param_names = list(sig.parameters.keys())[1:]  # self を除外

            if len(param_names) == 1 and param_names[0] in ["config", "configuration"]:
                # 設定オブジェクト全体を渡す
                return analyzer_class(config)
            else:
                # 個別パラメーターを渡す
                kwargs = {}
                for param_name in param_names:
                    if param_name in config:
                        kwargs[param_name] = config[param_name]
                return analyzer_class(**kwargs)
        else:
            # パラメーターなしのコンストラクター
            return analyzer_class()

    def _generate_cache_key(
        self,
        analyzer_type: AnalyzerType,
        plugin_name: Optional[str],
        config: Optional[Dict[str, Any]],
    ) -> str:
        """キャッシュキー生成"""
        import hashlib
        import json

        key_data = {
            "type": analyzer_type.value,
            "plugin": plugin_name,
            "config": config or {},
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


# グローバルファクトリーインスタンス
_global_analyzer_factory: Optional[RiskAnalyzerFactory] = None


def get_analyzer_factory() -> RiskAnalyzerFactory:
    """グローバル分析器ファクトリー取得"""
    global _global_analyzer_factory
    if _global_analyzer_factory is None:
        _global_analyzer_factory = RiskAnalyzerFactory()
    return _global_analyzer_factory


def create_analyzer(
    analyzer_type: AnalyzerType,
    config: Optional[Dict[str, Any]] = None,
    plugin_name: Optional[str] = None,
) -> IRiskAnalyzer:
    """分析器作成（便利関数）"""
    factory = get_analyzer_factory()
    return factory.create_analyzer(analyzer_type, config, plugin_name)


def register_analyzer_type(
    analyzer_type: AnalyzerType,
    module_path: str,
    class_name: str,
    config_schema: Optional[Dict[str, Any]] = None,
) -> bool:
    """分析器タイプ登録（便利関数）"""
    factory = get_analyzer_factory()
    return factory.register_analyzer(
        analyzer_type, module_path, class_name, config_schema
    )
