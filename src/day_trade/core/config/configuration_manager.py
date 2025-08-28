"""
設定管理システム

環境別設定、シークレット管理、動的設定更新を提供する
包括的な設定管理システム。
"""

import json
import os
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import threading
import logging
from functools import wraps


# ================================
# 設定関連の列挙型
# ================================

class Environment(Enum):
    """実行環境"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigFormat(Enum):
    """設定ファイル形式"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"


class ConfigSensitivity(Enum):
    """設定の機密度"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


# ================================
# 設定データクラス
# ================================

@dataclass
class ConfigItem:
    """設定アイテム"""
    key: str
    value: Any
    sensitivity: ConfigSensitivity = ConfigSensitivity.PUBLIC
    description: str = ""
    default_value: Any = None
    validation_func: Optional[Callable[[Any], bool]] = None
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "default"
    
    def validate(self) -> bool:
        """値の検証"""
        if self.validation_func:
            return self.validation_func(self.value)
        return True
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """辞書形式への変換"""
        result = {
            'key': self.key,
            'sensitivity': self.sensitivity.value,
            'description': self.description,
            'last_updated': self.last_updated.isoformat(),
            'source': self.source
        }
        
        # 機密情報の処理
        if include_sensitive or self.sensitivity in [ConfigSensitivity.PUBLIC, ConfigSensitivity.INTERNAL]:
            result['value'] = self.value
        else:
            result['value'] = '***REDACTED***'
        
        return result


@dataclass
class EnvironmentConfig:
    """環境別設定"""
    environment: Environment
    config_items: Dict[str, ConfigItem] = field(default_factory=dict)
    base_path: Optional[Path] = None
    last_loaded: datetime = field(default_factory=datetime.now)
    
    def get_item(self, key: str) -> Optional[ConfigItem]:
        """設定アイテム取得"""
        return self.config_items.get(key)
    
    def set_item(self, key: str, item: ConfigItem):
        """設定アイテム設定"""
        self.config_items[key] = item
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """設定値取得"""
        item = self.get_item(key)
        if item and item.validate():
            return item.value
        return default
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """辞書形式への変換"""
        return {
            'environment': self.environment.value,
            'last_loaded': self.last_loaded.isoformat(),
            'config_items': {
                key: item.to_dict(include_sensitive) 
                for key, item in self.config_items.items()
            }
        }


# ================================
# 設定プロバイダー
# ================================

class ConfigProvider(ABC):
    """設定プロバイダー基底クラス"""
    
    @abstractmethod
    def load_config(self, environment: Environment) -> Dict[str, Any]:
        """設定読み込み"""
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any], environment: Environment) -> bool:
        """設定保存"""
        pass
    
    def get_name(self) -> str:
        """プロバイダー名取得"""
        return self.__class__.__name__


class FileConfigProvider(ConfigProvider):
    """ファイル設定プロバイダー"""
    
    def __init__(self, base_path: Path, format: ConfigFormat = ConfigFormat.JSON):
        self.base_path = Path(base_path)
        self.format = format
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_config_file_path(self, environment: Environment) -> Path:
        """設定ファイルパス取得"""
        if self.format == ConfigFormat.JSON:
            return self.base_path / f"{environment.value}.json"
        elif self.format == ConfigFormat.YAML:
            return self.base_path / f"{environment.value}.yaml"
        else:
            return self.base_path / f"{environment.value}.env"
    
    def load_config(self, environment: Environment) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        config_file = self._get_config_file_path(environment)
        
        if not config_file.exists():
            logging.warning(f"設定ファイルが存在しません: {config_file}")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if self.format == ConfigFormat.JSON:
                    return json.load(f)
                elif self.format == ConfigFormat.YAML:
                    return yaml.safe_load(f) or {}
                else:
                    # ENV形式の簡単な実装
                    config = {}
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                config[key.strip()] = value.strip()
                    return config
        
        except Exception as e:
            logging.error(f"設定ファイル読み込みエラー: {e}")
            return {}
    
    def save_config(self, config: Dict[str, Any], environment: Environment) -> bool:
        """設定ファイル保存"""
        config_file = self._get_config_file_path(environment)
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                if self.format == ConfigFormat.JSON:
                    json.dump(config, f, indent=2, ensure_ascii=False, default=str)
                elif self.format == ConfigFormat.YAML:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    # ENV形式
                    for key, value in config.items():
                        f.write(f"{key}={value}\n")
            
            logging.info(f"設定ファイル保存成功: {config_file}")
            return True
        
        except Exception as e:
            logging.error(f"設定ファイル保存エラー: {e}")
            return False


class EnvironmentVariableProvider(ConfigProvider):
    """環境変数プロバイダー"""
    
    def __init__(self, prefix: str = "DAY_TRADE_"):
        self.prefix = prefix
    
    def load_config(self, environment: Environment) -> Dict[str, Any]:
        """環境変数から設定読み込み"""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # プレフィックスを除去してキー名を取得
                config_key = key[len(self.prefix):].lower()
                
                # 型変換を試行
                config[config_key] = self._convert_value(value)
        
        return config
    
    def save_config(self, config: Dict[str, Any], environment: Environment) -> bool:
        """環境変数への設定保存（実際には設定しない）"""
        # 環境変数は外部で設定されるため、ここでは何もしない
        logging.info("環境変数プロバイダーでは設定保存をサポートしていません")
        return False
    
    def _convert_value(self, value: str) -> Any:
        """値の型変換"""
        # bool変換
        if value.lower() in ['true', '1', 'yes', 'on']:
            return True
        elif value.lower() in ['false', '0', 'no', 'off']:
            return False
        
        # int変換を試行
        try:
            return int(value)
        except ValueError:
            pass
        
        # float変換を試行
        try:
            return float(value)
        except ValueError:
            pass
        
        # 文字列として返す
        return value


# ================================
# 設定管理マネージャー
# ================================

class ConfigurationManager:
    """統合設定管理マネージャー"""
    
    def __init__(self, environment: Environment = None):
        self.current_environment = environment or self._detect_environment()
        self.environments: Dict[Environment, EnvironmentConfig] = {}
        self.providers: List[ConfigProvider] = []
        self.watchers: List[Callable[[str, Any, Any], None]] = []
        self._lock = threading.RLock()
        
        # デフォルトプロバイダーを追加
        self._setup_default_providers()
        self._load_all_configs()
    
    def _detect_environment(self) -> Environment:
        """環境自動検出"""
        env_name = os.environ.get('DAY_TRADE_ENV', 'development').lower()
        
        for env in Environment:
            if env.value == env_name:
                return env
        
        logging.warning(f"不明な環境名: {env_name}、developmentを使用")
        return Environment.DEVELOPMENT
    
    def _setup_default_providers(self):
        """デフォルトプロバイダー設定"""
        # 設定ファイルプロバイダー
        config_dir = Path("config")
        self.add_provider(FileConfigProvider(config_dir, ConfigFormat.JSON))
        
        # 環境変数プロバイダー
        self.add_provider(EnvironmentVariableProvider())
    
    def add_provider(self, provider: ConfigProvider):
        """設定プロバイダー追加"""
        self.providers.append(provider)
        logging.info(f"設定プロバイダー追加: {provider.get_name()}")
    
    def add_watcher(self, watcher: Callable[[str, Any, Any], None]):
        """設定変更監視関数追加"""
        self.watchers.append(watcher)
    
    def _load_all_configs(self):
        """全設定読み込み"""
        for env in Environment:
            self.environments[env] = EnvironmentConfig(environment=env)
            self._load_environment_config(env)
    
    def _load_environment_config(self, environment: Environment):
        """環境別設定読み込み"""
        env_config = self.environments[environment]
        
        # 各プロバイダーから設定を読み込み
        for provider in self.providers:
            try:
                config_data = provider.load_config(environment)
                self._merge_config_data(env_config, config_data, provider.get_name())
            except Exception as e:
                logging.error(f"設定読み込みエラー ({provider.get_name()}): {e}")
        
        env_config.last_loaded = datetime.now()
        logging.info(f"環境設定読み込み完了: {environment.value}")
    
    def _merge_config_data(self, env_config: EnvironmentConfig, config_data: Dict[str, Any], source: str):
        """設定データマージ"""
        for key, value in config_data.items():
            # 機密度判定
            sensitivity = self._determine_sensitivity(key, value)
            
            # 検証関数設定
            validation_func = self._get_validation_function(key)
            
            config_item = ConfigItem(
                key=key,
                value=value,
                sensitivity=sensitivity,
                validation_func=validation_func,
                source=source
            )
            
            # 既存の値があれば監視者に通知
            old_item = env_config.get_item(key)
            if old_item and old_item.value != value:
                self._notify_watchers(key, old_item.value, value)
            
            env_config.set_item(key, config_item)
    
    def _determine_sensitivity(self, key: str, value: Any) -> ConfigSensitivity:
        """設定の機密度判定"""
        key_lower = key.lower()
        
        # シークレット系
        if any(secret_word in key_lower for secret_word in ['password', 'secret', 'key', 'token', 'credential']):
            return ConfigSensitivity.SECRET
        
        # 機密情報系
        if any(conf_word in key_lower for conf_word in ['database', 'db_', 'api_', 'private']):
            return ConfigSensitivity.CONFIDENTIAL
        
        # 内部情報系
        if any(internal_word in key_lower for internal_word in ['host', 'port', 'url', 'path']):
            return ConfigSensitivity.INTERNAL
        
        # パブリック
        return ConfigSensitivity.PUBLIC
    
    def _get_validation_function(self, key: str) -> Optional[Callable[[Any], bool]]:
        """設定キーに基づく検証関数取得"""
        key_lower = key.lower()
        
        # ポート番号
        if 'port' in key_lower:
            return lambda x: isinstance(x, int) and 1 <= x <= 65535
        
        # URL
        if 'url' in key_lower:
            return lambda x: isinstance(x, str) and (x.startswith('http://') or x.startswith('https://'))
        
        # パス
        if 'path' in key_lower or 'dir' in key_lower:
            return lambda x: isinstance(x, str) and len(x) > 0
        
        return None
    
    def _notify_watchers(self, key: str, old_value: Any, new_value: Any):
        """設定変更監視者に通知"""
        for watcher in self.watchers:
            try:
                watcher(key, old_value, new_value)
            except Exception as e:
                logging.error(f"設定変更監視関数エラー: {e}")
    
    def get_config(self, key: str, default: Any = None, environment: Environment = None) -> Any:
        """設定値取得"""
        env = environment or self.current_environment
        
        with self._lock:
            env_config = self.environments.get(env)
            if env_config:
                return env_config.get_value(key, default)
            return default
    
    def set_config(self, key: str, value: Any, environment: Environment = None, persist: bool = True):
        """設定値設定"""
        env = environment or self.current_environment
        
        with self._lock:
            env_config = self.environments.get(env)
            if not env_config:
                env_config = EnvironmentConfig(environment=env)
                self.environments[env] = env_config
            
            # 既存の値を取得
            old_item = env_config.get_item(key)
            old_value = old_item.value if old_item else None
            
            # 機密度と検証関数を設定
            sensitivity = self._determine_sensitivity(key, value)
            validation_func = self._get_validation_function(key)
            
            config_item = ConfigItem(
                key=key,
                value=value,
                sensitivity=sensitivity,
                validation_func=validation_func,
                source="manual"
            )
            
            # 検証
            if not config_item.validate():
                raise ValueError(f"設定値の検証に失敗しました: {key} = {value}")
            
            env_config.set_item(key, config_item)
            
            # 監視者に通知
            if old_value != value:
                self._notify_watchers(key, old_value, value)
            
            # 永続化
            if persist:
                self._persist_config(key, value, env)
            
            logging.info(f"設定値設定: {key} (環境: {env.value})")
    
    def _persist_config(self, key: str, value: Any, environment: Environment):
        """設定永続化"""
        # ファイルプロバイダーに保存
        file_providers = [p for p in self.providers if isinstance(p, FileConfigProvider)]
        
        for provider in file_providers:
            try:
                # 現在の設定を取得
                current_config = provider.load_config(environment)
                current_config[key] = value
                
                # 保存
                provider.save_config(current_config, environment)
                logging.info(f"設定永続化成功: {key} ({provider.get_name()})")
                break
            except Exception as e:
                logging.error(f"設定永続化エラー ({provider.get_name()}): {e}")
    
    def reload_config(self, environment: Environment = None):
        """設定再読み込み"""
        env = environment or self.current_environment
        
        with self._lock:
            self._load_environment_config(env)
            logging.info(f"設定再読み込み完了: {env.value}")
    
    def get_all_configs(self, environment: Environment = None, include_sensitive: bool = False) -> Dict[str, Any]:
        """全設定取得"""
        env = environment or self.current_environment
        
        with self._lock:
            env_config = self.environments.get(env)
            if env_config:
                return env_config.to_dict(include_sensitive)
            return {}
    
    def get_config_summary(self) -> Dict[str, Any]:
        """設定サマリー取得"""
        with self._lock:
            summary = {
                'current_environment': self.current_environment.value,
                'providers': [p.get_name() for p in self.providers],
                'environments': {}
            }
            
            for env, env_config in self.environments.items():
                config_count = len(env_config.config_items)
                sensitivity_counts = {}
                
                for item in env_config.config_items.values():
                    sens = item.sensitivity.value
                    sensitivity_counts[sens] = sensitivity_counts.get(sens, 0) + 1
                
                summary['environments'][env.value] = {
                    'config_count': config_count,
                    'last_loaded': env_config.last_loaded.isoformat(),
                    'sensitivity_distribution': sensitivity_counts
                }
            
            return summary
    
    def validate_all_configs(self, environment: Environment = None) -> Dict[str, List[str]]:
        """全設定検証"""
        env = environment or self.current_environment
        validation_errors = {}
        
        with self._lock:
            env_config = self.environments.get(env)
            if env_config:
                for key, item in env_config.config_items.items():
                    if not item.validate():
                        if key not in validation_errors:
                            validation_errors[key] = []
                        validation_errors[key].append(f"検証に失敗しました: {item.value}")
        
        return validation_errors


# ================================
# 設定デコレーター
# ================================

def config_required(config_key: str, default: Any = None):
    """設定値必須デコレーター"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # グローバル設定マネージャーから値を取得
            config_value = get_global_config_manager().get_config(config_key, default)
            
            if config_value is None:
                raise ValueError(f"必須設定が見つかりません: {config_key}")
            
            # 関数の引数に設定値を追加
            kwargs[config_key] = config_value
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def environment_specific(environments: List[Environment]):
    """環境限定デコレーター"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_env = get_global_config_manager().current_environment
            
            if current_env not in environments:
                raise RuntimeError(f"この関数は {[e.value for e in environments]} 環境でのみ実行可能です。現在の環境: {current_env.value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# ================================
# グローバル設定マネージャー
# ================================

_global_config_manager: Optional[ConfigurationManager] = None

def get_global_config_manager() -> ConfigurationManager:
    """グローバル設定マネージャー取得"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigurationManager()
    return _global_config_manager

def init_global_config_manager(environment: Environment = None) -> ConfigurationManager:
    """グローバル設定マネージャー初期化"""
    global _global_config_manager
    _global_config_manager = ConfigurationManager(environment)
    return _global_config_manager


# ================================
# 便利関数
# ================================

def get_config(key: str, default: Any = None) -> Any:
    """設定値取得（便利関数）"""
    return get_global_config_manager().get_config(key, default)

def set_config(key: str, value: Any, persist: bool = True):
    """設定値設定（便利関数）"""
    get_global_config_manager().set_config(key, value, persist=persist)

def reload_config():
    """設定再読み込み（便利関数）"""
    get_global_config_manager().reload_config()


# エクスポート
__all__ = [
    # 列挙型
    'Environment', 'ConfigFormat', 'ConfigSensitivity',
    
    # データクラス
    'ConfigItem', 'EnvironmentConfig',
    
    # プロバイダー
    'ConfigProvider', 'FileConfigProvider', 'EnvironmentVariableProvider',
    
    # メインクラス
    'ConfigurationManager',
    
    # デコレーター
    'config_required', 'environment_specific',
    
    # グローバル関数
    'get_global_config_manager', 'init_global_config_manager',
    'get_config', 'set_config', 'reload_config'
]