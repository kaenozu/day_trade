"""
Configuration manager module tests
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import yaml
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class MockConfigManager:
    """Mock configuration manager for testing"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config.json"
        self.config_data = {}
        self.defaults = {}
        self.watchers = []
        self.validation_rules = {}
        self.config_history = []
    
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file"""
        path = config_path or self.config_path
        
        try:
            if path.endswith('.json'):
                return self._load_json_config(path)
            elif path.endswith('.yaml') or path.endswith('.yml'):
                return self._load_yaml_config(path)
            else:
                raise ValueError(f"Unsupported config format: {path}")
        except FileNotFoundError:
            return self.defaults.copy()
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def _load_json_config(self, path: str) -> Dict[str, Any]:
        """Load JSON configuration"""
        # Mock JSON loading
        mock_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "trading_db",
                "user": "trader"
            },
            "api": {
                "base_url": "https://api.example.com",
                "timeout": 30,
                "retry_count": 3
            },
            "trading": {
                "max_position_size": 0.1,
                "risk_limit": 0.05,
                "stop_loss": 0.02
            },
            "logging": {
                "level": "INFO",
                "file": "trading.log",
                "max_size": 100
            }
        }
        self.config_data = mock_config
        return mock_config
    
    def _load_yaml_config(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration"""
        # Mock YAML loading
        mock_config = {
            "system": {
                "environment": "development",
                "debug": True,
                "workers": 4
            },
            "cache": {
                "type": "redis",
                "host": "localhost",
                "port": 6379,
                "ttl": 3600
            }
        }
        self.config_data = mock_config
        return mock_config
    
    def save_config(self, config_data: Dict[str, Any] = None, config_path: str = None) -> bool:
        """Save configuration to file"""
        data = config_data or self.config_data
        path = config_path or self.config_path
        
        try:
            # Store in history
            self.config_history.append({
                'timestamp': datetime.now(),
                'config': data.copy(),
                'path': path
            })
            
            self.config_data = data
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to save config: {e}")
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def set_value(self, key: str, value: Any) -> bool:
        """Set configuration value by dot notation key"""
        keys = key.split('.')
        config = self.config_data
        
        try:
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the final key
            config[keys[-1]] = value
            
            # Validate if rules exist
            if key in self.validation_rules:
                if not self._validate_value(key, value):
                    return False
            
            # Notify watchers
            self._notify_watchers(key, value)
            
            return True
        except Exception:
            return False
    
    def add_watcher(self, key: str, callback: callable) -> None:
        """Add configuration change watcher"""
        self.watchers.append({
            'key': key,
            'callback': callback
        })
    
    def remove_watcher(self, key: str, callback: callable) -> bool:
        """Remove configuration change watcher"""
        for i, watcher in enumerate(self.watchers):
            if watcher['key'] == key and watcher['callback'] == callback:
                del self.watchers[i]
                return True
        return False
    
    def _notify_watchers(self, key: str, value: Any) -> None:
        """Notify configuration change watchers"""
        for watcher in self.watchers:
            if watcher['key'] == key or key.startswith(watcher['key'] + '.'):
                try:
                    watcher['callback'](key, value)
                except Exception:
                    pass  # Don't let watcher errors break config updates
    
    def validate_config(self, config_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate configuration data"""
        data = config_data or self.config_data
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        required_fields = [
            'database.host',
            'database.port',
            'database.name',
            'api.base_url'
        ]
        
        for field in required_fields:
            if self.get_value(field) is None:
                validation_result['errors'].append(f"Required field missing: {field}")
                validation_result['valid'] = False
        
        # Check data types and ranges
        type_checks = {
            'database.port': (int, lambda x: 1 <= x <= 65535),
            'api.timeout': (int, lambda x: x > 0),
            'api.retry_count': (int, lambda x: x >= 0),
            'trading.max_position_size': (float, lambda x: 0 < x <= 1),
            'trading.risk_limit': (float, lambda x: 0 < x <= 1)
        }
        
        for field, (expected_type, validator) in type_checks.items():
            value = self.get_value(field)
            if value is not None:
                if not isinstance(value, expected_type):
                    validation_result['errors'].append(f"Invalid type for {field}: expected {expected_type.__name__}")
                    validation_result['valid'] = False
                elif not validator(value):
                    validation_result['errors'].append(f"Invalid value for {field}: {value}")
                    validation_result['valid'] = False
        
        return validation_result
    
    def add_validation_rule(self, key: str, validator: callable) -> None:
        """Add custom validation rule"""
        self.validation_rules[key] = validator
    
    def _validate_value(self, key: str, value: Any) -> bool:
        """Validate single value against rules"""
        if key in self.validation_rules:
            return self.validation_rules[key](value)
        return True
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config_data.get(section, {})
    
    def update_section(self, section: str, data: Dict[str, Any]) -> bool:
        """Update entire configuration section"""
        try:
            self.config_data[section] = data
            
            # Notify watchers for all keys in section
            for key, value in data.items():
                full_key = f"{section}.{key}"
                self._notify_watchers(full_key, value)
            
            return True
        except Exception:
            return False
    
    def merge_config(self, new_config: Dict[str, Any]) -> bool:
        """Merge new configuration with existing"""
        try:
            self._deep_merge(self.config_data, new_config)
            return True
        except Exception:
            return False
    
    def _deep_merge(self, base_dict: Dict[str, Any], merge_dict: Dict[str, Any]) -> None:
        """Recursively merge dictionaries"""
        for key, value in merge_dict.items():
            if (key in base_dict and isinstance(base_dict[key], dict) 
                and isinstance(value, dict)):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_config_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        return self.config_history.copy()
    
    def rollback_config(self, steps: int = 1) -> bool:
        """Rollback configuration to previous state"""
        if len(self.config_history) < steps:
            return False
        
        try:
            # Get the config from history
            target_config = self.config_history[-(steps + 1)]
            self.config_data = target_config['config'].copy()
            
            # Remove rolled back entries
            self.config_history = self.config_history[:-(steps)]
            
            return True
        except (IndexError, KeyError):
            return False


class MockEnvironmentConfig:
    """Mock environment-specific configuration manager"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.env_configs = {
            "development": {
                "database": {
                    "host": "localhost",
                    "debug": True
                },
                "api": {
                    "timeout": 10,
                    "debug": True
                }
            },
            "testing": {
                "database": {
                    "host": "test-db",
                    "debug": True
                },
                "api": {
                    "timeout": 5,
                    "debug": True
                }
            },
            "production": {
                "database": {
                    "host": "prod-db",
                    "debug": False
                },
                "api": {
                    "timeout": 30,
                    "debug": False
                }
            }
        }
    
    def get_environment(self) -> str:
        """Get current environment"""
        return self.environment
    
    def set_environment(self, environment: str) -> bool:
        """Set current environment"""
        if environment in self.env_configs:
            self.environment = environment
            return True
        return False
    
    def get_env_config(self, environment: str = None) -> Dict[str, Any]:
        """Get configuration for specific environment"""
        env = environment or self.environment
        return self.env_configs.get(env, {})
    
    def load_environment_overrides(self) -> Dict[str, Any]:
        """Load environment-specific configuration overrides"""
        base_config = {
            "database": {
                "host": "default-host",
                "port": 5432,
                "debug": False
            },
            "api": {
                "base_url": "https://api.example.com",
                "timeout": 30,
                "debug": False
            }
        }
        
        env_overrides = self.get_env_config()
        
        # Merge environment-specific overrides
        merged_config = base_config.copy()
        for section, values in env_overrides.items():
            if section in merged_config:
                merged_config[section].update(values)
            else:
                merged_config[section] = values
        
        return merged_config


class TestConfigManager:
    """Test configuration manager functionality"""
    
    def test_config_loading(self):
        config_manager = MockConfigManager()
        
        # Test JSON loading
        json_config = config_manager.load_config("config.json")
        assert "database" in json_config
        assert "api" in json_config
        assert json_config["database"]["host"] == "localhost"
        
        # Test YAML loading
        yaml_config = config_manager.load_config("config.yaml")
        assert "system" in yaml_config
        assert "cache" in yaml_config
        
        # Test unsupported format
        with pytest.raises(ValueError):
            config_manager.load_config("config.txt")
    
    def test_config_saving(self):
        config_manager = MockConfigManager()
        
        test_config = {
            "test_section": {
                "value1": "test",
                "value2": 42
            }
        }
        
        # Save config
        assert config_manager.save_config(test_config) == True
        assert config_manager.config_data == test_config
        
        # Check history
        history = config_manager.get_config_history()
        assert len(history) == 1
        assert history[0]['config'] == test_config
    
    def test_get_set_values(self):
        config_manager = MockConfigManager()
        config_manager.load_config("config.json")
        
        # Test getting values
        assert config_manager.get_value("database.host") == "localhost"
        assert config_manager.get_value("database.port") == 5432
        assert config_manager.get_value("nonexistent.key", "default") == "default"
        
        # Test setting values
        assert config_manager.set_value("database.host", "new-host") == True
        assert config_manager.get_value("database.host") == "new-host"
        
        # Test setting nested value
        assert config_manager.set_value("new.nested.key", "value") == True
        assert config_manager.get_value("new.nested.key") == "value"
    
    def test_config_watchers(self):
        config_manager = MockConfigManager()
        config_manager.load_config("config.json")
        
        # Track watcher calls
        watcher_calls = []
        
        def test_watcher(key: str, value: Any):
            watcher_calls.append((key, value))
        
        # Add watcher
        config_manager.add_watcher("database.host", test_watcher)
        
        # Change watched value
        config_manager.set_value("database.host", "watched-host")
        
        # Check watcher was called
        assert len(watcher_calls) == 1
        assert watcher_calls[0] == ("database.host", "watched-host")
        
        # Remove watcher
        assert config_manager.remove_watcher("database.host", test_watcher) == True
        
        # Change value again - watcher should not be called
        config_manager.set_value("database.host", "another-host")
        assert len(watcher_calls) == 1  # Still only one call
    
    def test_config_validation(self):
        config_manager = MockConfigManager()
        config_manager.load_config("config.json")
        
        # Valid configuration
        validation_result = config_manager.validate_config()
        assert validation_result['valid'] == True
        assert len(validation_result['errors']) == 0
        
        # Invalid configuration - missing required field
        invalid_config = {"api": {"timeout": 30}}
        validation_result = config_manager.validate_config(invalid_config)
        assert validation_result['valid'] == False
        assert len(validation_result['errors']) > 0
        
        # Test type validation
        config_manager.set_value("database.port", "not_a_number")
        validation_result = config_manager.validate_config()
        assert validation_result['valid'] == False
    
    def test_custom_validation_rules(self):
        config_manager = MockConfigManager()
        
        # Add custom validation rule
        def validate_positive_number(value):
            return isinstance(value, (int, float)) and value > 0
        
        config_manager.add_validation_rule("custom.value", validate_positive_number)
        
        # Valid value
        assert config_manager.set_value("custom.value", 10) == True
        
        # Invalid value (should fail validation)
        assert config_manager.set_value("custom.value", -5) == False
    
    def test_section_operations(self):
        config_manager = MockConfigManager()
        config_manager.load_config("config.json")
        
        # Get section
        database_section = config_manager.get_section("database")
        assert "host" in database_section
        assert "port" in database_section
        
        # Update section
        new_database_config = {
            "host": "updated-host",
            "port": 3306,
            "new_field": "new_value"
        }
        
        assert config_manager.update_section("database", new_database_config) == True
        assert config_manager.get_value("database.host") == "updated-host"
        assert config_manager.get_value("database.new_field") == "new_value"
    
    def test_config_merging(self):
        config_manager = MockConfigManager()
        config_manager.load_config("config.json")
        
        # Merge new configuration
        merge_config = {
            "database": {
                "password": "secret123",  # New field
                "port": 3306             # Override existing
            },
            "new_section": {
                "key": "value"
            }
        }
        
        assert config_manager.merge_config(merge_config) == True
        
        # Check merged values
        assert config_manager.get_value("database.password") == "secret123"
        assert config_manager.get_value("database.port") == 3306
        assert config_manager.get_value("database.host") == "localhost"  # Preserved
        assert config_manager.get_value("new_section.key") == "value"
    
    def test_config_rollback(self):
        config_manager = MockConfigManager()
        
        # Initial configuration
        config1 = {"version": 1, "data": "first"}
        config_manager.save_config(config1)
        
        # Second configuration
        config2 = {"version": 2, "data": "second"}
        config_manager.save_config(config2)
        
        # Third configuration
        config3 = {"version": 3, "data": "third"}
        config_manager.save_config(config3)
        
        # Current should be config3
        assert config_manager.get_value("version") == 3
        
        # Rollback one step
        assert config_manager.rollback_config(1) == True
        assert config_manager.get_value("version") == 2
        
        # Rollback two steps (should fail - not enough history)
        assert config_manager.rollback_config(3) == False
        
        # Current should still be config2
        assert config_manager.get_value("version") == 2


class TestEnvironmentConfig:
    """Test environment-specific configuration"""
    
    def test_environment_management(self):
        env_config = MockEnvironmentConfig("development")
        
        # Check initial environment
        assert env_config.get_environment() == "development"
        
        # Change environment
        assert env_config.set_environment("production") == True
        assert env_config.get_environment() == "production"
        
        # Invalid environment
        assert env_config.set_environment("invalid") == False
        assert env_config.get_environment() == "production"  # Unchanged
    
    def test_environment_configs(self):
        env_config = MockEnvironmentConfig("development")
        
        # Development config
        dev_config = env_config.get_env_config("development")
        assert dev_config["database"]["debug"] == True
        assert dev_config["api"]["timeout"] == 10
        
        # Production config
        prod_config = env_config.get_env_config("production")
        assert prod_config["database"]["debug"] == False
        assert prod_config["api"]["timeout"] == 30
        
        # Current environment config
        current_config = env_config.get_env_config()
        assert current_config == dev_config
    
    def test_environment_overrides(self):
        env_config = MockEnvironmentConfig("testing")
        
        merged_config = env_config.load_environment_overrides()
        
        # Should have base values
        assert merged_config["api"]["base_url"] == "https://api.example.com"
        
        # Should have environment-specific overrides
        assert merged_config["database"]["host"] == "test-db"
        assert merged_config["database"]["debug"] == True
        assert merged_config["api"]["timeout"] == 5


class TestConfigIntegration:
    """Test configuration manager integration scenarios"""
    
    def test_multi_format_config_loading(self):
        """Test loading and merging multiple config formats"""
        config_manager = MockConfigManager()
        
        # Load base JSON config
        json_config = config_manager.load_config("base_config.json")
        config_manager.config_data = json_config
        
        # Load and merge YAML overrides
        yaml_config = config_manager.load_config("overrides.yaml")
        config_manager.merge_config(yaml_config)
        
        # Should have values from both configs
        assert "database" in config_manager.config_data  # From JSON
        assert "system" in config_manager.config_data    # From YAML
    
    def test_config_with_environment_overrides(self):
        """Test configuration with environment-specific overrides"""
        config_manager = MockConfigManager()
        env_config = MockEnvironmentConfig("production")
        
        # Load base config
        base_config = config_manager.load_config("config.json")
        config_manager.config_data = base_config
        
        # Apply environment overrides
        env_overrides = env_config.load_environment_overrides()
        config_manager.merge_config(env_overrides)
        
        # Should have environment-specific values
        assert config_manager.get_value("database.debug") == False  # Production override
    
    def test_config_validation_with_watchers(self):
        """Test configuration validation combined with change watchers"""
        config_manager = MockConfigManager()
        config_manager.load_config("config.json")
        
        validation_errors = []
        
        def validation_watcher(key: str, value: Any):
            # Custom validation logic
            if key == "database.port" and (not isinstance(value, int) or value <= 0):
                validation_errors.append(f"Invalid port: {value}")
        
        # Add validation watcher
        config_manager.add_watcher("database.port", validation_watcher)
        
        # Set valid port
        config_manager.set_value("database.port", 5432)
        assert len(validation_errors) == 0
        
        # Set invalid port
        config_manager.set_value("database.port", -1)
        assert len(validation_errors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])