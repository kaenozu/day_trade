#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Source Configuration Manager

データソース設定管理モジュール
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .enums import DataSourceConfig


class DataSourceConfigManager:
    """データソース設定管理"""

    def __init__(self, config_path: Optional[Path] = None):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path("config/data_sources.yaml")
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, DataSourceConfig]:
        """設定ファイル読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                configs = {}
                for name, data in config_data.get('data_sources', {}).items():
                    # rate_limits辞書を個別パラメータに変換
                    if 'rate_limits' in data:
                        rate_limits = data.pop('rate_limits')
                        data['rate_limit_per_minute'] = rate_limits.get(
                            'requests_per_minute', 60
                        )
                        data['rate_limit_per_day'] = rate_limits.get(
                            'daily_limit', 1000
                        )

                    # DataSourceConfigで未対応のフィールドを除外
                    unsupported_fields = [
                        'quality_settings', 'retry_config', 'fallback_priority'
                    ]
                    for field in unsupported_fields:
                        data.pop(field, None)

                    configs[name] = DataSourceConfig(name=name, **data)

                self.logger.info(
                    f"Loaded {len(configs)} data source configurations"
                )
                return configs
            else:
                self.logger.warning(
                    f"Config file not found: {self.config_path}, using defaults"
                )
                return self._get_default_configs()

        except Exception as e:
            self.logger.error(f"Failed to load data source configs: {e}")
            return self._get_default_configs()

    def _get_default_configs(self) -> Dict[str, DataSourceConfig]:
        """デフォルト設定"""
        return {
            'yahoo_finance': DataSourceConfig(
                name='yahoo_finance',
                enabled=True,
                priority=1,
                timeout=30,
                rate_limit_per_minute=60,
                rate_limit_per_day=1000,
                quality_threshold=80.0,
                max_price_threshold=1000000
            ),
            'stooq': DataSourceConfig(
                name='stooq',
                enabled=True,
                priority=2,
                timeout=20,
                base_url="https://stooq.com/q/d/l/",
                rate_limit_per_minute=30,
                quality_threshold=70.0
            ),
            'mock': DataSourceConfig(
                name='mock',
                enabled=True,
                priority=99,
                timeout=1,
                quality_threshold=50.0
            )
        }

    def get_config(self, source_name: str) -> Optional[DataSourceConfig]:
        """設定取得"""
        return self.configs.get(source_name)

    def is_enabled(self, source_name: str) -> bool:
        """有効性確認"""
        config = self.get_config(source_name)
        return config.enabled if config else False

    def enable_source(self, source_name: str):
        """データソース有効化"""
        if source_name in self.configs:
            self.configs[source_name].enabled = True
            self.logger.info(f"Enabled data source: {source_name}")

    def disable_source(self, source_name: str):
        """データソース無効化"""
        if source_name in self.configs:
            self.configs[source_name].enabled = False
            self.logger.info(f"Disabled data source: {source_name}")

    def get_enabled_sources(self) -> List[str]:
        """有効なデータソース一覧"""
        return [
            name for name, config in self.configs.items()
            if config.enabled
        ]

    def save_configs(self):
        """設定保存"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            config_data = {
                'data_sources': {
                    name: {
                        'enabled': config.enabled,
                        'priority': config.priority,
                        'timeout': config.timeout,
                        'retry_count': config.retry_count,
                        'rate_limit_per_minute': config.rate_limit_per_minute,
                        'rate_limit_per_day': config.rate_limit_per_day,
                        'quality_threshold': config.quality_threshold,
                        'api_key': config.api_key,
                        'base_url': config.base_url,
                        'headers': config.headers,
                        'min_data_points': config.min_data_points,
                        'max_price_threshold': config.max_price_threshold,
                        'price_consistency_check': config.price_consistency_check,
                        'cache_enabled': config.cache_enabled,
                        'cache_ttl_seconds': config.cache_ttl_seconds
                    }
                    for name, config in self.configs.items()
                }
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_data, f, default_flow_style=False,
                    allow_unicode=True
                )

            self.logger.info(
                f"Saved data source configurations to {self.config_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to save configs: {e}")