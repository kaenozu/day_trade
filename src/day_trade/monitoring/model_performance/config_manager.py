#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - Configuration Manager
設定管理クラス
"""

import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EnhancedPerformanceConfigManager:
    """改善版性能設定管理クラス"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = (
            config_path or Path("config/performance_monitoring.yaml")
        )
        self.config = {}
        self.last_loaded = None
        self._load_configuration()

    def _load_configuration(self):
        """設定ファイルを読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self.last_loaded = datetime.now()
                logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            else:
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                self._create_default_config()

        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            self._load_default_config()

    def _create_default_config(self):
        """デフォルト設定ファイルの作成"""
        default_config = {
            'performance_thresholds': {
                'accuracy': {
                    'minimum_threshold': 0.75,
                    'warning_threshold': 0.80,
                    'target_threshold': 0.90,
                    'critical_threshold': 0.70
                },
                'confidence': {
                    'minimum_threshold': 0.65,
                    'warning_threshold': 0.75,
                    'target_threshold': 0.85
                },
                'prediction_accuracy': 85.0,
                'return_prediction': 75.0,
                'volatility_prediction': 80.0
            },
            'monitoring_symbols': {
                'primary_symbols': ['7203', '8306', '9984', '4751', '2914'],
                'dynamic_selection': {
                    'enabled': True,
                    'selection_criteria': {'min_market_cap': 100000000000}
                }
            },
            'monitoring': {
                'intervals': {
                    'real_time_seconds': 60,
                    'periodic_minutes': 15
                },
                'default_symbols': ["7203", "8306", "4751"],
                'dynamic_monitoring': {'enabled': True},
                'validation_hours': 168,
                'min_samples': 10
            },
            'retraining': {
                'triggers': {
                    'accuracy_degradation': {
                        'enabled': True,
                        'threshold_drop': 0.05,
                        'consecutive_failures': 3
                    }
                },
                'strategy': {
                    'granularity': 'selective',
                    'batch_size': 5,
                    'max_concurrent': 2
                },
                'granular_mode': True,
                'cooldown_hours': 24,
                'global_threshold': 80.0,
                'symbol_specific_threshold': 85.0,
                'partial_threshold': 88.0,
                'scopes': {
                    'global': {'estimated_time': 3600},
                    'partial': {'estimated_time': 1800},
                    'symbol': {'estimated_time': 600},
                    'incremental': {'estimated_time': 300}
                }
            },
            'database': {
                'connection': {
                    'path': 'data/model_performance.db'
                }
            },
            'models': {
                'supported_models': ['RandomForestClassifier', 'XGBoostClassifier']
            },
            'system': {
                'log_level': 'INFO'
            }
        }

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    default_config, f, default_flow_style=False, allow_unicode=True
                )
            logger.info(f"デフォルト設定ファイルを作成: {self.config_path}")
            self.config = default_config
        except Exception as e:
            logger.error(f"デフォルト設定ファイル作成エラー: {e}")
            self._load_default_config()

    def _load_default_config(self):
        """デフォルト設定を読み込み"""
        self.config = {
            'performance_thresholds': {
                'accuracy': {
                    'minimum_threshold': 0.75,
                    'warning_threshold': 0.80,
                    'target_threshold': 0.90,
                    'critical_threshold': 0.70
                },
                'confidence': {
                    'minimum_threshold': 0.65,
                    'warning_threshold': 0.75,
                    'target_threshold': 0.85
                }
            },
            'monitoring_symbols': {
                'primary_symbols': ['7203', '8306', '9984', '4751', '2914']
            },
            'monitoring': {
                'intervals': {
                    'real_time_seconds': 60,
                    'periodic_minutes': 15
                }
            },
            'retraining': {
                'triggers': {
                    'accuracy_degradation': {
                        'enabled': True,
                        'threshold_drop': 0.05,
                        'consecutive_failures': 3
                    }
                },
                'strategy': {
                    'granularity': 'selective',
                    'batch_size': 5,
                    'max_concurrent': 2
                }
            }
        }
        logger.info("デフォルト設定を使用します")

    def get_threshold(self, metric: str, symbol: str = None) -> float:
        """閾値取得（銘柄別対応）"""
        thresholds = self.config.get('performance_thresholds', {})

        # 銘柄別閾値チェック
        if symbol and 'symbol_specific' in thresholds:
            symbol_category = self._categorize_symbol(symbol)
            symbol_thresholds = thresholds['symbol_specific'].get(
                symbol_category, {}
            )
            if metric in symbol_thresholds:
                return symbol_thresholds[metric]

        # メトリック別閾値取得
        if metric in thresholds:
            if isinstance(thresholds[metric], dict):
                return thresholds[metric].get('target_threshold', 90.0)
            else:
                return thresholds[metric]

        # デフォルト閾値
        return 90.0

    def _categorize_symbol(self, symbol: str) -> str:
        """銘柄カテゴリ分類"""
        high_volume = ["7203", "8306", "9984", "6758"]
        if symbol in high_volume:
            return "high_volume_stocks"
        return "mid_volume_stocks"

    def get_monitoring_config(self) -> Dict[str, Any]:
        """監視設定取得"""
        return self.config.get('monitoring', {})

    def get_retraining_config(self) -> Dict[str, Any]:
        """再学習設定取得"""
        return self.config.get('retraining', {})

    def reload_if_modified(self):
        """設定ファイルが更新されていれば再読み込み"""
        try:
            if self.config_path.exists():
                file_mtime = datetime.fromtimestamp(
                    self.config_path.stat().st_mtime
                )
                if self.last_loaded is None or file_mtime > self.last_loaded:
                    self._load_configuration()
        except Exception as e:
            logger.error(f"設定ファイル更新チェックエラー: {e}")