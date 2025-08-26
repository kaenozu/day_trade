#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Web Dashboard Configuration - 拡張ウェブダッシュボード設定

設定クラス、列挙型、データクラスの定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml


class DashboardTheme(Enum):
    """ダッシュボードテーマ"""
    LIGHT = "light"
    DARK = "dark"
    FINANCIAL = "financial"
    CUSTOM = "custom"


class ChartType(Enum):
    """チャートタイプ"""
    CANDLESTICK = "candlestick"
    LINE = "line"
    AREA = "area"
    OHLC = "ohlc"
    VOLUME = "volume"
    INDICATORS = "indicators"


class UpdateFrequency(Enum):
    """更新頻度"""
    REAL_TIME = "real_time"  # 1秒
    HIGH = "high"           # 5秒
    MEDIUM = "medium"       # 30秒
    LOW = "low"             # 5分
    MANUAL = "manual"       # 手動


@dataclass
class DashboardConfig:
    """ダッシュボード設定"""
    # 基本設定
    theme: DashboardTheme = DashboardTheme.FINANCIAL
    update_frequency: UpdateFrequency = UpdateFrequency.MEDIUM
    auto_refresh: bool = True

    # 表示設定
    default_symbols: List[str] = field(default_factory=lambda: ["7203", "8306", "9984"])
    charts_per_row: int = 2
    show_volume: bool = True
    show_indicators: bool = True

    # データ設定
    data_retention_days: int = 30
    cache_duration_minutes: int = 5

    # 通知設定
    alerts_enabled: bool = True
    email_notifications: bool = False
    sound_alerts: bool = True

    # 高度な設定
    ml_predictions_enabled: bool = True
    performance_monitoring_enabled: bool = True
    custom_indicators: List[str] = field(default_factory=list)


@dataclass
class AlertConfig:
    """アラート設定"""
    alert_id: str
    symbol: str
    alert_type: str  # price_threshold, volume_spike, prediction_change, etc.
    condition: str   # >, <, ==, change_percent, etc.
    threshold: float
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class UserPreferences:
    """ユーザー設定"""
    user_id: str
    dashboard_config: DashboardConfig
    custom_layouts: Dict[str, Any] = field(default_factory=dict)
    watchlist: List[str] = field(default_factory=list)
    alerts: List[AlertConfig] = field(default_factory=list)
    saved_analyses: List[str] = field(default_factory=list)


def load_dashboard_config(config_path: Optional[Path] = None) -> DashboardConfig:
    """設定の読み込み"""
    import logging
    logger = logging.getLogger(__name__)
    
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            return DashboardConfig(**config_data)
        except Exception as e:
            logger.warning(f"設定読み込みエラー: {e}. デフォルト設定を使用")

    return DashboardConfig()