#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - Core Enums and Models
コア列挙型とモデル定義
"""

import time
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum


class PerformanceStatus(Enum):
    """性能ステータス"""
    EXCELLENT = "excellent"      # 目標閾値以上
    GOOD = "good"               # 警告閾値以上
    WARNING = "warning"         # 最低閾値以上
    CRITICAL = "critical"       # 最低閾値未満
    UNKNOWN = "unknown"         # 評価不可


class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RetrainingScope(Enum):
    """再学習スコープ"""
    NONE = "none"
    INCREMENTAL = "incremental"
    SYMBOL = "symbol"
    PARTIAL = "partial"
    GLOBAL = "global"


@dataclass
class PerformanceMetrics:
    """性能指標"""
    symbol: str
    model_type: str = "Default"
    timestamp: datetime = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    r2_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    prediction_time_ms: Optional[float] = None
    confidence_avg: Optional[float] = None
    sample_size: int = 0
    status: PerformanceStatus = PerformanceStatus.UNKNOWN
    prediction_accuracy: float = 0.0
    return_prediction: float = 0.0
    volatility_prediction: float = 0.0
    source: str = "validator"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerformanceAlert:
    """性能アラート"""
    id: str = None
    timestamp: datetime = None
    symbol: str = ""
    model_type: str = ""
    alert_level: AlertLevel = AlertLevel.INFO
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    message: str = ""
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    recommended_action: str = ""
    level: AlertLevel = None
    metric: str = ""
    threshold: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.id is None:
            self.id = f"alert_{self.symbol}_{self.metric_name}_{int(time.time())}"
        # レガシー属性の同期
        if self.level:
            self.alert_level = self.level
        if self.metric:
            self.metric_name = self.metric
        if self.threshold:
            self.threshold_value = self.threshold


@dataclass
class RetrainingTrigger:
    """再学習トリガー"""
    trigger_id: str
    timestamp: datetime
    trigger_type: str
    affected_symbols: List[str]
    affected_models: List[str]
    reason: str
    severity: AlertLevel
    recommended_action: str


@dataclass
class RetrainingResult:
    """再学習結果情報"""
    triggered: bool
    scope: RetrainingScope
    affected_symbols: List[str]
    improvement: float = 0.0
    duration: float = 0.0
    error: Optional[str] = None
    estimated_time: float = 0.0
    actual_benefit: float = 0.0