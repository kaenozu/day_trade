#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Types - 性能監視システム共通型定義

MLモデル性能監視システムで使用される共通のデータ型、Enum、dataclassを定義
"""

import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


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
    """性能指標
    
    MLモデルの性能を表現するメトリクス
    
    Attributes:
        symbol: 銘柄コード
        model_type: モデルタイプ
        timestamp: 測定時刻
        accuracy: 精度
        precision: 適合率
        recall: 再現率
        f1_score: F1スコア
        r2_score: 決定係数
        mse: 平均二乗誤差
        mae: 平均絶対誤差
        prediction_time_ms: 予測時間（ミリ秒）
        confidence_avg: 平均信頼度
        sample_size: サンプルサイズ
        status: 性能ステータス
        prediction_accuracy: 予測精度
        return_prediction: リターン予測精度
        volatility_prediction: ボラティリティ予測精度
        source: データソース
    """
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
        """初期化後処理"""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerformanceAlert:
    """性能アラート
    
    性能低下やしきい値違反を通知するアラート情報
    
    Attributes:
        id: アラートID
        timestamp: 発生時刻
        symbol: 対象銘柄
        model_type: 対象モデルタイプ
        alert_level: アラートレベル
        metric_name: メトリクス名
        current_value: 現在値
        threshold_value: しきい値
        message: メッセージ
        resolved: 解決済みフラグ
        resolved_at: 解決時刻
        recommended_action: 推奨アクション
    """
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
    # レガシー属性（後方互換性）
    level: AlertLevel = None
    metric: str = ""
    threshold: float = 0.0

    def __post_init__(self):
        """初期化後処理"""
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
    """再学習トリガー
    
    モデル再学習をトリガーする情報
    
    Attributes:
        trigger_id: トリガーID
        timestamp: 発生時刻
        trigger_type: トリガータイプ
        affected_symbols: 影響を受ける銘柄リスト
        affected_models: 影響を受けるモデルリスト
        reason: 理由
        severity: 重要度
        recommended_action: 推奨アクション
    """
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
    """再学習結果情報
    
    再学習実行の結果を表現する情報
    
    Attributes:
        triggered: 実行されたかどうか
        scope: 再学習のスコープ
        affected_symbols: 影響を受ける銘柄リスト
        improvement: 改善度（%）
        duration: 実行時間（秒）
        error: エラーメッセージ
        estimated_time: 推定実行時間（秒）
        actual_benefit: 実際の効果
    """
    triggered: bool
    scope: RetrainingScope
    affected_symbols: List[str]
    improvement: float = 0.0
    duration: float = 0.0
    error: Optional[str] = None
    estimated_time: float = 0.0
    actual_benefit: float = 0.0