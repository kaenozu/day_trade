#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - Alert Manager
アラート管理クラス
"""

import json
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

from .enums_and_models import (
    PerformanceAlert,
    PerformanceMetrics,
    RetrainingTrigger,
    AlertLevel
)
from .config_manager import EnhancedPerformanceConfigManager

logger = logging.getLogger(__name__)


class AlertManager:
    """アラート管理クラス"""

    def __init__(self, config_manager: EnhancedPerformanceConfigManager):
        self.config_manager = config_manager
        self.active_alerts = {}
        self.alerts = []

    async def check_performance_degradation(
            self, performance_data: Dict[str, PerformanceMetrics]
    ) -> List[PerformanceAlert]:
        """性能低下チェック"""
        alerts = []

        for key, metrics in performance_data.items():
            symbol = metrics.symbol
            model_type = metrics.model_type

            # 閾値チェック
            threshold_alerts = self._check_thresholds(metrics)
            alerts.extend(threshold_alerts)

            # 履歴比較チェック（PerformanceEvaluatorからの移行）
            # ここでは簡略化して基本的なチェックのみ
            if metrics.accuracy and metrics.accuracy < 0.75:
                alert = PerformanceAlert(
                    id=f"low_accuracy_{symbol}_{model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    model_type=model_type,
                    alert_level=AlertLevel.WARNING,
                    metric_name="accuracy",
                    current_value=metrics.accuracy,
                    threshold_value=0.75,
                    message=f"精度が低下しています: {metrics.accuracy:.3f}",
                    recommended_action="性能監視を強化"
                )
                alerts.append(alert)

        return alerts

    def _check_thresholds(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """閾値チェック"""
        alerts = []
        thresholds = self.config_manager.config.get('performance_thresholds', {})

        # 精度閾値チェック
        accuracy_thresholds = thresholds.get('accuracy', {})
        if metrics.accuracy is not None:
            if metrics.accuracy < accuracy_thresholds.get('critical_threshold', 0.70):
                alert = PerformanceAlert(
                    id=f"accuracy_critical_{metrics.symbol}_{metrics.model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=metrics.symbol,
                    model_type=metrics.model_type,
                    alert_level=AlertLevel.CRITICAL,
                    metric_name="accuracy",
                    current_value=metrics.accuracy,
                    threshold_value=accuracy_thresholds.get('critical_threshold', 0.70),
                    message=f"精度が緊急閾値を下回りました: {metrics.accuracy:.3f}",
                    recommended_action="即座に再学習を実行"
                )
                alerts.append(alert)
            elif metrics.accuracy < accuracy_thresholds.get('minimum_threshold', 0.75):
                alert = PerformanceAlert(
                    id=f"accuracy_warning_{metrics.symbol}_{metrics.model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=metrics.symbol,
                    model_type=metrics.model_type,
                    alert_level=AlertLevel.WARNING,
                    metric_name="accuracy",
                    current_value=metrics.accuracy,
                    threshold_value=accuracy_thresholds.get('minimum_threshold', 0.75),
                    message=f"精度が最低閾値を下回りました: {metrics.accuracy:.3f}",
                    recommended_action="再学習を検討"
                )
                alerts.append(alert)

        return alerts

    def log_alert(self, alert: PerformanceAlert):
        """アラートログ出力"""
        level_map = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }

        log_func = level_map.get(alert.alert_level, logger.info)
        log_func(
            f"[{alert.alert_level.value.upper()}] "
            f"{alert.symbol}_{alert.model_type}: {alert.message}"
        )

    async def trigger_retraining(
            self, alerts: List[PerformanceAlert]
    ) -> Optional[RetrainingTrigger]:
        """再学習トリガー（従来互換）"""
        if not alerts:
            return None

        # 重要なアラートのみ処理
        critical_alerts = [
            a for a in alerts 
            if a.alert_level in [AlertLevel.CRITICAL, AlertLevel.ERROR]
        ]

        if not critical_alerts:
            return None

        # 影響を受ける銘柄とモデルを特定
        affected_symbols = list(set(a.symbol for a in critical_alerts))
        affected_models = list(set(a.model_type for a in critical_alerts))

        # 再学習戦略決定
        strategy = self.config_manager.config.get(
            'retraining', {}
        ).get('strategy', {})
        granularity = strategy.get('granularity', 'selective')

        # 再学習トリガー作成
        trigger = RetrainingTrigger(
            trigger_id=f"retraining_{int(time.time())}",
            timestamp=datetime.now(),
            trigger_type=granularity,
            affected_symbols=affected_symbols,
            affected_models=affected_models,
            reason=f"{len(critical_alerts)}個の重要なアラートが発生",
            severity=max(alert.alert_level for alert in critical_alerts),
            recommended_action=self._generate_retraining_recommendation(
                critical_alerts, granularity
            )
        )

        return trigger

    def _generate_retraining_recommendation(
            self, alerts: List[PerformanceAlert], granularity: str
    ) -> str:
        """再学習推奨事項生成"""
        if granularity == 'selective':
            return (
                f"影響を受けた{len(set(a.symbol for a in alerts))}銘柄の"
                f"選択的再学習を実行"
            )
        elif granularity == 'full':
            return "システム全体の再学習が必要"
        else:
            return "増分学習による性能改善を実行"

    def add_alert_to_active(self, alert: PerformanceAlert):
        """アクティブアラートに追加"""
        self.active_alerts[alert.id] = alert

    def get_active_alert_count(self) -> int:
        """アクティブアラート数取得"""
        return len(self.active_alerts)

    def resolve_alert(self, alert_id: str):
        """アラート解決"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]
            logger.info(f"アラートを解決しました: {alert_id}")

    def get_alert_summary(self) -> Dict[str, Any]:
        """アラート要約取得"""
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = sum(
                1 for alert in self.active_alerts.values()
                if alert.alert_level == level
            )

        return {
            'total_active': len(self.active_alerts),
            'level_breakdown': level_counts,
            'latest_alerts': [
                {
                    'id': alert.id,
                    'symbol': alert.symbol,
                    'level': alert.alert_level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in sorted(
                    self.active_alerts.values(),
                    key=lambda a: a.timestamp,
                    reverse=True
                )[:10]
            ]
        }