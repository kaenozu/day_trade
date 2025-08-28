#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alert Manager - アラート管理

性能監視システムのアラート生成、管理、通知を行うモジュール
閾値チェック、履歴比較、アラートの永続化を提供
"""

import asyncio
import json
import logging
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from .config import EnhancedPerformanceConfigManager
from .types import PerformanceMetrics, PerformanceAlert, AlertLevel, RetrainingTrigger

logger = logging.getLogger(__name__)


class AlertManager:
    """アラート管理クラス
    
    性能メトリクスの監視、閾値チェック、アラート生成と管理を行います。
    履歴データとの比較や再学習トリガーの生成も担当します。
    
    Attributes:
        config_manager: 設定管理インスタンス
        active_alerts: アクティブなアラート辞書
        alert_history: アラート履歴リスト
        performance_history: 性能履歴辞書（銘柄・モデル別）
    """

    def __init__(
            self, config_manager: EnhancedPerformanceConfigManager,
            save_callback=None
    ):
        """初期化
        
        Args:
            config_manager: 設定管理インスタンス
            save_callback: アラート保存用コールバック関数
        """
        self.config_manager = config_manager
        self.active_alerts = {}
        self.alert_history = []
        self.performance_history = {}
        self.save_callback = save_callback

    async def check_and_generate_alerts(
            self, performance: PerformanceMetrics
    ) -> List[PerformanceAlert]:
        """性能アラートチェックと生成
        
        性能メトリクスを分析してアラートを生成します。
        
        Args:
            performance: 性能メトリクス
            
        Returns:
            生成されたアラートのリスト
        """
        alerts = []

        # 主要メトリクスのチェック
        metrics_to_check = {
            'accuracy': performance.accuracy,
            'prediction_accuracy': performance.prediction_accuracy,
            'return_prediction': performance.return_prediction,
            'volatility_prediction': performance.volatility_prediction,
            'confidence': performance.confidence_avg
        }

        for metric_name, value in metrics_to_check.items():
            if value is None:
                continue

            # 閾値取得（百分率を小数に変換）
            threshold = self.config_manager.get_threshold(metric_name, performance.symbol)
            if metric_name in ['prediction_accuracy', 'return_prediction', 'volatility_prediction']:
                threshold_value = threshold  # 既に百分率
                comparison_value = value
            else:
                threshold_value = threshold  # 小数値
                comparison_value = value

            if comparison_value < threshold_value:
                # アラートレベル決定
                level, action = self._determine_alert_level(
                    comparison_value, threshold_value, metric_name
                )

                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    alert_level=level,
                    symbol=performance.symbol,
                    model_type=performance.model_type,
                    metric_name=metric_name,
                    current_value=comparison_value,
                    threshold_value=threshold_value,
                    message=(
                        f"{metric_name}が閾値を下回りました: "
                        f"{comparison_value:.2f} < {threshold_value:.2f}"
                    ),
                    recommended_action=action
                )

                alerts.append(alert)
                
                # アラート保存
                if self.save_callback:
                    await self.save_callback(alert)
                    
                # アクティブアラートに追加
                self.active_alerts[alert.id] = alert

                logger.warning(
                    f"アラート生成: {performance.symbol} {metric_name} "
                    f"{comparison_value:.2f} < {threshold_value:.2f} ({level.value})"
                )

        return alerts

    def _determine_alert_level(
            self, current_value: float, threshold_value: float, metric_name: str
    ) -> tuple:
        """アラートレベルと推奨アクションを決定
        
        Args:
            current_value: 現在値
            threshold_value: 閾値
            metric_name: メトリクス名
            
        Returns:
            (アラートレベル, 推奨アクション)のタプル
        """
        degradation_ratio = (threshold_value - current_value) / threshold_value

        if degradation_ratio > 0.15:  # 15%以上の劣化
            level = AlertLevel.CRITICAL
            action = "即座に再学習を検討してください"
        elif degradation_ratio > 0.05:  # 5%以上の劣化
            level = AlertLevel.ERROR
            action = "再学習の準備を開始してください"
        elif degradation_ratio > 0.02:  # 2%以上の劣化
            level = AlertLevel.WARNING
            action = "監視を強化し、再学習を検討してください"
        else:
            level = AlertLevel.INFO
            action = "継続監視を行ってください"

        return level, action

    async def check_performance_degradation(
            self, performance_data: Dict[str, PerformanceMetrics]
    ) -> List[PerformanceAlert]:
        """性能低下チェック
        
        現在の性能データと履歴を比較して劣化を検知します。
        
        Args:
            performance_data: 銘柄・モデル別性能データ
            
        Returns:
            検出されたアラートのリスト
        """
        alerts = []

        for key, metrics in performance_data.items():
            # 現在のメトリクスをアラートチェック
            threshold_alerts = await self._check_thresholds(metrics)
            alerts.extend(threshold_alerts)

            # 履歴比較チェック
            history_alerts = await self._check_performance_history(key, metrics)
            alerts.extend(history_alerts)

            # 履歴に追加
            self._add_to_history(key, metrics)

        return alerts

    async def _check_thresholds(
            self, metrics: PerformanceMetrics
    ) -> List[PerformanceAlert]:
        """閾値チェック
        
        Args:
            metrics: 性能メトリクス
            
        Returns:
            閾値違反アラートのリスト
        """
        alerts = []
        thresholds = self.config_manager.config.get('performance_thresholds', {})

        # 精度閾値チェック
        accuracy_thresholds = thresholds.get('accuracy', {})
        if metrics.accuracy is not None:
            critical_threshold = accuracy_thresholds.get('critical_threshold', 0.70)
            minimum_threshold = accuracy_thresholds.get('minimum_threshold', 0.75)

            if metrics.accuracy < critical_threshold:
                alert = PerformanceAlert(
                    id=f"accuracy_critical_{metrics.symbol}_{metrics.model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=metrics.symbol,
                    model_type=metrics.model_type,
                    alert_level=AlertLevel.CRITICAL,
                    metric_name="accuracy",
                    current_value=metrics.accuracy,
                    threshold_value=critical_threshold,
                    message=f"精度が緊急閾値を下回りました: {metrics.accuracy:.3f} < {critical_threshold:.3f}",
                    recommended_action="即座に再学習を実行してください"
                )
                alerts.append(alert)
                
            elif metrics.accuracy < minimum_threshold:
                alert = PerformanceAlert(
                    id=f"accuracy_warning_{metrics.symbol}_{metrics.model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=metrics.symbol,
                    model_type=metrics.model_type,
                    alert_level=AlertLevel.WARNING,
                    metric_name="accuracy",
                    current_value=metrics.accuracy,
                    threshold_value=minimum_threshold,
                    message=f"精度が最低閾値を下回りました: {metrics.accuracy:.3f} < {minimum_threshold:.3f}",
                    recommended_action="再学習を検討してください"
                )
                alerts.append(alert)

        return alerts

    async def _check_performance_history(
            self, key: str, current_metrics: PerformanceMetrics
    ) -> List[PerformanceAlert]:
        """履歴比較チェック
        
        Args:
            key: パフォーマンスキー（銘柄_モデル）
            current_metrics: 現在のメトリクス
            
        Returns:
            履歴比較アラートのリスト
        """
        alerts = []

        if key not in self.performance_history or len(self.performance_history[key]) < 3:
            return alerts

        history = self.performance_history[key]
        recent_accuracy = [m.accuracy for m in history[-3:] if m.accuracy is not None]

        if len(recent_accuracy) >= 3 and current_metrics.accuracy is not None:
            avg_recent = np.mean(recent_accuracy[:-1])  # 最新を除く平均
            degradation = avg_recent - current_metrics.accuracy

            retraining_config = self.config_manager.get_retraining_config()
            threshold_drop = (retraining_config.get('triggers', {})
                            .get('accuracy_degradation', {})
                            .get('threshold_drop', 0.05))

            if degradation > threshold_drop:
                alert = PerformanceAlert(
                    id=f"degradation_{current_metrics.symbol}_{current_metrics.model_type}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=current_metrics.symbol,
                    model_type=current_metrics.model_type,
                    alert_level=AlertLevel.ERROR,
                    metric_name="accuracy_degradation",
                    current_value=current_metrics.accuracy,
                    threshold_value=avg_recent - threshold_drop,
                    message=f"精度が大幅に低下しました: {degradation:.3f}ポイント低下",
                    recommended_action="再学習を実行してください"
                )
                alerts.append(alert)

        return alerts

    def _add_to_history(self, key: str, metrics: PerformanceMetrics):
        """履歴にメトリクスを追加
        
        Args:
            key: パフォーマンスキー
            metrics: 性能メトリクス
        """
        if key not in self.performance_history:
            self.performance_history[key] = []
            
        self.performance_history[key].append(metrics)
        
        # 履歴サイズ制限（最新100件まで）
        max_history = 100
        if len(self.performance_history[key]) > max_history:
            self.performance_history[key] = self.performance_history[key][-max_history:]

    async def trigger_retraining(
            self, alerts: List[PerformanceAlert]
    ) -> Optional[RetrainingTrigger]:
        """再学習トリガー生成
        
        Args:
            alerts: アラートのリスト
            
        Returns:
            再学習トリガー（生成されない場合はNone）
        """
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
        retraining_config = self.config_manager.get_retraining_config()
        strategy = retraining_config.get('strategy', {})
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

        logger.info(f"再学習トリガー生成: {trigger.trigger_id}")
        return trigger

    def _generate_retraining_recommendation(
            self, alerts: List[PerformanceAlert], granularity: str
    ) -> str:
        """再学習推奨事項生成
        
        Args:
            alerts: アラートのリスト
            granularity: 再学習粒度
            
        Returns:
            推奨事項の説明
        """
        symbol_count = len(set(a.symbol for a in alerts))
        
        if granularity == 'selective':
            return f"影響を受けた{symbol_count}銘柄の選択的再学習を実行"
        elif granularity == 'full':
            return "システム全体の再学習が必要"
        else:
            return "増分学習による性能改善を実行"

    def resolve_alert(self, alert_id: str) -> bool:
        """アラートを解決済みにする
        
        Args:
            alert_id: アラートID
            
        Returns:
            解決に成功した場合はTrue
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].resolved_at = datetime.now()
            logger.info(f"アラート解決: {alert_id}")
            return True
        else:
            logger.warning(f"アラートが見つかりません: {alert_id}")
            return False

    def get_active_alerts(self) -> List[PerformanceAlert]:
        """アクティブなアラートを取得
        
        Returns:
            アクティブなアラートのリスト
        """
        return [alert for alert in self.active_alerts.values() if not alert.resolved]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """アラート統計情報を取得
        
        Returns:
            統計情報辞書
        """
        active_alerts = self.get_active_alerts()
        
        stats = {
            'total_active': len(active_alerts),
            'by_level': {},
            'by_symbol': {},
            'by_metric': {},
            'recent_count': 0
        }

        # レベル別集計
        for level in AlertLevel:
            stats['by_level'][level.value] = sum(
                1 for alert in active_alerts if alert.alert_level == level
            )

        # 銘柄別集計
        for alert in active_alerts:
            symbol = alert.symbol
            if symbol not in stats['by_symbol']:
                stats['by_symbol'][symbol] = 0
            stats['by_symbol'][symbol] += 1

        # メトリクス別集計
        for alert in active_alerts:
            metric = alert.metric_name
            if metric not in stats['by_metric']:
                stats['by_metric'][metric] = 0
            stats['by_metric'][metric] += 1

        # 最近1時間のアラート数
        one_hour_ago = datetime.now().timestamp() - 3600
        stats['recent_count'] = sum(
            1 for alert in active_alerts
            if alert.timestamp.timestamp() > one_hour_ago
        )

        return stats

    def cleanup_old_alerts(self, hours_old: int = 24):
        """古いアラートをクリーンアップ
        
        Args:
            hours_old: クリーンアップ対象の時間（時間）
        """
        cutoff_time = datetime.now().timestamp() - (hours_old * 3600)
        
        # 解決済みかつ古いアラートを削除
        alerts_to_remove = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.resolved and alert.timestamp.timestamp() < cutoff_time
        ]
        
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
            
        if alerts_to_remove:
            logger.info(f"古いアラートをクリーンアップ: {len(alerts_to_remove)}件")