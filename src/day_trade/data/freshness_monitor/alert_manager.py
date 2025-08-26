#!/usr/bin/env python3
"""
アラート管理機能
データ品質・鮮度問題のアラート評価、生成、通知を管理します
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from .database_operations import DatabaseOperations
from .enums import AlertSeverity, FreshnessStatus
from .models import (
    DataAlert,
    DataSourceConfig,
    FreshnessCheck,
    IntegrityCheck,
    MonitoringStats
)

# 依存コンポーネントのインポート
try:
    from ...monitoring.advanced_anomaly_detection_alerts import AdvancedAnomalyAlertSystem
    ANOMALY_SYSTEM_AVAILABLE = True
except ImportError:
    ANOMALY_SYSTEM_AVAILABLE = False


class AlertManager:
    """アラート管理システム
    
    データ品質・鮮度の問題を監視し、適切なアラートを生成・管理します。
    重要度に応じたアラート分類、通知機能、解決管理を提供します。
    """
    
    def __init__(self, db_operations: DatabaseOperations):
        """アラート管理システムを初期化
        
        Args:
            db_operations: データベース操作インスタンス
        """
        self.logger = logging.getLogger(__name__)
        self.db_operations = db_operations
        
        # アクティブアラート管理
        self.active_alerts: Dict[str, List[DataAlert]] = defaultdict(list)
        
        # コールバック管理
        self.alert_callbacks: List[Callable] = []
        
        # 統計情報
        self.stats = MonitoringStats()
        
        # 外部システム初期化
        self.anomaly_system = None
        if ANOMALY_SYSTEM_AVAILABLE:
            try:
                self.anomaly_system = AdvancedAnomalyAlertSystem()
                self.logger.info("異常検知アラートシステム初期化完了")
            except Exception as e:
                self.logger.warning(f"異常検知システム初期化エラー: {e}")
    
    async def evaluate_alerts(
        self,
        config: DataSourceConfig,
        freshness_check: FreshnessCheck,
        integrity_checks: List[IntegrityCheck]
    ):
        """アラート評価・生成
        
        鮮度チェックと整合性チェックの結果に基づいて、
        適切なアラートを評価・生成します。
        
        Args:
            config: データソース設定
            freshness_check: 鮮度チェック結果
            integrity_checks: 整合性チェック結果リスト
        """
        alerts_to_generate = []
        
        try:
            # 鮮度アラート評価
            freshness_alert = self._evaluate_freshness_alert(
                config, freshness_check
            )
            if freshness_alert:
                alerts_to_generate.append(freshness_alert)
                self.stats.freshness_violations += 1
            
            # 整合性アラート評価
            integrity_alerts = self._evaluate_integrity_alerts(
                config, integrity_checks
            )
            alerts_to_generate.extend(integrity_alerts)
            self.stats.integrity_violations += len(integrity_alerts)
            
            # 品質アラート評価
            quality_alert = self._evaluate_quality_alert(
                config, freshness_check
            )
            if quality_alert:
                alerts_to_generate.append(quality_alert)
            
            # 複合アラート評価（複数の問題が同時発生した場合）
            composite_alert = self._evaluate_composite_alert(
                config, freshness_check, integrity_checks
            )
            if composite_alert:
                alerts_to_generate.append(composite_alert)
            
            # アラート生成・通知
            for alert in alerts_to_generate:
                await self._generate_and_notify_alert(alert, config)
                
        except Exception as e:
            self.logger.error(f"アラート評価エラー ({config.source_id}): {e}")
    
    def _evaluate_freshness_alert(
        self,
        config: DataSourceConfig,
        freshness_check: FreshnessCheck
    ) -> DataAlert:
        """鮮度アラート評価
        
        Args:
            config: データソース設定
            freshness_check: 鮮度チェック結果
            
        Returns:
            鮮度アラート、問題がない場合はNone
        """
        if freshness_check.status == FreshnessStatus.FRESH:
            return None
        
        # 重要度判定
        if freshness_check.status == FreshnessStatus.EXPIRED:
            severity = (
                AlertSeverity.CRITICAL
                if freshness_check.age_seconds > config.freshness_threshold * 3
                else AlertSeverity.ERROR
                if freshness_check.age_seconds > config.freshness_threshold * 2
                else AlertSeverity.WARNING
            )
        else:  # STALE
            severity = AlertSeverity.WARNING
        
        # アラートメッセージ作成
        status_msg = {
            FreshnessStatus.STALE: "やや古い",
            FreshnessStatus.EXPIRED: "期限切れ"
        }.get(freshness_check.status, "不明")
        
        message = (
            f"データが{status_msg}状態です: "
            f"{freshness_check.age_seconds:.0f}秒経過 "
            f"(閾値: {config.freshness_threshold}秒)"
        )
        
        return DataAlert(
            alert_id=f"freshness_{config.source_id}_{int(time.time())}",
            source_id=config.source_id,
            severity=severity,
            alert_type="data_freshness",
            message=message,
            timestamp=freshness_check.timestamp,
            metadata={
                "age_seconds": freshness_check.age_seconds,
                "threshold": config.freshness_threshold,
                "last_update": freshness_check.last_update.isoformat(),
                "status": freshness_check.status.value,
            },
        )
    
    def _evaluate_integrity_alerts(
        self,
        config: DataSourceConfig,
        integrity_checks: List[IntegrityCheck]
    ) -> List[DataAlert]:
        """整合性アラート評価
        
        Args:
            config: データソース設定
            integrity_checks: 整合性チェック結果リスト
            
        Returns:
            整合性アラートのリスト
        """
        alerts = []
        
        for check in integrity_checks:
            if check.passed:
                continue
            
            # 重要度判定（チェック種別と問題の深刻度に基づく）
            severity = self._determine_integrity_severity(check)
            
            alert = DataAlert(
                alert_id=f"integrity_{config.source_id}_{check.check_type}_{int(time.time())}",
                source_id=config.source_id,
                severity=severity,
                alert_type="data_integrity",
                message=f"整合性チェック失敗 ({check.check_type}): {', '.join(check.issues_found)}",
                timestamp=check.timestamp,
                metadata={
                    "check_type": check.check_type,
                    "issues": check.issues_found,
                    "metrics": check.metrics,
                    "baseline_comparison": check.baseline_comparison,
                },
            )
            alerts.append(alert)
        
        return alerts
    
    def _determine_integrity_severity(self, check: IntegrityCheck) -> AlertSeverity:
        """整合性チェックの重要度判定
        
        Args:
            check: 整合性チェック結果
            
        Returns:
            アラート重要度
        """
        # チェック種別による重要度マッピング
        severity_map = {
            "data_availability": AlertSeverity.CRITICAL,
            "record_count": AlertSeverity.ERROR,
            "data_quality": AlertSeverity.WARNING,
            "baseline_comparison": AlertSeverity.WARNING,
            "metadata_integrity": AlertSeverity.INFO,
            "anomaly_detection": AlertSeverity.ERROR,
            "trend_analysis": AlertSeverity.WARNING,
        }
        
        base_severity = severity_map.get(check.check_type, AlertSeverity.WARNING)
        
        # 問題の数による重要度調整
        issue_count = len(check.issues_found)
        if issue_count > 3:
            # 問題が多い場合は重要度を上げる
            if base_severity == AlertSeverity.WARNING:
                return AlertSeverity.ERROR
            elif base_severity == AlertSeverity.ERROR:
                return AlertSeverity.CRITICAL
        
        return base_severity
    
    def _evaluate_quality_alert(
        self,
        config: DataSourceConfig,
        freshness_check: FreshnessCheck
    ) -> DataAlert:
        """品質アラート評価
        
        Args:
            config: データソース設定
            freshness_check: 鮮度チェック結果
            
        Returns:
            品質アラート、問題がない場合はNone
        """
        if (
            freshness_check.quality_score is None
            or freshness_check.quality_score >= config.quality_threshold
        ):
            return None
        
        # 品質スコアに基づく重要度判定
        quality_score = freshness_check.quality_score
        threshold = config.quality_threshold
        
        deviation = threshold - quality_score
        if deviation > 40:  # 40ポイント以上の差
            severity = AlertSeverity.CRITICAL
        elif deviation > 20:  # 20-40ポイントの差
            severity = AlertSeverity.ERROR
        else:  # 20ポイント以下の差
            severity = AlertSeverity.WARNING
        
        return DataAlert(
            alert_id=f"quality_{config.source_id}_{int(time.time())}",
            source_id=config.source_id,
            severity=severity,
            alert_type="data_quality",
            message=f"データ品質が閾値を下回っています: {quality_score:.1f} < {threshold}",
            timestamp=freshness_check.timestamp,
            metadata={
                "quality_score": quality_score,
                "threshold": threshold,
                "deviation": deviation,
            },
        )
    
    def _evaluate_composite_alert(
        self,
        config: DataSourceConfig,
        freshness_check: FreshnessCheck,
        integrity_checks: List[IntegrityCheck]
    ) -> DataAlert:
        """複合アラート評価
        
        複数の問題が同時に発生している場合の統合アラート
        
        Args:
            config: データソース設定
            freshness_check: 鮮度チェック結果
            integrity_checks: 整合性チェック結果リスト
            
        Returns:
            複合アラート、単独問題の場合はNone
        """
        issues = []
        
        # 鮮度問題
        if freshness_check.status != FreshnessStatus.FRESH:
            issues.append(f"鮮度: {freshness_check.status.value}")
        
        # 整合性問題
        failed_checks = [check for check in integrity_checks if not check.passed]
        if failed_checks:
            issues.append(f"整合性: {len(failed_checks)}個の問題")
        
        # 品質問題
        if (
            freshness_check.quality_score is not None
            and freshness_check.quality_score < config.quality_threshold
        ):
            issues.append(f"品質: {freshness_check.quality_score:.1f}")
        
        # 複数問題がある場合のみ複合アラートを生成
        if len(issues) < 2:
            return None
        
        return DataAlert(
            alert_id=f"composite_{config.source_id}_{int(time.time())}",
            source_id=config.source_id,
            severity=AlertSeverity.CRITICAL,
            alert_type="composite_issue",
            message=f"複数のデータ問題を検出: {', '.join(issues)}",
            timestamp=freshness_check.timestamp,
            metadata={
                "issue_count": len(issues),
                "issues": issues,
                "freshness_status": freshness_check.status.value,
                "integrity_failures": len(failed_checks),
                "quality_score": freshness_check.quality_score,
            },
        )
    
    async def _generate_and_notify_alert(
        self, 
        alert: DataAlert, 
        config: DataSourceConfig
    ):
        """アラート生成・通知
        
        Args:
            alert: データアラート
            config: データソース設定
        """
        try:
            # データベース保存
            await self.db_operations.save_alert(alert)
            
            # アクティブアラートに追加
            self.active_alerts[config.source_id].append(alert)
            self.stats.alerts_generated += 1
            
            # 外部異常検知システム連携
            if self.anomaly_system and alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                try:
                    await self._notify_anomaly_system(alert, config)
                except Exception as e:
                    self.logger.warning(f"異常検知システム通知エラー: {e}")
            
            # コールバック実行
            await self._execute_alert_callbacks(alert)
            
            # ログ出力
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨",
            }.get(alert.severity, "❓")
            
            self.logger.warning(
                f"{severity_emoji} アラート生成: {alert.alert_type} - {alert.message}"
            )
            
        except Exception as e:
            self.logger.error(f"アラート生成・通知エラー: {e}")
    
    async def _notify_anomaly_system(
        self, 
        alert: DataAlert, 
        config: DataSourceConfig
    ):
        """異常検知システム通知
        
        Args:
            alert: データアラート
            config: データソース設定
        """
        if not self.anomaly_system:
            return
        
        # 異常検知システム向けのデータ変換
        anomaly_data = {
            "source_id": alert.source_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity.value,
            "message": alert.message,
            "metadata": alert.metadata,
            "config": {
                "source_type": config.source_type.value,
                "monitoring_level": config.monitoring_level.value,
            }
        }
        
        # 異常検知システムにアラート送信（実装依存）
        # await self.anomaly_system.process_alert(anomaly_data)
    
    async def _execute_alert_callbacks(self, alert: DataAlert):
        """アラートコールバック実行
        
        Args:
            alert: データアラート
        """
        for callback in self.alert_callbacks:
            try:
                if hasattr(callback, '__call__'):
                    await callback(alert)
            except Exception as e:
                self.logger.error(f"アラートコールバックエラー: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """アラートコールバック追加
        
        Args:
            callback: アラート発生時に実行するコールバック関数
        """
        self.alert_callbacks.append(callback)
        self.logger.info("アラートコールバック追加")
    
    def remove_alert_callback(self, callback: Callable):
        """アラートコールバック削除
        
        Args:
            callback: 削除するコールバック関数
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            self.logger.info("アラートコールバック削除")
    
    async def resolve_alert(
        self, 
        alert_id: str, 
        resolution_notes: str = ""
    ):
        """アラート解決
        
        Args:
            alert_id: アラートID
            resolution_notes: 解決ノート
        """
        try:
            # データベース更新
            await self.db_operations.resolve_alert(alert_id, resolution_notes)
            
            # アクティブアラートから削除
            for source_alerts in self.active_alerts.values():
                for alert in source_alerts[:]:
                    if alert.alert_id == alert_id:
                        alert.resolved = True
                        alert.resolved_at = datetime.now(timezone.utc)
                        alert.resolution_notes = resolution_notes
                        source_alerts.remove(alert)
                        break
            
            self.logger.info(f"✅ アラート解決: {alert_id}")
            
        except Exception as e:
            self.logger.error(f"アラート解決エラー: {e}")
    
    def get_active_alerts(
        self, 
        source_id: str = None, 
        severity: AlertSeverity = None
    ) -> List[DataAlert]:
        """アクティブアラート取得
        
        Args:
            source_id: 特定のソースIDでフィルタ（オプション）
            severity: 特定の重要度でフィルタ（オプション）
            
        Returns:
            アクティブアラートのリスト
        """
        alerts = []
        
        sources = [source_id] if source_id else self.active_alerts.keys()
        
        for sid in sources:
            for alert in self.active_alerts[sid]:
                if not alert.resolved:
                    if severity is None or alert.severity == severity:
                        alerts.append(alert)
        
        # タイムスタンプで降順ソート
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """アラート統計取得
        
        Returns:
            アラート統計情報
        """
        return {
            "total_alerts_generated": self.stats.alerts_generated,
            "freshness_violations": self.stats.freshness_violations,
            "integrity_violations": self.stats.integrity_violations,
            "active_alerts_by_source": {
                source_id: len([a for a in alerts if not a.resolved])
                for source_id, alerts in self.active_alerts.items()
            },
            "active_alerts_by_severity": self._count_alerts_by_severity(),
        }
    
    def _count_alerts_by_severity(self) -> Dict[str, int]:
        """重要度別アクティブアラート数カウント
        
        Returns:
            重要度別のアクティブアラート数
        """
        count = {severity.value: 0 for severity in AlertSeverity}
        
        for alerts in self.active_alerts.values():
            for alert in alerts:
                if not alert.resolved:
                    count[alert.severity.value] += 1
        
        return count