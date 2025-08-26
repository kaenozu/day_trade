#!/usr/bin/env python3
"""
メイン監視システム
全コンポーネントを統合した高度データ鮮度・整合性監視システムのメインモジュール
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .alert_manager import AlertManager
from .dashboard import DashboardManager
from .database_operations import DatabaseOperations
from .enums import MonitoringLevel
from .freshness_checker import FreshnessChecker
from .integrity_checker import IntegrityChecker
from .models import DataSourceConfig, DashboardData, MonitoringStats
from .recovery_manager import RecoveryManager
from .sla_metrics import SLAMetricsCalculator

# 依存コンポーネントのインポート
try:
    from ...monitoring.advanced_anomaly_detection_alerts import (
        AdvancedAnomalyAlertSystem,
    )
    from ...monitoring.structured_logging_enhancement import (
        StructuredLoggingEnhancementSystem,
    )
    from ...utils.unified_cache_manager import UnifiedCacheManager
    from ..comprehensive_data_quality_system import ComprehensiveDataQualitySystem
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


class AdvancedFreshnessMonitor:
    """高度データ鮮度・整合性監視システム
    
    データソースの鮮度、整合性、品質を包括的に監視し、
    自動回復、アラート管理、SLAメトリクス計算、ダッシュボード機能を提供します。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """監視システムを初期化
        
        Args:
            config_path: 設定ファイルのパス（オプション）
        """
        self.logger = logging.getLogger(__name__)
        
        # 設定管理
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # コアコンポーネント初期化
        self.db_operations = DatabaseOperations()
        self.freshness_checker = FreshnessChecker()
        self.integrity_checker = IntegrityChecker(self.db_operations)
        self.alert_manager = AlertManager(self.db_operations)
        self.recovery_manager = RecoveryManager(self.db_operations)
        self.sla_calculator = SLAMetricsCalculator(self.db_operations)
        self.dashboard_manager = DashboardManager(self.db_operations)
        
        # 監視データ（インメモリ）
        self.recent_checks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_stats = MonitoringStats()
        
        # 外部システム連携
        self.quality_system = None
        self.anomaly_detector = None
        self.cache_manager = None
        
        # コンポーネント初期化
        self._initialize_external_components()
        
        # 設定読み込み
        if config_path:
            self.load_config(config_path)
        
        self.logger.info("高度データ鮮度監視システム初期化完了")
    
    def _initialize_external_components(self):
        """外部コンポーネント初期化"""
        if not DEPENDENCIES_AVAILABLE:
            self.logger.warning("一部の依存コンポーネントが利用できません")
            return
        
        try:
            self.quality_system = ComprehensiveDataQualitySystem()
            self.anomaly_detector = AdvancedAnomalyAlertSystem()
            self.cache_manager = UnifiedCacheManager()
            self.logger.info("外部コンポーネント初期化完了")
        except Exception as e:
            self.logger.warning(f"外部コンポーネント初期化エラー: {e}")
    
    def add_data_source(self, config: DataSourceConfig):
        """データソース追加
        
        Args:
            config: データソース設定
        """
        self.data_sources[config.source_id] = config
        
        # データソース状態初期化
        from .models import DataSourceState
        initial_state = DataSourceState(
            source_id=config.source_id,
            current_status="unknown",
            consecutive_failures=0,
            recovery_attempts=0,
            metadata={
                "source_type": config.source_type.value,
                "monitoring_level": config.monitoring_level.value,
                "sla_target": config.sla_target,
            }
        )
        
        asyncio.create_task(
            self.db_operations.update_source_state(
                config.source_id, initial_state
            )
        )
        
        self.logger.info(f"データソース追加: {config.source_id}")
    
    def remove_data_source(self, source_id: str):
        """データソース削除
        
        Args:
            source_id: データソースID
        """
        if source_id in self.data_sources:
            del self.data_sources[source_id]
            
            # アクティブアラートを解決
            active_alerts = self.alert_manager.get_active_alerts(source_id)
            for alert in active_alerts:
                asyncio.create_task(
                    self.alert_manager.resolve_alert(
                        alert.alert_id, 
                        "データソース削除により自動解決"
                    )
                )
            
            self.logger.info(f"データソース削除: {source_id}")
    
    def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            self.logger.warning("監視は既に実行中です")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("🔍 データ鮮度・整合性監視開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("⏹️ データ鮮度・整合性監視停止")
    
    def _monitoring_loop(self):
        """監視ループ（スレッド実行）"""
        self.logger.info("監視ループ開始")
        
        while self.monitoring_active:
            try:
                # 各データソースをチェック
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                for source_config in self.data_sources.values():
                    loop.run_until_complete(
                        self._check_data_source(source_config)
                    )
                
                # SLAメトリクス計算（毎時0分）
                current_time = datetime.now(timezone.utc)
                if current_time.minute == 0:
                    loop.run_until_complete(
                        self.sla_calculator.calculate_hourly_sla_metrics(
                            self.data_sources
                        )
                    )
                
                loop.close()
                
                # 10秒待機
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(30)  # エラー時は長めに待機
        
        self.logger.info("監視ループ終了")
    
    async def _check_data_source(self, config: DataSourceConfig):
        """データソースチェック
        
        Args:
            config: データソース設定
        """
        try:
            source_id = config.source_id
            current_time = datetime.now(timezone.utc)
            
            # 鮮度チェック実行
            freshness_result = await self.freshness_checker.check_freshness(config)
            
            # 整合性チェック実行（監視レベルに応じて）
            integrity_results = []
            if config.monitoring_level in [
                MonitoringLevel.COMPREHENSIVE,
                MonitoringLevel.CRITICAL,
            ]:
                integrity_results = await self.integrity_checker.check_integrity(config)
            
            # 結果保存
            await self.db_operations.save_freshness_check(freshness_result)
            for integrity_result in integrity_results:
                await self.db_operations.save_integrity_check(integrity_result)
            
            # インメモリデータ更新
            self.recent_checks[source_id].append(freshness_result)
            self.monitoring_stats.total_checks_performed += 1
            
            # アラート評価
            await self.alert_manager.evaluate_alerts(
                config, freshness_result, integrity_results
            )
            
            # データソース状態更新
            await self._update_source_state(
                config, freshness_result, integrity_results
            )
            
            # 回復アクション評価
            await self.recovery_manager.evaluate_recovery_actions(
                config, freshness_result, integrity_results
            )
            
        except Exception as e:
            self.logger.error(f"データソースチェックエラー ({config.source_id}): {e}")
    
    async def _update_source_state(
        self,
        config: DataSourceConfig,
        freshness_result,
        integrity_results: List
    ):
        """データソース状態更新
        
        Args:
            config: データソース設定
            freshness_result: 鮮度チェック結果
            integrity_results: 整合性チェック結果リスト
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # 現在の状態取得
            source_state = self.db_operations.get_source_state(config.source_id)
            if not source_state:
                from .models import DataSourceState
                source_state = DataSourceState(
                    source_id=config.source_id,
                    consecutive_failures=0,
                    recovery_attempts=0
                )
            
            # 総合健全性判定
            from .enums import FreshnessStatus
            is_healthy = (
                freshness_result.status == FreshnessStatus.FRESH and
                all(check.passed for check in integrity_results)
            )
            
            # 状態更新
            if is_healthy:
                source_state.consecutive_failures = 0
                source_state.current_status = "healthy"
                source_state.last_success = current_time
            else:
                source_state.consecutive_failures += 1
                source_state.current_status = "unhealthy"
            
            source_state.last_check = current_time
            
            await self.db_operations.update_source_state(
                config.source_id, source_state
            )
            
        except Exception as e:
            self.logger.error(f"データソース状態更新エラー ({config.source_id}): {e}")
    
    async def get_monitoring_dashboard(self, hours: int = 24) -> DashboardData:
        """監視ダッシュボードデータ取得
        
        Args:
            hours: 表示対象期間（時間）
            
        Returns:
            ダッシュボードデータ
        """
        try:
            return await self.dashboard_manager.get_monitoring_dashboard(
                self.data_sources, hours
            )
        except Exception as e:
            self.logger.error(f"ダッシュボードデータ取得エラー: {e}")
            return DashboardData(
                overview={"error": str(e)},
                source_summary=[],
                recent_alerts=[],
                sla_summary=[],
                generated_at=datetime.now(timezone.utc).isoformat(),
                time_range_hours=hours,
                error=str(e),
            )
    
    async def generate_health_report(self, period_days: int = 7) -> Dict[str, Any]:
        """健全性レポート生成
        
        Args:
            period_days: レポート期間（日数）
            
        Returns:
            健全性レポート
        """
        try:
            return await self.dashboard_manager.generate_health_report(
                self.data_sources, period_days
            )
        except Exception as e:
            self.logger.error(f"健全性レポート生成エラー: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """システム統計情報取得
        
        Returns:
            統合システム統計情報
        """
        return {
            "monitoring_stats": self.monitoring_stats.to_dict(),
            "alert_stats": self.alert_manager.get_alert_statistics(),
            "recovery_stats": self.recovery_manager.get_recovery_statistics(),
            "data_sources": {
                "total_count": len(self.data_sources),
                "by_type": self._count_sources_by_type(),
                "by_monitoring_level": self._count_sources_by_monitoring_level(),
            },
            "system_status": {
                "monitoring_active": self.monitoring_active,
                "external_components": {
                    "quality_system": self.quality_system is not None,
                    "anomaly_detector": self.anomaly_detector is not None,
                    "cache_manager": self.cache_manager is not None,
                },
            },
        }
    
    def _count_sources_by_type(self) -> Dict[str, int]:
        """データソース種別別カウント"""
        count = {}
        for config in self.data_sources.values():
            source_type = config.source_type.value
            count[source_type] = count.get(source_type, 0) + 1
        return count
    
    def _count_sources_by_monitoring_level(self) -> Dict[str, int]:
        """監視レベル別カウント"""
        count = {}
        for config in self.data_sources.values():
            level = config.monitoring_level.value
            count[level] = count.get(level, 0) + 1
        return count
    
    def add_alert_callback(self, callback: Callable):
        """アラートコールバック追加
        
        Args:
            callback: アラート発生時に実行するコールバック関数
        """
        self.alert_manager.add_alert_callback(callback)
    
    def add_recovery_callback(self, action, callback: Callable):
        """回復アクションコールバック追加
        
        Args:
            action: 回復アクション種別
            callback: 実行するコールバック関数
        """
        self.recovery_manager.add_recovery_callback(action, callback)
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """アラート解決
        
        Args:
            alert_id: アラートID
            resolution_notes: 解決ノート
        """
        await self.alert_manager.resolve_alert(alert_id, resolution_notes)
    
    def load_config(self, config_path: str):
        """設定ファイル読み込み
        
        Args:
            config_path: 設定ファイルのパス
        """
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
            
            for source_data in config_data.get("data_sources", []):
                from .enums import DataSourceType, MonitoringLevel, RecoveryAction
                
                config = DataSourceConfig(
                    source_id=source_data["source_id"],
                    source_type=DataSourceType(source_data["source_type"]),
                    endpoint_url=source_data.get("endpoint_url"),
                    connection_params=source_data.get("connection_params", {}),
                    expected_frequency=source_data.get("expected_frequency", 60),
                    freshness_threshold=source_data.get("freshness_threshold", 300),
                    quality_threshold=source_data.get("quality_threshold", 80.0),
                    monitoring_level=MonitoringLevel(
                        source_data.get("monitoring_level", "standard")
                    ),
                    enable_recovery=source_data.get("enable_recovery", True),
                    recovery_strategy=RecoveryAction(
                        source_data.get("recovery_strategy", "retry")
                    ),
                    max_retry_attempts=source_data.get("max_retry_attempts", 3),
                    sla_target=source_data.get("sla_target", 99.9),
                )
                self.add_data_source(config)
            
            self.logger.info(
                f"設定読み込み完了: {len(self.data_sources)}個のデータソース"
            )
            
        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")


# Factory function
def create_advanced_freshness_monitor(
    config_path: Optional[str] = None,
) -> AdvancedFreshnessMonitor:
    """高度データ鮮度監視システム作成
    
    Args:
        config_path: 設定ファイルのパス（オプション）
        
    Returns:
        高度データ鮮度監視システムのインスタンス
    """
    return AdvancedFreshnessMonitor(config_path)