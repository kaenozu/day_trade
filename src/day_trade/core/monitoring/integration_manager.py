"""
統合監視システム統合管理器

各コンポーネントとの統合を管理する。
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

from .unified_monitoring_system import UnifiedMonitoringSystem, get_global_monitoring_system
from .dashboard_server import DashboardServer, start_dashboard_server
from ..config.configuration_manager import ConfigurationManager
from ..security.security_manager import SecurityManager
from ..performance.performance_optimizer import PerformanceMonitor, get_global_performance_optimizer
from ..microservices_architecture import ServiceRegistry
from ..event_driven_architecture import EventBus


@dataclass
class IntegrationConfig:
    """統合設定"""
    enable_dashboard: bool = True
    dashboard_host: str = 'localhost'
    dashboard_port: int = 8080
    enable_microservices_monitoring: bool = True
    enable_event_monitoring: bool = True
    enable_performance_integration: bool = True
    enable_security_monitoring: bool = True
    auto_start: bool = True
    # セキュリティ設定
    require_authentication: bool = False
    max_connections: int = 100
    rate_limit_requests_per_minute: int = 1000


class MonitoringIntegrationManager:
    """監視システム統合管理器"""
    
    def __init__(self, config: Optional[IntegrationConfig] = None,
                 config_manager: Optional[ConfigurationManager] = None,
                 security_manager: Optional[SecurityManager] = None):
        self.config = config or IntegrationConfig()
        self.config_manager = config_manager
        self.security_manager = security_manager
        
        # コア監視システム
        self.monitoring_system = get_global_monitoring_system()
        self.dashboard_server: Optional[DashboardServer] = None
        
        # 統合コンポーネント
        self.performance_monitor = get_global_performance_optimizer()
        self.service_registry: Optional[ServiceRegistry] = None
        self.event_bus: Optional[EventBus] = None
        
        # 統合状態
        self.integration_status = {
            "monitoring_system": False,
            "dashboard_server": False,
            "microservices": False,
            "event_system": False,
            "performance": False,
            "security": False
        }
        
        # 統合タスク
        self.integration_tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """統合初期化"""
        logging.info("監視システム統合を初期化します")
        
        try:
            # コア監視システム初期化
            await self._initialize_monitoring_system()
            
            # パフォーマンス監視統合
            if self.config.enable_performance_integration:
                await self._integrate_performance_monitoring()
                
            # セキュリティ監視統合
            if self.config.enable_security_monitoring:
                await self._integrate_security_monitoring()
                
            # マイクロサービス監視統合
            if self.config.enable_microservices_monitoring:
                await self._integrate_microservices_monitoring()
                
            # イベント監視統合
            if self.config.enable_event_monitoring:
                await self._integrate_event_monitoring()
                
            # ダッシュボードサーバー
            if self.config.enable_dashboard:
                await self._initialize_dashboard()
                
            logging.info("監視システム統合初期化が完了しました")
            
        except Exception as e:
            logging.error(f"監視システム統合初期化エラー: {e}")
            raise
            
    async def _initialize_monitoring_system(self):
        """監視システム初期化"""
        try:
            # 追加のカスタムメトリクス設定
            self._setup_custom_metrics()
            
            # カスタムアラートルール設定
            self._setup_custom_alert_rules()
            
            # 監視システム開始
            await self.monitoring_system.start()
            
            self.integration_status["monitoring_system"] = True
            logging.info("コア監視システムが初期化されました")
            
        except Exception as e:
            logging.error(f"監視システム初期化エラー: {e}")
            raise
            
    def _setup_custom_metrics(self):
        """カスタムメトリクス設定"""
        # ビジネス固有のメトリクス
        custom_metrics = [
            ("business.active_positions", "GAUGE", "アクティブポジション数"),
            ("business.daily_volume", "COUNTER", "日次取引量"),
            ("business.risk_score", "GAUGE", "リスクスコア"),
            ("system.database.connections", "GAUGE", "データベース接続数"),
            ("system.cache.hit_rate", "GAUGE", "キャッシュヒット率"),
            ("system.queue.length", "GAUGE", "キュー長"),
        ]
        
        for name, metric_type, description in custom_metrics:
            self.monitoring_system.metrics_storage.create_metric(
                name, getattr(self.monitoring_system.metrics_storage.MetricType, metric_type), description
            )
            
    def _setup_custom_alert_rules(self):
        """カスタムアラートルール設定"""
        from .unified_monitoring_system import ThresholdAlertRule, AlertLevel
        
        # ビジネス固有のアラート
        custom_rules = [
            ("リスクスコア警告", AlertLevel.WARNING, "business.risk_score", 80.0, "gt"),
            ("リスクスコア危険", AlertLevel.CRITICAL, "business.risk_score", 95.0, "gt"),
            ("データベース接続不足", AlertLevel.ERROR, "system.database.connections", 5.0, "lt"),
            ("キャッシュヒット率低下", AlertLevel.WARNING, "system.cache.hit_rate", 70.0, "lt"),
        ]
        
        for name, level, metric_name, threshold, comparison in custom_rules:
            rule = ThresholdAlertRule(name, level, metric_name, threshold, comparison)
            self.monitoring_system.alert_manager.add_alert_rule(rule)
            
    async def _integrate_performance_monitoring(self):
        """パフォーマンス監視統合"""
        try:
            if not hasattr(self.performance_monitor, 'get_performance_metrics'):
                logging.warning("パフォーマンスモニターが利用できません")
                return
                
            # パフォーマンスメトリクスの定期収集タスク
            async def collect_performance_metrics():
                while True:
                    try:
                        # パフォーマンス統計取得
                        if hasattr(self.performance_monitor, 'get_statistics'):
                            stats = self.performance_monitor.get_statistics()
                            
                            # メトリクス記録
                            for key, value in stats.items():
                                metric_name = f"performance.{key}"
                                self.monitoring_system.record_business_metric(metric_name, value)
                                
                    except Exception as e:
                        logging.error(f"パフォーマンスメトリクス収集エラー: {e}")
                        
                    await asyncio.sleep(60)  # 1分間隔
                    
            task = asyncio.create_task(collect_performance_metrics())
            self.integration_tasks.append(task)
            
            self.integration_status["performance"] = True
            logging.info("パフォーマンス監視統合が完了しました")
            
        except Exception as e:
            logging.error(f"パフォーマンス監視統合エラー: {e}")
            
    async def _integrate_security_monitoring(self):
        """セキュリティ監視統合"""
        try:
            if not self.security_manager:
                logging.warning("セキュリティマネージャーが利用できません")
                return
                
            # セキュリティイベント監視
            async def monitor_security_events():
                while True:
                    try:
                        # セキュリティ統計取得（例）
                        if hasattr(self.security_manager, 'get_security_stats'):
                            stats = self.security_manager.get_security_stats()
                            
                            # セキュリティメトリクス記録
                            for key, value in stats.items():
                                metric_name = f"security.{key}"
                                self.monitoring_system.record_business_metric(metric_name, value)
                                
                    except Exception as e:
                        logging.error(f"セキュリティメトリクス収集エラー: {e}")
                        
                    await asyncio.sleep(30)  # 30秒間隔
                    
            task = asyncio.create_task(monitor_security_events())
            self.integration_tasks.append(task)
            
            self.integration_status["security"] = True
            logging.info("セキュリティ監視統合が完了しました")
            
        except Exception as e:
            logging.error(f"セキュリティ監視統合エラー: {e}")
            
    async def _integrate_microservices_monitoring(self):
        """マイクロサービス監視統合"""
        try:
            # サービスレジストリとの統合
            async def monitor_services():
                while True:
                    try:
                        if self.service_registry:
                            # サービス統計収集
                            services = getattr(self.service_registry, 'get_all_services', lambda: {})()
                            
                            # サービス数記録
                            self.monitoring_system.record_business_metric(
                                "microservices.active_services", len(services)
                            )
                            
                            # 各サービスの健全性チェック
                            for service_name, instances in services.items():
                                healthy_count = sum(1 for instance in instances.values() 
                                                 if getattr(instance, 'is_healthy', lambda: True)())
                                
                                self.monitoring_system.record_business_metric(
                                    f"microservices.{service_name}.healthy_instances", healthy_count
                                )
                                
                    except Exception as e:
                        logging.error(f"マイクロサービスメトリクス収集エラー: {e}")
                        
                    await asyncio.sleep(45)  # 45秒間隔
                    
            task = asyncio.create_task(monitor_services())
            self.integration_tasks.append(task)
            
            self.integration_status["microservices"] = True
            logging.info("マイクロサービス監視統合が完了しました")
            
        except Exception as e:
            logging.error(f"マイクロサービス監視統合エラー: {e}")
            
    async def _integrate_event_monitoring(self):
        """イベント監視統合"""
        try:
            # イベントバスとの統合
            async def monitor_events():
                while True:
                    try:
                        if self.event_bus:
                            # イベント統計収集
                            if hasattr(self.event_bus, 'get_event_stats'):
                                stats = self.event_bus.get_event_stats()
                                
                                # イベントメトリクス記録
                                for event_type, count in stats.items():
                                    metric_name = f"events.{event_type}.count"
                                    self.monitoring_system.record_business_metric(metric_name, count)
                                    
                    except Exception as e:
                        logging.error(f"イベントメトリクス収集エラー: {e}")
                        
                    await asyncio.sleep(30)  # 30秒間隔
                    
            task = asyncio.create_task(monitor_events())
            self.integration_tasks.append(task)
            
            self.integration_status["event_system"] = True
            logging.info("イベント監視統合が完了しました")
            
        except Exception as e:
            logging.error(f"イベント監視統合エラー: {e}")
            
    async def _initialize_dashboard(self):
        """ダッシュボード初期化"""
        try:
            # セキュリティ設定
            security_config = {
                "require_authentication": self.config.require_authentication,
                "max_connections": self.config.max_connections,
                "rate_limit_requests_per_minute": self.config.rate_limit_requests_per_minute
            }
            
            self.dashboard_server = await start_dashboard_server(
                host=self.config.dashboard_host,
                port=self.config.dashboard_port,
                monitoring_system=self.monitoring_system,
                security_config=security_config
            )
            
            self.integration_status["dashboard_server"] = True
            logging.info(f"ダッシュボードサーバーが開始されました: http://{self.config.dashboard_host}:{self.config.dashboard_port}")
            
        except Exception as e:
            logging.error(f"ダッシュボード初期化エラー: {e}")
            raise
            
    def set_service_registry(self, service_registry: 'ServiceRegistry'):
        """サービスレジストリ設定"""
        self.service_registry = service_registry
        
    def set_event_bus(self, event_bus: 'EventBus'):
        """イベントバス設定"""
        self.event_bus = event_bus
        
    def get_integration_status(self) -> Dict[str, Any]:
        """統合状態取得"""
        return {
            "status": self.integration_status.copy(),
            "dashboard_url": (f"http://{self.config.dashboard_host}:{self.config.dashboard_port}"
                            if self.integration_status["dashboard_server"] else None),
            "monitoring_metrics": self.monitoring_system.get_monitoring_status(),
            "active_integrations": sum(1 for status in self.integration_status.values() if status),
            "total_integrations": len(self.integration_status)
        }
        
    async def shutdown(self):
        """統合システム停止"""
        logging.info("監視システム統合を停止します")
        
        try:
            # 統合タスク停止
            for task in self.integration_tasks:
                task.cancel()
                
            await asyncio.gather(*self.integration_tasks, return_exceptions=True)
            self.integration_tasks.clear()
            
            # ダッシュボードサーバー停止
            if self.dashboard_server:
                await self.dashboard_server.stop()
                
            # 監視システム停止
            await self.monitoring_system.stop()
            
            # 統合状態リセット
            for key in self.integration_status:
                self.integration_status[key] = False
                
            logging.info("監視システム統合停止が完了しました")
            
        except Exception as e:
            logging.error(f"監視システム統合停止エラー: {e}")


# グローバル統合管理器
_global_integration_manager: Optional[MonitoringIntegrationManager] = None


def get_global_integration_manager() -> MonitoringIntegrationManager:
    """グローバル統合管理器取得"""
    global _global_integration_manager
    if _global_integration_manager is None:
        _global_integration_manager = MonitoringIntegrationManager()
    return _global_integration_manager


async def initialize_integrated_monitoring(config: Optional[IntegrationConfig] = None,
                                          config_manager: Optional[ConfigurationManager] = None,
                                          security_manager: Optional[SecurityManager] = None) -> MonitoringIntegrationManager:
    """統合監視システム初期化"""
    global _global_integration_manager
    
    if _global_integration_manager is None:
        _global_integration_manager = MonitoringIntegrationManager(
            config, config_manager, security_manager
        )
        
    await _global_integration_manager.initialize()
    return _global_integration_manager


async def shutdown_integrated_monitoring():
    """統合監視システム停止"""
    global _global_integration_manager
    
    if _global_integration_manager:
        await _global_integration_manager.shutdown()
        _global_integration_manager = None