#!/usr/bin/env python3
"""
Day Trade System - Core Module

リファクタリング済み統合コアシステム
"""

# バージョン情報
__version__ = "2.0.0"
__author__ = "Day Trade Development Team"
__description__ = "Unified and refactored core system for day trading platform"

# 既存コア機能（後方互換性のため保持）
try:
    from ..models.enums import AlertType
    from .alerts import AlertCondition, AlertManager, AlertPriority
    from .portfolio import PortfolioManager
    from .trade_manager import (
        Position, RealizedPnL, Trade, TradeManager, TradeStatus, TradeType,
    )
    from .watchlist import AlertNotification, WatchlistManager
    from .optimization_strategy import (
        OptimizationConfig, OptimizationLevel, OptimizationStrategy,
        OptimizationStrategyFactory, get_optimized_implementation,
        optimization_strategy,
    )
except ImportError:
    # 新しいリファクタリング版のみの場合
    pass

# 新しい統合システム
from .base import (
    BaseComponent, BaseConfig, HealthStatus, SystemStatus,
    ComponentRegistry, ConfigManager
)

from .enhanced_dependency_injection import (
    EnhancedDIContainer, LifecycleScope, InjectionStrategy,
    ServiceDescriptor, InjectionContext, injectable, inject,
    get_container, configure_container
)

from .common_infrastructure import (
    BaseStorage, InMemoryStorage, BaseTaskProcessor, BaseDataProcessor,
    SystemMetrics, TaskConfig, TaskResult, Priority, ProcessingMode,
    create_storage, create_task_processor, create_data_processor
)

from .system_integrations import (
    SystemOrchestrator, EventBus, CrossSystemDataBridge, IntegrationConfig,
    SystemState, IntegrationEvent, create_system_orchestrator,
    create_event_bus, create_data_bridge
)

from .unified_management import (
    UnifiedConfigManager, UnifiedMetricsCollector, ConfigSchema,
    ConfigEntry, MetricDefinition, MetricPoint, AlertRule,
    create_config_manager, create_metrics_collector
)

from .global_configuration import (
    SystemConfigSchema, GlobalConfigManager, get_global_config,
    initialize_global_config
)

from .unified_error_system import (
    UnifiedErrorHandler, ErrorContext, ErrorInfo, ErrorSeverity,
    ErrorCategory, RecoveryAction
)

from .standardized_error_handlers import (
    StandardizedErrorManager, SystemSpecificErrorHandler,
    DisasterRecoveryErrorHandler, MLAutoRetrainErrorHandler,
    HFTOptimizationErrorHandler, get_standardized_error_manager,
    initialize_error_handling
)

from .unified_logging_system import (
    UnifiedLoggingSystem, LogEntry, LoggingConfig, LogLevel,
    LogFormat, LogContext, StructuredLogFormatter,
    get_unified_logging_system, initialize_unified_logging,
    setup_component_logging, log_execution_time
)

from .system_adapters import (
    DisasterRecoveryAdapter, MLAutoRetrainAdapter, HFTOptimizationAdapter,
    SystemAdapterRegistry, setup_integrated_system
)

from .testing_framework import (
    TestResult, TestSuite, TestAssertion, MockManager, MockConfig,
    TestEnvironment, BaseTestCase, IntegrationTestCase, SystemTestRunner,
    test_environment, test_case, mock_service, get_test_runner
)

# 高度な企業級システム
from .performance_optimization_engine import (
    PerformanceOptimizationSystem, OptimizationLevel, OptimizationPolicy,
    PerformanceProfiler, MemoryProfiler, AutoTuner, CacheManager,
    get_performance_system, performance_optimized
)

from .distributed_microservices import (
    DistributedMicroservicesSystem, ServiceRegistry, LoadBalancer, MessageBroker,
    ServiceMesh, LoadBalancingStrategy, ServiceType, ServiceStatus,
    get_microservices_system, service_call, microservice
)

from .realtime_monitoring_dashboard import (
    RealtimeMonitoringSystem, DashboardManager, AlertManager as RealtimeAlertManager,
    MetricsCollector, WebSocketServer, SystemHealth, MetricType,
    get_monitoring_system, monitored
)

from .auto_scaling_load_balancer import (
    AutoScalingLoadBalancerSystem, AutoScaler, LoadBalancerManager,
    ScalingRule, ScalingMetric, HealthChecker, TrafficPredictor,
    get_autoscaling_system, auto_scalable
)

from .security_authentication_system import (
    SecurityAuthenticationSystem, JWTManager, ThreatDetectionManager,
    EncryptionManager, RateLimiter, AuthenticationProvider, SecurityPolicy,
    get_security_system, require_authentication, secure_endpoint
)

from .data_pipeline_etl_automation import (
    DataPipelineETLSystem, ETLPipeline, DataTransformer, DataQualityValidator,
    PipelineScheduler, DataSource, DataDestination, PipelineConfig,
    get_etl_system, etl_pipeline
)

from .ai_ml_prediction_integration import (
    AIMLPredictionSystem, ModelManager, PredictionService, BaseMLModel,
    LinearRegressionModel, TimeSeriesModel, EnsembleModel, ModelType,
    get_aiml_system, ml_model
)

from .cloud_native_architecture import (
    CloudNativeSystem, ContainerOrchestrator, ServiceMesh as CloudServiceMesh,
    KubernetesManager, ServiceManifest, DeploymentStrategy, CloudProvider,
    get_cloud_native_system, cloud_native_service
)

from .performance import (
    MetricType, PerformanceMetric, PerformanceMonitor,
    OptimizedBuffer, AdaptiveCache, MemoryOptimizer, PerformanceOptimizer,
    performance_optimized, get_global_performance_optimizer
)

from .monitoring import (
    UnifiedMonitoringSystem, get_global_monitoring_system, monitor_performance,
    IntegrationConfig, MonitoringIntegrationManager, initialize_integrated_monitoring,
    start_dashboard_server, stop_dashboard_server
)

# エクスポートリスト（後方互換性 + 新機能）
__all__ = [
    # バージョン情報
    '__version__', '__author__', '__description__',
    
    # 既存機能（後方互換性）
    "TradeManager", "Trade", "Position", "RealizedPnL", "TradeType", "TradeStatus",
    "WatchlistManager", "AlertType", "AlertCondition", "AlertNotification",
    "AlertPriority", "AlertManager", "PortfolioManager",
    "OptimizationLevel", "OptimizationConfig", "OptimizationStrategy",
    "OptimizationStrategyFactory", "optimization_strategy", "get_optimized_implementation",
    
    # 新しい統合システム
    'BaseComponent', 'BaseConfig', 'HealthStatus', 'SystemStatus',
    'ComponentRegistry', 'ConfigManager',
    'EnhancedDIContainer', 'LifecycleScope', 'InjectionStrategy',
    'ServiceDescriptor', 'InjectionContext', 'injectable', 'inject',
    'get_container', 'configure_container',
    'BaseStorage', 'InMemoryStorage', 'BaseTaskProcessor', 'BaseDataProcessor',
    'SystemMetrics', 'TaskConfig', 'TaskResult', 'Priority', 'ProcessingMode',
    'create_storage', 'create_task_processor', 'create_data_processor',
    'SystemOrchestrator', 'EventBus', 'CrossSystemDataBridge', 'IntegrationConfig',
    'SystemState', 'IntegrationEvent', 'create_system_orchestrator',
    'create_event_bus', 'create_data_bridge',
    'UnifiedConfigManager', 'UnifiedMetricsCollector', 'ConfigSchema',
    'ConfigEntry', 'MetricDefinition', 'MetricPoint', 'AlertRule',
    'create_config_manager', 'create_metrics_collector',
    'SystemConfigSchema', 'GlobalConfigManager', 'get_global_config',
    'initialize_global_config',
    'UnifiedErrorHandler', 'ErrorContext', 'ErrorInfo', 'ErrorSeverity',
    'ErrorCategory', 'RecoveryAction', 'StandardizedErrorManager',
    'SystemSpecificErrorHandler', 'DisasterRecoveryErrorHandler',
    'MLAutoRetrainErrorHandler', 'HFTOptimizationErrorHandler',
    'get_standardized_error_manager', 'initialize_error_handling',
    'UnifiedLoggingSystem', 'LogEntry', 'LoggingConfig', 'LogLevel',
    'LogFormat', 'LogContext', 'StructuredLogFormatter',
    'get_unified_logging_system', 'initialize_unified_logging',
    'setup_component_logging', 'log_execution_time',
    'DisasterRecoveryAdapter', 'MLAutoRetrainAdapter', 'HFTOptimizationAdapter',
    'SystemAdapterRegistry', 'setup_integrated_system',
    'TestResult', 'TestSuite', 'TestAssertion', 'MockManager', 'MockConfig',
    'TestEnvironment', 'BaseTestCase', 'IntegrationTestCase', 'SystemTestRunner',
    'test_environment', 'test_case', 'mock_service', 'get_test_runner',
    
    # 高度な企業級システム
    'PerformanceOptimizationSystem', 'OptimizationLevel', 'OptimizationPolicy',
    'PerformanceProfiler', 'MemoryProfiler', 'AutoTuner', 'CacheManager',
    'get_performance_system', 'performance_optimized',
    'DistributedMicroservicesSystem', 'ServiceRegistry', 'LoadBalancer', 'MessageBroker',
    'ServiceMesh', 'LoadBalancingStrategy', 'ServiceType', 'ServiceStatus',
    'get_microservices_system', 'service_call', 'microservice',
    'RealtimeMonitoringSystem', 'DashboardManager', 'RealtimeAlertManager',
    'MetricsCollector', 'WebSocketServer', 'SystemHealth', 'MetricType',
    'get_monitoring_system', 'monitored',
    'AutoScalingLoadBalancerSystem', 'AutoScaler', 'LoadBalancerManager',
    'ScalingRule', 'ScalingMetric', 'HealthChecker', 'TrafficPredictor',
    'get_autoscaling_system', 'auto_scalable',
    'SecurityAuthenticationSystem', 'JWTManager', 'ThreatDetectionManager',
    'EncryptionManager', 'RateLimiter', 'AuthenticationProvider', 'SecurityPolicy',
    'get_security_system', 'require_authentication', 'secure_endpoint',
    'DataPipelineETLSystem', 'ETLPipeline', 'DataTransformer', 'DataQualityValidator',
    'PipelineScheduler', 'DataSource', 'DataDestination', 'PipelineConfig',
    'get_etl_system', 'etl_pipeline',
    'AIMLPredictionSystem', 'ModelManager', 'PredictionService', 'BaseMLModel',
    'LinearRegressionModel', 'TimeSeriesModel', 'EnsembleModel', 'ModelType',
    'get_aiml_system', 'ml_model',
    'CloudNativeSystem', 'ContainerOrchestrator', 'CloudServiceMesh',
    'KubernetesManager', 'ServiceManifest', 'DeploymentStrategy', 'CloudProvider',
    'get_cloud_native_system', 'cloud_native_service',
    
    # パフォーマンス最適化
    'PerformanceMetric', 'PerformanceMonitor', 'OptimizedBuffer',
    'AdaptiveCache', 'MemoryOptimizer', 'PerformanceOptimizer',
    'performance_optimized', 'get_global_performance_optimizer',
    
    # 統合監視システム
    'UnifiedMonitoringSystem', 'get_global_monitoring_system', 'monitor_performance',
    'IntegrationConfig', 'MonitoringIntegrationManager', 'initialize_integrated_monitoring',
    'start_dashboard_server', 'stop_dashboard_server',
]


async def initialize_core_system(config: SystemConfigSchema = None) -> SystemOrchestrator:
    """
    コアシステム初期化
    
    リファクタリング済み統合システムの完全初期化を実行します。
    """
    # 1. DIコンテナ初期化
    container = get_container()
    await container.start()
    
    # 2. グローバル設定初期化
    global_config = get_global_config()
    config_manager = create_config_manager(BaseConfig(name="global_config"))
    await global_config.initialize(config_manager)
    
    # 3. ログシステム初期化
    logging_config = LoggingConfig(
        level=LogLevel.INFO,
        format=LogFormat.STRUCTURED,
        output_file="logs/day_trade_system.log",
        console_enabled=True,
        file_enabled=True
    )
    logging_system = await initialize_unified_logging(logging_config)
    
    # 4. エラーハンドリング初期化
    error_manager = await initialize_error_handling()
    
    # 5. メトリクス収集初期化
    metrics_collector = create_metrics_collector(BaseConfig(name="global_metrics"))
    await metrics_collector.start()
    
    # 6. システム統合・オーケストレーション
    orchestrator = await setup_integrated_system()
    
    # 7. システム開始
    await orchestrator.start()
    
    # ログ出力
    system_logger = setup_component_logging("core_system", logging_system)
    system_logger.info("Day Trade Core System initialized successfully")
    system_logger.info(f"Version: {__version__}")
    system_logger.info("All systems operational")
    
    return orchestrator
