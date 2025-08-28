#!/usr/bin/env python3
"""
標準化エラーハンドラー

システム固有のエラーハンドラーと統合エラー管理を提供します。
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import threading

from .unified_system_error import (
    UnifiedSystemError, ValidationError, SecurityError, ErrorSeverity, 
    ErrorCategory, ErrorRecoveryAction
)

# 後方互換性のため
RecoveryAction = ErrorRecoveryAction
from .base import BaseComponent, BaseConfig, HealthStatus, SystemStatus

logger = logging.getLogger(__name__)


@dataclass
class SystemErrorProfile:
    """システムエラープロファイル"""
    system_name: str
    error_patterns: Dict[str, ErrorCategory] = field(default_factory=dict)
    recovery_strategies: Dict[ErrorCategory, RecoveryAction] = field(default_factory=dict)
    retry_configs: Dict[ErrorCategory, Dict[str, Any]] = field(default_factory=dict)
    escalation_thresholds: Dict[ErrorSeverity, int] = field(default_factory=dict)
    custom_handlers: Dict[str, Callable] = field(default_factory=dict)


class SystemSpecificErrorHandler(ABC):
    """システム固有エラーハンドラー基底クラス"""
    
    def __init__(self, system_name: str, error_profile: SystemErrorProfile):
        self.system_name = system_name
        self.error_profile = error_profile
        self._error_counts: Dict[str, int] = {}
        self._last_errors: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    @abstractmethod
    async def handle_system_error(self, error_info: ErrorInfo) -> bool:
        """システム固有エラー処理"""
        pass
    
    @abstractmethod
    def categorize_error(self, exception: Exception, context: ErrorContext) -> ErrorCategory:
        """エラーカテゴリ分類"""
        pass
    
    @abstractmethod
    def determine_severity(self, exception: Exception, context: ErrorContext) -> ErrorSeverity:
        """エラー重要度判定"""
        pass
    
    def get_recovery_action(self, category: ErrorCategory) -> RecoveryAction:
        """復旧アクション決定"""
        return self.error_profile.recovery_strategies.get(category, RecoveryAction.ESCALATE)
    
    def should_retry(self, category: ErrorCategory, attempt_count: int) -> bool:
        """リトライ判定"""
        retry_config = self.error_profile.retry_configs.get(category, {})
        max_attempts = retry_config.get('max_attempts', 3)
        return attempt_count < max_attempts
    
    def get_retry_delay(self, category: ErrorCategory, attempt_count: int) -> float:
        """リトライ遅延計算"""
        retry_config = self.error_profile.retry_configs.get(category, {})
        base_delay = retry_config.get('base_delay', 1.0)
        backoff_factor = retry_config.get('backoff_factor', 2.0)
        max_delay = retry_config.get('max_delay', 60.0)
        
        delay = base_delay * (backoff_factor ** attempt_count)
        return min(delay, max_delay)
    
    async def track_error(self, error_info: ErrorInfo):
        """エラー追跡"""
        with self._lock:
            error_key = f"{error_info.category.value}_{type(error_info.exception).__name__}"
            self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
            self._last_errors[error_key] = time.time()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """エラー統計取得"""
        with self._lock:
            return {
                'error_counts': self._error_counts.copy(),
                'last_errors': self._last_errors.copy(),
                'total_errors': sum(self._error_counts.values())
            }


class DisasterRecoveryErrorHandler(SystemSpecificErrorHandler):
    """災害復旧システム用エラーハンドラー"""
    
    def __init__(self):
        error_profile = SystemErrorProfile(
            system_name="disaster_recovery",
            error_patterns={
                "BackupError": ErrorCategory.SYSTEM,
                "StorageError": ErrorCategory.SYSTEM,
                "CompressionError": ErrorCategory.SYSTEM,
                "NetworkError": ErrorCategory.NETWORK,
                "PermissionError": ErrorCategory.AUTHORIZATION
            },
            recovery_strategies={
                ErrorCategory.SYSTEM: RecoveryAction.RETRY,
                ErrorCategory.NETWORK: RecoveryAction.RETRY,
                ErrorCategory.AUTHORIZATION: RecoveryAction.ESCALATE,
                ErrorCategory.DATABASE: RecoveryAction.RETRY
            },
            retry_configs={
                ErrorCategory.SYSTEM: {
                    'max_attempts': 3,
                    'base_delay': 2.0,
                    'backoff_factor': 2.0,
                    'max_delay': 30.0
                },
                ErrorCategory.NETWORK: {
                    'max_attempts': 5,
                    'base_delay': 1.0,
                    'backoff_factor': 1.5,
                    'max_delay': 15.0
                }
            }
        )
        super().__init__("disaster_recovery", error_profile)
    
    async def handle_system_error(self, error_info: ErrorInfo) -> bool:
        """災害復旧システムエラー処理"""
        await self.track_error(error_info)
        
        if error_info.category == ErrorCategory.SYSTEM:
            return await self._handle_system_error(error_info)
        elif error_info.category == ErrorCategory.NETWORK:
            return await self._handle_network_error(error_info)
        elif error_info.category == ErrorCategory.AUTHORIZATION:
            return await self._handle_permission_error(error_info)
        else:
            return False
    
    def categorize_error(self, exception: Exception, context: ErrorContext) -> ErrorCategory:
        """エラーカテゴリ分類"""
        exception_name = type(exception).__name__
        
        if exception_name in self.error_profile.error_patterns:
            return self.error_profile.error_patterns[exception_name]
        elif "permission" in str(exception).lower():
            return ErrorCategory.AUTHORIZATION
        elif "network" in str(exception).lower() or "connection" in str(exception).lower():
            return ErrorCategory.NETWORK
        elif "storage" in str(exception).lower() or "disk" in str(exception).lower():
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN
    
    def determine_severity(self, exception: Exception, context: ErrorContext) -> ErrorSeverity:
        """エラー重要度判定"""
        exception_str = str(exception).lower()
        
        if "critical" in exception_str or "fatal" in exception_str:
            return ErrorSeverity.CRITICAL
        elif "backup" in context.operation and "failed" in exception_str:
            return ErrorSeverity.HIGH
        elif isinstance(exception, PermissionError):
            return ErrorSeverity.HIGH
        elif "network" in exception_str:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    async def _handle_system_error(self, error_info: ErrorInfo) -> bool:
        """システムエラー処理"""
        logger.warning(f"Disaster Recovery system error: {error_info.message}")
        
        # バックアップ処理の場合は代替ディレクトリを試行
        if "backup" in error_info.context.operation:
            # 代替バックアップ処理（実装省略）
            return True
        
        return False
    
    async def _handle_network_error(self, error_info: ErrorInfo) -> bool:
        """ネットワークエラー処理"""
        logger.info(f"Disaster Recovery network error, will retry: {error_info.message}")
        return True  # リトライ可能
    
    async def _handle_permission_error(self, error_info: ErrorInfo) -> bool:
        """権限エラー処理"""
        logger.error(f"Disaster Recovery permission error: {error_info.message}")
        
        # 管理者への通知（実装省略）
        return False  # 自動復旧不可


class MLAutoRetrainErrorHandler(SystemSpecificErrorHandler):
    """機械学習自動再学習用エラーハンドラー"""
    
    def __init__(self):
        error_profile = SystemErrorProfile(
            system_name="ml_auto_retrain",
            error_patterns={
                "ModelLoadError": ErrorCategory.SYSTEM,
                "DataValidationError": ErrorCategory.VALIDATION,
                "TrainingError": ErrorCategory.BUSINESS_LOGIC,
                "MemoryError": ErrorCategory.SYSTEM,
                "TimeoutError": ErrorCategory.SYSTEM
            },
            recovery_strategies={
                ErrorCategory.VALIDATION: RecoveryAction.FALLBACK,
                ErrorCategory.BUSINESS_LOGIC: RecoveryAction.RETRY,
                ErrorCategory.SYSTEM: RecoveryAction.RETRY,
                ErrorCategory.DATABASE: RecoveryAction.RETRY
            },
            retry_configs={
                ErrorCategory.BUSINESS_LOGIC: {
                    'max_attempts': 2,
                    'base_delay': 5.0,
                    'backoff_factor': 1.5,
                    'max_delay': 30.0
                },
                ErrorCategory.SYSTEM: {
                    'max_attempts': 3,
                    'base_delay': 10.0,
                    'backoff_factor': 2.0,
                    'max_delay': 120.0
                }
            }
        )
        super().__init__("ml_auto_retrain", error_profile)
    
    async def handle_system_error(self, error_info: ErrorInfo) -> bool:
        """ML システムエラー処理"""
        await self.track_error(error_info)
        
        if error_info.category == ErrorCategory.VALIDATION:
            return await self._handle_data_validation_error(error_info)
        elif error_info.category == ErrorCategory.BUSINESS_LOGIC:
            return await self._handle_training_error(error_info)
        elif error_info.category == ErrorCategory.SYSTEM:
            return await self._handle_resource_error(error_info)
        else:
            return False
    
    def categorize_error(self, exception: Exception, context: ErrorContext) -> ErrorCategory:
        """エラーカテゴリ分類"""
        exception_name = type(exception).__name__
        exception_str = str(exception).lower()
        
        if exception_name in self.error_profile.error_patterns:
            return self.error_profile.error_patterns[exception_name]
        elif "validation" in exception_str or "invalid" in exception_str:
            return ErrorCategory.VALIDATION
        elif "training" in exception_str or "model" in exception_str:
            return ErrorCategory.BUSINESS_LOGIC
        elif "memory" in exception_str or "resource" in exception_str:
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN
    
    def determine_severity(self, exception: Exception, context: ErrorContext) -> ErrorSeverity:
        """エラー重要度判定"""
        exception_str = str(exception).lower()
        
        if isinstance(exception, MemoryError):
            return ErrorSeverity.CRITICAL
        elif "training" in context.operation and "failed" in exception_str:
            return ErrorSeverity.HIGH
        elif "validation" in exception_str:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    async def _handle_data_validation_error(self, error_info: ErrorInfo) -> bool:
        """データ検証エラー処理"""
        logger.warning(f"ML data validation error: {error_info.message}")
        
        # 代替データセットまたはクリーニング処理（実装省略）
        return True
    
    async def _handle_training_error(self, error_info: ErrorInfo) -> bool:
        """学習エラー処理"""
        logger.info(f"ML training error, will retry with reduced parameters: {error_info.message}")
        
        # パラメーター調整して再試行（実装省略）
        return True
    
    async def _handle_resource_error(self, error_info: ErrorInfo) -> bool:
        """リソースエラー処理"""
        logger.warning(f"ML resource error: {error_info.message}")
        
        # バッチサイズ削減やガベージコレクション（実装省略）
        return True


class HFTOptimizationErrorHandler(SystemSpecificErrorHandler):
    """高頻度取引最適化用エラーハンドラー"""
    
    def __init__(self):
        error_profile = SystemErrorProfile(
            system_name="hft_optimization",
            error_patterns={
                "OrderError": ErrorCategory.BUSINESS_LOGIC,
                "MarketDataError": ErrorCategory.EXTERNAL_API,
                "LatencyError": ErrorCategory.NETWORK,
                "RiskError": ErrorCategory.BUSINESS_LOGIC,
                "ConnectionError": ErrorCategory.NETWORK
            },
            recovery_strategies={
                ErrorCategory.BUSINESS_LOGIC: RecoveryAction.FALLBACK,
                ErrorCategory.EXTERNAL_API: RecoveryAction.RETRY,
                ErrorCategory.NETWORK: RecoveryAction.RETRY,
                ErrorCategory.SYSTEM: RecoveryAction.ESCALATE
            },
            retry_configs={
                ErrorCategory.EXTERNAL_API: {
                    'max_attempts': 5,
                    'base_delay': 0.1,
                    'backoff_factor': 1.2,
                    'max_delay': 1.0
                },
                ErrorCategory.NETWORK: {
                    'max_attempts': 10,
                    'base_delay': 0.05,
                    'backoff_factor': 1.1,
                    'max_delay': 0.5
                }
            }
        )
        super().__init__("hft_optimization", error_profile)
    
    async def handle_system_error(self, error_info: ErrorInfo) -> bool:
        """HFT システムエラー処理"""
        await self.track_error(error_info)
        
        if error_info.category == ErrorCategory.BUSINESS_LOGIC:
            return await self._handle_trading_error(error_info)
        elif error_info.category == ErrorCategory.EXTERNAL_API:
            return await self._handle_market_data_error(error_info)
        elif error_info.category == ErrorCategory.NETWORK:
            return await self._handle_latency_error(error_info)
        else:
            return False
    
    def categorize_error(self, exception: Exception, context: ErrorContext) -> ErrorCategory:
        """エラーカテゴリ分類"""
        exception_name = type(exception).__name__
        exception_str = str(exception).lower()
        
        if exception_name in self.error_profile.error_patterns:
            return self.error_profile.error_patterns[exception_name]
        elif "order" in exception_str or "trade" in exception_str:
            return ErrorCategory.BUSINESS_LOGIC
        elif "market" in exception_str or "price" in exception_str:
            return ErrorCategory.EXTERNAL_API
        elif "latency" in exception_str or "timeout" in exception_str:
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.UNKNOWN
    
    def determine_severity(self, exception: Exception, context: ErrorContext) -> ErrorSeverity:
        """エラー重要度判定"""
        exception_str = str(exception).lower()
        
        if "risk" in exception_str or "loss" in exception_str:
            return ErrorSeverity.CRITICAL
        elif "order" in context.operation and "rejected" in exception_str:
            return ErrorSeverity.HIGH
        elif "latency" in exception_str:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    async def _handle_trading_error(self, error_info: ErrorInfo) -> bool:
        """取引エラー処理"""
        logger.warning(f"HFT trading error: {error_info.message}")
        
        # ポジション制限やリスクチェック強化（実装省略）
        return False  # 安全のため自動復旧しない
    
    async def _handle_market_data_error(self, error_info: ErrorInfo) -> bool:
        """マーケットデータエラー処理"""
        logger.info(f"HFT market data error, will retry: {error_info.message}")
        
        # 代替データソース（実装省略）
        return True
    
    async def _handle_latency_error(self, error_info: ErrorInfo) -> bool:
        """レイテンシエラー処理"""
        logger.info(f"HFT latency error: {error_info.message}")
        
        # ルート最適化やキャッシュクリア（実装省略）
        return True


class StandardizedErrorManager(BaseComponent):
    """標準化エラー管理"""
    
    def __init__(self, name: str = "standardized_error_manager"):
        super().__init__(name, BaseConfig(name=name))
        self.unified_handler = UnifiedErrorHandler()
        self.system_handlers: Dict[str, SystemSpecificErrorHandler] = {}
        self.error_routing: Dict[str, str] = {}  # component -> handler_name
        self._lock = threading.RLock()
    
    async def start(self) -> bool:
        """エラー管理開始"""
        try:
            await self.unified_handler.initialize()
            
            # システム固有ハンドラー登録
            self._register_system_handlers()
            
            # 統一ハンドラーへの登録
            self.unified_handler.add_handler(self._route_to_system_handler)
            
            self.status = SystemStatus.RUNNING
            logger.info("Standardized Error Manager started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Standardized Error Manager: {e}")
            return False
    
    async def stop(self) -> bool:
        """エラー管理停止"""
        try:
            self.status = SystemStatus.STOPPING
            self.status = SystemStatus.STOPPED
            return True
        except Exception as e:
            logger.error(f"Failed to stop Standardized Error Manager: {e}")
            return False
    
    def _register_system_handlers(self):
        """システム固有ハンドラー登録"""
        with self._lock:
            # 災害復旧
            dr_handler = DisasterRecoveryErrorHandler()
            self.system_handlers["disaster_recovery"] = dr_handler
            self.error_routing["disaster_recovery_adapter"] = "disaster_recovery"
            self.error_routing["backup_system"] = "disaster_recovery"
            
            # ML自動再学習
            ml_handler = MLAutoRetrainErrorHandler()
            self.system_handlers["ml_auto_retrain"] = ml_handler
            self.error_routing["ml_auto_retrain_adapter"] = "ml_auto_retrain"
            self.error_routing["ml_system"] = "ml_auto_retrain"
            
            # HFT最適化
            hft_handler = HFTOptimizationErrorHandler()
            self.system_handlers["hft_optimization"] = hft_handler
            self.error_routing["hft_optimization_adapter"] = "hft_optimization"
            self.error_routing["trading_engine"] = "hft_optimization"
            
            logger.info(f"Registered {len(self.system_handlers)} system error handlers")
    
    async def _route_to_system_handler(self, error_info: ErrorInfo) -> bool:
        """システム固有ハンドラーへルーティング"""
        component = error_info.context.component
        
        # ルーティング先決定
        handler_name = self.error_routing.get(component)
        if not handler_name or handler_name not in self.system_handlers:
            return False
        
        # システム固有ハンドラー実行
        system_handler = self.system_handlers[handler_name]
        
        try:
            # エラーカテゴリと重要度を再判定
            error_info.category = system_handler.categorize_error(
                error_info.exception, error_info.context
            )
            error_info.severity = system_handler.determine_severity(
                error_info.exception, error_info.context
            )
            
            # システム固有処理
            handled = await system_handler.handle_system_error(error_info)
            
            logger.debug(f"System handler {handler_name} processed error: {handled}")
            return handled
            
        except Exception as e:
            logger.error(f"System handler {handler_name} failed: {e}")
            return False
    
    def register_component_routing(self, component_name: str, handler_name: str):
        """コンポーネントルーティング登録"""
        with self._lock:
            if handler_name not in self.system_handlers:
                logger.warning(f"Handler {handler_name} not found for component {component_name}")
                return
            
            self.error_routing[component_name] = handler_name
            logger.info(f"Error routing registered: {component_name} -> {handler_name}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """エラー統計取得"""
        statistics = {}
        
        with self._lock:
            for handler_name, handler in self.system_handlers.items():
                statistics[handler_name] = handler.get_error_statistics()
        
        return statistics
    
    async def health_check(self) -> HealthStatus:
        """健全性チェック"""
        try:
            if self.status != SystemStatus.RUNNING:
                return HealthStatus.UNHEALTHY
            
            # 各ハンドラーのエラー統計確認
            for handler_name, handler in self.system_handlers.items():
                stats = handler.get_error_statistics()
                total_errors = stats.get('total_errors', 0)
                
                # エラー率が高い場合は劣化状態
                if total_errors > 100:  # しきい値
                    return HealthStatus.DEGRADED
            
            return HealthStatus.HEALTHY
            
        except Exception:
            return HealthStatus.UNHEALTHY


# グローバルエラー管理インスタンス
_global_error_manager: Optional[StandardizedErrorManager] = None


def get_standardized_error_manager() -> StandardizedErrorManager:
    """標準化エラー管理取得"""
    global _global_error_manager
    if _global_error_manager is None:
        _global_error_manager = StandardizedErrorManager()
    return _global_error_manager


async def initialize_error_handling():
    """エラーハンドリング初期化"""
    manager = get_standardized_error_manager()
    await manager.start()
    return manager