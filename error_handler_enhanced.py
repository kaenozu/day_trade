#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Error Handler - 強化エラーハンドリングシステム
Issue #946対応: 例外処理統一 + 自動リカバリ + エラー分析
"""

import sys
import traceback
import logging
import functools
import asyncio
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import inspect

# エラー通知用（オプショナル）
try:
    import smtplib
    from email.mime.text import MIMEText
    HAS_EMAIL = True
except ImportError:
    HAS_EMAIL = False


class ErrorSeverity(Enum):
    """エラー重要度"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """エラーカテゴリ"""
    NETWORK = "NETWORK"
    DATABASE = "DATABASE"
    AI_MODEL = "AI_MODEL"
    AUTHENTICATION = "AUTHENTICATION"
    VALIDATION = "VALIDATION"
    SYSTEM = "SYSTEM"
    BUSINESS_LOGIC = "BUSINESS_LOGIC"
    EXTERNAL_API = "EXTERNAL_API"


class RecoveryStrategy(Enum):
    """リカバリ戦略"""
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    IGNORE = "IGNORE"
    ESCALATE = "ESCALATE"
    RESTART_COMPONENT = "RESTART_COMPONENT"


@dataclass
class ErrorEvent:
    """エラーイベント"""
    error_id: str
    timestamp: datetime
    exception_type: str
    message: str
    stack_trace: str
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    function_name: str
    user_id: Optional[str]
    request_id: Optional[str]
    context: Dict[str, Any]
    recovery_attempted: bool
    recovery_success: bool
    recovery_strategy: Optional[RecoveryStrategy]
    retry_count: int


@dataclass
class RecoveryAction:
    """リカバリアクション"""
    strategy: RecoveryStrategy
    max_retries: int
    retry_delay: float
    backoff_multiplier: float
    timeout: float
    fallback_function: Optional[Callable]
    escalation_threshold: int


class ErrorClassifier:
    """エラー分類システム"""
    
    def __init__(self):
        self.classification_rules = {
            # ネットワークエラー
            ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.HIGH),
            TimeoutError: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            
            # データベースエラー
            Exception: (ErrorCategory.DATABASE, ErrorSeverity.HIGH),  # SQLite等
            
            # バリデーションエラー
            ValueError: (ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            TypeError: (ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            
            # システムエラー
            MemoryError: (ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL),
            OSError: (ErrorCategory.SYSTEM, ErrorSeverity.HIGH),
            
            # 一般的なエラー
            RuntimeError: (ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.MEDIUM),
            KeyError: (ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.LOW),
            IndexError: (ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.LOW),
        }
        
        # メッセージベース分類
        self.message_patterns = {
            'api': ErrorCategory.EXTERNAL_API,
            'auth': ErrorCategory.AUTHENTICATION,
            'model': ErrorCategory.AI_MODEL,
            'network': ErrorCategory.NETWORK,
            'database': ErrorCategory.DATABASE,
        }
    
    def classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> tuple[ErrorCategory, ErrorSeverity]:
        """エラー分類"""
        # 例外タイプベース分類
        exc_type = type(exception)
        if exc_type in self.classification_rules:
            return self.classification_rules[exc_type]
        
        # メッセージベース分類
        error_msg = str(exception).lower()
        for pattern, category in self.message_patterns.items():
            if pattern in error_msg:
                severity = self._infer_severity_from_message(error_msg)
                return category, severity
        
        # コンテキストベース分類
        if context:
            component = context.get('component', '').lower()
            if 'ai' in component or 'model' in component:
                return ErrorCategory.AI_MODEL, ErrorSeverity.MEDIUM
            elif 'database' in component or 'db' in component:
                return ErrorCategory.DATABASE, ErrorSeverity.HIGH
        
        # デフォルト分類
        return ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM
    
    def _infer_severity_from_message(self, message: str) -> ErrorSeverity:
        """メッセージから重要度推定"""
        critical_keywords = ['crash', 'fatal', 'critical', 'severe']
        high_keywords = ['failed', 'error', 'exception', 'timeout']
        low_keywords = ['warning', 'info', 'notice']
        
        if any(keyword in message for keyword in critical_keywords):
            return ErrorSeverity.CRITICAL
        elif any(keyword in message for keyword in high_keywords):
            return ErrorSeverity.HIGH
        elif any(keyword in message for keyword in low_keywords):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM


class ErrorRecoveryManager:
    """エラーリカバリ管理"""
    
    def __init__(self):
        self.recovery_strategies: Dict[ErrorCategory, RecoveryAction] = {
            ErrorCategory.NETWORK: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_retries=3,
                retry_delay=1.0,
                backoff_multiplier=2.0,
                timeout=30.0,
                fallback_function=None,
                escalation_threshold=5
            ),
            ErrorCategory.DATABASE: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_retries=2,
                retry_delay=0.5,
                backoff_multiplier=1.5,
                timeout=10.0,
                fallback_function=None,
                escalation_threshold=3
            ),
            ErrorCategory.AI_MODEL: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                max_retries=1,
                retry_delay=0.1,
                backoff_multiplier=1.0,
                timeout=5.0,
                fallback_function=self._ai_model_fallback,
                escalation_threshold=10
            ),
            ErrorCategory.EXTERNAL_API: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_retries=3,
                retry_delay=2.0,
                backoff_multiplier=2.0,
                timeout=60.0,
                fallback_function=None,
                escalation_threshold=5
            ),
            ErrorCategory.VALIDATION: RecoveryAction(
                strategy=RecoveryStrategy.IGNORE,
                max_retries=0,
                retry_delay=0.0,
                backoff_multiplier=1.0,
                timeout=0.0,
                fallback_function=None,
                escalation_threshold=1
            ),
            ErrorCategory.SYSTEM: RecoveryAction(
                strategy=RecoveryStrategy.ESCALATE,
                max_retries=1,
                retry_delay=0.1,
                backoff_multiplier=1.0,
                timeout=5.0,
                fallback_function=None,
                escalation_threshold=1
            )
        }
        
        self.component_restart_functions: Dict[str, Callable] = {}
        self.escalation_handlers: List[Callable] = []
    
    def register_component_restart(self, component_name: str, restart_func: Callable):
        """コンポーネント再起動関数登録"""
        self.component_restart_functions[component_name] = restart_func
    
    def register_escalation_handler(self, handler: Callable):
        """エスカレーション処理登録"""
        self.escalation_handlers.append(handler)
    
    async def execute_recovery(self, error_event: ErrorEvent, target_function: Callable, *args, **kwargs) -> Any:
        """リカバリ実行"""
        recovery_action = self.recovery_strategies.get(
            error_event.category,
            self.recovery_strategies[ErrorCategory.SYSTEM]
        )
        
        if recovery_action.strategy == RecoveryStrategy.RETRY:
            return await self._execute_retry(error_event, target_function, recovery_action, *args, **kwargs)
        elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
            return await self._execute_fallback(error_event, target_function, recovery_action, *args, **kwargs)
        elif recovery_action.strategy == RecoveryStrategy.RESTART_COMPONENT:
            return await self._execute_restart(error_event, target_function, recovery_action, *args, **kwargs)
        elif recovery_action.strategy == RecoveryStrategy.ESCALATE:
            await self._execute_escalation(error_event)
            raise Exception(f"Escalated error: {error_event.message}")
        else:  # IGNORE
            logging.warning(f"Ignoring error: {error_event.message}")
            return None
    
    async def _execute_retry(self, error_event: ErrorEvent, target_function: Callable, 
                           recovery_action: RecoveryAction, *args, **kwargs) -> Any:
        """リトライ実行"""
        retry_delay = recovery_action.retry_delay
        
        for attempt in range(recovery_action.max_retries):
            try:
                await asyncio.sleep(retry_delay)
                
                if asyncio.iscoroutinefunction(target_function):
                    result = await target_function(*args, **kwargs)
                else:
                    result = target_function(*args, **kwargs)
                
                error_event.recovery_success = True
                error_event.retry_count = attempt + 1
                logging.info(f"Recovery successful after {attempt + 1} retries")
                return result
                
            except Exception as e:
                retry_delay *= recovery_action.backoff_multiplier
                logging.warning(f"Retry {attempt + 1} failed: {e}")
                
                if attempt == recovery_action.max_retries - 1:
                    error_event.recovery_success = False
                    error_event.retry_count = recovery_action.max_retries
                    raise e
    
    async def _execute_fallback(self, error_event: ErrorEvent, target_function: Callable,
                              recovery_action: RecoveryAction, *args, **kwargs) -> Any:
        """フォールバック実行"""
        if recovery_action.fallback_function:
            try:
                if asyncio.iscoroutinefunction(recovery_action.fallback_function):
                    result = await recovery_action.fallback_function(*args, **kwargs)
                else:
                    result = recovery_action.fallback_function(*args, **kwargs)
                
                error_event.recovery_success = True
                logging.info(f"Fallback function executed successfully")
                return result
                
            except Exception as e:
                error_event.recovery_success = False
                logging.error(f"Fallback function failed: {e}")
                raise e
        else:
            error_event.recovery_success = False
            logging.error("No fallback function available")
            raise Exception("No fallback available")
    
    async def _execute_restart(self, error_event: ErrorEvent, target_function: Callable,
                             recovery_action: RecoveryAction, *args, **kwargs) -> Any:
        """コンポーネント再起動実行"""
        component = error_event.component
        
        if component in self.component_restart_functions:
            try:
                restart_func = self.component_restart_functions[component]
                
                if asyncio.iscoroutinefunction(restart_func):
                    await restart_func()
                else:
                    restart_func()
                
                # 再起動後に元の関数を実行
                await asyncio.sleep(1.0)  # 起動待ち
                
                if asyncio.iscoroutinefunction(target_function):
                    result = await target_function(*args, **kwargs)
                else:
                    result = target_function(*args, **kwargs)
                
                error_event.recovery_success = True
                logging.info(f"Component {component} restarted successfully")
                return result
                
            except Exception as e:
                error_event.recovery_success = False
                logging.error(f"Component restart failed: {e}")
                raise e
        else:
            error_event.recovery_success = False
            logging.error(f"No restart function for component: {component}")
            raise Exception(f"Cannot restart component: {component}")
    
    async def _execute_escalation(self, error_event: ErrorEvent):
        """エスカレーション実行"""
        for handler in self.escalation_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_event)
                else:
                    handler(error_event)
            except Exception as e:
                logging.error(f"Escalation handler failed: {e}")
    
    def _ai_model_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """AI モデルフォールバック関数"""
        return {
            'signal_type': 'HOLD',
            'confidence': 0.5,
            'strength': 0.5,
            'risk_level': 'MEDIUM',
            'fallback': True,
            'message': 'AI model unavailable, using fallback'
        }


class ErrorAnalytics:
    """エラー分析システム"""
    
    def __init__(self):
        self.error_patterns: Dict[str, List[ErrorEvent]] = defaultdict(list)
        self.error_frequency: Dict[str, int] = defaultdict(int)
        self.recovery_success_rate: Dict[ErrorCategory, float] = {}
        
    def analyze_error_patterns(self, error_events: List[ErrorEvent]) -> Dict[str, Any]:
        """エラーパターン分析"""
        # エラー頻度分析
        category_frequency = defaultdict(int)
        severity_frequency = defaultdict(int)
        component_frequency = defaultdict(int)
        
        for event in error_events:
            category_frequency[event.category.value] += 1
            severity_frequency[event.severity.value] += 1
            component_frequency[event.component] += 1
        
        # 時系列分析
        time_series = self._analyze_time_series(error_events)
        
        # 成功率分析
        success_rates = self._analyze_recovery_success_rates(error_events)
        
        return {
            'total_errors': len(error_events),
            'category_distribution': dict(category_frequency),
            'severity_distribution': dict(severity_frequency),
            'component_distribution': dict(component_frequency),
            'time_series_analysis': time_series,
            'recovery_success_rates': success_rates,
            'recommendations': self._generate_recommendations(error_events)
        }
    
    def _analyze_time_series(self, error_events: List[ErrorEvent]) -> Dict[str, Any]:
        """時系列分析"""
        if not error_events:
            return {}
        
        # 時間別エラー数
        hourly_counts = defaultdict(int)
        for event in error_events:
            hour = event.timestamp.hour
            hourly_counts[hour] += 1
        
        # トレンド分析
        recent_events = [e for e in error_events if 
                        (datetime.now() - e.timestamp).days <= 7]
        
        return {
            'hourly_distribution': dict(hourly_counts),
            'recent_errors_count': len(recent_events),
            'peak_error_hour': max(hourly_counts, key=hourly_counts.get) if hourly_counts else 0
        }
    
    def _analyze_recovery_success_rates(self, error_events: List[ErrorEvent]) -> Dict[str, float]:
        """リカバリ成功率分析"""
        category_stats = defaultdict(lambda: {'total': 0, 'success': 0})
        
        for event in error_events:
            if event.recovery_attempted:
                category_stats[event.category.value]['total'] += 1
                if event.recovery_success:
                    category_stats[event.category.value]['success'] += 1
        
        success_rates = {}
        for category, stats in category_stats.items():
            if stats['total'] > 0:
                success_rates[category] = stats['success'] / stats['total']
        
        return success_rates
    
    def _generate_recommendations(self, error_events: List[ErrorEvent]) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []
        
        # 頻出エラー分析
        component_errors = defaultdict(int)
        for event in error_events:
            component_errors[event.component] += 1
        
        # 最も多いコンポーネント
        if component_errors:
            top_component = max(component_errors, key=component_errors.get)
            if component_errors[top_component] > 5:
                recommendations.append(f"Review and optimize {top_component} component (high error frequency)")
        
        # 重要度別推奨
        critical_errors = [e for e in error_events if e.severity == ErrorSeverity.CRITICAL]
        if critical_errors:
            recommendations.append("Immediate attention required for critical errors")
        
        # リカバリ成功率が低い
        success_rates = self._analyze_recovery_success_rates(error_events)
        for category, rate in success_rates.items():
            if rate < 0.5:
                recommendations.append(f"Improve recovery strategy for {category} errors")
        
        return recommendations


class EnhancedErrorHandler:
    """強化エラーハンドリングシステム"""
    
    def __init__(self):
        self.error_classifier = ErrorClassifier()
        self.recovery_manager = ErrorRecoveryManager()
        self.error_analytics = ErrorAnalytics()
        
        self.error_events: deque = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.last_error_time = {}
        
        # 設定
        self.duplicate_threshold = 5  # 同じエラーの重複閾値
        self.escalation_cooldown = 300  # 5分間のクールダウン
        
        # 通知設定
        self.notification_enabled = False
        self.notification_email = None
        
    def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorEvent:
        """エラー処理"""
        # コンテキスト準備
        if context is None:
            context = {}
        
        # 実行情報取得
        frame = inspect.currentframe().f_back
        function_name = frame.f_code.co_name if frame else 'unknown'
        component = context.get('component', self._infer_component_from_stack())
        
        # エラー分類
        category, severity = self.error_classifier.classify_error(exception, context)
        
        # エラーイベント作成
        error_event = ErrorEvent(
            error_id=self._generate_error_id(),
            timestamp=datetime.now(),
            exception_type=type(exception).__name__,
            message=str(exception),
            stack_trace=traceback.format_exc(),
            severity=severity,
            category=category,
            component=component,
            function_name=function_name,
            user_id=context.get('user_id'),
            request_id=context.get('request_id'),
            context=context,
            recovery_attempted=False,
            recovery_success=False,
            recovery_strategy=None,
            retry_count=0
        )
        
        # 重複チェック
        error_key = f"{error_event.exception_type}_{error_event.component}_{error_event.function_name}"
        self.error_counts[error_key] += 1
        
        if self._should_suppress_duplicate(error_key):
            logging.debug(f"Suppressing duplicate error: {error_key}")
            return error_event
        
        # ログ出力
        self._log_error_event(error_event)
        
        # エラー記録
        self.error_events.append(error_event)
        self.last_error_time[error_key] = datetime.now()
        
        # 通知送信
        if self.notification_enabled and severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_error_notification(error_event)
        
        return error_event
    
    def _infer_component_from_stack(self) -> str:
        """スタックトレースからコンポーネント推定"""
        stack = traceback.extract_stack()
        
        for frame in reversed(stack):
            filename = frame.filename.lower()
            if 'ai_engine' in filename:
                return 'ai_engine'
            elif 'quantum' in filename:
                return 'quantum_ai'
            elif 'risk' in filename:
                return 'risk_management'
            elif 'trading' in filename:
                return 'trading_engine'
            elif 'web' in filename:
                return 'web_ui'
        
        return 'system'
    
    def _generate_error_id(self) -> str:
        """エラーID生成"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"ERROR_{timestamp}_{hash(time.time()) % 10000:04d}"
    
    def _should_suppress_duplicate(self, error_key: str) -> bool:
        """重複エラー抑制判定"""
        if self.error_counts[error_key] <= self.duplicate_threshold:
            return False
        
        last_time = self.last_error_time.get(error_key)
        if last_time and (datetime.now() - last_time).seconds < 60:  # 1分以内
            return True
        
        return False
    
    def _log_error_event(self, error_event: ErrorEvent):
        """エラーイベントログ出力"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_event.severity, logging.ERROR)
        
        logging.log(
            log_level,
            f"[{error_event.severity.value}] {error_event.component}.{error_event.function_name}: "
            f"{error_event.message} (ID: {error_event.error_id})"
        )
    
    def _send_error_notification(self, error_event: ErrorEvent):
        """エラー通知送信"""
        if not HAS_EMAIL or not self.notification_email:
            return
        
        try:
            subject = f"[{error_event.severity.value}] Error in {error_event.component}"
            body = f"""
            Error Details:
            - ID: {error_event.error_id}
            - Time: {error_event.timestamp}
            - Component: {error_event.component}
            - Function: {error_event.function_name}
            - Message: {error_event.message}
            - Category: {error_event.category.value}
            
            Stack Trace:
            {error_event.stack_trace}
            """
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = 'daytrade@system.local'
            msg['To'] = self.notification_email
            
            # 実際のメール送信はスキップ（設定が必要）
            logging.info(f"Error notification would be sent: {subject}")
            
        except Exception as e:
            logging.error(f"Failed to send error notification: {e}")
    
    async def handle_with_recovery(self, target_function: Callable, *args, **kwargs) -> Any:
        """リカバリ付きエラー処理"""
        try:
            if asyncio.iscoroutinefunction(target_function):
                return await target_function(*args, **kwargs)
            else:
                return target_function(*args, **kwargs)
        
        except Exception as e:
            error_event = self.handle_error(e, {'function': target_function.__name__})
            
            # リカバリ実行
            error_event.recovery_attempted = True
            
            try:
                result = await self.recovery_manager.execute_recovery(
                    error_event, target_function, *args, **kwargs
                )
                return result
            
            except Exception as recovery_error:
                logging.error(f"Recovery failed: {recovery_error}")
                raise recovery_error
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """エラーサマリー取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_events if e.timestamp > cutoff_time]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'period_hours': hours,
                'analysis': {}
            }
        
        analysis = self.error_analytics.analyze_error_patterns(recent_errors)
        
        return {
            'total_errors': len(recent_errors),
            'period_hours': hours,
            'analysis': analysis,
            'top_errors': self._get_top_errors(recent_errors)
        }
    
    def _get_top_errors(self, error_events: List[ErrorEvent], limit: int = 5) -> List[Dict[str, Any]]:
        """上位エラー取得"""
        error_groups = defaultdict(list)
        
        for event in error_events:
            key = f"{event.exception_type}_{event.component}"
            error_groups[key].append(event)
        
        top_errors = sorted(
            error_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:limit]
        
        return [
            {
                'error_type': key.split('_')[0],
                'component': key.split('_', 1)[1] if '_' in key else 'unknown',
                'count': len(events),
                'latest_occurrence': max(e.timestamp for e in events).isoformat(),
                'recovery_rate': sum(1 for e in events if e.recovery_success) / len(events)
            }
            for key, events in top_errors
        ]


# グローバルインスタンス
enhanced_error_handler = EnhancedErrorHandler()


def handle_errors(component: str = None):
    """エラーハンドリングデコレータ"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = {'component': component or func.__module__, 'function': func.__name__}
            return await enhanced_error_handler.handle_with_recovery(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = {'component': component or func.__module__, 'function': func.__name__}
            try:
                return func(*args, **kwargs)
            except Exception as e:
                enhanced_error_handler.handle_error(e, context)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_error_summary(hours: int = 24) -> Dict[str, Any]:
    """エラーサマリー取得（便利関数）"""
    return enhanced_error_handler.get_error_summary(hours)


async def test_error_handler():
    """エラーハンドラーテスト"""
    print("=== Enhanced Error Handler Test ===")
    
    @handle_errors(component="test_module")
    def test_function_with_error():
        """テスト関数（エラーありバージョン）"""
        raise ValueError("This is a test error")
    
    @handle_errors(component="test_module")
    def test_function_normal():
        """テスト関数（正常バージョン）"""
        return "Success"
    
    # 正常動作テスト
    result = test_function_normal()
    print(f"Normal function result: {result}")
    
    # エラー処理テスト
    try:
        test_function_with_error()
    except ValueError as e:
        print(f"Caught error: {e}")
    
    # リカバリ付き処理テスト
    async def recoverable_function():
        """リカバリテスト用関数"""
        import random
        if random.random() < 0.7:  # 70%の確率でエラー
            raise ConnectionError("Network error")
        return "Network success"
    
    try:
        result = await enhanced_error_handler.handle_with_recovery(recoverable_function)
        print(f"Recovery result: {result}")
    except Exception as e:
        print(f"Recovery failed: {e}")
    
    # エラーサマリー
    summary = enhanced_error_handler.get_error_summary()
    print(f"Error summary: {summary['total_errors']} errors in last 24h")
    
    if summary['analysis']:
        print("Error categories:", summary['analysis'].get('category_distribution', {}))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_error_handler())