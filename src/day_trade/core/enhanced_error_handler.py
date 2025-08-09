#!/usr/bin/env python3
"""
強化エラーハンドリングシステム
Phase E: ユーザーエクスペリエンス強化

多言語対応・詳細診断・自動復旧機能付きエラー処理
"""

import json
import traceback
import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .optimization_strategy import OptimizationConfig
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ErrorSeverity(Enum):
    """エラー重要度"""
    LOW = "low"           # 軽微なエラー（継続可能）
    MEDIUM = "medium"     # 中程度（機能制限）
    HIGH = "high"         # 重要（処理停止）
    CRITICAL = "critical" # 致命的（システム停止）


class ErrorCategory(Enum):
    """エラーカテゴリ"""
    DATA_ERROR = "data_error"                 # データ関連エラー
    NETWORK_ERROR = "network_error"           # ネットワークエラー
    COMPUTATION_ERROR = "computation_error"   # 計算エラー
    CONFIGURATION_ERROR = "config_error"      # 設定エラー
    SYSTEM_ERROR = "system_error"            # システムエラー
    USER_INPUT_ERROR = "user_input_error"    # ユーザー入力エラー


@dataclass
class ErrorContext:
    """エラーコンテキスト情報"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    error_message: str
    technical_details: str
    user_impact: str
    suggested_actions: List[str]
    auto_recovery_attempted: bool = False
    recovery_successful: bool = False


class MultiLanguageErrorMessages:
    """多言語エラーメッセージ"""
    
    def __init__(self):
        self.messages = {
            'ja': {
                'data_error': {
                    'missing_data': "データが不足しています。データソースを確認してください。",
                    'invalid_format': "データ形式が正しくありません。CSV またはExcel形式で提供してください。",
                    'empty_dataset': "データセットが空です。有効なデータを入力してください。",
                    'date_parsing_error': "日付の解析に失敗しました。日付形式を確認してください。"
                },
                'network_error': {
                    'connection_timeout': "ネットワーク接続がタイムアウトしました。インターネット接続を確認してください。",
                    'api_rate_limit': "APIの利用制限に達しました。しばらく時間をおいて再試行してください。",
                    'server_unavailable': "サーバーが利用できません。後ほど再試行してください。"
                },
                'computation_error': {
                    'insufficient_memory': "メモリが不足しています。他のアプリケーションを終了するか、データサイズを縮小してください。",
                    'calculation_overflow': "計算結果が範囲を超えました。データの値を確認してください。",
                    'convergence_failure': "計算が収束しませんでした。パラメーターを調整してください。"
                },
                'config_error': {
                    'missing_config': "設定ファイルが見つかりません。設定を確認してください。",
                    'invalid_config': "設定値が不正です。設定ファイルを確認してください。",
                    'permission_denied': "ファイルへのアクセス権限がありません。権限を確認してください。"
                },
                'system_error': {
                    'disk_space_full': "ディスク容量が不足しています。不要なファイルを削除してください。",
                    'dependency_missing': "必要なライブラリがインストールされていません。依存関係を確認してください。"
                },
                'user_input_error': {
                    'invalid_parameter': "パラメーターが不正です。入力値を確認してください。",
                    'missing_required_field': "必須フィールドが入力されていません。"
                }
            },
            'en': {
                'data_error': {
                    'missing_data': "Data is missing. Please check your data source.",
                    'invalid_format': "Invalid data format. Please provide CSV or Excel format.",
                    'empty_dataset': "Dataset is empty. Please provide valid data.",
                    'date_parsing_error': "Failed to parse date. Please check date format."
                },
                'network_error': {
                    'connection_timeout': "Network connection timed out. Please check your internet connection.",
                    'api_rate_limit': "API rate limit reached. Please wait and try again.",
                    'server_unavailable': "Server is unavailable. Please try again later."
                },
                'computation_error': {
                    'insufficient_memory': "Insufficient memory. Please close other applications or reduce data size.",
                    'calculation_overflow': "Calculation overflow. Please check data values.",
                    'convergence_failure': "Calculation failed to converge. Please adjust parameters."
                },
                'config_error': {
                    'missing_config': "Configuration file not found. Please check settings.",
                    'invalid_config': "Invalid configuration. Please check config file.",
                    'permission_denied': "Permission denied. Please check file permissions."
                },
                'system_error': {
                    'disk_space_full': "Disk space full. Please delete unnecessary files.",
                    'dependency_missing': "Required library not installed. Please check dependencies."
                },
                'user_input_error': {
                    'invalid_parameter': "Invalid parameter. Please check input values.",
                    'missing_required_field': "Required field is missing."
                }
            }
        }
    
    def get_message(self, category: ErrorCategory, error_type: str, language: str = 'ja') -> str:
        """エラーメッセージ取得"""
        try:
            return self.messages[language][category.value][error_type]
        except KeyError:
            # フォールバック処理
            return f"Error occurred: {category.value}.{error_type}"


class AutoRecoveryManager:
    """自動復旧管理"""
    
    def __init__(self):
        self.recovery_strategies = {
            ErrorCategory.DATA_ERROR: self._recover_data_error,
            ErrorCategory.NETWORK_ERROR: self._recover_network_error,
            ErrorCategory.COMPUTATION_ERROR: self._recover_computation_error,
            ErrorCategory.CONFIGURATION_ERROR: self._recover_config_error,
        }
    
    def attempt_recovery(self, error_context: ErrorContext, **kwargs) -> bool:
        """復旧試行"""
        if error_context.category in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_context.category](error_context, **kwargs)
            except Exception as e:
                logger.error(f"自動復旧失敗: {e}")
                return False
        return False
    
    def _recover_data_error(self, error_context: ErrorContext, **kwargs) -> bool:
        """データエラー復旧"""
        if 'missing_data' in error_context.technical_details.lower():
            # 代替データソース試行
            if 'fallback_data' in kwargs:
                logger.info("代替データを使用して復旧試行")
                return True
            
            # データ補間試行
            if 'enable_interpolation' in kwargs:
                logger.info("データ補間による復旧試行")
                return True
        
        return False
    
    def _recover_network_error(self, error_context: ErrorContext, **kwargs) -> bool:
        """ネットワークエラー復旧"""
        if 'timeout' in error_context.technical_details.lower():
            # タイムアウト延長
            retry_count = kwargs.get('retry_count', 0)
            max_retries = kwargs.get('max_retries', 3)
            
            if retry_count < max_retries:
                logger.info(f"ネットワーク再試行 ({retry_count + 1}/{max_retries})")
                time.sleep(min(2 ** retry_count, 10))  # 指数バックオフ
                return True
        
        return False
    
    def _recover_computation_error(self, error_context: ErrorContext, **kwargs) -> bool:
        """計算エラー復旧"""
        if 'memory' in error_context.technical_details.lower():
            # バッチサイズ縮小
            if 'reduce_batch_size' in kwargs:
                logger.info("バッチサイズを縮小して復旧試行")
                return True
            
            # キャッシュクリア
            if 'clear_cache' in kwargs:
                logger.info("キャッシュをクリアして復旧試行")
                import gc
                gc.collect()
                return True
        
        return False
    
    def _recover_config_error(self, error_context: ErrorContext, **kwargs) -> bool:
        """設定エラー復旧"""
        if 'missing_config' in error_context.technical_details.lower():
            # デフォルト設定作成
            if 'create_default_config' in kwargs:
                logger.info("デフォルト設定を作成して復旧試行")
                return True
        
        return False


class ErrorAnalyzer:
    """エラー分析・診断"""
    
    def __init__(self):
        self.error_patterns = self._load_error_patterns()
    
    def _load_error_patterns(self) -> Dict[str, Any]:
        """エラーパターン読み込み"""
        return {
            'data_errors': {
                'patterns': [
                    {'keyword': 'DataFrame is empty', 'category': ErrorCategory.DATA_ERROR, 'type': 'empty_dataset'},
                    {'keyword': 'KeyError', 'category': ErrorCategory.DATA_ERROR, 'type': 'missing_column'},
                    {'keyword': 'ParserError', 'category': ErrorCategory.DATA_ERROR, 'type': 'invalid_format'},
                ],
            },
            'network_errors': {
                'patterns': [
                    {'keyword': 'ConnectionError', 'category': ErrorCategory.NETWORK_ERROR, 'type': 'connection_failure'},
                    {'keyword': 'TimeoutError', 'category': ErrorCategory.NETWORK_ERROR, 'type': 'connection_timeout'},
                    {'keyword': '429', 'category': ErrorCategory.NETWORK_ERROR, 'type': 'api_rate_limit'},
                ],
            },
            'computation_errors': {
                'patterns': [
                    {'keyword': 'MemoryError', 'category': ErrorCategory.COMPUTATION_ERROR, 'type': 'insufficient_memory'},
                    {'keyword': 'OverflowError', 'category': ErrorCategory.COMPUTATION_ERROR, 'type': 'calculation_overflow'},
                    {'keyword': 'ZeroDivisionError', 'category': ErrorCategory.COMPUTATION_ERROR, 'type': 'division_by_zero'},
                ],
            }
        }
    
    def analyze_error(self, exception: Exception, context: Dict[str, Any]) -> ErrorContext:
        """エラー分析"""
        error_message = str(exception)
        error_type = type(exception).__name__
        
        # エラーカテゴリとタイプの特定
        category, error_subtype = self._classify_error(error_message, error_type)
        
        # 重要度判定
        severity = self._determine_severity(exception, category, context)
        
        # エラーコンテキスト生成
        error_context = ErrorContext(
            error_id=self._generate_error_id(),
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            component=context.get('component', 'unknown'),
            operation=context.get('operation', 'unknown'),
            error_message=error_message,
            technical_details=self._get_technical_details(exception),
            user_impact=self._assess_user_impact(category, severity),
            suggested_actions=self._generate_suggestions(category, error_subtype)
        )
        
        return error_context
    
    def _classify_error(self, error_message: str, error_type: str) -> tuple:
        """エラー分類"""
        for category_name, category_info in self.error_patterns.items():
            for pattern in category_info['patterns']:
                if pattern['keyword'].lower() in error_message.lower() or pattern['keyword'] in error_type:
                    return pattern['category'], pattern['type']
        
        # デフォルト分類
        return ErrorCategory.SYSTEM_ERROR, 'unknown_error'
    
    def _determine_severity(self, exception: Exception, category: ErrorCategory, context: Dict[str, Any]) -> ErrorSeverity:
        """重要度判定"""
        # 致命的エラー
        if isinstance(exception, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # カテゴリ別重要度
        severity_mapping = {
            ErrorCategory.DATA_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.NETWORK_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.COMPUTATION_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.CONFIGURATION_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.SYSTEM_ERROR: ErrorSeverity.CRITICAL,
            ErrorCategory.USER_INPUT_ERROR: ErrorSeverity.LOW,
        }
        
        return severity_mapping.get(category, ErrorSeverity.MEDIUM)
    
    def _generate_error_id(self) -> str:
        """エラーID生成"""
        import uuid
        return f"ERR_{uuid.uuid4().hex[:8]}"
    
    def _get_technical_details(self, exception: Exception) -> str:
        """技術的詳細取得"""
        return traceback.format_exc()
    
    def _assess_user_impact(self, category: ErrorCategory, severity: ErrorSeverity) -> str:
        """ユーザー影響評価"""
        impact_matrix = {
            (ErrorCategory.DATA_ERROR, ErrorSeverity.LOW): "一部機能が制限される可能性があります",
            (ErrorCategory.DATA_ERROR, ErrorSeverity.MEDIUM): "データ分析が実行できません",
            (ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM): "リアルタイムデータの取得が困難です",
            (ErrorCategory.COMPUTATION_ERROR, ErrorSeverity.HIGH): "計算処理が停止しました",
            (ErrorCategory.SYSTEM_ERROR, ErrorSeverity.CRITICAL): "システムが利用できません",
        }
        
        return impact_matrix.get((category, severity), "システムに影響が生じています")
    
    def _generate_suggestions(self, category: ErrorCategory, error_type: str) -> List[str]:
        """改善提案生成"""
        suggestions_db = {
            ErrorCategory.DATA_ERROR: {
                'empty_dataset': [
                    "有効なデータファイルが選択されているか確認してください",
                    "データの日付範囲が正しく設定されているか確認してください"
                ],
                'missing_column': [
                    "必要な列（Date, Open, High, Low, Close, Volume）が含まれているか確認してください",
                    "列名が正しいスペルで記載されているか確認してください"
                ]
            },
            ErrorCategory.NETWORK_ERROR: [
                "インターネット接続を確認してください",
                "VPNまたはファイアウォール設定を確認してください",
                "しばらく時間をおいて再試行してください"
            ],
            ErrorCategory.COMPUTATION_ERROR: [
                "利用可能メモリを増やすため、他のアプリケーションを終了してください",
                "データサイズを縮小してください",
                "バッチ処理サイズを小さくしてください"
            ]
        }
        
        if category in suggestions_db:
            if isinstance(suggestions_db[category], dict):
                return suggestions_db[category].get(error_type, ["専門家にお問い合わせください"])
            else:
                return suggestions_db[category]
        
        return ["エラーの詳細をサポートにお問い合わせください"]


class EnhancedErrorHandler:
    """強化エラーハンドリングシステム"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None, language: str = 'ja'):
        self.config = config or OptimizationConfig()
        self.language = language
        
        self.message_manager = MultiLanguageErrorMessages()
        self.recovery_manager = AutoRecoveryManager()
        self.analyzer = ErrorAnalyzer()
        
        # エラー履歴
        self.error_history: List[ErrorContext] = []
        
        logger.info(f"強化エラーハンドリングシステム初期化完了 (言語: {language})")
    
    def handle_error(
        self,
        exception: Exception,
        context: Dict[str, Any],
        auto_recovery: bool = True,
        **recovery_kwargs
    ) -> ErrorContext:
        """エラー処理"""
        # エラー分析
        error_context = self.analyzer.analyze_error(exception, context)
        
        # 多言語メッセージ取得
        if hasattr(error_context, 'error_subtype'):
            localized_message = self.message_manager.get_message(
                error_context.category,
                error_context.error_subtype,
                self.language
            )
            error_context.error_message = localized_message
        
        # 自動復旧試行
        if auto_recovery and error_context.severity != ErrorSeverity.CRITICAL:
            logger.info(f"自動復旧試行: {error_context.error_id}")
            error_context.auto_recovery_attempted = True
            error_context.recovery_successful = self.recovery_manager.attempt_recovery(
                error_context, **recovery_kwargs
            )
        
        # エラー履歴記録
        self.error_history.append(error_context)
        
        # ログ出力
        self._log_error(error_context)
        
        return error_context
    
    def _log_error(self, error_context: ErrorContext):
        """エラーログ出力"""
        log_level_mapping = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        
        log_level = log_level_mapping.get(error_context.severity, logging.ERROR)
        
        logger.log(log_level, 
            f"エラー発生 [{error_context.error_id}]: "
            f"{error_context.component}.{error_context.operation} - "
            f"{error_context.error_message}"
        )
        
        if error_context.auto_recovery_attempted:
            if error_context.recovery_successful:
                logger.info(f"自動復旧成功 [{error_context.error_id}]")
            else:
                logger.warning(f"自動復旧失敗 [{error_context.error_id}]")
    
    def get_user_friendly_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """ユーザー向けエラー情報"""
        return {
            'error_id': error_context.error_id,
            'severity': error_context.severity.value,
            'message': error_context.error_message,
            'user_impact': error_context.user_impact,
            'suggestions': error_context.suggested_actions,
            'recovery_attempted': error_context.auto_recovery_attempted,
            'recovery_successful': error_context.recovery_successful,
            'timestamp': error_context.timestamp.isoformat()
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """エラー統計取得"""
        if not self.error_history:
            return {'total_errors': 0}
        
        total_errors = len(self.error_history)
        severity_counts = {}
        category_counts = {}
        recovery_success_rate = 0
        
        for error_context in self.error_history:
            # 重要度別集計
            severity = error_context.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # カテゴリ別集計
            category = error_context.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # 自動復旧成功率
        recovery_attempts = sum(1 for ec in self.error_history if ec.auto_recovery_attempted)
        if recovery_attempts > 0:
            recovery_successes = sum(1 for ec in self.error_history if ec.recovery_successful)
            recovery_success_rate = recovery_successes / recovery_attempts
        
        return {
            'total_errors': total_errors,
            'severity_distribution': severity_counts,
            'category_distribution': category_counts,
            'recovery_success_rate': recovery_success_rate,
            'recent_errors': len([ec for ec in self.error_history if (datetime.now() - ec.timestamp).days <= 1])
        }


# グローバルエラーハンドラーインスタンス
_global_error_handler: Optional[EnhancedErrorHandler] = None

def get_global_error_handler(language: str = 'ja') -> EnhancedErrorHandler:
    """グローバルエラーハンドラー取得"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = EnhancedErrorHandler(language=language)
    return _global_error_handler

def handle_error_gracefully(
    operation_name: str,
    component_name: str = "unknown"
) -> Callable:
    """デコレータ: エラーをグレースフルに処理"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_global_error_handler()
                context = {
                    'operation': operation_name,
                    'component': component_name,
                    'function': func.__name__
                }
                error_context = error_handler.handle_error(e, context)
                
                # 復旧成功時は処理継続、失敗時は例外再発生
                if error_context.recovery_successful:
                    logger.info(f"エラー復旧により処理継続: {operation_name}")
                    return func(*args, **kwargs)  # 復旧後再試行
                else:
                    raise e  # 元の例外を再発生
        
        return wrapper
    return decorator