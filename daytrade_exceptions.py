#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Exceptions Module - 統一例外処理システム
"""

from typing import Optional, Dict, Any


class DayTradeError(Exception):
    """デイトレードシステム基底例外"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = self._get_timestamp()
    
    def _get_timestamp(self) -> str:
        """エラー発生時刻を取得"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """例外情報を辞書形式で返す"""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'error_code': self.error_code,
            'details': self.details,
            'timestamp': self.timestamp
        }


class DataSourceError(DayTradeError):
    """データソース関連例外"""
    pass


class YFinanceError(DataSourceError):
    """yfinanceデータ取得例外"""
    pass


class PriceDataError(DataSourceError):
    """価格データ関連例外"""
    pass


class MLPredictionError(DayTradeError):
    """ML予測関連例外"""
    pass


class ModelLoadError(MLPredictionError):
    """MLモデル読み込み例外"""
    pass


class PredictionAccuracyError(MLPredictionError):
    """予測精度関連例外"""
    pass


class AnalysisEngineError(DayTradeError):
    """分析エンジン関連例外"""
    pass


class TechnicalAnalysisError(AnalysisEngineError):
    """テクニカル分析例外"""
    pass


class SignalGenerationError(AnalysisEngineError):
    """シグナル生成例外"""
    pass


class WebDashboardError(DayTradeError):
    """Webダッシュボード関連例外"""
    pass


class APIError(WebDashboardError):
    """API関連例外"""
    pass


class ChartGenerationError(WebDashboardError):
    """チャート生成例外"""
    pass


class ConfigurationError(DayTradeError):
    """設定関連例外"""
    pass


class ValidationError(DayTradeError):
    """バリデーション例外"""
    pass


class SystemResourceError(DayTradeError):
    """システムリソース関連例外"""
    pass


class DatabaseError(DayTradeError):
    """データベース関連例外"""
    pass


class NetworkError(DayTradeError):
    """ネットワーク関連例外"""
    pass


class TimeoutError(DayTradeError):
    """タイムアウト例外"""
    pass


def handle_exception(func):
    """例外処理デコレータ"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DayTradeError:
            # 既知の例外はそのまま再発生
            raise
        except Exception as e:
            # 未知の例外をDayTradeErrorでラップ
            raise DayTradeError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={'function': func.__name__, 'original_error': str(e)}
            ) from e
    return wrapper


def handle_async_exception(func):
    """非同期例外処理デコレータ"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except DayTradeError:
            # 既知の例外はそのまま再発生
            raise
        except Exception as e:
            # 未知の例外をDayTradeErrorでラップ
            raise DayTradeError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={'function': func.__name__, 'original_error': str(e)}
            ) from e
    return wrapper


class ExceptionLogger:
    """例外ログシステム"""
    
    def __init__(self):
        from daytrade_logging import get_logger
        self.logger = get_logger("exceptions")
    
    def log_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """例外をログに記録"""
        if isinstance(exception, DayTradeError):
            self._log_daytrade_error(exception, context)
        else:
            self._log_general_exception(exception, context)
    
    def _log_daytrade_error(self, error: DayTradeError, context: Optional[Dict[str, Any]] = None):
        """DayTradeError専用ログ"""
        log_data = error.to_dict()
        if context:
            log_data['context'] = context
        
        if error.error_code and error.error_code.startswith('CRITICAL'):
            self.logger.critical(f"Critical Error: {log_data}")
        elif isinstance(error, (DataSourceError, MLPredictionError)):
            self.logger.error(f"System Error: {log_data}")
        else:
            self.logger.warning(f"Handled Error: {log_data}")
    
    def _log_general_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """一般例外ログ"""
        self.logger.exception(
            f"Unhandled Exception: {exception.__class__.__name__}: {str(exception)}"
        )
        if context:
            self.logger.error(f"Exception Context: {context}")


class ErrorRecoveryManager:
    """エラー回復管理システム"""
    
    def __init__(self):
        from daytrade_logging import get_logger
        self.logger = get_logger("error_recovery")
        self.retry_counts = {}
        self.max_retries = 3
    
    def should_retry(self, error: Exception, operation_id: str) -> bool:
        """リトライすべきかどうかを判定"""
        if not isinstance(error, (NetworkError, TimeoutError, DataSourceError)):
            return False
        
        current_count = self.retry_counts.get(operation_id, 0)
        if current_count >= self.max_retries:
            self.logger.warning(f"Max retries exceeded for {operation_id}")
            return False
        
        self.retry_counts[operation_id] = current_count + 1
        self.logger.info(f"Retry {current_count + 1}/{self.max_retries} for {operation_id}")
        return True
    
    def reset_retry_count(self, operation_id: str):
        """リトライカウントをリセット"""
        if operation_id in self.retry_counts:
            del self.retry_counts[operation_id]
    
    def get_fallback_data(self, data_type: str, symbol: Optional[str] = None) -> Any:
        """フォールバックデータを取得"""
        self.logger.info(f"Using fallback data for {data_type}, symbol: {symbol}")
        
        if data_type == "price_data":
            return {
                'price': 1000.0,
                'change': 0.0,
                'volume': 100000,
                'is_fallback': True
            }
        elif data_type == "ml_prediction":
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'prediction': 0,
                'is_fallback': True
            }
        elif data_type == "technical_indicators":
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'sma20': 1000.0,
                'is_fallback': True
            }
        else:
            return {'is_fallback': True}


# グローバルインスタンス
_exception_logger = None
_error_recovery_manager = None


def get_exception_logger() -> ExceptionLogger:
    """グローバル例外ログシステムを取得"""
    global _exception_logger
    if _exception_logger is None:
        _exception_logger = ExceptionLogger()
    return _exception_logger


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """グローバルエラー回復管理システムを取得"""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
    return _error_recovery_manager


def log_and_handle_exception(exception: Exception, context: Optional[Dict[str, Any]] = None):
    """例外をログに記録し、適切に処理"""
    get_exception_logger().log_exception(exception, context)


if __name__ == "__main__":
    # テスト実行
    from daytrade_logging import setup_logging
    
    setup_logging(debug=True)
    
    # 各種例外のテスト
    try:
        raise YFinanceError("yfinance connection failed", error_code="YFINANCE_001")
    except DayTradeError as e:
        log_and_handle_exception(e, {'symbol': '7203', 'operation': 'price_fetch'})
    
    try:
        raise MLPredictionError("Model prediction failed", error_code="ML_002")
    except DayTradeError as e:
        log_and_handle_exception(e, {'model': 'SimpleML', 'symbol': '8306'})
    
    # エラー回復テスト
    recovery_manager = get_error_recovery_manager()
    
    error = NetworkError("Connection timeout")
    operation_id = "price_fetch_7203"
    
    for i in range(5):
        if recovery_manager.should_retry(error, operation_id):
            print(f"Retrying operation {operation_id}")
        else:
            print(f"Using fallback for {operation_id}")
            fallback_data = recovery_manager.get_fallback_data("price_data", "7203")
            print(f"Fallback data: {fallback_data}")
            break
    
    print("例外処理システムテスト完了")