#!/usr/bin/env python3
"""
システムサービス実装
Issue #918 項目3対応: 依存性注入パターンの導入

具体的なサービス実装クラス群
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .dependency_injection import (
    IConfigurationService, ILoggingService, IAnalyzerService,
    IDashboardService, IDataProviderService, injectable, singleton
)
from ..config.config_manager import ConfigManager
from ..utils.logging_config import get_context_logger


@singleton(IConfigurationService)
@injectable
class ConfigurationService(IConfigurationService):
    """設定サービス実装"""

    def __init__(self):
        self._config_manager = ConfigManager()
        self._config = self._config_manager.get_config()
        self.logger = get_context_logger(__name__, "ConfigurationService")

    def get_config(self) -> Dict[str, Any]:
        """設定を取得"""
        return self._config or {}

    def get_analysis_config(self) -> Dict[str, Any]:
        """分析設定を取得"""
        try:
            analysis_config = self._config.get('analysis', {}) if self._config else {}
            return {
                'technical_indicators': analysis_config.get('technical_indicators', {}),
                'confidence': analysis_config.get('confidence', {}),
                'data_periods': analysis_config.get('data_periods', {})
            }
        except Exception as e:
            self.logger.warning(f"Failed to get analysis config: {e}")
            # デフォルト設定を返す
            return {
                'technical_indicators': {
                    'rsi': {'period': 14, 'overbought_threshold': 70, 'oversold_threshold': 30},
                    'sma': {'short_period': 20, 'long_period': 50},
                    'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
                },
                'confidence': {'default_confidence': 0.85, 'minimum_confidence': 0.60},
                'data_periods': {'default_period': '1y'}
            }


@singleton(ILoggingService)
@injectable
class LoggingService(ILoggingService):
    """ログサービス実装"""

    def __init__(self):
        self._loggers = {}

    def get_logger(self, name: str, context: str = None):
        """ロガーを取得"""
        key = f"{name}:{context}" if context else name

        if key not in self._loggers:
            self._loggers[key] = get_context_logger(name, context)

        return self._loggers[key]


@singleton(IAnalyzerService)
@injectable
class AnalyzerService(IAnalyzerService):
    """分析サービス実装"""

    def __init__(self, config_service: IConfigurationService, logging_service: ILoggingService):
        self._config_service = config_service
        self._logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "AnalyzerService")
        self._analyzer = None

    def _get_analyzer(self):
        """アナライザーの遅延初期化"""
        if self._analyzer is None:
            try:
                from ..analysis.advanced_technical_analyzer import AdvancedTechnicalAnalyzer
                analysis_config = self._config_service.get_analysis_config()
                self._analyzer = AdvancedTechnicalAnalyzer(config=analysis_config)
                self.logger.info("Analyzer initialized successfully")
            except ImportError as e:
                self.logger.error(f"Failed to import analyzer: {e}")
                # フォールバック実装
                self._analyzer = self._create_fallback_analyzer()
            except Exception as e:
                self.logger.error(f"Failed to initialize analyzer: {e}")
                self._analyzer = self._create_fallback_analyzer()

        return self._analyzer

    def _create_fallback_analyzer(self):
        """フォールバック用の簡易アナライザー"""
        class FallbackAnalyzer:
            def __init__(self):
                pass

            def analyze(self, symbol: str, **kwargs):
                return {"status": "fallback", "symbol": symbol, "message": "Using fallback analyzer"}

        return FallbackAnalyzer()

    def analyze(self, symbol: str, **kwargs):
        """分析実行"""
        analyzer = self._get_analyzer()
        return analyzer.analyze(symbol, **kwargs)


@singleton(IDashboardService)
@injectable
class DashboardService(IDashboardService):
    """ダッシュボードサービス実装"""

    def __init__(self, config_service: IConfigurationService, logging_service: ILoggingService):
        self._config_service = config_service
        self._logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "DashboardService")
        self._dashboard = None

    def _get_dashboard(self):
        """ダッシュボードの遅延初期化"""
        if self._dashboard is None:
            try:
                from ..dashboard.web_dashboard import WebDashboard
                self._dashboard = WebDashboard()
                self.logger.info("Dashboard initialized successfully")
            except ImportError as e:
                self.logger.error(f"Failed to import dashboard: {e}")
                self._dashboard = self._create_fallback_dashboard()
            except Exception as e:
                self.logger.error(f"Failed to initialize dashboard: {e}")
                self._dashboard = self._create_fallback_dashboard()

        return self._dashboard

    def _create_fallback_dashboard(self):
        """フォールバック用の簡易ダッシュボード"""
        class FallbackDashboard:
            def start_dashboard(self, **kwargs):
                print("Dashboard service not available - using fallback")
                return {"status": "fallback", "message": "Dashboard service not available"}

        return FallbackDashboard()

    def start_dashboard(self, **kwargs):
        """ダッシュボード開始"""
        dashboard = self._get_dashboard()
        return dashboard.start_dashboard(**kwargs)


@singleton(IDataProviderService)
@injectable
class DataProviderService(IDataProviderService):
    """データプロバイダーサービス実装"""

    def __init__(self, config_service: IConfigurationService, logging_service: ILoggingService):
        self._config_service = config_service
        self._logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "DataProviderService")
        self._data_provider = None

    def _get_data_provider(self):
        """データプロバイダーの遅延初期化"""
        if self._data_provider is None:
            try:
                # 実際のデータプロバイダーを使用
                from ..data.stock_data_provider import StockDataProvider
                self._data_provider = StockDataProvider()
                self.logger.info("Data provider initialized successfully")
            except ImportError as e:
                self.logger.warning(f"Stock data provider not available: {e}")
                self._data_provider = self._create_fallback_provider()
            except Exception as e:
                self.logger.error(f"Failed to initialize data provider: {e}")
                self._data_provider = self._create_fallback_provider()

        return self._data_provider

    def _create_fallback_provider(self):
        """フォールバック用の簡易データプロバイダー"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        class FallbackDataProvider:
            def get_stock_data(self, symbol: str, period: str):
                # 模擬データを生成
                days = 30
                dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
                np.random.seed(hash(symbol) % 2147483647)

                base_price = 1000 + (hash(symbol) % 1000)
                prices = []
                current_price = base_price

                for _ in range(days):
                    change = np.random.normal(0, 0.02)
                    current_price *= (1 + change)
                    prices.append(current_price)

                return pd.DataFrame({
                    'Date': dates,
                    'Open': [p * 0.99 for p in prices],
                    'High': [p * 1.02 for p in prices],
                    'Low': [p * 0.98 for p in prices],
                    'Close': prices,
                    'Volume': np.random.randint(1000000, 50000000, days)
                })

        return FallbackDataProvider()

    def get_stock_data(self, symbol: str, period: str):
        """株価データ取得"""
        provider = self._get_data_provider()
        return provider.get_stock_data(symbol, period)


def register_default_services():
    """デフォルトサービスを登録"""
    from .dependency_injection import get_container
    from .database_services import register_database_services
    from .async_services import register_async_services
    from .security_services import register_security_services

    container = get_container()

    # 既に登録されていない場合のみ登録
    if not container.is_registered(IConfigurationService):
        container.register_singleton(IConfigurationService, ConfigurationService)

    if not container.is_registered(ILoggingService):
        container.register_singleton(ILoggingService, LoggingService)

    if not container.is_registered(IAnalyzerService):
        container.register_singleton(IAnalyzerService, AnalyzerService)

    if not container.is_registered(IDashboardService):
        container.register_singleton(IDashboardService, DashboardService)

    if not container.is_registered(IDataProviderService):
        container.register_singleton(IDataProviderService, DataProviderService)

    # データベース最適化サービス登録
    register_database_services()

    # 非同期・並行処理サービス登録
    register_async_services()

    # セキュリティサービス登録
    register_security_services()