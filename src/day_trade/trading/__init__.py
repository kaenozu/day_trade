"""
取引管理システム

モジュラー設計による包括的な取引管理機能
"""

# メインインターフェース
from .trade_manager import TradeManager

# コア機能
from .core.trade_executor import TradeExecutor
from .core.position_manager import PositionManager
from .core.risk_calculator import RiskCalculator
from .core.types import Trade, Position, RealizedPnL, TradeStatus, TradeType

# 永続化機能
from .persistence.db_manager import TradeDatabaseManager
from .persistence.batch_processor import TradeBatchProcessor
from .persistence.data_cleaner import DataCleaner

# 分析機能
from .analytics.portfolio_analyzer import PortfolioAnalyzer
from .analytics.tax_calculator import TaxCalculator
from .analytics.report_exporter import ReportExporter

# 検証機能
from .validation.trade_validator import TradeValidator
from .validation.compliance_checker import ComplianceChecker
from .validation.id_generator import IDGenerator

__all__ = [
    # メインインターフェース
    "TradeManager",

    # コア機能
    "TradeExecutor",
    "PositionManager",
    "RiskCalculator",
    "Trade",
    "Position",
    "RealizedPnL",
    "TradeStatus",
    "TradeType",

    # 永続化機能
    "TradeDatabaseManager",
    "TradeBatchProcessor",
    "DataCleaner",

    # 分析機能
    "PortfolioAnalyzer",
    "TaxCalculator",
    "ReportExporter",

    # 検証機能
    "TradeValidator",
    "ComplianceChecker",
    "IDGenerator",
]
