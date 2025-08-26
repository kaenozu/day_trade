#!/usr/bin/env python3
"""
TOPIX500 Analysis System - Modular Components

分割されたTOPIX500分析システムのモジュラーコンポーネント
後方互換性を維持したインターフェース
"""

# データクラスのインポート
# 各分析コンポーネントのインポート
from .batch_processor import BatchProcessor
from .comprehensive_analyzer import ComprehensiveAnalyzer
from .data_classes import (
    BatchProcessingTask,
    PerformanceMetrics,
    SectorAnalysisResult,
    TOPIX500AnalysisResult,
    TOPIX500Symbol,
)
from .data_loader import DataLoader
from .market_analyzer import MarketAnalyzer
from .performance_manager import PerformanceManager
from .sector_analyzer import SectorAnalyzer
from .single_symbol_analyzer import SingleSymbolAnalyzer

# 後方互換性のためのメインクラス（従来のTOPIX500AnalysisSystemと同等）
from .topix500_analysis_system import TOPIX500AnalysisSystem

# パッケージレベルでエクスポートするクラス・関数
__all__ = [
    # データクラス
    "TOPIX500Symbol",
    "PerformanceMetrics",
    "SectorAnalysisResult",
    "TOPIX500AnalysisResult",
    "BatchProcessingTask",
    # 分析コンポーネント
    "DataLoader",
    "SingleSymbolAnalyzer",
    "SectorAnalyzer",
    "MarketAnalyzer",
    "BatchProcessor",
    "ComprehensiveAnalyzer",
    "PerformanceManager",
    # メインクラス（後方互換性）
    "TOPIX500AnalysisSystem",
]

# パッケージ情報
__version__ = "2.0.0"
__description__ = "TOPIX500 Analysis System - Modular Architecture"
__author__ = "Day Trade System Team"
