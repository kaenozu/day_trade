#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Analysis Package - 高度技術分析パッケージ

後方互換性を維持するためのimport設定
"""

# 型定義とデータクラス
from .types import (
    AnalysisType,
    TechnicalSignal,
    PatternMatch,
    TechnicalAnalysisResult
)

# コアクラス
from .indicators import AdvancedTechnicalIndicators
from .patterns import PatternRecognition
from .signals import SignalAnalyzer
from .database import DatabaseManager
from .analyzer import AdvancedTechnicalAnalysis

# テスト機能
from .test_runner import TestRunner, run_advanced_technical_analysis_test

# グローバルインスタンス（後方互換性のため）
advanced_technical_analysis = AdvancedTechnicalAnalysis()

# パッケージ情報
__version__ = "1.0.0"
__author__ = "Day Trade System"
__description__ = "Advanced technical analysis system with modular architecture"

# エクスポートリスト
__all__ = [
    # 型定義
    "AnalysisType",
    "TechnicalSignal", 
    "PatternMatch",
    "TechnicalAnalysisResult",
    
    # クラス
    "AdvancedTechnicalIndicators",
    "PatternRecognition",
    "SignalAnalyzer",
    "DatabaseManager",
    "AdvancedTechnicalAnalysis",
    
    # テスト
    "TestRunner",
    "run_advanced_technical_analysis_test",
    
    # グローバルインスタンス
    "advanced_technical_analysis"
]