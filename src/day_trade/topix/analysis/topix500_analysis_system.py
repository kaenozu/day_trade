#!/usr/bin/env python3
"""
TOPIX500 Analysis System - Main System Class

統合されたTOPIX500分析システム（後方互換性維持）
内部的にはモジュラー化されたコンポーネントを使用
"""

import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

from .comprehensive_analyzer import ComprehensiveAnalyzer
from .data_classes import TOPIX500AnalysisResult
from .performance_manager import PerformanceManager

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TOPIX500AnalysisSystem:
    """
    TOPIX500 Analysis System

    500銘柄大規模分析・セクター別分析・高性能処理
    統合最適化基盤（Issues #322-325）+ Issue #315全機能統合

    内部的にモジュラー化されたコンポーネントを使用し、
    後方互換性を維持したインターフェースを提供
    """

    def __init__(
        self,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        max_concurrent_symbols: int = 50,
        max_concurrent_sectors: int = 10,
        memory_limit_gb: float = 1.0,
        processing_timeout: int = 20,
        batch_size: int = 25,
    ):
        """
        TOPIX500分析システム初期化

        Args:
            enable_cache: キャッシュ有効化
            enable_parallel: 並列処理有効化
            max_concurrent_symbols: 最大同時分析銘柄数
            max_concurrent_sectors: 最大同時分析セクター数
            memory_limit_gb: メモリ使用制限（GB）
            processing_timeout: 処理タイムアウト（秒）
            batch_size: バッチサイズ
        """
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel
        self.max_concurrent_symbols = max_concurrent_symbols
        self.max_concurrent_sectors = max_concurrent_sectors
        self.memory_limit_gb = memory_limit_gb
        self.processing_timeout = processing_timeout
        self.batch_size = batch_size

        # モジュラー化されたコンポーネントを初期化
        self.comprehensive_analyzer = ComprehensiveAnalyzer(
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
            max_concurrent_symbols=max_concurrent_symbols,
            max_concurrent_sectors=max_concurrent_sectors,
            memory_limit_gb=memory_limit_gb,
            processing_timeout=processing_timeout,
        )

        self.performance_manager = PerformanceManager(
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
        )

        # 後方互換性のためのプロパティ
        self.topix500_symbols = {}
        self.sector_mapping = {}
        self.symbol_data_cache = {}
        self.stats = self.performance_manager.stats

        logger.info("TOPIX500 Analysis System（統合最適化版）初期化完了")
        logger.info(f"  - 統合キャッシュ: {self.enable_cache}")
        logger.info(f"  - 並列処理: {self.enable_parallel}")
        logger.info(f"  - 最大同時銘柄数: {self.max_concurrent_symbols}")
        logger.info(f"  - メモリ制限: {self.memory_limit_gb}GB")
        logger.info(f"  - 処理タイムアウト: {self.processing_timeout}秒")
        logger.info(f"  - バッチサイズ: {self.batch_size}")

    async def load_topix500_master_data(
        self, master_data_path: Optional[str] = None
    ) -> bool:
        """
        TOPIX500マスターデータ読み込み

        Args:
            master_data_path: マスターデータファイルパス（Noneの場合は模擬データ生成）

        Returns:
            bool: 読み込み成功フラグ
        """
        success = await self.comprehensive_analyzer.load_master_data(master_data_path)

        if success:
            # 後方互換性のためのプロパティ更新
            self.topix500_symbols = self.comprehensive_analyzer.data_loader.get_symbols()
            self.sector_mapping = self.comprehensive_analyzer.data_loader.get_sector_mapping()

        return success

    async def analyze_single_symbol_comprehensive(
        self, symbol: str, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        単一銘柄包括的分析（Issue #315全機能統合）

        Args:
            symbol: 銘柄コード
            data: 価格データ

        Returns:
            Dict[str, Any]: 包括分析結果
        """
        return await self.comprehensive_analyzer.single_symbol_analyzer.analyze_single_symbol_comprehensive(
            symbol, data
        )

    async def analyze_sector_batch(
        self, sector: str, symbols_data: Dict[str, pd.DataFrame]
    ):
        """
        セクター別バッチ分析

        Args:
            sector: セクター名
            symbols_data: セクター内銘柄データ（symbol -> DataFrame）

        Returns:
            SectorAnalysisResult: セクター分析結果
        """
        return await self.comprehensive_analyzer.sector_analyzer.analyze_sector_batch(
            sector, symbols_data, self.comprehensive_analyzer.single_symbol_analyzer
        )

    async def analyze_topix500_comprehensive(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        target_symbols: Optional[List[str]] = None,
    ) -> TOPIX500AnalysisResult:
        """
        TOPIX500包括分析

        Args:
            symbols_data: 全銘柄データ（symbol -> DataFrame）
            target_symbols: 分析対象銘柄（Noneの場合は全銘柄）

        Returns:
            TOPIX500AnalysisResult: TOPIX500分析結果
        """
        result = await self.comprehensive_analyzer.analyze_topix500_comprehensive(
            symbols_data, target_symbols
        )

        # 統計更新
        self.performance_manager.update_stats(
            successful_symbols=result.successful_analyses,
            failed_symbols=result.failed_analyses,
            processing_time=result.total_processing_time,
            sector_analyses=len(result.sector_results),
        )

        return result

    async def analyze_batch_comprehensive(
        self,
        stock_data: Dict[str, pd.DataFrame],
        enable_sector_analysis: bool = True,
        enable_ml_prediction: bool = True,
    ) -> Dict[str, Any]:
        """
        包括的バッチ分析実行

        Args:
            stock_data: 株式データ辞書
            enable_sector_analysis: セクター分析有効化
            enable_ml_prediction: ML予測有効化

        Returns:
            包括的分析結果
        """
        result = await self.comprehensive_analyzer.analyze_batch_comprehensive(
            stock_data, enable_sector_analysis, enable_ml_prediction
        )

        # 統計更新
        performance_metrics = result.get("performance_metrics")
        if performance_metrics:
            self.performance_manager.update_stats(
                successful_symbols=performance_metrics.successful_symbols,
                failed_symbols=performance_metrics.failed_symbols,
                processing_time=performance_metrics.processing_time_seconds,
                memory_usage=performance_metrics.peak_memory_mb,
                batch_analyses=1,
                sector_analyses=performance_metrics.sector_count,
            )

        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        return self.performance_manager.get_performance_stats(
            max_concurrent_symbols=self.max_concurrent_symbols,
            memory_limit_gb=self.memory_limit_gb,
            processing_timeout=self.processing_timeout,
            topix500_symbols_count=len(self.topix500_symbols),
            sector_mapping_count=len(self.sector_mapping),
            symbol_data_cache_count=len(self.symbol_data_cache),
        )

    def shutdown(self):
        """システムシャットダウン"""
        logger.info("TOPIX500分析システムシャットダウン開始")

        # 各コンポーネントのシャットダウン処理があれば実行
        if hasattr(self.comprehensive_analyzer, "shutdown"):
            self.comprehensive_analyzer.shutdown()

        logger.info("TOPIX500分析システムシャットダウン完了")
