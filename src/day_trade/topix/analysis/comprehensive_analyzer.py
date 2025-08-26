#!/usr/bin/env python3
"""
TOPIX500 Analysis System - Comprehensive Analyzer

TOPIX500包括的分析のメイン機能
"""

import asyncio
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .batch_processor import BatchProcessor
from .data_classes import TOPIX500AnalysisResult
from .data_loader import DataLoader
from .market_analyzer import MarketAnalyzer
from .sector_analyzer import SectorAnalyzer
from .single_symbol_analyzer import SingleSymbolAnalyzer

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class ComprehensiveAnalyzer:
    """TOPIX500包括的分析機能"""

    def __init__(
        self,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        max_concurrent_symbols: int = 50,
        max_concurrent_sectors: int = 10,
        memory_limit_gb: float = 1.0,
        processing_timeout: int = 20,
    ):
        """
        包括的分析器初期化

        Args:
            enable_cache: キャッシュ有効化
            enable_parallel: 並列処理有効化
            max_concurrent_symbols: 最大同時分析銘柄数
            max_concurrent_sectors: 最大同時分析セクター数
            memory_limit_gb: メモリ使用制限（GB）
            processing_timeout: 処理タイムアウト（秒）
        """
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel
        self.max_concurrent_sectors = max_concurrent_sectors

        # コンポーネント初期化
        self.data_loader = DataLoader()
        self.single_symbol_analyzer = SingleSymbolAnalyzer(
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
            max_concurrent_symbols=max_concurrent_symbols,
        )
        self.sector_analyzer = SectorAnalyzer(
            enable_parallel=enable_parallel,
            max_concurrent_symbols=max_concurrent_symbols,
        )
        self.market_analyzer = MarketAnalyzer()
        self.batch_processor = BatchProcessor(
            enable_parallel=enable_parallel,
            max_concurrent_symbols=max_concurrent_symbols,
            memory_limit_gb=memory_limit_gb,
            processing_timeout=processing_timeout,
        )

        # 統計
        self.stats = {
            "total_analyses": 0,
            "batch_analyses": 0,
            "sector_analyses": 0,
            "successful_symbols": 0,
            "failed_symbols": 0,
        }

        logger.info("TOPIX500包括的分析器初期化完了")

    async def load_master_data(self, master_data_path: Optional[str] = None) -> bool:
        """マスターデータ読み込み"""
        return await self.data_loader.load_topix500_master_data(master_data_path)

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
        start_time = time.time()
        analysis_timestamp = datetime.now()

        try:
            logger.info("TOPIX500包括分析開始")

            # 分析対象銘柄決定
            if target_symbols:
                target_data = {
                    s: symbols_data[s] for s in target_symbols if s in symbols_data
                }
            else:
                target_data = symbols_data

            logger.info(f"分析対象: {len(target_data)}銘柄")

            # セクター別データ分割
            sector_data = {}
            topix500_symbols = self.data_loader.get_symbols()

            for symbol, data in target_data.items():
                if symbol in topix500_symbols:
                    sector = topix500_symbols[symbol].sector
                    if sector not in sector_data:
                        sector_data[sector] = {}
                    sector_data[sector][symbol] = data

            logger.info(f"分析対象セクター: {len(sector_data)}セクター")

            # セクター別並列分析
            sector_results = {}
            successful_analyses = 0
            failed_analyses = 0

            if self.enable_parallel:
                # 並列セクター分析
                sector_tasks = []
                for sector, sector_symbols_data in sector_data.items():
                    task = self.sector_analyzer.analyze_sector_batch(
                        sector, sector_symbols_data, self.single_symbol_analyzer
                    )
                    sector_tasks.append((sector, task))

                # セマフォで同時実行数制御
                semaphore = asyncio.Semaphore(self.max_concurrent_sectors)

                async def analyze_sector_with_semaphore(sector, task):
                    async with semaphore:
                        return sector, await task

                results = await asyncio.gather(
                    *[
                        analyze_sector_with_semaphore(sector, task)
                        for sector, task in sector_tasks
                    ],
                    return_exceptions=True,
                )

                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"セクター分析エラー: {result}")
                        failed_analyses += 1
                    else:
                        sector, sector_result = result
                        sector_results[sector] = sector_result
                        successful_analyses += sector_result.symbol_count

            else:
                # 順次セクター分析
                for sector, sector_symbols_data in sector_data.items():
                    try:
                        sector_result = await self.sector_analyzer.analyze_sector_batch(
                            sector, sector_symbols_data, self.single_symbol_analyzer
                        )
                        sector_results[sector] = sector_result
                        successful_analyses += sector_result.symbol_count
                    except Exception as e:
                        logger.error(f"セクター分析エラー: {sector} - {e}")
                        failed_analyses += len(sector_symbols_data)

            # 上位推奨銘柄抽出
            top_recommendations = await self.market_analyzer.extract_top_recommendations(
                sector_results
            )

            # 市場全体概観
            market_overview = await self.market_analyzer.calculate_market_overview(
                sector_results
            )

            # リスク分布
            risk_distribution = await self.market_analyzer.calculate_risk_distribution(
                sector_results
            )

            # パフォーマンス統計
            processing_performance = await self.market_analyzer.calculate_performance_statistics(
                time.time() - start_time,
                sector_results,
                successful_analyses,
                self.single_symbol_analyzer.stats.get("cache_hits", 0),
                self.single_symbol_analyzer.stats.get("total_analyses", 1),
            )

            # 統計更新
            self.stats["total_analyses"] += 1
            self.stats["batch_analyses"] += 1
            self.stats["sector_analyses"] += len(sector_results)
            self.stats["successful_symbols"] += successful_analyses
            self.stats["failed_symbols"] += failed_analyses

            result = TOPIX500AnalysisResult(
                analysis_timestamp=analysis_timestamp,
                total_symbols_analyzed=len(target_data),
                successful_analyses=successful_analyses,
                failed_analyses=failed_analyses,
                sector_results=sector_results,
                top_recommendations=top_recommendations,
                market_overview=market_overview,
                risk_distribution=risk_distribution,
                processing_performance=processing_performance,
                total_processing_time=time.time() - start_time,
            )

            logger.info(
                f"TOPIX500包括分析完了: {successful_analyses}銘柄成功, {failed_analyses}銘柄失敗 ({result.total_processing_time:.1f}s)"
            )
            logger.info(
                f"処理性能: {processing_performance['symbols_per_second']:.1f}銘柄/秒"
            )

            return result

        except Exception as e:
            logger.error(f"TOPIX500包括分析エラー: {e}")
            traceback.print_exc()

            # エラー時のデフォルト結果
            return TOPIX500AnalysisResult(
                analysis_timestamp=analysis_timestamp,
                total_symbols_analyzed=len(symbols_data) if symbols_data else 0,
                successful_analyses=0,
                failed_analyses=len(symbols_data) if symbols_data else 0,
                sector_results={},
                top_recommendations=[],
                market_overview={},
                risk_distribution={},
                processing_performance={"error": str(e)},
                total_processing_time=time.time() - start_time,
            )

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
        return await self.batch_processor.analyze_batch_comprehensive(
            stock_data, enable_sector_analysis, enable_ml_prediction, self.sector_analyzer
        )
