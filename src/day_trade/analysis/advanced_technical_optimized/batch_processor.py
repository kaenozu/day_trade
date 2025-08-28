#!/usr/bin/env python3
"""
バッチ処理機能モジュール
Issue #315: 高度テクニカル指標・ML機能拡張

複数銘柄バッチ分析（Issue #323 並列処理活用）:
- 並列バッチ分析実行
- シーケンシャルバッチ分析実行
- 単一銘柄並列分析
"""

import asyncio
import time
from typing import Any, Dict, List

import pandas as pd

from .analysis_helpers import AnalysisHelpers
from .bollinger_bands_analyzer import BollingerBandsAnalyzer
from .core_system import CoreAdvancedTechnicalSystem
from .ichimoku_analyzer import IchimokuAnalyzer

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class BatchProcessor(
    CoreAdvancedTechnicalSystem, BollingerBandsAnalyzer, IchimokuAnalyzer
):
    """
    バッチ処理機能（Issue #323 並列処理活用）

    複数銘柄の高度テクニカル分析を効率的に実行:
    - 並列バッチ分析
    - シーケンシャルバッチ分析
    - 単一銘柄並列分析
    """

    def __init__(self, **kwargs):
        """初期化"""
        super().__init__(**kwargs)
        self.analysis_helpers = AnalysisHelpers()

    async def batch_analyze_symbols(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        analysis_types: List[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        複数銘柄バッチ分析（Issue #323 並列処理活用）

        Args:
            symbols_data: {symbol: DataFrame} 形式の株価データ
            analysis_types: 分析種類リスト ["bb", "ichimoku", "ma", "fibonacci"]
        """
        if analysis_types is None:
            analysis_types = ["bb", "ichimoku", "ma"]

        logger.info(
            f"バッチ分析開始: {len(symbols_data)}銘柄, {len(analysis_types)}種類"
        )
        start_time = time.time()

        # Issue #323: 並列処理による高速バッチ実行
        if self.parallel_enabled and len(symbols_data) > 1:
            try:
                results = await self._execute_parallel_batch_analysis(
                    symbols_data, analysis_types
                )
                self.performance_stats["parallel_analyses"] += 1
            except Exception as e:
                logger.warning(f"並列処理失敗、シーケンシャル実行: {e}")
                results = await self._execute_sequential_batch_analysis(
                    symbols_data, analysis_types
                )
        else:
            results = await self._execute_sequential_batch_analysis(
                symbols_data, analysis_types
            )

        processing_time = time.time() - start_time
        logger.info(f"バッチ分析完了: {len(results)}銘柄 ({processing_time:.2f}秒)")

        return results

    async def _execute_sequential_batch_analysis(
        self, symbols_data: Dict[str, pd.DataFrame], analysis_types: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """シーケンシャルバッチ分析実行"""
        results = {}

        for symbol, data in symbols_data.items():
            symbol_results = await self._analyze_single_symbol(
                symbol, data, analysis_types
            )
            results[symbol] = symbol_results

        return results

    async def _execute_parallel_batch_analysis(
        self, symbols_data: Dict[str, pd.DataFrame], analysis_types: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """並列バッチ分析実行（Issue #323活用）"""
        # 並列タスク作成
        tasks = []
        symbols = list(symbols_data.keys())

        for symbol in symbols:
            data = symbols_data[symbol]
            task = self._analyze_single_symbol_parallel(
                symbol, data, analysis_types
            )
            tasks.append(task)

        # 並列実行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果整理
        final_results = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                logger.error(f"並列分析エラー {symbol}: {result}")
                final_results[symbol] = {}
            else:
                final_results[symbol] = result

        return final_results

    async def _analyze_single_symbol(
        self, symbol: str, data: pd.DataFrame, analysis_types: List[str]
    ) -> Dict[str, Any]:
        """単一銘柄分析（シーケンシャル）"""
        results = {}

        if "bb" in analysis_types:
            results["bollinger_bands"] = (
                await self.analyze_bollinger_bands_optimized(data, symbol)
            )

        if "ichimoku" in analysis_types:
            results["ichimoku_cloud"] = (
                await self.analyze_ichimoku_cloud_optimized(data, symbol)
            )

        if "ma" in analysis_types:
            results["complex_ma"] = (
                await self.analysis_helpers.analyze_complex_ma_optimized(
                    data, symbol
                )
            )

        if "fibonacci" in analysis_types:
            results["fibonacci"] = (
                await self.analysis_helpers.analyze_fibonacci_optimized(
                    data, symbol
                )
            )

        return results

    async def _analyze_single_symbol_parallel(
        self, symbol: str, data: pd.DataFrame, analysis_types: List[str]
    ) -> Dict[str, Any]:
        """単一銘柄並列分析"""
        results = {}

        # 分析タスクを並列実行
        analysis_tasks = []

        if "bb" in analysis_types:
            analysis_tasks.append(
                (
                    "bollinger_bands",
                    self.analyze_bollinger_bands_optimized(data, symbol),
                )
            )

        if "ichimoku" in analysis_types:
            analysis_tasks.append(
                (
                    "ichimoku_cloud",
                    self.analyze_ichimoku_cloud_optimized(data, symbol),
                )
            )

        if "ma" in analysis_types:
            analysis_tasks.append(
                (
                    "complex_ma",
                    self.analysis_helpers.analyze_complex_ma_optimized(
                        data, symbol
                    ),
                )
            )

        if "fibonacci" in analysis_types:
            analysis_tasks.append(
                (
                    "fibonacci",
                    self.analysis_helpers.analyze_fibonacci_optimized(
                        data, symbol
                    ),
                )
            )

        # 並列実行
        if analysis_tasks:
            task_results = await asyncio.gather(
                *[task[1] for task in analysis_tasks], return_exceptions=True
            )

            for i, (analysis_name, _) in enumerate(analysis_tasks):
                result = task_results[i]
                if not isinstance(result, Exception):
                    results[analysis_name] = result
                else:
                    logger.error(
                        f"分析エラー {symbol}-{analysis_name}: {result}"
                    )

        return results