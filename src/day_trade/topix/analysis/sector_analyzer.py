#!/usr/bin/env python3
"""
TOPIX500 Analysis System - Sector Analyzer

セクター別分析機能
"""

import asyncio
import time
import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data_classes import SectorAnalysisResult

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class SectorAnalyzer:
    """セクター分析機能"""

    def __init__(
        self,
        enable_parallel: bool = True,
        max_concurrent_symbols: int = 50,
    ):
        """
        セクター分析器初期化

        Args:
            enable_parallel: 並列処理有効化
            max_concurrent_symbols: 最大同時分析銘柄数
        """
        self.enable_parallel = enable_parallel
        self.max_concurrent_symbols = max_concurrent_symbols
        logger.info("セクター分析器初期化完了")

    async def analyze_sector_batch(
        self, sector: str, symbols_data: Dict[str, pd.DataFrame],
        single_symbol_analyzer
    ) -> SectorAnalysisResult:
        """
        セクター別バッチ分析

        Args:
            sector: セクター名
            symbols_data: セクター内銘柄データ（symbol -> DataFrame）
            single_symbol_analyzer: 単一銘柄分析器

        Returns:
            SectorAnalysisResult: セクター分析結果
        """
        start_time = time.time()

        try:
            logger.info(f"セクター分析開始: {sector} ({len(symbols_data)}銘柄)")

            # セクター内銘柄の包括分析
            symbol_results = {}

            if self.enable_parallel:
                # 並列実行
                tasks = []
                for symbol, data in symbols_data.items():
                    task = single_symbol_analyzer.analyze_single_symbol_comprehensive(
                        symbol, data
                    )
                    tasks.append(task)

                # セマフォで同時実行数制御
                semaphore = asyncio.Semaphore(self.max_concurrent_symbols)

                async def analyze_with_semaphore(task):
                    async with semaphore:
                        return await task

                results = await asyncio.gather(
                    *[analyze_with_semaphore(task) for task in tasks],
                    return_exceptions=True,
                )

                for i, result in enumerate(results):
                    symbol = list(symbols_data.keys())[i]
                    if isinstance(result, Exception):
                        logger.warning(f"銘柄分析エラー: {symbol} - {result}")
                        symbol_results[symbol] = {
                            "error": str(result),
                            "integrated_score": 0.0,
                        }
                    else:
                        symbol_results[symbol] = result

            else:
                # 順次実行
                for symbol, data in symbols_data.items():
                    result = await single_symbol_analyzer.analyze_single_symbol_comprehensive(
                        symbol, data
                    )
                    symbol_results[symbol] = result

            # セクター統計計算
            valid_results = [r for r in symbol_results.values() if "error" not in r]

            if not valid_results:
                logger.warning(f"セクター分析失敗: {sector} - 有効な結果なし")
                return self._create_default_sector_result(sector, len(symbols_data))

            # ボラティリティ・リターン統計
            volatilities = []
            returns = []
            scores = []

            for result in valid_results:
                if result.get("volatility_prediction"):
                    vol_data = result["volatility_prediction"]
                    volatilities.append(vol_data.get("final_volatility_forecast", 0.2))

                # リターン推定（データから簡易計算）
                symbol = result["symbol"]
                if symbol in symbols_data:
                    data = symbols_data[symbol]
                    if len(data) > 1:
                        ret = (
                            (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1)
                            * 252
                            / len(data)
                        )
                        returns.append(ret)

                scores.append(result.get("integrated_score", 0.0))

            avg_volatility = np.mean(volatilities) if volatilities else 0.2
            avg_return = np.mean(returns) if returns else 0.0
            avg_score = np.mean(scores)

            # セクターモメンタム（上位銘柄のスコア重み付き平均）
            sorted_scores = sorted(scores, reverse=True)
            top_scores = sorted_scores[: min(5, len(sorted_scores))]
            sector_momentum = np.mean(top_scores) if top_scores else 0.0

            # リスクレベル判定
            risk_level = await self._determine_sector_risk_level(
                avg_volatility, avg_score
            )

            # 推奨配分
            recommended_allocation = await self._calculate_sector_allocation(
                avg_score, risk_level, len(valid_results)
            )

            # トップパフォーマー
            top_performers = sorted(
                valid_results,
                key=lambda x: x.get("integrated_score", 0.0),
                reverse=True,
            )[:3]
            top_performer_symbols = [p["symbol"] for p in top_performers]

            # セクターローテーションシグナル
            rotation_signal = await self._generate_sector_rotation_signal(
                avg_score, sector_momentum, avg_volatility
            )

            result = SectorAnalysisResult(
                sector=sector,
                symbol_count=len(valid_results),
                avg_volatility=avg_volatility,
                avg_return=avg_return,
                sector_momentum=sector_momentum,
                risk_level=risk_level,
                recommended_allocation=recommended_allocation,
                top_performers=top_performer_symbols,
                sector_rotation_signal=rotation_signal,
                processing_time=time.time() - start_time,
            )

            logger.info(
                f"セクター分析完了: {sector} - {rotation_signal} (スコア: {avg_score:.3f}) ({result.processing_time:.3f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"セクター分析エラー: {sector} - {e}")
            traceback.print_exc()
            return self._create_default_sector_result(sector, len(symbols_data))

    async def perform_sector_analysis(
        self, stock_data: Dict[str, pd.DataFrame], symbol_results: Dict[str, Any]
    ) -> Dict[str, SectorAnalysisResult]:
        """セクター分析実行"""
        sector_groups = {}

        # セクター別銘柄グループ化
        for symbol in stock_data:
            # 簡易セクター分類（銘柄コード先頭2桁ベース）
            if symbol.startswith(("72", "79")):
                sector = "Automotive/Banks"
            elif symbol.startswith(("43", "45")):
                sector = "Pharmaceuticals"
            elif symbol.startswith(("99", "98")):
                sector = "Technology"
            elif symbol.startswith(("82", "83")):
                sector = "Retail"
            else:
                sector = "Others"

            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(symbol)

        # セクター別分析
        sector_results = {}
        for sector, symbols in sector_groups.items():
            try:
                # セクター内銘柄の成果計算
                sector_scores = []
                successful_symbols = []

                for symbol in symbols:
                    result = symbol_results.get(symbol, {})
                    if result.get("success", False):
                        # 特徴量から簡易スコア計算
                        features = result.get("features", {})
                        score = (
                            features.get("price_change_pct", 0.0) / 100.0
                        )  # -1.0 to 1.0
                        sector_scores.append(score)
                        successful_symbols.append(symbol)

                if sector_scores:
                    avg_score = np.mean(sector_scores)
                    volatility = np.std(sector_scores)

                    # トレンド判定
                    if avg_score > 0.1:
                        trend = "BULLISH"
                    elif avg_score < -0.1:
                        trend = "BEARISH"
                    else:
                        trend = "NEUTRAL"

                    # トップパフォーマー抽出
                    if len(sector_scores) >= 3:
                        top_indices = np.argsort(sector_scores)[-3:]
                        top_performers = [successful_symbols[i] for i in top_indices]
                    else:
                        top_performers = successful_symbols

                else:
                    avg_score = 0.0
                    volatility = 0.0
                    trend = "NEUTRAL"
                    top_performers = []

                sector_result = SectorAnalysisResult(
                    sector_name=sector,
                    symbol_count=len(symbols),
                    symbols=symbols,
                    avg_performance_score=avg_score,
                    sector_trend=trend,
                    sector_volatility=volatility,
                    top_performers=top_performers,
                    sector_metrics={
                        "total_symbols": len(symbols),
                        "successful_symbols": len(successful_symbols),
                        "success_rate": (
                            len(successful_symbols) / len(symbols)
                            if len(symbols) > 0
                            else 0.0
                        ),
                    },
                )

                sector_results[sector] = sector_result

            except Exception as e:
                logger.error(f"セクター分析エラー {sector}: {e}")
                # デフォルトセクター結果
                sector_results[sector] = SectorAnalysisResult(
                    sector_name=sector,
                    symbol_count=len(symbols),
                    symbols=symbols,
                    avg_performance_score=0.0,
                    sector_trend="NEUTRAL",
                    sector_volatility=0.0,
                    top_performers=[],
                    sector_metrics={"error": str(e)},
                )

        logger.info(f"セクター分析完了: {len(sector_results)}セクター")
        return sector_results

    async def _determine_sector_risk_level(
        self, avg_volatility: float, avg_score: float
    ) -> str:
        """セクターリスクレベル判定"""
        if avg_volatility < 0.15 and avg_score > 0.7:
            return "low"
        elif avg_volatility < 0.25 and avg_score > 0.5:
            return "medium"
        elif avg_volatility < 0.40:
            return "high"
        else:
            return "extreme"

    async def _calculate_sector_allocation(
        self, avg_score: float, risk_level: str, symbol_count: int
    ) -> float:
        """セクター推奨配分計算"""
        base_allocation = 0.1  # 10%ベース

        # スコア調整
        score_multiplier = avg_score

        # リスク調整
        risk_multipliers = {"low": 1.5, "medium": 1.0, "high": 0.7, "extreme": 0.3}
        risk_multiplier = risk_multipliers.get(risk_level, 1.0)

        # 銘柄数調整（分散効果）
        size_multiplier = min(1.5, 1.0 + symbol_count / 50)

        allocation = (
            base_allocation * score_multiplier * risk_multiplier * size_multiplier
        )
        return max(0.01, min(0.3, allocation))  # 1%-30%の範囲

    async def _generate_sector_rotation_signal(
        self, avg_score: float, momentum: float, volatility: float
    ) -> str:
        """セクターローテーションシグナル生成"""
        # 複合指標計算
        composite_score = avg_score * 0.5 + momentum * 0.3 + (1 - volatility) * 0.2

        if composite_score > 0.7:
            return "overweight"
        elif composite_score > 0.4:
            return "neutral"
        else:
            return "underweight"

    def _create_default_sector_result(
        self, sector: str, symbol_count: int
    ) -> SectorAnalysisResult:
        """デフォルトセクター結果作成"""
        return SectorAnalysisResult(
            sector=sector,
            symbol_count=symbol_count,
            avg_volatility=0.25,
            avg_return=0.05,
            sector_momentum=0.5,
            risk_level="medium",
            recommended_allocation=0.1,
            top_performers=[],
            sector_rotation_signal="neutral",
            processing_time=0.0,
        )
