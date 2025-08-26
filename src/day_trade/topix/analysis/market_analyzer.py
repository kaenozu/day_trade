#!/usr/bin/env python3
"""
TOPIX500 Analysis System - Market Analyzer

市場全体の分析・統計計算機能
"""

from typing import Any, Dict, List

from .data_classes import SectorAnalysisResult

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class MarketAnalyzer:
    """市場分析機能"""

    def __init__(self):
        """市場分析器初期化"""
        logger.info("市場分析器初期化完了")

    async def extract_top_recommendations(
        self, sector_results: Dict[str, SectorAnalysisResult]
    ) -> List[Dict[str, Any]]:
        """上位推奨銘柄抽出"""
        recommendations = []

        for sector, result in sector_results.items():
            for symbol in result.top_performers:
                recommendations.append(
                    {
                        "symbol": symbol,
                        "sector": sector,
                        "sector_signal": result.sector_rotation_signal,
                        "sector_allocation": result.recommended_allocation,
                        "risk_level": result.risk_level,
                    }
                )

        # セクター配分でソート
        recommendations.sort(key=lambda x: x["sector_allocation"], reverse=True)

        return recommendations[:20]  # 上位20銘柄

    async def calculate_market_overview(
        self, sector_results: Dict[str, SectorAnalysisResult]
    ) -> Dict[str, float]:
        """市場全体概観計算"""
        if not sector_results:
            return {}

        total_symbols = sum(r.symbol_count for r in sector_results.values())

        # 加重平均計算
        weighted_volatility = (
            sum(r.avg_volatility * r.symbol_count for r in sector_results.values())
            / total_symbols
        )

        weighted_return = (
            sum(r.avg_return * r.symbol_count for r in sector_results.values())
            / total_symbols
        )

        # セクター分散
        overweight_count = sum(
            1
            for r in sector_results.values()
            if r.sector_rotation_signal == "overweight"
        )
        neutral_count = sum(
            1 for r in sector_results.values() if r.sector_rotation_signal == "neutral"
        )
        underweight_count = sum(
            1
            for r in sector_results.values()
            if r.sector_rotation_signal == "underweight"
        )

        return {
            "market_volatility": weighted_volatility,
            "market_return": weighted_return,
            "total_symbols": total_symbols,
            "total_sectors": len(sector_results),
            "overweight_sectors": overweight_count,
            "neutral_sectors": neutral_count,
            "underweight_sectors": underweight_count,
            "market_sentiment": overweight_count / max(len(sector_results), 1),
        }

    async def calculate_risk_distribution(
        self, sector_results: Dict[str, SectorAnalysisResult]
    ) -> Dict[str, int]:
        """リスク分布計算"""
        risk_counts = {"low": 0, "medium": 0, "high": 0, "extreme": 0}

        for result in sector_results.values():
            risk_counts[result.risk_level] += result.symbol_count

        return risk_counts

    async def calculate_performance_statistics(
        self,
        total_time: float,
        sector_results: Dict[str, SectorAnalysisResult],
        successful_analyses: int,
        cache_hits: int,
        total_analyses: int,
    ) -> Dict[str, float]:
        """パフォーマンス統計計算"""
        return {
            "total_processing_time": total_time,
            "avg_sector_processing_time": (
                sum(r.processing_time for r in sector_results.values()) / len(sector_results)
                if sector_results
                else 0.0
            ),
            "symbols_per_second": successful_analyses / max(total_time, 0.001),
            "cache_hit_rate": cache_hits / max(total_analyses, 1),
            "success_rate": successful_analyses / max(successful_analyses + (len(sector_results) * 10), 1),
        }
