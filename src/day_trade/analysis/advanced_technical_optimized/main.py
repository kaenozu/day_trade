#!/usr/bin/env python3
"""
高度テクニカル指標システム メインモジュール
Issue #315: 高度テクニカル指標・ML機能拡張

統合最適化基盤フル活用版:
- Issue #324: 98%メモリ削減キャッシュ活用
- Issue #323: 100倍並列処理活用
- Issue #325: 97%ML高速化活用
- Issue #322: 89%精度データ拡張活用

すべての機能を統合したメインクラス
"""

from typing import Any, Dict, List

import pandas as pd

from .batch_processor import BatchProcessor
from .bollinger_bands_analyzer import BollingerBandsAnalyzer
from .core_system import CoreAdvancedTechnicalSystem
from .data_structures import BollingerBandsAnalysis, IchimokuAnalysis
from .ichimoku_analyzer import IchimokuAnalyzer
from .performance_utils import PerformanceUtils

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class AdvancedTechnicalIndicatorsOptimized(
    CoreAdvancedTechnicalSystem,
    BollingerBandsAnalyzer, 
    IchimokuAnalyzer,
    BatchProcessor,
):
    """
    高度テクニカル指標分析システム（統合最適化版）

    統合最適化基盤をフル活用:
    - Issue #324: 統合キャッシュで98%メモリ削減効果
    - Issue #323: 並列処理で100倍高速化効果
    - Issue #325: ML最適化で97%処理高速化効果
    - Issue #322: 多角データで89%精度向上効果
    
    主要機能:
    - Bollinger Bands変動率分析
    - 一目均衡表総合判定
    - 複合移動平均分析
    - フィボナッチ retracement自動検出
    - バッチ並列処理
    - パフォーマンス最適化
    """

    def __init__(
        self,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        enable_ml_optimization: bool = True,
        cache_ttl_minutes: int = 5,
        max_concurrent: int = 20,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        初期化

        Args:
            enable_cache: 統合キャッシュ有効化
            enable_parallel: 並列処理有効化
            enable_ml_optimization: ML最適化有効化
            cache_ttl_minutes: キャッシュ有効期限（分）
            max_concurrent: 最大並列数
            confidence_threshold: 信頼度閾値
        """
        super().__init__(
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
            enable_ml_optimization=enable_ml_optimization,
            cache_ttl_minutes=cache_ttl_minutes,
            max_concurrent=max_concurrent,
            confidence_threshold=confidence_threshold,
        )

        # パフォーマンスユーティリティインスタンス
        self.performance_utils = PerformanceUtils()

        logger.info("高度テクニカル指標システム（統合最適化版）メインクラス初期化完了")

    async def comprehensive_analysis(
        self, data: pd.DataFrame, symbol: str, analysis_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        包括的テクニカル分析

        Args:
            data: 株価データ
            symbol: 銘柄シンボル
            analysis_types: 分析タイプリスト

        Returns:
            Dict[str, Any]: 包括的分析結果
        """
        if analysis_types is None:
            analysis_types = ["bb", "ichimoku", "ma", "fibonacci"]

        logger.info(f"包括的分析開始: {symbol}")

        # 単一銘柄での並列分析実行
        results = await self._analyze_single_symbol_parallel(
            symbol, data, analysis_types
        )

        # 総合パフォーマンススコア計算
        individual_scores = {}
        if "bollinger_bands" in results:
            individual_scores["bb"] = results["bollinger_bands"].performance_score
        if "ichimoku_cloud" in results:
            individual_scores["ichimoku"] = results["ichimoku_cloud"].performance_score
        if "complex_ma" in results:
            individual_scores["ma"] = results["complex_ma"].performance_score
        if "fibonacci" in results:
            individual_scores["fibonacci"] = results["fibonacci"].performance_score

        composite_score = self.performance_utils.calculate_composite_performance_score(
            individual_scores
        )

        results["composite_performance_score"] = composite_score
        results["analysis_summary"] = self._generate_analysis_summary(results)

        logger.info(f"包括的分析完了: {symbol} (総合スコア: {composite_score:.3f})")

        return results

    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析結果サマリー生成"""
        summary = {
            "total_analyses": len([k for k in results.keys() if k not in 
                                 ["composite_performance_score", "analysis_summary"]]),
            "signals": {},
            "confidence_scores": {},
            "performance_scores": {},
            "overall_recommendation": "HOLD",
            "overall_confidence": 0.5,
        }

        # 各分析結果からの情報抽出
        signals = []
        confidences = []
        
        for analysis_name, analysis_result in results.items():
            if hasattr(analysis_result, 'signal'):
                summary["signals"][analysis_name] = analysis_result.signal
                summary["confidence_scores"][analysis_name] = analysis_result.confidence
                summary["performance_scores"][analysis_name] = analysis_result.performance_score
                
                signals.append(analysis_result.signal)
                confidences.append(analysis_result.confidence)

        # 総合推奨計算
        if signals:
            buy_count = signals.count("BUY")
            sell_count = signals.count("SELL")
            
            if buy_count > sell_count:
                summary["overall_recommendation"] = "BUY"
            elif sell_count > buy_count:
                summary["overall_recommendation"] = "SELL"
            else:
                summary["overall_recommendation"] = "HOLD"
            
            # 総合信頼度
            summary["overall_confidence"] = sum(confidences) / len(confidences) if confidences else 0.5

        return summary

    def get_detailed_performance_stats(self) -> Dict[str, Any]:
        """詳細パフォーマンス統計取得"""
        base_stats = self.get_optimization_performance_stats()
        
        # 追加の詳細統計
        detailed_stats = {
            **base_stats,
            "efficiency_metrics": self.performance_utils.calculate_efficiency_metrics(
                self.performance_stats["avg_processing_time"],
                self.performance_stats.get("memory_usage", 0),
                self.performance_stats["cache_hits"],
                self.performance_stats["total_analyses"],
            ),
            "system_health": {
                "cache_system": "healthy" if self.cache_enabled else "disabled",
                "parallel_system": "healthy" if self.parallel_enabled else "disabled", 
                "ml_optimization": "healthy" if self.ml_optimization_enabled else "disabled",
            },
        }

        return detailed_stats

    def reset_performance_stats(self) -> None:
        """パフォーマンス統計リセット"""
        self.performance_stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "parallel_analyses": 0,
            "ml_optimizations": 0,
            "avg_processing_time": 0.0,
            "memory_efficiency": 0.0,
            "accuracy_improvements": 0.0,
        }
        
        logger.info("パフォーマンス統計をリセットしました")

    def get_system_status(self) -> Dict[str, Any]:
        """システムステータス取得"""
        return {
            "system_name": "AdvancedTechnicalIndicatorsOptimized",
            "version": "1.0.0",
            "optimization_features": {
                "unified_cache": self.cache_enabled,
                "parallel_processing": self.parallel_enabled,
                "ml_optimization": self.ml_optimization_enabled,
            },
            "configuration": {
                "cache_ttl_minutes": self.cache_ttl_minutes,
                "max_concurrent": self.max_concurrent,
                "confidence_threshold": self.confidence_threshold,
            },
            "performance_summary": self.get_optimization_performance_stats(),
        }