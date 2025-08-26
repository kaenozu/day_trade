#!/usr/bin/env python3
"""
TOPIX500 Analysis System - Single Symbol Analyzer

単一銘柄の包括的分析機能
"""

import asyncio
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

try:
    from ...analysis.multi_timeframe_analysis_optimized import (
        MultiTimeframeAnalysisOptimized,
    )
    from ...ml.advanced_ml_models import AdvancedMLModels
    from ...risk.volatility_prediction_system import VolatilityPredictionSystem
    from ...utils.logging_config import get_context_logger
    from ...utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class SingleSymbolAnalyzer:
    """単一銘柄分析機能"""

    def __init__(
        self,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        max_concurrent_symbols: int = 50,
    ):
        """
        単一銘柄分析器初期化

        Args:
            enable_cache: キャッシュ有効化
            enable_parallel: 並列処理有効化
            max_concurrent_symbols: 最大同時分析銘柄数
        """
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel
        self.cache_manager = None

        if enable_cache:
            try:
                from ...utils.unified_cache_manager import UnifiedCacheManager
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=256, l2_memory_mb=512, l3_disk_mb=2048
                )
            except ImportError:
                pass

        # Issue #315統合システム初期化
        try:
            self.multiframe_analyzer = MultiTimeframeAnalysisOptimized(
                enable_cache=enable_cache,
                enable_parallel=enable_parallel,
                max_concurrent=max_concurrent_symbols,
            )

            self.ml_models = AdvancedMLModels(
                enable_cache=enable_cache,
                enable_parallel=enable_parallel,
                max_concurrent=max_concurrent_symbols,
            )

            self.volatility_predictor = VolatilityPredictionSystem(
                enable_cache=enable_cache,
                enable_parallel=enable_parallel,
                max_concurrent=max_concurrent_symbols,
            )
        except ImportError:
            self.multiframe_analyzer = None
            self.ml_models = None
            self.volatility_predictor = None

        # 統計
        self.stats = {
            "cache_hits": 0,
            "total_analyses": 0,
        }

        logger.info("単一銘柄分析器初期化完了")

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
        start_time = time.time()

        # キャッシュチェック
        cache_key = None
        if self.enable_cache and self.cache_manager:
            try:
                from ...utils.unified_cache_manager import generate_unified_cache_key
                cache_key = generate_unified_cache_key(
                    "topix500_comprehensive", symbol, str(len(data))
                )
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    logger.debug(f"包括分析キャッシュヒット: {symbol}")
                    self.stats["cache_hits"] += 1
                    return cached_result
            except ImportError:
                pass

        try:
            logger.debug(f"包括分析開始: {symbol}")

            # 並列実行で全分析を同時実行
            if (
                self.enable_parallel
                and len(data) >= 50
                and self.multiframe_analyzer
                and self.ml_models
                and self.volatility_predictor
            ):
                # 十分なデータがある場合は全分析実行
                tasks = [
                    self.multiframe_analyzer.analyze_multi_timeframe(data, symbol),
                    self.ml_models.extract_advanced_features(data, symbol),
                    self.volatility_predictor.integrated_volatility_forecast(
                        data, symbol
                    ),
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)
                multiframe_result, features_result, volatility_result = results

                # エラーハンドリング
                if isinstance(multiframe_result, Exception):
                    logger.warning(
                        f"マルチタイムフレーム分析エラー: {symbol} - {multiframe_result}"
                    )
                    multiframe_result = None

                if isinstance(features_result, Exception):
                    logger.warning(f"特徴量抽出エラー: {symbol} - {features_result}")
                    features_result = None

                if isinstance(volatility_result, Exception):
                    logger.warning(
                        f"ボラティリティ予測エラー: {symbol} - {volatility_result}"
                    )
                    volatility_result = None

                # ML予測（特徴量が取得できた場合のみ）
                ml_ensemble_result = None
                if features_result and self.ml_models:
                    try:
                        ml_ensemble_result = await self.ml_models.ensemble_prediction(
                            data, symbol, features_result
                        )
                    except Exception as e:
                        logger.warning(f"アンサンブル予測エラー: {symbol} - {e}")

            else:
                # データ不足または並列無効の場合は簡易分析
                logger.warning(f"データ不足または並列無効: {symbol} - {len(data)}日分")
                multiframe_result = None
                features_result = None
                volatility_result = None
                ml_ensemble_result = None

            # 結果統合
            comprehensive_result = {
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_length": len(data),
                "multiframe_analysis": (
                    asdict(multiframe_result) if multiframe_result else None
                ),
                "advanced_features": (
                    asdict(features_result) if features_result else None
                ),
                "volatility_prediction": (
                    asdict(volatility_result) if volatility_result else None
                ),
                "ml_ensemble": (
                    asdict(ml_ensemble_result) if ml_ensemble_result else None
                ),
                "processing_time": time.time() - start_time,
            }

            # 統合スコア計算
            comprehensive_result["integrated_score"] = (
                await self._calculate_integrated_score(
                    multiframe_result,
                    features_result,
                    volatility_result,
                    ml_ensemble_result,
                )
            )

            # キャッシュ保存
            if self.enable_cache and cache_key and self.cache_manager:
                self.cache_manager.put(cache_key, comprehensive_result)

            logger.debug(
                f"包括分析完了: {symbol} - スコア: {comprehensive_result['integrated_score']:.3f} ({comprehensive_result['processing_time']:.3f}s)"
            )

            self.stats["total_analyses"] += 1
            return comprehensive_result

        except Exception as e:
            logger.error(f"包括分析エラー: {symbol} - {e}")
            traceback.print_exc()
            self.stats["total_analyses"] += 1
            return {
                "symbol": symbol,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "integrated_score": 0.0,
            }

    async def _calculate_integrated_score(
        self, multiframe_result, features_result, volatility_result, ml_result
    ) -> float:
        """統合スコア計算"""
        score = 0.0
        components = 0

        # マルチタイムフレーム（30%）
        if multiframe_result:
            mf_score = multiframe_result.weighted_confidence * (
                1 - multiframe_result.risk_adjusted_score
            )
            score += mf_score * 0.3
            components += 0.3

        # ML予測（25%）
        if ml_result:
            ml_score = ml_result.weighted_confidence
            score += ml_score * 0.25
            components += 0.25

        # ボラティリティ（25%）
        if volatility_result:
            vol_score = volatility_result.confidence_level * (
                1 - volatility_result.integrated_risk_score
            )
            score += vol_score * 0.25
            components += 0.25

        # 特徴量品質（20%）
        if features_result:
            feature_score = min(
                1.0, features_result.feature_count / 30
            )  # 30特徴量で満点
            score += feature_score * 0.2
            components += 0.2

        # 正規化
        if components > 0:
            score = score / components

        return max(0.0, min(1.0, score))
