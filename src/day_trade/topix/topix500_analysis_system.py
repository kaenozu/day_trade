#!/usr/bin/env python3
"""
TOPIX500 Analysis System
Issue #314: TOPIX500全銘柄対応

85銘柄→500銘柄への大規模拡張
統合最適化基盤（Issues #322-325）+ Issue #315全機能統合
"""

import asyncio
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from ..analysis.multi_timeframe_analysis_optimized import (
        MultiTimeframeAnalysisOptimized,
    )
    from ..data.advanced_parallel_ml_engine import AdvancedParallelMLEngine
    from ..ml.advanced_ml_models import AdvancedMLModels
    from ..risk.volatility_prediction_system import VolatilityPredictionSystem
    from ..utils.logging_config import get_context_logger
    from ..utils.performance_monitor import PerformanceMonitor
    from ..utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class UnifiedCacheManager:
        def __init__(self, **kwargs):
            pass

        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"topix500_{hash(str(args) + str(kwargs))}"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TOPIX500Symbol:
    """TOPIX500銘柄情報"""

    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    weight_in_index: float
    listing_date: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    processing_time_seconds: float
    avg_time_per_symbol_ms: float
    peak_memory_mb: float
    cache_hit_rate: float
    throughput_symbols_per_second: float
    sector_count: int
    error_messages: List[str] = field(default_factory=list)


@dataclass
class SectorAnalysisResult:
    """セクター分析結果"""

    sector_name: str
    symbol_count: int
    symbols: List[str] = field(default_factory=list)
    avg_performance_score: float = 0.0
    sector_trend: str = "NEUTRAL"  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    sector_volatility: float = 0.0
    top_performers: List[str] = field(default_factory=list)
    sector_metrics: Dict[str, Any] = field(default_factory=dict)

    # 既存フィールドとの互換性
    sector: str = field(init=False)
    avg_volatility: float = field(init=False)
    avg_return: float = field(init=False)
    sector_momentum: float = field(init=False)
    risk_level: str = field(init=False)
    recommended_allocation: float = field(init=False)
    sector_rotation_signal: str = field(init=False)
    processing_time: float = field(init=False)

    def __post_init__(self):
        # 既存フィールドの値を設定
        self.sector = self.sector_name
        self.avg_volatility = self.sector_volatility
        self.avg_return = self.avg_performance_score
        self.sector_momentum = self.avg_performance_score
        self.risk_level = "medium"
        self.recommended_allocation = max(0.0, min(1.0, self.avg_performance_score))
        self.sector_rotation_signal = "neutral"
        self.processing_time = 0.0


@dataclass
class TOPIX500AnalysisResult:
    """TOPIX500総合分析結果"""

    analysis_timestamp: datetime
    total_symbols_analyzed: int
    successful_analyses: int
    failed_analyses: int
    sector_results: Dict[str, SectorAnalysisResult]
    top_recommendations: List[Dict[str, Any]]
    market_overview: Dict[str, float]
    risk_distribution: Dict[str, int]
    processing_performance: Dict[str, float]
    total_processing_time: float


@dataclass
class BatchProcessingTask:
    """バッチ処理タスク"""

    task_id: str
    symbols: List[str]
    analysis_types: List[str]
    priority: float = 1.0
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 2


class TOPIX500AnalysisSystem:
    """
    TOPIX500 Analysis System

    500銘柄大規模分析・セクター別分析・高性能処理
    統合最適化基盤（Issues #322-325）+ Issue #315全機能統合
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

        # 統合最適化基盤初期化
        if self.enable_cache:
            self.cache_manager = UnifiedCacheManager(
                l1_memory_mb=256, l2_memory_mb=512, l3_disk_mb=2048
            )
            logger.info("統合キャッシュシステム有効化（Issue #324統合）")
        else:
            self.cache_manager = None

        if self.enable_parallel:
            self.parallel_engine = AdvancedParallelMLEngine(
                cpu_workers=max_concurrent_symbols,
                enable_monitoring=True,
                memory_limit_gb=memory_limit_gb,
            )
            logger.info("高度並列処理システム有効化（Issue #323統合）")
        else:
            self.parallel_engine = None

        # Issue #315統合システム初期化
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

        # TOPIX500データ管理
        self.topix500_symbols = {}
        self.sector_mapping = {}
        self.symbol_data_cache = {}

        # パフォーマンス統計
        self.stats = {
            "total_analyses": 0,
            "batch_analyses": 0,
            "sector_analyses": 0,
            "cache_hits": 0,
            "processing_times": [],
            "memory_usage": [],
            "successful_symbols": 0,
            "failed_symbols": 0,
        }

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
        try:
            logger.info("TOPIX500マスターデータ読み込み開始")

            if master_data_path and Path(master_data_path).exists():
                # 実際のファイルから読み込み
                df = pd.read_csv(master_data_path)
                symbols_data = df.to_dict("records")
            else:
                # 模擬データ生成
                symbols_data = await self._generate_mock_topix500_data()

            # TOPIX500銘柄情報構築
            for data in symbols_data:
                symbol = TOPIX500Symbol(
                    symbol=data["symbol"],
                    name=data.get("name", f"Company_{data['symbol']}"),
                    sector=data.get("sector", "Technology"),
                    industry=data.get("industry", "Software"),
                    market_cap=data.get("market_cap", 100000000000),
                    weight_in_index=data.get("weight", 0.2),
                    listing_date=pd.to_datetime(data.get("listing_date", "2020-01-01")),
                    is_active=data.get("is_active", True),
                )

                self.topix500_symbols[symbol.symbol] = symbol

                # セクターマッピング構築
                if symbol.sector not in self.sector_mapping:
                    self.sector_mapping[symbol.sector] = []
                self.sector_mapping[symbol.sector].append(symbol.symbol)

            logger.info(
                f"TOPIX500マスターデータ読み込み完了: {len(self.topix500_symbols)}銘柄"
            )
            logger.info(f"セクター数: {len(self.sector_mapping)}")

            # セクター別銘柄数表示
            for sector, symbols in self.sector_mapping.items():
                logger.info(f"  {sector}: {len(symbols)}銘柄")

            return True

        except Exception as e:
            logger.error(f"TOPIX500マスターデータ読み込みエラー: {e}")
            traceback.print_exc()
            return False

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
        if self.enable_cache:
            cache_key = generate_unified_cache_key(
                "topix500_comprehensive", symbol, str(len(data))
            )
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug(f"包括分析キャッシュヒット: {symbol}")
                self.stats["cache_hits"] += 1
                return cached_result

        try:
            logger.debug(f"包括分析開始: {symbol}")

            # 並列実行で全分析を同時実行
            if self.enable_parallel and len(data) >= 50:
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
                if features_result:
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
                "multiframe_analysis": asdict(multiframe_result)
                if multiframe_result
                else None,
                "advanced_features": asdict(features_result)
                if features_result
                else None,
                "volatility_prediction": asdict(volatility_result)
                if volatility_result
                else None,
                "ml_ensemble": asdict(ml_ensemble_result)
                if ml_ensemble_result
                else None,
                "processing_time": time.time() - start_time,
            }

            # 統合スコア計算
            comprehensive_result[
                "integrated_score"
            ] = await self._calculate_integrated_score(
                multiframe_result,
                features_result,
                volatility_result,
                ml_ensemble_result,
            )

            # キャッシュ保存
            if self.enable_cache and cache_key:
                self.cache_manager.put(cache_key, comprehensive_result)

            logger.debug(
                f"包括分析完了: {symbol} - スコア: {comprehensive_result['integrated_score']:.3f} ({comprehensive_result['processing_time']:.3f}s)"
            )

            return comprehensive_result

        except Exception as e:
            logger.error(f"包括分析エラー: {symbol} - {e}")
            traceback.print_exc()
            return {
                "symbol": symbol,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "integrated_score": 0.0,
            }

    async def analyze_sector_batch(
        self, sector: str, symbols_data: Dict[str, pd.DataFrame]
    ) -> SectorAnalysisResult:
        """
        セクター別バッチ分析

        Args:
            sector: セクター名
            symbols_data: セクター内銘柄データ（symbol -> DataFrame）

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
                    task = self.analyze_single_symbol_comprehensive(symbol, data)
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
                    result = await self.analyze_single_symbol_comprehensive(
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
            for symbol, data in target_data.items():
                if symbol in self.topix500_symbols:
                    sector = self.topix500_symbols[symbol].sector
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
                    task = self.analyze_sector_batch(sector, sector_symbols_data)
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
                        sector_result = await self.analyze_sector_batch(
                            sector, sector_symbols_data
                        )
                        sector_results[sector] = sector_result
                        successful_analyses += sector_result.symbol_count
                    except Exception as e:
                        logger.error(f"セクター分析エラー: {sector} - {e}")
                        failed_analyses += len(sector_symbols_data)

            # 上位推奨銘柄抽出
            top_recommendations = await self._extract_top_recommendations(
                sector_results
            )

            # 市場全体概観
            market_overview = await self._calculate_market_overview(sector_results)

            # リスク分布
            risk_distribution = await self._calculate_risk_distribution(sector_results)

            # パフォーマンス統計
            processing_performance = {
                "total_processing_time": time.time() - start_time,
                "avg_sector_processing_time": np.mean(
                    [sr.processing_time for sr in sector_results.values()]
                )
                if sector_results
                else 0.0,
                "symbols_per_second": successful_analyses
                / max(time.time() - start_time, 0.001),
                "cache_hit_rate": self.stats["cache_hits"]
                / max(self.stats["total_analyses"], 1),
                "success_rate": successful_analyses
                / max(successful_analyses + failed_analyses, 1),
            }

            # 統計更新
            self.stats["total_analyses"] += 1
            self.stats["batch_analyses"] += 1
            self.stats["sector_analyses"] += len(sector_results)
            self.stats["successful_symbols"] += successful_analyses
            self.stats["failed_symbols"] += failed_analyses
            self.stats["processing_times"].append(
                processing_performance["total_processing_time"]
            )

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

    async def _generate_mock_topix500_data(self) -> List[Dict[str, Any]]:
        """TOPIX500模擬データ生成"""
        sectors = [
            "Technology",
            "Financials",
            "Consumer Discretionary",
            "Industrials",
            "Health Care",
            "Consumer Staples",
            "Materials",
            "Energy",
            "Utilities",
            "Real Estate",
            "Communication Services",
        ]

        industries = {
            "Technology": ["Software", "Semiconductors", "Hardware", "IT Services"],
            "Financials": ["Banks", "Insurance", "Securities", "REITs"],
            "Consumer Discretionary": ["Retail", "Automotive", "Media", "Restaurants"],
            "Industrials": ["Machinery", "Transportation", "Construction", "Aerospace"],
            "Health Care": [
                "Pharmaceuticals",
                "Medical Devices",
                "Biotechnology",
                "Healthcare Services",
            ],
        }

        symbols_data = []

        for i in range(500):
            sector = np.random.choice(sectors)
            industry_list = industries.get(sector, ["Other"])
            industry = np.random.choice(industry_list)

            # 銘柄コード生成（4桁数字）
            symbol = f"{1000 + i:04d}"

            # 市場キャップ（対数正規分布）
            market_cap = np.random.lognormal(22, 1.5)  # 約100億円中心

            # インデックス重み（市場キャップベース）
            weight = min(market_cap / 1000000000000 * 10, 5.0)  # 最大5%

            symbol_data = {
                "symbol": symbol,
                "name": f"Company_{symbol}",
                "sector": sector,
                "industry": industry,
                "market_cap": market_cap,
                "weight": weight,
                "listing_date": "2018-01-01",
                "is_active": True,
            }

            symbols_data.append(symbol_data)

        logger.info(f"TOPIX500模擬データ生成完了: {len(symbols_data)}銘柄")
        return symbols_data

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

    async def _extract_top_recommendations(
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

    async def _calculate_market_overview(
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

    async def _calculate_risk_distribution(
        self, sector_results: Dict[str, SectorAnalysisResult]
    ) -> Dict[str, int]:
        """リスク分布計算"""
        risk_counts = {"low": 0, "medium": 0, "high": 0, "extreme": 0}

        for result in sector_results.values():
            risk_counts[result.risk_level] += result.symbol_count

        return risk_counts

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

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        return {
            "total_analyses": self.stats["total_analyses"],
            "batch_analyses": self.stats["batch_analyses"],
            "sector_analyses": self.stats["sector_analyses"],
            "cache_hit_rate": self.stats["cache_hits"]
            / max(self.stats["total_analyses"], 1),
            "successful_symbols": self.stats["successful_symbols"],
            "failed_symbols": self.stats["failed_symbols"],
            "success_rate": self.stats["successful_symbols"]
            / max(self.stats["successful_symbols"] + self.stats["failed_symbols"], 1),
            "avg_processing_time": np.mean(self.stats["processing_times"])
            if self.stats["processing_times"]
            else 0.0,
            "system_status": {
                "cache_enabled": self.enable_cache,
                "parallel_enabled": self.enable_parallel,
                "max_concurrent_symbols": self.max_concurrent_symbols,
                "memory_limit_gb": self.memory_limit_gb,
                "processing_timeout": self.processing_timeout,
            },
            "topix500_data": {
                "loaded_symbols": len(self.topix500_symbols),
                "sectors": len(self.sector_mapping),
                "cache_size": len(self.symbol_data_cache),
            },
            "optimization_benefits": {
                "scale_expansion": "85銘柄 → 500銘柄 (6倍拡張)",
                "processing_target": "500銘柄を20秒以内",
                "memory_efficiency": "1GB以内メモリ使用",
                "sector_analysis": "セクター別最適化投資戦略",
            },
        }

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
        start_time = time.time()
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory

        try:
            logger.info(f"包括的バッチ分析開始: {len(stock_data)}銘柄")

            # データ検証
            valid_stock_data = {}
            for symbol, data in stock_data.items():
                if (
                    isinstance(data, pd.DataFrame)
                    and not data.empty
                    and len(data) >= 10
                ):
                    valid_stock_data[symbol] = data
                else:
                    logger.warning(f"無効データスキップ: {symbol}")

            logger.info(f"有効データ: {len(valid_stock_data)}/{len(stock_data)}銘柄")

            # バッチ処理実行
            if self.parallel_engine:
                # 並列処理でバッチ分析
                (
                    symbol_results,
                    processing_time,
                ) = self.parallel_engine.batch_process_symbols(
                    valid_stock_data,
                    use_cache=True,
                    timeout_per_symbol=min(30, self.processing_timeout),
                )
            else:
                # 順次処理フォールバック
                symbol_results = {}
                for symbol, data in valid_stock_data.items():
                    try:
                        # 簡易分析実行
                        result = {
                            "success": True,
                            "features": {
                                "price_change_pct": (
                                    data["Close"].iloc[-1] - data["Close"].iloc[0]
                                )
                                / data["Close"].iloc[0]
                                * 100
                            },
                            "prediction": {"signal": "HOLD", "confidence": 0.5},
                        }
                        symbol_results[symbol] = result
                    except Exception as e:
                        symbol_results[symbol] = {"success": False, "error": str(e)}

                processing_time = time.time() - start_time

            # メモリ使用量監視
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)

            # セクター分析実行
            sector_analysis = {}
            if enable_sector_analysis:
                try:
                    sector_analysis = await self._perform_sector_analysis(
                        valid_stock_data, symbol_results
                    )
                except Exception as e:
                    logger.error(f"セクター分析エラー: {e}")
                    sector_analysis = {}

            # 統計計算
            successful_count = sum(
                1 for r in symbol_results.values() if r.get("success", False)
            )
            failed_count = len(symbol_results) - successful_count
            cache_hit_rate = getattr(self.parallel_engine, "processing_stats", {}).get(
                "cache_hit_rate", 0.0
            )

            total_time = time.time() - start_time

            # パフォーマンスメトリクス作成
            performance_metrics = PerformanceMetrics(
                total_symbols=len(stock_data),
                successful_symbols=successful_count,
                failed_symbols=failed_count,
                processing_time_seconds=total_time,
                avg_time_per_symbol_ms=(total_time / len(stock_data) * 1000)
                if len(stock_data) > 0
                else 0.0,
                peak_memory_mb=peak_memory - initial_memory,
                cache_hit_rate=cache_hit_rate,
                throughput_symbols_per_second=len(stock_data) / total_time
                if total_time > 0
                else 0.0,
                sector_count=len(sector_analysis),
                error_messages=[],
            )

            # 統計更新
            self.stats["total_analyses"] += len(stock_data)
            self.stats["batch_analyses"] += 1
            self.stats["successful_symbols"] += successful_count
            self.stats["failed_symbols"] += failed_count
            self.stats["processing_times"].append(total_time)
            self.stats["memory_usage"].append(peak_memory)

            logger.info(
                f"包括的バッチ分析完了: {successful_count}/{len(stock_data)}銘柄 "
                f"{total_time:.2f}秒 ({performance_metrics.throughput_symbols_per_second:.1f}銘柄/秒)"
            )

            return {
                "symbol_results": symbol_results,
                "sector_analysis": sector_analysis,
                "performance_metrics": performance_metrics,
                "analysis_summary": {
                    "total_symbols": len(stock_data),
                    "successful_symbols": successful_count,
                    "failed_symbols": failed_count,
                    "success_rate": successful_count / len(stock_data)
                    if len(stock_data) > 0
                    else 0.0,
                    "processing_time_seconds": total_time,
                    "sectors_analyzed": len(sector_analysis),
                },
            }

        except Exception as e:
            logger.error(f"包括的バッチ分析エラー: {e}")
            total_time = time.time() - start_time

            return {
                "symbol_results": {},
                "sector_analysis": {},
                "performance_metrics": PerformanceMetrics(
                    total_symbols=len(stock_data),
                    successful_symbols=0,
                    failed_symbols=len(stock_data),
                    processing_time_seconds=total_time,
                    avg_time_per_symbol_ms=0.0,
                    peak_memory_mb=0.0,
                    cache_hit_rate=0.0,
                    throughput_symbols_per_second=0.0,
                    sector_count=0,
                    error_messages=[str(e)],
                ),
                "analysis_summary": {
                    "total_symbols": len(stock_data),
                    "successful_symbols": 0,
                    "failed_symbols": len(stock_data),
                    "success_rate": 0.0,
                    "processing_time_seconds": total_time,
                    "sectors_analyzed": 0,
                },
            }

    async def _perform_sector_analysis(
        self, stock_data: Dict[str, pd.DataFrame], symbol_results: Dict[str, Any]
    ) -> Dict[str, SectorAnalysisResult]:
        """セクター分析実行"""
        sector_groups = {}

        # セクター別銘柄グループ化
        for symbol in stock_data.keys():
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
                        "success_rate": len(successful_symbols) / len(symbols)
                        if len(symbols) > 0
                        else 0.0,
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

    def shutdown(self):
        """システムシャットダウン"""
        logger.info("TOPIX500分析システムシャットダウン開始")

        if self.parallel_engine and hasattr(self.parallel_engine, "shutdown"):
            self.parallel_engine.shutdown()

        if self.cache_manager and hasattr(self.cache_manager, "shutdown"):
            self.cache_manager.shutdown()

        logger.info("TOPIX500分析システムシャットダウン完了")
