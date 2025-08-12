#!/usr/bin/env python3
"""
特徴量パイプライン統合システム
Issue #380: 機械学習モデルの最適化 - 特徴量生成の効率化

特徴量ストアと既存MLシステムの統合インターフェース
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..analysis.feature_engineering_unified import FeatureConfig, FeatureResult
from ..core.optimization_strategy import OptimizationConfig, OptimizationLevel
from ..utils.logging_config import get_context_logger
from .feature_store import FeatureStoreConfig, create_feature_store

# パフォーマンス最適化エンジン統合
try:
    from ..performance import (
        GPUConfig,
        HFTConfig,
        get_gpu_accelerator,
        get_hft_optimizer,
    )

    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False

logger = get_context_logger(__name__)


@dataclass
class PipelineConfig:
    """パイプライン設定"""

    # 特徴量ストア設定
    feature_store_config: FeatureStoreConfig

    # 最適化設定
    optimization_config: OptimizationConfig

    # キャッシュ戦略
    cache_strategy: str = "aggressive"  # conservative, balanced, aggressive

    # 並列処理設定
    enable_parallel_generation: bool = True
    max_parallel_symbols: int = 4

    # バッチサイズ設定
    batch_size: int = 100

    # 自動クリーンアップ
    auto_cleanup: bool = True

    # HFT最適化設定
    enable_hft_optimization: bool = True
    hft_target_latency_us: float = 50.0

    # GPU加速設定
    enable_gpu_acceleration: bool = True
    gpu_batch_size: int = 512


class FeaturePipeline:
    """特徴量パイプライン - 統合特徴量生成システム"""

    def __init__(self, config: PipelineConfig = None):
        """初期化"""
        self.config = config or self._default_config()

        # 特徴量ストアの初期化
        self.feature_store = create_feature_store(self.config.feature_store_config)

        # パフォーマンス最適化エンジン初期化
        self.hft_optimizer = None
        self.gpu_accelerator = None

        if PERFORMANCE_OPTIMIZATION_AVAILABLE:
            if self.config.enable_hft_optimization:
                hft_config = HFTConfig(
                    target_latency_us=self.config.hft_target_latency_us,
                    preallocated_memory_mb=50,
                    enable_simd=True,
                )
                self.hft_optimizer = get_hft_optimizer(hft_config)

            if self.config.enable_gpu_acceleration:
                gpu_config = GPUConfig(
                    batch_size=self.config.gpu_batch_size,
                    gpu_memory_limit_mb=1024,
                    cpu_fallback=True,
                )
                self.gpu_accelerator = get_gpu_accelerator(gpu_config)

        # パフォーマンス統計
        self.pipeline_stats = {
            "total_requests": 0,
            "cache_efficiency": 0.0,
            "avg_generation_time": 0.0,
            "duplicate_calculations_avoided": 0,
            "total_time_saved_seconds": 0.0,
            # HFT統計
            "hft_predictions": 0,
            "hft_avg_latency_us": 0.0,
            "hft_under_target_rate": 0.0,
            # GPU統計
            "gpu_accelerated_operations": 0,
            "gpu_speedup_ratio": 1.0,
        }

        logger.info(
            "特徴量パイプライン初期化完了",
            extra={
                "config": {
                    "cache_strategy": self.config.cache_strategy,
                    "parallel_generation": self.config.enable_parallel_generation,
                    "batch_size": self.config.batch_size,
                }
            },
        )

    @classmethod
    def _default_config(cls) -> PipelineConfig:
        """デフォルト設定"""
        return PipelineConfig(
            feature_store_config=FeatureStoreConfig(
                base_path="data/features",
                max_cache_age_days=7,
                max_cache_size_mb=512,
                enable_compression=True,
            ),
            optimization_config=OptimizationConfig(
                level=OptimizationLevel.ADAPTIVE,
                performance_monitoring=True,
                cache_enabled=True,
            ),
        )

    def generate_features_for_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
        feature_config: FeatureConfig,
        force_regenerate: bool = False,
    ) -> FeatureResult:
        """単一銘柄の特徴量生成"""
        start_time = time.time()

        # 日付範囲の取得
        start_date = (
            data.index.min().strftime("%Y-%m-%d") if hasattr(data.index, "min") else "1900-01-01"
        )
        end_date = (
            data.index.max().strftime("%Y-%m-%d") if hasattr(data.index, "max") else "2100-01-01"
        )

        # 強制再生成でなければキャッシュを試行
        if not force_regenerate:
            cached_result = self.feature_store.load_feature(
                symbol, start_date, end_date, feature_config
            )
            if cached_result:
                generation_time = time.time() - start_time
                self.pipeline_stats["total_requests"] += 1
                logger.info(
                    "キャッシュから特徴量取得",
                    extra={
                        "symbol": symbol,
                        "cache_hit": True,
                        "generation_time_ms": round(generation_time * 1000, 2),
                    },
                )
                return cached_result

        # 特徴量生成
        logger.info(
            "特徴量生成実行",
            extra={
                "symbol": symbol,
                "data_shape": data.shape,
                "force_regenerate": force_regenerate,
            },
        )

        feature_result = self.feature_store.get_or_generate_feature(
            symbol=symbol,
            data=data,
            start_date=start_date,
            end_date=end_date,
            feature_config=feature_config,
            optimization_config=self.config.optimization_config,
        )

        generation_time = time.time() - start_time
        self.pipeline_stats["total_requests"] += 1
        self.pipeline_stats["avg_generation_time"] = (
            self.pipeline_stats["avg_generation_time"] * (self.pipeline_stats["total_requests"] - 1)
            + generation_time
        ) / self.pipeline_stats["total_requests"]

        return feature_result

    def batch_generate_features(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        feature_config: FeatureConfig,
        force_regenerate: bool = False,
    ) -> Dict[str, FeatureResult]:
        """バッチ特徴量生成"""
        logger.info(
            "バッチ特徴量生成開始",
            extra={
                "symbols_count": len(symbols_data),
                "force_regenerate": force_regenerate,
                "batch_size": self.config.batch_size,
            },
        )

        results = {}
        total_symbols = len(symbols_data)
        processed_symbols = 0

        # バッチ処理
        symbol_batches = self._create_batches(list(symbols_data.keys()), self.config.batch_size)

        for batch_idx, symbol_batch in enumerate(symbol_batches):
            batch_start_time = time.time()

            logger.info(
                f"バッチ {batch_idx + 1}/{len(symbol_batches)} 処理開始",
                extra={"batch_symbols": len(symbol_batch), "symbols": symbol_batch},
            )

            # バッチ内処理
            batch_data = {symbol: symbols_data[symbol] for symbol in symbol_batch}

            if not force_regenerate:
                # バッチキャッシュ確認・生成
                batch_results = self._batch_process_with_cache(batch_data, feature_config)
            else:
                # 強制再生成
                batch_results = self._batch_process_force_regenerate(batch_data, feature_config)

            results.update(batch_results)
            processed_symbols += len(batch_results)

            batch_time = time.time() - batch_start_time
            logger.info(
                f"バッチ {batch_idx + 1} 完了",
                extra={
                    "processed_symbols": len(batch_results),
                    "batch_time_seconds": round(batch_time, 2),
                    "symbols_per_second": round(len(batch_results) / batch_time, 2),
                },
            )

        # 統計更新
        store_stats = self.feature_store.get_stats()
        self.pipeline_stats["cache_efficiency"] = store_stats.get("cache_hit_rate_percent", 0)

        logger.info(
            "バッチ特徴量生成完了",
            extra={
                "total_symbols": total_symbols,
                "processed_symbols": processed_symbols,
                "success_rate": f"{processed_symbols/total_symbols*100:.1f}%",
                "cache_hit_rate": f"{store_stats.get('cache_hit_rate_percent', 0):.1f}%",
            },
        )

        return results

    async def ultra_fast_prediction(
        self, symbol: str, prices: np.ndarray, volumes: np.ndarray, model_weights: np.ndarray = None
    ) -> Dict[str, Any]:
        """超高速予測（HFT対応 <50μs目標）"""
        if not PERFORMANCE_OPTIMIZATION_AVAILABLE or self.hft_optimizer is None:
            return self._fallback_prediction(symbol, prices, volumes)

        start_time = time.perf_counter_ns()

        try:
            # デフォルトモデル重み
            if model_weights is None:
                model_weights = np.random.normal(0, 0.1, 8).astype(np.float64)

            # HFT最適化エンジンで予測
            result = self.hft_optimizer.predict_ultra_fast(prices, volumes)

            # 統計更新
            if result.get("under_target", False):
                self.pipeline_stats["hft_predictions"] += 1
                current_avg = self.pipeline_stats["hft_avg_latency_us"]
                total_preds = self.pipeline_stats["hft_predictions"]
                latency = result.get("latency_us", 0)

                self.pipeline_stats["hft_avg_latency_us"] = (
                    current_avg * (total_preds - 1) + latency
                ) / total_preds

                # 目標達成率更新
                under_target_count = sum(
                    1 for _ in range(total_preds) if _ < self.config.hft_target_latency_us
                )
                self.pipeline_stats["hft_under_target_rate"] = under_target_count / total_preds

            total_time_us = (time.perf_counter_ns() - start_time) / 1000.0

            return {
                "symbol": symbol,
                "prediction": result.get("prediction", 0.0),
                "confidence": min(1.0, 1.0 / max(result.get("latency_us", 1), 1)),
                "latency_us": total_time_us,
                "hft_optimized": True,
                "under_target": total_time_us < self.config.hft_target_latency_us,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"HFT予測エラー {symbol}: {e}")
            return self._fallback_prediction(symbol, prices, volumes)

    def _fallback_prediction(
        self, symbol: str, prices: np.ndarray, volumes: np.ndarray
    ) -> Dict[str, Any]:
        """フォールバック予測"""
        start_time = time.perf_counter_ns()

        # 簡単な移動平均予測
        if len(prices) >= 5:
            prediction = float(np.mean(prices[-5:]))
        else:
            prediction = float(prices[-1]) if len(prices) > 0 else 0.0

        latency_us = (time.perf_counter_ns() - start_time) / 1000.0

        return {
            "symbol": symbol,
            "prediction": prediction,
            "confidence": 0.5,
            "latency_us": latency_us,
            "hft_optimized": False,
            "under_target": False,
            "timestamp": time.time(),
        }

    async def gpu_batch_feature_generation(
        self, symbols_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """GPU加速バッチ特徴量生成"""
        if not PERFORMANCE_OPTIMIZATION_AVAILABLE or self.gpu_accelerator is None:
            return self._cpu_batch_features(symbols_data)

        start_time = time.perf_counter()

        try:
            # データ準備
            all_features = {}

            for symbol, data in symbols_data.items():
                prices = data.get("prices", np.array([]))
                volumes = data.get("volumes", np.array([]))

                if len(prices) > 0 and len(volumes) > 0:
                    # GPU特徴量計算
                    features = self.gpu_accelerator.compute_features_gpu(
                        prices, volumes, feature_dim=7
                    )

                    if features is not None:
                        all_features[symbol] = features

            # 統計更新
            self.pipeline_stats["gpu_accelerated_operations"] += len(all_features)

            gpu_time = time.perf_counter() - start_time
            logger.info(f"GPU特徴量生成完了: {len(all_features)}銘柄, {gpu_time*1000:.2f}ms")

            return all_features

        except Exception as e:
            logger.error(f"GPU特徴量生成エラー: {e}")
            return self._cpu_batch_features(symbols_data)

    def _cpu_batch_features(
        self, symbols_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """CPU フォールバック特徴量生成"""
        results = {}

        for symbol, data in symbols_data.items():
            prices = data.get("prices", np.array([]))
            volumes = data.get("volumes", np.array([]))

            if len(prices) >= 20 and len(volumes) >= 20:
                # 基本的な特徴量のみ
                n = len(prices)
                features = np.zeros((n, 5))

                for i in range(20, n):
                    features[i, 0] = np.mean(prices[i - 5 : i])  # MA5
                    features[i, 1] = np.mean(prices[i - 20 : i])  # MA20
                    features[i, 2] = prices[i]  # 現在価格
                    features[i, 3] = volumes[i]  # 現在ボリューム
                    features[i, 4] = (prices[i] - prices[i - 1]) / prices[i - 1]  # 変化率

                results[symbol] = features

        return results

    def _create_batches(self, items: List[str], batch_size: int) -> List[List[str]]:
        """バッチ作成"""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i : i + batch_size])
        return batches

    def _batch_process_with_cache(
        self, batch_data: Dict[str, pd.DataFrame], feature_config: FeatureConfig
    ) -> Dict[str, FeatureResult]:
        """キャッシュ考慮バッチ処理"""
        results = {}

        # 日付範囲の統一（バッチ内で統一）
        all_dates = []
        for data in batch_data.values():
            if hasattr(data.index, "min") and hasattr(data.index, "max"):
                all_dates.extend([data.index.min(), data.index.max()])

        if all_dates:
            start_date = min(all_dates).strftime("%Y-%m-%d")
            end_date = max(all_dates).strftime("%Y-%m-%d")
        else:
            start_date, end_date = "1900-01-01", "2100-01-01"

        # バッチ特徴量生成（FeatureStoreのbatch_generate_featuresを使用）
        batch_results = self.feature_store.batch_generate_features(
            symbols=list(batch_data.keys()),
            data_dict=batch_data,
            start_date=start_date,
            end_date=end_date,
            feature_config=feature_config,
            optimization_config=self.config.optimization_config,
        )

        return batch_results

    def _batch_process_force_regenerate(
        self, batch_data: Dict[str, pd.DataFrame], feature_config: FeatureConfig
    ) -> Dict[str, FeatureResult]:
        """強制再生成バッチ処理"""
        results = {}

        for symbol, data in batch_data.items():
            try:
                result = self.generate_features_for_symbol(
                    symbol, data, feature_config, force_regenerate=True
                )
                results[symbol] = result
            except Exception as e:
                logger.error(f"強制再生成エラー: {symbol} - {e}")
                continue

        return results

    def optimize_feature_config(
        self, symbol: str, data: pd.DataFrame, base_config: FeatureConfig = None
    ) -> FeatureConfig:
        """特徴量設定の最適化"""
        if base_config is None:
            base_config = FeatureConfig.default()

        # データサイズベースの最適化
        data_size = len(data)

        optimized_config = base_config

        if data_size > 100000:
            # 大規模データの場合
            optimized_config.chunk_size = 20000
            optimized_config.enable_vectorization = True
            optimized_config.enable_dtype_optimization = True
            optimized_config.enable_copy_elimination = True
        elif data_size > 10000:
            # 中規模データの場合
            optimized_config.chunk_size = 10000
            optimized_config.enable_vectorization = True
            optimized_config.enable_dtype_optimization = True
        else:
            # 小規模データの場合
            optimized_config.chunk_size = 5000
            optimized_config.enable_vectorization = False

        logger.info(
            "特徴量設定最適化完了",
            extra={
                "symbol": symbol,
                "data_size": data_size,
                "chunk_size": optimized_config.chunk_size,
                "vectorization": optimized_config.enable_vectorization,
                "dtype_optimization": optimized_config.enable_dtype_optimization,
            },
        )

        return optimized_config

    def precompute_features_for_symbols(
        self,
        symbols: List[str],
        data_provider_func: callable,
        feature_config: FeatureConfig = None,
    ) -> Dict[str, str]:
        """銘柄リストの特徴量事前計算"""
        if feature_config is None:
            feature_config = FeatureConfig.default()

        logger.info("特徴量事前計算開始", extra={"symbols_count": len(symbols)})

        feature_ids = {}
        success_count = 0

        for symbol in symbols:
            try:
                # データ取得
                data = data_provider_func(symbol)
                if data is None or data.empty:
                    logger.warning(f"データが取得できませんでした: {symbol}")
                    continue

                # 特徴量生成
                result = self.generate_features_for_symbol(symbol, data, feature_config)
                feature_ids[symbol] = result.metadata.get("feature_id", "")
                success_count += 1

            except Exception as e:
                logger.error(f"特徴量事前計算エラー: {symbol} - {e}")
                continue

        logger.info(
            "特徴量事前計算完了",
            extra={
                "success_count": success_count,
                "total_count": len(symbols),
                "success_rate": f"{success_count/len(symbols)*100:.1f}%",
            },
        )

        return feature_ids

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """パイプライン統計の取得"""
        store_stats = self.feature_store.get_stats()

        stats = {
            **self.pipeline_stats,
            "feature_store_stats": store_stats,
            "cache_size_mb": store_stats.get("cache_size_mb", 0),
            "features_in_cache": store_stats.get("features_in_cache", 0),
        }

        # パフォーマンス最適化統計追加
        if PERFORMANCE_OPTIMIZATION_AVAILABLE:
            if self.hft_optimizer:
                hft_report = self.hft_optimizer.get_optimization_report()
                stats["hft_optimization"] = hft_report

            if self.gpu_accelerator:
                gpu_report = self.gpu_accelerator.get_gpu_report()
                stats["gpu_acceleration"] = gpu_report

        return stats

    def cleanup(self, force: bool = False):
        """リソースクリーンアップ"""
        if self.config.auto_cleanup or force:
            self.feature_store.cleanup_cache(force=force)

            # パフォーマンス最適化エンジンクリーンアップ
            if self.hft_optimizer:
                self.hft_optimizer.cleanup()
            if self.gpu_accelerator:
                self.gpu_accelerator.cleanup()

            logger.info("パイプラインクリーンアップ完了")

    def __enter__(self):
        """コンテキストマネージャー入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー出口"""
        self.cleanup()


# 便利関数
def create_feature_pipeline(config: PipelineConfig = None) -> FeaturePipeline:
    """特徴量パイプラインの作成"""
    return FeaturePipeline(config)


def create_optimized_pipeline() -> FeaturePipeline:
    """最適化されたパイプラインの作成"""
    config = PipelineConfig(
        feature_store_config=FeatureStoreConfig(
            base_path="data/features_optimized",
            max_cache_age_days=14,
            max_cache_size_mb=1024,
            enable_compression=True,
            cleanup_on_startup=True,
        ),
        optimization_config=OptimizationConfig(
            level=OptimizationLevel.ADAPTIVE,
            performance_monitoring=True,
            cache_enabled=True,
            parallel_processing=True,
        ),
        cache_strategy="aggressive",
        enable_parallel_generation=True,
        batch_size=50,
    )
    return FeaturePipeline(config)
