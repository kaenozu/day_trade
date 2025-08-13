#!/usr/bin/env python3
"""
特徴量パイプライン統合システム
Issue #380: 機械学習モデルの最適化 - 特徴量生成の効率化

特徴量ストアと既存MLシステムの統合インターフェース
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# 並列処理ライブラリ
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

import multiprocessing as mp

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
except Exception:
    # Issue #715テスト時: パフォーマンス最適化を一時的に無効化
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
    max_parallel_symbols: int = min(4, max(1, mp.cpu_count() - 1))  # CPU数に応じて調整
    parallel_backend: str = 'threading'  # 'threading', 'multiprocessing', 'joblib'
    enable_batch_parallel: bool = True  # バッチ内並列処理
    enable_symbol_parallel: bool = True  # シンボル間並列処理

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
            # Issue #715対応: 並列処理統計
            "parallel_batches_processed": 0,
            "parallel_symbols_processed": 0,
            "parallel_time_saved_seconds": 0.0,
            "parallel_efficiency_ratio": 1.0,
            "parallel_backend_used": self.config.parallel_backend,
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
            data.index.min().strftime("%Y-%m-%d")
            if hasattr(data.index, "min")
            else "1900-01-01"
        )
        end_date = (
            data.index.max().strftime("%Y-%m-%d")
            if hasattr(data.index, "max")
            else "2100-01-01"
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
            self.pipeline_stats["avg_generation_time"]
            * (self.pipeline_stats["total_requests"] - 1)
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
        symbol_batches = self._create_batches(
            list(symbols_data.keys()), self.config.batch_size
        )

        for batch_idx, symbol_batch in enumerate(symbol_batches):
            batch_start_time = time.time()

            logger.info(
                f"バッチ {batch_idx + 1}/{len(symbol_batches)} 処理開始",
                extra={
                    "batch_symbols": len(symbol_batch),
                    "symbols": symbol_batch,
                    "parallel_enabled": self.config.enable_parallel_generation and self.config.enable_batch_parallel
                },
            )

            # バッチ内処理
            batch_data = {symbol: symbols_data[symbol] for symbol in symbol_batch}

            # Issue #715対応: 並列処理の適用
            if self.config.enable_parallel_generation and self.config.enable_batch_parallel and len(symbol_batch) > 1:
                # 並列バッチ処理
                batch_results = self._parallel_process_symbols(batch_data, feature_config, force_regenerate)
            else:
                # 従来の順次処理
                if not force_regenerate:
                    # バッチキャッシュ確認・生成
                    batch_results = self._batch_process_with_cache(
                        batch_data, feature_config
                    )
                else:
                    # 強制再生成
                    batch_results = self._batch_process_force_regenerate(
                        batch_data, feature_config
                    )

            results.update(batch_results)
            processed_symbols += len(batch_results)

            batch_time = time.time() - batch_start_time
            symbols_per_second = len(batch_results) / max(batch_time, 0.001)  # ゼロ除算回避

            logger.info(
                f"バッチ {batch_idx + 1} 完了",
                extra={
                    "processed_symbols": len(batch_results),
                    "batch_time_seconds": round(batch_time, 2),
                    "symbols_per_second": round(symbols_per_second, 2),
                    "parallel_processing": self.config.enable_parallel_generation and self.config.enable_batch_parallel,
                    "backend": self.config.parallel_backend if self.config.enable_parallel_generation else "sequential"
                },
            )

        # 統計更新
        store_stats = self.feature_store.get_stats()
        self.pipeline_stats["cache_efficiency"] = store_stats.get(
            "cache_hit_rate_percent", 0
        )

        logger.info(
            "バッチ特徴量生成完了",
            extra={
                "total_symbols": total_symbols,
                "processed_symbols": processed_symbols,
                "success_rate": f"{processed_symbols / total_symbols * 100:.1f}%",
                "cache_hit_rate": f"{store_stats.get('cache_hit_rate_percent', 0):.1f}%",
            },
        )

        return results

    async def ultra_fast_prediction(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        model_weights: np.ndarray = None,
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
                    1
                    for _ in range(total_preds)
                    if _ < self.config.hft_target_latency_us
                )
                self.pipeline_stats["hft_under_target_rate"] = (
                    under_target_count / total_preds
                )

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
            logger.info(
                f"GPU特徴量生成完了: {len(all_features)}銘柄, {gpu_time * 1000:.2f}ms"
            )

            return all_features

        except Exception as e:
            logger.error(f"GPU特徴量生成エラー: {e}")
            return self._cpu_batch_features(symbols_data)

    def _cpu_batch_features(
        self, symbols_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """CPU フォールバック特徴量生成"""
        # Issue #715対応: CPU処理も並列化可能にする
        if self.config.enable_parallel_generation and len(symbols_data) > 1:
            return self._cpu_batch_features_parallel(symbols_data)
        else:
            return self._cpu_batch_features_sequential(symbols_data)

    def _cpu_batch_features_sequential(
        self, symbols_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """CPU順次特徴量生成（Issue #716対応: 最適化バッチ処理使用）"""
        # Issue #716対応: 最適化されたバッチ処理メソッドを使用
        return self._compute_features_optimized_batch(symbols_data)

    def _cpu_batch_features_parallel(
        self, symbols_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """CPU並列特徴量生成（Issue #716対応: 最適化バッチ処理使用）"""
        # Issue #716対応: まず最適化バッチ処理を試行
        try:
            # 最適化されたバッチ処理（同じ長さのデータをグループ化してベクトル化）
            batch_results = self._compute_features_optimized_batch(symbols_data)
            if len(batch_results) > 0:
                return batch_results
        except Exception as e:
            logger.warning(f"最適化バッチ処理失敗、個別並列処理にフォールバック: {e}")

        # フォールバック: 従来の個別シンボル並列処理
        results = {}

        if self.config.parallel_backend == 'joblib' and JOBLIB_AVAILABLE:
            try:
                results_list = Parallel(n_jobs=self.config.max_parallel_symbols)(
                    delayed(self._compute_single_symbol_features)(symbol, data)
                    for symbol, data in symbols_data.items()
                )

                # 結果を辞書に変換（Noneでないもの）
                for i, (symbol, data) in enumerate(symbols_data.items()):
                    if i < len(results_list) and results_list[i] is not None:
                        results[symbol] = results_list[i]

            except Exception as e:
                logger.error(f"CPU並列特徴量生成エラー: {e}")
                return self._cpu_batch_features_sequential(symbols_data)
        else:
            # Threading並列処理
            with ThreadPoolExecutor(max_workers=self.config.max_parallel_symbols) as executor:
                future_to_symbol = {
                    executor.submit(self._compute_single_symbol_features, symbol, data): symbol
                    for symbol, data in symbols_data.items()
                }

                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[symbol] = result
                    except Exception as e:
                        logger.error(f"CPU Threading特徴量生成エラー ({symbol}): {e}")

        return results

    def _compute_single_symbol_features(
        self, symbol: str, data: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        単一シンボル特徴量計算（Issue #716対応: ベクトル化最適化）
        """
        try:
            prices = data.get("prices", np.array([]))
            volumes = data.get("volumes", np.array([]))

            if len(prices) >= 20 and len(volumes) >= 20:
                return self._compute_features_vectorized(prices, volumes)

            return None

        except Exception as e:
            logger.error(f"単一シンボル特徴量計算エラー ({symbol}): {e}")
            return None

    def _compute_features_vectorized(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Issue #716対応: ベクトル化された特徴量計算

        Args:
            prices: 価格配列
            volumes: 出来高配列

        Returns:
            特徴量配列 (n_samples, n_features)
        """
        n = len(prices)
        features = np.zeros((n, 5))

        # Issue #716対応: Pythonループを排除し、NumPyベクトル化操作で置換

        # MA5 (5日移動平均) - ベクトル化計算
        # pandas.rolling風の実装をnumpyで実現
        ma5 = np.zeros(n)
        ma5[4:] = np.convolve(prices, np.ones(5)/5, mode='valid')
        features[:, 0] = ma5

        # MA20 (20日移動平均) - ベクトル化計算
        ma20 = np.zeros(n)
        ma20[19:] = np.convolve(prices, np.ones(20)/20, mode='valid')
        features[:, 1] = ma20

        # 現在価格 - そのまま代入
        features[:, 2] = prices

        # 現在ボリューム - そのまま代入
        features[:, 3] = volumes

        # 変化率 - ベクトル化計算
        price_changes = np.zeros(n)
        price_changes[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        features[:, 4] = price_changes

        return features

    def _compute_features_optimized_batch(self, symbols_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Issue #716対応: バッチ最適化された特徴量計算

        複数シンボルのデータを一度に処理し、メモリアクセスパターンを最適化

        Args:
            symbols_data: シンボル別データ辞書

        Returns:
            シンボル別特徴量辞書
        """
        results = {}

        # データの前処理とバリデーション
        valid_symbols = []
        valid_prices = []
        valid_volumes = []

        for symbol, data in symbols_data.items():
            prices = data.get("prices", np.array([]))
            volumes = data.get("volumes", np.array([]))

            if len(prices) >= 20 and len(volumes) >= 20:
                valid_symbols.append(symbol)
                valid_prices.append(prices)
                valid_volumes.append(volumes)

        if not valid_symbols:
            return results

        # バッチ処理: 同じ長さのデータをグループ化
        length_groups = {}
        for i, (symbol, prices, volumes) in enumerate(zip(valid_symbols, valid_prices, valid_volumes)):
            length = len(prices)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append((i, symbol, prices, volumes))

        # 長さごとにバッチ処理
        for length, group_data in length_groups.items():
            if len(group_data) == 1:
                # 単一シンボル: 通常の処理
                _, symbol, prices, volumes = group_data[0]
                results[symbol] = self._compute_features_vectorized(prices, volumes)
            else:
                # 複数シンボル: バッチベクトル化処理
                batch_results = self._compute_features_batch_vectorized(group_data)
                results.update(batch_results)

        return results

    def _compute_features_batch_vectorized(self, group_data: List[Tuple[int, str, np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Issue #716対応: 同じ長さのデータに対するバッチベクトル化処理

        Args:
            group_data: (index, symbol, prices, volumes) のリスト

        Returns:
            シンボル別特徴量辞書
        """
        if not group_data:
            return {}

        # データを3次元配列に変換 (n_symbols, n_timepoints, n_features)
        n_symbols = len(group_data)
        n_timepoints = len(group_data[0][2])  # すべて同じ長さ

        # 価格と出来高データを2次元配列に変換
        prices_batch = np.array([data[2] for data in group_data])  # (n_symbols, n_timepoints)
        volumes_batch = np.array([data[3] for data in group_data])  # (n_symbols, n_timepoints)

        # バッチ特徴量配列を初期化
        features_batch = np.zeros((n_symbols, n_timepoints, 5))

        # Issue #716対応: MA5とMA20をより効率的に計算（ループを最小化）
        # MA5: より効率的な実装（先頭ゼロ埋め + 有効部分のみconvolve）
        for i in range(n_symbols):
            if n_timepoints >= 5:
                # 最初の4つは0、5番目からconvolveの結果
                ma5_valid = np.convolve(prices_batch[i], np.ones(5)/5, mode='valid')
                features_batch[i, 4:, 0] = ma5_valid
            # features_batch[i, :4, 0] は初期化時に既に0

        # MA20: より効率的な実装
        for i in range(n_symbols):
            if n_timepoints >= 20:
                # 最初の19個は0、20番目からconvolveの結果
                ma20_valid = np.convolve(prices_batch[i], np.ones(20)/20, mode='valid')
                features_batch[i, 19:, 1] = ma20_valid
            # features_batch[i, :19, 1] は初期化時に既に0

        # 現在価格とボリューム: 直接代入
        features_batch[:, :, 2] = prices_batch
        features_batch[:, :, 3] = volumes_batch

        # 変化率: バッチ計算
        price_changes_batch = np.zeros_like(prices_batch)
        price_changes_batch[:, 1:] = (prices_batch[:, 1:] - prices_batch[:, :-1]) / prices_batch[:, :-1]
        features_batch[:, :, 4] = price_changes_batch

        # 結果を辞書に変換
        results = {}
        for i, (_, symbol, _, _) in enumerate(group_data):
            results[symbol] = features_batch[i]

        return results

    def _process_single_symbol_in_batch(
        self, symbol: str, data: pd.DataFrame, feature_config: FeatureConfig,
        force_regenerate: bool
    ) -> Tuple[str, Optional[FeatureResult]]:
        """
        Issue #715対応: バッチ内単一シンボル処理（並列化用）

        Args:
            symbol: シンボル名
            data: データ
            feature_config: 特徴量設定
            force_regenerate: 強制再生成フラグ

        Returns:
            (シンボル, 特徴量結果) のタプル
        """
        try:
            if not force_regenerate:
                # キャッシュから取得を試行
                start_date = data.index.min().strftime("%Y-%m-%d") if hasattr(data.index, "min") else "1900-01-01"
                end_date = data.index.max().strftime("%Y-%m-%d") if hasattr(data.index, "max") else "2100-01-01"

                cached_result = self.feature_store.load_feature(
                    symbol, start_date, end_date, feature_config
                )
                if cached_result:
                    return symbol, cached_result

            # 特徴量生成
            result = self.feature_store.get_or_generate_feature(
                symbol=symbol,
                data=data,
                start_date=data.index.min().strftime("%Y-%m-%d") if hasattr(data.index, "min") else "1900-01-01",
                end_date=data.index.max().strftime("%Y-%m-%d") if hasattr(data.index, "max") else "2100-01-01",
                feature_config=feature_config,
                optimization_config=self.config.optimization_config,
            )
            return symbol, result

        except Exception as e:
            logger.error(f"バッチ内シンボル処理エラー: {symbol} - {e}")
            return symbol, None

    def _parallel_process_symbols(
        self, symbols_data: Dict[str, pd.DataFrame], feature_config: FeatureConfig,
        force_regenerate: bool = False
    ) -> Dict[str, FeatureResult]:
        """
        Issue #715対応: シンボル並列処理

        Args:
            symbols_data: シンボル別データ辞書
            feature_config: 特徴量設定
            force_regenerate: 強制再生成フラグ

        Returns:
            シンボル別特徴量結果辞書
        """
        results = {}
        symbols = list(symbols_data.keys())

        if not self.config.enable_parallel_generation or len(symbols) <= 1:
            # 並列処理無効または単一シンボルの場合は順次処理
            for symbol in symbols:
                symbol_result, feature_result = self._process_single_symbol_in_batch(
                    symbol, symbols_data[symbol], feature_config, force_regenerate
                )
                if feature_result:
                    results[symbol_result] = feature_result
            return results

        # 並列処理実行
        if self.config.parallel_backend == 'joblib' and JOBLIB_AVAILABLE:
            return self._parallel_process_joblib(symbols_data, feature_config, force_regenerate)
        elif self.config.parallel_backend == 'multiprocessing':
            return self._parallel_process_multiprocessing(symbols_data, feature_config, force_regenerate)
        else:  # threading (default)
            return self._parallel_process_threading(symbols_data, feature_config, force_regenerate)

    def _parallel_process_joblib(
        self, symbols_data: Dict[str, pd.DataFrame], feature_config: FeatureConfig,
        force_regenerate: bool
    ) -> Dict[str, FeatureResult]:
        """Joblib並列処理"""
        try:
            results_list = Parallel(n_jobs=self.config.max_parallel_symbols)(
                delayed(self._process_single_symbol_in_batch)(
                    symbol, data, feature_config, force_regenerate
                )
                for symbol, data in symbols_data.items()
            )

            results = {}
            for symbol, result in results_list:
                if result:
                    results[symbol] = result

            logger.info(f"Joblib並列処理完了: {len(results)}/{len(symbols_data)} シンボル成功")
            return results

        except Exception as e:
            logger.error(f"Joblib並列処理エラー: {e}")
            # フォールバック: 順次処理
            return self._parallel_process_threading(symbols_data, feature_config, force_regenerate)

    def _parallel_process_threading(
        self, symbols_data: Dict[str, pd.DataFrame], feature_config: FeatureConfig,
        force_regenerate: bool
    ) -> Dict[str, FeatureResult]:
        """Threading並列処理"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_symbols) as executor:
            # 全タスクをサブミット
            future_to_symbol = {
                executor.submit(
                    self._process_single_symbol_in_batch,
                    symbol, data, feature_config, force_regenerate
                ): symbol
                for symbol, data in symbols_data.items()
            }

            # 結果収集
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result_symbol, result = future.result()
                    if result:
                        results[result_symbol] = result
                except Exception as e:
                    logger.error(f"Threading並列処理エラー ({symbol}): {e}")

        logger.info(f"Threading並列処理完了: {len(results)}/{len(symbols_data)} シンボル成功")
        return results

    def _parallel_process_multiprocessing(
        self, symbols_data: Dict[str, pd.DataFrame], feature_config: FeatureConfig,
        force_regenerate: bool
    ) -> Dict[str, FeatureResult]:
        """Multiprocessing並列処理（注意：プロセス間でのオブジェクト共有制限あり）"""
        # 注意: multiprocessingは複雑なオブジェクト（self.feature_store等）の共有が困難
        # そのため、threading方式にフォールバック
        logger.warning("Multiprocessing並列処理はthreadingにフォールバック")
        return self._parallel_process_threading(symbols_data, feature_config, force_regenerate)

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

        logger.info(
            "特徴量事前計算開始",
            extra={
                "symbols_count": len(symbols),
                "parallel_enabled": self.config.enable_parallel_generation and self.config.enable_symbol_parallel,
                "max_workers": self.config.max_parallel_symbols
            }
        )

        # Issue #715対応: シンボル間並列処理
        if self.config.enable_parallel_generation and self.config.enable_symbol_parallel and len(symbols) > 1:
            return self._precompute_parallel(symbols, data_provider_func, feature_config)
        else:
            return self._precompute_sequential(symbols, data_provider_func, feature_config)

    def _precompute_sequential(
        self, symbols: List[str], data_provider_func: callable, feature_config: FeatureConfig
    ) -> Dict[str, str]:
        """順次事前計算"""
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
            "順次事前計算完了",
            extra={
                "success_count": success_count,
                "total_count": len(symbols),
                "success_rate": f"{success_count / len(symbols) * 100:.1f}%",
            },
        )

        return feature_ids

    def _precompute_parallel(
        self, symbols: List[str], data_provider_func: callable, feature_config: FeatureConfig
    ) -> Dict[str, str]:
        """並列事前計算"""
        feature_ids = {}

        # 並列処理バックエンドの選択
        if self.config.parallel_backend == 'joblib' and JOBLIB_AVAILABLE:
            return self._precompute_parallel_joblib(symbols, data_provider_func, feature_config)
        else:  # threading
            return self._precompute_parallel_threading(symbols, data_provider_func, feature_config)

    def _precompute_single_symbol(
        self, symbol: str, data_provider_func: callable, feature_config: FeatureConfig
    ) -> Tuple[str, Optional[str]]:
        """単一シンボル事前計算（並列化用）"""
        try:
            # データ取得
            data = data_provider_func(symbol)
            if data is None or data.empty:
                logger.warning(f"データが取得できませんでした: {symbol}")
                return symbol, None

            # 特徴量生成
            result = self.generate_features_for_symbol(symbol, data, feature_config)
            feature_id = result.metadata.get("feature_id", "")
            return symbol, feature_id

        except Exception as e:
            logger.error(f"特徴量事前計算エラー: {symbol} - {e}")
            return symbol, None

    def _precompute_parallel_joblib(
        self, symbols: List[str], data_provider_func: callable, feature_config: FeatureConfig
    ) -> Dict[str, str]:
        """Joblib並列事前計算"""
        try:
            results_list = Parallel(n_jobs=self.config.max_parallel_symbols)(
                delayed(self._precompute_single_symbol)(symbol, data_provider_func, feature_config)
                for symbol in symbols
            )

            feature_ids = {}
            success_count = 0
            for symbol, feature_id in results_list:
                if feature_id:
                    feature_ids[symbol] = feature_id
                    success_count += 1

            logger.info(
                "Joblib並列事前計算完了",
                extra={
                    "success_count": success_count,
                    "total_count": len(symbols),
                    "success_rate": f"{success_count / len(symbols) * 100:.1f}%",
                    "backend": "joblib"
                },
            )

            return feature_ids

        except Exception as e:
            logger.error(f"Joblib並列事前計算エラー: {e}")
            # フォールバック: threading
            return self._precompute_parallel_threading(symbols, data_provider_func, feature_config)

    def _precompute_parallel_threading(
        self, symbols: List[str], data_provider_func: callable, feature_config: FeatureConfig
    ) -> Dict[str, str]:
        """Threading並列事前計算"""
        feature_ids = {}
        success_count = 0

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_symbols) as executor:
            # 全タスクをサブミット
            future_to_symbol = {
                executor.submit(self._precompute_single_symbol, symbol, data_provider_func, feature_config): symbol
                for symbol in symbols
            }

            # 結果収集
            for future in as_completed(future_to_symbol):
                original_symbol = future_to_symbol[future]
                try:
                    symbol, feature_id = future.result()
                    if feature_id:
                        feature_ids[symbol] = feature_id
                        success_count += 1
                except Exception as e:
                    logger.error(f"Threading並列事前計算エラー ({original_symbol}): {e}")

        logger.info(
            "Threading並列事前計算完了",
            extra={
                "success_count": success_count,
                "total_count": len(symbols),
                "success_rate": f"{success_count / len(symbols) * 100:.1f}%",
                "backend": "threading"
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
