#!/usr/bin/env python3
"""
バッチ処理モジュール

特徴量の並列バッチ生成機能を提供します。
Issue #719対応: バッチ処理並列化対応
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base import JOBLIB_AVAILABLE, FeatureStoreConfig
from .feature_operations import FeatureOperations
from ...analysis.feature_engineering_unified import FeatureConfig, FeatureResult
from ...core.optimization_strategy import OptimizationConfig
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

if JOBLIB_AVAILABLE:
    from joblib import Parallel, delayed


class BatchProcessor:
    """
    バッチ処理クラス
    
    複数シンボルの特徴量を並列または順次で生成します。
    """

    def __init__(
        self,
        config: FeatureStoreConfig,
        feature_operations: FeatureOperations,
    ):
        """
        初期化
        
        Args:
            config: 特徴量ストア設定
            feature_operations: 特徴量操作クラス
        """
        self.config = config
        self.feature_operations = feature_operations

    def batch_generate_features(
        self,
        metadata_index: dict,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """
        バッチ特徴量生成
        
        Args:
            metadata_index: メタデータインデックス
            symbols: 銘柄コードのリスト
            data_dict: 銘柄別データ辞書
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            optimization_config: 最適化設定
            
        Returns:
            銘柄別特徴量結果の辞書
        """
        logger.info(
            "バッチ特徴量生成開始",
            extra={
                "symbols_count": len(symbols),
                "start_date": start_date,
                "end_date": end_date,
                "parallel_enabled": self.config.enable_parallel_batch_processing,
                "parallel_backend": self.config.parallel_backend,
                "max_workers": self.config.max_parallel_workers,
            },
        )

        # 並列化が有効で複数シンボルがある場合は並列処理
        if (self.config.enable_parallel_batch_processing and len(symbols) > 1):
            return self._batch_generate_features_parallel(
                metadata_index, symbols, data_dict, start_date, end_date, 
                feature_config, optimization_config
            )
        else:
            return self._batch_generate_features_sequential(
                metadata_index, symbols, data_dict, start_date, end_date, 
                feature_config, optimization_config
            )

    def _batch_generate_features_sequential(
        self,
        metadata_index: dict,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """順次バッチ特徴量生成"""
        results = {}
        cache_hits = 0
        cache_misses = 0

        for symbol in symbols:
            if symbol not in data_dict:
                logger.warning(f"データが見つかりません: {symbol}")
                continue

            try:
                # キャッシュ確認
                if self.feature_operations.has_feature(
                    metadata_index, symbol, start_date, end_date, feature_config
                ):
                    cached_result = self.feature_operations.load_feature(
                        metadata_index, symbol, start_date, end_date, feature_config
                    )
                    if cached_result:
                        results[symbol] = cached_result
                        cache_hits += 1
                        continue

                # 生成
                data = data_dict[symbol]
                feature_result = self.feature_operations.get_or_generate_feature(
                    metadata_index,
                    symbol,
                    data,
                    start_date,
                    end_date,
                    feature_config,
                    optimization_config,
                )
                results[symbol] = feature_result
                cache_misses += 1

            except Exception as e:
                logger.error(f"バッチ特徴量生成エラー: {symbol} - {e}")
                continue

        logger.info(
            "順次バッチ特徴量生成完了",
            extra={
                "processed_symbols": len(results),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": (
                    f"{cache_hits / (cache_hits + cache_misses) * 100:.1f}%"
                    if (cache_hits + cache_misses) > 0
                    else "0%"
                ),
            },
        )

        return results

    def _batch_generate_features_parallel(
        self,
        metadata_index: dict,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """並列バッチ特徴量生成"""
        results = {}
        total_cache_hits = 0
        total_cache_misses = 0

        # バックエンドに応じた並列処理
        if self.config.parallel_backend == "joblib" and JOBLIB_AVAILABLE:
            results, total_cache_hits, total_cache_misses = self._parallel_process_joblib(
                metadata_index, symbols, data_dict, start_date, end_date, 
                feature_config, optimization_config
            )
        else:
            # Threading並列処理（デフォルト）
            results, total_cache_hits, total_cache_misses = self._parallel_process_threading(
                metadata_index, symbols, data_dict, start_date, end_date, 
                feature_config, optimization_config
            )

        logger.info(
            "並列バッチ特徴量生成完了",
            extra={
                "processed_symbols": len(results),
                "cache_hits": total_cache_hits,
                "cache_misses": total_cache_misses,
                "cache_hit_rate": (
                    f"{total_cache_hits / (total_cache_hits + total_cache_misses) * 100:.1f}%"
                    if (total_cache_hits + total_cache_misses) > 0
                    else "0%"
                ),
                "parallel_backend": self.config.parallel_backend,
            },
        )

        return results

    def _process_single_symbol(
        self,
        metadata_index: dict,
        symbol: str,
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Tuple[str, Optional[FeatureResult], bool, str]:
        """
        単一シンボル処理（並列実行用）

        Returns:
            (symbol, result, is_cache_hit, error_message)
        """
        if symbol not in data_dict:
            return symbol, None, False, f"データが見つかりません: {symbol}"

        try:
            # キャッシュ確認
            if self.feature_operations.has_feature(
                metadata_index, symbol, start_date, end_date, feature_config
            ):
                cached_result = self.feature_operations.load_feature(
                    metadata_index, symbol, start_date, end_date, feature_config
                )
                if cached_result:
                    return symbol, cached_result, True, ""

            # 生成
            data = data_dict[symbol]
            feature_result = self.feature_operations.get_or_generate_feature(
                metadata_index,
                symbol,
                data,
                start_date,
                end_date,
                feature_config,
                optimization_config,
            )
            return symbol, feature_result, False, ""

        except Exception as e:
            error_msg = f"バッチ特徴量生成エラー: {symbol} - {e}"
            return symbol, None, False, error_msg

    def _parallel_process_threading(
        self,
        metadata_index: dict,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Tuple[Dict[str, FeatureResult], int, int]:
        """
        Threading並列処理

        Returns:
            (results, cache_hits, cache_misses)
        """
        results = {}
        cache_hits = 0
        cache_misses = 0

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
            # 各シンボルを並列で処理
            future_to_symbol = {
                executor.submit(
                    self._process_single_symbol,
                    metadata_index, symbol, data_dict, start_date, end_date, 
                    feature_config, optimization_config
                ): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                try:
                    symbol, result, is_cache_hit, error_msg = future.result()

                    if result is not None:
                        results[symbol] = result
                        if is_cache_hit:
                            cache_hits += 1
                        else:
                            cache_misses += 1
                    elif error_msg:
                        logger.error(error_msg)

                except Exception as e:
                    symbol = future_to_symbol[future]
                    logger.error(f"Threading並列処理エラー ({symbol}): {e}")

        return results, cache_hits, cache_misses

    def _parallel_process_joblib(
        self,
        metadata_index: dict,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Tuple[Dict[str, FeatureResult], int, int]:
        """
        Joblib並列処理

        Returns:
            (results, cache_hits, cache_misses)
        """
        try:
            # Joblib並列実行
            results_list = Parallel(n_jobs=self.config.max_parallel_workers)(
                delayed(self._process_single_symbol)(
                    metadata_index, symbol, data_dict, start_date, end_date, 
                    feature_config, optimization_config
                )
                for symbol in symbols
            )

            # 結果を集約
            results = {}
            cache_hits = 0
            cache_misses = 0

            for symbol, result, is_cache_hit, error_msg in results_list:
                if result is not None:
                    results[symbol] = result
                    if is_cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                elif error_msg:
                    logger.error(error_msg)

            return results, cache_hits, cache_misses

        except Exception as e:
            logger.error(f"Joblib並列処理エラー: {e}")
            # フォールバック: Threading並列処理
            logger.warning("Joblib処理失敗、Threading並列処理にフォールバック")
            return self._parallel_process_threading(
                metadata_index, symbols, data_dict, start_date, end_date, 
                feature_config, optimization_config
            )


class ChunkedBatchProcessor:
    """
    チャンク化バッチ処理クラス
    
    大量のシンボルをチャンクに分けて効率的に処理します。
    """

    def __init__(
        self,
        config: FeatureStoreConfig,
        feature_operations: FeatureOperations,
    ):
        """
        初期化
        
        Args:
            config: 特徴量ストア設定
            feature_operations: 特徴量操作クラス
        """
        self.config = config
        self.feature_operations = feature_operations
        self.batch_processor = BatchProcessor(config, feature_operations)

    def process_in_chunks(
        self,
        metadata_index: dict,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """
        チャンク単位でのバッチ処理
        
        Args:
            metadata_index: メタデータインデックス
            symbols: 銘柄コードのリスト
            data_dict: 銘柄別データ辞書
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            optimization_config: 最適化設定
            
        Returns:
            銘柄別特徴量結果の辞書
        """
        chunk_size = self.config.batch_chunk_size
        total_results = {}

        logger.info(
            f"チャンク化バッチ処理開始: {len(symbols)}件をチャンクサイズ{chunk_size}で処理"
        )

        # シンボルをチャンクに分割
        for i in range(0, len(symbols), chunk_size):
            chunk_symbols = symbols[i:i + chunk_size]
            
            logger.info(f"チャンク処理 ({i // chunk_size + 1}): {len(chunk_symbols)}件")

            # チャンク内データを抽出
            chunk_data_dict = {
                symbol: data_dict[symbol] 
                for symbol in chunk_symbols 
                if symbol in data_dict
            }

            # チャンク処理実行
            chunk_results = self.batch_processor.batch_generate_features(
                metadata_index,
                chunk_symbols,
                chunk_data_dict,
                start_date,
                end_date,
                feature_config,
                optimization_config,
            )

            # 結果をマージ
            total_results.update(chunk_results)

            logger.info(f"チャンク完了: {len(chunk_results)}件処理済み")

        logger.info(
            f"チャンク化バッチ処理完了: 合計{len(total_results)}件処理"
        )

        return total_results