#!/usr/bin/env python3
"""
Dask Core Data Processor
Issue #384: 並列処理のさらなる強化 - Core Processor Module

Daskクラスター管理と基本データ処理機能を提供する
"""

import asyncio
import os
import shutil
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Dask関連インポート（オプショナル）
try:
    import dask
    import dask.array as da
    import dask.dataframe as dd
    from dask import compute
    from dask.delayed import delayed
    from dask.distributed import Client, LocalCluster, as_completed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn(
        "Daskが利用できません。分散処理機能は制限されます。", UserWarning, stacklevel=2
    )

# プロジェクトモジュール
try:
    from ...analysis.technical_analysis import TechnicalAnalyzer
    from ...data.enhanced_stock_fetcher import create_enhanced_stock_fetcher
    from ...data.stock_fetcher import StockFetcher
    from ...utils.logging_config import get_context_logger, log_performance_metric
    from ...utils.performance_monitor import PerformanceMonitor
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    # モッククラス
    class StockFetcher:
        def get_historical_data(self, symbol):
            return pd.DataFrame({"timestamp": [datetime.now()], "close": [100]})

    class TechnicalAnalyzer:
        def analyze(self, data):
            return data

    def create_enhanced_stock_fetcher(**kwargs):
        return StockFetcher()

    class PerformanceMonitor:
        def __init__(self):
            pass

        def start_monitoring(self):
            pass

        def stop_monitoring(self):
            return {}


logger = get_context_logger(__name__)


class DaskDataProcessor:
    """
    Dask分散データプロセッサー

    大規模データセットの分散処理、out-of-core計算、
    メモリ効率的な股価データ分析を提供
    """

    def __init__(
        self,
        enable_distributed: bool = True,
        n_workers: int = None,
        threads_per_worker: int = 2,
        memory_limit: str = "2GB",
        temp_dir: str = None,
        chunk_size: int = 10000,
    ):
        """
        初期化

        Args:
            enable_distributed: 分散処理有効化
            n_workers: ワーカー数（Noneで自動設定）
            threads_per_worker: ワーカーあたりスレッド数
            memory_limit: ワーカーメモリ制限
            temp_dir: 一時ディレクトリ
            chunk_size: データチャンクサイズ
        """
        if not DASK_AVAILABLE:
            logger.warning(
                "Daskが利用できないため、シングルスレッド処理にフォールバック"
            )
            enable_distributed = False

        self.enable_distributed = enable_distributed
        self.n_workers = n_workers or min(8, os.cpu_count() or 4)
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.chunk_size = chunk_size

        # 一時ディレクトリ設定
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="dask_day_trade_"))

        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Daskクライアント
        self.client = None
        self.cluster = None

        # パフォーマンス統計
        self.stats = {
            "processed_symbols": 0,
            "total_processing_time_ms": 0,
            "average_processing_time_ms": 0,
            "memory_peak_mb": 0,
            "tasks_executed": 0,
            "cache_hits": 0,
        }

        # データ処理コンポーネント
        self.stock_fetcher = create_enhanced_stock_fetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.performance_monitor = PerformanceMonitor()

        # 初期化
        if self.enable_distributed:
            self._initialize_dask_cluster()

        logger.info(
            f"DaskDataProcessor初期化完了: "
            f"distributed={enable_distributed}, workers={self.n_workers}"
        )

    def _initialize_dask_cluster(self):
        """Daskクラスター初期化"""
        if not DASK_AVAILABLE:
            return

        try:
            # ローカルクラスター設定
            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
                local_directory=str(self.temp_dir),
                silence_logs=logging.ERROR,
                dashboard_address=":8787",
            )

            # クライアント接続
            self.client = Client(self.cluster)

            logger.info(f"Daskクラスター開始: {self.client.dashboard_link}")
            logger.info(f"ワーカー: {len(self.client.scheduler_info()['workers'])}台")

        except Exception as e:
            logger.error(f"Daskクラスター初期化失敗: {e}")
            self.enable_distributed = False

    async def process_multiple_symbols_parallel(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        include_technical: bool = True,
        store_intermediate: bool = True,
    ) -> pd.DataFrame:
        """
        複数銘柄の並列データ処理

        Args:
            symbols: 処理対象銘柄リスト
            start_date: 開始日
            end_date: 終了日
            include_technical: テクニカル分析含有
            store_intermediate: 中間結果保存

        Returns:
            統合されたデータフレーム
        """
        logger.info(f"複数銘柄並列処理開始: {len(symbols)}銘柄")
        start_time = time.time()

        try:
            if self.enable_distributed and DASK_AVAILABLE:
                result = await self._process_with_dask(
                    symbols, start_date, end_date, include_technical, store_intermediate
                )
            else:
                result = await self._process_sequential(
                    symbols, start_date, end_date, include_technical, store_intermediate
                )

            # 統計更新
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats(len(symbols), processing_time_ms)

            logger.info(f"並列処理完了: {len(symbols)}銘柄, {processing_time_ms:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"並列処理エラー: {e}")
            raise

    async def _process_with_dask(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        include_technical: bool,
        store_intermediate: bool,
    ) -> pd.DataFrame:
        """Dask分散処理実行"""

        # 遅延タスク生成
        delayed_tasks = []
        for symbol in symbols:
            task = delayed(self._process_single_symbol_delayed)(
                symbol, start_date, end_date, include_technical
            )
            delayed_tasks.append(task)

        # 分散実行
        logger.info(f"Dask分散実行: {len(delayed_tasks)}タスク")

        try:
            # 並列計算実行
            results = compute(
                *delayed_tasks, scheduler="distributed" if self.client else "threads"
            )

            # 結果統合
            valid_results = [r for r in results if r is not None and not r.empty]

            if not valid_results:
                logger.warning("有効なデータが取得できませんでした")
                return pd.DataFrame()

            # DataFrameの統合
            combined_df = pd.concat(valid_results, ignore_index=True)

            # 中間結果保存（オプション）
            if store_intermediate:
                self._save_intermediate_results(combined_df, symbols)

            logger.info(f"Dask処理完了: {len(combined_df)}レコード統合")
            return combined_df

        except Exception as e:
            logger.error(f"Dask分散処理エラー: {e}")
            # フォールバックでシーケンシャル処理
            logger.info("シーケンシャル処理にフォールバック")
            return await self._process_sequential(
                symbols, start_date, end_date, include_technical, store_intermediate
            )

    def _process_single_symbol_delayed(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        include_technical: bool,
    ) -> pd.DataFrame:
        """
        単一銘柄処理（遅延実行用）

        注意: この関数はDaskワーカーで実行されるため、
        クラスメンバーへの直接アクセスは制限される
        """
        try:
            # 新しいStockFetcherインスタンス（ワーカー内で）
            fetcher = create_enhanced_stock_fetcher()

            # データ取得
            historical_data = fetcher.get_historical_data(
                symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

            if historical_data is None or historical_data.empty:
                logger.debug(f"データなし: {symbol}")
                return pd.DataFrame()

            # 基本データクリーニング
            df = historical_data.copy()
            df["symbol"] = symbol
            df["fetch_timestamp"] = datetime.now()

            # テクニカル分析（オプション）
            if include_technical:
                try:
                    analyzer = TechnicalAnalyzer()
                    df = analyzer.analyze(df)
                except Exception as e:
                    logger.debug(f"テクニカル分析失敗 {symbol}: {e}")

            logger.debug(f"処理完了 {symbol}: {len(df)}レコード")
            return df

        except Exception as e:
            logger.error(f"銘柄処理エラー {symbol}: {e}")
            return pd.DataFrame()

    async def _process_sequential(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        include_technical: bool,
        store_intermediate: bool,
    ) -> pd.DataFrame:
        """シーケンシャル処理（フォールバック）"""
        results = []

        for symbol in symbols:
            try:
                result = self._process_single_symbol_delayed(
                    symbol, start_date, end_date, include_technical
                )
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                logger.error(f"シーケンシャル処理エラー {symbol}: {e}")
                continue

        if results:
            combined_df = pd.concat(results, ignore_index=True)

            if store_intermediate:
                self._save_intermediate_results(combined_df, symbols)

            return combined_df
        else:
            return pd.DataFrame()

    def _save_intermediate_results(self, df: pd.DataFrame, symbols: List[str]):
        """中間結果保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"dask_processing_results_{timestamp}_{len(symbols)}symbols.parquet"
            )
            filepath = self.temp_dir / filename

            # Parquet形式で高効率保存
            df.to_parquet(filepath, compression="snappy")
            logger.info(f"中間結果保存: {filepath}")

        except Exception as e:
            logger.warning(f"中間結果保存失敗: {e}")

    def process_large_dataset_out_of_core(
        self,
        data_source: Union[str, Path, List[str]],
        processing_function: Callable,
        output_path: Optional[str] = None,
        **processing_kwargs,
    ) -> Union[pd.DataFrame, str]:
        """
        大規模データセットのout-of-core処理

        Args:
            data_source: データソース（ファイルパスまたはファイルリスト）
            processing_function: 処理関数
            output_path: 出力パス（Noneの場合はメモリ返却）
            **processing_kwargs: 処理関数への追加引数

        Returns:
            処理結果DataFrameまたは出力パス
        """
        if not DASK_AVAILABLE:
            raise RuntimeError("Daskが必要です")

        logger.info(f"大規模データセット処理開始: {data_source}")
        start_time = time.time()

        try:
            # データソース解析
            if isinstance(data_source, (str, Path)):
                # 単一ファイルまたはパターン
                if str(data_source).endswith(".csv"):
                    dd_df = dd.read_csv(data_source)
                elif str(data_source).endswith(".parquet"):
                    dd_df = dd.read_parquet(data_source)
                else:
                    # ディレクトリ内の全CSVファイル
                    pattern = str(Path(data_source) / "*.csv")
                    dd_df = dd.read_csv(pattern)

            elif isinstance(data_source, list):
                # 複数ファイル
                dd_dfs = []
                for file_path in data_source:
                    if str(file_path).endswith(".csv"):
                        dd_dfs.append(dd.read_csv(file_path))
                    elif str(file_path).endswith(".parquet"):
                        dd_dfs.append(dd.read_parquet(file_path))

                if dd_dfs:
                    dd_df = dd.concat(dd_dfs)
                else:
                    raise ValueError("有効なデータファイルがありません")
            else:
                raise ValueError("サポートされていないデータソース形式")

            # 分散処理適用
            logger.info(f"データフレーム分割数: {dd_df.npartitions}")

            # 処理関数適用
            processed_df = dd_df.map_partitions(
                processing_function, **processing_kwargs, meta=dd_df
            )

            # 結果計算
            if output_path:
                # ファイル出力
                processed_df.to_parquet(
                    output_path, compression="snappy", write_index=False
                )

                result = output_path
                logger.info(f"結果をファイル出力: {output_path}")
            else:
                # メモリ返却
                result = processed_df.compute()
                logger.info(f"結果をメモリ返却: {len(result)}レコード")

            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(f"大規模データセット処理完了: {processing_time_ms:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"大規模データセット処理エラー: {e}")
            raise

    def _update_stats(self, symbol_count: int, processing_time_ms: float):
        """統計情報更新"""
        self.stats["processed_symbols"] += symbol_count
        self.stats["total_processing_time_ms"] += processing_time_ms
        self.stats["tasks_executed"] += 1

        if self.stats["tasks_executed"] > 0:
            self.stats["average_processing_time_ms"] = (
                self.stats["total_processing_time_ms"] / self.stats["tasks_executed"]
            )

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        stats = self.stats.copy()

        if self.client and DASK_AVAILABLE:
            try:
                # Daskクラスター統計
                scheduler_info = self.client.scheduler_info()
                stats["dask_workers"] = len(scheduler_info["workers"])
                stats["dask_tasks"] = sum(
                    w.get("executing", 0) for w in scheduler_info["workers"].values()
                )
                stats["dask_memory_usage"] = sum(
                    w.get("memory", 0) for w in scheduler_info["workers"].values()
                )
            except Exception as e:
                logger.debug(f"Dask統計取得失敗: {e}")

        return stats

    def get_health_status(self) -> Dict[str, Any]:
        """ヘルスステータス"""
        health = {
            "status": "healthy",
            "distributed_enabled": self.enable_distributed,
            "dask_available": DASK_AVAILABLE,
            "temp_dir_exists": self.temp_dir.exists(),
            "stats": self.get_stats(),
        }

        if self.client and DASK_AVAILABLE:
            try:
                scheduler_info = self.client.scheduler_info()
                worker_count = len(scheduler_info["workers"])

                if worker_count == 0:
                    health["status"] = "critical"
                elif worker_count < self.n_workers / 2:
                    health["status"] = "degraded"

                health["workers_active"] = worker_count
                health["workers_expected"] = self.n_workers

            except Exception as e:
                health["status"] = "degraded"
                health["dask_error"] = str(e)

        return health

    def cleanup(self):
        """クリーンアップ"""
        logger.info("DaskDataProcessor クリーンアップ開始")

        try:
            # Daskクラスター終了
            if self.client:
                self.client.close()

            if self.cluster:
                self.cluster.close()

            # 一時ディレクトリ削除
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)

            logger.info("クリーンアップ完了")

        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")