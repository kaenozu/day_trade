#!/usr/bin/env python3
"""
Daskデータプロセッサー
Issue #384: 並列処理のさらなる強化 - Phase 1実装

大規模データセットの分散処理とout-of-core計算による
スケーラブルな株価データ分析
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
    warnings.warn("Daskが利用できません。分散処理機能は制限されます。", UserWarning, stacklevel=2)

# プロジェクトモジュール
try:
    from ..analysis.technical_analysis import TechnicalAnalyzer
    from ..data.enhanced_stock_fetcher import create_enhanced_stock_fetcher
    from ..data.stock_fetcher import StockFetcher
    from ..utils.logging_config import get_context_logger, log_performance_metric
    from ..utils.performance_monitor import PerformanceMonitor
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
            logger.warning("Daskが利用できないため、シングルスレッド処理にフォールバック")
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
            results = compute(*delayed_tasks, scheduler="distributed" if self.client else "threads")

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
            filename = f"dask_processing_results_{timestamp}_{len(symbols)}symbols.parquet"
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
                processed_df.to_parquet(output_path, compression="snappy", write_index=False)

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

    def analyze_market_correlation_distributed(
        self,
        symbols: List[str],
        analysis_period_days: int = 252,
        correlation_window: int = 30,
    ) -> pd.DataFrame:
        """
        分散相関分析

        Args:
            symbols: 分析対象銘柄
            analysis_period_days: 分析期間（日数）
            correlation_window: 相関計算ウィンドウ

        Returns:
            相関分析結果
        """
        if not DASK_AVAILABLE or not self.enable_distributed:
            logger.warning("分散処理が利用できないため、制限された分析を実行")
            return self._analyze_correlation_sequential(
                symbols, analysis_period_days, correlation_window
            )

        logger.info(f"分散相関分析開始: {len(symbols)}銘柄")

        try:
            # 価格データ取得（分散）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=analysis_period_days)

            # 価格データマトリックス構築
            price_data = {}

            @delayed
            def fetch_price_data(symbol):
                try:
                    fetcher = create_enhanced_stock_fetcher()
                    data = fetcher.get_historical_data(
                        symbol,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )
                    if data is not None and not data.empty:
                        return symbol, data["close"].values
                    return symbol, None
                except Exception as e:
                    logger.debug(f"価格データ取得失敗 {symbol}: {e}")
                    return symbol, None

            # 全銘柄の価格データを並列取得
            price_tasks = [fetch_price_data(symbol) for symbol in symbols]
            price_results = compute(*price_tasks)

            # 有効なデータのみ抽出
            valid_data = {}
            for symbol, prices in price_results:
                if prices is not None:
                    valid_data[symbol] = prices

            if len(valid_data) < 2:
                logger.warning("相関分析に十分なデータがありません")
                return pd.DataFrame()

            # 価格データ長を統一
            min_length = min(len(prices) for prices in valid_data.values())
            aligned_data = {symbol: prices[-min_length:] for symbol, prices in valid_data.items()}

            # 相関マトリックス計算（分散）
            symbols_list = list(aligned_data.keys())
            correlation_matrix = np.zeros((len(symbols_list), len(symbols_list)))

            @delayed
            def calculate_correlation_pair(i, j, symbol_i, symbol_j):
                if i == j:
                    return i, j, 1.0
                try:
                    prices_i = np.array(aligned_data[symbol_i])
                    prices_j = np.array(aligned_data[symbol_j])

                    # ローリング相関計算
                    returns_i = np.diff(prices_i) / prices_i[:-1]
                    returns_j = np.diff(prices_j) / prices_j[:-1]

                    correlation = np.corrcoef(returns_i, returns_j)[0, 1]
                    return i, j, correlation if not np.isnan(correlation) else 0.0
                except Exception as e:
                    logger.debug(f"相関計算エラー {symbol_i}-{symbol_j}: {e}")
                    return i, j, 0.0

            # 相関ペア計算タスク生成
            correlation_tasks = []
            for i, symbol_i in enumerate(symbols_list):
                for j, symbol_j in enumerate(symbols_list):
                    if i <= j:  # 対称性利用
                        task = calculate_correlation_pair(i, j, symbol_i, symbol_j)
                        correlation_tasks.append(task)

            # 相関計算実行
            correlation_results = compute(*correlation_tasks)

            # 相関マトリックス構築
            for i, j, corr in correlation_results:
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr  # 対称性

            # 結果DataFrame作成
            correlation_df = pd.DataFrame(
                correlation_matrix, index=symbols_list, columns=symbols_list
            )

            # メタデータ追加
            correlation_df.attrs = {
                "analysis_date": datetime.now().isoformat(),
                "analysis_period_days": analysis_period_days,
                "symbols_count": len(symbols_list),
                "processing_method": "dask_distributed",
            }

            logger.info(f"分散相関分析完了: {len(symbols_list)}銘柄の相関マトリックス")
            return correlation_df

        except Exception as e:
            logger.error(f"分散相関分析エラー: {e}")
            # フォールバックでシーケンシャル処理
            return self._analyze_correlation_sequential(
                symbols, analysis_period_days, correlation_window
            )

    def _analyze_correlation_sequential(
        self, symbols: List[str], analysis_period_days: int, correlation_window: int
    ) -> pd.DataFrame:
        """シーケンシャル相関分析（フォールバック）"""
        logger.info(f"シーケンシャル相関分析実行: {len(symbols)}銘柄")

        try:
            # 簡略化された相関分析
            price_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=analysis_period_days)

            for symbol in symbols:
                try:
                    data = self.stock_fetcher.get_historical_data(
                        symbol,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )
                    if data is not None and not data.empty:
                        price_data[symbol] = data["close"].values
                except Exception as e:
                    logger.debug(f"データ取得失敗 {symbol}: {e}")

            if len(price_data) < 2:
                return pd.DataFrame()

            # 最小長に合わせる
            min_length = min(len(prices) for prices in price_data.values())
            aligned_data = {symbol: prices[-min_length:] for symbol, prices in price_data.items()}

            # 相関計算
            df = pd.DataFrame(aligned_data)
            returns = df.pct_change().dropna()
            correlation_df = returns.corr()

            correlation_df.attrs = {
                "analysis_date": datetime.now().isoformat(),
                "analysis_period_days": analysis_period_days,
                "symbols_count": len(price_data),
                "processing_method": "sequential_fallback",
            }

            return correlation_df

        except Exception as e:
            logger.error(f"シーケンシャル相関分析エラー: {e}")
            return pd.DataFrame()

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


class DaskStockAnalyzer:
    """Dask特化型株価分析器"""

    def __init__(self, dask_processor: DaskDataProcessor):
        self.dask_processor = dask_processor

    async def analyze_portfolio_performance_distributed(
        self,
        portfolio_symbols: List[str],
        benchmark_symbol: str = "SPY",
        analysis_period_days: int = 252,
    ) -> Dict[str, Any]:
        """分散ポートフォリオパフォーマンス分析"""

        logger.info(f"分散ポートフォリオ分析: {len(portfolio_symbols)}銘柄 vs {benchmark_symbol}")

        try:
            all_symbols = portfolio_symbols + [benchmark_symbol]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=analysis_period_days)

            # 価格データ並列取得
            price_data = await self.dask_processor.process_multiple_symbols_parallel(
                all_symbols,
                start_date,
                end_date,
                include_technical=False,
                store_intermediate=False,
            )

            if price_data.empty:
                logger.warning("ポートフォリオ分析用データが不足しています")
                return {}

            # ポートフォリオ分析実行（分散）
            analysis_results = await self._perform_distributed_portfolio_analysis(
                price_data, portfolio_symbols, benchmark_symbol
            )

            return analysis_results

        except Exception as e:
            logger.error(f"分散ポートフォリオ分析エラー: {e}")
            return {}

    async def _perform_distributed_portfolio_analysis(
        self,
        price_data: pd.DataFrame,
        portfolio_symbols: List[str],
        benchmark_symbol: str,
    ) -> Dict[str, Any]:
        """分散ポートフォリオ分析実行"""

        if not DASK_AVAILABLE:
            return self._perform_sequential_portfolio_analysis(
                price_data, portfolio_symbols, benchmark_symbol
            )

        try:
            # シンボル別データ分割
            symbol_groups = price_data.groupby("symbol")

            @delayed
            def calculate_symbol_metrics(symbol, group_data):
                """銘柄別メトリクス計算"""
                try:
                    prices = group_data.sort_values("timestamp")["close"]
                    returns = prices.pct_change().dropna()

                    metrics = {
                        "symbol": symbol,
                        "total_return": (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
                        "volatility": returns.std() * np.sqrt(252) * 100,
                        "sharpe_ratio": (
                            (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                            if returns.std() > 0
                            else 0
                        ),
                        "max_drawdown": self._calculate_max_drawdown(prices),
                        "var_95": np.percentile(returns, 5) * 100,
                        "data_points": len(prices),
                    }

                    return metrics

                except Exception as e:
                    logger.debug(f"メトリクス計算エラー {symbol}: {e}")
                    return {"symbol": symbol, "error": str(e)}

            # 各銘柄のメトリクス並列計算
            metric_tasks = [
                calculate_symbol_metrics(symbol, group) for symbol, group in symbol_groups
            ]

            metric_results = compute(*metric_tasks)

            # 結果整理
            portfolio_metrics = []
            benchmark_metrics = None

            for metrics in metric_results:
                if "error" not in metrics:
                    if metrics["symbol"] in portfolio_symbols:
                        portfolio_metrics.append(metrics)
                    elif metrics["symbol"] == benchmark_symbol:
                        benchmark_metrics = metrics

            # ポートフォリオ全体統計
            if portfolio_metrics:
                portfolio_total_return = sum(m["total_return"] for m in portfolio_metrics) / len(
                    portfolio_metrics
                )
                portfolio_avg_volatility = sum(m["volatility"] for m in portfolio_metrics) / len(
                    portfolio_metrics
                )
                portfolio_avg_sharpe = sum(m["sharpe_ratio"] for m in portfolio_metrics) / len(
                    portfolio_metrics
                )
            else:
                portfolio_total_return = portfolio_avg_volatility = portfolio_avg_sharpe = 0

            analysis_results = {
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_period_days": (
                    price_data["timestamp"].max() - price_data["timestamp"].min()
                ).days,
                "portfolio_summary": {
                    "symbols_count": len(portfolio_metrics),
                    "average_return": portfolio_total_return,
                    "average_volatility": portfolio_avg_volatility,
                    "average_sharpe_ratio": portfolio_avg_sharpe,
                    "processing_method": "dask_distributed",
                },
                "individual_metrics": portfolio_metrics,
                "benchmark_metrics": benchmark_metrics,
                "relative_performance": {
                    "alpha": portfolio_total_return
                    - (benchmark_metrics["total_return"] if benchmark_metrics else 0),
                    "outperformed_benchmark": portfolio_total_return
                    > (benchmark_metrics["total_return"] if benchmark_metrics else 0),
                },
            }

            return analysis_results

        except Exception as e:
            logger.error(f"分散ポートフォリオ分析実行エラー: {e}")
            return self._perform_sequential_portfolio_analysis(
                price_data, portfolio_symbols, benchmark_symbol
            )

    def _perform_sequential_portfolio_analysis(
        self,
        price_data: pd.DataFrame,
        portfolio_symbols: List[str],
        benchmark_symbol: str,
    ) -> Dict[str, Any]:
        """シーケンシャルポートフォリオ分析（フォールバック）"""

        try:
            results = {
                "analysis_timestamp": datetime.now().isoformat(),
                "portfolio_summary": {
                    "symbols_count": len(portfolio_symbols),
                    "processing_method": "sequential_fallback",
                },
                "individual_metrics": [],
                "benchmark_metrics": None,
                "relative_performance": {},
            }

            # 簡略化分析
            symbol_groups = price_data.groupby("symbol")

            for symbol, group in symbol_groups:
                try:
                    prices = group.sort_values("timestamp")["close"]
                    returns = prices.pct_change().dropna()

                    metrics = {
                        "symbol": symbol,
                        "total_return": (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
                        "volatility": returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0,
                        "data_points": len(prices),
                    }

                    if symbol in portfolio_symbols:
                        results["individual_metrics"].append(metrics)
                    elif symbol == benchmark_symbol:
                        results["benchmark_metrics"] = metrics

                except Exception as e:
                    logger.debug(f"シーケンシャルメトリクス計算エラー {symbol}: {e}")

            return results

        except Exception as e:
            logger.error(f"シーケンシャルポートフォリオ分析エラー: {e}")
            return {}

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """最大ドローダウン計算"""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min() * 100
        except Exception:
            return 0.0


class DaskBatchProcessor:
    """Dask特化バッチプロセッサー"""

    def __init__(self, dask_processor: DaskDataProcessor):
        self.dask_processor = dask_processor

    async def process_market_data_pipeline(
        self,
        symbols: List[str],
        pipeline_steps: List[str],
        start_date: datetime,
        end_date: datetime,
        output_format: str = "parquet",
    ) -> Dict[str, Any]:
        """市場データ処理パイプライン"""

        logger.info(
            f"市場データパイプライン開始: {len(symbols)}銘柄, {len(pipeline_steps)}ステップ"
        )

        try:
            # ステップ1: 基本データ取得
            if "fetch_data" in pipeline_steps:
                raw_data = await self.dask_processor.process_multiple_symbols_parallel(
                    symbols, start_date, end_date, include_technical=False
                )
            else:
                raw_data = pd.DataFrame()

            # ステップ2: テクニカル分析
            if "technical_analysis" in pipeline_steps and not raw_data.empty:
                enhanced_data = await self._apply_technical_analysis_distributed(raw_data)
            else:
                enhanced_data = raw_data

            # ステップ3: データクリーニング
            if "data_cleaning" in pipeline_steps and not enhanced_data.empty:
                cleaned_data = await self._apply_data_cleaning_distributed(enhanced_data)
            else:
                cleaned_data = enhanced_data

            # ステップ4: 出力
            output_path = None
            if not cleaned_data.empty and output_format:
                output_path = await self._save_pipeline_output(cleaned_data, output_format)

            results = {
                "pipeline_timestamp": datetime.now().isoformat(),
                "symbols_processed": len(symbols),
                "steps_executed": pipeline_steps,
                "records_processed": len(cleaned_data) if not cleaned_data.empty else 0,
                "output_path": output_path,
                "processing_successful": not cleaned_data.empty,
            }

            logger.info(f"パイプライン完了: {results['records_processed']}レコード処理")
            return results

        except Exception as e:
            logger.error(f"市場データパイプラインエラー: {e}")
            return {"error": str(e), "processing_successful": False}

    async def _apply_technical_analysis_distributed(self, data: pd.DataFrame) -> pd.DataFrame:
        """分散テクニカル分析適用"""

        if not DASK_AVAILABLE:
            return self._apply_technical_analysis_sequential(data)

        try:

            @delayed
            def apply_technical_to_symbol(symbol_data):
                try:
                    analyzer = TechnicalAnalyzer()
                    return analyzer.analyze(symbol_data)
                except Exception as e:
                    logger.debug(f"テクニカル分析エラー: {e}")
                    return symbol_data

            # シンボル別に分散処理
            symbol_groups = data.groupby("symbol")
            analysis_tasks = [
                apply_technical_to_symbol(group.reset_index(drop=True))
                for symbol, group in symbol_groups
            ]

            analyzed_results = compute(*analysis_tasks)

            # 結果統合
            if analyzed_results:
                return pd.concat([r for r in analyzed_results if not r.empty], ignore_index=True)
            else:
                return data

        except Exception as e:
            logger.error(f"分散テクニカル分析エラー: {e}")
            return self._apply_technical_analysis_sequential(data)

    def _apply_technical_analysis_sequential(self, data: pd.DataFrame) -> pd.DataFrame:
        """シーケンシャルテクニカル分析（フォールバック）"""
        try:
            analyzer = TechnicalAnalyzer()
            return analyzer.analyze(data)
        except Exception as e:
            logger.error(f"シーケンシャルテクニカル分析エラー: {e}")
            return data

    async def _apply_data_cleaning_distributed(self, data: pd.DataFrame) -> pd.DataFrame:
        """分散データクリーニング"""

        if not DASK_AVAILABLE or data.empty:
            return self._apply_data_cleaning_sequential(data)

        try:

            @delayed
            def clean_data_chunk(chunk):
                try:
                    # 基本クリーニング
                    chunk = chunk.dropna(subset=["close", "volume"])
                    chunk = chunk[chunk["volume"] > 0]
                    chunk = chunk[chunk["close"] > 0]

                    # 異常値除去（簡易）
                    for col in ["close", "high", "low", "volume"]:
                        if col in chunk.columns:
                            q1 = chunk[col].quantile(0.01)
                            q99 = chunk[col].quantile(0.99)
                            chunk = chunk[(chunk[col] >= q1) & (chunk[col] <= q99)]

                    return chunk

                except Exception as e:
                    logger.debug(f"データクリーニングエラー: {e}")
                    return chunk

            # データを適当なチャンクに分割
            chunk_size = max(1000, len(data) // (self.dask_processor.n_workers * 4))
            data_chunks = [data.iloc[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

            # 分散クリーニング実行
            cleaning_tasks = [clean_data_chunk(chunk) for chunk in data_chunks]
            cleaned_chunks = compute(*cleaning_tasks)

            # 結果統合
            if cleaned_chunks:
                cleaned_data = pd.concat(
                    [c for c in cleaned_chunks if not c.empty], ignore_index=True
                )
                return cleaned_data
            else:
                return data

        except Exception as e:
            logger.error(f"分散データクリーニングエラー: {e}")
            return self._apply_data_cleaning_sequential(data)

    def _apply_data_cleaning_sequential(self, data: pd.DataFrame) -> pd.DataFrame:
        """シーケンシャルデータクリーニング"""
        try:
            if data.empty:
                return data

            # 基本クリーニング
            cleaned = data.dropna(subset=["close"])
            cleaned = cleaned[cleaned["close"] > 0]

            if "volume" in cleaned.columns:
                cleaned = cleaned[cleaned["volume"] >= 0]

            return cleaned

        except Exception as e:
            logger.error(f"シーケンシャルデータクリーニングエラー: {e}")
            return data

    async def _save_pipeline_output(self, data: pd.DataFrame, output_format: str) -> Optional[str]:
        """パイプライン出力保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if output_format.lower() == "parquet":
                output_path = self.dask_processor.temp_dir / f"pipeline_output_{timestamp}.parquet"
                data.to_parquet(output_path, compression="snappy", index=False)
            elif output_format.lower() == "csv":
                output_path = self.dask_processor.temp_dir / f"pipeline_output_{timestamp}.csv"
                data.to_csv(output_path, index=False)
            else:
                logger.warning(f"サポートされていない出力形式: {output_format}")
                return None

            logger.info(f"パイプライン出力保存: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"パイプライン出力保存エラー: {e}")
            return None


# 便利関数
def create_dask_data_processor(
    enable_distributed: bool = True, n_workers: int = None, **kwargs
) -> DaskDataProcessor:
    """DaskDataProcessorファクトリ関数"""
    return DaskDataProcessor(enable_distributed=enable_distributed, n_workers=n_workers, **kwargs)


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #384 Dask分散処理テスト ===")

        processor = None
        try:
            # プロセッサー初期化
            processor = create_dask_data_processor(
                enable_distributed=True, n_workers=4, memory_limit="1GB"
            )

            # テストデータ
            test_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            # 1. 並列データ処理テスト
            print("\n1. 並列データ処理テスト")
            result_data = await processor.process_multiple_symbols_parallel(
                test_symbols, start_date, end_date, include_technical=True
            )
            print(f"処理結果: {len(result_data)}レコード")

            # 2. 相関分析テスト
            print("\n2. 分散相関分析テスト")
            analyzer = DaskStockAnalyzer(processor)
            portfolio_analysis = await analyzer.analyze_portfolio_performance_distributed(
                test_symbols[:3], benchmark_symbol=test_symbols[-1]
            )
            print(f"ポートフォリオ分析結果: {portfolio_analysis.get('portfolio_summary', {})}")

            # 3. バッチパイプラインテスト
            print("\n3. バッチパイプラインテスト")
            batch_processor = DaskBatchProcessor(processor)
            pipeline_result = await batch_processor.process_market_data_pipeline(
                test_symbols,
                ["fetch_data", "technical_analysis", "data_cleaning"],
                start_date,
                end_date,
            )
            print(f"パイプライン結果: {pipeline_result.get('records_processed', 0)}レコード処理")

            # 統計情報
            print("\n4. パフォーマンス統計")
            stats = processor.get_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # ヘルスチェック
            print("\n5. ヘルスステータス")
            health = processor.get_health_status()
            print(f"  状態: {health['status']}")
            print(f"  分散処理: {health['distributed_enabled']}")
            print(f"  Dask利用可能: {health['dask_available']}")

        except Exception as e:
            print(f"テスト実行エラー: {e}")

        finally:
            if processor:
                processor.cleanup()

    asyncio.run(main())
    print("\n=== Dask分散処理テスト完了 ===")
