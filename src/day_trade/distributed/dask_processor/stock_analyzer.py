#!/usr/bin/env python3
"""
Dask Stock Analyzer
Issue #384: 並列処理のさらなる強化 - Stock Analysis Module

Dask分散処理による株価分析とポートフォリオパフォーマンス評価を提供する
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Dask関連インポート（オプショナル）
try:
    from dask import compute
    from dask.delayed import delayed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# プロジェクトモジュール
try:
    from ...data.enhanced_stock_fetcher import create_enhanced_stock_fetcher
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def create_enhanced_stock_fetcher(**kwargs):
        class MockFetcher:
            def get_historical_data(self, symbol, start_date, end_date):
                return pd.DataFrame({
                    "timestamp": [datetime.now()],
                    "close": [100]
                })
        return MockFetcher()


logger = get_context_logger(__name__)


class DaskStockAnalyzer:
    """Dask特化型株価分析器"""

    def __init__(self, dask_processor):
        """
        初期化

        Args:
            dask_processor: DaskDataProcessorインスタンス
        """
        self.dask_processor = dask_processor

    async def analyze_portfolio_performance_distributed(
        self,
        portfolio_symbols: List[str],
        benchmark_symbol: str = "SPY",
        analysis_period_days: int = 252,
    ) -> Dict[str, Any]:
        """
        分散ポートフォリオパフォーマンス分析

        Args:
            portfolio_symbols: ポートフォリオ銘柄リスト
            benchmark_symbol: ベンチマーク銘柄（デフォルト：SPY）
            analysis_period_days: 分析期間（日数）

        Returns:
            ポートフォリオ分析結果
        """
        logger.info(
            f"分散ポートフォリオ分析: {len(portfolio_symbols)}銘柄 vs {benchmark_symbol}"
        )

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
        """
        分散ポートフォリオ分析実行

        Args:
            price_data: 価格データ
            portfolio_symbols: ポートフォリオ銘柄
            benchmark_symbol: ベンチマーク銘柄

        Returns:
            分析結果
        """
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
                calculate_symbol_metrics(symbol, group)
                for symbol, group in symbol_groups
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
                portfolio_total_return = sum(
                    m["total_return"] for m in portfolio_metrics
                ) / len(portfolio_metrics)
                portfolio_avg_volatility = sum(
                    m["volatility"] for m in portfolio_metrics
                ) / len(portfolio_metrics)
                portfolio_avg_sharpe = sum(
                    m["sharpe_ratio"] for m in portfolio_metrics
                ) / len(portfolio_metrics)
            else:
                portfolio_total_return = portfolio_avg_volatility = (
                    portfolio_avg_sharpe
                ) = 0

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
        """
        シーケンシャルポートフォリオ分析（フォールバック）

        Args:
            price_data: 価格データ
            portfolio_symbols: ポートフォリオ銘柄
            benchmark_symbol: ベンチマーク銘柄

        Returns:
            分析結果
        """
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
                        "volatility": (
                            returns.std() * np.sqrt(252) * 100
                            if len(returns) > 1
                            else 0
                        ),
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
        """
        最大ドローダウン計算

        Args:
            prices: 価格データ

        Returns:
            最大ドローダウン（%）
        """
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min() * 100
        except Exception:
            return 0.0

    async def analyze_market_correlation_distributed(
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
        if not DASK_AVAILABLE or not self.dask_processor.enable_distributed:
            logger.warning("分散処理が利用できないため、制限された分析を実行")
            return await self._analyze_correlation_sequential(
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
            aligned_data = {
                symbol: prices[-min_length:] for symbol, prices in valid_data.items()
            }

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
            return await self._analyze_correlation_sequential(
                symbols, analysis_period_days, correlation_window
            )

    async def _analyze_correlation_sequential(
        self, symbols: List[str], analysis_period_days: int, correlation_window: int
    ) -> pd.DataFrame:
        """
        シーケンシャル相関分析（フォールバック）

        Args:
            symbols: 分析対象銘柄
            analysis_period_days: 分析期間
            correlation_window: 相関ウィンドウ

        Returns:
            相関分析結果
        """
        logger.info(f"シーケンシャル相関分析実行: {len(symbols)}銘柄")

        try:
            # 簡略化された相関分析
            price_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=analysis_period_days)

            for symbol in symbols:
                try:
                    data = self.dask_processor.stock_fetcher.get_historical_data(
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
            aligned_data = {
                symbol: prices[-min_length:] for symbol, prices in price_data.items()
            }

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