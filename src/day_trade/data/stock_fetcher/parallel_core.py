"""
並列処理コア機能
並列実行エンジンと結果処理
"""

from typing import Dict, List, Optional

import pandas as pd

# 並列処理サポート (Issue #383)
try:
    from ...utils.parallel_executor_manager import get_global_executor_manager

    PARALLEL_SUPPORT = True
except ImportError:
    PARALLEL_SUPPORT = False


class ParallelProcessorCore:
    """
    並列処理のコア機能を提供するクラス
    """

    def __init__(self, stock_fetcher):
        """
        Args:
            stock_fetcher: メインのStockFetcherインスタンス
        """
        self.stock_fetcher = stock_fetcher
        self.logger = stock_fetcher.logger

    def parallel_get_historical_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        複数銘柄のヒストリカルデータを並列取得 (I/Oバウンド最適化)

        Args:
            symbols: 銘柄コードリスト
            period: データ期間
            interval: データ間隔
            max_concurrent: 最大同時実行数

        Returns:
            銘柄別ヒストリカルデータ辞書
        """
        if not PARALLEL_SUPPORT:
            self.logger.warning("並列処理サポートが無効です。従来の方法で実行します")
            return self._sequential_get_historical_data(symbols, period, interval)

        self.logger.info(f"並列ヒストリカルデータ取得開始: {len(symbols)}銘柄")

        # 並列実行タスクを準備
        tasks = []
        for symbol in symbols:
            task = (
                self.stock_fetcher.get_historical_data,
                (symbol, period, interval),
                {},
            )
            tasks.append(task)

        # 並列実行マネージャーで実行
        parallel_manager = get_global_executor_manager()
        results = parallel_manager.execute_batch(
            tasks, max_concurrent=max_concurrent or min(len(symbols), 10)
        )

        # 結果を整理
        symbol_results = {}
        for symbol, exec_result in zip(symbols, results):
            if exec_result.success:
                symbol_results[symbol] = exec_result.result
            else:
                self.logger.error(
                    f"ヒストリカルデータ取得失敗: {symbol}, エラー: {exec_result.error}"
                )
                symbol_results[symbol] = None

        # 統計情報をログ出力
        successful_count = sum(
            1 for result in symbol_results.values() if result is not None
        )
        total_time = max(r.execution_time_ms for r in results) if results else 0

        self.logger.info(
            f"並列ヒストリカルデータ取得完了: {successful_count}/{len(symbols)}銘柄成功, "
            f"実行時間: {total_time:.1f}ms"
        )

        return symbol_results

    def parallel_get_current_prices(
        self, symbols: List[str], max_concurrent: Optional[int] = None
    ) -> Dict[str, Optional[Dict]]:
        """
        複数銘柄の現在価格を並列取得 (I/Oバウンド最適化)

        Args:
            symbols: 銘柄コードリスト
            max_concurrent: 最大同時実行数

        Returns:
            銘柄別現在価格辞書
        """
        if not PARALLEL_SUPPORT:
            self.logger.warning("並列処理サポートが無効です。従来の方法で実行します")
            return {
                symbol: self.stock_fetcher.get_current_price(symbol)
                for symbol in symbols
            }

        self.logger.info(f"並列現在価格取得開始: {len(symbols)}銘柄")

        # 並列実行タスクを準備
        tasks = []
        for symbol in symbols:
            task = (self.stock_fetcher.get_current_price, (symbol,), {})
            tasks.append(task)

        # 並列実行マネージャーで実行（I/Oバウンドとしてヒント）
        parallel_manager = get_global_executor_manager()
        results = parallel_manager.execute_batch(
            tasks,
            max_concurrent=max_concurrent
            or min(len(symbols), 15),  # 価格データは軽いので多め
        )

        # 結果を整理
        symbol_results = {}
        for symbol, exec_result in zip(symbols, results):
            if exec_result.success:
                symbol_results[symbol] = exec_result.result
            else:
                self.logger.error(
                    f"現在価格取得失敗: {symbol}, エラー: {exec_result.error}"
                )
                symbol_results[symbol] = None

        # 統計情報をログ出力
        successful_count = sum(
            1 for result in symbol_results.values() if result is not None
        )
        total_time = max(r.execution_time_ms for r in results) if results else 0

        self.logger.info(
            f"並列現在価格取得完了: {successful_count}/{len(symbols)}銘柄成功, "
            f"実行時間: {total_time:.1f}ms"
        )

        return symbol_results

    def parallel_get_company_info(
        self, symbols: List[str], max_concurrent: Optional[int] = None
    ) -> Dict[str, Optional[Dict]]:
        """
        複数銘柄の企業情報を並列取得 (I/Oバウンド最適化)

        Args:
            symbols: 銘柄コードリスト
            max_concurrent: 最大同時実行数

        Returns:
            銘柄別企業情報辞書
        """
        if not PARALLEL_SUPPORT:
            self.logger.warning("並列処理サポートが無効です。従来の方法で実行します")
            return {
                symbol: self.stock_fetcher.get_company_info(symbol)
                for symbol in symbols
            }

        self.logger.info(f"並列企業情報取得開始: {len(symbols)}銘柄")

        # 並列実行タスクを準備
        tasks = []
        for symbol in symbols:
            task = (self.stock_fetcher.get_company_info, (symbol,), {})
            tasks.append(task)

        # 並列実行マネージャーで実行（I/Oバウンドとしてヒント）
        parallel_manager = get_global_executor_manager()
        results = parallel_manager.execute_batch(
            tasks,
            max_concurrent=max_concurrent
            or min(len(symbols), 8),  # 企業情報は重めなので控えめ
        )

        # 結果を整理
        symbol_results = {}
        for symbol, exec_result in zip(symbols, results):
            if exec_result.success:
                symbol_results[symbol] = exec_result.result
            else:
                self.logger.error(
                    f"企業情報取得失敗: {symbol}, エラー: {exec_result.error}"
                )
                symbol_results[symbol] = None

        # 統計情報をログ出力
        successful_count = sum(
            1 for result in symbol_results.values() if result is not None
        )
        total_time = max(r.execution_time_ms for r in results) if results else 0

        self.logger.info(
            f"並列企業情報取得完了: {successful_count}/{len(symbols)}銘柄成功, "
            f"実行時間: {total_time:.1f}ms"
        )

        return symbol_results

    def _sequential_get_historical_data(
        self, symbols: List[str], period: str, interval: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """シーケンシャル実行フォールバック（並列処理が利用できない場合）"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.stock_fetcher.get_historical_data(
                    symbol, period, interval
                )
            except Exception as e:
                self.logger.error(f"シーケンシャル取得失敗: {symbol}, エラー: {e}")
                results[symbol] = None
        return results