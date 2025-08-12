#!/usr/bin/env python3
"""
TOPIX500並列処理エンジン
Issue #314: TOPIX500全銘柄対応

500銘柄の高速並列処理システム
目標: 20秒以内で500銘柄処理
"""

import asyncio
import gc
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psutil

try:
    from ..analysis.technical_analyzer import TechnicalAnalyzer
    from ..data.stock_fetcher import StockFetcher
    from ..data.topix500_master import TOPIX500MasterManager
    from ..utils.logging_config import get_context_logger
except ImportError:
    # テスト用フォールバック
    import sys
    from pathlib import Path

    # プロジェクトルートを追加
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from src.day_trade.data.stock_fetcher import StockFetcher
        from src.day_trade.data.topix500_master import TOPIX500MasterManager
    except ImportError:
        # モックオブジェクト作成
        class TOPIX500MasterManager:
            def get_all_active_symbols(self):
                return ["7203", "8306", "9984", "6758", "4689"]

            def create_balanced_batches(self, batch_size):
                symbols = self.get_all_active_symbols()
                return [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]

            def record_processing_performance(self, **kwargs):
                pass

        class StockFetcher:
            def fetch_stock_data(self, symbol, days=100):
                import numpy as np
                import pandas as pd

                dates = pd.date_range(start="2024-01-01", periods=days)
                data = pd.DataFrame(
                    {
                        "Open": np.random.uniform(2000, 3000, days),
                        "High": np.random.uniform(2100, 3100, days),
                        "Low": np.random.uniform(1900, 2900, days),
                        "Close": np.random.uniform(2000, 3000, days),
                        "Volume": np.random.randint(1000000, 10000000, days),
                    },
                    index=dates,
                )
                return data

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


@dataclass
class ProcessingResult:
    """処理結果データクラス"""

    symbol: str
    success: bool
    processing_time: float
    memory_used: float
    data_size: int
    analysis_results: Optional[Dict] = None
    error_message: Optional[str] = None


@dataclass
class BatchProcessingStats:
    """バッチ処理統計"""

    batch_id: int
    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    processing_time: float
    memory_peak: float
    avg_memory_per_symbol: float
    throughput: float  # symbols/second


class MemoryManager:
    """メモリ効率管理システム"""

    def __init__(self, max_memory_gb: float = 1.0):
        """
        初期化

        Args:
            max_memory_gb: 最大メモリ使用量（GB）
        """
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss

    def get_current_memory_usage(self) -> float:
        """現在のメモリ使用量取得（MB）"""
        current = self.process.memory_info().rss
        return (current - self.initial_memory) / 1024 / 1024

    def check_memory_limit(self) -> bool:
        """メモリ制限チェック"""
        current_usage = self.get_current_memory_usage() * 1024 * 1024
        return current_usage < self.max_memory_bytes

    def force_cleanup(self):
        """強制メモリクリーンアップ"""
        gc.collect()

    def get_available_memory_ratio(self) -> float:
        """利用可能メモリ率取得"""
        used = self.get_current_memory_usage() * 1024 * 1024
        return max(0, (self.max_memory_bytes - used) / self.max_memory_bytes)


class TOPIX500ParallelEngine:
    """
    TOPIX500並列処理エンジン

    500銘柄を効率的に並列処理するための高性能エンジン
    """

    def __init__(
        self,
        max_workers: int = None,
        batch_size: int = 50,
        memory_limit_gb: float = 1.0,
        processing_timeout: int = 300,
    ):
        """
        初期化

        Args:
            max_workers: 最大ワーカー数（Noneの場合はCPUコア数）
            batch_size: バッチサイズ
            memory_limit_gb: メモリ制限（GB）
            processing_timeout: 処理タイムアウト（秒）
        """
        self.max_workers = max_workers or min(mp.cpu_count(), 10)
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        self.processing_timeout = processing_timeout

        # コンポーネント初期化
        self.master_manager = TOPIX500MasterManager()
        self.memory_manager = MemoryManager(memory_limit_gb)

        # 統計情報
        self.processing_stats = []
        self.total_processing_time = 0
        self.peak_memory_usage = 0

        logger.info("TOPIX500並列処理エンジン初期化完了")
        logger.info(f"  最大ワーカー数: {self.max_workers}")
        logger.info(f"  バッチサイズ: {self.batch_size}")
        logger.info(f"  メモリ制限: {self.memory_limit_gb}GB")

    def process_single_symbol_batch(self, symbols_batch: List[str]) -> List[ProcessingResult]:
        """
        単一バッチの処理（プロセス内実行用）

        Args:
            symbols_batch: 処理対象銘柄バッチ

        Returns:
            処理結果リスト
        """
        batch_start_time = time.time()
        results = []

        try:
            # 局所的なインスタンス作成（プロセス分離）
            stock_fetcher = StockFetcher()
            # technical_analyzer = TechnicalAnalyzer()

            for symbol in symbols_batch:
                symbol_start_time = time.time()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024

                try:
                    # データ取得
                    stock_data = stock_fetcher.fetch_stock_data(symbol, days=100)

                    if stock_data is None or stock_data.empty:
                        results.append(
                            ProcessingResult(
                                symbol=symbol,
                                success=False,
                                processing_time=time.time() - symbol_start_time,
                                memory_used=0,
                                data_size=0,
                                error_message="データ取得失敗",
                            )
                        )
                        continue

                    # 簡易分析（詳細分析は別途実装）
                    analysis_results = {
                        "symbol": symbol,
                        "current_price": float(stock_data["Close"].iloc[-1]),
                        "volume": int(stock_data["Volume"].iloc[-1]),
                        "price_change": float(stock_data["Close"].pct_change().iloc[-1]),
                        "volatility": float(
                            stock_data["Close"].pct_change().rolling(20).std().iloc[-1]
                        ),
                        "data_points": len(stock_data),
                    }

                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                    processing_time = time.time() - symbol_start_time

                    results.append(
                        ProcessingResult(
                            symbol=symbol,
                            success=True,
                            processing_time=processing_time,
                            memory_used=memory_after - memory_before,
                            data_size=len(stock_data),
                            analysis_results=analysis_results,
                        )
                    )

                    # メモリクリーンアップ
                    del stock_data
                    gc.collect()

                except Exception as e:
                    processing_time = time.time() - symbol_start_time
                    results.append(
                        ProcessingResult(
                            symbol=symbol,
                            success=False,
                            processing_time=processing_time,
                            memory_used=0,
                            data_size=0,
                            error_message=str(e),
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"バッチ処理エラー: {e}")
            # エラー時は空の結果を返す
            return [
                ProcessingResult(
                    symbol=symbol,
                    success=False,
                    processing_time=0,
                    memory_used=0,
                    data_size=0,
                    error_message=f"バッチ処理エラー: {str(e)}",
                )
                for symbol in symbols_batch
            ]

    async def process_all_symbols_async(
        self, symbols: Optional[List[str]] = None
    ) -> Tuple[List[ProcessingResult], Dict[str, Any]]:
        """
        全銘柄非同期並列処理

        Args:
            symbols: 処理対象銘柄リスト（Noneの場合は全アクティブ銘柄）

        Returns:
            (処理結果リスト, 統計情報辞書)
        """
        start_time = time.time()

        # 銘柄リスト取得
        if symbols is None:
            symbols = self.master_manager.get_all_active_symbols()

        if not symbols:
            logger.warning("処理対象銘柄がありません")
            return [], {}

        logger.info(f"非同期並列処理開始: {len(symbols)}銘柄")

        # バッチ作成
        symbol_batches = self.master_manager.create_balanced_batches(self.batch_size)

        # 実際の銘柄リストに合わせてバッチを調整
        if len(symbols) != sum(len(batch) for batch in symbol_batches):
            # 必要に応じてバッチを再作成
            symbol_batches = [
                symbols[i : i + self.batch_size] for i in range(0, len(symbols), self.batch_size)
            ]

        logger.info(f"バッチ分割完了: {len(symbol_batches)}バッチ")

        # 並列処理実行
        all_results = []
        batch_stats = []

        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # バッチを並列実行
                future_to_batch = {
                    executor.submit(self.process_single_symbol_batch, batch): i
                    for i, batch in enumerate(symbol_batches)
                }

                for future in as_completed(future_to_batch, timeout=self.processing_timeout):
                    batch_id = future_to_batch[future]
                    batch_start_time = time.time()

                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)

                        # バッチ統計計算
                        successful = sum(1 for r in batch_results if r.success)
                        failed = len(batch_results) - successful
                        batch_time = time.time() - batch_start_time

                        batch_stats.append(
                            BatchProcessingStats(
                                batch_id=batch_id,
                                total_symbols=len(batch_results),
                                successful_symbols=successful,
                                failed_symbols=failed,
                                processing_time=batch_time,
                                memory_peak=max(r.memory_used for r in batch_results),
                                avg_memory_per_symbol=sum(r.memory_used for r in batch_results)
                                / len(batch_results),
                                throughput=len(batch_results) / batch_time if batch_time > 0 else 0,
                            )
                        )

                        logger.info(f"バッチ{batch_id}完了: {successful}/{len(batch_results)}成功")

                    except Exception as e:
                        logger.error(f"バッチ{batch_id}処理エラー: {e}")
                        # エラー時の結果を追加
                        batch_symbols = symbol_batches[batch_id]
                        error_results = [
                            ProcessingResult(
                                symbol=symbol,
                                success=False,
                                processing_time=0,
                                memory_used=0,
                                data_size=0,
                                error_message=f"バッチ処理エラー: {str(e)}",
                            )
                            for symbol in batch_symbols
                        ]
                        all_results.extend(error_results)

        except Exception as e:
            logger.error(f"並列処理エラー: {e}")
            # 全体的なエラーの場合
            error_results = [
                ProcessingResult(
                    symbol=symbol,
                    success=False,
                    processing_time=0,
                    memory_used=0,
                    data_size=0,
                    error_message=f"並列処理エラー: {str(e)}",
                )
                for symbol in symbols
            ]
            return error_results, {}

        total_time = time.time() - start_time

        # 統計情報計算
        successful_results = [r for r in all_results if r.success]
        failed_results = [r for r in all_results if not r.success]

        statistics = {
            "total_symbols": len(symbols),
            "successful_symbols": len(successful_results),
            "failed_symbols": len(failed_results),
            "success_rate": len(successful_results) / len(symbols) * 100,
            "total_processing_time": total_time,
            "avg_time_per_symbol": sum(r.processing_time for r in all_results) / len(all_results),
            "throughput": len(symbols) / total_time,
            "memory_usage": sum(r.memory_used for r in all_results),
            "avg_memory_per_symbol": sum(r.memory_used for r in all_results) / len(all_results),
            "batch_count": len(symbol_batches),
            "batch_stats": batch_stats,
            "target_achieved": total_time <= 20.0,  # 20秒目標
            "memory_limit_ok": sum(r.memory_used for r in all_results)
            <= self.memory_limit_gb * 1024,
        }

        # パフォーマンス記録
        self.master_manager.record_processing_performance(
            total_symbols=len(symbols),
            successful_symbols=len(successful_results),
            processing_time=total_time,
            memory_usage=statistics["memory_usage"],
            error_details=(
                f"失敗率: {statistics['failed_symbols']/len(symbols)*100:.1f}%"
                if failed_results
                else None
            ),
        )

        logger.info(f"並列処理完了: {total_time:.1f}秒, {statistics['success_rate']:.1f}%成功率")

        return all_results, statistics

    def process_all_symbols(
        self, symbols: Optional[List[str]] = None
    ) -> Tuple[List[ProcessingResult], Dict[str, Any]]:
        """
        全銘柄処理（同期版）

        Args:
            symbols: 処理対象銘柄リスト

        Returns:
            (処理結果リスト, 統計情報辞書)
        """
        return asyncio.run(self.process_all_symbols_async(symbols))

    def generate_performance_report(
        self, results: List[ProcessingResult], statistics: Dict[str, Any]
    ) -> str:
        """
        パフォーマンスレポート生成

        Args:
            results: 処理結果リスト
            statistics: 統計情報

        Returns:
            レポート文字列
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        report = f"""
TOPIX500並列処理パフォーマンスレポート
{'='*50}
実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【基本統計】
処理銘柄数: {statistics['total_symbols']}
成功銘柄数: {statistics['successful_symbols']}
失敗銘柄数: {statistics['failed_symbols']}
成功率: {statistics['success_rate']:.1f}%

【パフォーマンス指標】
総処理時間: {statistics['total_processing_time']:.1f}秒
平均処理時間/銘柄: {statistics['avg_time_per_symbol']:.3f}秒
スループット: {statistics['throughput']:.1f} 銘柄/秒
20秒目標達成: {'○' if statistics['target_achieved'] else '×'}

【メモリ使用量】
総メモリ使用量: {statistics['memory_usage']:.1f}MB
平均メモリ/銘柄: {statistics['avg_memory_per_symbol']:.1f}MB
メモリ制限内: {'○' if statistics['memory_limit_ok'] else '×'}

【バッチ処理統計】
バッチ数: {statistics['batch_count']}
"""

        if "batch_stats" in statistics and statistics["batch_stats"]:
            report += "\nバッチ別詳細:\n"
            for batch_stat in statistics["batch_stats"]:
                report += f"  バッチ{batch_stat.batch_id}: {batch_stat.successful_symbols}/{batch_stat.total_symbols}成功 "
                report += (
                    f"({batch_stat.processing_time:.1f}秒, {batch_stat.throughput:.1f}銘柄/秒)\n"
                )

        if failed:
            report += "\n【失敗銘柄詳細】\n"
            for result in failed[:10]:  # 最初の10件
                report += f"  {result.symbol}: {result.error_message}\n"
            if len(failed) > 10:
                report += f"  ... 他{len(failed)-10}銘柄\n"

        report += f"\n{'='*50}\n"

        return report

    def optimize_performance(self) -> Dict[str, Any]:
        """
        パフォーマンス最適化提案

        Returns:
            最適化提案辞書
        """
        cpu_count = mp.cpu_count()
        memory_info = psutil.virtual_memory()

        # 最適化提案計算
        optimal_workers = min(cpu_count, 12)  # 最大12並列
        optimal_batch_size = max(25, min(100, 500 // optimal_workers))  # 動的バッチサイズ

        recommendations = {
            "current_workers": self.max_workers,
            "optimal_workers": optimal_workers,
            "current_batch_size": self.batch_size,
            "optimal_batch_size": optimal_batch_size,
            "system_cpu_cores": cpu_count,
            "system_memory_gb": memory_info.total / 1024 / 1024 / 1024,
            "recommended_memory_limit": min(2.0, memory_info.available / 1024 / 1024 / 1024 * 0.8),
            "optimization_tips": [
                f"ワーカー数を{optimal_workers}に調整",
                f"バッチサイズを{optimal_batch_size}に調整",
                "プロセスプールの再利用を考慮",
                "メモリ使用量の監視強化",
                "失敗銘柄の再試行機構追加",
            ],
        }

        return recommendations


def process_symbol_batch_worker(symbols_batch: List[str]) -> List[ProcessingResult]:
    """
    ワーカープロセス用のバッチ処理関数

    Args:
        symbols_batch: 処理対象銘柄バッチ

    Returns:
        処理結果リスト
    """
    engine = TOPIX500ParallelEngine()
    return engine.process_single_symbol_batch(symbols_batch)


if __name__ == "__main__":
    print("=== TOPIX500並列処理エンジン テスト ===")

    try:
        # エンジン初期化
        engine = TOPIX500ParallelEngine(
            max_workers=4,  # テスト用に制限
            batch_size=10,
            memory_limit_gb=0.5,
        )

        print("1. テスト用銘柄リスト作成...")
        test_symbols = [
            "7203",
            "8306",
            "9984",
            "6758",
            "4689",
            "8058",
            "8031",
            "4568",
            "9501",
            "8801",
        ]
        print(f"   テスト銘柄: {len(test_symbols)}銘柄")

        print("2. 並列処理実行...")
        start_time = time.time()
        results, stats = engine.process_all_symbols(test_symbols)
        execution_time = time.time() - start_time

        print(f"   実行時間: {execution_time:.1f}秒")
        print(f"   成功率: {stats['success_rate']:.1f}%")
        print(f"   スループット: {stats['throughput']:.1f} 銘柄/秒")

        print("3. パフォーマンスレポート生成...")
        report = engine.generate_performance_report(results, stats)
        print("   レポート生成完了")

        print("4. パフォーマンス最適化提案...")
        recommendations = engine.optimize_performance()
        print(f"   推奨ワーカー数: {recommendations['optimal_workers']}")
        print(f"   推奨バッチサイズ: {recommendations['optimal_batch_size']}")

        print("\n=== テスト結果 ===")
        print(f"処理銘柄数: {len(test_symbols)}")
        print(f"成功銘柄数: {sum(1 for r in results if r.success)}")
        print(f"実行時間: {execution_time:.1f}秒")
        print(f"目標時間(20秒)内: {'○' if execution_time <= 20 else '×'}")

        print("\n[OK] TOPIX500並列処理エンジン テスト完了！")

    except Exception as e:
        print(f"[NG] テストエラー: {e}")
        import traceback

        traceback.print_exc()
