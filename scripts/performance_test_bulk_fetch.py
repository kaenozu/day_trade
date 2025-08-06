#!/usr/bin/env python3
"""
銘柄一括取得のパフォーマンステスト

大量の銘柄データの取得・処理・保存のパフォーマンスを測定し、
最適化の効果を検証する。

機能:
- 銘柄データ取得の並列処理性能測定
- データベース書き込み性能測定
- メモリ使用量監視
- 処理時間の詳細分析
- ボトルネック特定

Usage:
    python scripts/performance_test_bulk_fetch.py
    python scripts/performance_test_bulk_fetch.py --symbols 100 --threads 5
    python scripts/performance_test_bulk_fetch.py --profile --output-report results.json
"""

import argparse
import json
import logging
import sys
import threading
import time
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import Mock

import psutil

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.data.stock_master import StockMasterManager, create_stock_master_manager
from src.day_trade.models.database import DatabaseConfig, DatabaseManager
from src.day_trade.models.stock import Stock

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """パフォーマンス測定結果"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success_count: int
    error_count: int
    total_count: int
    throughput: float  # 件/秒
    memory_usage_mb: float
    cpu_usage_percent: float

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.total_count if self.total_count > 0 else 0.0


class PerformanceMonitor:
    """パフォーマンス監視"""

    def __init__(self):
        self.process = psutil.Process()
        self.metrics_history: List[Dict[str, Any]] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval: float = 1.0):
        """監視開始"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()

    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self, interval: float):
        """監視ループ"""
        while self.monitoring:
            try:
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': self.process.cpu_percent(),
                    'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                    'memory_percent': self.process.memory_percent(),
                    'threads_count': self.process.num_threads(),
                    'connections_count': len(self.process.connections()) if hasattr(self.process, 'connections') else 0
                }
                self.metrics_history.append(metrics)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            time.sleep(interval)

    def get_peak_metrics(self) -> Dict[str, float]:
        """ピーク値を取得"""
        if not self.metrics_history:
            return {}

        return {
            'peak_cpu_percent': max(m['cpu_percent'] for m in self.metrics_history),
            'peak_memory_mb': max(m['memory_mb'] for m in self.metrics_history),
            'peak_memory_percent': max(m['memory_percent'] for m in self.metrics_history),
            'peak_threads': max(m['threads_count'] for m in self.metrics_history),
            'peak_connections': max(m['connections_count'] for m in self.metrics_history)
        }

    def get_average_metrics(self) -> Dict[str, float]:
        """平均値を取得"""
        if not self.metrics_history:
            return {}

        count = len(self.metrics_history)
        return {
            'avg_cpu_percent': sum(m['cpu_percent'] for m in self.metrics_history) / count,
            'avg_memory_mb': sum(m['memory_mb'] for m in self.metrics_history) / count,
            'avg_memory_percent': sum(m['memory_percent'] for m in self.metrics_history) / count,
            'avg_threads': sum(m['threads_count'] for m in self.metrics_history) / count,
            'avg_connections': sum(m['connections_count'] for m in self.metrics_history) / count
        }


class BulkFetchPerformanceTester:
    """銘柄一括取得パフォーマンステスター"""

    def __init__(self, use_mock: bool = True):
        """
        Args:
            use_mock: モックを使用するか（実際のAPI呼び出しを避けるため）
        """
        self.use_mock = use_mock
        self.monitor = PerformanceMonitor()
        self.results: List[PerformanceMetrics] = []

        # テスト用銘柄コードリスト
        self.test_symbols = self._generate_test_symbols(1000)

        if use_mock:
            self.stock_fetcher = self._create_mock_fetcher()
        else:
            self.stock_fetcher = StockFetcher()
            logger.warning("実際のAPIを使用します。レート制限にご注意ください")

    def _generate_test_symbols(self, count: int) -> List[str]:
        """テスト用銘柄コードを生成"""
        # 実際の日本株コード風の4桁数字を生成
        symbols = []
        for i in range(1000, 1000 + count):
            symbols.append(str(i))
        return symbols

    def _create_mock_fetcher(self) -> Mock:
        """モックStockFetcherを作成"""
        mock_fetcher = Mock()

        def mock_get_current_price(symbol):
            # ランダムな遅延を追加してリアルな処理時間をシミュレート
            import random
            time.sleep(random.uniform(0.01, 0.1))

            return {
                'current_price': 1000 + hash(symbol) % 5000,
                'change': random.uniform(-100, 100),
                'change_percent': random.uniform(-5, 5),
                'volume': random.randint(10000, 100000),
                'market_cap': random.randint(100000000, 10000000000)
            }

        def mock_get_company_info(symbol):
            time.sleep(random.uniform(0.05, 0.2))

            return {
                'name': f'テスト会社{symbol}',
                'sector': random.choice(['テクノロジー', '金融', '製造業', 'ヘルスケア']),
                'industry': f'業界{hash(symbol) % 10}',
                'market_cap': random.randint(100000000, 10000000000)
            }

        mock_fetcher.get_current_price.side_effect = mock_get_current_price
        mock_fetcher.get_company_info.side_effect = mock_get_company_info

        return mock_fetcher

    def test_sequential_fetch(self, symbols: List[str]) -> PerformanceMetrics:
        """シーケンシャル（逐次）取得のテスト"""
        logger.info(f"シーケンシャル取得テスト開始: {len(symbols)}銘柄")

        start_time = time.perf_counter()
        memory_start = self.monitor.process.memory_info().rss / 1024 / 1024

        success_count = 0
        error_count = 0

        self.monitor.start_monitoring()

        for symbol in symbols:
            try:
                if self.use_mock:
                    self.stock_fetcher.get_current_price(symbol)
                else:
                    self.stock_fetcher.get_current_price(symbol)
                success_count += 1

            except Exception as e:
                logger.debug(f"取得エラー {symbol}: {e}")
                error_count += 1

        self.monitor.stop_monitoring()

        end_time = time.perf_counter()
        duration = end_time - start_time
        memory_end = self.monitor.process.memory_info().rss / 1024 / 1024
        peak_metrics = self.monitor.get_peak_metrics()

        metrics = PerformanceMetrics(
            operation="sequential_fetch",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success_count=success_count,
            error_count=error_count,
            total_count=len(symbols),
            throughput=success_count / duration if duration > 0 else 0,
            memory_usage_mb=memory_end - memory_start,
            cpu_usage_percent=peak_metrics.get('peak_cpu_percent', 0)
        )

        logger.info(f"シーケンシャル取得完了: {duration:.2f}秒, {metrics.throughput:.1f}件/秒")

        self.results.append(metrics)
        return metrics

    def test_parallel_fetch(self, symbols: List[str], max_workers: int = 5) -> PerformanceMetrics:
        """並列取得のテスト"""
        logger.info(f"並列取得テスト開始: {len(symbols)}銘柄, {max_workers}スレッド")

        start_time = time.perf_counter()
        memory_start = self.monitor.process.memory_info().rss / 1024 / 1024

        success_count = 0
        error_count = 0

        self.monitor.start_monitoring()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 全タスクをサブミット
            future_to_symbol = {
                executor.submit(self._fetch_single_stock, symbol): symbol
                for symbol in symbols
            }

            # 結果を収集
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        error_count += 1

                except Exception as e:
                    logger.debug(f"並列取得エラー {symbol}: {e}")
                    error_count += 1

        self.monitor.stop_monitoring()

        end_time = time.perf_counter()
        duration = end_time - start_time
        memory_end = self.monitor.process.memory_info().rss / 1024 / 1024
        peak_metrics = self.monitor.get_peak_metrics()

        metrics = PerformanceMetrics(
            operation=f"parallel_fetch_{max_workers}",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success_count=success_count,
            error_count=error_count,
            total_count=len(symbols),
            throughput=success_count / duration if duration > 0 else 0,
            memory_usage_mb=memory_end - memory_start,
            cpu_usage_percent=peak_metrics.get('peak_cpu_percent', 0)
        )

        logger.info(f"並列取得完了: {duration:.2f}秒, {metrics.throughput:.1f}件/秒")

        self.results.append(metrics)
        return metrics

    def _fetch_single_stock(self, symbol: str) -> bool:
        """単一銘柄の取得"""
        try:
            if self.use_mock:
                self.stock_fetcher.get_current_price(symbol)
            else:
                self.stock_fetcher.get_current_price(symbol)
            return True

        except Exception as e:
            logger.debug(f"取得失敗 {symbol}: {e}")
            return False

    def test_database_bulk_insert(self, record_count: int) -> PerformanceMetrics:
        """データベース一括挿入のテスト"""
        logger.info(f"データベース一括挿入テスト開始: {record_count}件")

        # テスト用データ作成
        test_stocks = []
        for i in range(record_count):
            test_stocks.append({
                'code': f"{1000 + i:04d}",
                'name': f'テスト会社{i}',
                'market': 'テストマーケット',
                'sector': 'テストセクター',
                'industry': 'テスト業界'
            })

        # テスト用インメモリDBを使用
        config = DatabaseConfig.for_testing()
        db_manager = DatabaseManager(config)
        db_manager.create_tables()

        start_time = time.perf_counter()
        memory_start = self.monitor.process.memory_info().rss / 1024 / 1024

        success_count = 0
        error_count = 0

        self.monitor.start_monitoring()

        try:
            with db_manager.session_scope() as session:
                for stock_data in test_stocks:
                    try:
                        stock = Stock(**stock_data)
                        session.add(stock)
                        success_count += 1

                    except Exception as e:
                        logger.debug(f"挿入エラー {stock_data['code']}: {e}")
                        error_count += 1

        except Exception as e:
            logger.error(f"データベース操作エラー: {e}")
            error_count += record_count - success_count

        self.monitor.stop_monitoring()

        end_time = time.perf_counter()
        duration = end_time - start_time
        memory_end = self.monitor.process.memory_info().rss / 1024 / 1024
        peak_metrics = self.monitor.get_peak_metrics()

        metrics = PerformanceMetrics(
            operation="database_bulk_insert",
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            success_count=success_count,
            error_count=error_count,
            total_count=record_count,
            throughput=success_count / duration if duration > 0 else 0,
            memory_usage_mb=memory_end - memory_start,
            cpu_usage_percent=peak_metrics.get('peak_cpu_percent', 0)
        )

        logger.info(f"データベース挿入完了: {duration:.2f}秒, {metrics.throughput:.1f}件/秒")

        self.results.append(metrics)
        return metrics

    def run_comprehensive_test(
        self,
        symbol_counts: List[int] = [10, 50, 100, 500],
        thread_counts: List[int] = [1, 3, 5, 10]
    ) -> List[PerformanceMetrics]:
        """包括的なパフォーマンステスト"""
        logger.info("=== 包括的パフォーマンステスト開始 ===")

        all_results = []

        for symbol_count in symbol_counts:
            test_symbols = self.test_symbols[:symbol_count]

            # シーケンシャルテスト
            sequential_result = self.test_sequential_fetch(test_symbols)
            all_results.append(sequential_result)

            # 並列テスト（複数のスレッド数）
            for thread_count in thread_counts:
                if thread_count > 1:
                    parallel_result = self.test_parallel_fetch(test_symbols, thread_count)
                    all_results.append(parallel_result)

            # 間隔を空ける
            time.sleep(1)

        # データベーステスト
        for record_count in [100, 500, 1000]:
            db_result = self.test_database_bulk_insert(record_count)
            all_results.append(db_result)

        logger.info("=== 包括的パフォーマンステスト完了 ===")

        self.results.extend(all_results)
        return all_results

    def generate_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポートを生成"""
        if not self.results:
            return {}

        # 結果を操作タイプ別にグループ化
        grouped_results = {}
        for metric in self.results:
            operation_type = metric.operation.split('_')[0]  # 例: "parallel_fetch_5" -> "parallel"
            if operation_type not in grouped_results:
                grouped_results[operation_type] = []
            grouped_results[operation_type].append(metric)

        # 統計情報を計算
        report = {
            'test_summary': {
                'total_tests': len(self.results),
                'test_timestamp': datetime.now().isoformat(),
                'use_mock': self.use_mock
            },
            'performance_by_operation': {}
        }

        for operation_type, metrics_list in grouped_results.items():
            # 各操作タイプの統計
            throughputs = [m.throughput for m in metrics_list]
            durations = [m.duration for m in metrics_list]
            success_rates = [m.success_rate for m in metrics_list]

            report['performance_by_operation'][operation_type] = {
                'test_count': len(metrics_list),
                'throughput': {
                    'min': min(throughputs),
                    'max': max(throughputs),
                    'avg': sum(throughputs) / len(throughputs),
                    'unit': '件/秒'
                },
                'duration': {
                    'min': min(durations),
                    'max': max(durations),
                    'avg': sum(durations) / len(durations),
                    'unit': '秒'
                },
                'success_rate': {
                    'min': min(success_rates),
                    'max': max(success_rates),
                    'avg': sum(success_rates) / len(success_rates),
                    'unit': '%'
                }
            }

        # 詳細結果
        report['detailed_results'] = [asdict(metric) for metric in self.results]

        return report

    def find_optimal_configuration(self) -> Dict[str, Any]:
        """最適な設定を特定"""
        if not self.results:
            return {}

        # 並列処理の結果から最適なスレッド数を特定
        parallel_results = [r for r in self.results if 'parallel_fetch' in r.operation]

        if parallel_results:
            best_parallel = max(parallel_results, key=lambda x: x.throughput)
            optimal_threads = int(best_parallel.operation.split('_')[-1])
        else:
            optimal_threads = 1

        # データベース処理の最適バッチサイズを特定
        db_results = [r for r in self.results if 'database' in r.operation]
        optimal_batch_size = 100  # デフォルト

        if db_results:
            best_db = max(db_results, key=lambda x: x.throughput)
            optimal_batch_size = best_db.total_count

        return {
            'optimal_thread_count': optimal_threads,
            'optimal_batch_size': optimal_batch_size,
            'best_throughput': best_parallel.throughput if parallel_results else 0,
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """パフォーマンス改善の推奨事項を生成"""
        recommendations = []

        if not self.results:
            return recommendations

        # 並列処理の効果を分析
        sequential_results = [r for r in self.results if r.operation == 'sequential_fetch']
        parallel_results = [r for r in self.results if 'parallel_fetch' in r.operation]

        if sequential_results and parallel_results:
            seq_avg = sum(r.throughput for r in sequential_results) / len(sequential_results)
            par_avg = sum(r.throughput for r in parallel_results) / len(parallel_results)

            improvement = (par_avg - seq_avg) / seq_avg * 100 if seq_avg > 0 else 0

            if improvement > 50:
                recommendations.append(f"並列処理により{improvement:.1f}%の性能向上が期待できます")
            elif improvement < 20:
                recommendations.append("並列処理の効果が限定的です。I/Oバウンドな処理の見直しを検討してください")

        # メモリ使用量の分析
        high_memory_results = [r for r in self.results if r.memory_usage_mb > 100]
        if high_memory_results:
            recommendations.append("メモリ使用量が高い処理があります。バッチサイズの調整を検討してください")

        # エラー率の分析
        high_error_results = [r for r in self.results if r.success_rate < 0.9]
        if high_error_results:
            recommendations.append("エラー率が高い処理があります。エラーハンドリングとリトライ機能の強化を検討してください")

        return recommendations


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="銘柄一括取得パフォーマンステスト",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--symbols', '-s',
        type=int,
        default=100,
        help='テスト対象銘柄数（デフォルト: 100）'
    )

    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=5,
        help='並列処理のスレッド数（デフォルト: 5）'
    )

    parser.add_argument(
        '--comprehensive',
        action='store_true',
        help='包括的なテストを実行'
    )

    parser.add_argument(
        '--use-real-api',
        action='store_true',
        help='実際のAPIを使用（デフォルトはモック）'
    )

    parser.add_argument(
        '--output-report',
        type=str,
        help='レポートファイルの出力パス'
    )

    parser.add_argument(
        '--profile',
        action='store_true',
        help='詳細プロファイリングを実行'
    )

    args = parser.parse_args()

    try:
        logger.info("=== 銘柄一括取得パフォーマンステスト開始 ===")

        # テスター初期化
        tester = BulkFetchPerformanceTester(use_mock=not args.use_real_api)

        if args.comprehensive:
            # 包括的テスト
            tester.run_comprehensive_test()
        else:
            # 基本テスト
            test_symbols = tester.test_symbols[:args.symbols]

            # シーケンシャルテスト
            tester.test_sequential_fetch(test_symbols)

            # 並列テスト
            tester.test_parallel_fetch(test_symbols, args.threads)

            # データベーステスト
            tester.test_database_bulk_insert(args.symbols)

        # レポート生成
        report = tester.generate_performance_report()
        optimal_config = tester.find_optimal_configuration()

        # 結果表示
        logger.info("=== テスト結果サマリー ===")
        logger.info(f"総テスト数: {report['test_summary']['total_tests']}")

        for operation, stats in report['performance_by_operation'].items():
            logger.info(f"\n{operation.upper()}:")
            logger.info(f"  平均スループット: {stats['throughput']['avg']:.1f} 件/秒")
            logger.info(f"  平均実行時間: {stats['duration']['avg']:.2f} 秒")
            logger.info(f"  平均成功率: {stats['success_rate']['avg']:.1%}")

        logger.info(f"\n=== 最適化推奨 ===")
        logger.info(f"推奨スレッド数: {optimal_config.get('optimal_thread_count', 'N/A')}")
        logger.info(f"推奨バッチサイズ: {optimal_config.get('optimal_batch_size', 'N/A')}")

        for recommendation in optimal_config.get('recommendations', []):
            logger.info(f"- {recommendation}")

        # レポートファイル出力
        if args.output_report:
            full_report = {
                'performance_report': report,
                'optimal_configuration': optimal_config
            }

            with open(args.output_report, 'w', encoding='utf-8') as f:
                json.dump(full_report, f, ensure_ascii=False, indent=2)

            logger.info(f"詳細レポートを出力: {args.output_report}")

        logger.info("=== パフォーマンステスト完了 ===")

        return 0

    except KeyboardInterrupt:
        logger.info("テストが中断されました")
        return 1

    except Exception as e:
        logger.error(f"予期しないエラー: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
