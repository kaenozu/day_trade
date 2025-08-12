#!/usr/bin/env python3
"""
Issue #126: パフォーマンスベンチマーク測定スクリプト

現在の銘柄一括取得・更新処理のパフォーマンスを測定し、改善前後の比較を行う。
"""

import sys
import time
from pathlib import Path
from typing import List

from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.data.stock_master import StockMasterManager
from src.day_trade.utils.logging_config import setup_logging

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PerformanceBenchmark:
    """パフォーマンスベンチマーククラス"""

    def __init__(self):
        self.stock_manager = StockMasterManager()
        self.stock_fetcher = StockFetcher()
        self.results = {}

    def measure_time(self, func_name: str, func, *args, **kwargs):
        """関数の実行時間を測定"""
        print(f"測定開始: {func_name}")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time

            self.results[func_name] = {
                "execution_time": execution_time,
                "success": True,
                "result_count": len(result) if hasattr(result, "__len__") else "N/A",
                "error": None,
            }

            print(
                f"OK {func_name}: {execution_time:.2f}秒 ({self.results[func_name]['result_count']}件)"
            )
            return result

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time

            self.results[func_name] = {
                "execution_time": execution_time,
                "success": False,
                "result_count": 0,
                "error": str(e),
            }

            print(f"NG {func_name}: {execution_time:.2f}秒 (エラー: {e})")
            return None

    def get_test_codes(self, limit: int = 50) -> List[str]:
        """テスト用の銘柄コード一覧を取得"""
        stocks = self.stock_manager.search_stocks(limit=limit)
        codes = []
        for stock in stocks:
            try:
                codes.append(stock.code)
            except Exception:
                # セッション問題を回避するため、ハードコードされたテスト用コードを使用
                codes = [
                    "7203",
                    "6758",
                    "9984",
                    "9432",
                    "8306",
                    "6861",
                    "6367",
                    "4063",
                    "8031",
                    "6981",
                ]
                break

        print(f"テスト用銘柄コード: {len(codes)}件 ({codes[:5]}...)")
        return codes

    def benchmark_current_implementation(self):
        """現在の実装のベンチマーク"""
        print("\n=== 現在の実装のパフォーマンスベンチマーク ===")

        # 1. 銘柄検索のベンチマーク
        print("\n1. 銘柄検索パフォーマンス:")
        self.measure_time("銘柄検索(100件)", self.stock_manager.search_stocks, limit=100)

        self.measure_time("銘柄検索(500件)", self.stock_manager.search_stocks, limit=500)

        # 2. セクター検索のベンチマーク
        print("\n2. セクター検索パフォーマンス:")
        sectors = self.stock_manager.get_all_sectors()
        if sectors:
            test_sector = sectors[0]
            self.measure_time(
                f"セクター検索('{test_sector}')",
                self.stock_manager.search_stocks_by_sector,
                test_sector,
                limit=100,
            )

        # 3. 現在価格取得のベンチマーク
        print("\n3. 現在価格取得パフォーマンス:")
        test_codes = self.get_test_codes(10)

        # 個別取得 vs バルク取得
        if test_codes:
            # 個別取得（従来方式）
            def get_prices_individually(codes):
                results = {}
                for code in codes:
                    try:
                        price_data = self.stock_fetcher.get_current_price(code)
                        results[code] = price_data
                    except Exception:
                        results[code] = None
                return results

            self.measure_time(
                f"個別価格取得({len(test_codes)}件)",
                get_prices_individually,
                test_codes,
            )

            # バルク取得（既存版）
            self.measure_time(
                f"バルク価格取得({len(test_codes)}件)",
                self.stock_fetcher.bulk_get_current_prices,
                test_codes,
                batch_size=len(test_codes),
            )

            # バルク取得（最適化版）
            self.measure_time(
                f"最適化バルク価格取得({len(test_codes)}件)",
                self.stock_fetcher.bulk_get_current_prices_optimized,
                test_codes,
                batch_size=len(test_codes),
            )

        # 4. 大量データ処理のベンチマーク
        print("\n4. 大量データ処理パフォーマンス:")
        large_test_codes = self.get_test_codes(50)

        if large_test_codes:
            # 大量バルク価格取得（既存版）
            self.measure_time(
                f"大量バルク価格取得({len(large_test_codes)}件)",
                self.stock_fetcher.bulk_get_current_prices,
                large_test_codes,
                batch_size=20,  # バッチサイズ20で処理
                delay=0.1,
            )

            # 大量バルク価格取得（最適化版）
            self.measure_time(
                f"大量最適化バルク価格取得({len(large_test_codes)}件)",
                self.stock_fetcher.bulk_get_current_prices_optimized,
                large_test_codes,
                batch_size=25,  # バッチサイズ25で処理
                delay=0.05,
            )

            # 企業情報バルク取得
            self.measure_time(
                f"企業情報バルク取得({len(large_test_codes)}件)",
                self.stock_fetcher.bulk_get_company_info,
                large_test_codes[:10],  # 企業情報は時間がかかるので10件に制限
                batch_size=5,
                delay=0.2,
            )

    def benchmark_database_operations(self):
        """データベース操作のベンチマーク"""
        print("\n=== データベース操作ベンチマーク ===")

        # 1. 大量検索
        print("\n1. 大量検索パフォーマンス:")
        self.measure_time("大量銘柄検索(1000件)", self.stock_manager.search_stocks, limit=1000)

        # 2. 業種検索
        print("\n2. 業種検索パフォーマンス:")
        industries = self.stock_manager.get_all_industries()
        if industries:
            test_industry = industries[0]
            self.measure_time(
                f"業種検索('{test_industry}')",
                self.stock_manager.search_stocks_by_industry,
                test_industry,
                limit=200,
            )

        # 3. 名前検索
        print("\n3. 名前検索パフォーマンス:")
        self.measure_time(
            "名前検索('株式')",
            self.stock_manager.search_stocks_by_name,
            "株式",
            limit=100,
        )

    def print_summary(self):
        """ベンチマーク結果のサマリを表示"""
        print("\n=== ベンチマーク結果サマリ ===")
        print("-" * 60)
        print(f"{'測定項目':<30} {'実行時間':<10} {'件数':<8} {'結果':<6}")
        print("-" * 60)

        for name, result in self.results.items():
            status = "OK" if result["success"] else "NG"
            print(
                f"{name:<30} {result['execution_time']:<10.2f}秒 {str(result['result_count']):<8} {status:<6}"
            )

        print("-" * 60)

        # 成功したテストのみの統計
        successful_tests = [r for r in self.results.values() if r["success"]]
        if successful_tests:
            total_time = sum(r["execution_time"] for r in successful_tests)
            avg_time = total_time / len(successful_tests)

            print(f"総実行時間: {total_time:.2f}秒")
            print(f"平均実行時間: {avg_time:.2f}秒")
            print(f"成功テスト: {len(successful_tests)}/{len(self.results)}")

        # エラーがあった場合は詳細表示
        failed_tests = [
            (name, result) for name, result in self.results.items() if not result["success"]
        ]
        if failed_tests:
            print("\n警告: エラー詳細:")
            for name, result in failed_tests:
                print(f"  {name}: {result['error']}")

    def save_results(self, filename: str = "benchmark_results.json"):
        """結果をJSONファイルに保存"""
        import json
        from datetime import datetime

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "current_implementation",
            "results": self.results,
        }

        output_path = project_root / "benchmarks" / filename
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nデータ: ベンチマーク結果を保存しました: {output_path}")


def main():
    """メイン関数"""
    setup_logging()

    print("開始: Issue #126 パフォーマンスベンチマーク測定")
    print("=" * 50)

    benchmark = PerformanceBenchmark()

    try:
        # 現在の実装ベンチマーク
        benchmark.benchmark_current_implementation()

        # データベース操作ベンチマーク
        benchmark.benchmark_database_operations()

        # 結果サマリ表示
        benchmark.print_summary()

        # 結果保存
        benchmark.save_results("benchmark_current.json")

        print("\n完了: ベンチマーク測定完了!")

    except Exception as e:
        print(f"\nエラー: ベンチマーク測定中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
