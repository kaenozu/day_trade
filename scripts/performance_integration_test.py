#!/usr/bin/env python3
"""
Issue #126: 統合パフォーマンステスト

最適化された一括取得・更新機能の統合テストを実行し、
既存実装との性能比較を行う。
"""

import sys
import time
from pathlib import Path

from src.day_trade.data.stock_master import StockMasterManager
from src.day_trade.utils.logging_config import setup_logging

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PerformanceIntegrationTest:
    """統合パフォーマンステストクラス"""

    def __init__(self):
        self.stock_manager = StockMasterManager()
        self.test_codes = ["7203", "6758", "9984", "9432", "8306"]

    def test_existing_vs_optimized(self):
        """既存実装 vs 最適化実装の性能比較"""
        print("=== 統合パフォーマンステスト: 既存 vs 最適化 ===\n")

        # 1. 既存の一括取得・更新
        print("1. 既存の一括企業情報取得・更新:")
        start_time = time.time()

        try:
            result_existing = self.stock_manager.bulk_fetch_and_update_companies(
                self.test_codes, batch_size=5, delay=0.2
            )

            elapsed_existing = time.time() - start_time

            print(
                f"   結果: 成功 {result_existing.get('updated', 0)}/{result_existing.get('total', 0)}件"
            )
            print(f"   実行時間: {elapsed_existing:.2f}秒")
            print(f"   平均: {elapsed_existing / len(self.test_codes):.2f}秒/件\n")

        except Exception as e:
            elapsed_existing = time.time() - start_time
            print(f"   エラー: {e}")
            print(f"   実行時間: {elapsed_existing:.2f}秒\n")
            result_existing = {
                "total": len(self.test_codes),
                "updated": 0,
                "failed": len(self.test_codes),
            }

        # 2. 最適化された一括価格取得・更新
        print("2. 最適化された一括価格取得・更新:")
        start_time = time.time()

        try:
            result_optimized = (
                self.stock_manager.bulk_fetch_and_update_prices_optimized(
                    self.test_codes, batch_size=5, delay=0.1
                )
            )

            elapsed_optimized = time.time() - start_time

            print(
                f"   結果: 成功 {result_optimized.get('updated', 0)}/{result_optimized.get('total', 0)}件"
            )
            print(f"   実行時間: {elapsed_optimized:.2f}秒")
            print(f"   平均: {elapsed_optimized / len(self.test_codes):.2f}秒/件\n")

        except Exception as e:
            elapsed_optimized = time.time() - start_time
            print(f"   エラー: {e}")
            print(f"   実行時間: {elapsed_optimized:.2f}秒\n")
            result_optimized = {
                "total": len(self.test_codes),
                "updated": 0,
                "failed": len(self.test_codes),
            }

        # 3. 比較結果表示
        print("=== 性能比較結果 ===")
        print(
            f"既存実装:     {elapsed_existing:.2f}秒 ({elapsed_existing / len(self.test_codes):.2f}秒/件)"
        )
        print(
            f"最適化実装:   {elapsed_optimized:.2f}秒 ({elapsed_optimized / len(self.test_codes):.2f}秒/件)"
        )

        if elapsed_existing > 0 and elapsed_optimized > 0:
            improvement = (
                (elapsed_existing - elapsed_optimized) / elapsed_existing
            ) * 100
            speedup = elapsed_existing / elapsed_optimized
            print(f"改善率:       {improvement:.1f}%")
            print(f"高速化倍率:   {speedup:.1f}x")

        print()
        return {
            "existing": {"time": elapsed_existing, "result": result_existing},
            "optimized": {"time": elapsed_optimized, "result": result_optimized},
        }

    def test_large_scale_performance(self):
        """大規模データでのパフォーマンステスト"""
        print("=== 大規模データパフォーマンステスト ===\n")

        # より大きなデータセットを取得
        large_test_codes = []
        try:
            stocks = self.stock_manager.search_stocks(limit=20)
            for stock in stocks:
                try:
                    large_test_codes.append(stock.code)
                except Exception:
                    # セッション問題回避
                    break

            if not large_test_codes:
                # フォールバック用のテストコード
                large_test_codes = [
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
                    "7974",
                    "4755",
                    "3382",
                    "6098",
                    "7733",
                    "4689",
                    "3659",
                    "2914",
                    "9613",
                    "4324",
                ][:15]  # 15件でテスト

        except Exception as e:
            print(f"テストデータ取得エラー: {e}")
            # デフォルトのテストコードを使用
            large_test_codes = [
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

        print(f"大規模テスト対象: {len(large_test_codes)}件")
        print(f"銘柄コード: {large_test_codes}")
        print()

        # 最適化された一括処理の大規模テスト
        print("最適化された大規模一括価格更新:")
        start_time = time.time()

        try:
            result = self.stock_manager.bulk_fetch_and_update_prices_optimized(
                large_test_codes,
                batch_size=10,  # 適度なバッチサイズ
                delay=0.05,  # 短い遅延
            )

            elapsed = time.time() - start_time

            print(
                f"   結果: 成功 {result.get('updated', 0)}/{result.get('total', 0)}件"
            )
            print(f"   実行時間: {elapsed:.2f}秒")
            print(f"   平均: {elapsed / len(large_test_codes):.2f}秒/件")
            print(f"   スループット: {len(large_test_codes) / elapsed:.1f}件/秒")

            return {
                "codes_count": len(large_test_codes),
                "time": elapsed,
                "result": result,
                "throughput": len(large_test_codes) / elapsed if elapsed > 0 else 0,
            }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   エラー: {e}")
            print(f"   実行時間: {elapsed:.2f}秒")

            return {
                "codes_count": len(large_test_codes),
                "time": elapsed,
                "result": {
                    "total": len(large_test_codes),
                    "updated": 0,
                    "failed": len(large_test_codes),
                },
                "throughput": 0,
            }

    def generate_summary_report(self, comparison_result, large_scale_result):
        """サマリーレポート生成"""
        print("\n" + "=" * 50)
        print("Issue #126 パフォーマンス改善レポート")
        print("=" * 50)

        print("\n【基本性能比較】")
        existing_time = comparison_result["existing"]["time"]
        optimized_time = comparison_result["optimized"]["time"]

        if existing_time > 0 and optimized_time > 0:
            improvement = ((existing_time - optimized_time) / existing_time) * 100
            speedup = existing_time / optimized_time

            print(f"- 既存実装:     {existing_time:.2f}秒")
            print(f"- 最適化実装:   {optimized_time:.2f}秒")
            print(f"- 改善率:       {improvement:.1f}%")
            print(f"- 高速化倍率:   {speedup:.1f}x")

        print("\n【大規模処理性能】")
        print("- 処理銘柄数:   {}件".format(large_scale_result["codes_count"]))
        print("- 実行時間:     {:.2f}秒".format(large_scale_result["time"]))
        print("- スループット: {:.1f}件/秒".format(large_scale_result["throughput"]))

        success_rate = 0
        if large_scale_result["result"].get("total", 0) > 0:
            success_rate = (
                large_scale_result["result"].get("updated", 0)
                / large_scale_result["result"]["total"]
            ) * 100
        print(f"- 成功率:       {success_rate:.1f}%")

        print("\n【実装済み最適化技術】")
        print("- yfinance.download()による真の一括取得")
        print("- マルチスレッド並列処理 (threads=True)")
        print("- 最適化されたバッチサイズとレート制限")
        print("- SQLAlchemyセッション管理改善")
        print("- 銘柄数制限機能との統合")

        print("\n【次の改善候補】")
        print("- SQLAlchemy bulk_update_mappings()の使用")
        print("- 非同期処理（asyncio）の導入")
        print("- データベース接続プールの最適化")
        print("- キャッシュ戦略の見直し")


def main():
    """メイン関数"""
    setup_logging()

    print("開始: Issue #126 統合パフォーマンステスト")
    print("-" * 50)

    test = PerformanceIntegrationTest()

    try:
        # 基本性能比較テスト
        comparison_result = test.test_existing_vs_optimized()

        # 大規模性能テスト
        large_scale_result = test.test_large_scale_performance()

        # サマリーレポート生成
        test.generate_summary_report(comparison_result, large_scale_result)

        print("\n完了: 統合パフォーマンステスト完了!")

    except Exception as e:
        print(f"\nエラー: 統合テスト中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
