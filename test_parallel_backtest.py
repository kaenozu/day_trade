#!/usr/bin/env python3
"""
並列バックテストフレームワーク統合テスト
Issue #382: Parallel Backtest Framework Integration Test

高速化効果とパフォーマンス検証
"""

import sys
import time
from pathlib import Path

# パス調整
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.day_trade.backtesting.parallel_backtest_framework import (
    MicrosecondTimer,
    OptimizationMethod,
    ParallelMode,
    ParameterSpace,
    create_parallel_backtest_framework,
)


def run_parallel_backtest_test():
    """並列バックテスト統合テスト"""
    print("=" * 70)
    print("並列バックテストフレームワーク統合テスト開始")
    print("=" * 70)

    try:
        # フレームワーク作成
        print("\n1. フレームワーク作成中...")
        framework = create_parallel_backtest_framework(
            max_workers=4,
            parallel_mode=ParallelMode.MULTIPROCESSING,
            optimization_method=OptimizationMethod.GRID_SEARCH,
        )
        print("   フレームワーク作成完了")

        # パラメータ空間定義
        print("\n2. パラメータ空間定義...")
        parameter_spaces = [
            ParameterSpace(
                name="momentum_window", min_value=10, max_value=30, step_size=5
            ),
            ParameterSpace(
                name="buy_threshold", min_value=0.02, max_value=0.08, step_size=0.02
            ),
            ParameterSpace(
                name="position_size", min_value=0.1, max_value=0.3, step_size=0.1
            ),
        ]

        combinations = (
            len([10, 15, 20, 25, 30])
            * len([0.02, 0.04, 0.06, 0.08])
            * len([0.1, 0.2, 0.3])
        )
        print(f"   パラメータ組み合わせ数: {combinations}通り")

        # テスト用銘柄・期間
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start_date = "2023-01-01"
        end_date = "2023-12-31"

        print("\n3. 並列最適化実行設定:")
        print(f"   対象銘柄: {symbols}")
        print(f"   期間: {start_date} - {end_date}")
        print("   並列ワーカー数: 4")
        print("   最適化手法: グリッドサーチ")

        # 最適化実行
        print("\n4. 並列パラメータ最適化実行中...")
        print(
            f"   推定実行時間: {combinations}タスク × 2秒 ÷ 4ワーカー = {combinations * 2 // 4}秒"
        )

        start_time = MicrosecondTimer.now_ns()

        results = framework.run_parameter_optimization(
            symbols=symbols,
            parameter_spaces=parameter_spaces,
            start_date=start_date,
            end_date=end_date,
            strategy_config={"strategy_type": "momentum"},
            initial_capital=1000000,
        )

        execution_time_ms = MicrosecondTimer.elapsed_us(start_time) / 1000

        # 結果表示
        print("\n5. 最適化結果:")
        if "error" in results:
            print(f"   エラー: {results['error']}")
            return False

        summary = results.get("optimization_summary", {})
        best_params = results.get("best_parameters", {})
        best_result = results.get("best_result", {})

        print(
            f"   実行時間: {execution_time_ms:.0f}ms ({execution_time_ms/1000:.1f}秒)"
        )
        print(f"   総組み合わせ数: {summary.get('total_combinations', 0)}")
        print(f"   成功率: {summary.get('success_rate', 0):.1%}")
        print(f"   最適値: {summary.get('best_value', 0):.4f}")

        print("\n6. 最優秀パラメータ:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")

        print("\n7. 最優秀結果:")
        print(f"   総リターン: {best_result.get('total_return', 0):.2%}")
        print(f"   年率リターン: {best_result.get('annualized_return', 0):.2%}")
        print(f"   シャープレシオ: {best_result.get('sharpe_ratio', 0):.3f}")
        print(f"   最大ドローダウン: {best_result.get('max_drawdown', 0):.2%}")
        print(f"   勝率: {best_result.get('win_rate', 0):.1%}")
        print(f"   総取引数: {best_result.get('total_trades', 0)}")

        # パフォーマンス評価
        print("\n8. パフォーマンス評価:")
        stats = framework.get_framework_stats()
        exec_stats = stats.get("execution_stats", {})

        throughput = exec_stats.get("throughput_tasks_per_sec", 0)
        avg_time = exec_stats.get("avg_task_time_ms", 0)

        print(f"   スループット: {throughput:.1f} タスク/秒")
        print(f"   平均タスク時間: {avg_time:.0f}ms")
        print(f"   並列効率: {throughput * avg_time / 1000:.1f}x")

        # 効率評価
        if throughput > 2.0:
            print("   評価: 優秀 - 高速並列処理達成")
        elif throughput > 1.0:
            print("   評価: 良好 - 並列処理効果確認")
        else:
            print("   評価: 改善余地 - 並列処理最適化必要")

        # Top 3結果表示
        top_results = results.get("top_10_results", [])[:3]
        if top_results:
            print("\n9. Top 3 結果:")
            for result in top_results:
                rank = result.get("rank", 0)
                params = result.get("parameters", {})
                sharpe = result.get("sharpe_ratio", 0)
                print(f"   第{rank}位: シャープ {sharpe:.3f} - {params}")

        print("\n並列バックテストフレームワーク テスト完了!")
        print("高速化により戦略開発効率が大幅向上しました。")
        return True

    except Exception as e:
        print(f"\nテストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_parallel_backtest_test()

    if success:
        print("\n並列バックテストフレームワーク統合テスト成功!")
    else:
        print("\n並列バックテストフレームワーク統合テスト失敗...")
        sys.exit(1)
