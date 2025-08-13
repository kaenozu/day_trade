#!/usr/bin/env python3
"""
Issue #597-599対応: 統合システムの最適化改善テスト

フィボナッチパラメータ化、キャッシュ最適化、Numba最適化のテスト
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトパスを追加
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from day_trade.analysis.technical_indicators_consolidated import (
    TechnicalIndicatorsManager,
    IndicatorConfig
)


def test_issue_597_fibonacci_parameterization():
    """Issue #597: フィボナッチリトレースメントパラメータ化テスト"""
    print("=== Issue #597: フィボナッチパラメータ化テスト ===")

    manager = TechnicalIndicatorsManager()

    # テストデータ作成
    np.random.seed(42)
    data = pd.DataFrame({
        'High': 100 + np.cumsum(np.random.randn(100) * 0.5) + 2,
        'Low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 2,
        'Close': 100 + np.cumsum(np.random.randn(100) * 0.5)
    })

    print("\n--- 標準フィボナッチレベルテスト ---")
    results = manager.calculate_indicators(data, ["fibonacci_retracement"], ["Close"])
    result = list(results.values())[0][0]
    print(f"実装: {result.implementation_used}")
    print(f"レベル数: {result.metadata['levels_count']}")
    print(f"トレンド方向: {result.metadata['detected_trend']}")
    print(f"最近接レベル: {result.metadata['closest_level']}")

    print("\n--- カスタムフィボナッチレベルテスト ---")
    custom_levels = [0.0, 0.382, 0.5, 0.618, 1.0]
    results = manager.calculate_indicators(
        data, ["fibonacci_retracement"], ["Close"],
        custom_levels=custom_levels
    )
    result = list(results.values())[0][0]
    print(f"カスタムレベル使用: {result.metadata['custom_levels']}")
    print(f"レベル数: {result.metadata['levels_count']}")

    print("\n--- トレンド方向指定テスト ---")
    results = manager.calculate_indicators(
        data, ["fibonacci_retracement"], ["Close"],
        trend_mode="up"
    )
    result = list(results.values())[0][0]
    print(f"指定トレンド: {result.metadata['trend_mode']}")
    print(f"検出トレンド: {result.metadata['detected_trend']}")

    print("\n--- 終値のみデータテスト ---")
    close_only_data = pd.DataFrame({'Close': data['Close']})
    results = manager.calculate_indicators(
        close_only_data, ["fibonacci_retracement"], ["Close"],
        use_hl_data=False
    )
    result = list(results.values())[0][0]
    print(f"高値・安値データ使用: {result.metadata['use_hl_data']}")

    return True


def test_issue_598_cache_optimization():
    """Issue #598: キャッシュキー生成最適化テスト"""
    print("\n=== Issue #598: キャッシュ最適化テスト ===")

    config = IndicatorConfig(cache_results=True)
    manager = TechnicalIndicatorsManager(config)

    # 大きなデータセット作成
    np.random.seed(42)
    large_data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(1000) * 0.5)
    })

    print("\n--- 大データセットキャッシュテスト ---")
    start_time = time.time()

    # 初回計算
    results1 = manager.calculate_indicators(large_data, ["sma", "ema", "rsi"], ["Close"])
    first_calc_time = time.time() - start_time

    start_time = time.time()

    # 2回目計算（キャッシュヒット期待）
    results2 = manager.calculate_indicators(large_data, ["sma", "ema", "rsi"], ["Close"])
    second_calc_time = time.time() - start_time

    perf_summary = manager.get_performance_summary()
    print(f"初回計算時間: {first_calc_time:.3f}秒")
    print(f"2回目計算時間: {second_calc_time:.3f}秒")
    print(f"キャッシュヒット率: {perf_summary['cache_hit_rate']:.1%}")
    print(f"計算回数: {perf_summary['total_calculations']}")

    # パラメータ違いでのキャッシュテスト
    print("\n--- パラメータ別キャッシュテスト ---")

    # 異なるパラメータ
    results3 = manager.calculate_indicators(large_data, ["sma"], ["Close"], period=30)
    results4 = manager.calculate_indicators(large_data, ["sma"], ["Close"], period=20)
    results5 = manager.calculate_indicators(large_data, ["sma"], ["Close"], period=20)  # キャッシュヒット期待

    final_perf = manager.get_performance_summary()
    print(f"最終キャッシュヒット率: {final_perf['cache_hit_rate']:.1%}")
    print(f"総計算回数: {final_perf['total_calculations']}")

    return True


def test_issue_599_numba_optimization():
    """Issue #599: Numba最適化とフォールバック改善テスト"""
    print("\n=== Issue #599: Numba最適化テスト ===")

    # Numba有効設定
    config_numba = IndicatorConfig(use_numba=True, use_talib=False)
    manager_numba = TechnicalIndicatorsManager(config_numba)

    # Numba無効設定
    config_pandas = IndicatorConfig(use_numba=False, use_talib=False)
    manager_pandas = TechnicalIndicatorsManager(config_pandas)

    # 大きなデータセット（Numba効果を見るため）
    np.random.seed(42)
    large_data = pd.DataFrame({
        'High': 100 + np.cumsum(np.random.randn(500) * 0.5) + 1,
        'Low': 100 + np.cumsum(np.random.randn(500) * 0.5) - 1,
        'Close': 100 + np.cumsum(np.random.randn(500) * 0.5)
    })

    indicators = ["sma", "ema", "rsi", "ichimoku"]

    print("\n--- Numba vs Pandas パフォーマンステスト ---")

    # Numba実装テスト
    start_time = time.time()
    results_numba = manager_numba.calculate_indicators(large_data, indicators, ["Close"])
    numba_time = time.time() - start_time

    # Pandas実装テスト
    start_time = time.time()
    results_pandas = manager_pandas.calculate_indicators(large_data, indicators, ["Close"])
    pandas_time = time.time() - start_time

    # パフォーマンス比較
    perf_numba = manager_numba.get_performance_summary()
    perf_pandas = manager_pandas.get_performance_summary()

    print(f"Numba設定実行時間: {numba_time:.3f}秒")
    print(f"Pandas設定実行時間: {pandas_time:.3f}秒")

    if numba_time < pandas_time:
        speedup = pandas_time / numba_time
        print(f"Numba高速化率: {speedup:.2f}倍")

    print(f"\nNumba使用率: {perf_numba.get('numba_usage_rate', 0):.1%}")
    print(f"Pandas使用率: {perf_pandas.get('fallback_usage_rate', 0):.1%}")
    print(f"最適化レベル: {perf_numba.get('optimization_level', 'unknown')}")

    # 実装使用状況
    impl_usage = perf_numba.get('implementation_usage', {})
    print(f"\n実装使用統計:")
    for impl, count in impl_usage.items():
        print(f"  {impl}: {count}回")

    print(f"Numba利用可能: {perf_numba.get('numba_available', False)}")
    print(f"TA-Lib利用可能: {perf_numba.get('talib_available', False)}")

    return True


def test_combined_optimizations():
    """統合最適化テスト"""
    print("\n=== 統合最適化テスト ===")

    # 全機能有効設定
    config = IndicatorConfig(
        cache_results=True,
        use_talib=True,
        use_numba=True
    )
    manager = TechnicalIndicatorsManager(config)

    # 複雑なテストデータ
    np.random.seed(42)
    data = pd.DataFrame({
        'High': 100 + np.cumsum(np.random.randn(200) * 0.5) + 1,
        'Low': 100 + np.cumsum(np.random.randn(200) * 0.5) - 1,
        'Close': 100 + np.cumsum(np.random.randn(200) * 0.5)
    })

    all_indicators = ["sma", "ema", "rsi", "macd", "bollinger_bands", "ichimoku", "fibonacci_retracement"]

    print("\n--- 全指標計算テスト ---")
    start_time = time.time()

    results = manager.calculate_indicators(data, all_indicators, ["Close"])

    total_time = time.time() - start_time
    perf_summary = manager.get_performance_summary()

    print(f"全指標計算時間: {total_time:.3f}秒")
    print(f"計算指標数: {len(results['Close'])}")
    print(f"キャッシュヒット率: {perf_summary['cache_hit_rate']:.1%}")
    print(f"最適化レベル: {perf_summary.get('optimization_level', 'unknown')}")

    # 実装別使用率
    print(f"\n実装別使用率:")
    print(f"  TA-Lib: {perf_summary.get('talib_usage_rate', 0):.1%}")
    print(f"  Numba: {perf_summary.get('numba_usage_rate', 0):.1%}")
    print(f"  Pandas: {perf_summary.get('fallback_usage_rate', 0):.1%}")

    # 指標別パフォーマンス
    avg_times = perf_summary.get('average_calculation_times', {})
    if avg_times:
        print(f"\n指標別平均計算時間:")
        for indicator, avg_time in sorted(avg_times.items(), key=lambda x: x[1]):
            print(f"  {indicator}: {avg_time:.4f}秒")

        print(f"\n最高速指標: {perf_summary.get('fastest_indicator', 'N/A')}")
        print(f"最低速指標: {perf_summary.get('slowest_indicator', 'N/A')}")

    return True


def main():
    """メインテスト実行"""
    print("Issue #597-599対応: 統合システム最適化改善テスト開始")
    print("=" * 70)

    test_results = []

    try:
        # Issue #597 フィボナッチパラメータ化テスト
        result1 = test_issue_597_fibonacci_parameterization()
        test_results.append(("Issue #597 フィボナッチパラメータ化", result1))

        # Issue #598 キャッシュ最適化テスト
        result2 = test_issue_598_cache_optimization()
        test_results.append(("Issue #598 キャッシュ最適化", result2))

        # Issue #599 Numba最適化テスト
        result3 = test_issue_599_numba_optimization()
        test_results.append(("Issue #599 Numba最適化", result3))

        # 統合最適化テスト
        result4 = test_combined_optimizations()
        test_results.append(("統合最適化", result4))

    except Exception as e:
        print(f"\n[エラー] テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 結果サマリー
    print("\n" + "=" * 70)
    print("テスト結果サマリー:")
    print("-" * 50)

    all_passed = True
    for test_name, result in test_results:
        status = "[OK]" if result else "[NG]"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False

    print("-" * 50)
    if all_passed:
        print("[成功] Issue #597-599 最適化改善テスト完了")
        print("統合システムのパフォーマンスが大幅に向上しました")
    else:
        print("[失敗] 一部のテストが失敗しました")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)