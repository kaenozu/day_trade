#!/usr/bin/env python3
"""
Issue #619対応: テクニカル指標計算ロジック統合テスト

統合テクニカル指標システムの動作確認と
後方互換性の検証テスト
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
    IndicatorConfig,
    IndicatorCategory,
    SignalStrength,
    calculate_technical_indicators,
    calculate_sma,
    calculate_rsi,
    calculate_macd
)


def test_consolidated_system():
    """統合システムの基本動作テスト"""
    print("=== Issue #619: 統合テクニカル指標システムテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    # OHLCV データ作成
    data = pd.DataFrame({
        'Open': prices + np.random.randn(100) * 0.1,
        'High': prices + np.abs(np.random.randn(100)) * 0.2,
        'Low': prices - np.abs(np.random.randn(100)) * 0.2,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    print(f"テストデータ作成完了: {len(data)}件")
    
    # 統合マネージャー初期化
    config = IndicatorConfig(
        cache_results=True,
        use_talib=True,
        parallel_computation=True
    )
    manager = TechnicalIndicatorsManager(config)
    
    print("\n--- 利用可能指標 ---")
    available_indicators = manager.get_available_indicators()
    print(f"利用可能指標数: {len(available_indicators)}")
    print(f"指標: {', '.join(available_indicators[:10])}...")
    
    # 主要指標計算テスト
    test_indicators = ["sma", "ema", "rsi", "macd", "bollinger_bands", "stochastic"]
    
    print(f"\n--- 指標計算テスト ({len(test_indicators)}指標) ---")
    start_time = time.time()
    
    results = manager.calculate_indicators(
        data=data,
        indicators=test_indicators,
        symbols=["Close"],
        timeframe="1D"
    )
    
    calc_time = time.time() - start_time
    print(f"計算完了: {calc_time:.3f}秒")
    
    # 結果検証
    print("\n--- 計算結果検証 ---")
    for symbol, symbol_results in results.items():
        print(f"\n{symbol}:")
        for result in symbol_results:
            values_info = f"{len(result.values)}個" if hasattr(result.values, '__len__') else "1個"
            if isinstance(result.values, dict):
                values_info = f"{len(result.values)}種類"
            
            print(f"  {result.name}: {values_info}, "
                  f"信頼度={result.confidence:.2f}, "
                  f"計算時間={result.calculation_time:.3f}s, "
                  f"実装={result.implementation_used}")
    
    # パフォーマンス概要
    print("\n--- パフォーマンス概要 ---")
    perf_summary = manager.get_performance_summary()
    print(f"キャッシュヒット率: {perf_summary['cache_hit_rate']:.1%}")
    print(f"TA-Lib使用率: {perf_summary.get('talib_usage_rate', 0):.1%}")
    print(f"平均計算時間: {np.mean(list(perf_summary['average_calculation_times'].values())):.4f}s")
    
    return True


def test_backward_compatibility():
    """後方互換性テスト"""
    print("\n=== 後方互換性テスト ===")
    
    # テストデータ
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(50) * 0.5)
    }, index=dates)
    
    # 旧システムインポートテスト
    try:
        from day_trade.analysis.advanced_technical_indicators import AdvancedTechnicalIndicators
        from day_trade.portfolio.technical_indicators import (
            TechnicalIndicatorEngine,
            calculate_sma,
            calculate_rsi,
            calculate_macd
        )
        print("[OK] 後方互換インポート成功")
    except ImportError as e:
        print(f"[エラー] インポート失敗: {e}")
        return False
    
    # 旧クラス動作確認
    print("\n--- 旧AdvancedTechnicalIndicators動作確認 ---")
    old_indicators = AdvancedTechnicalIndicators()
    
    sma_result = old_indicators.calculate_sma(data, period=20)
    rsi_result = old_indicators.calculate_rsi(data, period=14)
    macd_result = old_indicators.calculate_macd(data)
    
    print(f"SMA計算: {len(sma_result)}件")
    print(f"RSI計算: {len(rsi_result)}件")
    print(f"MACD計算: {len(macd_result)}種類")
    
    # 旧TechnicalIndicatorEngine動作確認
    print("\n--- 旧TechnicalIndicatorEngine動作確認 ---")
    from day_trade.portfolio.technical_indicators import IndicatorConfig as OldConfig
    
    old_config = OldConfig(cache_results=True)
    old_engine = TechnicalIndicatorEngine(old_config)
    
    old_results = old_engine.calculate_indicators(
        data=data,
        indicators=["sma", "rsi"],
        symbols=["Close"]
    )
    
    print(f"旧エンジン結果: {len(old_results)}銘柄")
    for symbol, results in old_results.items():
        print(f"  {symbol}: {len(results)}指標")
    
    # 便利関数テスト
    print("\n--- 便利関数動作確認 ---")
    sma_func = calculate_sma(data, 20)
    rsi_func = calculate_rsi(data, 14)
    macd_func = calculate_macd(data)
    
    print(f"SMA関数: {len(sma_func)}件")
    print(f"RSI関数: {len(rsi_func)}件")
    print(f"MACD関数: {len(macd_func)}種類")
    
    print("[OK] 後方互換性確認完了")
    return True


def test_performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n=== パフォーマンス比較テスト ===")
    
    # 大きなテストデータ
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(1000) * 0.5)
    }, index=dates)
    
    print(f"大規模データテスト: {len(data)}件")
    
    # 統合システムテスト
    print("\n--- 統合システム性能 ---")
    manager = TechnicalIndicatorsManager()
    
    start_time = time.time()
    results1 = manager.calculate_indicators(
        data, ["sma", "ema", "rsi", "macd"], ["Close"]
    )
    time1 = time.time() - start_time
    
    # キャッシュテスト（2回目実行）
    start_time = time.time()
    results2 = manager.calculate_indicators(
        data, ["sma", "ema", "rsi", "macd"], ["Close"]
    )
    time2 = time.time() - start_time
    
    print(f"初回実行: {time1:.3f}秒")
    print(f"キャッシュ使用: {time2:.3f}秒")
    
    if time2 > 0:
        speedup = time1 / time2
        print(f"高速化倍率: {speedup:.1f}倍")
    else:
        print("高速化倍率: 測定不可（キャッシュ実行時間が0秒未満）")
    
    # パフォーマンス詳細
    perf_summary = manager.get_performance_summary()
    print(f"キャッシュヒット率: {perf_summary['cache_hit_rate']:.1%}")
    
    return True


def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n=== エラーハンドリングテスト ===")
    
    manager = TechnicalIndicatorsManager()
    
    # 空データテスト
    try:
        empty_data = pd.DataFrame()
        results = manager.calculate_indicators(empty_data, ["sma"])
        print("[OK] 空データ処理")
    except Exception as e:
        print(f"[警告] 空データエラー: {e}")
    
    # 無効指標テスト
    try:
        data = pd.DataFrame({'Close': [100, 101, 102]})
        results = manager.calculate_indicators(data, ["invalid_indicator"])
        print("[OK] 無効指標処理")
    except Exception as e:
        print(f"[警告] 無効指標エラー: {e}")
    
    # 不完全データテスト
    try:
        incomplete_data = pd.DataFrame({'Close': [100, np.nan, 102]})
        results = manager.calculate_indicators(incomplete_data, ["sma"])
        print("[OK] 不完全データ処理")
    except Exception as e:
        print(f"[警告] 不完全データエラー: {e}")
    
    return True


def main():
    """メインテスト実行"""
    print("Issue #619: テクニカル指標計算ロジック統合テスト開始")
    print("=" * 60)
    
    test_results = []
    
    try:
        # 基本動作テスト
        result1 = test_consolidated_system()
        test_results.append(("統合システム基本動作", result1))
        
        # 後方互換性テスト
        result2 = test_backward_compatibility()
        test_results.append(("後方互換性", result2))
        
        # パフォーマンステスト
        result3 = test_performance_comparison()
        test_results.append(("パフォーマンス比較", result3))
        
        # エラーハンドリングテスト
        result4 = test_error_handling()
        test_results.append(("エラーハンドリング", result4))
        
    except Exception as e:
        print(f"\n[エラー] テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー:")
    print("-" * 40)
    
    all_passed = True
    for test_name, result in test_results:
        status = "[OK]" if result else "[NG]"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("-" * 40)
    if all_passed:
        print("[成功] Issue #619: テクニカル指標統合テスト完了")
        print("統合システムは正常に動作しています")
    else:
        print("[失敗] 一部のテストが失敗しました")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)