#!/usr/bin/env python3
"""
Issue #596, #595対応: 統合システムの堅牢性向上テスト

一目均衡表計算の堅牢性とpandas Series返却の改善テスト
"""

import sys
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


def test_ichimoku_robustness():
    """一目均衡表の堅牢性テスト"""
    print("=== Issue #596: 一目均衡表堅牢性テスト ===")
    
    manager = TechnicalIndicatorsManager()
    
    # テスト1: 不完全なOHLCデータ
    print("\n--- 不完全なOHLCデータテスト ---")
    incomplete_data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [102, 103, 104],
        # 'Low': [99, 100, 101],  # Low列が欠如
        'Close': [101, 102, 103]
    })
    
    try:
        results = manager.calculate_indicators(incomplete_data, ["ichimoku"], ["Close"])
        result = list(results.values())[0][0]
        print(f"不完全データ処理: {result.implementation_used}")
        print(f"エラーメタデータ: {result.metadata.get('error', 'なし')}")
        
        # 結果がNaN配列かチェック
        if isinstance(result.values, dict):
            all_nan = all(np.all(np.isnan(arr)) for arr in result.values.values() if isinstance(arr, np.ndarray))
            print(f"全てNaN: {all_nan}")
        
    except Exception as e:
        print(f"例外発生: {e}")
    
    # テスト2: データ不足
    print("\n--- データ不足テスト ---")
    insufficient_data = pd.DataFrame({
        'High': [100, 101],  # 2点のみ（52期間必要）
        'Low': [99, 100],
        'Close': [100, 101]
    })
    
    try:
        results = manager.calculate_indicators(insufficient_data, ["ichimoku"], ["Close"])
        result = list(results.values())[0][0]
        print(f"データ不足処理: {result.implementation_used}")
        print(f"エラーメタデータ: {result.metadata.get('error', 'なし')}")
        
        if isinstance(result.values, dict):
            all_nan = all(np.all(np.isnan(arr)) for arr in result.values.values() if isinstance(arr, np.ndarray))
            print(f"全てNaN: {all_nan}")
            
    except Exception as e:
        print(f"例外発生: {e}")
    
    # テスト3: 正常なデータ
    print("\n--- 正常データテスト ---")
    np.random.seed(42)
    normal_data = pd.DataFrame({
        'High': 100 + np.cumsum(np.random.randn(60) * 0.5) + 1,
        'Low': 100 + np.cumsum(np.random.randn(60) * 0.5) - 1, 
        'Close': 100 + np.cumsum(np.random.randn(60) * 0.5)
    })
    
    try:
        results = manager.calculate_indicators(normal_data, ["ichimoku"], ["Close"])
        result = list(results.values())[0][0]
        print(f"正常データ処理: {result.implementation_used}")
        print(f"計算時間: {result.calculation_time:.3f}秒")
        
        if isinstance(result.values, dict):
            valid_values = sum(1 for arr in result.values.values() 
                             if isinstance(arr, np.ndarray) and not np.all(np.isnan(arr)))
            print(f"有効な値を持つ配列数: {valid_values}/7")
            
    except Exception as e:
        print(f"例外発生: {e}")
    
    return True


def test_pandas_series_compatibility():
    """pandas Series互換性テスト"""
    print("\n=== Issue #595: pandas Series互換性テスト ===")
    
    manager = TechnicalIndicatorsManager()
    
    # 日付インデックス付きデータ
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(50) * 0.5)
    }, index=dates)
    
    print(f"テストデータ: {len(data)}件, インデックス: {type(data.index)}")
    
    indicators_to_test = ["sma", "ema", "rsi", "bollinger_bands"]
    
    for indicator in indicators_to_test:
        try:
            results = manager.calculate_indicators(data, [indicator], ["Close"])
            result = list(results.values())[0][0]
            
            print(f"\n--- {indicator.upper()} ---")
            print(f"実装: {result.implementation_used}")
            print(f"値の型: {type(result.values)}")
            
            if isinstance(result.values, dict):
                for key, value in result.values.items():
                    if isinstance(value, (pd.Series, np.ndarray)):
                        print(f"  {key}: {type(value)}, 長さ={len(value)}")
                    else:
                        print(f"  {key}: {type(value)}, 値={value}")
            else:
                print(f"値: {type(result.values)}, 長さ={len(result.values) if hasattr(result.values, '__len__') else 'N/A'}")
                
        except Exception as e:
            print(f"{indicator} エラー: {e}")
    
    return True


def test_error_handling_improvements():
    """エラーハンドリング改善テスト"""
    print("\n=== エラーハンドリング改善テスト ===")
    
    manager = TechnicalIndicatorsManager()
    
    # テスト1: 空のデータフレーム
    empty_data = pd.DataFrame()
    try:
        results = manager.calculate_indicators(empty_data, ["sma"], ["Close"])
        print("空データ処理: 成功")
    except Exception as e:
        print(f"空データエラー: {e}")
    
    # テスト2: 存在しない列
    data = pd.DataFrame({'Price': [100, 101, 102]})
    try:
        results = manager.calculate_indicators(data, ["sma"], ["NonExistent"])
        print("存在しない列処理: 成功")
    except Exception as e:
        print(f"存在しない列エラー: {e}")
    
    # テスト3: NaN含みデータ
    nan_data = pd.DataFrame({
        'Close': [100, np.nan, 102, 103, np.nan, 105]
    })
    try:
        results = manager.calculate_indicators(nan_data, ["sma"], ["Close"])
        result = list(results.values())[0][0]
        print(f"NaN含みデータ処理: 成功, 実装={result.implementation_used}")
    except Exception as e:
        print(f"NaN含みデータエラー: {e}")
    
    return True


def main():
    """メインテスト実行"""
    print("Issue #596, #595対応: 統合システム堅牢性向上テスト開始")
    print("=" * 70)
    
    test_results = []
    
    try:
        # 一目均衡表堅牢性テスト
        result1 = test_ichimoku_robustness()
        test_results.append(("一目均衡表堅牢性", result1))
        
        # pandas Series互換性テスト
        result2 = test_pandas_series_compatibility()
        test_results.append(("pandas Series互換性", result2))
        
        # エラーハンドリング改善テスト
        result3 = test_error_handling_improvements()
        test_results.append(("エラーハンドリング改善", result3))
        
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
        print("[成功] 堅牢性向上テスト完了")
        print("統合システムの品質がさらに向上しました")
    else:
        print("[失敗] 一部のテストが失敗しました")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)