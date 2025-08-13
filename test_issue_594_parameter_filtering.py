#!/usr/bin/env python3
"""
Issue #594対応: パラメータフィルタリング簡略化テスト
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

def test_issue_594_parameter_filtering():
    """Issue #594: パラメータフィルタリング簡略化テスト"""
    print("=== Issue #594: パラメータフィルタリング簡略化テスト ===")
    
    manager = TechnicalIndicatorsManager()
    
    # テストデータ作成
    np.random.seed(42)
    data = pd.DataFrame({
        'High': 100 + np.cumsum(np.random.randn(60) * 0.5) + 1,
        'Low': 100 + np.cumsum(np.random.randn(60) * 0.5) - 1,
        'Close': 100 + np.cumsum(np.random.randn(60) * 0.5)
    })
    
    print("\n--- デフォルトパラメータテスト ---")
    # デフォルトパラメータで計算
    results = manager.calculate_indicators(data, ["sma", "ema", "rsi"], ["Close"])
    for symbol_results in results.values():
        for result in symbol_results:
            print(f"指標: {result.name}, メタデータ: {result.metadata}")
    
    print("\n--- カスタムパラメータテスト ---")
    # カスタムパラメータで計算
    results = manager.calculate_indicators(
        data, ["sma", "ema", "rsi"], ["Close"],
        period=30  # 全指標に適用される共通パラメータ
    )
    for symbol_results in results.values():
        for result in symbol_results:
            print(f"指標: {result.name}, メタデータ: {result.metadata}")
    
    print("\n--- MACD特殊パラメータテスト ---")
    # MACD特殊パラメータ
    results = manager.calculate_indicators(
        data, ["macd"], ["Close"],
        fast_period=8, slow_period=21, signal_period=5
    )
    for symbol_results in results.values():
        for result in symbol_results:
            print(f"指標: {result.name}, メタデータ: {result.metadata}")
    
    print("\n--- 無効パラメータ除外テスト ---")
    # 無効なパラメータが除外されることを確認
    results = manager.calculate_indicators(
        data, ["sma"], ["Close"],
        period=25,
        invalid_param="should_be_ignored",  # 無効パラメータ
        another_invalid=123
    )
    for symbol_results in results.values():
        for result in symbol_results:
            print(f"指標: {result.name}, メタデータ: {result.metadata}")
            # invalid_param が含まれていないことを確認
            assert "invalid_param" not in result.metadata
            assert "another_invalid" not in result.metadata
    
    print("\n--- フィボナッチ複合パラメータテスト ---")
    # フィボナッチの複合パラメータ
    results = manager.calculate_indicators(
        data, ["fibonacci_retracement"], ["Close"],
        period=50,
        trend_mode="up",
        custom_levels=[0.0, 0.382, 0.618, 1.0]
    )
    for symbol_results in results.values():
        for result in symbol_results:
            print(f"指標: {result.name}, メタデータ: {result.metadata}")
    
    return True


def main():
    """メインテスト実行"""
    print("Issue #594対応: パラメータフィルタリング簡略化テスト開始")
    print("=" * 70)
    
    try:
        success = test_issue_594_parameter_filtering()
        
        if success:
            print("\n[成功] Issue #594 パラメータフィルタリング簡略化テスト完了")
            print("パラメータ処理が大幅に簡略化されました")
        else:
            print("\n[失敗] テストが失敗しました")
            
        return success
        
    except Exception as e:
        print(f"\n[エラー] テスト実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)