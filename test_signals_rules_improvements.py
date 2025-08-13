#!/usr/bin/env python3
"""
Issues #650, #651, #653, #654 テストケース

signals.pyのルール系改善をテスト
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.analysis.signals import (
    MACDCrossoverRule,
    MACDDeathCrossRule,
    BollingerBandRule,
    PatternBreakoutRule,
    GoldenCrossRule,
    DeadCrossRule,
    _get_shared_config
)

def create_test_data():
    """テスト用のサンプルデータを作成"""
    dates = pd.date_range(end='2024-12-01', periods=100, freq='D')
    np.random.seed(42)
    
    # 価格データ
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(100) * 0.1,
        'High': prices + np.abs(np.random.randn(100)) * 0.2,
        'Low': prices - np.abs(np.random.randn(100)) * 0.2,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100),
    })
    df.set_index('Date', inplace=True)
    
    # 技術指標データ
    indicators = pd.DataFrame({
        'RSI': 50 + np.random.randn(100) * 15,
        'MACD': np.random.randn(100) * 0.5,
        'MACD_Signal': np.random.randn(100) * 0.4,
        'BB_Upper': prices + 2,
        'BB_Lower': prices - 2,
    }, index=dates)
    
    # パターンデータ
    patterns = {
        'crosses': pd.DataFrame({
            'Golden_Cross': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'Dead_Cross': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'Golden_Confidence': np.random.uniform(0, 100, 100),
            'Dead_Confidence': np.random.uniform(0, 100, 100),
        }, index=dates),
        'breakouts': pd.DataFrame({
            'Upward_Breakout': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'Downward_Breakout': np.random.choice([True, False], 100, p=[0.1, 0.9]),
            'Upward_Confidence': np.random.uniform(0, 100, 100),
            'Downward_Confidence': np.random.uniform(0, 100, 100),
        }, index=dates)
    }
    
    return df, indicators, patterns

def test_issue_650_macd_validation():
    """Issue #650: MACD lookback期間検証テスト"""
    print("=== Issue #650: MACD lookback期間検証テスト ===")
    
    df, indicators, patterns = create_test_data()
    config = _get_shared_config()
    
    # MACDCrossoverRuleのテスト
    macd_rule = MACDCrossoverRule(lookback=2)
    
    # 通常データでのテスト
    result, confidence = macd_rule.evaluate(df, indicators, patterns, config)
    print(f"  通常データ評価結果: {result}, 信頼度: {confidence}")
    
    # 不十分なデータでのテスト
    short_indicators = indicators.head(20)  # 26期間未満
    result_short, confidence_short = macd_rule.evaluate(df.head(20), short_indicators, patterns, config)
    
    if not result_short and confidence_short == 0.0:
        print("  [PASS] 不十分なデータで適切にFalseを返しました")
    else:
        print("  [FAIL] 不十分なデータでの処理に問題があります")
    
    print()

def test_issue_651_bollinger_robustness():
    """Issue #651: ボリンジャーバンド計算堅牢性テスト"""
    print("=== Issue #651: ボリンジャーバンド計算堅牢性テスト ===")
    
    df, indicators, patterns = create_test_data()
    
    # 低価格でのテスト
    low_price_df = df.copy()
    low_price_df['Close'] = 0.01  # 非常に低い価格
    
    bb_rule = BollingerBandRule(position="lower")
    result, confidence = bb_rule.evaluate(low_price_df, indicators, patterns)
    
    print(f"  低価格データ評価結果: {result}, 信頼度: {confidence}")
    
    # ゼロ価格でのテスト
    zero_price_df = df.copy()
    zero_price_df['Close'] = 0.0
    
    result_zero, confidence_zero = bb_rule.evaluate(zero_price_df, indicators, patterns)
    
    if confidence_zero >= 1.0:  # フォールバック信頼度
        print("  [PASS] ゼロ価格でフォールバック処理が動作しました")
    else:
        print("  [FAIL] ゼロ価格での処理に問題があります")
    
    print()

def test_issue_653_pattern_data_consistency():
    """Issue #653: パターンデータ一貫性テスト"""
    print("=== Issue #653: パターンデータ一貫性テスト ===")
    
    df, indicators, _ = create_test_data()
    
    breakout_rule = PatternBreakoutRule(direction="upward")
    
    # 正常なパターンデータ
    normal_patterns = {'breakouts': pd.DataFrame({'Upward_Breakout': [True], 'Upward_Confidence': [75.0]})}
    result_normal, conf_normal = breakout_rule.evaluate(df, indicators, normal_patterns)
    print(f"  正常データ評価結果: {result_normal}, 信頼度: {conf_normal}")
    
    # Noneパターンデータ
    none_patterns = {'breakouts': None}
    result_none, conf_none = breakout_rule.evaluate(df, indicators, none_patterns)
    
    if not result_none and conf_none == 0.0:
        print("  [PASS] Noneデータで適切にFalseを返しました")
    else:
        print("  [FAIL] Noneデータでの処理に問題があります")
    
    # 非DataFrameパターンデータ
    invalid_patterns = {'breakouts': "invalid"}
    result_invalid, conf_invalid = breakout_rule.evaluate(df, indicators, invalid_patterns)
    
    if not result_invalid and conf_invalid == 0.0:
        print("  [PASS] 無効データで適切にFalseを返しました")
    else:
        print("  [FAIL] 無効データでの処理に問題があります")
    
    print()

def test_issue_654_cross_rule_robustness():
    """Issue #654: ゴールデン/デッドクロスルール堅牢性テスト"""
    print("=== Issue #654: ゴールデン/デッドクロスルール堅牢性テスト ===")
    
    df, indicators, _ = create_test_data()
    
    golden_rule = GoldenCrossRule()
    
    # 空のクロスデータ
    empty_patterns = {'crosses': pd.DataFrame()}
    result_empty, conf_empty = golden_rule.evaluate(df, indicators, empty_patterns)
    
    if not result_empty and conf_empty == 0.0:
        print("  [PASS] 空データで適切にFalseを返しました")
    else:
        print("  [FAIL] 空データでの処理に問題があります")
    
    # 小さなクロスデータ
    small_patterns = {
        'crosses': pd.DataFrame({
            'Golden_Cross': [True],
            'Golden_Confidence': [80.0]
        })
    }
    result_small, conf_small = golden_rule.evaluate(df, indicators, small_patterns)
    print(f"  小データ評価結果: {result_small}, 信頼度: {conf_small}")
    
    if result_small and conf_small > 0:
        print("  [PASS] 小データでも適切に処理されました")
    else:
        print("  [INFO] 小データでの処理結果を確認しました")
    
    print()

def run_all_tests():
    """全テストを実行"""
    print("signals.py ルール系改善テスト開始\\n")
    
    test_issue_650_macd_validation()
    test_issue_651_bollinger_robustness()
    test_issue_653_pattern_data_consistency()
    test_issue_654_cross_rule_robustness()
    
    print("テスト完了")

if __name__ == "__main__":
    run_all_tests()