#!/usr/bin/env python3
"""
データ品質システムの動作テスト
Issue #420対応
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

# テストデータ生成
def create_test_financial_data():
    """金融データのテストセット生成"""
    np.random.seed(42)

    # 正常データ
    clean_data = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'] * 25,
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(110, 220, 100),
        'low': np.random.uniform(90, 180, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.randint(1000000, 100000000, 100)
    })

    # 問題のあるデータ
    dirty_data = clean_data.copy()

    # 欠損値を意図的に作成
    dirty_data.loc[10:15, 'close'] = np.nan
    dirty_data.loc[20:22, 'volume'] = np.nan

    # 異常値を作成
    dirty_data.loc[30, 'high'] = -100  # 負の価格
    dirty_data.loc[31, 'volume'] = -1000  # 負の出来高
    dirty_data.loc[32, 'low'] = 1000000  # 異常に高いlow

    # 重複データ
    dirty_data = pd.concat([dirty_data, dirty_data.iloc[:5]], ignore_index=True)

    # 形式の不整合
    dirty_data.loc[40, 'symbol'] = 'invalid_symbol_format_12345'

    return clean_data, dirty_data

def test_basic_data_quality():
    """基本的なデータ品質テスト"""
    print("=== データ品質システム基本テスト ===")

    clean_data, dirty_data = create_test_financial_data()

    print(f"\n[統計] テストデータ概要:")
    print(f"  正常データ: {len(clean_data)}行")
    print(f"  問題データ: {len(dirty_data)}行")
    print(f"  欠損値: {dirty_data.isnull().sum().sum()}箇所")
    print(f"  重複行: {dirty_data.duplicated().sum()}行")

    # 基本的な品質チェック
    quality_issues = []

    # 1. 欠損値チェック
    missing_data = dirty_data.isnull().sum()
    for col, count in missing_data[missing_data > 0].items():
        quality_issues.append(f"欠損値: {col}列に{count}件")

    # 2. 重複チェック
    duplicates = dirty_data.duplicated().sum()
    if duplicates > 0:
        quality_issues.append(f"重複: {duplicates}行")

    # 3. 異常値チェック
    negative_prices = dirty_data[(dirty_data['high'] < 0) | (dirty_data['low'] < 0) | (dirty_data['close'] < 0)]
    if len(negative_prices) > 0:
        quality_issues.append(f"異常値: 負の価格{len(negative_prices)}件")

    negative_volume = dirty_data[dirty_data['volume'] < 0]
    if len(negative_volume) > 0:
        quality_issues.append(f"異常値: 負の出来高{len(negative_volume)}件")

    # 4. OHLC整合性チェック
    ohlc_issues = dirty_data[
        (dirty_data['high'] < dirty_data['low']) |
        (dirty_data['high'] < dirty_data['open']) |
        (dirty_data['high'] < dirty_data['close']) |
        (dirty_data['low'] > dirty_data['open']) |
        (dirty_data['low'] > dirty_data['close'])
    ]
    if len(ohlc_issues) > 0:
        quality_issues.append(f"OHLC整合性エラー: {len(ohlc_issues)}件")

    print(f"\n[検出] 検出された品質問題: {len(quality_issues)}件")
    for issue in quality_issues:
        print(f"  - {issue}")

    # データクリーニングの基本的な実装
    cleaned_data = dirty_data.copy()

    # 重複削除
    cleaned_data = cleaned_data.drop_duplicates()

    # 負の値を修正
    cleaned_data.loc[cleaned_data['high'] < 0, 'high'] = np.nan
    cleaned_data.loc[cleaned_data['low'] < 0, 'low'] = np.nan
    cleaned_data.loc[cleaned_data['close'] < 0, 'close'] = np.nan
    cleaned_data.loc[cleaned_data['volume'] < 0, 'volume'] = np.nan

    # 前方補完で欠損値を修正
    cleaned_data = cleaned_data.fillna(method='ffill')

    print(f"\n[清掃] クリーニング後:")
    print(f"  データ行数: {len(dirty_data)} → {len(cleaned_data)}")
    print(f"  欠損値: {dirty_data.isnull().sum().sum()} → {cleaned_data.isnull().sum().sum()}")

    return len(quality_issues) == 0

async def test_advanced_data_system():
    """高度なデータシステムのテスト"""
    print("\n=== 高度なデータ品質システムテスト ===")

    try:
        # 包括的データ品質システムのインポートを試行
        from src.day_trade.data.comprehensive_data_quality_system import ComprehensiveDataQualitySystem

        print("[利用可能] 包括的データ品質システム利用可能")

        # システム初期化
        dq_system = ComprehensiveDataQualitySystem(
            base_path=Path("test_data_quality"),
            enable_monitoring=True
        )

        clean_data, dirty_data = create_test_financial_data()

        # データ品質評価
        quality_result = await dq_system.assess_data_quality(
            data=dirty_data,
            dataset_name="test_financial_data"
        )

        print(f"  品質スコア: {quality_result.get('overall_score', 'N/A')}")
        print(f"  検出問題数: {len(quality_result.get('issues', []))}")

        # 自動修復を試行
        if hasattr(dq_system, 'auto_repair_data'):
            repaired_data = await dq_system.auto_repair_data(dirty_data)
            print(f"  修復後データ行数: {len(repaired_data)}")

        return True

    except ImportError as e:
        print(f"[エラー] 高度システム利用不可: {e}")
        return False
    except Exception as e:
        print(f"[エラー] システムエラー: {e}")
        return False

def test_data_validation_rules():
    """データバリデーションルールテスト"""
    print("\n=== データバリデーションルールテスト ===")

    clean_data, dirty_data = create_test_financial_data()

    # 金融データ特有のバリデーションルール
    validation_rules = {
        'price_positive': lambda df: (df[['open', 'high', 'low', 'close']] >= 0).all().all(),
        'volume_positive': lambda df: (df['volume'] >= 0).all(),
        'ohlc_consistency': lambda df: (
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ).all(),
        'symbol_format': lambda df: df['symbol'].str.len().le(10).all(),
        'no_missing_critical': lambda df: df[['symbol', 'date', 'close']].isnull().sum().sum() == 0
    }

    print("[検証] バリデーションルール適用:")

    for rule_name, rule_func in validation_rules.items():
        try:
            is_valid = rule_func(clean_data)
            print(f"  正常データ - {rule_name}: {'[合格] PASS' if is_valid else '[不合格] FAIL'}")

            is_valid_dirty = rule_func(dirty_data)
            print(f"  問題データ - {rule_name}: {'[合格] PASS' if is_valid_dirty else '[不合格] FAIL'}")

        except Exception as e:
            print(f"  {rule_name}: [警告] エラー - {e}")

    return True

def main():
    """メインテスト実行"""
    print("[開始] データ品質保証システム 総合テスト開始")

    # 基本テスト
    basic_result = test_basic_data_quality()

    # バリデーションルールテスト
    validation_result = test_data_validation_rules()

    # 高度システムテスト
    advanced_result = asyncio.run(test_advanced_data_system())

    # 総合評価
    print(f"\n[結果] テスト結果サマリー:")
    print(f"  基本データ品質: {'[合格] PASS' if basic_result else '[不合格] FAIL'}")
    print(f"  バリデーションルール: {'[合格] PASS' if validation_result else '[不合格] FAIL'}")
    print(f"  高度システム: {'[合格] PASS' if advanced_result else '[不合格] FAIL'}")

    if all([basic_result, validation_result, advanced_result]):
        print(f"\n[成功] 全テスト成功: データ品質システム正常動作")
        return True
    else:
        print(f"\n[警告] 一部テスト失敗: システム改善が必要")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
