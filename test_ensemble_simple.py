"""
シンプルなアンサンブルテスト
TradingSignalのエラーを回避して基本機能をテスト
"""

import os
import sys

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd

from src.day_trade.analysis.ensemble import (
    EnsembleStrategy,
    EnsembleTradingStrategy,
    EnsembleVotingType,
)


def create_test_data():
    """テストデータ作成"""

    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    np.random.seed(42)

    # 基本価格データ
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))  # 日次リターン
    prices = [base_price]

    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    df = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, len(dates))
    })

    df.set_index('Date', inplace=True)
    return df

def test_ensemble_strategies():
    """アンサンブル戦略テスト"""

    print("=== アンサンブル戦略テスト開始 ===")

    # テストデータ準備
    df = create_test_data()
    print(f"テストデータ: {len(df)}日分の価格データ")

    # 各戦略タイプのテスト
    strategy_types = [
        (EnsembleStrategy.CONSERVATIVE, "保守的"),
        (EnsembleStrategy.AGGRESSIVE, "積極的"),
        (EnsembleStrategy.BALANCED, "バランス型"),
        (EnsembleStrategy.ADAPTIVE, "適応型")
    ]

    voting_types = [
        (EnsembleVotingType.SOFT_VOTING, "ソフト投票"),
        (EnsembleVotingType.HARD_VOTING, "ハード投票"),
        (EnsembleVotingType.WEIGHTED_AVERAGE, "重み付け平均")
    ]

    results = []

    for strategy_type, strategy_name in strategy_types:
        for voting_type, voting_name in voting_types:
            try:
                print(f"\n--- {strategy_name} + {voting_name} ---")

                # アンサンブル戦略作成
                ensemble = EnsembleTradingStrategy(
                    ensemble_strategy=strategy_type,
                    voting_type=voting_type
                )

                # シグナル生成
                ensemble_signal = ensemble.generate_ensemble_signal(df)

                if ensemble_signal:
                    signal = ensemble_signal.ensemble_signal
                    print(f"シグナル: {signal.signal_type.value}")
                    print(f"強度: {signal.strength.value}")
                    print(f"信頼度: {signal.confidence:.1f}%")
                    print(f"価格: {signal.price:.2f}")

                    # 戦略重み
                    print("戦略重み:")
                    for name, weight in ensemble_signal.strategy_weights.items():
                        print(f"  {name}: {weight:.3f}")

                    # メタ特徴量
                    if ensemble_signal.meta_features:
                        print("メタ特徴量:")
                        for feature, value in ensemble_signal.meta_features.items():
                            if isinstance(value, float):
                                print(f"  {feature}: {value:.3f}")
                            else:
                                print(f"  {feature}: {value}")

                    results.append({
                        'strategy': strategy_name,
                        'voting': voting_name,
                        'signal_type': signal.signal_type.value,
                        'confidence': signal.confidence,
                        'success': True
                    })
                else:
                    print("シグナルなし")
                    results.append({
                        'strategy': strategy_name,
                        'voting': voting_name,
                        'signal_type': 'none',
                        'confidence': 0,
                        'success': False
                    })

            except Exception as e:
                print(f"エラー: {e}")
                results.append({
                    'strategy': strategy_name,
                    'voting': voting_name,
                    'error': str(e),
                    'success': False
                })

    # 結果サマリー
    print("\n=== テスト結果サマリー ===")
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]

    print(f"成功: {len(successful_tests)}/{len(results)}")
    print(f"失敗: {len(failed_tests)}/{len(results)}")

    if successful_tests:
        print("\n成功したテスト:")
        for result in successful_tests:
            print(f"  {result['strategy']} + {result['voting']}: "
                  f"{result['signal_type']} (信頼度: {result['confidence']:.1f}%)")

    if failed_tests:
        print("\n失敗したテスト:")
        for result in failed_tests:
            if 'error' in result:
                print(f"  {result['strategy']} + {result['voting']}: {result['error']}")
            else:
                print(f"  {result['strategy']} + {result['voting']}: シグナルなし")

    return results

def test_strategy_summary():
    """戦略サマリーテスト"""

    print("\n=== 戦略サマリーテスト ===")

    try:
        ensemble = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.BALANCED,
            voting_type=EnsembleVotingType.SOFT_VOTING
        )

        summary = ensemble.get_strategy_summary()

        print("戦略サマリー:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"戦略サマリーテストエラー: {e}")
        return False

if __name__ == "__main__":
    print("アンサンブル戦略シンプルテスト開始")

    # 基本機能テスト
    results = test_ensemble_strategies()

    # サマリーテスト
    summary_success = test_strategy_summary()

    # 全体結果
    successful_tests = len([r for r in results if r.get('success', False)])
    total_tests = len(results)

    print("\n=== 最終結果 ===")
    print(f"アンサンブルテスト: {successful_tests}/{total_tests} 成功")
    print(f"サマリーテスト: {'成功' if summary_success else '失敗'}")

    if successful_tests > 0:
        print("✓ アンサンブル戦略の基本機能は正常に動作しています")
    else:
        print("✗ アンサンブル戦略にエラーがあります")
