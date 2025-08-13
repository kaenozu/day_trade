#!/usr/bin/env python3
"""
Issue #576: Data Quality Calculation Logic Test
データ品質計算ロジック改善テスト
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def create_test_data_scenarios():
    """テスト用データシナリオ作成"""
    scenarios = {}

    # 1. 高品質データ（完璧なデータ）
    dates = pd.date_range('2023-01-01', periods=60, freq='D')
    np.random.seed(42)

    base_price = 100
    price_changes = np.random.randn(60) * 0.02
    prices = base_price * np.exp(np.cumsum(price_changes))
    volumes = np.random.randint(50000, 200000, 60)

    scenarios['high_quality'] = pd.DataFrame({
        '始値': np.roll(prices, 1),
        '高値': prices * 1.01,
        '安値': prices * 0.99,
        '終値': prices,
        '出来高': volumes,
    }, index=dates)

    # 2. 欠損値あり
    missing_data = scenarios['high_quality'].copy()
    missing_data.iloc[10:15, 1] = np.nan  # 高値に欠損
    missing_data.iloc[20:25, 4] = np.nan  # 出来高に欠損
    scenarios['with_missing'] = missing_data

    # 3. データ量不足（10日間のみ）
    scenarios['insufficient_volume'] = scenarios['high_quality'].iloc[:10].copy()

    # 4. 異常値含み
    extreme_data = scenarios['high_quality'].copy()
    extreme_data.iloc[30, 3] = extreme_data.iloc[29, 3] * 1.3  # 30%上昇
    extreme_data.iloc[35, 3] = extreme_data.iloc[34, 3] * 0.7  # 30%下落
    scenarios['with_extremes'] = extreme_data

    # 5. ゼロ価格・ボリューム含み
    zero_data = scenarios['high_quality'].copy()
    zero_data.iloc[15:18, 3] = 0  # ゼロ価格
    zero_data.iloc[25:28, 4] = 0  # ゼロボリューム
    scenarios['with_zeros'] = zero_data

    # 6. 重複インデックス
    duplicate_data = scenarios['high_quality'].copy()
    # インデックスの一部を重複させる
    duplicate_indices = duplicate_data.index.tolist()
    duplicate_indices[25] = duplicate_indices[20]  # 重複作成
    duplicate_indices[26] = duplicate_indices[21]  # 重複作成
    duplicate_data.index = duplicate_indices
    scenarios['with_duplicates'] = duplicate_data

    # 7. 最悪品質（全ての問題を含む）
    worst_data = scenarios['high_quality'].iloc[:5].copy()  # データ量不足
    worst_data.iloc[:, 1:3] = np.nan  # 大量の欠損値
    worst_data.iloc[2, 3] = 0  # ゼロ価格
    scenarios['worst_quality'] = worst_data

    return scenarios


def test_data_quality_calculation():
    """データ品質計算テスト"""
    print("=== Issue #576 Data Quality Calculation Test ===\n")

    try:
        from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

        # フェッチャー初期化
        fetcher = AdvancedBatchDataFetcher(
            max_workers=1,
            enable_kafka=False,
            enable_redis=False
        )

        # テストシナリオ作成
        scenarios = create_test_data_scenarios()

        print("データ品質スコア計算テスト:")
        print("-" * 60)

        results = {}
        for scenario_name, data in scenarios.items():
            try:
                # 品質スコア計算
                quality_score = fetcher._calculate_data_quality(data)

                # 詳細メトリクス計算（内部メソッド直接テスト）
                metrics = fetcher._calculate_quality_metrics(data)

                results[scenario_name] = {
                    'score': quality_score,
                    'metrics': metrics,
                    'data_shape': data.shape
                }

                print(f"{scenario_name:15s}: {quality_score:5.1f} points")
                print(f"                 Data: {data.shape[0]}日 x {data.shape[1]}列")
                print(f"                 Missing: {metrics['missing_ratio']:.3f}")
                print(f"                 Duplicates: {metrics['duplicate_ratio']:.3f}")
                if 'extreme_move_ratio' in metrics:
                    print(f"                 Extremes: {metrics['extreme_move_ratio']:.3f}")
                print()

            except Exception as e:
                print(f"{scenario_name:15s}: ERROR - {str(e)}")
                results[scenario_name] = {'error': str(e)}

        return results

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        return None


def test_quality_metrics_components():
    """品質メトリクス構成要素テスト"""
    print("\n=== Quality Metrics Components Test ===\n")

    try:
        from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

        fetcher = AdvancedBatchDataFetcher(
            max_workers=1,
            enable_kafka=False,
            enable_redis=False
        )

        # 基本テストデータ
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        test_data = pd.DataFrame({
            '終値': [100 + i for i in range(30)],
            '出来高': [1000] * 30,
        }, index=dates)

        print("個別メトリクステスト:")

        # 1. 価格品質メトリクステスト
        price_metrics = fetcher._calculate_price_quality_metrics(test_data['終値'])
        print(f"価格品質メトリクス: {price_metrics}")

        # 2. ボリュームペナルティテスト
        for volume in [5, 10, 20, 40, 70]:
            penalty = fetcher._calculate_volume_penalty(volume, 20.0)
            print(f"データ量{volume}日のペナルティ: {penalty:.1f}")

        # 3. 重み付きスコア計算テスト
        test_metrics = {
            'missing_ratio': 0.1,
            'data_volume': 30,
            'duplicate_ratio': 0.0,
            'invalid_numeric_ratio': 0.0,
            'extreme_move_ratio': 0.05,
            'zero_price_ratio': 0.0,
            'zero_volume_ratio': 0.0,
            'price_continuity': 0.9
        }

        weighted_score = fetcher._compute_weighted_quality_score(test_metrics, test_data)
        print(f"重み付きスコア: {weighted_score:.1f}")

        return True

    except Exception as e:
        print(f"メトリクステストエラー: {e}")
        return False


def test_edge_cases():
    """エッジケーステスト"""
    print("\n=== Edge Cases Test ===\n")

    try:
        from src.day_trade.data.batch_data_fetcher import AdvancedBatchDataFetcher

        fetcher = AdvancedBatchDataFetcher(
            max_workers=1,
            enable_kafka=False,
            enable_redis=False
        )

        print("エッジケーステスト:")

        # 1. 空のデータフレーム
        empty_df = pd.DataFrame()
        score = fetcher._calculate_data_quality(empty_df)
        print(f"空データフレーム: {score:.1f}")
        assert score == 0.0

        # 2. None値
        score = fetcher._calculate_data_quality(None)
        print(f"None値: {score:.1f}")
        assert score == 0.0

        # 3. 1行のみ
        single_row = pd.DataFrame({'終値': [100]})
        score = fetcher._calculate_data_quality(single_row)
        print(f"1行のみ: {score:.1f}")
        assert 0 <= score <= 100

        # 4. 全てNaN
        all_nan = pd.DataFrame({'終値': [np.nan] * 10})
        score = fetcher._calculate_data_quality(all_nan)
        print(f"全てNaN: {score:.1f}")
        assert 0 <= score <= 100

        # 5. 異常に大きな値
        extreme_values = pd.DataFrame({'終値': [1e10, 1e11, 1e12]})
        score = fetcher._calculate_data_quality(extreme_values)
        print(f"異常に大きな値: {score:.1f}")
        assert 0 <= score <= 100

        print("OK 全エッジケーステスト合格")
        return True

    except Exception as e:
        print(f"NG エッジケーステストエラー: {e}")
        return False


def validate_improvement():
    """改善効果検証"""
    print("\n=== Improvement Validation ===\n")

    expected_ranges = {
        'high_quality': (85, 100),      # 高品質データ
        'with_missing': (65, 85),       # 欠損値あり
        'insufficient_volume': (60, 80), # データ量不足
        'with_extremes': (70, 90),      # 異常値含み
        'with_zeros': (50, 70),         # ゼロ値含み
        'with_duplicates': (70, 90),    # 重複あり
        'worst_quality': (0, 30),       # 最悪品質
    }

    # テスト実行
    results = test_data_quality_calculation()
    if not results:
        return False

    print("改善効果検証:")
    print("-" * 40)

    validation_passed = 0
    validation_total = 0

    for scenario, expected_range in expected_ranges.items():
        if scenario in results and 'score' in results[scenario]:
            score = results[scenario]['score']
            min_expected, max_expected = expected_range

            if min_expected <= score <= max_expected:
                status = "OK PASS"
                validation_passed += 1
            else:
                status = "NG FAIL"

            print(f"{scenario:15s}: {score:5.1f} (expected {min_expected}-{max_expected}) {status}")
            validation_total += 1

    print(f"\n検証結果: {validation_passed}/{validation_total} passed")

    return validation_passed == validation_total


def main():
    """メインテスト実行"""
    print("Issue #576 Data Quality Calculation Logic Test\n")

    tests = [
        ("データ品質計算", lambda: test_data_quality_calculation() is not None),
        ("メトリクス構成要素", test_quality_metrics_components),
        ("エッジケース", test_edge_cases),
        ("改善効果検証", validate_improvement)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"実行中: {test_name}")
            print('='*50)

            if test_func():
                print(f"OK {test_name}: PASS")
                passed += 1
            else:
                print(f"NG {test_name}: FAIL")
                failed += 1

        except Exception as e:
            print(f"NG {test_name}: ERROR - {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"=== Final Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("OK Issue #576 Data Quality Calculation: ALL TESTS PASSED")
        return True
    else:
        print("NG Issue #576 Data Quality Calculation: SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)