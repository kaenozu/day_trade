"""
テクニカル指標付きアンサンブルテスト
テクニカル指標を計算してアンサンブル戦略をテスト
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


def calculate_technical_indicators(df):
    """テクニカル指標計算"""

    indicators = pd.DataFrame(index=df.index)
    close = df['Close']
    df['High']
    df['Low']
    volume = df['Volume']

    # RSI計算
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = 100 - (100 / (1 + rs))

    # MACD計算
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    indicators['MACD'] = ema12 - ema26
    indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()

    # 移動平均
    indicators['SMA_20'] = close.rolling(window=20).mean()
    indicators['SMA_50'] = close.rolling(window=50).mean()

    # ボリンジャーバンド
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    indicators['BB_Upper'] = sma_20 + (std_20 * 2)
    indicators['BB_Lower'] = sma_20 - (std_20 * 2)
    indicators['BB_Middle'] = sma_20

    # 出来高指標
    indicators['Volume_SMA'] = volume.rolling(window=20).mean()
    indicators['Volume_Ratio'] = volume / indicators['Volume_SMA']

    return indicators

def create_test_data():
    """テストデータ作成"""

    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    np.random.seed(42)

    # より現実的な価格データ生成
    n_days = len(dates)
    base_price = 100

    # トレンド成分
    trend = np.linspace(0, 20, n_days)  # 上昇トレンド

    # 季節性成分
    seasonality = 5 * np.sin(np.linspace(0, 4*np.pi, n_days))

    # ノイズ成分
    noise = np.random.normal(0, 2, n_days)

    # 価格系列
    close_prices = base_price + trend + seasonality + noise

    # OHLV生成
    opens = close_prices + np.random.normal(0, 0.5, n_days)
    highs = np.maximum(opens, close_prices) + np.abs(np.random.normal(0, 1, n_days))
    lows = np.minimum(opens, close_prices) - np.abs(np.random.normal(0, 1, n_days))
    volumes = np.random.lognormal(12, 0.5, n_days).astype(int)

    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': close_prices,
        'Volume': volumes
    })

    df.set_index('Date', inplace=True)
    return df

def test_ensemble_with_indicators():
    """テクニカル指標付きアンサンブルテスト"""

    print("=== テクニカル指標付きアンサンブルテスト開始 ===")

    # テストデータとテクニカル指標
    df = create_test_data()
    indicators = calculate_technical_indicators(df)

    print(f"テストデータ: {len(df)}日分")
    print(f"テクニカル指標: {list(indicators.columns)}")

    # テクニカル指標の統計情報
    print("\nテクニカル指標統計:")
    for col in ['RSI', 'MACD', 'SMA_20']:
        if col in indicators.columns:
            latest_value = indicators[col].iloc[-1]
            if not pd.isna(latest_value):
                print(f"  {col}: {latest_value:.2f}")

    # アンサンブル戦略テスト
    test_configs = [
        (EnsembleStrategy.CONSERVATIVE, EnsembleVotingType.SOFT_VOTING, "保守的+ソフト投票"),
        (EnsembleStrategy.AGGRESSIVE, EnsembleVotingType.HARD_VOTING, "積極的+ハード投票"),
        (EnsembleStrategy.BALANCED, EnsembleVotingType.SOFT_VOTING, "バランス+ソフト投票"),
    ]

    results = []

    for strategy_type, voting_type, description in test_configs:
        try:
            print(f"\n--- {description} ---")

            # アンサンブル戦略作成
            ensemble = EnsembleTradingStrategy(
                ensemble_strategy=strategy_type,
                voting_type=voting_type
            )

            # シグナル生成（テクニカル指標付き）
            ensemble_signal = ensemble.generate_ensemble_signal(
                df=df,
                indicators=indicators,
                patterns={}  # パターン認識は省略
            )

            if ensemble_signal:
                signal = ensemble_signal.ensemble_signal
                print(f"シグナル: {signal.signal_type.value}")
                print(f"強度: {signal.strength.value}")
                print(f"信頼度: {signal.confidence:.1f}%")

                # 戦略別貢献度
                print("戦略別貢献度:")
                for strategy_name, score in ensemble_signal.voting_scores.items():
                    print(f"  {strategy_name}: {score:.3f}")

                # メタ特徴量
                if ensemble_signal.meta_features:
                    print("メタ特徴量:")
                    for feature, value in ensemble_signal.meta_features.items():
                        if isinstance(value, (int, float)):
                            print(f"  {feature}: {value:.3f}")
                        else:
                            print(f"  {feature}: {value}")

                results.append({
                    'config': description,
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'success': True
                })
            else:
                print("シグナルなし")
                results.append({
                    'config': description,
                    'success': False,
                    'reason': 'no_signal'
                })

        except Exception as e:
            print(f"エラー: {e}")
            results.append({
                'config': description,
                'success': False,
                'error': str(e)
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
            print(f"  {result['config']}: {result['signal_type']} (信頼度: {result['confidence']:.1f}%)")

    if failed_tests:
        print("\n失敗したテスト:")
        for result in failed_tests:
            if 'error' in result:
                print(f"  {result['config']}: {result['error']}")
            else:
                print(f"  {result['config']}: {result.get('reason', 'unknown')}")

    return len(successful_tests) > 0

def test_strategy_weights_analysis():
    """戦略重み分析テスト"""

    print("\n=== 戦略重み分析テスト ===")

    try:
        df = create_test_data()
        calculate_technical_indicators(df)

        # 各戦略タイプの重み比較
        strategy_types = [
            (EnsembleStrategy.CONSERVATIVE, "保守的"),
            (EnsembleStrategy.AGGRESSIVE, "積極的"),
            (EnsembleStrategy.BALANCED, "バランス"),
            (EnsembleStrategy.ADAPTIVE, "適応型")
        ]

        print("戦略タイプ別重み分析:")
        for strategy_type, name in strategy_types:
            ensemble = EnsembleTradingStrategy(
                ensemble_strategy=strategy_type,
                voting_type=EnsembleVotingType.SOFT_VOTING
            )

            print(f"\n{name}戦略:")
            for strategy_name, weight in ensemble.strategy_weights.items():
                print(f"  {strategy_name}: {weight:.3f}")

        return True

    except Exception as e:
        print(f"戦略重み分析エラー: {e}")
        return False

if __name__ == "__main__":
    print("テクニカル指標付きアンサンブルテスト開始")

    # メインテスト
    main_success = test_ensemble_with_indicators()

    # 戦略重み分析
    weights_success = test_strategy_weights_analysis()

    print("\n=== 最終結果 ===")
    print(f"メインテスト: {'成功' if main_success else '失敗'}")
    print(f"重み分析: {'成功' if weights_success else '失敗'}")

    if main_success and weights_success:
        print("✓ アンサンブル戦略は正常に動作しています")
    else:
        print("✗ アンサンブル戦略にエラーがあります")
