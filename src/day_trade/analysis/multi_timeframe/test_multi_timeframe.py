#!/usr/bin/env python3
"""
マルチタイムフレーム分析システム テストファイル
Issue #315: 高度テクニカル指標・ML機能拡張

元のテスト機能を分割後のシステムで実行
"""

import numpy as np
import pandas as pd

from .main import MultiTimeframeAnalyzer


def run_multi_timeframe_test():
    """マルチタイムフレーム分析システムのテストを実行"""
    print("=== マルチタイムフレーム分析システム テスト ===")

    # サンプルデータ生成（2年間の日足データ）
    dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)

    # より複雑な価格パターンを生成
    base_price = 2500
    trend_periods = [
        (0, 100, 0.001),  # 100日間上昇トレンド
        (100, 200, -0.0005),  # 100日間下降トレンド
        (200, 300, 0.0003),  # 100日間横ばい
        (300, 500, 0.0012),  # 200日間強い上昇
        (500, 600, -0.0008),  # 100日間調整
        (600, len(dates), 0.0005),  # 残り期間緩やかな上昇
    ]

    prices = [base_price]
    volatility = 0.02

    for i in range(1, len(dates)):
        # 現在のトレンド期間を特定
        current_trend = 0
        for start, end, trend in trend_periods:
            if start <= i < end:
                current_trend = trend
                break

        # ランダムウォーク + トレンド + 週末効果
        weekday_effect = -0.0002 if dates[i].weekday() == 4 else 0  # 金曜日効果
        seasonal_effect = 0.0005 * np.sin(2 * np.pi * i / 252)  # 年次季節性

        random_change = np.random.normal(
            current_trend + weekday_effect + seasonal_effect, volatility
        )
        new_price = prices[-1] * (1 + random_change)
        prices.append(max(new_price, 500))  # 価格下限設定

    # OHLCV生成
    sample_data = pd.DataFrame(index=dates)
    sample_data["Close"] = prices
    sample_data["Open"] = [p * np.random.uniform(0.995, 1.005) for p in prices]
    sample_data["High"] = [
        max(o, c) * np.random.uniform(1.000, 1.025)
        for o, c in zip(sample_data["Open"], sample_data["Close"])
    ]
    sample_data["Low"] = [
        min(o, c) * np.random.uniform(0.975, 1.000)
        for o, c in zip(sample_data["Open"], sample_data["Close"])
    ]
    sample_data["Volume"] = np.random.randint(1000000, 20000000, len(dates))

    try:
        analyzer = MultiTimeframeAnalyzer()

        print(f"サンプルデータ: {len(sample_data)}日分")
        print(
            f"価格範囲: {sample_data['Close'].min():.2f} - {sample_data['Close'].max():.2f}"
        )

        # 各時間軸でのリサンプリングテスト
        print("\n1. 時間軸リサンプリングテスト")
        for tf in ["daily", "weekly", "monthly"]:
            resampled = analyzer.resample_to_timeframe(sample_data, tf)
            print(f"✅ {tf}リサンプリング: {len(sample_data)} → {len(resampled)}期間")

        # 単一時間軸指標計算テスト
        print("\n2. 単一時間軸指標計算テスト")
        for tf in ["daily", "weekly", "monthly"]:
            tf_indicators = analyzer.calculate_timeframe_indicators(sample_data, tf)
            if not tf_indicators.empty:
                print(
                    f"✅ {tf}指標計算完了: {len(tf_indicators.columns)}指標, {len(tf_indicators)}期間"
                )

                # 最新データの表示
                latest = tf_indicators.iloc[-1]
                trend = latest.get("trend_direction", "unknown")
                strength = latest.get("trend_strength", 0)
                print(f"   最新トレンド: {trend} (強度: {strength:.1f})")
            else:
                print(f"❌ {tf}指標計算失敗")

        # マルチタイムフレーム統合分析テスト
        print("\n3. マルチタイムフレーム統合分析テスト")
        integrated_analysis = analyzer.analyze_multiple_timeframes(
            sample_data, "TEST_STOCK"
        )

        if "error" not in integrated_analysis:
            print("✅ 統合分析完了")

            # 時間軸別結果
            print("\n📊 時間軸別分析結果:")
            for tf, result in integrated_analysis["timeframes"].items():
                print(f"   {result['timeframe']}:")
                print(f"     トレンド: {result['trend_direction']}")
                print(f"     強度: {result['trend_strength']:.1f}")
                print(f"     現在価格: {result['current_price']:.2f}")

                if "technical_indicators" in result:
                    indicators = result["technical_indicators"]
                    if "rsi" in indicators:
                        print(f"     RSI: {indicators['rsi']:.1f}")
                    if "bb_position" in indicators:
                        print(f"     BB位置: {indicators['bb_position']:.2f}")

            # 統合結果
            integrated = integrated_analysis["integrated_analysis"]
            print("\n🔍 統合分析結果:")
            print(f"   総合トレンド: {integrated['overall_trend']}")
            print(f"   トレンド信頼度: {integrated['trend_confidence']:.1f}%")
            print(f"   整合性スコア: {integrated['consistency_score']:.1f}%")

            # 統合シグナル
            signal = integrated["integrated_signal"]
            print("\n📈 統合シグナル:")
            print(f"   アクション: {signal['action']}")
            print(f"   強度: {signal['strength']}")
            print(f"   シグナルスコア: {signal['signal_score']:.1f}")

            # リスク評価
            risk = integrated["risk_assessment"]
            print("\n⚠️  リスク評価:")
            print(f"   リスクレベル: {risk['risk_level']}")
            print(f"   リスクスコア: {risk['risk_score']:.1f}")
            print(f"   リスク要因数: {risk['total_risk_factors']}")
            if risk["risk_factors"]:
                for factor in risk["risk_factors"][:3]:  # 上位3個表示
                    print(f"     - {factor}")

            # 投資推奨
            recommendation = integrated["investment_recommendation"]
            print("\n💡 投資推奨:")
            print(f"   推奨: {recommendation['recommendation']}")
            print(f"   ポジションサイズ: {recommendation['position_size']}")
            print(f"   保有期間: {recommendation['holding_period']}")

            if recommendation.get("stop_loss_suggestion"):
                print(f"   ストップロス: {recommendation['stop_loss_suggestion']:.2f}")
            if recommendation.get("take_profit_suggestion"):
                print(f"   利益確定: {recommendation['take_profit_suggestion']:.2f}")

            print("\n📋 推奨理由:")
            for reason in recommendation.get("reasons", []):
                print(f"     - {reason}")

        else:
            print(f"❌ 統合分析エラー: {integrated_analysis['error']}")

        print("\n✅ マルチタイムフレーム分析システム テスト完了！")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_multi_timeframe_test()