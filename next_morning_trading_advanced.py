#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Next Morning Trading Advanced System - 翌朝場モード高度化システム
Issue #887対応：翌朝場モードの信頼性向上と本格運用に向けた機能改善

翌朝場モードを本格運用レベルに引き上げる包括的改善システム:
1. 予測ロジックの高度化（機械学習統合）
2. データソースの多様化と信頼性向上
3. リスク管理機能の統合
4. 専用バックテスト環境の構築
"""

import asyncio
import logging
import sys
import os

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

from trading_system.main_system import NextMorningTradingAdvanced


# テスト関数
async def test_next_morning_trading_advanced():
    """翌朝場モード高度化システムテスト"""
    print("=== Advanced Next Morning Trading System Test ===")

    system = NextMorningTradingAdvanced()

    # テスト銘柄
    test_symbols = ["7203", "4751", "9984"]

    print(f"\n[ {len(test_symbols)}銘柄の翌朝場予測テスト ]")

    for symbol in test_symbols:
        print(f"\n--- {symbol} 翌朝場予測 ---")

        try:
            # 予測実行
            prediction = await system.predict_next_morning(symbol, account_balance=1000000, risk_tolerance=0.05)

            print(f"予測日時: {prediction.prediction_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"市場方向: {prediction.market_direction.value}")
            print(f"予測変動率: {prediction.predicted_change_percent:+.2f}%")
            print(f"信頼度: {prediction.confidence.value} ({prediction.confidence_score:.1%})")
            print(f"使用モデル: {prediction.model_used}")
            print(f"データソース: {', '.join(prediction.data_sources)}")

            print(f"\n=== センチメント分析 ===")
            sentiment = prediction.market_sentiment
            print(f"総合センチメント: {sentiment.sentiment_score:+.2f} (信頼度: {sentiment.confidence:.1%})")
            print(f"テクニカル: {sentiment.technical_sentiment:+.2f}")
            print(f"ファンダメンタル: {sentiment.fundamental_sentiment:+.2f}")
            print(f"ニュース: {sentiment.news_sentiment:+.2f}")
            print(f"主要要因: {', '.join(sentiment.key_factors)}")

            print(f"\n=== リスク指標 ===")
            risk = prediction.risk_metrics
            print(f"ボラティリティ: {risk.volatility:.1%}")
            print(f"VaR(95%): {risk.var_95:+.2f}%")
            print(f"最大ドローダウン: {risk.maximum_drawdown:+.1%}")
            print(f"シャープレシオ: {risk.sharpe_ratio:.2f}")

            print(f"\n=== ポジション推奨 ===")
            pos = prediction.position_recommendation
            print(f"推奨方向: {pos.direction.value}")
            print(f"エントリー価格: ¥{pos.entry_price:,.0f}")
            print(f"目標価格: ¥{pos.target_price:,.0f}")
            print(f"損切り価格: ¥{pos.stop_loss_price:,.0f}")
            print(f"ポジションサイズ: {pos.position_size_percentage:.1f}%")
            print(f"リスクレベル: {pos.risk_level.value}")
            print(f"保有期間: {pos.holding_period}")
            print(f"根拠: {pos.rationale}")

        except Exception as e:
            print(f"❌ {symbol}の予測に失敗: {e}")

    # バックテストテスト
    print(f"\n[ バックテストテスト ]")
    try:
        backtest_result = await system.run_strategy_backtest("7203", months=6)

        print(f"バックテスト期間: {backtest_result.period_start.strftime('%Y-%m-%d')} - {backtest_result.period_end.strftime('%Y-%m-%d')}")
        print(f"総取引数: {backtest_result.total_trades}")
        print(f"勝率: {backtest_result.win_rate:.1%}")
        print(f"総リターン: {backtest_result.total_return:+.1%}")
        print(f"年率リターン: {backtest_result.annualized_return:+.1%}")
        print(f"ボラティリティ: {backtest_result.volatility:.1%}")
        print(f"シャープレシオ: {backtest_result.sharpe_ratio:.2f}")
        print(f"最大ドローダウン: {backtest_result.max_drawdown:+.1%}")
        print(f"プロフィットファクター: {backtest_result.profit_factor:.2f}")

    except Exception as e:
        print(f"❌ バックテストに失敗: {e}")

    print(f"\n=== テスト完了 ===")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # テスト実行
    asyncio.run(test_next_morning_trading_advanced())
