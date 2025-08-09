#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine バックテストシステム動作確認

LSTM-Transformer + PPO強化学習 + センチメント分析統合バックテスト
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Next-Gen バックテストエンジン
from src.day_trade.backtesting.nextgen_backtest_engine import (
    NextGenBacktestEngine,
    NextGenBacktestConfig,
    run_nextgen_backtest
)

async def test_nextgen_backtest_comprehensive():
    """包括的Next-Gen AIバックテストテスト"""

    print("=" * 60)
    print("Next-Gen AI Trading Engine バックテストシステム")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # テスト設定
    test_symbols = ["7203", "8306", "9984", "6758", "4689"]  # 日本主要株

    # バックテスト設定
    config = NextGenBacktestConfig(
        start_date="2023-01-01",
        end_date="2023-06-30",  # 短期間でテスト
        initial_capital=10000000.0,  # 1000万円
        max_position_size=0.15,  # 最大15%ポジション
        transaction_cost=0.001,  # 0.1%取引コスト

        # AI設定
        enable_ml_engine=True,
        enable_rl_agent=True,
        enable_sentiment=True,

        # ML設定
        ml_sequence_length=30,  # 短縮
        ml_prediction_threshold=0.5,

        # RL設定
        rl_training_episodes=20,  # 短縮
        rl_exploration_rate=0.1,

        # リスク管理
        max_drawdown=0.20,
        stop_loss=0.08,
        take_profit=0.15
    )

    print("テスト設定:")
    print(f"  対象銘柄: {test_symbols}")
    print(f"  期間: {config.start_date} ～ {config.end_date}")
    print(f"  初期資本: ¥{config.initial_capital:,.0f}")
    print(f"  ML予測: {'有効' if config.enable_ml_engine else '無効'}")
    print(f"  RL判断: {'有効' if config.enable_rl_agent else '無効'}")
    print(f"  センチメント: {'有効' if config.enable_sentiment else '無効'}")
    print()

    try:
        print("🚀 Next-Gen AIバックテスト実行開始...")

        # バックテスト実行
        result = await run_nextgen_backtest(test_symbols, config)

        print("✅ バックテスト実行完了")
        print()

        # 結果表示
        print("=" * 50)
        print("📊 バックテスト結果サマリー")
        print("=" * 50)

        # 基本パフォーマンス
        print("🎯 基本パフォーマンス:")
        print(f"  総リターン: {result.total_return:+.2%}")
        print(f"  年率リターン: {result.annualized_return:+.2%}")
        print(f"  シャープレシオ: {result.sharpe_ratio:.2f}")
        print(f"  最大ドローダウン: {result.max_drawdown:.2%}")
        print(f"  カルマーレシオ: {result.calmar_ratio:.2f}")
        print()

        # AI パフォーマンス
        print("🤖 AI パフォーマンス:")
        print(f"  ML予測精度: {result.ml_accuracy:.1%}")
        print(f"  RL成功率: {result.rl_success_rate:.1%}")
        print(f"  センチメント相関: {result.sentiment_correlation:.1%}")
        print()

        # 取引統計
        print("📈 取引統計:")
        print(f"  総取引数: {result.total_trades:,} 回")
        print(f"  勝率: {result.win_rate:.1%}")
        print(f"  平均保有期間: {result.avg_holding_period:.1f} 日")
        print()

        # システム性能
        print("⚡ システム性能:")
        print(f"  バックテスト実行時間: {result.backtest_duration:.2f} 秒")
        print(f"  AI判断ログ: {len(result.ai_decisions_log):,} 件")
        print(f"  エクイティカーブ: {len(result.equity_curve):,} ポイント")
        print()

        # 総合評価
        print("🏆 総合評価:")

        # パフォーマンス評価
        if result.total_return > 0.1:
            performance_grade = "A (優秀)"
        elif result.total_return > 0.05:
            performance_grade = "B (良好)"
        elif result.total_return > 0:
            performance_grade = "C (普通)"
        else:
            performance_grade = "D (要改善)"

        # リスク評価
        if result.max_drawdown < 0.05:
            risk_grade = "A (低リスク)"
        elif result.max_drawdown < 0.10:
            risk_grade = "B (中リスク)"
        elif result.max_drawdown < 0.20:
            risk_grade = "C (高リスク)"
        else:
            risk_grade = "D (危険)"

        # AI統合評価
        ai_avg_score = (result.ml_accuracy + result.rl_success_rate + result.sentiment_correlation) / 3
        if ai_avg_score > 0.8:
            ai_grade = "A (優秀)"
        elif ai_avg_score > 0.7:
            ai_grade = "B (良好)"
        elif ai_avg_score > 0.6:
            ai_grade = "C (普通)"
        else:
            ai_grade = "D (要改善)"

        print(f"  パフォーマンス: {performance_grade}")
        print(f"  リスク管理: {risk_grade}")
        print(f"  AI統合度: {ai_grade}")
        print()

        # 詳細分析
        if result.trades:
            print("📋 取引詳細（最初の5件）:")
            for i, trade in enumerate(result.trades[:5]):
                ml_conf = trade.ml_prediction.get('confidence', 0) if trade.ml_prediction else 0
                rl_conf = trade.rl_decision.get('confidence', 0) if trade.rl_decision else 0
                sent_score = trade.sentiment_analysis.get('score', 0) if trade.sentiment_analysis else 0

                print(f"  [{i+1}] {trade.timestamp.strftime('%m/%d')} "
                      f"{trade.action} {trade.symbol} "
                      f"qty:{trade.quantity:.1f} @¥{trade.price:.0f} "
                      f"(ML:{ml_conf:.2f} RL:{rl_conf:.2f} 感情:{sent_score:+.2f})")

        # 成功判定
        overall_success = (
            result.total_return > 0 and
            result.max_drawdown < 0.25 and
            result.total_trades > 0 and
            result.backtest_duration < 300  # 5分以内
        )

        print()
        print("=" * 50)
        if overall_success:
            print("🎉 Next-Gen AI バックテストシステム 動作確認成功!")
            print("   システムは期待通りに動作しています。")
        else:
            print("⚠️  システム改善の余地があります。")
            print("   パフォーマンスやリスク指標を確認してください。")

        print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return overall_success

    except Exception as e:
        print(f"❌ Next-Gen AIバックテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_components():
    """個別コンポーネントテスト"""

    print("\n" + "=" * 50)
    print("🔧 個別コンポーネントテスト")
    print("=" * 50)

    # エンジン初期化テスト
    print("1. バックテストエンジン初期化テスト...")
    try:
        config = NextGenBacktestConfig()
        engine = NextGenBacktestEngine(config)
        print("   ✅ エンジン初期化成功")
    except Exception as e:
        print(f"   ❌ エンジン初期化失敗: {e}")
        return False

    # 設定テスト
    print("2. 設定カスタマイズテスト...")
    try:
        custom_config = NextGenBacktestConfig(
            initial_capital=5000000.0,
            enable_ml_engine=False,
            enable_rl_agent=True,
            enable_sentiment=True
        )
        custom_engine = NextGenBacktestEngine(custom_config)
        print("   ✅ カスタム設定成功")
    except Exception as e:
        print(f"   ❌ カスタム設定失敗: {e}")
        return False

    # データ構造テスト
    print("3. データ構造テスト...")
    try:
        from src.day_trade.backtesting.nextgen_backtest_engine import NextGenTrade, NextGenBacktestResult

        # テスト取引作成
        test_trade = NextGenTrade(
            symbol="TEST",
            action="BUY",
            quantity=100,
            price=1000.0,
            timestamp=datetime.now()
        )

        trade_value = test_trade.get_trade_value()
        assert trade_value == 100000.0

        print("   ✅ データ構造テスト成功")
    except Exception as e:
        print(f"   ❌ データ構造テスト失敗: {e}")
        return False

    print("✅ 全個別コンポーネントテスト合格")
    return True

async def main():
    """メイン実行関数"""

    print("Next-Gen AI Trading Engine バックテストシステム 総合テスト")
    print("=" * 70)

    # 個別コンポーネントテスト
    component_success = await test_individual_components()

    if not component_success:
        print("❌ 個別コンポーネントテストに失敗したため、総合テストをスキップします。")
        return False

    # 総合バックテストテスト
    backtest_success = await test_nextgen_backtest_comprehensive()

    # 最終結果
    overall_success = component_success and backtest_success

    print("\n" + "=" * 70)
    print("🏁 最終結果")
    print("=" * 70)

    if overall_success:
        print("🎉 Next-Gen AI Trading Engine バックテストシステム")
        print("   全ての機能が正常に動作しました！")
        print()
        print("✨ システムの特徴:")
        print("   • LSTM-Transformer機械学習予測")
        print("   • PPO強化学習による意思決定")
        print("   • センチメント分析統合")
        print("   • 高度なリスク管理")
        print("   • 包括的なパフォーマンス分析")
    else:
        print("⚠️  一部の機能に問題があります。")
        print("   ログを確認して改善してください。")

    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
