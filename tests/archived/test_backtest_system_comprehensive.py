#!/usr/bin/env python3
"""
統合バックテストシステム総合テスト

Issue #323: 実データバックテスト機能開発
バックテストエンジン・リスクメトリクス・戦略評価の統合テスト
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

# プロジェクトルート設定
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

try:
    from day_trade.backtesting.backtest_engine import BacktestEngine
    from day_trade.backtesting.risk_metrics import RiskMetricsCalculator
    from day_trade.backtesting.strategy_evaluator import (
        StrategyEvaluator,
        create_sample_strategies,
    )
except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    print("簡易版でテストを実行します")

print("統合バックテストシステム総合テスト")
print("Issue #323: 実データバックテスト機能開発")
print("=" * 60)


def create_advanced_strategies() -> Dict[str, callable]:
    """高度な戦略作成"""

    def ml_momentum_strategy(
        lookback_data: Dict[str, callable], current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """ML強化モメンタム戦略"""
        import numpy as np

        signals = {}
        for symbol, data in lookback_data.items():
            if len(data) >= 30:
                # 価格データ
                prices = data["Close"].values
                volumes = data["Volume"].values

                # 複数期間リターン
                ret_5d = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
                ret_10d = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0
                ret_20d = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0

                # ボラティリティ
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0

                # 出来高トレンド
                volume_trend = (
                    (volumes[-5:].mean() / volumes[-20:-5].mean() - 1)
                    if len(volumes) >= 20
                    else 0
                )

                # 複合シグナル計算
                momentum_score = ret_5d * 0.4 + ret_10d * 0.3 + ret_20d * 0.3
                volume_score = min(1.0, max(-1.0, volume_trend))
                volatility_penalty = max(
                    0.5, 1 - volatility * 10
                )  # 高ボラティリティにペナルティ

                composite_score = momentum_score * volume_score * volatility_penalty

                # ポジションサイズ決定
                if composite_score > 0.08:  # 強いポジティブシグナル
                    signals[symbol] = 0.25
                elif composite_score > 0.03:
                    signals[symbol] = 0.15
                elif composite_score < -0.05:  # ネガティブシグナル
                    signals[symbol] = 0.02
                else:
                    signals[symbol] = 0.08

        # ウェイト正規化
        total_weight = sum(signals.values())
        if total_weight > 0:
            signals = {k: min(0.3, v / total_weight) for k, v in signals.items()}

        return signals

    def risk_parity_strategy(
        lookback_data: Dict[str, callable], current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """リスクパリティ戦略"""
        import numpy as np

        signals = {}
        volatilities = {}

        # 各銘柄のボラティリティ計算
        for symbol, data in lookback_data.items():
            if len(data) >= 20:
                returns = data["Close"].pct_change().dropna()
                if len(returns) >= 10:
                    vol = returns.std() * np.sqrt(252)  # 年率ボラティリティ
                    volatilities[symbol] = vol

        if volatilities:
            # 逆ボラティリティウェイト
            inv_vol = {k: 1 / v for k, v in volatilities.items() if v > 0}
            total_inv_vol = sum(inv_vol.values())

            if total_inv_vol > 0:
                signals = {k: v / total_inv_vol for k, v in inv_vol.items()}

        return signals

    def adaptive_momentum_strategy(
        lookback_data: Dict[str, callable], current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """適応的モメンタム戦略"""
        import numpy as np

        signals = {}

        for symbol, data in lookback_data.items():
            if len(data) >= 50:
                prices = data["Close"].values

                # 市場状況判定（トレンド vs レンジ）
                returns = np.diff(prices) / prices[:-1]
                recent_returns = returns[-20:]

                # トレンドの一貫性
                trend_consistency = np.corrcoef(
                    np.arange(len(recent_returns)), recent_returns
                )[0, 1]
                trend_consistency = (
                    0 if np.isnan(trend_consistency) else trend_consistency
                )

                # ボラティリティレジーム
                short_vol = np.std(returns[-10:]) if len(returns) >= 10 else 0
                long_vol = np.std(returns[-30:]) if len(returns) >= 30 else 0
                vol_regime = short_vol / long_vol if long_vol > 0 else 1

                # パフォーマンス
                performance = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0

                # 適応的ウェイト計算
                if abs(trend_consistency) > 0.3 and vol_regime < 1.5:  # トレンド相場
                    if performance > 0.02:  # ポジティブモメンタム
                        signals[symbol] = 0.2
                    elif performance < -0.02:
                        signals[symbol] = 0.05
                    else:
                        signals[symbol] = 0.1
                else:  # レンジ相場
                    # 平均回帰的アプローチ
                    sma_20 = np.mean(prices[-20:])
                    deviation = (prices[-1] - sma_20) / sma_20

                    if deviation < -0.05:  # 割安
                        signals[symbol] = 0.15
                    elif deviation > 0.05:  # 割高
                        signals[symbol] = 0.08
                    else:
                        signals[symbol] = 0.12

        # ウェイト正規化
        total_weight = sum(signals.values())
        if total_weight > 0:
            signals = {k: v / total_weight for k, v in signals.items()}

        return signals

    return {
        "ML強化モメンタム戦略": ml_momentum_strategy,
        "リスクパリティ戦略": risk_parity_strategy,
        "適応的モメンタム戦略": adaptive_momentum_strategy,
    }


def test_backtest_engine():
    """バックテストエンジンテスト"""
    print("\n=== バックテストエンジンテスト ===")

    try:
        engine = BacktestEngine(initial_capital=1000000)

        # テスト対象銘柄
        test_symbols = [
            "7203.T",  # トヨタ自動車
            "8306.T",  # 三菱UFJフィナンシャル・グループ
            "9984.T",  # ソフトバンクグループ
            "6758.T",  # ソニーグループ
            "9432.T",  # 日本電信電話
        ]

        # 過去半年のデータでテスト
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        print(
            f"テスト期間: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"対象銘柄: {len(test_symbols)}銘柄")

        # データ取得
        print("過去データ取得中...")
        historical_data = engine.load_historical_data(
            test_symbols, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        if not historical_data:
            print("[FAILED] データ取得失敗")
            return False

        print(f"データ取得成功: {len(historical_data)}銘柄")

        # シンプル戦略でテスト
        def test_strategy(lookback_data, current_prices):
            """テスト用等重み戦略"""
            symbols = list(lookback_data.keys())
            weight = 1.0 / len(symbols) if symbols else 0
            return {symbol: weight for symbol in symbols}

        # バックテスト実行
        print("バックテスト実行中...")
        start_time = time.time()
        results = engine.execute_backtest(
            historical_data, test_strategy, rebalance_frequency=5
        )
        execution_time = time.time() - start_time

        print(f"実行時間: {execution_time:.2f}秒")

        # 結果検証
        if results.final_value > 0:
            print("[OK] バックテスト実行成功")
            print(f"  初期資本: {results.initial_capital:,.0f}円")
            print(f"  最終価値: {results.final_value:,.0f}円")
            print(f"  総リターン: {results.total_return:.2%}")
            print(f"  年率リターン: {results.annualized_return:.2%}")
            print(f"  取引数: {results.total_trades}回")
            return True
        else:
            print("[FAILED] バックテスト結果異常")
            return False

    except Exception as e:
        print(f"[ERROR] バックテストエンジンテストエラー: {e}")
        return False


def test_risk_metrics():
    """リスクメトリクステスト"""
    print("\n=== リスクメトリクステスト ===")

    try:
        calculator = RiskMetricsCalculator()

        # サンプルデータ生成
        import numpy as np

        np.random.seed(42)

        n_days = 180
        returns = np.random.normal(0.0005, 0.015, n_days)  # 日次リターン
        portfolio_values = [1000000]

        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))

        # ベンチマークリターン
        benchmark_returns = np.random.normal(0.0003, 0.012, n_days)

        print("リスクメトリクス計算中...")
        start_time = time.time()
        metrics = calculator.calculate_metrics(
            returns.tolist(), portfolio_values, benchmark_returns.tolist()
        )
        calculation_time = time.time() - start_time

        print(f"計算時間: {calculation_time:.3f}秒")

        # 結果検証
        if metrics.annualized_return is not None:
            print("[OK] リスクメトリクス計算成功")
            print(f"  年率リターン: {metrics.annualized_return:.2%}")
            print(f"  ボラティリティ: {metrics.volatility:.2%}")
            print(f"  シャープレシオ: {metrics.sharpe_ratio:.3f}")
            print(f"  最大ドローダウン: {metrics.maximum_drawdown:.2%}")
            print(f"  VaR(95%): {metrics.var_95:.2%}")

            # レポート生成テスト
            report = calculator.generate_risk_report(metrics, "TOPIX")
            if len(report) > 100:  # レポートが適切な長さ
                print(f"[OK] レポート生成成功: {len(report)}文字")
                return True
            else:
                print("[FAILED] レポート生成失敗")
                return False
        else:
            print("[FAILED] リスクメトリクス計算失敗")
            return False

    except Exception as e:
        print(f"[ERROR] リスクメトリクステストエラー: {e}")
        return False


def test_strategy_evaluator():
    """戦略評価テスト"""
    print("\n=== 戦略評価テスト ===")

    try:
        # データ準備
        engine = BacktestEngine()
        test_symbols = ["7203.T", "8306.T", "9984.T"]

        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)  # 4ヶ月

        historical_data = engine.load_historical_data(
            test_symbols, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        if not historical_data:
            print("[SKIP] テストデータ不足のため戦略評価をスキップ")
            return True

        # 戦略取得
        basic_strategies = create_sample_strategies()
        advanced_strategies = create_advanced_strategies()
        all_strategies = {**basic_strategies, **advanced_strategies}

        print(f"評価戦略数: {len(all_strategies)}戦略")
        print(
            f"テストデータ: {len(historical_data)}銘柄, {len(list(historical_data.values())[0])}日分"
        )

        # 評価実行
        evaluator = StrategyEvaluator(initial_capital=1000000)

        print("戦略評価実行中...")
        start_time = time.time()
        evaluation = evaluator.evaluate_strategies(
            all_strategies,
            historical_data,
            parallel_execution=False,  # 安定性のため順次実行
        )
        evaluation_time = time.time() - start_time

        print(f"評価時間: {evaluation_time:.2f}秒")

        # 結果検証
        if evaluation.strategies:
            print("[OK] 戦略評価成功")
            print(f"  評価完了戦略: {len(evaluation.strategies)}戦略")
            print(f"  最優秀戦略: {evaluation.best_strategy}")

            if evaluation.strategies:
                best = evaluation.strategies[0]
                print(f"  最高スコア: {best.score:.1f}/100")
                print(f"  最高年率リターン: {best.risk_metrics.annualized_return:.2%}")

            # レポート生成テスト
            report = evaluator.generate_evaluation_report(evaluation)
            if len(report) > 200:
                print(f"[OK] 評価レポート生成成功: {len(report)}文字")
                return True
            else:
                print("[FAILED] 評価レポート生成失敗")
                return False
        else:
            print("[FAILED] 戦略評価失敗")
            return False

    except Exception as e:
        print(f"[ERROR] 戦略評価テストエラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メイン実行"""
    print("統合バックテストシステムの総合テストを開始します")

    test_results = []

    # 個別機能テスト
    tests = [
        ("バックテストエンジン", test_backtest_engine),
        ("リスクメトリクス", test_risk_metrics),
        ("戦略評価", test_strategy_evaluator),
    ]

    for test_name, test_function in tests:
        print(f"\n{'='*60}")
        try:
            success = test_function()
            test_results.append((test_name, success))
        except Exception as e:
            print(f"[CRITICAL ERROR] {test_name}テストで予期しないエラー: {e}")
            test_results.append((test_name, False))

    # 最終結果
    print(f"\n{'='*60}")
    print("統合バックテストシステム総合テスト結果")
    print(f"{'='*60}")

    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)

    for test_name, success in test_results:
        status = "[OK]" if success else "[NG]"
        print(f"  {status} {test_name}")

    success_rate = passed / total if total > 0 else 0
    print(f"\n成功率: {passed}/{total} ({success_rate:.1%})")

    if success_rate >= 0.8:
        print("\n[SUCCESS] 統合バックテストシステム: 準備完了")
        print("実データでの戦略検証が可能になりました")
        print("\n推奨次ステップ:")
        print("  1. 長期間（1年以上）でのバックテスト実行")
        print("  2. より多くの銘柄（20-50銘柄）での検証")
        print("  3. カスタム戦略の開発・評価")
        print("  4. リアルタイム取引システムとの統合")

        # 結果保存
        result_data = {
            "test_date": datetime.now().isoformat(),
            "system": "integrated_backtest_system",
            "test_results": [
                {"name": name, "passed": passed} for name, passed in test_results
            ],
            "success_rate": success_rate,
            "status": "READY" if success_rate >= 0.8 else "NEEDS_IMPROVEMENT",
        }

        with open("backtest_system_test_results.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        return True
    else:
        print("\n[FAILED] 一部機能に問題があります")
        print("問題の解決後に再テストしてください")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
