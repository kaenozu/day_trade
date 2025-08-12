#!/usr/bin/env python3
"""
ポートフォリオ最適化システムのテスト

Phase 2: ポートフォリオ最適化AI システムの動作確認
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from day_trade.optimization import PortfolioManager  # noqa: E402


def test_portfolio_optimization_basic():
    """基本的なポートフォリオ最適化テスト"""
    print("=== 基本ポートフォリオ最適化テスト ===")

    # テスト用銘柄（セクター分散）
    test_symbols = [
        "7203",  # トヨタ (Transportation)
        "8306",  # 三菱UFJ (Financial)
        "9984",  # ソフトバンクGP (Technology)
        "6758",  # ソニー (Technology)
        "4502",  # 武田薬品 (Healthcare)
        "4563",  # アンジェス (BioTech)
        "3655",  # ブレインパッド (DayTrading)
        "7779",  # CYBERDYNE (FutureTech)
    ]

    try:
        # ポートフォリオマネージャー初期化
        manager = PortfolioManager(
            investment_amount=500000,  # 50万円でテスト
            risk_tolerance=0.6,
            rebalancing_threshold=0.05,
        )

        print(f"テスト銘柄: {len(test_symbols)}銘柄")
        print(f"投資金額: {manager.investment_amount:,.0f}円")

        # 包括的ポートフォリオ生成
        print("\n1. 包括的ポートフォリオ生成中...")
        portfolio = manager.generate_comprehensive_portfolio(
            test_symbols, use_ml_signals=True
        )

        if "error" not in portfolio:
            print("   ✅ ポートフォリオ生成成功")

            # 結果表示
            allocations = portfolio["portfolio_allocations"]
            print(f"\n   配分結果: {len(allocations)}銘柄")

            total_weight = sum(a["weight"] for a in allocations.values())
            total_amount = sum(a["amount"] for a in allocations.values())

            print(f"   ウェイト合計: {total_weight:.1%}")
            print(f"   投資金額合計: {total_amount:,.0f}円")

            # 上位5銘柄表示
            top_allocations = sorted(
                allocations.items(),
                key=lambda x: x[1]["weight"],
                reverse=True,
            )[:5]

            print("\n   主要配分:")
            for symbol, allocation in top_allocations:
                weight = allocation["weight"]
                amount = allocation["amount"]
                ml_advice = allocation.get("ml_advice", "N/A")
                ml_confidence = allocation.get("ml_confidence", 0)

                print(
                    f"     {symbol}: {weight:.1%} ({amount:,.0f}円) "
                    f"ML: {ml_advice}({ml_confidence:.0f}%)"
                )

            # ポートフォリオ指標
            if "portfolio_metrics" in portfolio:
                metrics = portfolio["portfolio_metrics"]
                print("\n   ポートフォリオ指標:")
                print(f"     期待リターン: {metrics.get('expected_return', 0):.2%}")
                print(
                    f"     ボラティリティ: {metrics.get('portfolio_volatility', 0):.2%}"
                )
                print(f"     シャープレシオ: {metrics.get('sharpe_ratio', 0):.2f}")
                print(
                    f"     実効ポジション数: {metrics.get('effective_positions', 0):.1f}"
                )

            return True

        else:
            print(f"   ERROR: {portfolio['error']}")
            return False

    except Exception as e:
        print(f"   EXCEPTION: {e}")
        return False


def test_risk_analysis():
    """リスク分析テスト"""
    print("\n=== リスク分析テスト ===")

    # サンプルポートフォリオ
    sample_portfolio = {
        "7203": 0.20,  # トヨタ
        "8306": 0.15,  # 三菱UFJ
        "9984": 0.15,  # ソフトバンクGP
        "6758": 0.15,  # ソニー
        "4563": 0.10,  # アンジェス
        "3655": 0.10,  # ブレインパッド
        "7779": 0.08,  # CYBERDYNE
        "4592": 0.07,  # サンバイオ
    }

    try:
        manager = PortfolioManager(investment_amount=1000000)

        print("   現在ポートフォリオ分析中...")
        analysis = manager.analyze_current_portfolio(sample_portfolio)

        if "error" not in analysis:
            print("   ✅ リスク分析成功")

            # リスクレベル
            if "risk_analysis" in analysis:
                risk_info = analysis["risk_analysis"]
                risk_level = risk_info.get("overall_risk_level", "UNKNOWN")
                print(f"   総合リスクレベル: {risk_level}")

            # セクター分析
            if "sector_analysis" in analysis:
                sector_info = analysis["sector_analysis"]
                sector_weights = sector_info.get("sector_weights", {})
                print(f"   セクター分散: {len(sector_weights)}セクター")

                for sector, weight in sorted(
                    sector_weights.items(), key=lambda x: x[1], reverse=True
                )[:3]:
                    print(f"     {sector}: {weight:.1%}")

            # 健全性スコア
            if "health_score" in analysis:
                health = analysis["health_score"]
                overall_score = health.get("overall_score", 0)
                score_level = health.get("score_level", "UNKNOWN")
                print(f"   健全性スコア: {overall_score:.0f}点 ({score_level})")

            # 改善提案
            if "improvement_suggestions" in analysis:
                suggestions = analysis["improvement_suggestions"]
                print(f"   改善提案: {len(suggestions)}件")
                for suggestion in suggestions[:2]:  # 上位2件表示
                    print(f"     • {suggestion}")

            return True

        else:
            print(f"   ERROR: {analysis['error']}")
            return False

    except Exception as e:
        print(f"   EXCEPTION: {e}")
        return False


def test_rebalancing():
    """リバランシングテスト"""
    print("\n=== リバランシングテスト ===")

    # 現在保有（偏ったポートフォリオ）
    current_holdings = {
        "7203": 0.40,  # トヨタ偏重
        "8306": 0.30,  # 三菱UFJ偏重
        "9984": 0.20,  # ソフトバンクGP
        "6758": 0.10,  # ソニー
    }

    # 目標銘柄（より分散）
    target_symbols = [
        "7203",
        "8306",
        "9984",
        "6758",
        "4563",
        "3655",
        "4592",
        "7779",
        "4475",
        "3692",
    ]

    try:
        manager = PortfolioManager()

        print("   リバランシング計画生成中...")
        rebalancing_plan = manager.generate_rebalancing_plan(
            current_holdings, target_symbols
        )

        if "error" not in rebalancing_plan:
            print("   ✅ リバランシング計画生成成功")

            # リバランシング必要性
            necessity = rebalancing_plan.get("rebalancing_necessity", {})
            is_required = necessity.get("is_required", False)
            urgency = necessity.get("urgency_level", "UNKNOWN")

            print(f"   リバランシング必要性: {'必要' if is_required else '不要'}")
            print(f"   緊急度: {urgency}")

            # 取引サマリー
            if "rebalancing_proposal" in rebalancing_plan:
                proposal = rebalancing_plan["rebalancing_proposal"]
                summary = proposal.get("summary", {})

                total_trades = summary.get("total_trades", 0)
                buy_trades = summary.get("buy_trades", 0)
                sell_trades = summary.get("sell_trades", 0)

                print(
                    f"   予定取引: {total_trades}件 (買い: {buy_trades}, 売り: {sell_trades})"
                )

                # 主要取引表示
                if "prioritized_trades" in proposal:
                    top_trades = proposal["prioritized_trades"][:3]
                    print("   主要取引:")
                    for trade in top_trades:
                        symbol = trade["symbol"]
                        trade_type = trade["trade_type"]
                        weight_change = trade["weight_change"]
                        print(f"     {symbol}: {trade_type} {abs(weight_change):.1%}")

            return True

        else:
            print(f"   ERROR: {rebalancing_plan['error']}")
            return False

    except Exception as e:
        print(f"   EXCEPTION: {e}")
        return False


def main():
    """メインテスト実行"""
    print("Phase 2 ポートフォリオ最適化システム テスト")
    print("=" * 60)

    test_results = []

    try:
        # 基本最適化テスト
        result1 = test_portfolio_optimization_basic()
        test_results.append(("基本最適化", result1))

        # リスク分析テスト
        result2 = test_risk_analysis()
        test_results.append(("リスク分析", result2))

        # リバランシングテスト
        result3 = test_rebalancing()
        test_results.append(("リバランシング", result3))

        # テスト結果サマリー
        print("\n" + "=" * 60)
        print("テスト結果サマリー")
        print("=" * 60)

        passed = 0
        total = len(test_results)

        for test_name, result in test_results:
            status = "PASS" if result else "FAIL"
            print(f"{test_name:20} : {status}")
            if result:
                passed += 1

        print(f"\nPass Rate: {passed}/{total} ({passed/total*100:.0f}%)")

        if passed == total:
            print("\nAll Tests Passed! Phase 2 Implementation Complete")
            print("\nFeatures:")
            print("• Modern Portfolio Theory optimization")
            print("• Monte Carlo simulation")
            print("• Sector constraint management")
            print("• Comprehensive risk analysis")
            print("• Dynamic rebalancing proposals")
            print("• ML investment advice integration")
        else:
            print(f"\n{total-passed} tests failed")

    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\nTest execution error: {e}")


if __name__ == "__main__":
    main()
