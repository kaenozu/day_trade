#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
収益ポテンシャル分析 - 現在の推奨銘柄での期待収益計算
"""

import requests
import json
from datetime import datetime, timedelta

def analyze_profit_potential():
    """現在の推奨銘柄での収益ポテンシャル分析"""

    try:
        # 推奨銘柄データ取得
        response = requests.get('http://localhost:5000/api/recommendations-with-timing')
        data = response.json()

        if not data.get('recommendations'):
            print("❌ 推奨データの取得に失敗しました")
            return

        recommendations = data['recommendations']
        buy_recommendations = [r for r in recommendations if r['recommendation'] in ['BUY', 'STRONG_BUY']]

        print("=" * 80)
        print("スイングトレード収益ポテンシャル分析")
        print("=" * 80)
        print(f"分析時点: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        print(f"買い推奨銘柄: {len(buy_recommendations)}銘柄")
        print(f"総分析銘柄: {len(recommendations)}銘柄")
        print()

        # 投資シナリオ設定
        scenarios = [
            {"name": "少額分散投資", "total_capital": 500000, "risk_level": "低"},
            {"name": "標準ポートフォリオ", "total_capital": 1000000, "risk_level": "中"},
            {"name": "積極投資", "total_capital": 2000000, "risk_level": "高"}
        ]

        for scenario in scenarios:
            print(f"💰 【{scenario['name']}】 資金: {scenario['total_capital']:,}円")
            print("-" * 60)

            total_investment = 0
            total_expected_profit = 0
            total_max_loss = 0
            position_count = 0

            # 各推奨銘柄に対する投資配分
            for stock in buy_recommendations:
                if total_investment >= scenario['total_capital']:
                    break

                # 投資配分計算（信頼度とリスクレベルに基づく）
                confidence = stock.get('confidence', 0.7)
                expected_return = stock.get('expected_return', 10.0)
                current_price = stock.get('price', 0)

                if current_price <= 0:
                    continue

                # リスクレベル別の配分率
                if scenario['risk_level'] == "低":
                    allocation_rate = min(0.05, confidence * 0.08)  # 最大5%
                elif scenario['risk_level'] == "中":
                    allocation_rate = min(0.10, confidence * 0.12)  # 最大10%
                else:  # 高
                    allocation_rate = min(0.15, confidence * 0.15)  # 最大15%

                investment_amount = scenario['total_capital'] * allocation_rate

                # 100株単位に調整
                shares = int(investment_amount / current_price / 100) * 100
                if shares <= 0:
                    continue

                actual_investment = shares * current_price
                expected_profit = actual_investment * (expected_return / 100)
                max_loss = actual_investment * 0.05  # 5%ストップロス想定

                total_investment += actual_investment
                total_expected_profit += expected_profit
                total_max_loss += max_loss
                position_count += 1

                print(f"  {stock['symbol']} {stock['name'][:15]:15s} | "
                      f"{shares:3d}株 {actual_investment:8,.0f}円 | "
                      f"期待: +{expected_profit:6,.0f}円 ({expected_return:4.1f}%)")

            # サマリー計算
            expected_return_rate = (total_expected_profit / total_investment * 100) if total_investment > 0 else 0
            max_loss_rate = (total_max_loss / total_investment * 100) if total_investment > 0 else 0

            print("-" * 60)
            print(f"📊 投資サマリー:")
            print(f"   総投資額:     {total_investment:10,.0f}円 ({total_investment/scenario['total_capital']*100:.1f}%)")
            print(f"   ポジション数: {position_count:10d}銘柄")
            print(f"   期待利益:     {total_expected_profit:10,.0f}円 ({expected_return_rate:.1f}%)")
            print(f"   最大損失:     {total_max_loss:10,.0f}円 ({max_loss_rate:.1f}%)")
            print(f"   リスクリターン比: {total_expected_profit/total_max_loss:.1f}倍" if total_max_loss > 0 else "   リスクリターン比: N/A")
            print()

        # 個別銘柄詳細分析
        print("🔍 高期待銘柄詳細分析")
        print("=" * 80)

        # 期待リターン順でソート
        sorted_stocks = sorted(buy_recommendations,
                             key=lambda x: x.get('expected_return', 0) * x.get('confidence', 0),
                             reverse=True)

        for i, stock in enumerate(sorted_stocks[:5], 1):
            confidence = stock.get('confidence', 0.7)
            expected_return = stock.get('expected_return', 10.0)
            current_price = stock.get('price', 0)
            target_price = stock.get('target_price', current_price)

            print(f"#{i} {stock['symbol']} {stock['name']}")
            print(f"    現在価格: ¥{current_price:,.0f}")
            print(f"    目標価格: ¥{target_price:,.0f}")
            print(f"    期待リターン: {expected_return:.1f}%")
            print(f"    信頼度: {confidence:.1%}")
            print(f"    買いタイミング: {stock.get('buy_timing', 'N/A')}")
            print(f"    売りタイミング: {stock.get('sell_timing', 'N/A')}")
            print(f"    戦略: {stock.get('strategy_type', 'N/A')}")
            print()

        # 時期別収益予想
        print("📅 時期別収益予想")
        print("=" * 80)

        # 短期・中期・長期での収益予想
        time_horizons = {
            "1ヶ月後": 0.3,   # 期待リターンの30%
            "3ヶ月後": 0.7,   # 期待リターンの70%
            "6ヶ月後": 1.0,   # 期待リターンの100%
        }

        for horizon, multiplier in time_horizons.items():
            projected_profit = total_expected_profit * multiplier
            projected_rate = expected_return_rate * multiplier

            print(f"{horizon:8s}: +{projected_profit:8,.0f}円 ({projected_rate:4.1f}%)")

        print()
        print("⚠️  注意事項:")
        print("   • これは過去のテクニカル分析に基づく予想であり、将来の結果を保証するものではありません")
        print("   • 実際の市場は予想外の変動があり、損失が発生する可能性があります")
        print("   • 分散投資とリスク管理を心がけてください")
        print("   • 投資は自己責任で行ってください")

    except Exception as e:
        print(f"❌ 分析エラー: {e}")

if __name__ == "__main__":
    analyze_profit_potential()