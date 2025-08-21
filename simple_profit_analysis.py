#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

def simple_profit_analysis():
    try:
        response = requests.get('http://localhost:5000/api/recommendations-with-timing')
        data = response.json()

        recommendations = data['recommendations']
        buy_recs = [r for r in recommendations if r['recommendation'] in ['BUY', 'STRONG_BUY']]

        print("=" * 60)
        print("スイングトレード収益分析")
        print("=" * 60)

        total_capital = 1000000  # 100万円
        total_investment = 0
        total_expected = 0

        print(f"投資資金: {total_capital:,}円")
        print(f"買い推奨: {len(buy_recs)}銘柄")
        print()

        print("銘柄別期待収益:")
        print("-" * 60)

        for stock in buy_recs[:8]:  # 上位8銘柄
            price = stock.get('price', 0)
            expected_return = stock.get('expected_return', 10.0)
            confidence = stock.get('confidence', 0.7)

            if price <= 0:
                continue

            # 10万円ずつ投資と仮定
            investment = 100000
            shares = int(investment / price / 100) * 100  # 100株単位
            actual_investment = shares * price
            expected_profit = actual_investment * (expected_return / 100)

            total_investment += actual_investment
            total_expected += expected_profit

            print(f"{stock['symbol']} {stock['name'][:20]:20s}")
            print(f"  投資額: {actual_investment:8,.0f}円 ({shares}株)")
            print(f"  期待利益: {expected_profit:6,.0f}円 ({expected_return}%)")
            print(f"  信頼度: {confidence:.1%}")
            print()

        print("-" * 60)
        print("投資サマリー:")
        print(f"  総投資額: {total_investment:,.0f}円")
        print(f"  期待利益: {total_expected:,.0f}円")

        if total_investment > 0:
            return_rate = (total_expected / total_investment) * 100
            print(f"  期待利回り: {return_rate:.1f}%")

            # 時期別予想
            print()
            print("時期別収益予想:")
            print(f"  1ヶ月後: +{total_expected * 0.3:,.0f}円")
            print(f"  3ヶ月後: +{total_expected * 0.7:,.0f}円")
            print(f"  6ヶ月後: +{total_expected * 1.0:,.0f}円")

        print()
        print("※これは分析予想であり、実際の結果とは異なる場合があります")

    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    simple_profit_analysis()