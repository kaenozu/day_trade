#!/usr/bin/env python3
"""
デイトレードエンジンデモ

Issue #849対応: メインコードからデモを分離
"""

import asyncio
import sys
from pathlib import Path

# パス設定
sys.path.append(str(Path(__file__).parent.parent))

from day_trading_engine import (
    PersonalDayTradingEngine,
    DayTradingSignal,
    create_day_trading_engine
)

async def demo_daytrading_engine():
    """デイトレードエンジンデモ"""
    print("=== Day Trade Personal - デイトレード推奨エンジン（改善版） ===")

    engine = create_day_trading_engine()

    # 現在の時間帯アドバイス
    print(engine.get_session_advice())
    print()

    # 今日のデイトレード推奨
    print("今日のデイトレード推奨 TOP5:")
    print("-" * 50)

    recommendations = await engine.get_today_daytrading_recommendations(limit=5)

    for i, rec in enumerate(recommendations, 1):
        signal_icon = {
            DayTradingSignal.STRONG_BUY: "[強い買い]",
            DayTradingSignal.BUY: "[●買い●]",
            DayTradingSignal.STRONG_SELL: "[▼強い売り▼]",
            DayTradingSignal.SELL: "[▽売り▽]",
            DayTradingSignal.HOLD: "[■ホールド■]",
            DayTradingSignal.WAIT: "[…待機…]"
        }.get(rec.signal, "[?]")

        print(f"{i}. {rec.symbol} ({rec.name})")
        print(f"   シグナル: {signal_icon}")
        print(f"   エントリー: {rec.entry_timing}")
        print(f"   目標利確: +{rec.target_profit}% / 損切り: -{rec.stop_loss}%")
        print(f"   保有時間: {rec.holding_time}")
        print(f"   信頼度: {rec.confidence:.0f}% | リスク: {rec.risk_level}")
        print(f"   出来高: {rec.volume_trend} | 値動き: {rec.price_momentum}")
        print(f"   タイミングスコア: {rec.market_timing_score:.0f}/100")
        print()

    print("=== デモ完了 ===")

if __name__ == "__main__":
    asyncio.run(demo_daytrading_engine())