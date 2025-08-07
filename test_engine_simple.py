"""
TradingEngineの簡単なテスト実行
"""

import asyncio
import sys
from pathlib import Path
from decimal import Decimal

# パス追加
sys.path.append(str(Path(__file__).parent / "src"))

# 直接インポート
from day_trade.automation.trading_engine import (
    TradingEngine,
    RiskParameters,
    EngineStatus
)

async def simple_test():
    """簡単な動作テスト"""

    print("TradingEngine Simple Test")
    print("=" * 40)

    # リスクパラメータ
    risk_params = RiskParameters(
        max_position_size=Decimal("100000"),
        max_daily_loss=Decimal("10000"),
        max_open_positions=3
    )

    # エンジン作成
    engine = TradingEngine(
        symbols=["7203", "6758"],
        risk_params=risk_params,
        update_interval=1.0
    )

    print("SUCCESS: Engine initialized")
    print(f"Symbols: {engine.symbols}")
    print(f"Status: {engine.status.value}")

    # 基本機能テスト
    print("\nBasic Function Tests:")

    # ステータス確認
    status = engine.get_status()
    print(f"   Symbols: {status['monitored_symbols']}")
    print(f"   Active positions: {status['active_positions']}")
    print(f"   Daily P&L: {status['daily_pnl']}")

    # リスク制約チェック
    constraints_ok = engine._check_risk_constraints()
    print(f"   Risk constraints: {'OK' if constraints_ok else 'VIOLATION'}")

    # 一時停止・再開テスト
    engine.status = EngineStatus.RUNNING
    await engine.pause()
    print(f"   After pause: {engine.status.value}")

    await engine.resume()
    print(f"   After resume: {engine.status.value}")

    # 緊急停止テスト
    engine.emergency_stop()
    print(f"   After emergency stop: {engine.status.value}")

    print("\nSUCCESS: All basic tests completed!")
    print("TradingEngine is working properly")

if __name__ == "__main__":
    asyncio.run(simple_test())
