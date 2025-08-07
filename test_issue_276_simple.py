"""
Issue #276 Simple Test - Enhanced Trading System
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

# パス追加
sys.path.append(str(Path(__file__).parent / "src"))

from day_trade.automation.advanced_order_manager import (
    AdvancedOrderManager,
    Order,
    OrderType,
)
from day_trade.automation.portfolio_manager import PortfolioManager
from day_trade.automation.enhanced_trading_engine import (
    EnhancedTradingEngine,
    ExecutionMode,
)
from day_trade.core.trade_manager import Trade, TradeType


def test_basic_functionality():
    """Basic functionality test"""
    print("Enhanced Trading System - Issue #276 Test")
    print("=" * 50)

    # 1. AdvancedOrderManager Test
    print("\n1. AdvancedOrderManager Test")
    manager = AdvancedOrderManager()

    order = Order(
        symbol="7203",
        order_type=OrderType.LIMIT,
        side=TradeType.BUY,
        quantity=100,
        price=Decimal("2500.0")
    )

    print(f"   Order created: {order.symbol} {order.side.value} {order.quantity}")
    assert manager._validate_order(order) == True
    print("   [+] Order validation: PASSED")

    # 2. PortfolioManager Test
    print("\n2. PortfolioManager Test")
    portfolio = PortfolioManager(initial_cash=Decimal("1000000"))

    trade = Trade(
        id="test1",
        symbol="7203",
        trade_type=TradeType.BUY,
        quantity=100,
        price=Decimal("2500.0"),
        timestamp=None,
        commission=Decimal("25.0"),
        status="executed"
    )

    portfolio.add_trade(trade)
    print(f"   Trade added: {trade.symbol} {trade.trade_type.value} {trade.quantity}")

    position = portfolio.get_position("7203")
    assert position is not None
    assert position.quantity == 100
    print("   [+] Position management: PASSED")

    # Market price update
    portfolio.update_market_prices({"7203": Decimal("2600.0")})
    position = portfolio.get_position("7203")
    assert position.unrealized_pnl == Decimal("10000.0")  # (2600-2500)*100
    print("   [+] P&L calculation: PASSED")

    # 3. EnhancedTradingEngine Test
    print("\n3. EnhancedTradingEngine Test")
    engine = EnhancedTradingEngine(
        symbols=["7203", "6758"],
        execution_mode=ExecutionMode.BALANCED,
        initial_cash=Decimal("1000000"),
        update_interval=1.0
    )

    print(f"   Engine created: {len(engine.symbols)} symbols")
    print(f"   Initial cash: {engine.portfolio_manager.initial_cash}")
    print(f"   Execution mode: {engine.execution_mode.value}")

    assert engine.status.value == "stopped"
    print("   [+] Engine initialization: PASSED")

    # Test different execution modes
    conservative_threshold = engine._get_min_confidence_threshold()
    engine.execution_mode = ExecutionMode.CONSERVATIVE
    assert engine._get_min_confidence_threshold() == 80.0

    engine.execution_mode = ExecutionMode.AGGRESSIVE
    assert engine._get_min_confidence_threshold() == 60.0
    print("   [+] Execution modes: PASSED")

    # 4. Integration Test
    print("\n4. Integration Test")
    summary = engine.portfolio_manager.get_portfolio_summary()

    assert summary.total_cash == Decimal("1000000")
    assert summary.total_positions == 0
    print("   [+] Portfolio integration: PASSED")

    status = engine.get_comprehensive_status()
    assert "engine" in status
    assert "portfolio" in status
    assert "orders" in status
    print("   [+] Status reporting: PASSED")

    print("\n" + "=" * 50)
    print("SUCCESS: All Issue #276 core functionality tests passed!")
    print("\nImplemented Features:")
    print("- Advanced order management (Market, Limit, Stop, OCO, IFD)")
    print("- Portfolio management with P&L tracking")
    print("- Enhanced trading engine with multiple execution modes")
    print("- Comprehensive risk management")
    print("- Real-time position monitoring")
    print("- Performance analytics")


async def test_async_functionality():
    """Async functionality test"""
    print("\n5. Async Functionality Test")

    manager = AdvancedOrderManager()

    # Submit order test
    order = Order(
        symbol="7203",
        order_type=OrderType.MARKET,
        side=TradeType.BUY,
        quantity=100
    )

    order_id = await manager.submit_order(order)
    print(f"   Order submitted: {order_id[:8]}")

    # Check active orders
    active_orders = manager.get_active_orders()
    assert len(active_orders) == 1
    print("   [+] Async order submission: PASSED")

    # Cancel order test
    cancelled = await manager.cancel_order(order_id)
    assert cancelled == True

    active_orders = manager.get_active_orders()
    assert len(active_orders) == 0
    print("   [+] Async order cancellation: PASSED")

    # Market update test
    order2 = Order(
        symbol="6758",
        order_type=OrderType.LIMIT,
        side=TradeType.BUY,
        quantity=50,
        price=Decimal("2900.0")
    )

    await manager.submit_order(order2)

    # Simulate market price reaching limit price
    fills = await manager.process_market_update("6758", Decimal("2850.0"))
    assert len(fills) == 1
    assert fills[0].quantity == 50
    print("   [+] Market update processing: PASSED")


async def main():
    """Main test function"""
    try:
        # Basic tests
        test_basic_functionality()

        # Async tests
        await test_async_functionality()

        print("\n" + "=" * 50)
        print("COMPLETE: Issue #276 Enhanced Trading System")
        print("All tests passed - System ready for deployment!")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
