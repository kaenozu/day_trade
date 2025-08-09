"""
Safe Mode System Test (ASCII only)

Check:
1. Safe mode is enabled
2. Automatic trading is completely disabled
3. Only analysis functionality works
4. Safety settings are enforced
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import contextlib

from src.day_trade.automation.analysis_only_engine import AnalysisOnlyEngine
from src.day_trade.automation.trading_engine import TradingEngine
from src.day_trade.config.trading_mode_config import (
    get_current_trading_config,
    is_safe_mode,
)


def test_safe_mode_configuration():
    """Test safe mode configuration"""
    print("=" * 60)
    print("Safe Mode Configuration Test")
    print("=" * 60)

    config = get_current_trading_config()
    safe_mode = is_safe_mode()

    print(f"Safe mode: {'ENABLED' if safe_mode else 'DISABLED'}")
    print(f"Auto trading: {'DISABLED' if not config.enable_automatic_trading else 'ENABLED'}")
    print(f"Order execution: {'DISABLED' if not config.enable_order_execution else 'ENABLED'}")
    print(f"Order API: {'DISABLED' if config.disable_order_api else 'ENABLED'}")
    print(f"Manual confirmation: {'REQUIRED' if config.require_manual_confirmation else 'NOT REQUIRED'}")

    validation = config.validate_configuration()
    print("\nConfiguration validation:")
    for key, value in validation.items():
        status = "PASS" if value else "FAIL"
        print(f"  {status}: {key}")

    assert safe_mode, "Safe mode is not enabled"
    assert not config.enable_automatic_trading, "Automatic trading is enabled"
    assert not config.enable_order_execution, "Order execution is enabled"
    assert config.disable_order_api, "Order API is enabled"
    assert config.require_manual_confirmation, "Manual confirmation is not required"

    print("\n[OK] Safe mode configuration test: PASSED")


def test_trading_engine_safety():
    """Test TradingEngine safety"""
    print("\n" + "=" * 60)
    print("TradingEngine Safety Test")
    print("=" * 60)

    test_symbols = ["7203", "8306", "9984"]

    try:
        engine = TradingEngine(test_symbols)

        status = engine.get_status()
        print(f"Safe mode: {'ENABLED' if status['safe_mode'] else 'DISABLED'}")
        print(f"Trading disabled: {'YES' if status['trading_disabled'] else 'NO'}")
        print(f"Monitored symbols: {status['monitored_symbols']}")

        assert status['safe_mode'], "TradingEngine is not in safe mode"
        assert status['trading_disabled'], "Trading functionality is enabled"

        print("\n[OK] TradingEngine safety test: PASSED")

    except ValueError as e:
        if "safe mode" in str(e).lower():
            print("\n[OK] TradingEngine safety check: Works correctly (initialization rejected)")
        else:
            raise e


def test_analysis_only_engine():
    """Test analysis-only engine"""
    print("\n" + "=" * 60)
    print("Analysis-Only Engine Test")
    print("=" * 60)

    test_symbols = ["7203", "8306"]

    try:
        engine = AnalysisOnlyEngine(test_symbols, update_interval=5.0)

        status = engine.get_status()
        print(f"Engine status: {status['status']}")
        print(f"Monitored symbols: {status['monitored_symbols']}")
        print(f"Safe mode: {'ENABLED' if status['safe_mode'] else 'DISABLED'}")
        print(f"Trading disabled: {'YES' if status['trading_disabled'] else 'NO'}")

        assert status['safe_mode'], "Analysis engine is not in safe mode"
        assert status['trading_disabled'], "Trading functionality is enabled"

        recommendations = engine.get_symbol_recommendations("7203")
        print("\n7203 recommendations:")
        for rec in recommendations[:3]:  # First 3 only
            print(f"  - {rec}")

        print("\n[OK] Analysis-only engine test: PASSED")

    except Exception as e:
        print(f"\n[FAIL] Analysis-only engine test: ERROR - {e}")
        raise e


async def test_analysis_engine_operation():
    """Test analysis engine operation (short run)"""
    print("\n" + "=" * 60)
    print("Analysis Engine Operation Test")
    print("=" * 60)

    test_symbols = ["7203"]

    try:
        engine = AnalysisOnlyEngine(test_symbols, update_interval=2.0)

        print("Running analysis engine for 5 seconds...")

        start_task = asyncio.create_task(engine.start())

        await asyncio.sleep(5.0)

        await engine.stop()

        status = engine.get_status()
        print(f"Total analyses: {status['stats']['total_analyses']}")
        print(f"Successful analyses: {status['stats']['successful_analyses']}")
        print(f"Failed analyses: {status['stats']['failed_analyses']}")

        latest_report = engine.get_latest_report()
        if latest_report:
            print("Latest report:")
            print(f"  - Analyzed symbols: {latest_report.analyzed_symbols}")
            print(f"  - Strong signals: {latest_report.strong_signals}")
            print(f"  - Market sentiment: {latest_report.market_sentiment}")
            print(f"  - Analysis time: {latest_report.analysis_time_ms:.1f}ms")

        summary = engine.get_market_summary()
        print(f"\nMarket summary keys: {list(summary.keys())}")

        print("\n[OK] Analysis engine operation test: PASSED")

        if not start_task.done():
            start_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await start_task

    except Exception as e:
        print(f"\n[FAIL] Analysis engine operation test: ERROR - {e}")
        raise e


def test_system_security():
    """Test system security"""
    print("\n" + "=" * 60)
    print("System Security Test")
    print("=" * 60)

    config = get_current_trading_config()

    security_checks = {
        "Auto trading disabled": not config.enable_automatic_trading,
        "Order execution disabled": not config.enable_order_execution,
        "Order API disabled": config.disable_order_api,
        "Manual confirmation required": config.require_manual_confirmation,
        "All activity logging enabled": config.log_all_activities,
        "Analysis function enabled": config.enable_analysis,
        "Market data enabled": config.enable_market_data,
    }

    print("Security check results:")
    all_passed = True
    for check, passed in security_checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {check}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[OK] System security test: PASSED")
    else:
        print("\n[FAIL] System security test: FAILED")
        raise AssertionError("Security requirements not met")


async def main():
    """Main test execution"""
    print("Automatic Trading Disabling System Test")
    print("=" * 80)

    try:
        test_safe_mode_configuration()
        test_trading_engine_safety()
        test_analysis_only_engine()
        await test_analysis_engine_operation()
        test_system_security()

        print("\n" + "=" * 80)
        print("SUCCESS: All tests passed!")
        print("[OK] Automatic trading functions are completely disabled")
        print("[OK] System operates in safe mode")
        print("[OK] Only analysis functions are enabled, no trading execution")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"FAILURE: Test failed - {e}")
        print("=" * 80)
        raise e


if __name__ == "__main__":
    asyncio.run(main())
