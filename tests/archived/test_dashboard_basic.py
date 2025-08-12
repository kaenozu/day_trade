"""
Basic Analysis Dashboard Test (ASCII only)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.config.trading_mode_config import (
    get_current_trading_config,
    is_safe_mode,
)


def test_basic_configuration():
    """Basic configuration test"""
    print("=" * 60)
    print("Basic Configuration Test")
    print("=" * 60)

    # Safe mode test
    assert is_safe_mode(), "Safe mode is not enabled"
    print("[OK] Safe mode: ENABLED")

    # Configuration test
    config = get_current_trading_config()
    assert not config.enable_automatic_trading, "Automatic trading is enabled"
    assert not config.enable_order_execution, "Order execution is enabled"
    assert config.disable_order_api, "Order API is enabled"

    print("[OK] Automatic trading: DISABLED")
    print("[OK] Order execution: DISABLED")
    print("[OK] Order API: DISABLED")
    print("[OK] System is configured for analysis only")


def test_dashboard_imports():
    """Test dashboard module imports"""
    print("\n" + "=" * 60)
    print("Dashboard Import Test")
    print("=" * 60)

    try:
        from src.day_trade.dashboard.analysis_dashboard_server import app

        print("[OK] Dashboard server import: SUCCESS")

        # Check app type
        from fastapi import FastAPI

        assert isinstance(app, FastAPI), "App is not FastAPI instance"
        print("[OK] FastAPI app: VALID")

        # Check app title for safety indication
        assert "分析専用" in app.title or "analysis" in app.title.lower()
        print("[OK] App title indicates analysis-only system")

    except ImportError as e:
        print(f"[FAIL] Dashboard import failed: {e}")
        raise


def main():
    """Main test function"""
    print("Analysis Dashboard Basic Test")
    print("=" * 80)

    try:
        test_basic_configuration()
        test_dashboard_imports()

        print("\n" + "=" * 80)
        print("SUCCESS: Basic tests passed!")
        print("[OK] System is properly configured for analysis only")
        print("[OK] Dashboard components are importable")
        print("[OK] Safe mode is properly enforced")
        print()
        print("To start the dashboard:")
        print("  python run_analysis_dashboard.py")
        print("  Access: http://localhost:8000")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"FAILURE: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
