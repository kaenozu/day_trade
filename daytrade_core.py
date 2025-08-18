#!/usr/bin/env python3
"""
Day Trade Personal - Core Entry Point

Refactored lightweight main file
"""

import sys
from pathlib import Path

# Add system path
sys.path.append(str(Path(__file__).parent / "src"))

from src.day_trade.core.application import DayTradeApplication
from src.day_trade.core.lightweight_application import LightweightDayTradeApplication

# Alias definitions
DayTradeCore = DayTradeApplication
DayTradeCoreLight = LightweightDayTradeApplication


def main():
    """Main execution function"""
    try:
        app = DayTradeApplication()
        return app.run()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())