#!/usr/bin/env python3
"""Debug enum values"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.config.trading_mode_config import TradingMode

print("=== TradingMode Enum Debug ===")
for mode in TradingMode:
    print(f"{mode.name} = {mode.value!r}")

print(f"\nTradingMode.ANALYSIS_ONLY.value = {TradingMode.ANALYSIS_ONLY.value!r}")
print(f"TradingMode.ANALYSIS_ONLY == 'analysis_only': {TradingMode.ANALYSIS_ONLY == 'analysis_only'}")
print(f"TradingMode.ANALYSIS_ONLY.value == 'analysis_only': {TradingMode.ANALYSIS_ONLY.value == 'analysis_only'}")
