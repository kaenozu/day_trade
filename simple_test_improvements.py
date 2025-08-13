#!/usr/bin/env python3
"""
Issue #597-599改善の簡単テスト
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# プロジェクトパスを追加
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from day_trade.analysis.technical_indicators_consolidated import (
    TechnicalIndicatorsManager,
    IndicatorConfig
)

def simple_test():
    """基本動作テスト"""
    print("=== 基本動作テスト ===")

    manager = TechnicalIndicatorsManager()

    # テストデータ作成
    np.random.seed(42)
    data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(50) * 0.5)
    })

    # 基本指標テスト
    indicators = ["sma", "ema", "rsi"]
    results = manager.calculate_indicators(data, indicators, ["Close"])

    print(f"計算した指標数: {len(list(results.values())[0])}")

    for symbol_results in results.values():
        for result in symbol_results:
            print(f"指標: {result.name}, 実装: {result.implementation_used}")

    perf = manager.get_performance_summary()
    print(f"パフォーマンス: {perf}")

    return True

if __name__ == "__main__":
    simple_test()