#!/usr/bin/env python3
"""
簡易統合テスト - Issue #619対応確認
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

def test_basic_integration():
    """基本統合テスト"""
    print("=== 簡易統合テスト開始 ===")
    
    # テストデータ生成
    data = pd.DataFrame({
        'Close': [100, 101, 102, 101, 103, 104, 103, 105, 106, 105]
    })
    
    # 統合システムテスト
    from day_trade.analysis.technical_indicators_consolidated import TechnicalIndicatorsManager
    manager = TechnicalIndicatorsManager()
    
    results = manager.calculate_indicators(data, ["sma", "rsi"], ["Close"])
    print(f"統合システム: {len(list(results.values())[0])}指標計算完了")
    
    # 後方互換性テスト
    from day_trade.analysis.advanced_technical_indicators import AdvancedTechnicalIndicators
    old_system = AdvancedTechnicalIndicators()
    
    sma_result = old_system.calculate_sma(data, 5)
    print(f"後方互換性: SMA計算({len(sma_result)}件)完了")
    
    # ポートフォリオシステムテスト
    from day_trade.portfolio.technical_indicators import calculate_sma, get_consolidation_info
    
    sma_portfolio = calculate_sma(data, 5)
    consolidation_info = get_consolidation_info()
    print(f"ポートフォリオ統合: {consolidation_info['status']}")
    
    print("[成功] 全システム正常動作確認")
    return True

if __name__ == "__main__":
    try:
        test_basic_integration()
        print("✓ Issue #619統合テスト完了")
    except Exception as e:
        print(f"✗ テストエラー: {e}")
        sys.exit(1)