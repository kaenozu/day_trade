"""デバッグ用テストファイル"""

import pandas as pd
from src.day_trade.analysis.screener_enhanced import EnhancedStockScreener, ScreenerCondition, ScreenerCriteria
from src.day_trade.analysis.screening_strategies import ScreeningStrategyFactory

# ストラテジーファクトリーの確認
factory = ScreeningStrategyFactory()
print("利用可能な戦略:")
for name, strategy in factory.get_all_strategies().items():
    print(f"  {name}: {strategy}")

# RSI_OVERSOLD戦略を直接テスト
strategy = factory.get_strategy("RSI_OVERSOLD")
print(f"\nRSI_OVERSOLD戦略: {strategy}")

if strategy:
    # テストデータ
    df = pd.DataFrame({
        'Close': [100] * 50,
        'Volume': [1000] * 50,
        'High': [105] * 50,
        'Low': [95] * 50
    })

    indicators = pd.DataFrame({
        'RSI': [25] * 50,  # RSI過売り
        'SMA_20': [100] * 50,
        'SMA_50': [100] * 50
    })

    meets_condition, score = strategy.evaluate(df, indicators, threshold=30.0)
    print(f"条件満足: {meets_condition}, スコア: {score}")

# ScreenerConditionの値を確認
condition = ScreenerCondition.RSI_OVERSOLD
print(f"\nScreenerCondition.RSI_OVERSOLD.value: {condition.value}")

# 戦略の取得を再テスト
strategy2 = factory.get_strategy(condition.value)
print(f"戦略の取得結果: {strategy2}")
