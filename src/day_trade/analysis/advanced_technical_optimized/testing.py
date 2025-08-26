#!/usr/bin/env python3
"""
テスト関連コードモジュール
Issue #315: 高度テクニカル指標・ML機能拡張

統合最適化版高度テクニカル指標システムのテスト機能:
- システムテスト実行
- サンプルデータ生成
- テスト結果表示
"""

import asyncio

import numpy as np
import pandas as pd

from .main import AdvancedTechnicalIndicatorsOptimized

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


async def test_optimized_system() -> None:
    """統合最適化版高度テクニカル指標システムテスト"""
    print("=== 統合最適化版高度テクニカル指標システムテスト ===")

    # テストデータ生成
    test_data = generate_test_data()

    # システム初期化
    analyzer = AdvancedTechnicalIndicatorsOptimized(
        enable_cache=True,
        enable_parallel=True,
        enable_ml_optimization=True,
        max_concurrent=10,
    )

    # Bollinger Bands分析テスト
    print("\n🔍 Bollinger Bands最適化分析テスト...")
    bb_result = await analyzer.analyze_bollinger_bands_optimized(
        test_data, "TEST"
    )
    print(f"シグナル: {bb_result.signal} (信頼度: {bb_result.confidence:.2%})")
    print(f"パフォーマンススコア: {bb_result.performance_score:.3f}")

    # 一目均衡表分析テスト
    print("\n☁️ 一目均衡表最適化分析テスト...")
    ichimoku_result = await analyzer.analyze_ichimoku_cloud_optimized(
        test_data, "TEST"
    )
    print(
        f"総合シグナル: {ichimoku_result.overall_signal} "
        f"(信頼度: {ichimoku_result.confidence:.2%})"
    )
    print(f"雲の位置: {ichimoku_result.price_vs_cloud}")
    print(f"パフォーマンススコア: {ichimoku_result.performance_score:.3f}")

    # バッチ分析テスト
    print("\n⚡ 並列バッチ分析テスト...")
    batch_data = {
        "TEST1": test_data,
        "TEST2": test_data.copy(),
        "TEST3": test_data.copy(),
    }

    batch_results = await analyzer.batch_analyze_symbols(
        batch_data, ["bb", "ichimoku"]
    )
    print(f"バッチ分析完了: {len(batch_results)}銘柄")

    # パフォーマンス統計
    print("\n📊 統合最適化基盤パフォーマンス統計:")
    stats = analyzer.get_optimization_performance_stats()
    print(f"総分析回数: {stats['total_analyses']}")
    print(f"キャッシュヒット率: {stats['cache_hit_rate']:.1%}")
    print(f"平均処理時間: {stats['avg_processing_time_ms']:.1f}ms")

    print("\n🎯 統合最適化効果:")
    benefits = stats["optimization_benefits"]
    for benefit, value in benefits.items():
        print(f"  - {benefit}: {value}")

    print("\n✅ 統合最適化版高度テクニカル指標システムテスト完了")


def generate_test_data(periods: int = 100) -> pd.DataFrame:
    """
    テストデータ生成
    
    Args:
        periods: データ期間数
        
    Returns:
        pd.DataFrame: テスト用株価データ
    """
    dates = pd.date_range(start="2024-01-01", periods=periods)
    
    # ランダムウォークベースの株価データ生成
    base_price = 2000
    returns = np.random.normal(0, 0.02, periods)  # 2%の日次ボラティリティ
    
    prices = [base_price]
    for return_rate in returns[1:]:
        prices.append(prices[-1] * (1 + return_rate))
    
    # OHLCV データ生成
    closes = np.array(prices)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.01, periods)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.01, periods)))
    opens = closes * (1 + np.random.normal(0, 0.005, periods))
    volumes = np.random.randint(500000, 2000000, periods)

    test_data = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )

    return test_data


def generate_trending_test_data(
    periods: int = 100, trend: str = "up"
) -> pd.DataFrame:
    """
    トレンド付きテストデータ生成
    
    Args:
        periods: データ期間数
        trend: トレンド方向 ("up", "down", "sideways")
        
    Returns:
        pd.DataFrame: トレンド付きテスト用株価データ
    """
    dates = pd.date_range(start="2024-01-01", periods=periods)
    
    base_price = 2000
    
    # トレンド成分
    if trend == "up":
        trend_component = np.linspace(0, 0.3, periods)
    elif trend == "down":
        trend_component = np.linspace(0, -0.3, periods)
    else:  # sideways
        trend_component = np.sin(np.linspace(0, 4*np.pi, periods)) * 0.1
    
    # ランダム成分
    random_component = np.random.normal(0, 0.015, periods)
    
    # 価格生成
    returns = trend_component / periods + random_component
    prices = [base_price]
    for return_rate in returns[1:]:
        prices.append(prices[-1] * (1 + return_rate))
    
    # OHLCV データ生成
    closes = np.array(prices)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.008, periods)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.008, periods)))
    opens = closes * (1 + np.random.normal(0, 0.003, periods))
    volumes = np.random.randint(800000, 1500000, periods)

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


async def run_comprehensive_tests() -> None:
    """包括的テスト実行"""
    print("=== 包括的テスト実行 ===")
    
    # 基本テスト
    await test_optimized_system()
    
    # トレンド別テスト
    print("\n=== トレンド別テスト ===")
    
    analyzer = AdvancedTechnicalIndicatorsOptimized()
    
    for trend_type in ["up", "down", "sideways"]:
        print(f"\n📈 {trend_type.upper()}トレンドテスト...")
        trend_data = generate_trending_test_data(trend=trend_type)
        
        bb_result = await analyzer.analyze_bollinger_bands_optimized(
            trend_data, f"TREND_{trend_type.upper()}"
        )
        ichimoku_result = await analyzer.analyze_ichimoku_cloud_optimized(
            trend_data, f"TREND_{trend_type.upper()}"
        )
        
        print(f"  BB: {bb_result.signal} ({bb_result.confidence:.2%})")
        print(f"  一目: {ichimoku_result.overall_signal} ({ichimoku_result.confidence:.2%})")


# テスト実行のためのメイン関数
if __name__ == "__main__":
    # 基本テスト実行
    asyncio.run(test_optimized_system())
    
    # 包括的テスト実行（コメントアウト、必要に応じて有効化）
    # asyncio.run(run_comprehensive_tests())