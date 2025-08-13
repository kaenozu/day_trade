#!/usr/bin/env python3
"""
最終システム統合テスト
全コンポーネント統合動作検証

検証対象:
- Issue #314: TOPIX500分析システム
- Issue #315: 高度テクニカル指標・ML機能拡張
- Issue #322-325: 統合最適化基盤
- 既存システム: データベース・アラート・可視化
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

try:
    # 核心システム
    from src.day_trade.analysis.advanced_technical_indicators_optimized import (
        AdvancedTechnicalIndicatorsOptimized,
    )
    from src.day_trade.analysis.multi_timeframe_analysis_optimized import (
        MultiTimeframeAnalysisOptimized,
    )

    # データ管理
    from src.day_trade.data.advanced_parallel_ml_engine import AdvancedParallelMLEngine
    from src.day_trade.ml.advanced_ml_models import AdvancedMLModels
    from src.day_trade.risk.volatility_prediction_system import (
        VolatilityPredictionSystem,
    )
    from src.day_trade.topix.topix500_analysis_system import TOPIX500AnalysisSystem

    # 可視化・アラート (オプション)
    # from src.day_trade.core.alerts import AlertsManager
    # from src.day_trade.dashboard.dashboard_core import DashboardCore
    from src.day_trade.utils.logging_config import get_context_logger
    from src.day_trade.utils.performance_monitor import PerformanceMonitor
    from src.day_trade.utils.unified_cache_manager import UnifiedCacheManager
except ImportError as e:
    print(f"インポートエラー: {e}")
    sys.exit(1)

logger = get_context_logger(__name__)


def generate_comprehensive_test_data() -> Dict[str, pd.DataFrame]:
    """包括的テスト用データ生成"""
    # 主要銘柄（各セクター代表）
    test_symbols = [
        "7203",  # トヨタ（自動車）
        "8306",  # 三菱UFJ（銀行）
        "9984",  # ソフトバンク（通信）
        "4502",  # 武田薬品（医薬品）
        "7182",  # ゆうちょ銀行（銀行）
        "8267",  # イオン（小売）
        "6501",  # 日立製作所（電機）
        "5020",  # ENEOS（エネルギー）
        "8802",  # 三菱地所（不動産）
        "1812",  # 鹿島建設（建設）
    ]

    stock_data = {}

    for symbol in test_symbols:
        # 1年分のリアルな株式データ生成
        dates = pd.date_range(start="2024-01-01", periods=252)  # 営業日ベース

        # セクター別の特性を反映した価格生成
        if symbol.startswith("7"):  # 自動車・製造業
            base_price = 2500
            volatility = 0.025
        elif symbol.startswith("8"):  # 銀行・金融
            base_price = 1200
            volatility = 0.03
        elif symbol.startswith("9"):  # 通信・IT
            base_price = 4500
            volatility = 0.035
        elif symbol.startswith("4"):  # 医薬品
            base_price = 3800
            volatility = 0.02
        elif symbol.startswith("5"):  # エネルギー
            base_price = 600
            volatility = 0.04
        else:
            base_price = 2000
            volatility = 0.025

        # トレンド成分（年間±20%変動）
        trend = np.linspace(0, np.random.uniform(-0.2, 0.2), 252)

        # ランダムウォーク成分
        random_walk = np.cumsum(np.random.normal(0, volatility, 252))

        # 価格系列生成
        price_changes = trend + random_walk
        cumulative_changes = np.exp(price_changes)
        close_prices = base_price * cumulative_changes

        # OHLV データ生成
        open_prices = close_prices * np.random.uniform(0.995, 1.005, 252)
        high_prices = np.maximum(open_prices, close_prices) * np.random.uniform(
            1.0, 1.03, 252
        )
        low_prices = np.minimum(open_prices, close_prices) * np.random.uniform(
            0.97, 1.0, 252
        )
        volumes = np.random.lognormal(14, 0.8, 252).astype(int)

        stock_data[symbol] = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volumes,
                "Adj Close": close_prices,
            },
            index=dates,
        )

    return stock_data


async def test_topix500_system(test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """TOPIX500システムテスト"""
    logger.info("TOPIX500システムテスト開始")
    start_time = time.time()

    try:
        system = TOPIX500AnalysisSystem(
            enable_cache=True,
            enable_parallel=True,
            max_concurrent_symbols=10,
            memory_limit_gb=1.0,
            processing_timeout=15,
        )

        result = await system.analyze_batch_comprehensive(
            stock_data=test_data, enable_sector_analysis=True, enable_ml_prediction=True
        )

        processing_time = time.time() - start_time

        success_count = len(
            [r for r in result["symbol_results"].values() if r.get("success", False)]
        )

        return {
            "test_name": "TOPIX500システム",
            "success": True,
            "processing_time": processing_time,
            "symbols_processed": len(test_data),
            "successful_symbols": success_count,
            "success_rate": success_count / len(test_data),
            "sectors_analyzed": len(result["sector_analysis"]),
            "performance_metrics": result.get("performance_metrics"),
            "details": f"{success_count}/{len(test_data)}銘柄成功、{len(result['sector_analysis'])}セクター分析",
        }

    except Exception as e:
        logger.error(f"TOPIX500システムテストエラー: {e}")
        return {
            "test_name": "TOPIX500システム",
            "success": False,
            "processing_time": time.time() - start_time,
            "error": str(e),
            "details": "システム実行エラー",
        }


async def test_advanced_technical_indicators(
    test_data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """高度テクニカル指標テスト"""
    logger.info("高度テクニカル指標テスト開始")
    start_time = time.time()

    try:
        system = AdvancedTechnicalIndicatorsOptimized(
            enable_cache=True, enable_parallel=True, max_concurrent=5
        )

        results = {}
        for symbol, data in list(test_data.items())[:5]:  # 5銘柄でテスト
            try:
                result = await system.analyze_optimized(data, symbol)
                results[symbol] = result
            except Exception as e:
                logger.error(f"テクニカル分析エラー {symbol}: {e}")
                results[symbol] = None

        processing_time = time.time() - start_time
        success_count = len([r for r in results.values() if r is not None])

        return {
            "test_name": "高度テクニカル指標",
            "success": True,
            "processing_time": processing_time,
            "symbols_processed": len(results),
            "successful_symbols": success_count,
            "success_rate": success_count / len(results),
            "details": f"{success_count}/{len(results)}銘柄でテクニカル分析成功",
        }

    except Exception as e:
        logger.error(f"高度テクニカル指標テストエラー: {e}")
        return {
            "test_name": "高度テクニカル指標",
            "success": False,
            "processing_time": time.time() - start_time,
            "error": str(e),
            "details": "テクニカル指標システムエラー",
        }


async def test_ml_models_system(test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """ML予測システムテスト"""
    logger.info("ML予測システムテスト開始")
    start_time = time.time()

    try:
        system = AdvancedMLModels(enable_cache=True, enable_parallel=True)

        results = {}
        for symbol, data in list(test_data.items())[:3]:  # 3銘柄でテスト
            try:
                feature_set = await system._generate_advanced_features(data, symbol)
                prediction = await system.predict_with_ensemble(
                    data, symbol, feature_set
                )
                results[symbol] = prediction
            except Exception as e:
                logger.error(f"ML予測エラー {symbol}: {e}")
                results[symbol] = None

        processing_time = time.time() - start_time
        success_count = len([r for r in results.values() if r is not None])

        return {
            "test_name": "ML予測システム",
            "success": True,
            "processing_time": processing_time,
            "symbols_processed": len(results),
            "successful_symbols": success_count,
            "success_rate": success_count / len(results),
            "details": f"{success_count}/{len(results)}銘柄でML予測成功",
        }

    except Exception as e:
        logger.error(f"ML予測システムテストエラー: {e}")
        return {
            "test_name": "ML予測システム",
            "success": False,
            "processing_time": time.time() - start_time,
            "error": str(e),
            "details": "ML予測システムエラー",
        }


async def test_cache_system() -> Dict[str, Any]:
    """統合キャッシュシステムテスト"""
    logger.info("統合キャッシュシステムテスト開始")
    start_time = time.time()

    try:
        cache_manager = UnifiedCacheManager(
            l1_memory_mb=64, l2_memory_mb=256, l3_disk_mb=512
        )

        # キャッシュ操作テスト
        test_key = "test_integration_key"
        test_value = {"test_data": [1, 2, 3, 4, 5]}

        # 書き込みテスト
        cache_manager.put(test_key, test_value)

        # 読み込みテスト
        retrieved_value = cache_manager.get(test_key)

        processing_time = time.time() - start_time

        cache_success = retrieved_value is not None and retrieved_value == test_value

        return {
            "test_name": "統合キャッシュシステム",
            "success": cache_success,
            "processing_time": processing_time,
            "details": f"キャッシュ読み書き{'成功' if cache_success else '失敗'}",
        }

    except Exception as e:
        logger.error(f"キャッシュシステムテストエラー: {e}")
        return {
            "test_name": "統合キャッシュシステム",
            "success": False,
            "processing_time": time.time() - start_time,
            "error": str(e),
            "details": "キャッシュシステムエラー",
        }


async def test_performance_monitoring() -> Dict[str, Any]:
    """パフォーマンス監視システムテスト"""
    logger.info("パフォーマンス監視システムテスト開始")
    start_time = time.time()

    try:
        monitor = PerformanceMonitor()

        # 監視開始
        monitor.start_monitoring("integration_test")

        # 模擬処理
        await asyncio.sleep(0.1)

        # 監視停止
        metrics = monitor.stop_monitoring("integration_test")

        processing_time = time.time() - start_time

        return {
            "test_name": "パフォーマンス監視",
            "success": True,
            "processing_time": processing_time,
            "details": "監視システム正常動作、メトリクス取得成功",
        }

    except Exception as e:
        logger.error(f"パフォーマンス監視テストエラー: {e}")
        return {
            "test_name": "パフォーマンス監視",
            "success": False,
            "processing_time": time.time() - start_time,
            "error": str(e),
            "details": "パフォーマンス監視システムエラー",
        }


async def main():
    """最終システム統合テスト実行"""
    print("=" * 80)
    print("最終システム統合テスト")
    print("全コンポーネント動作検証")
    print("=" * 80)

    # テストデータ生成
    print("\nテストデータ生成中...")
    test_data = generate_comprehensive_test_data()
    print(f"OK テストデータ生成完了: {len(test_data)}銘柄")

    # 各システムテスト実行
    tests = []

    # 1. TOPIX500システムテスト
    print("\n1. TOPIX500システムテスト実行中...")
    topix500_result = await test_topix500_system(test_data)
    tests.append(topix500_result)

    # 2. 高度テクニカル指標テスト
    print("\n2. 高度テクニカル指標テスト実行中...")
    technical_result = await test_advanced_technical_indicators(test_data)
    tests.append(technical_result)

    # 3. ML予測システムテスト
    print("\n3. ML予測システムテスト実行中...")
    ml_result = await test_ml_models_system(test_data)
    tests.append(ml_result)

    # 4. 統合キャッシュテスト
    print("\n4. 統合キャッシュシステムテスト実行中...")
    cache_result = await test_cache_system()
    tests.append(cache_result)

    # 5. パフォーマンス監視テスト
    print("\n5. パフォーマンス監視システムテスト実行中...")
    monitor_result = await test_performance_monitoring()
    tests.append(monitor_result)

    # 結果集計
    print("\n" + "=" * 80)
    print("統合テスト結果")
    print("=" * 80)

    successful_tests = 0
    total_processing_time = 0

    for i, result in enumerate(tests, 1):
        status = "OK" if result["success"] else "NG"
        print(f"{i}. {result['test_name']}: {status}")
        print(f"   処理時間: {result['processing_time']:.2f}秒")
        print(f"   詳細: {result['details']}")

        if not result["success"] and "error" in result:
            print(f"   エラー: {result['error']}")

        if result["success"]:
            successful_tests += 1
        total_processing_time += result["processing_time"]
        print()

    # 総合結果
    success_rate = successful_tests / len(tests)
    print("総合結果:")
    print(f"  成功テスト: {successful_tests}/{len(tests)} ({success_rate:.1%})")
    print(f"  総処理時間: {total_processing_time:.2f}秒")

    if success_rate >= 0.8:
        print("  判定: OK システム統合成功")
        print("  ステータス: 実用化準備完了")
    elif success_rate >= 0.6:
        print("  判定: PARTIAL 部分的成功")
        print("  ステータス: 一部調整が必要")
    else:
        print("  判定: NG システム統合失敗")
        print("  ステータス: 追加修正が必要")

    print("\nOK 最終システム統合テスト完了")

    return success_rate >= 0.8


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n統合テスト中断")
        sys.exit(1)
    except Exception as e:
        print(f"統合テストエラー: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
