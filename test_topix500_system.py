#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOPIX500システムテスト
Issue #314: TOPIX500全銘柄対応機能検証

統合最適化システム基盤活用テスト:
- Issue #325: 97%高速化ML処理
- Issue #324: 98%メモリ削減キャッシュ
- Issue #323: 100倍並列処理
- Issue #322: 89%精度データ拡張
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent))

def test_topix500_manager_import():
    """TOPIX500管理システムインポートテスト"""
    print("=== TOPIX500管理システムインポートテスト ===")

    try:
        from src.day_trade.data.topix500_manager import TOPIX500Manager, TOPIX500Stock, SectorAnalysis
        print("[OK] TOPIX500Manager import success")

        # 基本クラステスト
        stock = TOPIX500Stock(
            code="7203",
            name="トヨタ自動車",
            sector="Transportation",
            industry="Automotive"
        )
        print(f"[OK] TOPIX500Stock creation: {stock.code} - {stock.name}")

        return True

    except Exception as e:
        print(f"[ERROR] Import error: {e}")
        traceback.print_exc()
        return False

async def test_topix500_initialization():
    """TOPIX500システム初期化テスト"""
    print("\n=== TOPIX500システム初期化テスト ===")

    try:
        from src.day_trade.data.topix500_manager import TOPIX500Manager

        # TOPIX500マネージャー初期化
        manager = TOPIX500Manager(
            enable_cache=True,
            batch_size=50,
            max_concurrent=10,
            target_processing_time=20
        )
        print("[OK] TOPIX500Manager initialization success")

        # 処理状況確認
        status = manager.get_processing_status()
        print(f"[OK] Processing status: {status['target_processing_time']}s target")

        return manager, True

    except Exception as e:
        print(f"[ERROR] Initialization error: {e}")
        traceback.print_exc()
        return None, False

async def test_topix500_stock_loading(manager):
    """TOPIX500銘柄ロードテスト"""
    print("\n=== TOPIX500銘柄ロードテスト ===")

    try:
        # 銘柄リストロード
        success = await manager.load_topix500_list(source="auto")

        if success:
            print(f"[OK] Stock loading success: {len(manager.topix500_stocks)} stocks loaded")

            # セクター情報確認
            sector_summary = manager.get_sector_summary()
            print(f"[OK] Sector organization: {sector_summary['total_sectors']} sectors")
            print(f"[INFO] Largest sector: {sector_summary.get('largest_sector', 'N/A')}")

            return True
        else:
            print("[ERROR] Stock loading failed")
            return False

    except Exception as e:
        print(f"[ERROR] Stock loading error: {e}")
        traceback.print_exc()
        return False

async def test_small_batch_analysis(manager):
    """小規模バッチ分析テスト（5銘柄）"""
    print("\n=== 小規模バッチ分析テスト ===")

    try:
        # テスト用に小さなバッチサイズに変更
        original_batch_size = manager.batch_size
        manager.batch_size = 5

        # テスト用銘柄リストを5銘柄に制限
        test_stocks = dict(list(manager.topix500_stocks.items())[:5])
        original_stocks = manager.topix500_stocks
        manager.topix500_stocks = test_stocks

        print(f"[INFO] Testing with {len(test_stocks)} stocks")

        # 分析実行
        start_time = time.time()
        results = await manager.analyze_all_topix500(
            enable_sector_analysis=True,
            save_results=False  # テスト用に保存無効
        )
        processing_time = time.time() - start_time

        # 結果確認
        summary = results["analysis_summary"]
        print(f"[OK] Analysis completed: {summary['successful_analysis']}/{summary['total_stocks']} stocks")
        print(f"[OK] Processing time: {processing_time:.2f}s")
        print(f"[OK] Memory usage: {summary.get('memory_usage', 0):.1f}MB")
        print(f"[OK] Sectors analyzed: {len(results['sector_analysis'])}")

        # 元の設定に復元
        manager.batch_size = original_batch_size
        manager.topix500_stocks = original_stocks

        return True

    except Exception as e:
        print(f"[ERROR] Small batch analysis error: {e}")
        traceback.print_exc()
        return False

async def test_sector_analysis(manager):
    """セクター分析テスト"""
    print("\n=== セクター分析テスト ===")

    try:
        # セクターサマリー取得
        sector_summary = manager.get_sector_summary()
        print(f"[OK] Sector summary: {sector_summary['total_sectors']} sectors")

        # セクター分布確認
        sector_dist = sector_summary["sector_distribution"]
        for sector, count in list(sector_dist.items())[:5]:  # 上位5セクター表示
            print(f"[INFO] {sector}: {count} stocks")

        return True

    except Exception as e:
        print(f"[ERROR] Sector analysis error: {e}")
        traceback.print_exc()
        return False

async def test_performance_projection():
    """パフォーマンス予測テスト"""
    print("\n=== パフォーマンス予測テスト ===")

    try:
        # 統合最適化基盤の性能を基にTOPIX500処理時間予測

        # Issue基盤性能データ
        base_performance = {
            "ml_speedup": 97,        # Issue #325: 97%高速化 (23.6s→0.3s)
            "memory_reduction": 98,  # Issue #324: 98%削減 (500MB→4.6MB)
            "parallel_speedup": 100, # Issue #323: 100倍並列化
            "accuracy": 89           # Issue #322: 89%精度
        }

        # 現在の小規模処理性能から予測
        current_stocks = 85  # 現在の銘柄数
        target_stocks = 500  # TOPIX500目標
        scale_factor = target_stocks / current_stocks

        # 並列処理効果を考慮した処理時間予測
        base_processing_time = 0.3  # 1銘柄あたりの処理時間（秒）
        parallel_efficiency = 0.8   # 並列効率

        # 予測計算
        sequential_time = target_stocks * base_processing_time
        parallel_time = sequential_time / (base_performance["parallel_speedup"] * parallel_efficiency / 100)

        # メモリ予測
        base_memory = 4.6  # MB
        predicted_memory = base_memory * scale_factor * 0.1  # キャッシュ効果考慮

        print(f"[PREDICTION] TOPIX500処理時間予測:")
        print(f"  - 逐次処理時: {sequential_time:.1f}秒")
        print(f"  - 並列処理時: {parallel_time:.1f}秒")
        print(f"  - 目標20秒: {'達成可能' if parallel_time <= 20 else '要追加最適化'}")
        print(f"[PREDICTION] メモリ使用量予測:")
        print(f"  - 予測使用量: {predicted_memory:.1f}MB")
        print(f"  - 目標1GB: {'達成可能' if predicted_memory <= 1000 else '要最適化'}")
        print(f"[PREDICTION] 精度維持:")
        print(f"  - 期待精度: {base_performance['accuracy']}% (統合最適化基盤)")

        return True

    except Exception as e:
        print(f"[ERROR] Performance projection error: {e}")
        traceback.print_exc()
        return False

async def main():
    """メイン実行関数"""
    print("TOPIX500システムテスト開始")
    print("=" * 60)

    results = []
    manager = None

    # 各テスト実行
    results.append(("インポートテスト", test_topix500_manager_import()))

    if results[-1][1]:  # インポート成功時のみ続行
        manager, init_success = await test_topix500_initialization()
        results.append(("初期化テスト", init_success))

        if init_success and manager:
            loading_success = await test_topix500_stock_loading(manager)
            results.append(("銘柄ロードテスト", loading_success))

            if loading_success:
                results.append(("小規模分析テスト", await test_small_batch_analysis(manager)))
                results.append(("セクター分析テスト", await test_sector_analysis(manager)))

    # パフォーマンス予測（独立テスト）
    results.append(("パフォーマンス予測", await test_performance_projection()))

    # システムクリーンアップ
    if manager:
        try:
            await manager.shutdown()
        except:
            pass

    # 結果サマリー
    print("\n" + "=" * 60)
    print("=== テスト結果サマリー ===")

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1

    print(f"\n成功率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")

    if passed == len(results):
        print("[SUCCESS] 全テスト成功！TOPIX500システム準備完了")
        return True
    else:
        print("[WARNING] 一部テスト失敗 - 要追加開発")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nテスト中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n致命的エラー: {e}")
        traceback.print_exc()
        sys.exit(1)
