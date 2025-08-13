"""
分析専用システムテスト

自動取引無効化後の統合分析システムの動作テスト
"""

import asyncio
import sys
from pathlib import Path

# パス追加
sys.path.append(str(Path(__file__).parent / "src"))

from day_trade.analysis.market_analysis_system import MarketAnalysisSystem
from day_trade.automation.risk_aware_trading_engine import MarketAnalysisEngine
from day_trade.config.trading_mode_config import (
    get_current_trading_config,
    is_safe_mode,
    log_current_configuration,
)


def test_safety_configuration():
    """安全設定テスト"""
    print("=" * 60)
    print("安全設定確認テスト")
    print("=" * 60)

    # 1. セーフモード確認
    safe_mode = is_safe_mode()
    print(f"1. セーフモード: {'有効' if safe_mode else '無効'}")
    assert safe_mode, "セーフモードが無効化されています"

    # 2. 取引設定確認
    config = get_current_trading_config()
    print(f"2. 現在のモード: {config.current_mode.value}")
    print(f"3. 自動取引: {'無効' if not config.is_trading_enabled() else '有効'}")

    # 3. 設定詳細出力
    print("\n有効な機能:")
    for feature in config.get_enabled_features():
        print(f"  + {feature}")

    print("\n無効な機能:")
    for feature in config.get_disabled_features():
        print(f"  - {feature}")

    print("\n[OK] 安全設定テスト合格")


def test_market_analysis_system():
    """市場分析システムテスト"""
    print("\n" + "=" * 60)
    print("市場分析システムテスト")
    print("=" * 60)

    symbols = ["7203", "6758"]

    try:
        # 1. システム初期化
        analysis_system = MarketAnalysisSystem(symbols)
        print("[OK] 市場分析システム初期化成功")

        # 2. 分析サマリー取得
        summary = analysis_system.get_analysis_summary()
        print("[OK] 分析サマリー取得成功")
        print(f"  - システム名: {summary['system_info']['system_name']}")
        print(f"  - モード: {summary['system_info']['mode']}")
        print(f"  - セーフモード: {summary['system_info']['safe_mode']}")

    except Exception as e:
        print(f"[FAIL] 市場分析システムテスト失敗: {e}")
        raise


async def test_analysis_engine():
    """分析エンジンテスト"""
    print("\n" + "=" * 60)
    print("分析エンジンテスト")
    print("=" * 60)

    symbols = ["7203"]

    try:
        # 1. エンジン初期化（自動取引無効確認）
        engine = MarketAnalysisEngine(symbols=symbols, emergency_stop_enabled=False)
        print("[OK] 分析エンジン初期化成功")

        # 2. ステータス確認
        status = engine.get_comprehensive_status()
        print("[OK] ステータス取得成功")
        print(f"  - エンジンタイプ: {status['engine_type']}")
        print(f"  - セーフモード: {status['safe_mode']}")
        print(f"  - 自動取引: {status['automatic_trading']}")

        # 3. 短時間実行テスト
        print("  - 短時間実行テスト開始...")

        # 非同期でエンジンを開始
        start_task = asyncio.create_task(engine.start())

        # 3秒間実行
        await asyncio.sleep(3.0)

        # エンジン停止
        await engine.stop()

        # 開始タスクの完了を待つ
        try:
            await asyncio.wait_for(start_task, timeout=2.0)
        except asyncio.TimeoutError:
            start_task.cancel()

        print("  - 短時間実行完了")
        print("[OK] 分析エンジンテスト合格")

    except Exception as e:
        print(f"[FAIL] 分析エンジンテスト失敗: {e}")
        raise


def test_system_integration():
    """システム統合テスト"""
    print("\n" + "=" * 60)
    print("システム統合テスト")
    print("=" * 60)

    try:
        # 統合分析システムのインポート確認
        from day_trade.core.integrated_analysis_system import IntegratedAnalysisSystem

        symbols = ["7203"]

        # システム初期化
        integrated_system = IntegratedAnalysisSystem(symbols)
        print("[OK] 統合分析システム初期化成功")

        # システム状態確認
        status = integrated_system.get_system_status()
        print("[OK] システム状態取得成功")
        print(f"  - システム名: {status['system_info']['name']}")
        print(f"  - セーフモード: {status['system_info']['safe_mode']}")
        print(f"  - 自動取引: {status['system_info']['automatic_trading']}")

    except Exception as e:
        print(f"[FAIL] システム統合テスト失敗: {e}")
        raise


async def main():
    """メインテスト関数"""
    print("分析専用システム動作確認テスト")
    print("=" * 80)

    try:
        # 1. 設定確認
        log_current_configuration()

        # 2. 安全設定テスト
        test_safety_configuration()

        # 3. 市場分析システムテスト
        test_market_analysis_system()

        # 4. 分析エンジンテスト
        await test_analysis_engine()

        # 5. システム統合テスト
        test_system_integration()

        print("\n" + "=" * 80)
        print("[SUCCESS] 全テスト合格！")
        print("[OK] システムは正常に分析専用モードで動作しています")
        print("[DISABLED] 自動取引機能は完全に無効化されています")
        print("=" * 80)

    except Exception as e:
        print(f"\n[FAIL] テスト失敗: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    asyncio.run(main())
