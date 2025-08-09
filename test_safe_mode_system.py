"""
自動取引無効化システムのテスト

【確認項目】
1. セーフモードが有効であること
2. 自動取引機能が完全に無効化されていること
3. 分析機能のみが動作すること
4. 安全設定が強制されていること
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import contextlib

from src.day_trade.automation.analysis_only_engine import AnalysisOnlyEngine
from src.day_trade.automation.trading_engine import TradingEngine
from src.day_trade.config.trading_mode_config import (
    get_current_trading_config,
    is_safe_mode,
    log_current_configuration,
)


def test_safe_mode_configuration():
    """セーフモード設定のテスト"""
    print("=" * 60)
    print("セーフモード設定テスト")
    print("=" * 60)

    # 設定の確認
    config = get_current_trading_config()
    safe_mode = is_safe_mode()

    print(f"セーフモード: {'有効' if safe_mode else '無効'}")
    print(f"自動取引: {'無効' if not config.enable_automatic_trading else '有効'}")
    print(f"注文実行: {'無効' if not config.enable_order_execution else '有効'}")
    print(f"注文API: {'無効' if config.disable_order_api else '有効'}")
    print(f"手動確認必須: {'有効' if config.require_manual_confirmation else '無効'}")

    # 安全性検証
    validation = config.validate_configuration()
    print("\n設定妥当性検証:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {'合格' if value else '不合格'}")

    assert safe_mode, "セーフモードが無効です"
    assert not config.enable_automatic_trading, "自動取引が有効になっています"
    assert not config.enable_order_execution, "注文実行が有効になっています"
    assert config.disable_order_api, "注文APIが有効になっています"
    assert config.require_manual_confirmation, "手動確認が無効になっています"

    print("\n✓ セーフモード設定テスト: 合格")


def test_trading_engine_safety():
    """TradingEngineの安全性テスト"""
    print("\n" + "=" * 60)
    print("TradingEngine 安全性テスト")
    print("=" * 60)

    test_symbols = ["7203", "8306", "9984"]

    try:
        # TradingEngineの初期化テスト
        engine = TradingEngine(test_symbols)

        # 設定確認
        status = engine.get_status()
        print(f"セーフモード: {'有効' if status['safe_mode'] else '無効'}")
        print(f"取引無効: {'有効' if status['trading_disabled'] else '無効'}")
        print(f"監視銘柄数: {status['monitored_symbols']}")

        # 安全性アサーション
        assert status['safe_mode'], "TradingEngineがセーフモードではありません"
        assert status['trading_disabled'], "取引機能が有効になっています"

        print("\n✓ TradingEngine安全性テスト: 合格")

    except ValueError as e:
        if "安全設定が無効です" in str(e):
            print("\n✓ TradingEngine安全チェック: 正常に機能（初期化拒否）")
        else:
            raise e


def test_analysis_only_engine():
    """分析専用エンジンのテスト"""
    print("\n" + "=" * 60)
    print("分析専用エンジンテスト")
    print("=" * 60)

    test_symbols = ["7203", "8306"]

    try:
        # AnalysisOnlyEngineの初期化テスト
        engine = AnalysisOnlyEngine(test_symbols, update_interval=5.0)

        # 状態確認
        status = engine.get_status()
        print(f"エンジンステータス: {status['status']}")
        print(f"監視銘柄数: {status['monitored_symbols']}")
        print(f"セーフモード: {'有効' if status['safe_mode'] else '無効'}")
        print(f"取引無効: {'有効' if status['trading_disabled'] else '無効'}")

        # 安全性アサーション
        assert status['safe_mode'], "分析エンジンがセーフモードではありません"
        assert status['trading_disabled'], "取引機能が有効になっています"

        # 推奨事項テスト
        recommendations = engine.get_symbol_recommendations("7203")
        print("\n7203の推奨事項:")
        for rec in recommendations:
            print(f"  - {rec}")

        print("\n✓ 分析専用エンジンテスト: 合格")

    except Exception as e:
        print(f"\n✗ 分析専用エンジンテスト: エラー - {e}")
        raise e


async def test_analysis_engine_operation():
    """分析エンジンの動作テスト（短時間実行）"""
    print("\n" + "=" * 60)
    print("分析エンジン動作テスト")
    print("=" * 60)

    test_symbols = ["7203"]

    try:
        engine = AnalysisOnlyEngine(test_symbols, update_interval=2.0)

        print("分析エンジンを5秒間実行します...")

        # エンジン開始
        start_task = asyncio.create_task(engine.start())

        # 5秒待機
        await asyncio.sleep(5.0)

        # エンジン停止
        await engine.stop()

        # 結果確認
        status = engine.get_status()
        print(f"総分析回数: {status['stats']['total_analyses']}")
        print(f"成功分析回数: {status['stats']['successful_analyses']}")
        print(f"失敗分析回数: {status['stats']['failed_analyses']}")

        # レポート確認
        latest_report = engine.get_latest_report()
        if latest_report:
            print("最新レポート:")
            print(f"  - 分析銘柄数: {latest_report.analyzed_symbols}")
            print(f"  - 強いシグナル: {latest_report.strong_signals}")
            print(f"  - 市場センチメント: {latest_report.market_sentiment}")
            print(f"  - 分析時間: {latest_report.analysis_time_ms:.1f}ms")

        # サマリー確認
        summary = engine.get_market_summary()
        print(f"\n市場サマリー: {summary}")

        print("\n✓ 分析エンジン動作テスト: 合格")

        # タスクをキャンセル
        if not start_task.done():
            start_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await start_task

    except Exception as e:
        print(f"\n✗ 分析エンジン動作テスト: エラー - {e}")
        raise e


def test_system_security():
    """システムセキュリティテスト"""
    print("\n" + "=" * 60)
    print("システムセキュリティテスト")
    print("=" * 60)

    config = get_current_trading_config()

    # セキュリティ要件確認
    security_checks = {
        "自動取引無効": not config.enable_automatic_trading,
        "注文実行無効": not config.enable_order_execution,
        "注文API無効": config.disable_order_api,
        "手動確認必須": config.require_manual_confirmation,
        "全活動ログ有効": config.log_all_activities,
        "分析機能有効": config.enable_analysis,
        "市場データ有効": config.enable_market_data,
    }

    print("セキュリティチェック結果:")
    all_passed = True
    for check, passed in security_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ システムセキュリティテスト: 合格")
    else:
        print("\n✗ システムセキュリティテスト: 不合格")
        raise AssertionError("セキュリティ要件を満たしていません")


async def main():
    """メインテスト実行"""
    print("自動取引無効化システム テスト開始")
    print("=" * 80)

    try:
        # 設定ログ出力
        log_current_configuration()

        # テスト実行
        test_safe_mode_configuration()
        test_trading_engine_safety()
        test_analysis_only_engine()
        await test_analysis_engine_operation()
        test_system_security()

        print("\n" + "=" * 80)
        print("🎉 全テスト合格！")
        print("✓ 自動取引機能は完全に無効化されています")
        print("✓ システムは安全なセーフモードで動作しています")
        print("✓ 分析機能のみが有効で、取引実行は一切行われません")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ テスト失敗: {e}")
        print("=" * 80)
        raise e


if __name__ == "__main__":
    asyncio.run(main())
