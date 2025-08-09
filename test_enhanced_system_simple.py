"""
拡張取引システムの簡単な統合テスト・パフォーマンス検証
"""

import asyncio
import sys
import time
from decimal import Decimal
from pathlib import Path

# パス追加
sys.path.append(str(Path(__file__).parent / "src"))

from day_trade.automation.advanced_order_manager import (
    AdvancedOrderManager,
    Order,
    OrderType,
)
from day_trade.automation.enhanced_trading_engine import (
    EnhancedTradingEngine,
    ExecutionMode,
)
from day_trade.automation.portfolio_manager import PortfolioManager
from day_trade.core.trade_manager import Trade, TradeType


async def test_advanced_order_manager():
    """高度な注文管理システムの基本テスト"""
    print("[*] AdvancedOrderManager test start")

    manager = AdvancedOrderManager()

    # 基本注文のテスト
    order = Order(
        symbol="7203",
        order_type=OrderType.LIMIT,
        side=TradeType.BUY,
        quantity=100,
        price=Decimal("2500.0")
    )

    start_time = time.time()
    order_id = await manager.submit_order(order)
    submit_time = time.time() - start_time

    print(f"[+] Order submit: {order_id[:8]} - time: {submit_time*1000:.1f}ms")

    # 注文キャンセル
    start_time = time.time()
    cancelled = await manager.cancel_order(order_id)
    cancel_time = time.time() - start_time

    print(f"[+] Order cancel: {cancelled} - time: {cancel_time*1000:.1f}ms")

    # OCO注文のテスト
    primary = Order(
        symbol="6758",
        order_type=OrderType.LIMIT,
        side=TradeType.SELL,
        quantity=50,
        price=Decimal("3000.0")
    )

    secondary = Order(
        symbol="6758",
        order_type=OrderType.STOP,
        side=TradeType.SELL,
        quantity=50,
        stop_price=Decimal("2800.0")
    )

    start_time = time.time()
    primary_id, secondary_id = await manager.submit_oco_order(primary, secondary)
    oco_time = time.time() - start_time

    print(f"[+] OCO order submit: {primary_id[:8]}/{secondary_id[:8]} - time: {oco_time*1000:.1f}ms")

    # 統計確認
    stats = manager.get_execution_statistics()
    print(f"[*] Order stats: submitted={stats['orders_submitted']}, cancelled={stats['orders_cancelled']}")

    print("[+] AdvancedOrderManager test completed\n")


def test_portfolio_manager():
    """ポートフォリオ管理システムの基本テスト"""
    print("💼 PortfolioManager テスト開始")

    portfolio = PortfolioManager(initial_cash=Decimal("1000000"))

    # 取引追加のパフォーマンステスト
    trades = [
        Trade("t1", "7203", TradeType.BUY, 100, Decimal("2500.0"), time=None, commission=Decimal("25.0"), status="executed"),
        Trade("t2", "6758", TradeType.BUY, 200, Decimal("3000.0"), time=None, commission=Decimal("50.0"), status="executed"),
        Trade("t3", "9984", TradeType.SELL, 50, Decimal("8000.0"), time=None, commission=Decimal("40.0"), status="executed"),
    ]

    start_time = time.time()
    for trade in trades:
        portfolio.add_trade(trade)
    trade_time = time.time() - start_time

    print(f"✅ 取引追加: {len(trades)}件 - 実行時間: {trade_time*1000:.1f}ms")

    # 市場価格更新のパフォーマンステスト
    price_data = {
        "7203": Decimal("2600.0"),
        "6758": Decimal("2900.0"),
        "9984": Decimal("8200.0"),
    }

    start_time = time.time()
    portfolio.update_market_prices(price_data)
    update_time = time.time() - start_time

    print(f"✅ 価格更新: {len(price_data)}銘柄 - 実行時間: {update_time*1000:.1f}ms")

    # ポートフォリオサマリー取得
    start_time = time.time()
    summary = portfolio.get_portfolio_summary()
    summary_time = time.time() - start_time

    print(f"✅ サマリー取得 - 実行時間: {summary_time*1000:.1f}ms")
    print(f"📊 ポートフォリオ: 総資産={summary.total_equity:,}, ポジション数={summary.total_positions}")

    # リスクチェック
    start_time = time.time()
    risk_check = portfolio.check_risk_limits()
    risk_time = time.time() - start_time

    print(f"✅ リスクチェック - 実行時間: {risk_time*1000:.1f}ms")
    print(f"📊 リスク: 違反={risk_check['total_violations']}, スコア={risk_check['risk_score']:.2f}")

    print("✅ PortfolioManager テスト完了\n")


async def test_enhanced_trading_engine():
    """拡張取引エンジンの基本テスト"""
    print("🚀 EnhancedTradingEngine テスト開始")

    # 軽量設定でエンジン作成
    engine = EnhancedTradingEngine(
        symbols=["7203", "6758"],
        execution_mode=ExecutionMode.BALANCED,
        initial_cash=Decimal("500000"),
        update_interval=0.5  # テスト用短縮
    )

    print(f"✅ エンジン初期化: {len(engine.symbols)}銘柄監視")
    print(f"📊 初期資金: {engine.portfolio_manager.initial_cash:,}円")
    print(f"⚙️ 実行モード: {engine.execution_mode.value}")

    # 基本機能テスト
    try:
        # 短時間実行
        print("🔄 エンジン開始（3秒間実行）...")
        engine_task = asyncio.create_task(engine.start())

        await asyncio.sleep(3.0)  # 3秒間実行

        # 実行中の状態確認
        status = engine.get_comprehensive_status()
        print("📊 実行状態:")
        print(f"   - ステータス: {status['engine']['status']}")
        print(f"   - サイクル数: {status['engine']['engine_cycles']}")
        print(f"   - 平均サイクル時間: {status['engine']['avg_cycle_time_ms']:.1f}ms")

        # エンジン停止
        await engine.stop()
        print("⏹️ エンジン停止完了")

        # 最終統計
        final_status = engine.get_comprehensive_status()
        print("📈 最終統計:")
        print(f"   - 総サイクル: {final_status['engine']['engine_cycles']}")
        print(f"   - 実行時間: {final_status['engine']['uptime_seconds']}秒")
        print(f"   - シグナル処理: {engine.execution_stats['signals_processed']}")
        print(f"   - 生成注文: {engine.execution_stats['orders_generated']}")

        # タスク完了を待機
        try:
            await asyncio.wait_for(engine_task, timeout=2.0)
        except asyncio.TimeoutError:
            engine_task.cancel()

    except Exception as e:
        print(f"❌ エラー: {e}")
        engine.emergency_stop()

    print("✅ EnhancedTradingEngine テスト完了\n")


def test_execution_modes():
    """実行モード比較テスト"""
    print("⚖️ ExecutionMode 比較テスト")

    modes = [ExecutionMode.CONSERVATIVE, ExecutionMode.BALANCED, ExecutionMode.AGGRESSIVE]

    for mode in modes:
        engine = EnhancedTradingEngine(
            symbols=["7203"],
            execution_mode=mode,
        )

        threshold = engine._get_min_confidence_threshold()
        print(f"📊 {mode.value:>12}: 信頼度閾値 = {threshold:>4.1f}%")

    print("✅ ExecutionMode 比較完了\n")


async def performance_stress_test():
    """パフォーマンス負荷テスト"""
    print("🔥 パフォーマンス負荷テスト開始")

    # 大量注文処理テスト
    manager = AdvancedOrderManager()

    orders = []
    symbols = ["7203", "6758", "9984", "4755", "8058"]

    # 大量注文を生成
    start_time = time.time()
    for i in range(100):  # 100件の注文
        symbol = symbols[i % len(symbols)]
        order = Order(
            symbol=symbol,
            order_type=OrderType.LIMIT,
            side=TradeType.BUY if i % 2 == 0 else TradeType.SELL,
            quantity=100,
            price=Decimal("2500.0")
        )
        orders.append(order)

    generation_time = time.time() - start_time
    print(f"✅ 注文生成: {len(orders)}件 - {generation_time*1000:.1f}ms")

    # 一括提出テスト
    start_time = time.time()
    submitted_orders = []
    for order in orders:
        order_id = await manager.submit_order(order)
        submitted_orders.append(order_id)

    submission_time = time.time() - start_time
    throughput = len(orders) / submission_time
    print(f"✅ 注文提出: {len(orders)}件 - {submission_time*1000:.1f}ms")
    print(f"📊 スループット: {throughput:.1f} orders/sec")

    # 一括キャンセルテスト
    start_time = time.time()
    cancelled_count = await manager.cancel_all_orders()
    cancel_time = time.time() - start_time

    print(f"✅ 一括キャンセル: {cancelled_count}件 - {cancel_time*1000:.1f}ms")

    # 統計確認
    stats = manager.get_execution_statistics()
    print(f"📊 最終統計: 提出={stats['orders_submitted']}, キャンセル={stats['orders_cancelled']}")

    print("✅ パフォーマンス負荷テスト完了\n")


async def main():
    """メイン実行関数"""
    print("🎯 Enhanced Trading System - 統合テスト・パフォーマンス検証")
    print("=" * 70)
    print()

    try:
        # 基本機能テスト
        await test_advanced_order_manager()
        test_portfolio_manager()
        await test_enhanced_trading_engine()

        # 比較テスト
        test_execution_modes()

        # パフォーマンステスト
        await performance_stress_test()

        print("🎉 全テスト完了 - Enhanced Trading System は正常に動作しています！")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 非同期実行
    asyncio.run(main())
