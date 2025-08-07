"""
自動取引エンジンの使用例
リアルタイム取引の基本的な使用方法を示す
"""

import asyncio
import logging
import sys
from decimal import Decimal
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

from day_trade.automation.trading_engine import RiskParameters, TradingEngine


async def basic_trading_example():
    """基本的な自動取引の例"""

    print("🚀 自動取引エンジン実行例")
    print("=" * 50)

    # 監視する銘柄リスト
    symbols = ["7203", "6758", "9984"]  # トヨタ、ソニーグループ、ソフトバンクグループ

    # リスク管理パラメータを設定
    risk_params = RiskParameters(
        max_position_size=Decimal("500000"),  # 最大ポジション: 50万円
        max_daily_loss=Decimal("10000"),  # 日次最大損失: 1万円
        max_open_positions=3,  # 最大保有銘柄数: 3
        stop_loss_ratio=Decimal("0.02"),  # ストップロス: 2%
        take_profit_ratio=Decimal("0.05"),  # 利益確定: 5%
    )

    # 取引エンジンを初期化
    engine = TradingEngine(
        symbols=symbols,
        risk_params=risk_params,
        update_interval=2.0,  # 2秒間隔で更新
    )

    print(f"📊 監視銘柄: {', '.join(symbols)}")
    print(f"💰 最大ポジション: {risk_params.max_position_size:,}円")
    print(f"⚠️ 最大損失: {risk_params.max_daily_loss:,}円")
    print()

    try:
        # エンジンを開始
        print("✅ 取引エンジンを開始します...")

        # 非同期でエンジンを実行
        engine_task = asyncio.create_task(engine.start())

        # 30秒間実行
        for i in range(15):  # 2秒間隔で15回 = 30秒
            await asyncio.sleep(2)

            # ステータスを表示
            status = engine.get_status()
            print(f"🔄 [{i+1:2d}/15] ステータス: {status['status']}")
            print(f"   アクティブポジション: {status['active_positions']}")
            print(f"   実行した注文数: {status['execution_stats']['orders_executed']}")
            print(
                f"   生成シグナル数: {status['execution_stats']['signals_generated']}"
            )
            print(f"   日次損益: {status['daily_pnl']:+.0f}円")

            if status["execution_stats"]["avg_execution_time"] > 0:
                print(
                    f"   平均実行時間: {status['execution_stats']['avg_execution_time']*1000:.1f}ms"
                )
            print()

        print("⏹️ 取引を停止します...")
        await engine.stop()

        # 最終結果を表示
        final_status = engine.get_status()
        print("📋 最終結果:")
        print(
            f"   実行した注文数: {final_status['execution_stats']['orders_executed']}"
        )
        print(
            f"   生成シグナル数: {final_status['execution_stats']['signals_generated']}"
        )
        print(f"   最終損益: {final_status['daily_pnl']:+.0f}円")

        # タスク完了を待機
        try:
            await asyncio.wait_for(engine_task, timeout=2.0)
        except asyncio.TimeoutError:
            engine_task.cancel()

    except KeyboardInterrupt:
        print("\n🛑 ユーザーによる中断")
        await engine.stop()
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        engine.emergency_stop()


async def risk_management_example():
    """リスク管理の例"""

    print("🛡️ リスク管理機能の例")
    print("=" * 50)

    # より厳しいリスク設定
    strict_risk_params = RiskParameters(
        max_position_size=Decimal("100000"),  # 最大ポジション: 10万円
        max_daily_loss=Decimal("5000"),  # 日次最大損失: 5000円
        max_open_positions=2,  # 最大保有銘柄数: 2
        stop_loss_ratio=Decimal("0.015"),  # ストップロス: 1.5%
        take_profit_ratio=Decimal("0.03"),  # 利益確定: 3%
    )

    engine = TradingEngine(
        symbols=["7203", "6758"], risk_params=strict_risk_params, update_interval=1.0
    )

    print("💡 厳格なリスク管理設定:")
    print(f"   最大ポジション: {strict_risk_params.max_position_size:,}円")
    print(f"   最大損失: {strict_risk_params.max_daily_loss:,}円")
    print(f"   ストップロス: {strict_risk_params.stop_loss_ratio*100:.1f}%")
    print(f"   利益確定: {strict_risk_params.take_profit_ratio*100:.1f}%")
    print()

    try:
        # 短時間実行
        print("✅ リスク管理テストを開始...")
        engine_task = asyncio.create_task(engine.start())

        # 10秒間監視
        for i in range(10):
            await asyncio.sleep(1)

            status = engine.get_status()
            constraints_ok = engine._check_risk_constraints()

            print(
                f"⚖️ [{i+1:2d}/10] リスク制約: {'✅ OK' if constraints_ok else '⚠️ 制約違反'}"
            )
            print(
                f"   ポジション数: {status['active_positions']}/{strict_risk_params.max_open_positions}"
            )
            print(
                f"   日次損益: {status['daily_pnl']:+.0f}円 (制限: {-strict_risk_params.max_daily_loss:+.0f}円)"
            )

        await engine.stop()

        try:
            await asyncio.wait_for(engine_task, timeout=2.0)
        except asyncio.TimeoutError:
            engine_task.cancel()

    except Exception as e:
        print(f"❌ エラー: {e}")
        engine.emergency_stop()


async def emergency_stop_example():
    """緊急停止機能の例"""

    print("🚨 緊急停止機能の例")
    print("=" * 50)

    engine = TradingEngine(symbols=["7203"], update_interval=0.5)

    try:
        print("✅ エンジン開始...")
        engine_task = asyncio.create_task(engine.start())

        # 3秒後に緊急停止
        await asyncio.sleep(3)

        print("🚨 緊急停止を実行します...")
        engine.emergency_stop()

        # 状態確認
        status = engine.get_status()
        print(f"📊 緊急停止後の状態: {status['status']}")
        print(f"📊 保留中の注文: {status['pending_orders']} (すべてキャンセルされます)")

        try:
            await asyncio.wait_for(engine_task, timeout=1.0)
        except asyncio.TimeoutError:
            engine_task.cancel()

    except Exception as e:
        print(f"❌ エラー: {e}")


async def main():
    """メイン実行関数"""

    print("🎯 自動取引エンジン デモンストレーション")
    print("=" * 60)
    print()

    # ログレベルを設定（詳細すぎる場合は WARNING に変更）
    logging.basicConfig(level=logging.INFO)

    try:
        # 基本的な使用例
        await basic_trading_example()
        print()

        # リスク管理の例
        await risk_management_example()
        print()

        # 緊急停止の例
        await emergency_stop_example()

    except KeyboardInterrupt:
        print("\n👋 デモを終了します")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")

    print("\n✨ デモンストレーション完了")


if __name__ == "__main__":
    # 非同期実行
    asyncio.run(main())
