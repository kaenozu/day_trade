"""
実用的なテストカバレッジ検証
実際のコード実行によるカバレッジ測定
"""

import asyncio

from src.day_trade.automation.analysis_only_engine import AnalysisOnlyEngine


def test_analysis_engine_basic():
    """基本的な分析エンジンテスト"""
    print("=== AnalysisOnlyEngine 基本動作テスト ===")

    symbols = ["7203", "6758", "4689"]
    engine = AnalysisOnlyEngine(symbols)

    # 基本プロパティの確認
    assert engine.symbols == symbols
    assert engine.status.value == "stopped"
    assert engine.stats["total_analyses"] == 0

    # レポート取得テスト
    latest_report = engine.get_latest_report()
    assert latest_report is None  # 初期状態では None

    # 状態取得テスト
    status = engine.get_status()
    assert "status" in status
    assert status["monitored_symbols"] == len(symbols)
    assert status["trading_disabled"] is True

    print("基本動作テスト完了")


async def test_analysis_engine_async():
    """非同期処理テスト"""
    print("=== AnalysisOnlyEngine 非同期処理テスト ===")

    symbols = ["7203"]
    engine = AnalysisOnlyEngine(symbols)

    # 停止テスト
    await engine.stop()
    assert engine.status.value == "stopped"

    # 一時停止・再開テスト
    engine.status = engine.status.__class__.RUNNING  # 手動で RUNNING に設定
    await engine.pause()
    assert engine.status.value == "paused"

    await engine.resume()
    assert engine.status.value == "running"

    print("非同期処理テスト完了")


def test_market_analysis_data_classes():
    """データクラステスト"""
    from datetime import datetime
    from decimal import Decimal

    from src.day_trade.automation.analysis_only_engine import (
        AnalysisReport,
        AnalysisStatus,
        MarketAnalysis,
    )

    print("=== データクラステスト ===")

    # MarketAnalysis テスト
    analysis = MarketAnalysis(
        symbol="7203",
        current_price=Decimal("2500.0"),
        analysis_timestamp=datetime.now(),
    )

    assert analysis.symbol == "7203"
    assert analysis.current_price == Decimal("2500.0")
    assert analysis.recommendations == []

    # AnalysisReport テスト
    report = AnalysisReport(
        timestamp=datetime.now(),
        total_symbols=3,
        analyzed_symbols=2,
        strong_signals=1,
        medium_signals=1,
        weak_signals=0,
        market_sentiment="中性",
        top_recommendations=[],
        analysis_time_ms=100.0,
    )

    assert report.total_symbols == 3
    assert report.analyzed_symbols == 2
    assert report.market_sentiment == "中性"

    # AnalysisStatus テスト
    assert AnalysisStatus.STOPPED.value == "stopped"
    assert AnalysisStatus.RUNNING.value == "running"

    print("データクラステスト完了")


def main():
    """メインテスト実行"""
    try:
        # 基本テスト
        test_analysis_engine_basic()

        # 非同期テスト
        asyncio.run(test_analysis_engine_async())

        # データクラステスト
        test_market_analysis_data_classes()

        print("\n=== 全テスト完了 ===")
        print("AnalysisOnlyEngine の基本機能は正常に動作しています")

    except Exception as e:
        print(f"テスト失敗: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
