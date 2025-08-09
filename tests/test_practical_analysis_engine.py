"""
実際のAnalysisOnlyEngineを使用する統合テスト
モックを使わずに実際の動作をテストします
"""

from datetime import datetime
from decimal import Decimal

import pytest

from src.day_trade.automation.analysis_only_engine import (
    AnalysisOnlyEngine,
    AnalysisReport,
    AnalysisStatus,
    MarketAnalysis,
)


class TestAnalysisOnlyEngineIntegration:
    """AnalysisOnlyEngine の統合テスト"""

    def test_engine_initialization_real(self):
        """実際の初期化テスト"""
        symbols = ['7203', '6758', '4689']
        engine = AnalysisOnlyEngine(symbols)

        assert engine.symbols == symbols
        assert engine.status == AnalysisStatus.STOPPED
        assert engine.stats["total_analyses"] == 0
        assert engine.stats["successful_analyses"] == 0
        assert engine.stats["failed_analyses"] == 0
        assert len(engine.market_analyses) == 0
        assert len(engine.analysis_history) == 0

    def test_status_management(self):
        """ステータス管理テスト"""
        symbols = ['7203']
        engine = AnalysisOnlyEngine(symbols)

        # 初期ステータス確認
        assert engine.status == AnalysisStatus.STOPPED

        # get_status メソッドテスト
        status_info = engine.get_status()
        assert status_info["status"] == "stopped"
        assert status_info["monitored_symbols"] == 1
        assert status_info["trading_disabled"] is True
        assert status_info["safe_mode"] is True

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """非同期操作テスト"""
        symbols = ['7203']
        engine = AnalysisOnlyEngine(symbols)

        # 停止操作テスト
        await engine.stop()
        assert engine.status == AnalysisStatus.STOPPED
        assert engine._stop_event.is_set()

        # 一時停止・再開テスト
        engine.status = AnalysisStatus.RUNNING
        await engine.pause()
        assert engine.status == AnalysisStatus.PAUSED

        await engine.resume()
        assert engine.status == AnalysisStatus.RUNNING

    def test_data_classes(self):
        """データクラスの動作テスト"""
        # MarketAnalysis テスト
        analysis = MarketAnalysis(
            symbol="7203",
            current_price=Decimal("2500.0"),
            analysis_timestamp=datetime.now(),
        )

        assert analysis.symbol == "7203"
        assert analysis.current_price == Decimal("2500.0")
        assert analysis.signal is None
        assert analysis.volatility is None
        assert analysis.recommendations == []

        # オプションフィールド付きテスト
        analysis_with_data = MarketAnalysis(
            symbol="6758",
            current_price=Decimal("1800.0"),
            analysis_timestamp=datetime.now(),
            volatility=0.25,
            volume_trend="増加",
            price_trend="上昇",
            recommendations=["買い検討", "注意深く監視"]
        )

        assert analysis_with_data.volatility == 0.25
        assert analysis_with_data.volume_trend == "増加"
        assert analysis_with_data.price_trend == "上昇"
        assert len(analysis_with_data.recommendations) == 2

    def test_analysis_report_generation(self):
        """分析レポート生成テスト"""
        symbols = ['7203', '6758']
        engine = AnalysisOnlyEngine(symbols)

        # 模擬的な分析データを追加
        analysis1 = MarketAnalysis(
            symbol="7203",
            current_price=Decimal("2500.0"),
            analysis_timestamp=datetime.now(),
            recommendations=["テスト推奨1"]
        )

        analysis2 = MarketAnalysis(
            symbol="6758",
            current_price=Decimal("1800.0"),
            analysis_timestamp=datetime.now(),
            recommendations=["テスト推奨2"]
        )

        engine.market_analyses["7203"] = analysis1
        engine.market_analyses["6758"] = analysis2

        # レポート生成テスト
        report = engine._generate_analysis_report(150.0)

        assert isinstance(report, AnalysisReport)
        assert report.total_symbols == len(symbols)
        assert report.analyzed_symbols == 2
        assert report.analysis_time_ms == 150.0
        assert report.market_sentiment in ["強気", "弱気", "中性"]

    def test_statistics_update(self):
        """統計更新テスト"""
        symbols = ['7203']
        engine = AnalysisOnlyEngine(symbols)

        # 成功時の統計更新
        engine._update_statistics(100.0, success=True)

        assert engine.stats["total_analyses"] == 1
        assert engine.stats["successful_analyses"] == 1
        assert engine.stats["failed_analyses"] == 0
        assert engine.stats["avg_analysis_time"] == 100.0
        assert engine.stats["last_analysis"] is not None

        # 失敗時の統計更新
        engine._update_statistics(200.0, success=False)

        assert engine.stats["total_analyses"] == 2
        assert engine.stats["successful_analyses"] == 1
        assert engine.stats["failed_analyses"] == 1

    def test_analysis_data_access(self):
        """分析データアクセステスト"""
        symbols = ['7203', '6758']
        engine = AnalysisOnlyEngine(symbols)

        # 空の状態でのテスト
        assert engine.get_latest_analysis("7203") is None
        assert len(engine.get_all_analyses()) == 0
        assert engine.get_latest_report() is None

        # データ追加後のテスト
        analysis = MarketAnalysis(
            symbol="7203",
            current_price=Decimal("2500.0"),
            analysis_timestamp=datetime.now(),
        )
        engine.market_analyses["7203"] = analysis

        retrieved_analysis = engine.get_latest_analysis("7203")
        assert retrieved_analysis == analysis

        all_analyses = engine.get_all_analyses()
        assert len(all_analyses) == 1
        assert "7203" in all_analyses

    def test_recommendations_generation(self):
        """推奨事項生成テスト"""
        symbols = ['7203']
        engine = AnalysisOnlyEngine(symbols)

        # シグナルなしでの推奨事項
        recommendations = engine._generate_recommendations(
            "7203",
            Decimal("2500.0"),
            None
        )

        assert len(recommendations) > 0
        assert any("現在価格" in rec for rec in recommendations)
        assert any("シグナル: なし" in rec for rec in recommendations)
        assert any("分析情報です" in rec for rec in recommendations)
        assert any("自動取引は実行されません" in rec for rec in recommendations)

    def test_market_summary(self):
        """市場サマリーテスト"""
        symbols = ['7203']
        engine = AnalysisOnlyEngine(symbols)

        # レポートなしの場合
        summary = engine.get_market_summary()
        assert "error" in summary

        # レポートありの場合
        report = AnalysisReport(
            timestamp=datetime.now(),
            total_symbols=1,
            analyzed_symbols=1,
            strong_signals=1,
            medium_signals=0,
            weak_signals=0,
            market_sentiment="強気",
            top_recommendations=[{"symbol": "7203", "action": "buy"}],
            analysis_time_ms=100.0
        )
        engine.analysis_history.append(report)

        summary = engine.get_market_summary()
        assert "総銘柄数" in summary
        assert summary["総銘柄数"] == 1
        assert summary["強いシグナル"] == 1
        assert summary["市場センチメント"] == "強気"
        assert "注意" in summary

    def test_symbol_recommendations(self):
        """銘柄別推奨事項テスト"""
        symbols = ['7203']
        engine = AnalysisOnlyEngine(symbols)

        # データなしの場合
        recommendations = engine.get_symbol_recommendations("7203")
        assert len(recommendations) == 1
        assert "分析データがありません" in recommendations[0]

        # データありの場合
        analysis = MarketAnalysis(
            symbol="7203",
            current_price=Decimal("2500.0"),
            analysis_timestamp=datetime.now(),
            recommendations=["テスト推奨1", "テスト推奨2"]
        )
        engine.market_analyses["7203"] = analysis

        recommendations = engine.get_symbol_recommendations("7203")
        assert len(recommendations) == 2
        assert "テスト推奨1" in recommendations
        assert "テスト推奨2" in recommendations

    def test_volatility_calculation(self):
        """ボラティリティ計算テスト"""
        symbols = ['7203']
        engine = AnalysisOnlyEngine(symbols)

        import pandas as pd

        # 正常なデータでのテスト
        data = pd.DataFrame({
            'Close': [100.0, 102.0, 101.0, 103.0, 105.0]
        })

        volatility = engine._calculate_volatility(data)
        assert volatility is not None
        assert isinstance(volatility, float)
        assert volatility > 0

        # 空データでのテスト
        empty_data = pd.DataFrame()
        volatility_empty = engine._calculate_volatility(empty_data)
        assert volatility_empty is None

        # None データでのテスト
        volatility_none = engine._calculate_volatility(None)
        assert volatility_none is None

    def test_trend_analysis(self):
        """トレンド分析テスト"""
        symbols = ['7203']
        engine = AnalysisOnlyEngine(symbols)

        import pandas as pd

        # 出来高トレンドテスト
        volume_data = pd.DataFrame({
            'Volume': [1000, 1100, 1200, 1400, 1500, 1600]
        })

        volume_trend = engine._analyze_volume_trend(volume_data)
        assert volume_trend == "増加"

        # 価格トレンドテスト
        price_data = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120]
        })

        price_trend = engine._analyze_price_trend(price_data)
        assert price_trend == "上昇"
