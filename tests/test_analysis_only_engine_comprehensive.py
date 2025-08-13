#!/usr/bin/env python3
"""
AnalysisOnlyEngine 包括的ユニットテスト

カバレッジ向上のための詳細テストスイート
全メソッドと分岐条件をテストします
"""

import asyncio
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from day_trade.analysis.signals import SignalStrength, SignalType, TradingSignal
from day_trade.automation.analysis_only_engine import (
    AnalysisOnlyEngine,
    AnalysisReport,
    AnalysisStatus,
    MarketAnalysis,
)


class TestAnalysisOnlyEngineComprehensive:
    """AnalysisOnlyEngine 包括的テストクラス"""

    @pytest.fixture
    def sample_symbols(self):
        """テスト用銘柄リスト"""
        return ["7203.T", "6758.T", "9984.T"]

    @pytest.fixture
    def mock_signal_generator(self):
        """モックシグナルジェネレーター"""
        mock = Mock()
        mock.generate_signal.return_value = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=85.0,
            reasons=["強いモメンタム"],
            conditions_met={"momentum": True},
            timestamp=datetime.now(),
            price=Decimal("1500"),
        )
        return mock

    @pytest.fixture
    def mock_stock_fetcher(self):
        """モックストックフェッチャー"""
        mock = Mock()
        mock.get_current_price.return_value = {
            "current_price": 1500.0,
            "timestamp": datetime.now(),
        }
        mock.get_historical_data.return_value = self.create_sample_dataframe()
        return mock

    def create_sample_dataframe(self):
        """サンプル株価データフレーム作成"""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "Close": [1000 + i * 10 for i in range(30)],
                "Volume": [100000 + i * 1000 for i in range(30)],
                "High": [1010 + i * 10 for i in range(30)],
                "Low": [990 + i * 10 for i in range(30)],
            },
            index=dates,
        )

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_initialization_success(self, mock_safe_mode, sample_symbols):
        """正常な初期化テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        assert engine.symbols == sample_symbols
        assert engine.status == AnalysisStatus.STOPPED
        assert len(engine.market_analyses) == 0
        assert len(engine.analysis_history) == 0
        assert engine.stats["total_analyses"] == 0

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=False)
    def test_initialization_failure_not_safe_mode(self, mock_safe_mode, sample_symbols):
        """セーフモードでない場合の初期化失敗テスト"""
        with pytest.raises(ValueError, match="セーフモードでない場合は"):
            AnalysisOnlyEngine(symbols=sample_symbols)

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_initialization_with_custom_components(
        self, mock_safe_mode, sample_symbols, mock_signal_generator, mock_stock_fetcher
    ):
        """カスタムコンポーネント指定での初期化テスト"""
        engine = AnalysisOnlyEngine(
            symbols=sample_symbols,
            signal_generator=mock_signal_generator,
            stock_fetcher=mock_stock_fetcher,
            update_interval=60.0,
        )

        assert engine.signal_generator == mock_signal_generator
        assert engine.stock_fetcher == mock_stock_fetcher
        assert engine.update_interval == 60.0

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_start_stop_cycle(
        self, mock_safe_mode, sample_symbols, mock_stock_fetcher
    ):
        """開始・停止サイクルテスト"""
        engine = AnalysisOnlyEngine(
            symbols=sample_symbols,
            stock_fetcher=mock_stock_fetcher,
            update_interval=0.1,  # 高速テスト用
        )

        # 開始前状態確認
        assert engine.status == AnalysisStatus.STOPPED

        # 開始（短時間で停止）
        asyncio.create_task(engine.start())
        await asyncio.sleep(0.05)  # 短時間待機

        assert engine.status == AnalysisStatus.RUNNING

        # 停止
        await engine.stop()

        # 停止後状態確認
        assert engine.status == AnalysisStatus.STOPPED

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_already_running_warning(self, mock_safe_mode, sample_symbols):
        """既に実行中の場合の警告テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols, update_interval=0.1)

        # 状態を実行中に設定
        engine.status = AnalysisStatus.RUNNING

        with patch("day_trade.automation.analysis_only_engine.logger") as mock_logger:
            await engine.start()
            mock_logger.warning.assert_called_with("分析エンジンは既に実行中です")

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_pause_resume_cycle(self, mock_safe_mode, sample_symbols):
        """一時停止・再開サイクルテスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # 実行中状態に設定
        engine.status = AnalysisStatus.RUNNING

        # 一時停止
        await engine.pause()
        assert engine.status == AnalysisStatus.PAUSED

        # 再開
        await engine.resume()
        assert engine.status == AnalysisStatus.RUNNING

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_pause_resume_invalid_states(self, mock_safe_mode, sample_symbols):
        """無効な状態での一時停止・再開テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # 停止状態で一時停止（効果なし）
        engine.status = AnalysisStatus.STOPPED
        await engine.pause()
        assert engine.status == AnalysisStatus.STOPPED

        # 停止状態で再開（効果なし）
        await engine.resume()
        assert engine.status == AnalysisStatus.STOPPED

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_calculate_volatility_success(self, mock_safe_mode, sample_symbols):
        """ボラティリティ計算成功テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)
        data = self.create_sample_dataframe()

        volatility = engine._calculate_volatility(data)

        assert volatility is not None
        assert isinstance(volatility, float)
        assert volatility > 0

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_calculate_volatility_edge_cases(self, mock_safe_mode, sample_symbols):
        """ボラティリティ計算エッジケーステスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # None データ
        assert engine._calculate_volatility(None) is None

        # 空のデータフレーム
        empty_df = pd.DataFrame()
        assert engine._calculate_volatility(empty_df) is None

        # 1行のみのデータ
        single_row_df = pd.DataFrame({"Close": [100]})
        assert engine._calculate_volatility(single_row_df) is None

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_analyze_volume_trend_patterns(self, mock_safe_mode, sample_symbols):
        """出来高トレンド分析パターンテスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # 増加パターン
        increasing_data = pd.DataFrame(
            {"Volume": [1000, 1100, 1200, 2000, 2100, 2200]}  # 後半が増加
        )
        assert engine._analyze_volume_trend(increasing_data) == "増加"

        # 減少パターン
        decreasing_data = pd.DataFrame(
            {"Volume": [2000, 2100, 2200, 1000, 1100, 1200]}  # 後半が減少
        )
        assert engine._analyze_volume_trend(decreasing_data) == "減少"

        # 安定パターン
        stable_data = pd.DataFrame(
            {"Volume": [1500, 1400, 1600, 1500, 1400, 1600]}  # ほぼ同じ
        )
        assert engine._analyze_volume_trend(stable_data) == "安定"

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_analyze_volume_trend_edge_cases(self, mock_safe_mode, sample_symbols):
        """出来高トレンド分析エッジケーステスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # None データ
        assert engine._analyze_volume_trend(None) is None

        # 不十分なデータ
        insufficient_data = pd.DataFrame({"Volume": [100, 200]})
        assert engine._analyze_volume_trend(insufficient_data) is None

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_analyze_price_trend_patterns(self, mock_safe_mode, sample_symbols):
        """価格トレンド分析パターンテスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # 上昇トレンド
        uptrend_data = pd.DataFrame(
            {"Close": [900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350]}
        )
        assert engine._analyze_price_trend(uptrend_data) == "上昇"

        # 下降トレンド
        downtrend_data = pd.DataFrame(
            {"Close": [1350, 1300, 1250, 1200, 1150, 1100, 1050, 1000, 950, 900]}
        )
        assert engine._analyze_price_trend(downtrend_data) == "下降"

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_generate_recommendations_with_signal(self, mock_safe_mode, sample_symbols):
        """シグナル有りでの推奨事項生成テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=85.0,
            reasons=["テスト理由"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("1500"),
        )

        recommendations = engine._generate_recommendations(
            "7203.T", Decimal("1500"), signal
        )

        assert len(recommendations) > 5
        assert any("1,500円" in rec for rec in recommendations)
        assert any("買い注目" in rec for rec in recommendations)
        assert any("信頼度: 高" in rec for rec in recommendations)
        assert any("※ これは分析情報です" in rec for rec in recommendations)

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_generate_recommendations_without_signal(
        self, mock_safe_mode, sample_symbols
    ):
        """シグナル無しでの推奨事項生成テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        recommendations = engine._generate_recommendations(
            "7203.T", Decimal("1500"), None
        )

        assert len(recommendations) > 3
        assert any("シグナル: なし" in rec for rec in recommendations)

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_generate_recommendations_confidence_levels(
        self, mock_safe_mode, sample_symbols
    ):
        """信頼度レベル別推奨事項生成テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # 高信頼度
        high_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=85.0,
            reasons=["高信頼度"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("1000"),
        )
        high_recs = engine._generate_recommendations(
            "TEST", Decimal("1000"), high_signal
        )
        assert any("信頼度: 高" in rec for rec in high_recs)

        # 中信頼度
        medium_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=65.0,
            reasons=["中信頼度"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("1000"),
        )
        medium_recs = engine._generate_recommendations(
            "TEST", Decimal("1000"), medium_signal
        )
        assert any("信頼度: 中" in rec for rec in medium_recs)

        # 低信頼度
        low_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=45.0,
            reasons=["低信頼度"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("1000"),
        )
        low_recs = engine._generate_recommendations("TEST", Decimal("1000"), low_signal)
        assert any("信頼度: 低" in rec for rec in low_recs)

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_generate_analysis_report(self, mock_safe_mode, sample_symbols):
        """分析レポート生成テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # モック分析データ設定
        engine.market_analyses = {
            "7203.T": MarketAnalysis(
                symbol="7203.T",
                current_price=Decimal("1500"),
                analysis_timestamp=datetime.now(),
                signal=TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.STRONG,
                    confidence=85.0,
                    reasons=["強いシグナル"],
                    conditions_met={"test": True},
                    timestamp=datetime.now(),
                    price=Decimal("1500"),
                ),
            ),
            "6758.T": MarketAnalysis(
                symbol="6758.T",
                current_price=Decimal("2000"),
                analysis_timestamp=datetime.now(),
                signal=TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MEDIUM,
                    confidence=65.0,
                    reasons=["中程度シグナル"],
                    conditions_met={"test": True},
                    timestamp=datetime.now(),
                    price=Decimal("2000"),
                ),
            ),
        }

        report = engine._generate_analysis_report(1.5)

        assert isinstance(report, AnalysisReport)
        assert report.total_symbols == len(sample_symbols)
        assert report.analyzed_symbols == 2
        assert report.strong_signals == 1
        assert report.medium_signals == 1
        assert report.analysis_time_ms == 1500.0
        assert len(report.top_recommendations) >= 1

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_market_sentiment_calculation(self, mock_safe_mode, sample_symbols):
        """市場センチメント計算テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # 強気シナリオ（強いシグナルが多い）
        engine.market_analyses = {
            "A": MarketAnalysis(
                "A",
                Decimal("100"),
                datetime.now(),
                TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.STRONG,
                    confidence=85.0,
                    reasons=["強い"],
                    conditions_met={"test": True},
                    timestamp=datetime.now(),
                    price=Decimal("100"),
                ),
            ),
            "B": MarketAnalysis(
                "B",
                Decimal("200"),
                datetime.now(),
                TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.STRONG,
                    confidence=80.0,
                    reasons=["強い"],
                    conditions_met={"test": True},
                    timestamp=datetime.now(),
                    price=Decimal("200"),
                ),
            ),
            "C": MarketAnalysis(
                "C",
                Decimal("300"),
                datetime.now(),
                TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MEDIUM,
                    confidence=75.0,
                    reasons=["中程度"],
                    conditions_met={"test": True},
                    timestamp=datetime.now(),
                    price=Decimal("300"),
                ),
            ),
        }
        report = engine._generate_analysis_report(1.0)
        assert report.market_sentiment == "強気"

        # シグナル無しシナリオ
        engine.market_analyses = {
            "A": MarketAnalysis("A", Decimal("100"), datetime.now(), None),
        }
        report = engine._generate_analysis_report(1.0)
        assert report.market_sentiment == "中性"

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_update_statistics(self, mock_safe_mode, sample_symbols):
        """統計情報更新テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # 初回成功
        engine._update_statistics(1.5, success=True)
        assert engine.stats["total_analyses"] == 1
        assert engine.stats["successful_analyses"] == 1
        assert engine.stats["failed_analyses"] == 0
        assert engine.stats["avg_analysis_time"] == 1.5

        # 2回目失敗
        engine._update_statistics(2.0, success=False)
        assert engine.stats["total_analyses"] == 2
        assert engine.stats["successful_analyses"] == 1
        assert engine.stats["failed_analyses"] == 1
        # 移動平均計算確認
        expected_avg = 0.1 * 2.0 + 0.9 * 1.5
        assert abs(engine.stats["avg_analysis_time"] - expected_avg) < 0.001

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_data_retrieval_methods(self, mock_safe_mode, sample_symbols):
        """データ取得メソッドテスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # テストデータ設定
        test_analysis = MarketAnalysis(
            symbol="7203.T",
            current_price=Decimal("1500"),
            analysis_timestamp=datetime.now(),
            recommendations=["テスト推奨"],
        )
        engine.market_analyses["7203.T"] = test_analysis

        test_report = AnalysisReport(
            timestamp=datetime.now(),
            total_symbols=3,
            analyzed_symbols=1,
            strong_signals=0,
            medium_signals=0,
            weak_signals=0,
            market_sentiment="中性",
            top_recommendations=[],
            analysis_time_ms=100.0,
        )
        engine.analysis_history.append(test_report)

        # メソッドテスト
        assert engine.get_latest_analysis("7203.T") == test_analysis
        assert engine.get_latest_analysis("INVALID") is None

        all_analyses = engine.get_all_analyses()
        assert "7203.T" in all_analyses
        assert len(all_analyses) == 1

        assert engine.get_latest_report() == test_report

        history = engine.get_analysis_history(5)
        assert len(history) == 1
        assert history[0] == test_report

        recommendations = engine.get_symbol_recommendations("7203.T")
        assert "テスト推奨" in recommendations

        empty_recommendations = engine.get_symbol_recommendations("INVALID")
        assert "INVALIDの分析データがありません" in empty_recommendations

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_get_status(self, mock_safe_mode, sample_symbols):
        """状態取得テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        status = engine.get_status()

        assert status["status"] == AnalysisStatus.STOPPED.value
        assert status["monitored_symbols"] == len(sample_symbols)
        assert status["analyzed_symbols"] == 0
        assert status["safe_mode"] is True
        assert status["trading_disabled"] is True
        assert "stats" in status

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_get_market_summary_with_data(self, mock_safe_mode, sample_symbols):
        """データ有りでの市場サマリー取得テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        # テストレポート追加
        test_report = AnalysisReport(
            timestamp=datetime.now(),
            total_symbols=3,
            analyzed_symbols=2,
            strong_signals=1,
            medium_signals=1,
            weak_signals=0,
            market_sentiment="強気",
            top_recommendations=[{"symbol": "7203.T", "action": "buy"}],
            analysis_time_ms=150.0,
        )
        engine.analysis_history.append(test_report)

        summary = engine.get_market_summary()

        assert summary["総銘柄数"] == 3
        assert summary["分析済み銘柄数"] == 2
        assert summary["強いシグナル"] == 1
        assert summary["市場センチメント"] == "強気"
        assert "注意" in summary

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_get_market_summary_no_data(self, mock_safe_mode, sample_symbols):
        """データ無しでの市場サマリー取得テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        summary = engine.get_market_summary()
        assert "error" in summary
        assert summary["error"] == "分析データがありません"

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_perform_market_analysis_success(
        self, mock_safe_mode, sample_symbols, mock_stock_fetcher, mock_signal_generator
    ):
        """市場分析実行成功テスト"""
        engine = AnalysisOnlyEngine(
            symbols=["7203.T"],
            stock_fetcher=mock_stock_fetcher,
            signal_generator=mock_signal_generator,
        )

        await engine._perform_market_analysis()

        assert len(engine.market_analyses) == 1
        assert "7203.T" in engine.market_analyses
        analysis = engine.market_analyses["7203.T"]
        assert analysis.current_price == Decimal("1500.0")
        assert analysis.signal is not None

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_perform_market_analysis_data_fetch_failure(
        self,
        mock_safe_mode,
        sample_symbols,
    ):
        """データ取得失敗時の市場分析テスト"""
        mock_stock_fetcher = Mock()
        mock_stock_fetcher.get_current_price.return_value = None  # 失敗をシミュレート

        engine = AnalysisOnlyEngine(
            symbols=["7203.T"], stock_fetcher=mock_stock_fetcher
        )

        await engine._perform_market_analysis()

        # データ取得失敗により分析結果は空
        assert len(engine.market_analyses) == 0

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_error_handling_in_analysis_loop(
        self, mock_safe_mode, sample_symbols
    ):
        """分析ループでのエラーハンドリングテスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols, update_interval=0.01)

        # _perform_market_analysis でエラーを発生させる
        with patch.object(
            engine, "_perform_market_analysis", side_effect=Exception("テストエラー")
        ):
            # 短時間実行してエラー処理確認
            asyncio.create_task(engine._analysis_loop())
            await asyncio.sleep(0.05)
            engine._stop_event.set()

            # 統計にエラーが記録されることを確認
            assert engine.stats["failed_analyses"] > 0

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    def test_analysis_with_empty_historical_data(self, mock_safe_mode, sample_symbols):
        """空の履歴データでの分析テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols)

        empty_df = pd.DataFrame()

        # 各分析メソッドが空データを適切に処理するかテスト
        assert engine._calculate_volatility(empty_df) is None
        assert engine._analyze_volume_trend(empty_df) is None
        assert engine._analyze_price_trend(empty_df) is None

    @patch("day_trade.automation.analysis_only_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_analysis_loop_pause_behavior(self, mock_safe_mode, sample_symbols):
        """分析ループの一時停止動作テスト"""
        engine = AnalysisOnlyEngine(symbols=sample_symbols, update_interval=0.01)

        # 一時停止状態に設定
        engine.status = AnalysisStatus.PAUSED

        with patch.object(engine, "_perform_market_analysis") as mock_analysis:
            # 短時間実行
            asyncio.create_task(engine._analysis_loop())
            await asyncio.sleep(0.02)
            engine._stop_event.set()

            # 一時停止中は分析が実行されないことを確認
            mock_analysis.assert_not_called()


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v", "--tb=short"])
