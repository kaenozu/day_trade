#!/usr/bin/env python3
"""
TradingEngine 包括的ユニットテスト

市場分析エンジンコアの全機能をテストします
セーフモード動作を前提としたテスト設計
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
from day_trade.automation.trading_engine import (
    EngineStatus,
    MarketData,
    OrderRequest,
    OrderType,
    RiskParameters,
    TradingEngine,
)
from day_trade.core.trade_manager import Trade, TradeType


class TestTradingEngineComprehensive:
    """TradingEngine 包括的テストクラス"""

    @pytest.fixture
    def sample_symbols(self):
        """テスト用銘柄リスト"""
        return ["7203.T", "6758.T", "9984.T"]

    @pytest.fixture
    def sample_risk_params(self):
        """テスト用リスクパラメータ"""
        return RiskParameters(
            max_position_size=Decimal("500000"),
            max_daily_loss=Decimal("25000"),
            max_open_positions=5,
            stop_loss_ratio=Decimal("0.03"),
            take_profit_ratio=Decimal("0.06"),
        )

    @pytest.fixture
    def mock_trade_manager(self):
        """モックトレードマネージャー"""
        mock = Mock()
        mock.add_trade.return_value = None
        return mock

    @pytest.fixture
    def mock_signal_generator(self):
        """モックシグナルジェネレーター"""
        mock = Mock()
        mock.generate_signal.return_value = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=75.0,
            reasons=["テストシグナル"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("2000"),
        )
        return mock

    @pytest.fixture
    def mock_stock_fetcher(self):
        """モックストックフェッチャー"""
        mock = Mock()
        mock.get_current_price.return_value = {
            "current_price": 2000.0,
            "volume": 100000,
            "timestamp": datetime.now(),
        }
        mock.get_historical_data.return_value = self.create_sample_dataframe()
        return mock

    def create_sample_dataframe(self):
        """サンプル株価データフレーム作成"""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "Close": [2000 + i * 10 for i in range(30)],
                "Volume": [100000 + i * 1000 for i in range(30)],
                "High": [2010 + i * 10 for i in range(30)],
                "Low": [1990 + i * 10 for i in range(30)],
            },
            index=dates,
        )

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_initialization_success(
        self, mock_safe_mode, sample_symbols, sample_risk_params
    ):
        """正常な初期化テスト"""
        engine = TradingEngine(symbols=sample_symbols, risk_params=sample_risk_params)

        assert engine.symbols == sample_symbols
        assert engine.status == EngineStatus.STOPPED
        assert engine.risk_params == sample_risk_params
        assert len(engine.market_data) == 0
        assert len(engine.active_positions) == 0
        assert len(engine.pending_orders) == 0

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=False)
    def test_initialization_failure_not_safe_mode(self, mock_safe_mode, sample_symbols):
        """セーフモードでない場合の初期化失敗テスト"""
        with pytest.raises(ValueError, match="安全設定が無効です"):
            TradingEngine(symbols=sample_symbols)

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_initialization_with_custom_components(
        self,
        mock_safe_mode,
        sample_symbols,
        mock_trade_manager,
        mock_signal_generator,
        mock_stock_fetcher,
        sample_risk_params,
    ):
        """カスタムコンポーネント指定での初期化テスト"""
        engine = TradingEngine(
            symbols=sample_symbols,
            trade_manager=mock_trade_manager,
            signal_generator=mock_signal_generator,
            stock_fetcher=mock_stock_fetcher,
            risk_params=sample_risk_params,
            update_interval=2.0,
        )

        assert engine.trade_manager == mock_trade_manager
        assert engine.signal_generator == mock_signal_generator
        assert engine.stock_fetcher == mock_stock_fetcher
        assert engine.update_interval == 2.0

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_start_stop_cycle(
        self, mock_safe_mode, sample_symbols, mock_stock_fetcher
    ):
        """開始・停止サイクルテスト"""
        engine = TradingEngine(
            symbols=sample_symbols,
            stock_fetcher=mock_stock_fetcher,
            update_interval=0.1,  # 高速テスト用
        )

        # 開始前状態確認
        assert engine.status == EngineStatus.STOPPED

        # 開始（短時間で停止）
        asyncio.create_task(engine.start())
        await asyncio.sleep(0.05)  # 短時間待機

        assert engine.status == EngineStatus.RUNNING

        # 停止
        await engine.stop()

        # 停止後状態確認
        assert engine.status == EngineStatus.STOPPED
        assert len(engine.pending_orders) == 0

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_already_running_warning(self, mock_safe_mode, sample_symbols):
        """既に実行中の場合の警告テスト"""
        engine = TradingEngine(symbols=sample_symbols, update_interval=0.1)

        # 状態を実行中に設定
        engine.status = EngineStatus.RUNNING

        with patch("day_trade.automation.trading_engine.logger") as mock_logger:
            await engine.start()
            mock_logger.warning.assert_called_with("市場分析エンジンは既に実行中です")

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_pause_resume_cycle(self, mock_safe_mode, sample_symbols):
        """一時停止・再開サイクルテスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 実行中状態に設定
        engine.status = EngineStatus.RUNNING

        # 一時停止
        await engine.pause()
        assert engine.status == EngineStatus.PAUSED

        # 再開
        await engine.resume()
        assert engine.status == EngineStatus.RUNNING

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_pause_resume_invalid_states(self, mock_safe_mode, sample_symbols):
        """無効な状態での一時停止・再開テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 停止状態で一時停止（効果なし）
        engine.status = EngineStatus.STOPPED
        await engine.pause()
        assert engine.status == EngineStatus.STOPPED

        # 停止状態で再開（効果なし）
        await engine.resume()
        assert engine.status == EngineStatus.STOPPED

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_update_market_data_success(
        self, mock_safe_mode, sample_symbols, mock_stock_fetcher
    ):
        """市場データ更新成功テスト"""
        engine = TradingEngine(symbols=sample_symbols, stock_fetcher=mock_stock_fetcher)

        await engine._update_market_data()

        assert len(engine.market_data) == len(sample_symbols)
        for symbol in sample_symbols:
            assert symbol in engine.market_data
            assert engine.market_data[symbol].price == Decimal("2000.0")
            assert engine.market_data[symbol].volume == 100000

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_update_market_data_partial_failure(
        self, mock_safe_mode, sample_symbols
    ):
        """市場データ更新部分失敗テスト"""
        mock_stock_fetcher = Mock()
        # 最初の銘柄は成功、2番目は失敗、3番目は成功
        mock_stock_fetcher.get_current_price.side_effect = [
            {"current_price": 2000.0, "volume": 100000},
            None,  # 失敗をシミュレート
            {"current_price": 3000.0, "volume": 200000},
        ]

        engine = TradingEngine(symbols=sample_symbols, stock_fetcher=mock_stock_fetcher)

        await engine._update_market_data()

        # 成功した銘柄のみデータが登録される
        assert len(engine.market_data) == 2
        assert sample_symbols[0] in engine.market_data
        assert sample_symbols[1] not in engine.market_data  # 失敗
        assert sample_symbols[2] in engine.market_data

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_generate_signals_success(
        self, mock_safe_mode, sample_symbols, mock_stock_fetcher, mock_signal_generator
    ):
        """シグナル生成成功テスト"""
        engine = TradingEngine(
            symbols=sample_symbols,
            stock_fetcher=mock_stock_fetcher,
            signal_generator=mock_signal_generator,
        )

        # 市場データを事前に設定
        for symbol in sample_symbols:
            engine.market_data[symbol] = MarketData(
                symbol=symbol,
                price=Decimal("2000"),
                volume=100000,
                timestamp=datetime.now(),
            )

        signals = await engine._generate_signals()

        assert len(signals) == len(sample_symbols)
        for symbol, signal in signals:
            assert symbol in sample_symbols
            assert isinstance(signal, TradingSignal)

        # 統計が更新されていることを確認
        assert engine.execution_stats["signals_generated"] == len(sample_symbols)

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_generate_signals_no_market_data(
        self, mock_safe_mode, sample_symbols
    ):
        """市場データなしでのシグナル生成テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        signals = await engine._generate_signals()

        # 市場データがないのでシグナルは生成されない
        assert len(signals) == 0
        assert engine.execution_stats["signals_generated"] == 0

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_analyze_signals_high_confidence(
        self, mock_safe_mode, sample_symbols
    ):
        """高信頼度シグナルの分析テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 高信頼度シグナルを作成
        high_confidence_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=85.0,
            reasons=["強いモメンタム"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("2000"),
        )

        signals = [("7203.T", high_confidence_signal)]

        await engine._analyze_signals(signals)

        # 分析完了統計が更新されることを確認
        assert engine.execution_stats["analysis_completed"] == 1

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_analyze_signals_low_confidence(self, mock_safe_mode, sample_symbols):
        """低信頼度シグナルの分析テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 低信頼度シグナルを作成
        low_confidence_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=50.0,
            reasons=["弱いシグナル"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("2000"),
        )

        signals = [("7203.T", low_confidence_signal)]

        await engine._analyze_signals(signals)

        # 低信頼度なので分析完了統計は更新されない
        assert engine.execution_stats["analysis_completed"] == 0

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_log_signal_analysis_buy_signal(self, mock_safe_mode, sample_symbols):
        """買いシグナル分析ログテスト"""
        engine = TradingEngine(symbols=sample_symbols)

        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=80.0,
            reasons=["上昇トレンド"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("1000"),
        )

        with patch("day_trade.automation.trading_engine.logger") as mock_logger:
            engine._log_signal_analysis("7203.T", signal)

            # 買い推奨がログされることを確認
            mock_logger.info.assert_any_call(
                "【分析結果】取引提案 - 7203.T: 買い推奨 (信頼度: 80.0%, 強度: strong)"
            )

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_log_signal_analysis_sell_signal(self, mock_safe_mode, sample_symbols):
        """売りシグナル分析ログテスト"""
        engine = TradingEngine(symbols=sample_symbols)

        signal = TradingSignal(
            signal_type=SignalType.SELL,
            strength=SignalStrength.MEDIUM,
            confidence=75.0,
            reasons=["下降トレンド"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=Decimal("1000"),
        )

        with patch("day_trade.automation.trading_engine.logger") as mock_logger:
            engine._log_signal_analysis("7203.T", signal)

            # 売り推奨がログされることを確認
            mock_logger.info.assert_any_call(
                "【分析結果】取引提案 - 7203.T: 売り推奨 (信頼度: 75.0%, 強度: medium)"
            )

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_monitor_positions_buy_profit(self, mock_safe_mode, sample_symbols):
        """買いポジション利益確定推奨テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 買いポジション設定
        buy_trade = Trade(
            symbol="7203.T",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("1000"),  # エントリー価格
            timestamp=datetime.now(),
        )
        engine.active_positions["7203.T"] = [buy_trade]

        # 現在価格（利益確定レベル）
        engine.market_data["7203.T"] = MarketData(
            symbol="7203.T",
            price=Decimal("1060"),  # 6%上昇
            volume=100000,
            timestamp=datetime.now(),
        )

        with patch("day_trade.automation.trading_engine.logger") as mock_logger:
            await engine._monitor_positions()

            # 利益確定推奨がログされることを確認
            mock_logger.info.assert_any_call(
                "【分析】7203.T: 利益確定推奨 - 利益率 6.00% (+6000円)"
            )

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_monitor_positions_buy_loss(self, mock_safe_mode, sample_symbols):
        """買いポジション損切り推奨テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 買いポジション設定
        buy_trade = Trade(
            symbol="7203.T",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("1000"),  # エントリー価格
            timestamp=datetime.now(),
        )
        engine.active_positions["7203.T"] = [buy_trade]

        # 現在価格（損切りレベル）
        engine.market_data["7203.T"] = MarketData(
            symbol="7203.T",
            price=Decimal("970"),  # 3%下落
            volume=100000,
            timestamp=datetime.now(),
        )

        with patch("day_trade.automation.trading_engine.logger") as mock_logger:
            await engine._monitor_positions()

            # 損切り推奨がログされることを確認
            mock_logger.info.assert_any_call(
                "【分析】7203.T: 損切り推奨 - 損失率 -3.00% (-3000円)"
            )

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_monitor_positions_sell_profit(self, mock_safe_mode, sample_symbols):
        """売りポジション利益確定推奨テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 売りポジション設定
        sell_trade = Trade(
            symbol="7203.T",
            trade_type=TradeType.SELL,
            quantity=100,
            price=Decimal("1000"),  # エントリー価格
            timestamp=datetime.now(),
        )
        engine.active_positions["7203.T"] = [sell_trade]

        # 現在価格（利益確定レベル）
        engine.market_data["7203.T"] = MarketData(
            symbol="7203.T",
            price=Decimal("940"),  # 6%下落（売りなので利益）
            volume=100000,
            timestamp=datetime.now(),
        )

        with patch("day_trade.automation.trading_engine.logger") as mock_logger:
            await engine._monitor_positions()

            # 利益確定推奨がログされることを確認
            mock_logger.info.assert_any_call(
                "【分析】7203.T: 利益確定推奨 - 利益率 6.00% (+6000円)"
            )

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_calculate_daily_pnl_with_positions(self, mock_safe_mode, sample_symbols):
        """ポジション有りでの日次損益計算テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 今日の取引を設定
        today_trade = Trade(
            symbol="7203.T",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("1000"),
            timestamp=datetime.now(),  # 今日の取引
        )
        engine.active_positions["7203.T"] = [today_trade]

        # 現在価格設定
        engine.market_data["7203.T"] = MarketData(
            symbol="7203.T",
            price=Decimal("1050"),  # 50円上昇
            volume=100000,
            timestamp=datetime.now(),
        )

        daily_pnl = engine._calculate_daily_pnl()

        # 100株 × 50円 = 5000円の利益
        assert daily_pnl == Decimal("5000")

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_calculate_daily_pnl_no_positions(self, mock_safe_mode, sample_symbols):
        """ポジション無しでの日次損益計算テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        daily_pnl = engine._calculate_daily_pnl()

        # ポジションがないので損益は0
        assert daily_pnl == Decimal("0")

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_update_performance_stats(self, mock_safe_mode, sample_symbols):
        """パフォーマンス統計更新テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 初回更新
        engine._update_performance_stats(1.5)
        assert engine.execution_stats["avg_execution_time"] == 1.5
        assert engine.execution_stats["last_update"] is not None

        # 2回目更新（移動平均計算）
        engine._update_performance_stats(2.0)
        expected_avg = 0.1 * 2.0 + 0.9 * 1.5  # 指数移動平均
        assert abs(engine.execution_stats["avg_execution_time"] - expected_avg) < 0.001

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_get_status(self, mock_safe_mode, sample_symbols):
        """状態取得テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # テストデータ設定
        engine.active_positions["7203.T"] = [
            Trade("7203.T", TradeType.BUY, 100, Decimal("1000"), datetime.now())
        ]
        engine.pending_orders = [
            OrderRequest("6758.T", OrderType.LIMIT, TradeType.BUY, 50, Decimal("2000"))
        ]

        status = engine.get_status()

        assert status["status"] == EngineStatus.STOPPED.value
        assert status["monitored_symbols"] == len(sample_symbols)
        assert status["active_positions"] == 1
        assert status["pending_analysis"] == 1
        assert status["safe_mode"] is True
        assert "execution_stats" in status
        assert "market_data_age" in status

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_emergency_stop(self, mock_safe_mode, sample_symbols):
        """緊急停止テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 実行中状態に設定
        engine.status = EngineStatus.RUNNING
        engine.pending_orders = [
            OrderRequest("7203.T", OrderType.MARKET, TradeType.BUY, 100)
        ]

        with patch("day_trade.automation.trading_engine.logger") as mock_logger:
            engine.emergency_stop()

            assert engine.status == EngineStatus.STOPPED
            assert len(engine.pending_orders) == 0
            assert engine._stop_event.is_set()

            # 緊急停止ログが出力されることを確認
            mock_logger.critical.assert_any_call(
                "緊急停止が実行されました（分析エンジンのみ停止）"
            )

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_add_manual_trade(self, mock_safe_mode, sample_symbols, mock_trade_manager):
        """手動取引追加テスト"""
        engine = TradingEngine(symbols=sample_symbols, trade_manager=mock_trade_manager)

        manual_trade = Trade(
            symbol="7203.T",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2000"),
            timestamp=datetime.now(),
        )

        engine.add_manual_trade(manual_trade)

        # TradeManager に記録されることを確認
        mock_trade_manager.add_trade.assert_called_once_with(manual_trade)

        # アクティブポジションに追加されることを確認
        assert "7203.T" in engine.active_positions
        assert manual_trade in engine.active_positions["7203.T"]

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_get_trading_suggestions_with_data(self, mock_safe_mode, sample_symbols):
        """データ有りでの取引提案取得テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        # 市場データ設定
        engine.market_data["7203.T"] = MarketData(
            symbol="7203.T",
            price=Decimal("1500"),
            volume=100000,
            timestamp=datetime.now(),
        )

        suggestions = engine.get_trading_suggestions("7203.T")

        assert len(suggestions) > 0
        assert any("現在価格: 1500円" in s for s in suggestions)
        assert any("※ これは分析情報です" in s for s in suggestions)
        assert any("※ 自動取引は完全に無効化されています" in s for s in suggestions)

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_get_trading_suggestions_no_data(self, mock_safe_mode, sample_symbols):
        """データ無しでの取引提案取得テスト"""
        engine = TradingEngine(symbols=sample_symbols)

        suggestions = engine.get_trading_suggestions("INVALID")

        # データがない場合でも基本的な情報は返される
        assert len(suggestions) == 0 or any("分析情報" in s for s in suggestions)

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_main_loop_pause_behavior(self, mock_safe_mode, sample_symbols):
        """メインループの一時停止動作テスト"""
        engine = TradingEngine(symbols=sample_symbols, update_interval=0.01)

        # 一時停止状態に設定
        engine.status = EngineStatus.PAUSED

        with patch.object(engine, "_update_market_data") as mock_update:
            # 短時間実行
            asyncio.create_task(engine._main_loop())
            await asyncio.sleep(0.02)
            engine._stop_event.set()

            # 一時停止中は市場データ更新が実行されないことを確認
            mock_update.assert_not_called()

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    @pytest.mark.asyncio
    async def test_main_loop_error_handling(self, mock_safe_mode, sample_symbols):
        """メインループでのエラーハンドリングテスト"""
        engine = TradingEngine(symbols=sample_symbols, update_interval=0.01)

        # _update_market_data でエラーを発生させる
        with patch.object(
            engine, "_update_market_data", side_effect=Exception("テストエラー")
        ):
            # 短時間実行してエラー処理確認
            await engine._main_loop()

            # エラー発生によりステータスがERRORになることを確認
            assert engine.status == EngineStatus.ERROR

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_order_request_creation(self, mock_safe_mode):
        """注文リクエスト作成テスト"""
        order = OrderRequest(
            symbol="7203.T",
            order_type=OrderType.LIMIT,
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2000"),
        )

        assert order.symbol == "7203.T"
        assert order.order_type == OrderType.LIMIT
        assert order.trade_type == TradeType.BUY
        assert order.quantity == 100
        assert order.price == Decimal("2000")
        assert order.timestamp is not None

    @patch("day_trade.automation.trading_engine.is_safe_mode", return_value=True)
    def test_risk_parameters_creation(self, mock_safe_mode):
        """リスクパラメータ作成テスト"""
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            max_daily_loss=Decimal("10000"),
            max_open_positions=3,
            stop_loss_ratio=Decimal("0.05"),
            take_profit_ratio=Decimal("0.10"),
        )

        assert risk_params.max_position_size == Decimal("100000")
        assert risk_params.max_daily_loss == Decimal("10000")
        assert risk_params.max_open_positions == 3
        assert risk_params.stop_loss_ratio == Decimal("0.05")
        assert risk_params.take_profit_ratio == Decimal("0.10")


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v", "--tb=short"])
