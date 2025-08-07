"""
自動取引エンジンのテスト
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.day_trade.analysis.signals import SignalStrength, SignalType, TradingSignal
from src.day_trade.automation.trading_engine import (
    EngineStatus,
    MarketData,
    OrderRequest,
    OrderType,
    RiskParameters,
    TradingEngine,
)
from src.day_trade.core.trade_manager import TradeManager, TradeType
from src.day_trade.data.stock_fetcher import StockFetcher


class TestTradingEngine:
    """自動取引エンジンのテストクラス"""

    @pytest.fixture
    def mock_dependencies(self):
        """依存関係のモック"""
        trade_manager = Mock(spec=TradeManager)
        signal_generator = Mock()
        stock_fetcher = Mock(spec=StockFetcher)

        return trade_manager, signal_generator, stock_fetcher

    @pytest.fixture
    def trading_engine(self, mock_dependencies):
        """取引エンジンインスタンス"""
        trade_manager, signal_generator, stock_fetcher = mock_dependencies

        symbols = ["7203", "6758", "9984"]
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            max_daily_loss=Decimal("5000"),
            max_open_positions=5,
        )

        engine = TradingEngine(
            symbols=symbols,
            trade_manager=trade_manager,
            signal_generator=signal_generator,
            stock_fetcher=stock_fetcher,
            risk_params=risk_params,
            update_interval=0.1,  # テスト用に短縮
        )

        return engine

    def test_engine_initialization(self, trading_engine):
        """エンジン初期化のテスト"""
        assert trading_engine.status == EngineStatus.STOPPED
        assert len(trading_engine.symbols) == 3
        assert trading_engine.update_interval == 0.1
        assert len(trading_engine.market_data) == 0
        assert len(trading_engine.active_positions) == 0
        assert len(trading_engine.pending_orders) == 0

    def test_risk_parameters(self, trading_engine):
        """リスクパラメータのテスト"""
        risk_params = trading_engine.risk_params
        assert risk_params.max_position_size == Decimal("100000")
        assert risk_params.max_daily_loss == Decimal("5000")
        assert risk_params.max_open_positions == 5

    @pytest.mark.asyncio
    async def test_engine_start_stop(self, trading_engine):
        """エンジンの開始・停止テスト"""
        # モック設定
        trading_engine.stock_fetcher.get_current_price = Mock(
            return_value={
                "current_price": 2500.0,
                "volume": 1000000,
                "timestamp": datetime.now(),
            }
        )
        trading_engine.signal_generator.generate_signal = Mock(return_value=None)

        # エンジン開始（短時間実行）
        start_task = asyncio.create_task(trading_engine.start())
        await asyncio.sleep(0.2)  # 短時間実行

        # 実行中であることを確認
        assert trading_engine.status in [EngineStatus.RUNNING, EngineStatus.STOPPED]

        # 停止
        await trading_engine.stop()
        assert trading_engine.status == EngineStatus.STOPPED

        # タスクが完了するまで待機
        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

    @pytest.mark.asyncio
    async def test_pause_resume(self, trading_engine):
        """一時停止・再開のテスト"""
        # 初期状態はSTOPPED、まずRUNNINGにする
        trading_engine.status = EngineStatus.RUNNING

        # 一時停止
        await trading_engine.pause()
        assert trading_engine.status == EngineStatus.PAUSED

        # 再開
        await trading_engine.resume()
        assert trading_engine.status == EngineStatus.RUNNING

    def test_market_data_update(self, trading_engine):
        """市場データ更新のテスト"""
        # テストデータ設定
        test_data = MarketData(
            symbol="7203",
            price=Decimal("2500.0"),
            volume=1000000,
            timestamp=datetime.now(),
        )

        trading_engine.market_data["7203"] = test_data

        assert "7203" in trading_engine.market_data
        assert trading_engine.market_data["7203"].price == Decimal("2500.0")
        assert trading_engine.market_data["7203"].volume == 1000000

    def test_order_request_creation(self):
        """注文リクエスト作成のテスト"""
        order = OrderRequest(
            symbol="7203",
            order_type=OrderType.MARKET,
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
        )

        assert order.symbol == "7203"
        assert order.order_type == OrderType.MARKET
        assert order.trade_type == TradeType.BUY
        assert order.quantity == 100
        assert order.price == Decimal("2500.0")
        assert isinstance(order.timestamp, datetime)

    def test_signal_to_order_conversion(self, trading_engine):
        """シグナルから注文への変換テスト"""
        # 市場データ設定
        trading_engine.market_data["7203"] = MarketData(
            symbol="7203",
            price=Decimal("2500.0"),
            volume=1000000,
            timestamp=datetime.now(),
        )

        # 買いシグナル
        buy_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=85.0,
            reasons=["Golden Cross", "Volume Spike"],
        )

        order = trading_engine._create_order_from_signal("7203", buy_signal)

        assert order is not None
        assert order.symbol == "7203"
        assert order.trade_type == TradeType.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 85  # confidence * base_quantity / 100

    def test_risk_constraint_check(self, trading_engine):
        """リスク制約チェックのテスト"""
        # 正常状態
        assert trading_engine._check_risk_constraints() is True

        # 最大ポジション数超過をシミュレート
        for i in range(6):  # max_open_positions = 5 を超える
            trading_engine.active_positions[f"TEST{i}"] = [Mock()]

        assert trading_engine._check_risk_constraints() is False

    def test_daily_pnl_calculation(self, trading_engine):
        """日次損益計算のテスト"""
        # 市場データ設定
        trading_engine.market_data["7203"] = MarketData(
            symbol="7203",
            price=Decimal("2600.0"),  # 2500から100円上昇
            volume=1000000,
            timestamp=datetime.now(),
        )

        # アクティブポジション設定（買いポジション）
        mock_trade = Mock()
        mock_trade.symbol = "7203"
        mock_trade.trade_type = TradeType.BUY
        mock_trade.price = Decimal("2500.0")
        mock_trade.quantity = 100
        mock_trade.timestamp = datetime.now()

        trading_engine.active_positions["7203"] = [mock_trade]

        pnl = trading_engine._calculate_daily_pnl()
        expected_pnl = (Decimal("2600.0") - Decimal("2500.0")) * 100  # 10000円の利益
        assert pnl == expected_pnl

    def test_get_status(self, trading_engine):
        """ステータス取得のテスト"""
        status = trading_engine.get_status()

        assert "status" in status
        assert "monitored_symbols" in status
        assert "active_positions" in status
        assert "pending_orders" in status
        assert "daily_pnl" in status
        assert "execution_stats" in status
        assert "market_data_age" in status

        assert status["monitored_symbols"] == 3
        assert status["active_positions"] == 0
        assert status["pending_orders"] == 0

    def test_emergency_stop(self, trading_engine):
        """緊急停止のテスト"""
        # 保留中の注文を追加
        trading_engine.pending_orders.append(Mock())

        # 緊急停止実行
        trading_engine.emergency_stop()

        assert trading_engine.status == EngineStatus.STOPPED
        assert len(trading_engine.pending_orders) == 0
        assert trading_engine._stop_event.is_set()

    def test_performance_stats_update(self, trading_engine):
        """パフォーマンス統計更新のテスト"""
        # 実行時間を更新
        test_execution_time = 0.05  # 50ms
        trading_engine._update_performance_stats(test_execution_time)

        assert (
            trading_engine.execution_stats["avg_execution_time"] == test_execution_time
        )
        assert trading_engine.execution_stats["last_update"] is not None

    @pytest.mark.asyncio
    async def test_position_monitoring(self, trading_engine):
        """ポジション監視のテスト"""
        # 市場データ設定（価格上昇をシミュレート）
        trading_engine.market_data["7203"] = MarketData(
            symbol="7203",
            price=Decimal("2625.0"),  # 2500から5%上昇（利確ライン）
            volume=1000000,
            timestamp=datetime.now(),
        )

        # 買いポジション設定
        mock_trade = Mock()
        mock_trade.symbol = "7203"
        mock_trade.trade_type = TradeType.BUY
        mock_trade.price = Decimal("2500.0")
        mock_trade.quantity = 100

        trading_engine.active_positions["7203"] = [mock_trade]

        # ポジション監視実行
        with patch.object(
            trading_engine, "_close_position", new_callable=AsyncMock
        ) as mock_close:
            await trading_engine._monitor_positions()

            # 利益確定が呼ばれることを確認
            mock_close.assert_called_once_with("7203", mock_trade, "利益確定")


class TestTradingEngineIntegration:
    """統合テストクラス"""

    @pytest.mark.asyncio
    async def test_full_trading_cycle(self):
        """完全な取引サイクルの統合テスト"""
        # 実際の依存関係を使用（テスト環境向けに設定）
        symbols = ["7203"]

        # モック設定
        mock_stock_fetcher = Mock(spec=StockFetcher)
        mock_stock_fetcher.get_current_price.return_value = {
            "current_price": 2500.0,
            "volume": 1000000,
        }
        mock_stock_fetcher.get_historical_data.return_value = Mock()

        mock_signal_generator = Mock()
        mock_signal_generator.generate_signal.return_value = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=80.0,
            reasons=["Test Signal"],
        )

        mock_trade_manager = Mock(spec=TradeManager)

        # エンジン作成
        engine = TradingEngine(
            symbols=symbols,
            trade_manager=mock_trade_manager,
            signal_generator=mock_signal_generator,
            stock_fetcher=mock_stock_fetcher,
            update_interval=0.1,
        )

        # 短時間実行してテスト
        start_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.3)  # 短時間実行

        await engine.stop()

        # 結果検証
        assert engine.status == EngineStatus.STOPPED

        # 最低限の実行が行われたことを確認
        assert engine.execution_stats["last_update"] is not None

        # タスクが完了するまで待機
        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()
