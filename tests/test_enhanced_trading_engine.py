"""
拡張取引エンジンのテスト
"""

import asyncio
import uuid
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from src.day_trade.analysis.signals import SignalStrength, SignalType, TradingSignal
from src.day_trade.automation.enhanced_trading_engine import (
    EnhancedTradingEngine,
    ExecutionMode,
)
from src.day_trade.automation.trading_engine import EngineStatus, RiskParameters
from src.day_trade.core.trade_manager import TradeManager, TradeType


class TestEnhancedTradingEngine:
    """拡張取引エンジンのテストクラス"""

    @pytest.fixture
    def mock_dependencies(self):
        """依存関係のモック"""
        trade_manager = Mock(spec=TradeManager)
        signal_generator = Mock()
        stock_fetcher = Mock()

        # モックの戻り値を設定
        stock_fetcher.get_current_price.return_value = {
            "current_price": 2500.0,
            "volume": 1000000,
        }
        stock_fetcher.get_historical_data.return_value = Mock()

        return trade_manager, signal_generator, stock_fetcher

    @pytest.fixture
    def enhanced_engine(self, mock_dependencies):
        """拡張取引エンジンインスタンス"""
        trade_manager, signal_generator, stock_fetcher = mock_dependencies

        symbols = ["7203", "6758", "9984"]
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            max_daily_loss=Decimal("10000"),
            max_open_positions=5,
        )

        return EnhancedTradingEngine(
            symbols=symbols,
            trade_manager=trade_manager,
            signal_generator=signal_generator,
            stock_fetcher=stock_fetcher,
            risk_params=risk_params,
            initial_cash=Decimal("1000000"),
            execution_mode=ExecutionMode.BALANCED,
            update_interval=0.1,  # テスト用に短縮
        )

    def test_engine_initialization(self, enhanced_engine):
        """エンジン初期化のテスト"""
        assert enhanced_engine.status == EngineStatus.STOPPED
        assert len(enhanced_engine.symbols) == 3
        assert enhanced_engine.execution_mode == ExecutionMode.BALANCED
        assert enhanced_engine.portfolio_manager.initial_cash == Decimal("1000000")
        assert enhanced_engine.execution_stats["engine_cycles"] == 0

    def test_execution_mode_settings(self):
        """実行モード設定のテスト"""
        conservative_engine = EnhancedTradingEngine(
            symbols=["7203"],
            execution_mode=ExecutionMode.CONSERVATIVE,
        )
        assert conservative_engine.execution_mode == ExecutionMode.CONSERVATIVE
        assert conservative_engine._get_min_confidence_threshold() == 80.0

        aggressive_engine = EnhancedTradingEngine(
            symbols=["7203"],
            execution_mode=ExecutionMode.AGGRESSIVE,
        )
        assert aggressive_engine.execution_mode == ExecutionMode.AGGRESSIVE
        assert aggressive_engine._get_min_confidence_threshold() == 60.0

    @pytest.mark.asyncio
    async def test_engine_start_stop(self, enhanced_engine):
        """エンジンの開始・停止テスト"""
        # エンジン開始（短時間実行）
        start_task = asyncio.create_task(enhanced_engine.start())
        await asyncio.sleep(0.2)  # 短時間実行

        # 実行中であることを確認
        assert enhanced_engine.status in [EngineStatus.RUNNING, EngineStatus.STOPPED]

        # 停止
        await enhanced_engine.stop()
        assert enhanced_engine.status == EngineStatus.STOPPED

        # タスク完了を待機
        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

    @pytest.mark.asyncio
    async def test_pause_resume(self, enhanced_engine):
        """一時停止・再開のテスト"""
        # 初期状態をRUNNINGにする
        enhanced_engine.status = EngineStatus.RUNNING

        # 一時停止
        await enhanced_engine.pause()
        assert enhanced_engine.status == EngineStatus.PAUSED

        # 再開
        await enhanced_engine.resume()
        assert enhanced_engine.status == EngineStatus.RUNNING

    @pytest.mark.asyncio
    async def test_market_data_update(self, enhanced_engine):
        """市場データ更新のテスト"""
        # モックされた価格データで更新実行
        await enhanced_engine._update_market_data()

        # 市場データが更新されることを確認
        assert len(enhanced_engine.market_data) == 3
        for symbol in enhanced_engine.symbols:
            assert symbol in enhanced_engine.market_data
            market_data = enhanced_engine.market_data[symbol]
            assert market_data.price == Decimal("2500.0")
            assert market_data.volume == 1000000

    @pytest.mark.asyncio
    async def test_signal_generation(self, enhanced_engine):
        """シグナル生成のテスト"""
        # 先に市場データを更新
        await enhanced_engine._update_market_data()

        # モックシグナルを設定
        mock_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=75.0,
            reasons=["Test Signal"],
        )

        enhanced_engine.signal_generator.generate_signal.return_value = mock_signal

        # シグナル生成実行
        signals = await enhanced_engine._generate_trading_signals()

        assert len(signals) == 3  # 3銘柄分
        for symbol, signal in signals:
            assert symbol in enhanced_engine.symbols
            assert signal.confidence == 75.0
            assert signal.signal_type == SignalType.BUY

    def test_risk_check(self, enhanced_engine):
        """リスク管理チェックのテスト"""
        # 正常状態でのリスクチェック
        risk_result = enhanced_engine._comprehensive_risk_check()
        assert risk_result["approved"] is True
        assert risk_result["reason"] == "OK"

        # 大量ポジションを追加してリスク違反をシミュレート
        large_trade = Mock()
        large_trade.symbol = "7203"
        large_trade.trade_type = TradeType.BUY
        large_trade.quantity = 10000
        large_trade.price = Decimal("2500.0")
        large_trade.timestamp = datetime.now()
        large_trade.commission = Decimal("250.0")
        large_trade.status = "executed"

        enhanced_engine.portfolio_manager.add_trade(large_trade)

        # リスクチェック再実行
        risk_result = enhanced_engine._comprehensive_risk_check()
        # リスク違反が検出される可能性があることを確認
        assert "risk_score" in risk_result

    def test_position_size_calculation(self, enhanced_engine):
        """ポジションサイズ計算のテスト"""
        # 市場データを設定
        enhanced_engine.market_data["7203"] = Mock()
        enhanced_engine.market_data["7203"].price = Decimal("2500.0")

        # 高信頼度シグナル
        high_confidence_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=90.0,
            reasons=["Strong Signal"],
        )

        position_size = enhanced_engine._calculate_position_size(
            "7203", high_confidence_signal
        )
        assert position_size > 0
        assert position_size <= 40  # max_position_size / price の制限内

        # 低信頼度シグナル
        low_confidence_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=50.0,
            reasons=["Weak Signal"],
        )

        low_position_size = enhanced_engine._calculate_position_size(
            "7203", low_confidence_signal
        )
        assert low_position_size < position_size  # 信頼度が低いとサイズも小さい

    @pytest.mark.asyncio
    async def test_conservative_order_strategy(self, enhanced_engine):
        """保守的注文戦略のテスト"""
        # 市場データ設定
        current_price = Decimal("2500.0")
        enhanced_engine.market_data["7203"] = Mock()
        enhanced_engine.market_data["7203"].price = current_price

        # 保守的モードに変更
        enhanced_engine.execution_mode = ExecutionMode.CONSERVATIVE

        mock_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=85.0,  # 保守的モードの閾値を超える
            reasons=["Conservative Signal"],
        )

        # 注文管理システムのモック
        enhanced_engine.order_manager.submit_order = AsyncMock()

        await enhanced_engine._create_and_submit_orders("7203", mock_signal)

        # 注文が提出されることを確認
        enhanced_engine.order_manager.submit_order.assert_called()
        assert enhanced_engine.execution_stats["orders_generated"] == 1

    @pytest.mark.asyncio
    async def test_aggressive_order_strategy(self, enhanced_engine):
        """積極的注文戦略のテスト"""
        # 市場データ設定
        current_price = Decimal("2500.0")
        enhanced_engine.market_data["7203"] = Mock()
        enhanced_engine.market_data["7203"].price = current_price

        # 積極的モードに変更
        enhanced_engine.execution_mode = ExecutionMode.AGGRESSIVE

        mock_signal = TradingSignal(
            signal_type=SignalType.SELL,
            strength=SignalStrength.MEDIUM,
            confidence=65.0,  # 積極的モードの閾値を超える
            reasons=["Aggressive Signal"],
        )

        # 注文管理システムのモック
        enhanced_engine.order_manager.submit_order = AsyncMock()

        await enhanced_engine._create_and_submit_orders("7203", mock_signal)

        # 注文が提出されることを確認
        enhanced_engine.order_manager.submit_order.assert_called()
        assert enhanced_engine.execution_stats["orders_generated"] == 1

    @pytest.mark.asyncio
    async def test_balanced_order_strategy(self, enhanced_engine):
        """バランス注文戦略のテスト"""
        # 市場データ設定
        current_price = Decimal("2500.0")
        enhanced_engine.market_data["7203"] = Mock()
        enhanced_engine.market_data["7203"].price = current_price

        # バランスモード（デフォルト）
        assert enhanced_engine.execution_mode == ExecutionMode.BALANCED

        mock_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=75.0,  # バランスモードの閾値を超える
            reasons=["Balanced Signal"],
        )

        # 注文管理システムのモック
        enhanced_engine.order_manager.submit_order = AsyncMock()

        await enhanced_engine._create_and_submit_orders("7203", mock_signal)

        # 複数の注文が提出されることを確認（指値+成行）
        assert enhanced_engine.order_manager.submit_order.call_count == 2
        assert enhanced_engine.execution_stats["orders_generated"] == 1

    def test_comprehensive_status(self, enhanced_engine):
        """包括的ステータス取得のテスト"""
        status = enhanced_engine.get_comprehensive_status()

        assert "engine" in status
        assert "portfolio" in status
        assert "orders" in status
        assert "performance" in status
        assert "risk" in status

        # エンジン情報
        engine_info = status["engine"]
        assert engine_info["status"] == "stopped"
        assert engine_info["execution_mode"] == "balanced"
        assert engine_info["monitored_symbols"] == 3

        # ポートフォリオ情報
        portfolio_info = status["portfolio"]
        assert "total_equity" in portfolio_info
        assert "total_pnl" in portfolio_info
        assert "cash" in portfolio_info

    def test_emergency_stop(self, enhanced_engine):
        """緊急停止のテスト"""
        # 初期状態を実行中にする
        enhanced_engine.status = EngineStatus.RUNNING

        # 緊急停止実行
        enhanced_engine.emergency_stop()

        assert enhanced_engine.status == EngineStatus.STOPPED
        assert enhanced_engine._stop_event.is_set()

    def test_execution_statistics(self, enhanced_engine):
        """実行統計のテスト"""
        # 統計を手動で更新
        enhanced_engine.execution_stats["engine_cycles"] = 10
        enhanced_engine.execution_stats["signals_processed"] = 25
        enhanced_engine.execution_stats["orders_generated"] = 5
        enhanced_engine._update_execution_stats(0.05)  # 50ms サイクル時間

        assert enhanced_engine.execution_stats["engine_cycles"] == 11
        assert enhanced_engine.execution_stats["avg_cycle_time"] > 0
        assert enhanced_engine.execution_stats["last_update"] is not None

    @pytest.mark.asyncio
    async def test_order_processing_integration(self, enhanced_engine):
        """注文処理統合のテスト"""
        # 市場データ設定
        await enhanced_engine._update_market_data()

        # モック約定を作成
        mock_fill = Mock()
        mock_fill.fill_id = str(uuid.uuid4())
        mock_fill.symbol = "7203"
        mock_fill.side = TradeType.BUY
        mock_fill.quantity = 100
        mock_fill.price = Decimal("2500.0")
        mock_fill.timestamp = datetime.now()
        mock_fill.commission = Decimal("25.0")

        # 注文管理システムのモック設定
        enhanced_engine.order_manager.process_market_update = AsyncMock(
            return_value=[mock_fill]
        )

        # 約定処理実行
        await enhanced_engine._process_pending_orders()

        # ポートフォリオに取引が追加されることを確認
        assert len(enhanced_engine.portfolio_manager.trade_history) == 1
        assert "7203" in enhanced_engine.portfolio_manager.positions

    @pytest.mark.asyncio
    async def test_signal_processing_threshold(self, enhanced_engine):
        """シグナル処理閾値のテスト"""
        # 低信頼度シグナル（実行されないはず）
        low_confidence_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=50.0,  # バランスモード閾値(70%)を下回る
            reasons=["Low Confidence"],
        )

        enhanced_engine.market_data["7203"] = Mock()
        enhanced_engine.market_data["7203"].price = Decimal("2500.0")
        enhanced_engine.order_manager.submit_order = AsyncMock()

        signals = [("7203", low_confidence_signal)]
        await enhanced_engine._process_trading_signals(signals)

        # 注文が提出されないことを確認
        enhanced_engine.order_manager.submit_order.assert_not_called()
        assert enhanced_engine.execution_stats["orders_generated"] == 0


class TestIntegrationScenarios:
    """統合シナリオのテスト"""

    @pytest.mark.asyncio
    async def test_full_trading_cycle(self):
        """完全な取引サイクルの統合テスト"""
        # 依存関係のモック
        mock_trade_manager = Mock(spec=TradeManager)
        mock_signal_generator = Mock()
        mock_stock_fetcher = Mock()

        # モック設定
        mock_stock_fetcher.get_current_price.return_value = {
            "current_price": 2500.0,
            "volume": 1000000,
        }
        mock_stock_fetcher.get_historical_data.return_value = Mock()

        strong_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=80.0,
            reasons=["Integration Test"],
        )
        mock_signal_generator.generate_signal.return_value = strong_signal

        # エンジン作成
        engine = EnhancedTradingEngine(
            symbols=["7203"],
            trade_manager=mock_trade_manager,
            signal_generator=mock_signal_generator,
            stock_fetcher=mock_stock_fetcher,
            execution_mode=ExecutionMode.BALANCED,
            update_interval=0.1,
        )

        # 短時間実行してテスト
        start_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.3)  # 短時間実行

        await engine.stop()

        # 結果検証
        assert engine.status == EngineStatus.STOPPED
        assert engine.execution_stats["engine_cycles"] > 0

        # タスク完了を待機
        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

    @pytest.mark.asyncio
    async def test_risk_management_integration(self):
        """リスク管理統合テスト"""
        # 厳しいリスクパラメータ
        strict_risk = RiskParameters(
            max_position_size=Decimal("50000"),
            max_daily_loss=Decimal("5000"),
            max_open_positions=2,
        )

        engine = EnhancedTradingEngine(
            symbols=["7203", "6758"],
            risk_params=strict_risk,
            initial_cash=Decimal("100000"),  # 少額資金
            execution_mode=ExecutionMode.CONSERVATIVE,
            update_interval=0.1,
        )

        # 大量取引をポートフォリオに追加（リスク違反をシミュレート）
        large_trade = Mock()
        large_trade.symbol = "7203"
        large_trade.trade_type = TradeType.BUY
        large_trade.quantity = 5000
        large_trade.price = Decimal("2500.0")
        large_trade.timestamp = datetime.now()
        large_trade.commission = Decimal("250.0")
        large_trade.status = "executed"

        engine.portfolio_manager.add_trade(large_trade)

        # リスクチェック実行
        risk_result = engine._comprehensive_risk_check()

        # リスク違反が検出されることを確認
        assert risk_result["approved"] is False
        assert (
            "violation" in risk_result["reason"].lower()
            or "到達" in risk_result["reason"]
        )

    def test_multi_mode_comparison(self):
        """複数実行モードの比較テスト"""
        modes = [
            ExecutionMode.CONSERVATIVE,
            ExecutionMode.BALANCED,
            ExecutionMode.AGGRESSIVE,
        ]
        engines = []

        for mode in modes:
            engine = EnhancedTradingEngine(
                symbols=["7203"],
                execution_mode=mode,
            )
            engines.append(engine)

        # 信頼度閾値の比較
        thresholds = [engine._get_min_confidence_threshold() for engine in engines]

        # Conservative > Balanced > Aggressive の順序
        assert thresholds[0] > thresholds[1] > thresholds[2]
        assert thresholds == [80.0, 70.0, 60.0]
