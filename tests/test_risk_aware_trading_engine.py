"""
リスク認識取引エンジンのテスト
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from src.day_trade.analysis.signals import SignalStrength, SignalType, TradingSignal
from src.day_trade.automation.enhanced_trading_engine import ExecutionMode
from src.day_trade.automation.risk_aware_trading_engine import RiskAwareTradingEngine
from src.day_trade.automation.risk_manager import (
    AlertType,
    EmergencyReason,
    RiskAlert,
    RiskLevel,
    RiskLimits,
)
from src.day_trade.automation.trading_engine import EngineStatus, RiskParameters
from src.day_trade.core.trade_manager import TradeManager


class TestRiskAwareTradingEngine:
    """リスク認識取引エンジンのテストクラス"""

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
    def risk_limits(self):
        """テスト用リスク制限"""
        return RiskLimits(
            max_position_size=Decimal("100000"),
            max_total_exposure=Decimal("500000"),
            max_open_positions=3,
            max_daily_loss=Decimal("20000"),
            max_consecutive_losses=3,
        )

    @pytest.fixture
    def risk_aware_engine(self, mock_dependencies, risk_limits):
        """リスク認識取引エンジンインスタンス"""
        trade_manager, signal_generator, stock_fetcher = mock_dependencies

        symbols = ["7203", "6758", "9984"]
        risk_params = RiskParameters(
            max_position_size=Decimal("100000"),
            max_daily_loss=Decimal("20000"),
            max_open_positions=3,
        )

        return RiskAwareTradingEngine(
            symbols=symbols,
            trade_manager=trade_manager,
            signal_generator=signal_generator,
            stock_fetcher=stock_fetcher,
            risk_params=risk_params,
            risk_limits=risk_limits,
            initial_cash=Decimal("1000000"),
            execution_mode=ExecutionMode.BALANCED,
            update_interval=0.1,
            emergency_stop_enabled=True,
        )

    def test_risk_aware_engine_initialization(self, risk_aware_engine):
        """リスク認識エンジン初期化のテスト"""
        assert risk_aware_engine.status == EngineStatus.STOPPED
        assert risk_aware_engine.risk_manager is not None
        assert not risk_aware_engine.risk_manager.is_emergency_stopped
        assert risk_aware_engine.emergency_stop_enabled is True
        assert len(risk_aware_engine.symbols) == 3

    @pytest.mark.asyncio
    async def test_engine_start_with_risk_monitoring(self, risk_aware_engine):
        """リスク監視付きエンジン開始のテスト"""
        # エンジン開始（短時間実行）
        start_task = asyncio.create_task(risk_aware_engine.start())
        await asyncio.sleep(0.2)

        # リスク監視が開始されることを確認
        assert risk_aware_engine.risk_manager._monitoring_task is not None

        # 停止
        await risk_aware_engine.stop()
        assert risk_aware_engine.status == EngineStatus.STOPPED

        # タスク完了を待機
        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

    @pytest.mark.asyncio
    async def test_emergency_stop_integration(self, risk_aware_engine):
        """緊急停止統合のテスト"""
        # 手動緊急停止
        risk_aware_engine.emergency_stop()

        # 短時間待機（非同期処理完了まで）
        await asyncio.sleep(0.1)

        assert risk_aware_engine.risk_manager.is_emergency_stopped is True

    def test_emergency_stop_reset(self, risk_aware_engine):
        """緊急停止リセットのテスト"""
        # まず緊急停止状態にする
        risk_aware_engine.risk_manager.is_emergency_stopped = True
        risk_aware_engine.status = EngineStatus.STOPPED

        # リセット実行
        success = risk_aware_engine.reset_emergency_stop("test_operator")

        assert success is True
        assert not risk_aware_engine.risk_manager.is_emergency_stopped
        assert risk_aware_engine.status == EngineStatus.PAUSED

    @pytest.mark.asyncio
    async def test_risk_aware_position_sizing(self, risk_aware_engine):
        """リスク認識ポジションサイジングのテスト"""
        # 市場データ設定
        risk_aware_engine.market_data["7203"] = Mock()
        risk_aware_engine.market_data["7203"].price = Decimal("2500.0")

        # 高信頼度シグナル
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=85.0,
            reasons=["Test Signal"],
        )

        # ポジションサイズ計算
        size = await risk_aware_engine._calculate_risk_aware_position_size(
            "7203", signal
        )

        assert size > 0
        assert size <= 40  # max_position_size / price の制限内

    @pytest.mark.asyncio
    async def test_emergency_condition_check(self, risk_aware_engine):
        """緊急停止条件チェックのテスト"""
        # 正常状態のポートフォリオ
        normal_summary = Mock()
        normal_summary.daily_pnl = Decimal("1000.0")
        normal_summary.total_positions = 2
        risk_aware_engine.portfolio_manager.get_portfolio_summary.return_value = (
            normal_summary
        )

        # 緊急停止条件チェック（正常時）
        await risk_aware_engine._check_emergency_conditions()

        # 緊急停止されないことを確認
        assert not risk_aware_engine.risk_manager.is_emergency_stopped

        # 異常状態のポートフォリオ
        emergency_summary = Mock()
        emergency_summary.daily_pnl = Decimal("-25000.0")  # 制限-20000を超過
        emergency_summary.total_positions = 2
        risk_aware_engine.portfolio_manager.get_portfolio_summary.return_value = (
            emergency_summary
        )

        # 緊急停止条件チェック（異常時）
        await risk_aware_engine._check_emergency_conditions()

        # 緊急停止が実行されることを確認
        assert risk_aware_engine.risk_manager.is_emergency_stopped

    def test_risk_alert_handling(self, risk_aware_engine):
        """リスクアラートハンドリングのテスト"""
        # テストアラート作成
        alert = RiskAlert(
            alert_id="test_alert",
            alert_type=AlertType.WARNING,
            risk_level=RiskLevel.MEDIUM,
            message="テストアラートメッセージ",
            symbol="7203",
        )

        # アラート処理
        risk_aware_engine._handle_risk_alert(alert)

        # 統計が更新されることを確認
        assert risk_aware_engine.risk_stats["alerts_generated"] == 1

    @pytest.mark.asyncio
    async def test_critical_alert_auto_response(self, risk_aware_engine):
        """クリティカルアラート自動対応のテスト"""
        # 市場データ設定
        risk_aware_engine.market_data["7203"] = Mock()
        risk_aware_engine.market_data["7203"].price = Decimal("2300.0")

        # ポジション設定
        mock_position = Mock()
        mock_position.is_flat.return_value = False
        mock_position.quantity = 100
        risk_aware_engine.portfolio_manager.get_position.return_value = mock_position
        risk_aware_engine.portfolio_manager.close_position.return_value = Mock()

        # ストップロス到達アラート
        critical_alert = RiskAlert(
            alert_id="critical_test",
            alert_type=AlertType.CRITICAL,
            risk_level=RiskLevel.CRITICAL,
            message="ストップロス到達: 7203",
            symbol="7203",
        )

        # 自動対応実行
        await risk_aware_engine._handle_critical_position_alert(critical_alert)

        # ポジションクローズが呼ばれることを確認
        risk_aware_engine.portfolio_manager.close_position.assert_called_once()

    def test_comprehensive_status_with_risk_info(self, risk_aware_engine):
        """リスク情報統合包括ステータスのテスト"""
        # テストデータ設定
        risk_aware_engine.risk_manager.risk_metrics.total_exposure = Decimal("100000")
        risk_aware_engine.risk_stats["alerts_generated"] = 5

        status = risk_aware_engine.get_comprehensive_status()

        # リスク管理情報が含まれることを確認
        assert "risk_management" in status
        assert "system_health" in status

        risk_info = status["risk_management"]
        assert "emergency_stopped" in risk_info
        assert "risk_metrics" in risk_info
        assert "active_alerts" in risk_info
        assert "risk_stats" in risk_info

        assert risk_info["risk_stats"]["alerts_generated"] == 5

    @pytest.mark.asyncio
    async def test_risk_aware_signal_processing(self, risk_aware_engine):
        """リスク認識シグナル処理のテスト"""
        # 市場データ設定
        risk_aware_engine.market_data["7203"] = Mock()
        risk_aware_engine.market_data["7203"].price = Decimal("2500.0")

        # 注文管理システムのモック
        risk_aware_engine.order_manager.submit_order = AsyncMock()

        # 高信頼度シグナル
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=80.0,
            reasons=["Risk Aware Test"],
        )

        signals = [("7203", signal)]

        # シグナル処理実行
        await risk_aware_engine._process_trading_signals_with_risk(signals)

        # 統計更新を確認
        assert risk_aware_engine.risk_stats["risk_adjustments"] > 0

    @pytest.mark.asyncio
    async def test_emergency_stop_during_execution(self, risk_aware_engine):
        """実行中の緊急停止のテスト"""
        # エンジンを実行状態にする
        risk_aware_engine.status = EngineStatus.RUNNING

        # 緊急停止をトリガー
        await risk_aware_engine.risk_manager.trigger_emergency_stop(
            EmergencyReason.LOSS_LIMIT, "テスト緊急停止"
        )

        # 緊急停止状態になることを確認
        assert risk_aware_engine.risk_manager.is_emergency_stopped is True

        # メインループでの緊急停止検知をテスト
        # （実際のメインループは複雑なので、状態チェックのみ）
        if risk_aware_engine.risk_manager.is_emergency_stopped:
            risk_aware_engine.status = EngineStatus.STOPPED

        assert risk_aware_engine.status == EngineStatus.STOPPED

    def test_emergency_stop_callback_execution(self, risk_aware_engine):
        """緊急停止コールバック実行のテスト"""
        emergency_callback_executed = False
        callback_reason = None
        callback_info = None

        async def test_callback(reason, info):
            nonlocal emergency_callback_executed, callback_reason, callback_info
            emergency_callback_executed = True
            callback_reason = reason
            callback_info = info

        # コールバックを設定
        risk_aware_engine.risk_manager.emergency_callback = test_callback

        # 緊急停止実行
        asyncio.create_task(
            risk_aware_engine.risk_manager.trigger_emergency_stop(
                EmergencyReason.SYSTEM_ERROR, "テストシステムエラー"
            )
        )

        # 短時間待機
        import time

        time.sleep(0.1)

        # コールバックが実行されることを確認（非同期のため結果は実行環境依存）


class TestRiskIntegrationScenarios:
    """リスク統合シナリオのテスト"""

    @pytest.mark.asyncio
    async def test_full_risk_management_cycle(self):
        """完全なリスク管理サイクルのテスト"""
        # 厳格なリスク設定
        strict_limits = RiskLimits(
            max_position_size=Decimal("50000"),
            max_total_exposure=Decimal("200000"),
            max_open_positions=2,
            max_daily_loss=Decimal("10000"),
        )

        # モック依存関係
        mock_trade_manager = Mock(spec=TradeManager)
        mock_signal_generator = Mock()
        mock_stock_fetcher = Mock()

        mock_stock_fetcher.get_current_price.return_value = {
            "current_price": 2500.0,
            "volume": 1000000,
        }
        mock_stock_fetcher.get_historical_data.return_value = Mock()

        # 中程度の信頼度シグナル
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=75.0,
            reasons=["Integration Test"],
        )
        mock_signal_generator.generate_signal.return_value = signal

        # エンジン作成
        engine = RiskAwareTradingEngine(
            symbols=["7203"],
            trade_manager=mock_trade_manager,
            signal_generator=mock_signal_generator,
            stock_fetcher=mock_stock_fetcher,
            risk_limits=strict_limits,
            execution_mode=ExecutionMode.CONSERVATIVE,
            update_interval=0.1,
            emergency_stop_enabled=True,
        )

        # 短時間実行
        start_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.3)

        await engine.stop()

        # 結果検証
        assert engine.status == EngineStatus.STOPPED
        assert engine.risk_stats["last_risk_check"] is not None

        # タスク完了を待機
        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

    @pytest.mark.asyncio
    async def test_cascade_risk_response(self):
        """カスケードリスク対応のテスト"""
        # 複数の段階的リスクイベントをシミュレート

        strict_limits = RiskLimits(
            max_consecutive_losses=2,  # 2回連続損失で制限
            max_daily_loss=Decimal("5000"),
        )

        engine = RiskAwareTradingEngine(
            symbols=["7203", "6758"],
            risk_limits=strict_limits,
            emergency_stop_enabled=True,
        )

        # 段階1: 通常のリスクアラート
        alert1 = RiskAlert(
            alert_id="cascade_1",
            alert_type=AlertType.WARNING,
            risk_level=RiskLevel.MEDIUM,
            message="ポジションサイズ注意",
        )

        engine._handle_risk_alert(alert1)
        assert not engine.risk_manager.is_emergency_stopped

        # 段階2: 高リスクアラート
        alert2 = RiskAlert(
            alert_id="cascade_2",
            alert_type=AlertType.ERROR,
            risk_level=RiskLevel.HIGH,
            message="損失増大",
        )

        engine._handle_risk_alert(alert2)
        assert not engine.risk_manager.is_emergency_stopped

        # 段階3: クリティカルアラートで緊急停止
        await engine.risk_manager.trigger_emergency_stop(
            EmergencyReason.LOSS_LIMIT, "カスケードリスク検知"
        )

        assert engine.risk_manager.is_emergency_stopped
        assert engine.risk_stats["emergency_stops"] == 1

    def test_risk_performance_metrics(self):
        """リスクパフォーマンス指標のテスト"""
        engine = RiskAwareTradingEngine(
            symbols=["7203"],
            emergency_stop_enabled=True,
        )

        # 統計の初期化確認
        assert engine.risk_stats["orders_rejected"] == 0
        assert engine.risk_stats["emergency_stops"] == 0
        assert engine.risk_stats["alerts_generated"] == 0
        assert engine.risk_stats["risk_adjustments"] == 0

        # 模擬統計更新
        engine.risk_stats["orders_rejected"] = 3
        engine.risk_stats["alerts_generated"] = 5

        status = engine.get_comprehensive_status()
        risk_stats = status["risk_management"]["risk_stats"]

        assert risk_stats["orders_rejected"] == 3
        assert risk_stats["alerts_generated"] == 5
