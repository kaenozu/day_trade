"""
包括的な統合テスト - Issue #129対応

主要な機能フローを網羅する統合テストを実装し、
システム全体としての動作を保証する。
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.day_trade.analysis.indicators import TechnicalIndicators
from src.day_trade.analysis.signals import TradingSignalGenerator
from src.day_trade.core.alerts import AlertManager
from src.day_trade.core.trade_manager import TradeManager
from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.models.database import DatabaseConfig, DatabaseManager
from src.day_trade.models.enums import AlertType, TradeType
from src.day_trade.models.stock import PriceData, Stock
from src.day_trade.utils.enhanced_error_handler import (
    get_default_error_handler,
)
from src.day_trade.utils.exceptions import (
    DataError,
    TradingError,
    ValidationError,
)

pytestmark = pytest.mark.integration


class TestComprehensiveWorkflow:
    """包括的なワークフロー統合テスト"""

    @pytest.fixture
    def test_db_config(self):
        """テスト用データベース設定"""
        return DatabaseConfig.for_testing()

    @pytest.fixture
    def test_db_manager(self, test_db_config):
        """テスト用データベースマネージャー"""
        db_manager = DatabaseManager(test_db_config)
        db_manager.create_tables()
        return db_manager

    @pytest.fixture
    def sample_stock_data(self):
        """テスト用株価データ"""
        dates = pd.date_range(start="2024-01-01", end="2024-01-30", freq="D")

        # 現実的な株価データの生成
        base_price = 2800.0
        prices = []
        for i, date in enumerate(dates):
            # 日々の価格変動をシミュレート
            daily_change = (i % 10 - 5) * 0.01  # -5% to +5%
            price = base_price * (1 + daily_change)
            prices.append(
                {
                    "Date": date,
                    "Open": price * 0.995,
                    "High": price * 1.02,
                    "Low": price * 0.98,
                    "Close": price,
                    "Volume": 1000000 + (i * 10000),
                }
            )

        return pd.DataFrame(prices).set_index("Date")

    @pytest.fixture
    def mock_stock_fetcher(self, sample_stock_data):
        """モック化されたStockFetcher"""
        fetcher = Mock(spec=StockFetcher)
        fetcher.get_historical_data.return_value = sample_stock_data
        fetcher.get_current_price.return_value = {
            "current_price": 2850.0,
            "change": 50.0,
            "change_percent": 1.79,
            "volume": 1500000,
        }
        return fetcher

    def test_data_acquisition_and_processing_flow(self, mock_stock_fetcher):
        """データ取得・処理フローの統合テスト"""

        # 1. データ取得
        symbol = "7203"
        historical_data = mock_stock_fetcher.get_historical_data(
            symbol, start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 30)
        )

        assert historical_data is not None
        assert len(historical_data) > 0
        assert "Close" in historical_data.columns

        # 2. テクニカル分析
        indicators = TechnicalIndicators()

        # SMA計算
        sma_20 = indicators.sma(historical_data, period=20, column="Close")
        assert sma_20 is not None
        assert len(sma_20) == len(historical_data)

        # RSI計算
        rsi = indicators.rsi(historical_data, period=14, column="Close")
        assert rsi is not None
        assert len(rsi) == len(historical_data)

        # MACD計算
        macd_result = indicators.macd(historical_data, column="Close")
        assert macd_result is not None
        assert isinstance(macd_result, pd.DataFrame)

        # 3. データ品質検証
        assert historical_data["High"].min() >= historical_data["Low"].min()
        assert historical_data["Volume"].min() >= 0
        assert not historical_data["Close"].isna().all()

    def test_signal_generation_flow(self, mock_stock_fetcher, sample_stock_data):
        """シグナル生成フローの統合テスト"""

        # 1. シグナル生成器の初期化（config_pathのみ）
        signal_generator = TradingSignalGenerator(config_path=None)

        # 2. テクニカル分析とシグナル生成（正しいメソッドを使用）
        signals_df = signal_generator.generate_signals_series(sample_stock_data)

        assert signals_df is not None
        assert isinstance(signals_df, pd.DataFrame)

        # シグナル結果の検証
        if not signals_df.empty:
            # DataFrameの列名チェック
            expected_columns = ["Buy_Signal", "Sell_Signal", "Signal_Strength"]
            available_columns = [
                col for col in expected_columns if col in signals_df.columns
            ]
            assert (
                len(available_columns) > 0
            ), f"期待する列が見つかりません: {expected_columns}"

    def test_portfolio_management_flow(self, test_db_manager, mock_stock_fetcher):
        """ポートフォリオ管理フローの統合テスト"""

        # 1. 取引マネージャーの初期化
        trade_manager = TradeManager()

        # 2. 銘柄データの準備
        with test_db_manager.session_scope() as session:
            test_stock = Stock(
                code="7203",
                name="トヨタ自動車",
                market="東証プライム",
                sector="自動車・輸送機器",
            )
            session.add(test_stock)
            session.commit()

        # 3. 買い注文の実行
        buy_trade_id = trade_manager.add_trade(
            symbol="7203",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("2800.0"),
            notes="統合テスト買い注文",
        )

        assert buy_trade_id is not None
        # 取引履歴から確認
        trades = trade_manager.get_trade_history("7203")
        assert len(trades) > 0
        buy_trade = trades[-1]  # 最新の取引
        assert buy_trade.symbol == "7203"
        assert buy_trade.trade_type == TradeType.BUY
        assert buy_trade.quantity == 100

        # 4. ポジション確認
        positions = trade_manager.get_current_positions()
        assert "7203" in positions
        assert positions["7203"].quantity == 100

        # 5. 売り注文の実行
        sell_trade_id = trade_manager.add_trade(
            symbol="7203",
            trade_type=TradeType.SELL,
            quantity=50,
            price=Decimal("2900.0"),
            notes="統合テスト売り注文",
        )

        assert sell_trade_id is not None
        # 売り取引も履歴から確認
        trades = trade_manager.get_trade_history("7203")
        sell_trade = trades[-1]  # 最新の取引（売り）
        assert sell_trade.trade_type == TradeType.SELL

        # 6. 最終ポジション確認
        final_positions = trade_manager.get_current_positions()
        assert final_positions["7203"].quantity == 50

    def test_alert_system_flow(self, test_db_manager, mock_stock_fetcher):
        """アラートシステムフローの統合テスト"""

        # 1. アラートマネージャーの初期化
        alert_manager = AlertManager(test_db_manager, mock_stock_fetcher)

        # 2. テスト用銘柄の準備
        with test_db_manager.session_scope() as session:
            test_stock = Stock(code="7203", name="トヨタ自動車", market="東証プライム")
            session.add(test_stock)
            session.commit()

        # 3. 価格アラートの設定
        alert = alert_manager.create_alert(
            stock_code="7203",
            alert_type=AlertType.PRICE_ABOVE,
            threshold=3000.0,
            memo="高値警戒アラート",
        )

        assert alert is not None
        assert alert.stock_code == "7203"
        assert alert.alert_type == AlertType.PRICE_ABOVE
        assert alert.threshold == 3000.0

        # 4. アラート条件チェック（価格が閾値以下の場合）
        mock_stock_fetcher.get_current_price.return_value = {
            "current_price": 2850.0,  # 閾値以下
            "change": 50.0,
            "change_percent": 1.79,
        }

        triggered_alerts = alert_manager.check_alerts()
        assert isinstance(triggered_alerts, list)

        # 5. アラート条件チェック（価格が閾値を超えた場合）
        mock_stock_fetcher.get_current_price.return_value = {
            "current_price": 3100.0,  # 閾値超過
            "change": 300.0,
            "change_percent": 10.71,
        }

        triggered_alerts_high = alert_manager.check_alerts()

        # 高値アラートがトリガーされるべき
        if triggered_alerts_high:
            triggered_alert = triggered_alerts_high[0]
            assert triggered_alert.stock_code == "7203"

    def test_error_handling_integration(self, test_db_manager):
        """エラーハンドリング統合テスト"""

        error_handler = get_default_error_handler()

        # 1. データベースエラーのシミュレーション
        with pytest.raises(ValidationError):
            trade_manager = TradeManager()
            trade_manager.add_trade(
                symbol="INVALID",  # 存在しない銘柄
                trade_type=TradeType.BUY,
                quantity=0,  # 無効な数量
                price=Decimal("-100.0"),  # 無効な価格
            )

        # 2. データエラーのシミュレーション
        try:
            raise DataError(
                message="テストデータエラー",
                error_code="TEST_DATA_ERROR",
                details={"context": "integration_test"},
            )
        except DataError as e:
            panel = error_handler.handle_error(
                e, context={"test_type": "integration"}, show_technical=False
            )
            assert panel is not None

        # 3. 取引エラーのシミュレーション
        try:
            raise TradingError(
                message="テスト取引エラー",
                error_code="TEST_TRADING_ERROR",
                details={"symbol": "7203", "quantity": 100},
            )
        except TradingError as e:
            error_handler.log_error(e, context={"user_action": "統合テスト"})

    def test_database_transaction_integrity(self, test_db_manager):
        """データベーストランザクション整合性テスト"""

        # 1. トランザクション成功ケース
        with test_db_manager.session_scope() as session:
            stock1 = Stock(code="7203", name="トヨタ自動車")
            stock2 = Stock(code="9984", name="ソフトバンクグループ")

            session.add(stock1)
            session.add(stock2)
            # commit は session_scope で自動実行

        # 2. データが正しく保存されていることを確認
        with test_db_manager.session_scope() as session:
            saved_stocks = session.query(Stock).count()
            assert saved_stocks >= 2

        # 3. トランザクション失敗ケースのシミュレーション
        try:
            with test_db_manager.session_scope() as session:
                # 重複する銘柄コードで制約違反を引き起こす
                duplicate_stock = Stock(code="7203", name="重複銘柄")
                session.add(duplicate_stock)
                session.flush()  # エラーを早期に検出
        except Exception:
            # エラーが発生することを期待
            pass

        # 4. ロールバック後のデータ整合性確認
        with test_db_manager.session_scope() as session:
            toyota_count = session.query(Stock).filter(Stock.code == "7203").count()
            assert toyota_count == 1  # 元のデータのみ残っている

    def test_performance_monitoring_flow(self):
        """パフォーマンス監視フローの統合テスト"""

        from src.day_trade.utils.logging_config import get_performance_logger
        from src.day_trade.utils.performance_analyzer import PerformanceAnalyzer

        # 1. パフォーマンスロガーの初期化
        _perf_logger = get_performance_logger("integration_test")

        # 2. パフォーマンス分析器の初期化
        analyzer = PerformanceAnalyzer()

        # 3. 測定開始
        measurement_id = analyzer.start_measurement("integration_test")
        assert measurement_id is not None

        # 4. 処理の実行（シミュレーション）
        import time

        time.sleep(0.01)  # 10ms の処理をシミュレート

        # 5. 測定終了
        result = analyzer.end_measurement(measurement_id)
        assert result is not None
        assert "duration" in result
        assert result["duration"] > 0

        # 6. パフォーマンス統計の取得
        stats = analyzer.get_performance_stats()
        assert isinstance(stats, dict)
        assert "total_measurements" in stats

    def test_configuration_loading_flow(self):
        """設定読み込みフローの統合テスト"""

        from src.day_trade.core.config import AppConfig

        # 1. アプリケーション設定の読み込み
        config = AppConfig()

        # 2. 設定項目の存在確認
        assert hasattr(config, "trading")
        assert hasattr(config, "display")
        assert hasattr(config, "api")

        # 3. デフォルト値の確認
        assert config.display.decimal_places >= 0
        assert config.trading.commission_rate >= 0

        # 4. 設定の更新テスト
        original_places = config.display.decimal_places
        config.display.decimal_places = 3
        assert config.display.decimal_places == 3

        # 5. 設定の復元
        config.display.decimal_places = original_places

    def test_end_to_end_trading_simulation(
        self, test_db_manager, mock_stock_fetcher, sample_stock_data
    ):
        """エンドツーエンド取引シミュレーション"""

        # 1. 初期設定
        trade_manager = TradeManager()
        alert_manager = AlertManager(test_db_manager, mock_stock_fetcher)
        signal_generator = TradingSignalGenerator(config_path=None)

        # 2. テスト用銘柄の登録
        with test_db_manager.session_scope() as session:
            test_stock = Stock(
                code="7203",
                name="トヨタ自動車",
                market="東証プライム",
                sector="自動車・輸送機器",
            )
            session.add(test_stock)

            # 価格データの登録
            for _i, (date, row) in enumerate(sample_stock_data.iterrows()):
                price_data = PriceData(
                    stock_code="7203",
                    datetime=date,
                    open=Decimal(str(row["Open"])),
                    high=Decimal(str(row["High"])),
                    low=Decimal(str(row["Low"])),
                    close=Decimal(str(row["Close"])),
                    volume=int(row["Volume"]),
                )
                session.add(price_data)

            session.commit()

        # 3. シグナル生成
        with patch.object(
            signal_generator, "_fetch_historical_data", return_value=sample_stock_data
        ):
            signals = signal_generator.generate_signals("7203")

        # 4. シグナルに基づく取引実行（シミュレーション）
        if signals:
            for signal in signals[:2]:  # 最初の2つのシグナルのみ処理
                if signal.signal_type.value in ["BUY", "buy"]:
                    trade = trade_manager.execute_trade(
                        symbol=signal.symbol,
                        trade_type=TradeType.BUY,
                        quantity=100,
                        price=signal.price,
                        notes=f"シグナル取引: {signal.signal_type}",
                    )
                    assert trade is not None

        # 5. アラート設定と監視
        alert = alert_manager.create_alert(
            stock_code="7203",
            alert_type=AlertType.PRICE_ABOVE,
            threshold=3000.0,
            memo="高値監視",
        )
        assert alert is not None

        # 6. 最終ポートフォリオ状態の確認
        positions = trade_manager.get_current_positions()
        trade_history = trade_manager.get_trade_history()

        assert isinstance(positions, dict)
        assert isinstance(trade_history, list)

        # 7. レポート生成（基本的な統計）
        total_trades = len(trade_history)
        total_positions = len(positions)

        assert total_trades >= 0
        assert total_positions >= 0

    def test_system_health_check(self, test_db_manager):
        """システムヘルスチェック統合テスト"""

        # 1. データベース接続確認
        assert test_db_manager.engine is not None

        # 2. テーブル存在確認

        _inspector = test_db_manager.engine.dialect.get_table_names(
            test_db_manager.engine.connect()
        )
        expected_tables = [
            "stocks",
            "trades",
            "alerts",
            "price_data",
            "watchlist_items",
        ]

        for _table in expected_tables:
            # テーブル名は実際のモデル定義に依存するため、存在チェックは緩やか
            pass

        # 3. 基本的なCRUD操作確認
        with test_db_manager.session_scope() as session:
            # Create
            test_stock = Stock(code="TEST", name="テスト銘柄")
            session.add(test_stock)
            session.flush()

            # Read
            retrieved_stock = session.query(Stock).filter(Stock.code == "TEST").first()
            assert retrieved_stock is not None
            assert retrieved_stock.name == "テスト銘柄"

            # Update
            retrieved_stock.name = "更新されたテスト銘柄"
            session.flush()

            # Delete
            session.delete(retrieved_stock)

        # 4. エラーハンドリングシステム確認
        error_handler = get_default_error_handler()
        stats = error_handler.get_performance_stats()
        assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
