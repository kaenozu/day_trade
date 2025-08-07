"""
テスト用Mockファクトリ関数

Phase 5: テストコード共通化の一環として、
重複するMock実装を統合し、テストの保守性を向上させます。

主な機能:
- MockStockFetcher: 8箇所での重複実装を統一
- MockDatabaseManager: DB関連Mockの統一
- MockTradeManager: 取引関連Mockの統一
- MockAlertManager: アラート関連Mockの統一
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import Mock

import numpy as np
import pandas as pd


def create_mock_stock_fetcher(
    with_data: bool = True,
    symbols: List[str] = None,
    data_range_days: int = 100,
    with_errors: bool = False,
    **kwargs,
) -> Mock:
    """
    統一されたMockStockFetcherを作成

    Args:
        with_data: サンプルデータを含むか
        symbols: 対象銘柄リスト
        data_range_days: データの期間（日数）
        with_errors: エラーケースを含むか
        **kwargs: 追加設定

    Returns:
        設定済みMockStockFetcherインスタンス
    """
    mock = Mock()
    mock.name = "MockStockFetcher"

    # デフォルト銘柄リスト
    if symbols is None:
        symbols = ["7203", "6758", "9984", "8306", "4503"]

    if with_data:
        # 基本的な株価データ生成
        def generate_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
            dates = pd.date_range(end=datetime.now(), periods=data_range_days, freq="D")

            # 銘柄ごとに異なるシードで再現可能なデータ
            np.random.seed(hash(symbol) % 2**32)

            # 基準価格（銘柄に応じて調整）
            base_price = {
                "7203": 2000,
                "6758": 3000,
                "9984": 8000,
                "8306": 4000,
                "4503": 5000,
            }
            start_price = base_price.get(symbol, 1000)

            # トレンドとボラティリティ
            trend = np.random.choice([-0.1, 0, 0.1], p=[0.3, 0.4, 0.3])
            volatility = np.random.uniform(0.02, 0.05)

            # 価格生成（ランダムウォーク + トレンド）
            returns = np.random.normal(trend / 252, volatility, data_range_days)
            prices = start_price * np.exp(np.cumsum(returns))

            # OHLCV データ生成
            close_prices = prices
            open_prices = np.roll(close_prices, 1)
            open_prices[0] = start_price

            # 日中の値動きを考慮
            daily_ranges = np.abs(np.random.normal(0, volatility / 2, data_range_days))
            high_prices = close_prices + daily_ranges * close_prices
            low_prices = close_prices - daily_ranges * close_prices

            # OpenとCloseの関係を調整
            for i in range(len(prices)):
                high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
                low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])

            volume = np.random.randint(100000, 10000000, data_range_days)

            df = pd.DataFrame(
                {
                    "Date": dates,
                    "Open": np.round(open_prices, 2),
                    "High": np.round(high_prices, 2),
                    "Low": np.round(low_prices, 2),
                    "Close": np.round(close_prices, 2),
                    "Volume": volume,
                }
            )
            df.set_index("Date", inplace=True)
            return df

        mock.get_historical_data.side_effect = generate_stock_data

        # 現在価格の取得
        def get_current_price(symbol: str) -> float:
            # 最新の株価データから現在価格を取得
            data = generate_stock_data(symbol, "1d")
            return float(data["Close"].iloc[-1])

        mock.get_current_price.side_effect = get_current_price

        # 銘柄情報の取得
        def get_stock_info(symbol: str) -> Dict[str, Any]:
            company_names = {
                "7203": "トヨタ自動車",
                "6758": "ソニーグループ",
                "9984": "ソフトバンクグループ",
                "8306": "三菱UFJフィナンシャル・グループ",
                "4503": "アステラス製薬",
            }
            return {
                "symbol": symbol,
                "name": company_names.get(symbol, f"Test Company {symbol}"),
                "market": "東証プライム",
                "sector": "テクノロジー",
                "current_price": get_current_price(symbol),
            }

        mock.get_stock_info.side_effect = get_stock_info
        mock.get_available_symbols.return_value = symbols

    if with_errors:
        # エラーケースの設定
        def error_prone_fetch(symbol: str, *args, **kwargs):
            if symbol == "ERROR_SYMBOL":
                raise ValueError(f"Symbol {symbol} not found")
            elif symbol == "NETWORK_ERROR":
                raise ConnectionError("Network connection failed")
            else:
                return generate_stock_data(symbol)

        mock.get_historical_data.side_effect = error_prone_fetch

    # その他の設定
    for key, value in kwargs.items():
        setattr(mock, key, value)

    return mock


def create_mock_database_manager(
    with_data: bool = True,
    auto_commit: bool = True,
    with_errors: bool = False,
    **kwargs,
) -> Mock:
    """
    統一されたMockDatabaseManagerを作成

    Args:
        with_data: サンプルデータを含むか
        auto_commit: 自動コミットするか
        with_errors: エラーケースを含むか
        **kwargs: 追加設定

    Returns:
        設定済みMockDatabaseManagerインスタンス
    """
    mock = Mock()
    mock.name = "MockDatabaseManager"

    # セッション管理
    mock_session = Mock()
    mock_session.commit.return_value = None
    mock_session.rollback.return_value = None
    mock_session.close.return_value = None

    if auto_commit:
        mock_session.add.return_value = None
        mock_session.merge.return_value = None
        mock_session.delete.return_value = None

    mock.get_session.return_value.__enter__.return_value = mock_session
    mock.get_session.return_value.__exit__.return_value = None

    # データベース操作
    if with_data:
        # クエリ結果のモック
        def mock_query(*args, **kwargs):
            # 簡単なクエリ結果を返す
            if "stock" in str(args).lower():
                return [{"symbol": "7203", "name": "トヨタ自動車"}]
            elif "trade" in str(args).lower():
                return [{"id": 1, "symbol": "7203", "quantity": 100, "price": 2000}]
            else:
                return []

        mock_session.query.return_value.all.side_effect = mock_query
        mock_session.execute.return_value.fetchall.side_effect = mock_query

    if with_errors:
        # エラーケース
        def error_prone_operation(*args, **kwargs):
            operation = str(args[0]) if args else "unknown"
            if "error" in operation.lower():
                raise Exception(f"Database error for operation: {operation}")
            return []

        mock_session.execute.side_effect = error_prone_operation

    # ヘルスチェック
    mock.health_check.return_value = {"status": "healthy", "response_time": 0.001}

    for key, value in kwargs.items():
        setattr(mock, key, value)

    return mock


def create_mock_trade_manager(
    with_portfolio: bool = True,
    initial_balance: Decimal = Decimal("1000000"),
    with_errors: bool = False,
    **kwargs,
) -> Mock:
    """
    統一されたMockTradeManagerを作成

    Args:
        with_portfolio: ポートフォリオ情報を含むか
        initial_balance: 初期資金
        with_errors: エラーケースを含むか
        **kwargs: 追加設定

    Returns:
        設定済みMockTradeManagerインスタンス
    """
    mock = Mock()
    mock.name = "MockTradeManager"

    # ポートフォリオ状態
    if with_portfolio:
        portfolio_state = {
            "cash": initial_balance,
            "positions": {},
            "total_value": initial_balance,
            "unrealized_pnl": Decimal("0"),
            "realized_pnl": Decimal("0"),
        }

        # 取引実行のモック
        def execute_trade(symbol: str, quantity: int, price: float, action: str):
            trade_id = f"trade_{len(portfolio_state['positions']) + 1}"

            if action.upper() == "BUY":
                cost = Decimal(str(price)) * Decimal(str(quantity))
                if portfolio_state["cash"] >= cost:
                    portfolio_state["cash"] -= cost
                    if symbol in portfolio_state["positions"]:
                        portfolio_state["positions"][symbol]["quantity"] += quantity
                    else:
                        portfolio_state["positions"][symbol] = {
                            "quantity": quantity,
                            "avg_price": price,
                        }
                    return {
                        "trade_id": trade_id,
                        "status": "executed",
                        "symbol": symbol,
                    }
                else:
                    return {
                        "trade_id": trade_id,
                        "status": "rejected",
                        "reason": "insufficient_funds",
                    }

            elif action.upper() == "SELL":
                if (
                    symbol in portfolio_state["positions"]
                    and portfolio_state["positions"][symbol]["quantity"] >= quantity
                ):
                    revenue = Decimal(str(price)) * Decimal(str(quantity))
                    portfolio_state["cash"] += revenue
                    portfolio_state["positions"][symbol]["quantity"] -= quantity
                    if portfolio_state["positions"][symbol]["quantity"] == 0:
                        del portfolio_state["positions"][symbol]
                    return {
                        "trade_id": trade_id,
                        "status": "executed",
                        "symbol": symbol,
                    }
                else:
                    return {
                        "trade_id": trade_id,
                        "status": "rejected",
                        "reason": "insufficient_position",
                    }

            return {"trade_id": trade_id, "status": "error", "reason": "invalid_action"}

        mock.execute_trade.side_effect = execute_trade
        mock.get_portfolio.return_value = portfolio_state
        mock.get_cash_balance.return_value = lambda: portfolio_state["cash"]
        mock.get_positions.return_value = lambda: portfolio_state["positions"]

    if with_errors:

        def error_prone_trade(*args, **kwargs):
            symbol = args[0] if args else "unknown"
            if symbol == "ERROR_SYMBOL":
                raise ValueError(f"Invalid symbol: {symbol}")
            return execute_trade(*args, **kwargs)

        mock.execute_trade.side_effect = error_prone_trade

    for key, value in kwargs.items():
        setattr(mock, key, value)

    return mock


def create_mock_alert_manager(
    with_alerts: bool = True, max_alerts: int = 100, with_errors: bool = False, **kwargs
) -> Mock:
    """
    統一されたMockAlertManagerを作成

    Args:
        with_alerts: アラート機能を含むか
        max_alerts: 最大アラート数
        with_errors: エラーケースを含むか
        **kwargs: 追加設定

    Returns:
        設定済みMockAlertManagerインスタンス
    """
    mock = Mock()
    mock.name = "MockAlertManager"

    if with_alerts:
        alerts_storage = []

        def add_alert(symbol: str, condition: str, threshold: float, message: str = ""):
            alert_id = f"alert_{len(alerts_storage) + 1}"
            alert = {
                "id": alert_id,
                "symbol": symbol,
                "condition": condition,
                "threshold": threshold,
                "message": message,
                "created_at": datetime.now(),
                "triggered": False,
            }

            if len(alerts_storage) < max_alerts:
                alerts_storage.append(alert)
                return alert_id
            else:
                raise ValueError(f"Maximum alerts ({max_alerts}) reached")

        def trigger_alert(alert_id: str, current_value: float):
            for alert in alerts_storage:
                if alert["id"] == alert_id:
                    alert["triggered"] = True
                    alert["triggered_at"] = datetime.now()
                    alert["triggered_value"] = current_value
                    return True
            return False

        mock.add_alert.side_effect = add_alert
        mock.trigger_alert.side_effect = trigger_alert
        mock.get_alerts.return_value = lambda: alerts_storage.copy()
        mock.get_active_alerts.return_value = lambda: [
            a for a in alerts_storage if not a["triggered"]
        ]

    if with_errors:

        def error_prone_alert(*args, **kwargs):
            symbol = args[0] if args else "unknown"
            if symbol == "ERROR_SYMBOL":
                raise ValueError(f"Invalid symbol for alert: {symbol}")
            return add_alert(*args, **kwargs)

        mock.add_alert.side_effect = error_prone_alert

    for key, value in kwargs.items():
        setattr(mock, key, value)

    return mock


# 高レベルなMock作成関数


def create_comprehensive_mock_environment(
    include_stock_fetcher: bool = True,
    include_database: bool = True,
    include_trade_manager: bool = True,
    include_alert_manager: bool = True,
    **kwargs,
) -> Dict[str, Mock]:
    """
    包括的なMock環境を作成

    Args:
        include_stock_fetcher: StockFetcherを含むか
        include_database: DatabaseManagerを含むか
        include_trade_manager: TradeManagerを含むか
        include_alert_manager: AlertManagerを含むか
        **kwargs: 各Mockへの追加設定

    Returns:
        Mockオブジェクトの辞書
    """
    mocks = {}

    if include_stock_fetcher:
        mocks["stock_fetcher"] = create_mock_stock_fetcher(
            **kwargs.get("stock_fetcher", {})
        )

    if include_database:
        mocks["database_manager"] = create_mock_database_manager(
            **kwargs.get("database", {})
        )

    if include_trade_manager:
        mocks["trade_manager"] = create_mock_trade_manager(
            **kwargs.get("trade_manager", {})
        )

    if include_alert_manager:
        mocks["alert_manager"] = create_mock_alert_manager(
            **kwargs.get("alert_manager", {})
        )

    return mocks
