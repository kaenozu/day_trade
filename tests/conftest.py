"""
テスト用の共通フィクスチャとヘルパー関数

テスト全体で使用される共通のセットアップやデータを提供し、
テストの重複を削減し、保守性を向上させる。
"""

import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.day_trade.analysis.patterns import ChartPatternRecognizer
from src.day_trade.analysis.signals import TradingSignalGenerator
from src.day_trade.analysis.indicators import TechnicalIndicators
from src.day_trade.core.trade_manager import TradeManager
from src.day_trade.data.stock_fetcher import StockFetcher


@pytest.fixture(scope="session")
def sample_stock_data():
    """セッション全体で使用する基本的な株価データ"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    np.random.seed(42)  # 再現可能なデータ

    # トレンドのあるデータを生成
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    close_prices = trend + noise

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close_prices + np.random.randn(100) * 0.5,
            "High": close_prices + np.abs(np.random.randn(100)) * 1.5,
            "Low": close_prices - np.abs(np.random.randn(100)) * 1.5,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 5000000, 100),
        }
    )
    df.set_index("Date", inplace=True)
    return df


@pytest.fixture(scope="session")
def crossover_stock_data():
    """クロスオーバーパターン用のテストデータ"""
    dates = pd.date_range(end=datetime.now(), periods=50, freq="D")

    # 明確なクロスオーバーパターンを作成
    fast_trend = np.concatenate([
        np.linspace(100, 95, 20),   # 下降
        np.linspace(95, 105, 15),   # 上昇（ゴールデンクロス）
        np.linspace(105, 98, 15),   # 下降（デッドクロス）
    ])

    slow_trend = np.concatenate([
        np.linspace(102, 100, 20),  # 緩やかな下降
        np.linspace(100, 102, 15),  # 緩やかな上昇
        np.linspace(102, 100, 15),  # 緩やかな下降
    ])

    # ノイズを追加
    fast_prices = fast_trend + np.random.randn(50) * 0.5
    slow_prices = slow_trend + np.random.randn(50) * 0.3

    df = pd.DataFrame({
        "Date": dates,
        "Open": fast_prices + np.random.randn(50) * 0.2,
        "High": fast_prices + np.abs(np.random.randn(50)) * 0.8,
        "Low": fast_prices - np.abs(np.random.randn(50)) * 0.8,
        "Close": fast_prices,
        "Volume": np.random.randint(2000000, 8000000, 50),
    })
    df.set_index("Date", inplace=True)
    return df


@pytest.fixture
def pattern_recognizer():
    """ChartPatternRecognizerインスタンス"""
    return ChartPatternRecognizer()


@pytest.fixture
def mock_stock_fetcher():
    """モック化されたStockFetcher"""
    mock_fetcher = Mock(spec=StockFetcher)

    # 基本的な価格データをモック
    mock_fetcher.get_current_price.return_value = {
        "current_price": 1100.0,
        "change": 50.0,
        "change_percent": 4.76,
        "volume": 25000,
        "market_cap": 1000000000
    }

    # 企業情報をモック
    mock_fetcher.get_company_info.return_value = {
        "name": "テスト株式会社",
        "sector": "テクノロジー",
        "industry": "ソフトウェア",
        "market_cap": 1000000000
    }

    return mock_fetcher


@pytest.fixture
def trade_manager():
    """テスト用TradeManager（データベースなし）"""
    return TradeManager(
        commission_rate=Decimal("0.001"),
        tax_rate=Decimal("0.2"),
        load_from_db=False,
    )


@pytest.fixture
def signal_generator():
    """TradingSignalGenerator（デフォルト設定）"""
    config_path = "config/signal_rules.json"
    return TradingSignalGenerator(config_path)


@pytest.fixture
def temp_database():
    """テスト用一時データベース"""
    # 一時ファイルを作成
    temp_fd, temp_path = tempfile.mkstemp(suffix='.db')
    os.close(temp_fd)

    # 環境変数でテスト用DBを指定
    original_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite:///{temp_path}"

    yield temp_path

    # クリーンアップ
    if original_db_url:
        os.environ["DATABASE_URL"] = original_db_url
    else:
        os.environ.pop("DATABASE_URL", None)

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_trades_data():
    """テスト用取引データ"""
    return [
        {
            "symbol": "7203",
            "trade_type": "buy",
            "quantity": 100,
            "price": Decimal("2800.00"),
            "timestamp": datetime.now() - timedelta(days=10),
            "commission": Decimal("2.80")
        },
        {
            "symbol": "7203",
            "trade_type": "sell",
            "quantity": 100,
            "price": Decimal("2900.00"),
            "timestamp": datetime.now() - timedelta(days=5),
            "commission": Decimal("2.90")
        },
        {
            "symbol": "9984",
            "trade_type": "buy",
            "quantity": 50,
            "price": Decimal("9500.00"),
            "timestamp": datetime.now() - timedelta(days=8),
            "commission": Decimal("4.75")
        }
    ]


class TestHelpers:
    """テスト用のヘルパー関数群"""

    @staticmethod
    def create_price_data_with_pattern(
        pattern_type: str,
        periods: int = 100,
        base_price: float = 100.0
    ) -> pd.DataFrame:
        """特定のパターンを持つ価格データを生成"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="D")

        if pattern_type == "uptrend":
            trend = np.linspace(base_price, base_price * 1.3, periods)
        elif pattern_type == "downtrend":
            trend = np.linspace(base_price, base_price * 0.7, periods)
        elif pattern_type == "sideways":
            trend = np.full(periods, base_price) + np.sin(np.linspace(0, 4*np.pi, periods)) * 2
        else:
            trend = np.full(periods, base_price)

        noise = np.random.randn(periods) * (base_price * 0.02)
        close_prices = trend + noise

        return pd.DataFrame({
            "Date": dates,
            "Open": close_prices + np.random.randn(periods) * 0.5,
            "High": close_prices + np.abs(np.random.randn(periods)) * 1.5,
            "Low": close_prices - np.abs(np.random.randn(periods)) * 1.5,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 5000000, periods),
        }).set_index("Date")

    @staticmethod
    def assert_signal_structure(signal_data: Dict[str, Any]):
        """シグナルデータの構造をアサート"""
        required_fields = ["signal_type", "strength", "confidence", "timestamp"]
        for field in required_fields:
            assert field in signal_data, f"Missing required field: {field}"

        assert signal_data["confidence"] >= 0.0
        assert signal_data["confidence"] <= 100.0
        assert signal_data["strength"] in ["WEAK", "MEDIUM", "STRONG"]

    @staticmethod
    def assert_pattern_result_structure(pattern_result: pd.DataFrame, expected_columns: list):
        """パターン認識結果の構造をアサート"""
        assert isinstance(pattern_result, pd.DataFrame)
        for col in expected_columns:
            assert col in pattern_result.columns, f"Missing column: {col}"
        assert len(pattern_result) >= 0


# pytest設定でwarning filterを設定
@pytest.fixture(autouse=True)
def configure_warnings():
    """テスト実行時の警告を設定"""
    import warnings

    # Pandas関連の警告を無視
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
    warnings.filterwarnings("ignore", message=".*Bitwise inversion.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*R\\^2 score is not well-defined.*")
    warnings.filterwarnings("ignore", message=".*Remove `format_exc_info`.*")


# pytest設定マーカー
def pytest_configure(config):
    """pytest設定"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (run with --integration)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (run with --slow)"
    )
