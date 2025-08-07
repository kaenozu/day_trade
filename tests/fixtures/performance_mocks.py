"""
パフォーマンステスト用の統合モック設定
全ての重い処理をモック化してテストの高速化を実現する
"""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


class FastMockMixin:
    """
    高速化のための統合モッククラス
    time.sleep、DB操作、API呼び出しなどをすべてモック化
    """

    @pytest.fixture(autouse=True)
    def setup_performance_mocks(self, monkeypatch):
        """
        すべてのテストで自動的に適用される高速化モック
        """
        # time.sleepを無効化
        monkeypatch.setattr(time, "sleep", lambda x: None)

        # yfinanceの重い処理をモック化
        mock_ticker = Mock()
        mock_ticker.history.return_value = self._create_mock_stock_data()
        mock_ticker.info = {
            "shortName": "Mock Company",
            "regularMarketPrice": 1500.0,
            "previousClose": 1480.0,
            "marketCap": 5000000000,
        }

        # yfinance.Tickerをモック化
        monkeypatch.setattr("yfinance.Ticker", lambda symbol: mock_ticker)

        # データベースセッション作成の高速化
        self._setup_database_mocks(monkeypatch)

        # ファイルI/O操作の高速化
        self._setup_file_mocks(monkeypatch)

    def _create_mock_stock_data(self, days=100):
        """モック用の株価データを生成"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days), end=datetime.now(), freq="D"
        )

        # リアルな株価データをシミュレート
        base_price = 1500.0
        returns = np.random.normal(0.001, 0.02, len(dates))  # 日次リターン
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        return pd.DataFrame(
            {
                "Open": [p * np.random.uniform(0.99, 1.01) for p in prices],
                "High": [p * np.random.uniform(1.01, 1.05) for p in prices],
                "Low": [p * np.random.uniform(0.95, 0.99) for p in prices],
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

    def _setup_database_mocks(self, monkeypatch):
        """データベース操作のモック設定"""
        # SQLAlchemy bulk操作の高速化
        mock_bulk_insert = Mock()
        mock_bulk_insert.return_value = None

        # セッション操作をモック化
        mock_session = Mock()
        mock_session.bulk_insert_mappings = mock_bulk_insert
        mock_session.execute = Mock()
        mock_session.commit = Mock()
        mock_session.rollback = Mock()
        mock_session.flush = Mock()
        mock_session.close = Mock()

        # データベース接続をモック化
        monkeypatch.setattr("sqlalchemy.create_engine", lambda *args, **kwargs: Mock())

    def _setup_file_mocks(self, monkeypatch):
        """ファイル操作のモック設定"""
        # ファイル読み書きの高速化
        mock_open_data = {
            "config.json": '{"test": "data"}',
            "settings.yaml": "test: data",
        }

        def mock_open_func(filename, mode="r", *args, **kwargs):
            mock_file = Mock()
            if "r" in mode and filename in mock_open_data:
                mock_file.read.return_value = mock_open_data[filename]
                mock_file.__enter__.return_value = mock_file
                mock_file.__exit__.return_value = None
            return mock_file

        monkeypatch.setattr("builtins.open", mock_open_func)


@pytest.fixture
def fast_mocks():
    """
    高速化モックのファクトリー関数
    個別テストで必要に応じて使用
    """
    return FastMockMixin()


def mock_time_operations():
    """
    時間関連操作のモック化デコレーター
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch("time.sleep", return_value=None), patch(
                "time.time", return_value=1234567890.0
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def mock_heavy_operations():
    """
    重い操作の一括モック化デコレーター
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch("time.sleep", return_value=None), patch(
                "requests.get",
                return_value=Mock(json=lambda: {"data": "mock"}, status_code=200),
            ), patch("sqlalchemy.create_engine", return_value=Mock()):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# パフォーマンス計測用ユーティリティ
class PerformanceTimer:
    """テスト実行時間の計測"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            return 0
        return self.end_time - self.start_time


@pytest.fixture
def perf_timer():
    """パフォーマンス計測フィクスチャ"""
    return PerformanceTimer()
