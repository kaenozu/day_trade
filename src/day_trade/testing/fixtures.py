#!/usr/bin/env python3
"""
テストフィクスチャとデータ管理
Test Fixtures and Data Management

Issue #760: 包括的テスト自動化と検証フレームワークの構築
"""

import os
import json
import pickle
import logging
import tempfile
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Generator
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest
from unittest.mock import Mock, MagicMock
import factory
from factory import Factory, Sequence, LazyFunction
import uuid

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class TestDataSpec:
    """テストデータ仕様"""
    name: str
    data_type: str  # "stock", "market", "ml_model", "performance"
    size: int = 1000
    date_range: Tuple[str, str] = ("2023-01-01", "2024-01-01")
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)


class TestDataManager:
    """テストデータ管理"""

    def __init__(self, data_dir: str = "tests/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Any] = {}

    def get_stock_data(
        self,
        symbol: str = "7203.T",
        days: int = 100,
        start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """株価データ取得"""
        cache_key = f"stock_{symbol}_{days}_{start_date}"

        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        # データ生成
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days+10)
        else:
            start_date = pd.to_datetime(start_date)

        dates = pd.date_range(start=start_date, periods=days, freq='D')

        # リアルな株価データシミュレーション
        np.random.seed(42)  # 再現可能性のため

        # 初期価格
        initial_price = 2000.0
        prices = [initial_price]

        # ランダムウォークで価格生成
        for i in range(1, days):
            change = np.random.normal(0, 0.02)  # 平均0%, 標準偏差2%の変動
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # 価格は1円以上

        # OHLCデータ作成
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Open, High, Low 生成
            daily_volatility = np.random.normal(0, 0.01)
            high = close * (1 + abs(daily_volatility))
            low = close * (1 - abs(daily_volatility))
            open_price = close * (1 + np.random.normal(0, 0.005))

            # Volume 生成
            volume = int(np.random.lognormal(12, 0.5))  # 対数正規分布

            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume,
                'Symbol': symbol
            })

        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)

        # キャッシュに保存
        self.cache[cache_key] = df.copy()

        return df

    def get_market_data(self, market: str = "TOPIX", days: int = 100) -> pd.DataFrame:
        """市場データ取得"""
        cache_key = f"market_{market}_{days}"

        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        # 市場インデックスデータ生成
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        np.random.seed(123)
        initial_value = 2000.0
        values = [initial_value]

        for i in range(1, days):
            change = np.random.normal(0.0005, 0.015)  # 若干上昇トレンド
            new_value = values[-1] * (1 + change)
            values.append(max(new_value, 100.0))

        df = pd.DataFrame({
            'Date': dates,
            'Value': values,
            'Market': market
        })
        df.set_index('Date', inplace=True)

        self.cache[cache_key] = df.copy()
        return df

    def get_feature_data(
        self,
        samples: int = 1000,
        features: int = 10,
        target_type: str = "regression"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """機械学習用特徴量データ取得"""
        cache_key = f"features_{samples}_{features}_{target_type}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        np.random.seed(456)

        # 特徴量生成
        X = np.random.randn(samples, features)

        # ターゲット生成
        if target_type == "regression":
            # 線形関係 + ノイズ
            weights = np.random.randn(features)
            y = X @ weights + np.random.randn(samples) * 0.1
        elif target_type == "classification":
            # バイナリ分類
            weights = np.random.randn(features)
            linear_combination = X @ weights
            y = (linear_combination > np.median(linear_combination)).astype(int)
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

        data = (X.astype(np.float32), y.astype(np.float32))
        self.cache[cache_key] = data

        return data

    def get_performance_data(self, metric_count: int = 5, time_points: int = 100) -> pd.DataFrame:
        """パフォーマンスメトリクスデータ取得"""
        cache_key = f"performance_{metric_count}_{time_points}"

        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=time_points),
            periods=time_points,
            freq='H'
        )

        metrics = {}
        np.random.seed(789)

        for i in range(metric_count):
            metric_name = f"metric_{i+1}"
            # トレンドのあるランダムデータ
            trend = np.linspace(0, 1, time_points) * np.random.uniform(-0.5, 0.5)
            noise = np.random.normal(0, 0.1, time_points)
            values = trend + noise + np.random.uniform(0, 10)
            metrics[metric_name] = np.maximum(values, 0)  # 負の値を避ける

        df = pd.DataFrame(metrics, index=timestamps)

        self.cache[cache_key] = df.copy()
        return df

    def save_test_data(self, name: str, data: Any, format: str = "pickle") -> str:
        """テストデータ保存"""
        if format == "pickle":
            file_path = self.data_dir / f"{name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        elif format == "json":
            file_path = self.data_dir / f"{name}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "csv":
            file_path = self.data_dir / f"{name}.csv"
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path)
            else:
                raise ValueError("CSV format requires pandas DataFrame")
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Test data saved: {file_path}")
        return str(file_path)

    def load_test_data(self, name: str, format: str = "pickle") -> Any:
        """テストデータ読み込み"""
        if format == "pickle":
            file_path = self.data_dir / f"{name}.pkl"
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif format == "json":
            file_path = self.data_dir / f"{name}.json"
            with open(file_path, 'r') as f:
                return json.load(f)
        elif format == "csv":
            file_path = self.data_dir / f"{name}.csv"
            return pd.read_csv(file_path, index_col=0)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_cache(self) -> None:
        """キャッシュクリア"""
        self.cache.clear()
        logger.info("Test data cache cleared")


class MockDataGenerator:
    """モックデータ生成器"""

    @staticmethod
    def create_mock_model() -> Mock:
        """MLモデルモック作成"""
        model = Mock()

        # predict メソッドモック
        def mock_predict(X):
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            return np.random.randn(X.shape[0], 1)

        model.predict = Mock(side_effect=mock_predict)
        model.fit = Mock(return_value=model)
        model.score = Mock(return_value=0.85)

        return model

    @staticmethod
    def create_mock_api_response(
        status_code: int = 200,
        data: Optional[Dict] = None
    ) -> Mock:
        """API レスポンスモック作成"""
        response = Mock()
        response.status_code = status_code
        response.json = Mock(return_value=data or {"status": "success"})
        response.text = json.dumps(data or {"status": "success"})

        return response

    @staticmethod
    def create_mock_database() -> Mock:
        """データベースモック作成"""
        db = Mock()

        # 基本的なCRUD操作
        db.execute = Mock(return_value=True)
        db.fetchall = Mock(return_value=[])
        db.fetchone = Mock(return_value=None)
        db.commit = Mock(return_value=True)
        db.rollback = Mock(return_value=True)

        return db

    @staticmethod
    def create_mock_cache() -> Mock:
        """キャッシュモック作成"""
        cache = Mock()
        cache_data = {}

        def mock_get(key):
            return cache_data.get(key)

        def mock_set(key, value, ttl=None):
            cache_data[key] = value
            return True

        def mock_delete(key):
            return cache_data.pop(key, None) is not None

        cache.get = Mock(side_effect=mock_get)
        cache.set = Mock(side_effect=mock_set)
        cache.delete = Mock(side_effect=mock_delete)
        cache.clear = Mock(side_effect=lambda: cache_data.clear())

        return cache


# Factory Boy ファクトリー定義
class StockDataFactory(Factory):
    """株価データファクトリー"""

    class Meta:
        model = dict

    symbol = Sequence(lambda n: f"stock_{n:04d}")
    price = factory.LazyFunction(lambda: round(np.random.uniform(100, 5000), 2))
    volume = factory.LazyFunction(lambda: int(np.random.uniform(1000, 1000000)))
    timestamp = factory.LazyFunction(lambda: datetime.now())


class PerformanceMetricFactory(Factory):
    """パフォーマンスメトリクスファクトリー"""

    class Meta:
        model = dict

    metric_name = Sequence(lambda n: f"metric_{n}")
    value = factory.LazyFunction(lambda: round(np.random.uniform(0, 100), 3))
    timestamp = factory.LazyFunction(lambda: datetime.now())
    unit = "ms"


class FixtureRegistry:
    """フィクスチャレジストリ"""

    def __init__(self):
        self.fixtures: Dict[str, Callable] = {}
        self.data_manager = TestDataManager()

    def register(self, name: str, fixture_func: Callable) -> None:
        """フィクスチャ登録"""
        self.fixtures[name] = fixture_func
        logger.info(f"Fixture registered: {name}")

    def get(self, name: str) -> Any:
        """フィクスチャ取得"""
        if name not in self.fixtures:
            raise ValueError(f"Fixture not found: {name}")

        return self.fixtures[name]()

    def list_fixtures(self) -> List[str]:
        """登録フィクスチャ一覧"""
        return list(self.fixtures.keys())


class CommonFixtures:
    """共通フィクスチャ集"""

    @staticmethod
    @pytest.fixture
    def test_data_manager():
        """テストデータマネージャー フィクスチャ"""
        manager = TestDataManager()
        yield manager
        manager.clear_cache()

    @staticmethod
    @pytest.fixture
    def sample_stock_data():
        """サンプル株価データ フィクスチャ"""
        manager = TestDataManager()
        return manager.get_stock_data("TEST.T", days=30)

    @staticmethod
    @pytest.fixture
    def sample_ml_data():
        """サンプルML データ フィクスチャ"""
        manager = TestDataManager()
        return manager.get_feature_data(samples=100, features=5)

    @staticmethod
    @pytest.fixture
    def mock_model():
        """モックMLモデル フィクスチャ"""
        return MockDataGenerator.create_mock_model()

    @staticmethod
    @pytest.fixture
    def mock_api():
        """モックAPI フィクスチャ"""
        return MockDataGenerator.create_mock_api_response()

    @staticmethod
    @pytest.fixture
    def temp_directory():
        """一時ディレクトリ フィクスチャ"""
        temp_dir = tempfile.mkdtemp(prefix="test_")
        yield temp_dir

        # クリーンアップ
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# pytest プラグイン設定
def pytest_configure(config):
    """pytest 設定"""
    # カスタムマーカー登録
    config.addinivalue_line(
        "markers", "integration: integration test marker"
    )
    config.addinivalue_line(
        "markers", "performance: performance test marker"
    )
    config.addinivalue_line(
        "markers", "slow: slow test marker"
    )


# 使用例とテスト
async def test_fixtures_example():
    """フィクスチャ使用例"""

    # データマネージャー使用
    data_manager = TestDataManager()

    # 株価データ取得
    stock_data = data_manager.get_stock_data("7203.T", days=50)
    print(f"Stock data shape: {stock_data.shape}")
    print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")

    # ML データ取得
    X, y = data_manager.get_feature_data(samples=200, features=8)
    print(f"Feature data shape: X={X.shape}, y={y.shape}")

    # モックオブジェクト作成
    mock_model = MockDataGenerator.create_mock_model()
    predictions = mock_model.predict(X[:10])
    print(f"Mock predictions shape: {predictions.shape}")

    # ファクトリーでデータ生成
    stock_records = [StockDataFactory() for _ in range(5)]
    print(f"Generated {len(stock_records)} stock records")

    # フィクスチャレジストリ使用
    registry = FixtureRegistry()
    registry.register("test_stocks", lambda: data_manager.get_stock_data("TEST.T"))

    test_data = registry.get("test_stocks")
    print(f"Registry test data shape: {test_data.shape}")

    return {
        "stock_data_shape": stock_data.shape,
        "ml_data_shapes": (X.shape, y.shape),
        "mock_predictions_shape": predictions.shape,
        "factory_records_count": len(stock_records)
    }


if __name__ == "__main__":
    import asyncio
    result = asyncio.run(test_fixtures_example())
    print("\n=== Fixture Test Results ===")
    for key, value in result.items():
        print(f"{key}: {value}")