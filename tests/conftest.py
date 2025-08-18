#!/usr/bin/env python3
"""
pytestの共通設定とフィクスチャ
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """プロジェクトルートパスフィクスチャ"""
    return project_root


@pytest.fixture(scope="function")
def temp_dir():
    """一時ディレクトリフィクスチャ"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def sample_data():
    """サンプルデータフィクスチャ"""
    return {
        "test_symbols": ["7203", "8306", "9984"],
        "test_prices": [1000, 2000, 3000],
        "test_config": {
            "analysis": {"enabled": True},
            "alerts": {"enabled": False}
        }
    }


@pytest.fixture(scope="session")
def mock_data_provider():
    """モックデータプロバイダーフィクスチャ"""
    class MockDataProvider:
        def get_stock_price(self, symbol):
            return {"symbol": symbol, "price": 1000, "volume": 10000}

        def get_market_data(self):
            return {"market_open": True, "timestamp": "2024-01-01T09:00:00"}

    return MockDataProvider()


def pytest_configure(config):
    """pytest設定"""
    config.addinivalue_line(
        "markers", "unit: ユニットテストマーカー"
    )
    config.addinivalue_line(
        "markers", "integration: 統合テストマーカー"
    )
    config.addinivalue_line(
        "markers", "performance: パフォーマンステストマーカー"
    )
    config.addinivalue_line(
        "markers", "slow: 実行時間の長いテストマーカー"
    )


def pytest_collection_modifyitems(config, items):
    """テストアイテムの動的変更"""
    skip_slow = pytest.mark.skip(reason="--run-slow オプションが必要")

    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow", default=False):
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """カスタムオプション追加"""
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="実行時間の長いテストも実行"
    )
