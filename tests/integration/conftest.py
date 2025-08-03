"""
統合テスト用 pytest 設定

統合テスト実行時の共通設定とフィクスチャを定義します。
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import contextlib

# 統合テスト用のマーカーを定義
def pytest_configure(config):
    """pytest設定の初期化"""
    config.addinivalue_line(
        "markers", "integration: 統合テスト（複数コンポーネント間の連携テスト）"
    )
    config.addinivalue_line(
        "markers", "slow: 実行時間の長いテスト"
    )


@pytest.fixture(scope="session")
def integration_test_db():
    """統合テスト用のテスト専用データベース

    TODO: 実装時に以下を含める:
    - テスト用SQLiteデータベースの作成
    - スキーマの初期化
    - テストデータの投入
    - セッション終了時のクリーンアップ
    """
    # プレースホルダー実装
    test_db_path = tempfile.mktemp(suffix='.db')
    yield test_db_path
    # クリーンアップ（実装時に追加）
    with contextlib.suppress(Exception):
        Path(test_db_path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def test_data_directory():
    """統合テスト用のテストデータディレクトリ

    テスト用のCSVファイル、設定ファイル等を格納
    """
    temp_dir = tempfile.mkdtemp(prefix="day_trade_test_")
    yield Path(temp_dir)
    # クリーンアップ
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_external_apis():
    """外部API呼び出しのモック化

    統合テストでも外部APIの実際の呼び出しは避け、
    制御されたレスポンスを使用
    """
    with patch('yfinance.download') as mock_yf, \
         patch('pandas_datareader.data.DataReader') as mock_pdr:

        # デフォルトのモック動作を設定
        mock_yf.return_value = None
        mock_pdr.return_value = None

        yield {
            'yfinance': mock_yf,
            'pandas_datareader': mock_pdr
        }


@pytest.fixture
def integration_config():
    """統合テスト用の設定

    テスト環境に適した設定値を提供
    """
    return {
        'database': {
            'url': 'sqlite:///test.db',
            'echo': False
        },
        'api': {
            'timeout': 5,
            'max_retries': 2
        },
        'logging': {
            'level': 'WARNING'  # テスト時はログレベルを上げる
        }
    }


# 統合テスト用のカスタムアサーション
def assert_valid_stock_data(data):
    """株価データの妥当性をアサート

    統合テストで株価データの品質を確認
    """
    assert data is not None, "株価データがNoneです"
    # TODO: より詳細な検証ロジックを追加


def assert_performance_within_limits(execution_time, max_time=30.0):
    """パフォーマンス要件のアサート

    実行時間が許容範囲内であることを確認
    """
    assert execution_time <= max_time, f"実行時間 {execution_time}秒 が制限 {max_time}秒 を超過"


# pytest実行時のオプション設定
def pytest_addoption(parser):
    """カスタムコマンドラインオプションの追加"""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="統合テストを実行"
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="実行時間の長いテストも含めて実行"
    )


def pytest_collection_modifyitems(config, items):
    """テストアイテムの修正

    統合テストのマーカーに基づいてテストを選択/除外
    """
    if config.getoption("--integration"):
        # --integration フラグが指定された場合は統合テストのみ実行
        skip_unit = pytest.mark.skip(reason="--integration フラグにより単体テストをスキップ")
        for item in items:
            if "integration" not in item.keywords:
                item.add_marker(skip_unit)
    else:
        # デフォルトでは統合テストをスキップ
        skip_integration = pytest.mark.skip(reason="統合テストをスキップ（--integration を使用して実行）")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    
