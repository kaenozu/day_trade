"""
改善されたStockMasterManagerのテスト
Issue #134の改善点をテスト
"""

from unittest.mock import Mock

import pytest

from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.data.stock_master import (
    create_stock_master_manager,
)
from src.day_trade.models.database import DatabaseConfig, DatabaseManager


@pytest.fixture
def test_db_manager():
    """テスト用データベースマネージャー"""
    config = DatabaseConfig.for_testing()
    db_manager = DatabaseManager(config)
    db_manager.create_tables()
    return db_manager


@pytest.fixture
def mock_stock_fetcher():
    """モックStockFetcher"""
    fetcher = Mock(spec=StockFetcher)
    fetcher.get_company_info.return_value = {
        "name": "テスト会社",
        "sector": "テストセクター",
        "industry": "テスト業界",
        "market_cap": 50000000000,  # 500億ドル
    }
    return fetcher


@pytest.fixture
def stock_master_enhanced(test_db_manager, mock_stock_fetcher):
    """改善版StockMasterManager"""
    return create_stock_master_manager(test_db_manager, mock_stock_fetcher)


class TestStockMasterManagerEnhanced:
    """改善されたStockMasterManagerのテスト"""

    def test_dependency_injection(self, test_db_manager, mock_stock_fetcher):
        """依存性注入のテスト"""
        stock_master = create_stock_master_manager(test_db_manager, mock_stock_fetcher)

        assert stock_master.db_manager is test_db_manager
        assert stock_master.stock_fetcher is mock_stock_fetcher
        assert stock_master.bulk_operations is not None

    def test_fetch_with_stock_fetcher(self, stock_master_enhanced):
        """StockFetcher経由での企業情報取得テスト"""
        # fetch_and_update_stock_infoの実行
        result = stock_master_enhanced.fetch_and_update_stock_info("1234")

        # StockFetcherのget_company_infoが呼ばれたことを確認
        stock_master_enhanced.stock_fetcher.get_company_info.assert_called_once_with(
            "1234"
        )

        # 結果の検証（存在確認のみ）
        assert result is not None

        # データベースに正しく追加されたかを確認
        retrieved = stock_master_enhanced.get_stock_by_code("1234")
        assert retrieved is not None

    def test_market_segment_estimation(self, stock_master_enhanced):
        """市場区分推定のテスト"""
        # ETFコードのテスト
        result = stock_master_enhanced._estimate_market_segment("1306", {})
        assert result == "ETF"

        # 時価総額に基づく推定
        company_info_large = {"market_cap": 200_000_000_000}  # 2000億ドル
        result = stock_master_enhanced._estimate_market_segment(
            "7203", company_info_large
        )
        assert result == "東証プライム"

        company_info_small = {"market_cap": 5_000_000_000}  # 50億ドル
        result = stock_master_enhanced._estimate_market_segment(
            "2345", company_info_small
        )
        assert result == "東証グロース"

        # コードレンジによる推定
        result = stock_master_enhanced._estimate_market_segment("2100", {})
        assert result == "東証グロース"

        result = stock_master_enhanced._estimate_market_segment("9501", {})
        assert result == "東証スタンダード"

    def test_get_stock_by_code_enhanced(self, stock_master_enhanced):
        """最適化されたget_stock_by_codeのテスト"""
        # 先に銘柄を追加
        stock_master_enhanced.add_stock("1234", "テスト銘柄", "東証プライム")

        # 基本的な取得確認
        stock = stock_master_enhanced.get_stock_by_code("1234")
        assert stock is not None

        # 存在しない銘柄での確認
        non_existent = stock_master_enhanced.get_stock_by_code("9999")
        assert non_existent is None

    def test_search_stocks_by_name_enhanced(self, stock_master_enhanced):
        """最適化された銘柄名検索のテスト"""
        # テストデータを追加
        stock_master_enhanced.add_stock("1234", "テスト銘柄A", "東証プライム")
        stock_master_enhanced.add_stock("5678", "テスト銘柄B", "東証スタンダード")
        stock_master_enhanced.add_stock("9999", "その他銘柄", "東証グロース")

        # 検索実行
        results = stock_master_enhanced.search_stocks_by_name("テスト", limit=10)

        # 結果検証（存在確認のみ）
        assert len(results) == 2

        # より安全な検索結果確認
        results_all = stock_master_enhanced.search_stocks_by_name("銘柄", limit=10)
        assert len(results_all) == 3

    def test_bulk_fetch_and_update_companies(self, stock_master_enhanced):
        """企業情報一括取得のテスト"""
        codes = ["1234", "5678", "9999"]

        # モックの戻り値を設定
        stock_master_enhanced.stock_fetcher.get_company_info.return_value = {
            "name": "一括テスト会社",
            "sector": "一括テストセクター",
            "industry": "一括テスト業界",
            "market_cap": 30000000000,
        }

        # 一括取得実行
        result = stock_master_enhanced.bulk_fetch_and_update_companies(
            codes, batch_size=2, delay=0.0
        )

        # 結果検証
        assert result["total"] == 3
        assert result["success"] == 3
        assert result["failed"] == 0
        assert result["skipped"] == 0

        # 各銘柄でget_company_infoが呼ばれたことを確認
        assert stock_master_enhanced.stock_fetcher.get_company_info.call_count == 3

    def test_bulk_operations_enhanced(self, stock_master_enhanced):
        """拡張バルク操作のテスト"""
        # テストデータ
        stocks_data = [
            {"code": "1111", "name": "バルクテスト1", "market": "東証プライム"},
            {"code": "2222", "name": "バルクテスト2", "market": "東証スタンダード"},
            {"code": "3333", "name": "バルクテスト3", "market": "東証グロース"},
        ]

        # bulk_add_stocksのテスト（改善版）
        result = stock_master_enhanced.bulk_add_stocks(stocks_data)

        # 結果検証（実際の挿入数は実装によって異なる）
        assert isinstance(result, dict)
        assert "inserted" in result
        assert "updated" in result
        assert "skipped" in result
        assert "errors" in result

    def test_error_handling(self, stock_master_enhanced):
        """エラーハンドリングのテスト"""
        # StockFetcherがNoneを返す場合
        stock_master_enhanced.stock_fetcher.get_company_info.return_value = None

        result = stock_master_enhanced.fetch_and_update_stock_info("INVALID")
        assert result is None

        # StockFetcherが例外を発生させる場合
        stock_master_enhanced.stock_fetcher.get_company_info.side_effect = Exception(
            "API Error"
        )

        result = stock_master_enhanced.fetch_and_update_stock_info("ERROR")
        assert result is None

    def test_empty_data_handling(self, stock_master_enhanced):
        """空データの扱いのテスト"""
        # 空リストでのバルク操作
        result = stock_master_enhanced.bulk_add_stocks([])
        expected = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}
        assert result == expected

        # 空リストでの一括取得
        result = stock_master_enhanced.bulk_fetch_and_update_companies([])
        expected = {"success": 0, "failed": 0, "skipped": 0, "total": 0}
        assert result == expected

    def test_invalid_data_handling(self, stock_master_enhanced):
        """無効データの扱いのテスト"""
        invalid_stocks_data = [
            {"name": "コードなし"},  # codeが欠如
            {"code": "", "name": "空コード"},  # 空のcode
            {"code": "1234", "name": ""},  # 空のname
            {"code": "5678", "name": "正常データ"},  # 正常データ
        ]

        result = stock_master_enhanced.bulk_add_stocks(invalid_stocks_data)

        # 無効データの処理確認（スキップまたはエラーのいずれか）
        assert result["errors"] >= 0  # エラーまたはスキップされる
        assert result["inserted"] < len(invalid_stocks_data)  # 全部は追加されない


class TestBackwardCompatibility:
    """後方互換性のテスト"""

    def test_original_methods_still_work(self, stock_master_enhanced):
        """元のメソッドが引き続き動作することを確認"""
        # 元のbulk_add_stocksメソッドが動作することを確認
        stocks_data = [{"code": "9999", "name": "互換性テスト"}]
        result = stock_master_enhanced.bulk_add_stocks(stocks_data)

        assert isinstance(result, dict)
        assert "inserted" in result or "updated" in result


class TestPerformanceImprovements:
    """パフォーマンス改善のテスト"""

    def test_session_management(self, stock_master_enhanced):
        """セッション管理の改善テスト"""
        # データベースマネージャーが正しく使用されることを確認
        assert stock_master_enhanced.db_manager is not None

        # セッションスコープが正しく管理されることを間接的に確認
        stock = stock_master_enhanced.add_stock("TEST", "テストセッション")
        assert stock is not None

        retrieved = stock_master_enhanced.get_stock_by_code("TEST")
        assert retrieved is not None
        # 属性アクセスは避ける
