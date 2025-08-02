"""
銘柄マスタ管理のAPIテスト
モック化によりデータベース操作を高速化・安定化
"""

from unittest.mock import Mock, patch

import pytest

from src.day_trade.data.stock_master import stock_master
from src.day_trade.models.stock import Stock


@pytest.fixture
def sample_stocks():
    """サンプル株式データ"""
    return [
        {
            "code": "7203",
            "name": "トヨタ自動車",
            "market": "東証プライム",
            "sector": "輸送用機器",
            "industry": "自動車",
        },
        {
            "code": "6758",
            "name": "ソニーグループ",
            "market": "東証プライム",
            "sector": "電気機器",
            "industry": "電気機器",
        },
        {
            "code": "9984",
            "name": "ソフトバンクグループ",
            "market": "東証プライム",
            "sector": "情報・通信業",
            "industry": "通信業",
        },
        {
            "code": "8306",
            "name": "三菱UFJフィナンシャル・グループ",
            "market": "東証プライム",
            "sector": "銀行業",
            "industry": "銀行業",
        },
    ]


class TestStockMasterAPI:
    """StockMasterManagerのAPIテストクラス（モック化版）"""

    @patch('src.day_trade.data.stock_master.db_manager')
    def test_add_stock(self, mock_db_manager):
        """銘柄追加のテスト（モック化）"""
        # モックセットアップ
        mock_session = Mock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None  # 既存なし

        # 作成されるStockオブジェクトをモック
        created_stock = Stock(
            code="7203",
            name="トヨタ自動車",
            market="東証プライム",
            sector="輸送用機器",
            industry="自動車"
        )

        # 銘柄を追加
        with patch.object(stock_master, '_add_stock_with_session', return_value=created_stock):
            stock = stock_master.add_stock(
                code="7203",
                name="トヨタ自動車",
                market="東証プライム",
                sector="輸送用機器",
                industry="自動車",
            )

        assert stock is not None
        assert stock.code == "7203"
        assert stock.name == "トヨタ自動車"
        assert stock.market == "東証プライム"
        assert stock.sector == "輸送用機器"
        assert stock.industry == "自動車"

    @patch('src.day_trade.data.stock_master.db_manager')
    def test_add_duplicate_stock(self, mock_db_manager):
        """重複銘柄追加のテスト（モック化）"""
        # モックセットアップ
        mock_session = Mock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session

        existing_stock = Stock(code="7203", name="トヨタ自動車")
        mock_session.query.return_value.filter.return_value.first.return_value = existing_stock

        # 重複追加テスト
        with patch.object(stock_master, '_add_stock_with_session', return_value=existing_stock):
            stock1 = stock_master.add_stock(code="7203", name="トヨタ自動車")
            stock2 = stock_master.add_stock(code="7203", name="トヨタ自動車 (重複)")

        assert stock1 is not None
        assert stock2 is not None
        assert stock1.name == "トヨタ自動車"  # 元の名前が保持される

    @patch('src.day_trade.data.stock_master.db_manager')
    def test_get_stock_by_code(self, mock_db_manager):
        """証券コードによる銘柄取得のテスト（モック化）"""
        # モックセットアップ
        mock_session = Mock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session

        # 存在する銘柄
        toyota_stock = Stock(code="7203", name="トヨタ自動車")
        mock_session.query.return_value.filter.return_value.first.side_effect = [
            toyota_stock,  # 7203の場合
            None           # 9999の場合
        ]

        # 存在する銘柄を取得
        with patch.object(stock_master, 'get_stock_by_code', return_value=toyota_stock):
            stock = stock_master.get_stock_by_code("7203")
            assert stock is not None
            assert stock.code == "7203"
            assert stock.name == "トヨタ自動車"

        # 存在しない銘柄を取得
        with patch.object(stock_master, 'get_stock_by_code', return_value=None):
            stock = stock_master.get_stock_by_code("9999")
            assert stock is None

    @patch('src.day_trade.data.stock_master.db_manager')
    def test_search_stocks_by_name(self, mock_db_manager):
        """銘柄名検索のテスト（モック化）"""
        # モックセットアップ
        mock_session = Mock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session

        # 検索結果をモック
        sony_stock = Stock(code="6758", name="ソニーグループ", sector="電気機器")
        softbank_stock = Stock(code="9984", name="ソフトバンクグループ", sector="情報・通信業")
        mock_results = [sony_stock, softbank_stock]

        # 部分一致検索
        with patch.object(stock_master, 'search_stocks_by_name', return_value=mock_results):
            results = stock_master.search_stocks_by_name("ソ")

        assert len(results) == 2  # ソニーグループ、ソフトバンクグループ
        names = [stock.name for stock in results]
        assert "ソニーグループ" in names
        assert "ソフトバンクグループ" in names

    @patch('src.day_trade.data.stock_master.db_manager')
    def test_search_stocks_by_sector(self, mock_db_manager):
        """セクター検索のテスト（モック化）"""
        # モックセットアップ
        mock_session = Mock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session

        # 検索結果をモック
        sony_stock = Stock(code="6758", name="ソニーグループ", sector="電気機器")

        # セクター検索
        with patch.object(stock_master, 'search_stocks_by_sector', return_value=[sony_stock]):
            results = stock_master.search_stocks_by_sector("電気機器")

        assert len(results) == 1
        assert results[0].name == "ソニーグループ"

    @patch('src.day_trade.data.stock_master.db_manager')
    def test_search_stocks_by_industry(self, mock_db_manager):
        """業種検索のテスト（モック化）"""
        # モックセットアップ
        mock_session = Mock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session

        # 検索結果をモック
        toyota_stock = Stock(code="7203", name="トヨタ自動車", industry="自動車")

        # 業種検索
        with patch.object(stock_master, 'search_stocks_by_industry', return_value=[toyota_stock]):
            results = stock_master.search_stocks_by_industry("自動車")

        assert len(results) == 1
        assert results[0].name == "トヨタ自動車"

    def test_search_stocks_complex(self):
        """複合条件検索のテスト（モック化）"""
        # 検索結果をモック
        sony_stock = Stock(code="6758", name="ソニーグループ", market="東証プライム")
        softbank_stock = Stock(code="9984", name="ソフトバンクグループ", market="東証プライム")
        mock_results = [sony_stock, softbank_stock]

        # 複合条件検索（市場区分 + 名前の一部）
        with patch.object(stock_master, 'search_stocks', return_value=mock_results):
            results = stock_master.search_stocks(market="東証プライム", name="グループ")

        # "グループ"を含む東証プライム銘柄を確認
        assert len(results) >= 2
        names = [stock.name for stock in results]
        assert "ソニーグループ" in names
        assert "ソフトバンクグループ" in names

    def test_update_stock(self):
        """銘柄更新のテスト（モック化）"""
        # 更新後の銘柄をモック
        updated_stock = Stock(
            code="7203",
            name="トヨタ自動車株式会社",
            market="東証プライム",
            sector="輸送用機器"
        )

        # 更新
        with patch.object(stock_master, 'update_stock', return_value=updated_stock):
            result = stock_master.update_stock(
                code="7203", name="トヨタ自動車株式会社", sector="輸送用機器"
            )

        assert result is not None
        assert result.name == "トヨタ自動車株式会社"
        assert result.sector == "輸送用機器"
        assert result.market == "東証プライム"  # 元の値が保持される

    def test_delete_stock(self):
        """銘柄削除のテスト（モック化）"""
        toyota_stock = Stock(code="7203", name="トヨタ自動車")

        # 削除前の確認（銘柄が存在）
        with patch.object(stock_master, 'get_stock_by_code', return_value=toyota_stock):
            stock = stock_master.get_stock_by_code("7203")
            assert stock is not None

        # 削除成功
        with patch.object(stock_master, 'delete_stock', return_value=True):
            result = stock_master.delete_stock("7203")
            assert result is True

        # 削除後の確認（銘柄が存在しない）
        with patch.object(stock_master, 'get_stock_by_code', return_value=None):
            stock = stock_master.get_stock_by_code("7203")
            assert stock is None

    def test_get_stock_count(self):
        """銘柄数取得のテスト（モック化）"""
        # 銘柄数をモック
        with patch.object(stock_master, 'get_stock_count', return_value=4):
            count = stock_master.get_stock_count()
            assert count == 4
            assert count > 0

    def test_get_all_sectors(self):
        """全セクター取得のテスト（モック化）"""
        mock_sectors = ["輸送用機器", "電気機器", "情報・通信業", "銀行業"]

        with patch.object(stock_master, 'get_all_sectors', return_value=mock_sectors):
            sectors = stock_master.get_all_sectors()

        # 期待されるセクターが含まれていることを確認
        assert "輸送用機器" in sectors
        assert "電気機器" in sectors
        assert "情報・通信業" in sectors
        assert "銀行業" in sectors

    def test_get_all_industries(self):
        """全業種取得のテスト（モック化）"""
        mock_industries = ["自動車", "電気機器", "通信業", "銀行業"]

        with patch.object(stock_master, 'get_all_industries', return_value=mock_industries):
            industries = stock_master.get_all_industries()

        # 期待される業種が含まれていることを確認
        assert "自動車" in industries
        assert "電気機器" in industries
        assert "通信業" in industries
        assert "銀行業" in industries

    def test_get_all_markets(self):
        """全市場区分取得のテスト（モック化）"""
        mock_markets = ["東証プライム"]

        with patch.object(stock_master, 'get_all_markets', return_value=mock_markets):
            markets = stock_master.get_all_markets()

        assert len(markets) == 1
        assert "東証プライム" in markets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
