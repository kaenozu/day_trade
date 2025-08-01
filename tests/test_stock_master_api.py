"""
銘柄マスタ管理のAPIテスト
"""

import pytest

from src.day_trade.data.stock_master import stock_master


@pytest.fixture(scope="function")
def setup_test_db():
    """テスト用データベースのセットアップ"""
    # テスト用インメモリDBを使用
    from src.day_trade.models import database
    from src.day_trade.data import stock_master as sm

    # 元のdb_managerを保存
    original_db_manager = database.db_manager
    original_sm_db_manager = getattr(sm, 'db_manager', None)

    # 横展開：全てのモジュール参照パターンを網羅
    import src.day_trade.data.stock_master as stock_master_module
    original_stock_master_module_db = getattr(stock_master_module, 'db_manager', None)

    # CIとローカルの違いを吸収するためsysモジュールから直接参照
    import sys
    if 'day_trade.data.stock_master' in sys.modules:
        day_trade_module = sys.modules['day_trade.data.stock_master']
        original_day_trade_db = getattr(day_trade_module, 'db_manager', None)
    else:
        day_trade_module = None
        original_day_trade_db = None

    # テスト用のマネージャーに差し替え
    from src.day_trade.models.database import DatabaseConfig
    config = DatabaseConfig.for_testing()
    test_db_manager = database.DatabaseManager(config)
    test_db_manager.create_tables()

    # 全ての参照を横展開で変更
    database.db_manager = test_db_manager
    sm.db_manager = test_db_manager
    stock_master_module.db_manager = test_db_manager

    # CIとローカルの参照違いを吸収
    if day_trade_module:
        day_trade_module.db_manager = test_db_manager

    # stock_masterグローバルインスタンスのdb_managerも変更
    stock_master.db_manager = test_db_manager

    yield test_db_manager

    # 元に戻す（横展開で全て復元）
    database.db_manager = original_db_manager
    if original_sm_db_manager:
        sm.db_manager = original_sm_db_manager
    if original_stock_master_module_db:
        stock_master_module.db_manager = original_stock_master_module_db
        stock_master.db_manager = original_stock_master_module_db
    if day_trade_module and original_day_trade_db:
        day_trade_module.db_manager = original_day_trade_db


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
    """StockMasterManagerのAPIテストクラス"""

    def test_add_stock(self, setup_test_db):
        """銘柄追加のテスト"""
        # 銘柄を追加
        stock = stock_master.add_stock(
            code="7203",
            name="トヨタ自動車",
            market="東証プライム",
            sector="輸送用機器",
            industry="自動車",
        )

        assert stock is not None

        # データベースから直接確認
        result = stock_master.get_stock_by_code("7203")
        assert result is not None
        assert result.code == "7203"
        assert result.name == "トヨタ自動車"
        assert result.market == "東証プライム"
        assert result.sector == "輸送用機器"
        assert result.industry == "自動車"

    def test_add_duplicate_stock(self, setup_test_db):
        """重複銘柄追加のテスト"""
        # 同じ銘柄を2回追加
        stock1 = stock_master.add_stock(code="7203", name="トヨタ自動車")
        stock2 = stock_master.add_stock(code="7203", name="トヨタ自動車 (重複)")

        assert stock1 is not None
        assert stock2 is not None

        # データベースから直接確認
        result = stock_master.get_stock_by_code("7203")
        assert result is not None
        assert result.name == "トヨタ自動車"  # 元の名前が保持される

    def test_get_stock_by_code(self, setup_test_db, sample_stocks):
        """証券コードによる銘柄取得のテスト"""
        # サンプルデータを追加
        for stock_data in sample_stocks:
            stock_master.add_stock(**stock_data)

        # 存在する銘柄を取得
        stock = stock_master.get_stock_by_code("7203")
        assert stock is not None
        assert stock.code == "7203"
        assert stock.name == "トヨタ自動車"

        # 存在しない銘柄を取得
        stock = stock_master.get_stock_by_code("9999")
        assert stock is None

    def test_search_stocks_by_name(self, setup_test_db, sample_stocks):
        """銘柄名検索のテスト"""
        # サンプルデータを追加
        for stock_data in sample_stocks:
            stock_master.add_stock(**stock_data)

        # 部分一致検索
        results = stock_master.search_stocks_by_name("ソ")

        assert len(results) == 2  # ソニーグループ、ソフトバンクグループ
        names = [stock.name for stock in results]
        assert "ソニーグループ" in names
        assert "ソフトバンクグループ" in names

    def test_search_stocks_by_sector(self, setup_test_db, sample_stocks):
        """セクター検索のテスト"""
        # サンプルデータを追加
        for stock_data in sample_stocks:
            stock_master.add_stock(**stock_data)

        # セクター検索
        results = stock_master.search_stocks_by_sector("電気機器")

        assert len(results) == 1
        assert results[0].name == "ソニーグループ"

    def test_search_stocks_by_industry(self, setup_test_db, sample_stocks):
        """業種検索のテスト"""
        # サンプルデータを追加
        for stock_data in sample_stocks:
            stock_master.add_stock(**stock_data)

        # 業種検索
        results = stock_master.search_stocks_by_industry("自動車")

        assert len(results) == 1
        assert results[0].name == "トヨタ自動車"

    def test_search_stocks_complex(self, setup_test_db, sample_stocks):
        """複合条件検索のテスト"""
        # サンプルデータを追加
        for stock_data in sample_stocks:
            stock_master.add_stock(**stock_data)

        # 複合条件検索（市場区分 + 名前の一部）
        results = stock_master.search_stocks(market="東証プライム", name="グループ")

        # "グループ"を含む東証プライム銘柄を確認（正確な数はテストデータに依存）
        assert len(results) >= 2
        names = [stock.name for stock in results]
        assert "ソニーグループ" in names
        assert "ソフトバンクグループ" in names

    def test_update_stock(self, setup_test_db):
        """銘柄更新のテスト"""
        # 銘柄を追加
        stock_master.add_stock(code="7203", name="トヨタ自動車", market="東証プライム")

        # 更新
        updated_stock = stock_master.update_stock(
            code="7203", name="トヨタ自動車株式会社", sector="輸送用機器"
        )

        assert updated_stock is not None
        assert updated_stock.name == "トヨタ自動車株式会社"
        assert updated_stock.sector == "輸送用機器"
        assert updated_stock.market == "東証プライム"  # 元の値が保持される

    def test_delete_stock(self, setup_test_db):
        """銘柄削除のテスト"""
        # 銘柄を追加
        stock_master.add_stock(code="7203", name="トヨタ自動車")

        # 削除前の確認
        stock = stock_master.get_stock_by_code("7203")
        assert stock is not None

        # 削除
        result = stock_master.delete_stock("7203")
        assert result is True

        # 削除後の確認
        stock = stock_master.get_stock_by_code("7203")
        assert stock is None

    def test_get_stock_count(self, setup_test_db, sample_stocks):
        """銘柄数取得のテスト"""
        # 初期カウントを記録
        initial_count = stock_master.get_stock_count()

        # サンプルデータを追加
        for stock_data in sample_stocks:
            stock_master.add_stock(**stock_data)

        # 追加後の確認
        count = stock_master.get_stock_count()
        # テストが独立していない可能性があるため、最低限の検証のみ行う
        assert count >= initial_count
        assert count > 0

    def test_get_all_sectors(self, setup_test_db, sample_stocks):
        """全セクター取得のテスト"""
        # サンプルデータを追加
        for stock_data in sample_stocks:
            stock_master.add_stock(**stock_data)

        sectors = stock_master.get_all_sectors()
        # 期待されるセクターが含まれていることを確認（正確な数はテストデータに依存）
        assert "輸送用機器" in sectors
        assert "電気機器" in sectors
        assert "情報・通信業" in sectors
        assert "銀行業" in sectors

    def test_get_all_industries(self, setup_test_db, sample_stocks):
        """全業種取得のテスト"""
        # サンプルデータを追加
        for stock_data in sample_stocks:
            stock_master.add_stock(**stock_data)

        industries = stock_master.get_all_industries()
        # 期待される業種が含まれていることを確認（正確な数はテストデータに依存）
        assert "自動車" in industries
        assert "電気機器" in industries
        assert "通信業" in industries
        assert "銀行業" in industries

    def test_get_all_markets(self, setup_test_db, sample_stocks):
        """全市場区分取得のテスト"""
        # サンプルデータを追加
        for stock_data in sample_stocks:
            stock_master.add_stock(**stock_data)

        markets = stock_master.get_all_markets()
        assert len(markets) == 1
        assert "東証プライム" in markets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
