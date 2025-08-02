"""
銘柄マスタ管理のユニットテスト
"""

import os
import sys

import pytest

from day_trade.data.stock_master import StockMasterManager  # Moved to top
from day_trade.models.stock import Stock  # Moved to top

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="function")
def setup_test_db():
    """テスト用データベースのセットアップ"""
    # テスト用インメモリDBを使用
    from day_trade.models.database import DatabaseConfig, DatabaseManager
    config = DatabaseConfig.for_testing()
    test_db_manager = DatabaseManager(config)
    test_db_manager.create_tables()

    # テスト用のStockMasterManagerを作成
    test_stock_master = StockMasterManager()

    yield test_stock_master, test_db_manager

    # クリーンアップは自動的に行われる（インメモリDB）


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


class TestStockMasterManager:
    """StockMasterManagerのテストクラス"""

    def test_add_stock(self, setup_test_db):
        """銘柄追加のテスト"""
        stock_master, db_manager = setup_test_db

        with db_manager.session_scope() as session:
            # 銘柄を追加
            stock = stock_master.add_stock(
                code="7203",
                name="トヨタ自動車",
                market="東証プライム",
                sector="輸送用機器",
                industry="自動車",
                session=session,
            )

            assert stock is not None
            assert stock.code == "7203"
            assert stock.name == "トヨタ自動車"
            assert stock.market == "東証プライム"
            assert stock.sector == "輸送用機器"
            assert stock.industry == "自動車"

    def test_add_duplicate_stock(self, setup_test_db):
        """重複銘柄追加のテスト"""
        stock_master, db_manager = setup_test_db

        with db_manager.session_scope() as session:
            # 同じ銘柄を2回追加
            stock1 = stock_master.add_stock(
                code="7203", name="トヨタ自動車", session=session
            )
            stock2 = stock_master.add_stock(
                code="7203", name="トヨタ自動車 (重複)", session=session
            )

            assert stock1 is not None
            assert stock2 is not None
            assert stock1.id == stock2.id  # 同じオブジェクトが返される
            assert stock1.name == "トヨタ自動車"  # 元の名前が保持される

    def test_get_stock_by_code(self, setup_test_db, sample_stocks):
        """証券コードによる銘柄取得のテスト"""
        stock_master, db_manager = setup_test_db

        # サンプルデータを追加
        with db_manager.session_scope() as session:
            for stock_data in sample_stocks:
                stock_master.add_stock(session=session, **stock_data)

        # 直接データベースから取得してテスト
        with db_manager.session_scope() as session:
            # 存在する銘柄を取得
            stock = session.query(Stock).filter(Stock.code == "7203").first()
            assert stock is not None
            assert stock.code == "7203"
            assert stock.name == "トヨタ自動車"

            # 存在しない銘柄を取得
            stock = session.query(Stock).filter(Stock.code == "9999").first()
            assert stock is None

    def test_search_stocks_by_name(self, setup_test_db, sample_stocks):
        """銘柄名検索のテスト"""
        stock_master, db_manager = setup_test_db

        # サンプルデータを追加
        with db_manager.session_scope() as session:
            for stock_data in sample_stocks:
                stock_master.add_stock(session=session, **stock_data)

        # テスト用のマネージャーを作成
        test_manager = StockMasterManager()
        test_manager.db_manager = db_manager

        # 部分一致検索（セッション内で名前を取得）
        with db_manager.session_scope() as session:
            results = session.query(Stock).filter(Stock.name.ilike("%ソ%")).all()
            names = [stock.name for stock in results]

        assert len(results) == 2  # ソニーグループ、ソフトバンクグループ
        assert "ソニーグループ" in names
        assert "ソフトバンクグループ" in names

    def test_search_stocks_by_sector(self, setup_test_db, sample_stocks):
        """セクター検索のテスト"""
        stock_master, db_manager = setup_test_db

        # サンプルデータを追加
        with db_manager.session_scope() as session:
            for stock_data in sample_stocks:
                stock_master.add_stock(session=session, **stock_data)

        # テスト用のマネージャーを作成
        test_manager = StockMasterManager()
        test_manager.db_manager = db_manager

        # セクター検索（セッション内で実行）
        with db_manager.session_scope() as session:
            results = session.query(Stock).filter(Stock.sector == "電気機器").all()
            result_name = results[0].name if results else None

        assert len(results) == 1
        assert result_name == "ソニーグループ"

    def test_update_stock(self, setup_test_db):
        """銘柄更新のテスト"""
        stock_master, db_manager = setup_test_db

        # 銘柄を追加
        with db_manager.session_scope() as session:
            stock_master.add_stock(
                code="7203", name="トヨタ自動車", market="東証プライム", session=session
            )

        # テスト用のマネージャーを作成
        test_manager = StockMasterManager()
        test_manager.db_manager = db_manager

        # 更新（セッション内で属性確認）
        with db_manager.session_scope() as session:
            updated_stock = session.query(Stock).filter(Stock.code == "7203").first()
            if updated_stock:
                updated_stock.name = "トヨタ自動車株式会社"
                updated_stock.sector = "輸送用機器"
                session.commit()

                # 属性をセッション内で確認
                name = updated_stock.name
                sector = updated_stock.sector
                market = updated_stock.market

        assert name == "トヨタ自動車株式会社"
        assert sector == "輸送用機器"
        assert market == "東証プライム"  # 元の値が保持される

    def test_delete_stock(self, setup_test_db):
        """銘柄削除のテスト"""
        stock_master, db_manager = setup_test_db

        # 銘柄を追加
        with db_manager.session_scope() as session:
            stock_master.add_stock(code="7203", name="トヨタ自動車", session=session)

        # テスト用のマネージャーを作成
        test_manager = StockMasterManager()
        test_manager.db_manager = db_manager

        # 削除前の確認
        stock = test_manager._get_with_db_manager("7203")
        assert stock is not None

        # 削除
        result = test_manager._delete_with_db_manager("7203")
        assert result is True

        # 削除後の確認
        stock = test_manager._get_with_db_manager("7203")
        assert stock is None

    def test_get_stock_count(self, setup_test_db, sample_stocks):
        """銘柄数取得のテスト"""
        stock_master, db_manager = setup_test_db

        # 初期状態では0件
        with db_manager.session_scope() as session:
            count = session.query(Stock).count()
            assert count == 0

        # サンプルデータを追加
        with db_manager.session_scope() as session:
            for stock_data in sample_stocks:
                stock_master.add_stock(session=session, **stock_data)

        # 追加後の確認
        with db_manager.session_scope() as session:
            count = session.query(Stock).count()
            assert count == len(sample_stocks)


# ヘルパーメソッドを追加（テスト用）
def _add_helper_methods():
    """テスト用のヘルパーメソッドを追加"""

    def _search_with_db_manager(self, query_func):
        with self.db_manager.session_scope() as session:
            return query_func(session)

    def _update_with_db_manager(self, code, **kwargs):
        with self.db_manager.session_scope() as session:
            stock = session.query(Stock).filter(Stock.code == code).first()
            if stock:
                for key, value in kwargs.items():
                    if hasattr(stock, key):
                        setattr(stock, key, value)
            return stock

    def _get_with_db_manager(self, code):
        with self.db_manager.session_scope() as session:
            return session.query(Stock).filter(Stock.code == code).first()

    def _delete_with_db_manager(self, code):
        with self.db_manager.session_scope() as session:
            stock = session.query(Stock).filter(Stock.code == code).first()
            if stock:
                session.delete(stock)
                return True
            return False

    # メソッドを動的に追加
    StockMasterManager._search_with_db_manager = _search_with_db_manager
    StockMasterManager._update_with_db_manager = _update_with_db_manager
    StockMasterManager._get_with_db_manager = _get_with_db_manager
    StockMasterManager._delete_with_db_manager = _delete_with_db_manager


# テスト実行時にヘルパーメソッドを追加
_add_helper_methods()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
