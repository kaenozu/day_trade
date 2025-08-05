"""
models/stock.pyのテスト
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.day_trade.models.base import BaseModel
from src.day_trade.models.enums import AlertType, TradeType
from src.day_trade.models.stock import Alert, PriceData, Stock, Trade, WatchlistItem


@pytest.fixture
def session():
    """テスト用のデータベースセッション"""
    engine = create_engine("sqlite:///:memory:")
    BaseModel.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_stock(session):
    """テスト用の銘柄データ"""
    stock = Stock(
        code="7203",
        name="トヨタ自動車",
        market="東証プライム",
        sector="輸送用機器",
        industry="自動車"
    )
    session.add(stock)
    session.commit()
    return stock


@pytest.fixture
def sample_price_data(session, sample_stock):
    """テスト用の価格データ"""
    price_data = []
    base_date = datetime(2024, 1, 1)

    for i in range(10):
        data = PriceData(
            stock_code=sample_stock.code,
            datetime=base_date + timedelta(days=i),
            open=Decimal("1000.00") + i,
            high=Decimal("1010.00") + i,
            low=Decimal("990.00") + i,
            close=Decimal("1005.00") + i,
            volume=100000 + i * 1000
        )
        price_data.append(data)
        session.add(data)

    session.commit()
    return price_data


class TestStock:
    """Stockモデルのテスト"""

    def test_stock_creation(self, session):
        """銘柄作成のテスト"""
        stock = Stock(
            code="7203",
            name="トヨタ自動車",
            market="東証プライム",
            sector="輸送用機器",
            industry="自動車"
        )
        session.add(stock)
        session.commit()

        assert stock.id is not None
        assert stock.code == "7203"
        assert stock.name == "トヨタ自動車"
        assert stock.market == "東証プライム"
        assert stock.sector == "輸送用機器"
        assert stock.industry == "自動車"

    def test_get_by_sector(self, session, sample_stock):
        """セクター別取得のテスト"""
        # 別セクターの銘柄を追加
        stock2 = Stock(
            code="8306",
            name="三菱UFJフィナンシャル・グループ",
            market="東証プライム",
            sector="銀行業",
            industry="銀行"
        )
        session.add(stock2)
        session.commit()

        # セクター別で取得
        transport_stocks = Stock.get_by_sector(session, "輸送用機器")
        banking_stocks = Stock.get_by_sector(session, "銀行業")

        assert len(transport_stocks) == 1
        assert transport_stocks[0].code == "7203"
        assert len(banking_stocks) == 1
        assert banking_stocks[0].code == "8306"

    def test_search_by_name_or_code(self, session, sample_stock):
        """名前・コード検索のテスト"""
        # コードで検索
        result = Stock.search_by_name_or_code(session, "7203")
        assert len(result) == 1
        assert result[0].code == "7203"

        # 名前で検索
        result = Stock.search_by_name_or_code(session, "トヨタ")
        assert len(result) == 1
        assert result[0].name == "トヨタ自動車"

        # 部分一致で検索
        result = Stock.search_by_name_or_code(session, "トヨ")
        assert len(result) == 1

        # 該当なしの検索
        result = Stock.search_by_name_or_code(session, "存在しない")
        assert len(result) == 0

    def test_search_with_limit(self, session):
        """検索結果の件数制限テスト"""
        # 複数の銘柄を作成（名前に共通文字列を含む）
        for i in range(15):
            stock = Stock(
                code=f"000{i:02d}",
                name=f"テスト銘柄{i}",
                market="東証プライム"
            )
            session.add(stock)
        session.commit()

        # デフォルトの制限（50件）で検索
        result = Stock.search_by_name_or_code(session, "テスト")
        assert len(result) == 15

        # 制限を指定して検索
        result = Stock.search_by_name_or_code(session, "テスト", limit=5)
        assert len(result) == 5


class TestPriceData:
    """PriceDataモデルのテスト"""

    def test_price_data_creation(self, session, sample_stock):
        """価格データ作成のテスト"""
        price_data = PriceData(
            stock_code=sample_stock.code,
            datetime=datetime(2024, 1, 1),
            open=Decimal("1000.00"),
            high=Decimal("1010.00"),
            low=Decimal("990.00"),
            close=Decimal("1005.00"),
            volume=100000
        )
        session.add(price_data)
        session.commit()

        assert price_data.id is not None
        assert price_data.stock_code == "7203"
        assert price_data.open == Decimal("1000.00")
        assert price_data.close == Decimal("1005.00")
        assert price_data.volume == 100000

    def test_get_latest_prices(self, session, sample_price_data):
        """最新価格取得のテスト"""
        # 追加の銘柄と価格データ
        stock2 = Stock(code="8306", name="MUFG")
        session.add(stock2)

        price2 = PriceData(
            stock_code="8306",
            datetime=datetime(2024, 1, 5),
            close=Decimal("800.00"),
            volume=50000
        )
        session.add(price2)
        session.commit()

        # 最新価格を取得
        latest_prices = PriceData.get_latest_prices(session, ["7203", "8306"])

        assert len(latest_prices) == 2
        assert "7203" in latest_prices
        assert "8306" in latest_prices
        assert latest_prices["7203"].close == Decimal("1014.00")  # 最新（9日目のデータ）
        assert latest_prices["8306"].close == Decimal("800.00")

    def test_get_price_range(self, session, sample_price_data):
        """期間指定価格データ取得のテスト"""
        start_date = datetime(2024, 1, 3)
        end_date = datetime(2024, 1, 7)

        price_range = PriceData.get_price_range(session, "7203", start_date, end_date)

        assert len(price_range) == 5  # 3日〜7日の5日分
        # 時系列順にソートされているかチェック
        for i in range(1, len(price_range)):
            assert price_range[i].datetime > price_range[i-1].datetime

    def test_get_volume_spike_candidates(self, session, sample_price_data):
        """出来高急増検出のテスト"""
        # 現在日時を基準にして、最近のデータとして出来高急増データを追加
        base_date = datetime.now()
        spike_data = PriceData(
            stock_code="7203",
            datetime=base_date,
            close=Decimal("1020.00"),
            volume=500000  # 通常の5倍の出来高
        )
        session.add(spike_data)

        # 平均計算用の最近のデータも追加（基準日の2-5日前）
        for i in range(2, 6):
            normal_data = PriceData(
                stock_code="7203",
                datetime=base_date - timedelta(days=i),
                close=Decimal("1000.00"),
                volume=100000  # 通常の出来高
            )
            session.add(normal_data)

        session.commit()

        # 出来高急増銘柄を検索
        spike_candidates = PriceData.get_volume_spike_candidates(
            session,
            volume_threshold=2.0,
            days_back=30,
            reference_date=base_date
        )

        assert len(spike_candidates) >= 1
        assert spike_candidates[0].stock_code == "7203"
        assert spike_candidates[0].volume == 500000

    def test_get_volume_spike_with_reference_date(self, session, sample_price_data):
        """基準日指定での出来高急増検出テスト"""
        reference_date = datetime(2024, 1, 15)

        spike_candidates = PriceData.get_volume_spike_candidates(
            session,
            volume_threshold=2.0,
            days_back=20,
            reference_date=reference_date
        )

        # 基準日以前のデータのみが対象になることを確認
        # サンプルデータは2024年1月1-10日なので、該当なし
        assert len(spike_candidates) == 0


class TestTrade:
    """Tradeモデルのテスト"""

    def test_trade_creation(self, session, sample_stock):
        """取引データ作成のテスト"""
        trade = Trade(
            stock_code=sample_stock.code,
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("1000.00"),
            commission=Decimal("500.00"),
            trade_datetime=datetime.now()
        )
        session.add(trade)
        session.commit()

        assert trade.id is not None
        assert trade.stock_code == "7203"
        assert trade.trade_type == TradeType.BUY
        assert trade.quantity == 100
        assert trade.price == Decimal("1000.00")

    def test_total_amount_buy(self, session, sample_stock):
        """買い取引の総額計算テスト"""
        trade = Trade(
            stock_code=sample_stock.code,
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("1000.00"),
            commission=Decimal("500.00"),
            trade_datetime=datetime.now()
        )

        expected_total = Decimal("1000.00") * 100 + Decimal("500.00")  # 100,500
        assert trade.total_amount == expected_total

    def test_total_amount_sell(self, session, sample_stock):
        """売り取引の総額計算テスト"""
        trade = Trade(
            stock_code=sample_stock.code,
            trade_type=TradeType.SELL,
            quantity=100,
            price=Decimal("1000.00"),
            commission=Decimal("500.00"),
            trade_datetime=datetime.now()
        )

        expected_total = Decimal("1000.00") * 100 - Decimal("500.00")  # 99,500
        assert trade.total_amount == expected_total

    def test_total_amount_no_commission(self, session, sample_stock):
        """手数料なしの総額計算テスト"""
        trade = Trade(
            stock_code=sample_stock.code,
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("1000.00"),
            commission=None,
            trade_datetime=datetime.now()
        )

        expected_total = Decimal("1000.00") * 100  # 100,000
        assert trade.total_amount == expected_total

    def test_total_amount_zero_values(self, session, sample_stock):
        """ゼロ値での総額計算テスト"""
        trade = Trade(
            stock_code=sample_stock.code,
            trade_type=TradeType.BUY,
            quantity=0,
            price=None,
            trade_datetime=datetime.now()
        )

        assert trade.total_amount == Decimal('0')

    def test_create_buy_trade(self, session, sample_stock):
        """買い取引作成ヘルパーのテスト"""
        trade = Trade.create_buy_trade(
            session=session,
            stock_code=sample_stock.code,
            quantity=100,
            price=1000.0,
            commission=500.0,
            memo="テスト買い"
        )

        assert trade.trade_type == TradeType.BUY
        assert trade.quantity == 100
        assert trade.price == 1000.0
        assert trade.commission == 500.0
        assert trade.memo == "テスト買い"
        assert trade.id is not None  # flushされてIDが設定される

    def test_create_sell_trade(self, session, sample_stock):
        """売り取引作成ヘルパーのテスト"""
        trade = Trade.create_sell_trade(
            session=session,
            stock_code=sample_stock.code,
            quantity=100,
            price=1100.0,
            commission=500.0,
            memo="テスト売り"
        )

        assert trade.trade_type == TradeType.SELL
        assert trade.quantity == 100
        assert trade.price == 1100.0
        assert trade.memo == "テスト売り"

    def test_get_recent_trades(self, session, sample_stock):
        """最近の取引履歴取得のテスト"""
        # 新しい取引を作成
        recent_trade = Trade.create_buy_trade(
            session, sample_stock.code, 100, 1000.0
        )

        # 古い取引を作成
        old_trade = Trade(
            stock_code=sample_stock.code,
            trade_type=TradeType.SELL,
            quantity=50,
            price=Decimal("900.00"),
            trade_datetime=datetime.now() - timedelta(days=60)
        )
        session.add(old_trade)
        session.commit()

        # 最近30日の取引を取得
        recent_trades = Trade.get_recent_trades(session, days=30)

        assert len(recent_trades) == 1
        assert recent_trades[0].id == recent_trade.id

    def test_get_portfolio_summary(self, session, sample_stock):
        """ポートフォリオサマリー計算のテスト"""
        # 買い取引
        Trade.create_buy_trade(session, sample_stock.code, 100, 1000.0, 500.0)
        Trade.create_buy_trade(session, sample_stock.code, 50, 1200.0, 300.0)

        # 売り取引
        Trade.create_sell_trade(session, sample_stock.code, 30, 1100.0, 200.0)

        session.commit()

        summary = Trade.get_portfolio_summary(session)

        assert "7203" in summary["portfolio"]
        portfolio_data = summary["portfolio"]["7203"]

        # 保有数量: 100 + 50 - 30 = 120
        assert portfolio_data["quantity"] == 120

        # 総コスト: (100 * 1000 + 500) + (50 * 1200 + 300)
        expected_cost = Decimal("100500") + Decimal("60300")
        assert portfolio_data["total_cost"] == expected_cost

        # 売却収益: 30 * 1100 - 200
        expected_proceeds = Decimal("30") * Decimal("1100") - Decimal("200")
        assert summary["total_proceeds"] == expected_proceeds

    def test_get_portfolio_summary_with_date_filter(self, session, sample_stock):
        """日付フィルタ付きポートフォリオサマリーのテスト"""
        # 古い取引
        old_trade = Trade(
            stock_code=sample_stock.code,
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("800.00"),
            trade_datetime=datetime.now() - timedelta(days=60)
        )
        session.add(old_trade)

        # 新しい取引
        Trade.create_buy_trade(session, sample_stock.code, 50, 1000.0)
        session.commit()

        # 30日以内の取引のみでサマリー作成
        start_date = datetime.now() - timedelta(days=30)
        summary = Trade.get_portfolio_summary(session, start_date)

        assert summary["portfolio"]["7203"]["quantity"] == 50  # 新しい取引のみ


class TestWatchlistItem:
    """WatchlistItemモデルのテスト"""

    def test_watchlist_item_creation(self, session, sample_stock):
        """ウォッチリストアイテム作成のテスト"""
        item = WatchlistItem(
            stock_code=sample_stock.code,
            group_name="お気に入り",
            memo="良い銘柄"
        )
        session.add(item)
        session.commit()

        assert item.id is not None
        assert item.stock_code == "7203"
        assert item.group_name == "お気に入り"
        assert item.memo == "良い銘柄"

    def test_watchlist_item_default_group(self, session, sample_stock):
        """デフォルトグループのテスト"""
        item = WatchlistItem(stock_code=sample_stock.code)
        session.add(item)
        session.commit()

        assert item.group_name == "default"


class TestAlert:
    """Alertモデルのテスト"""

    def test_alert_creation(self, session, sample_stock):
        """アラート作成のテスト"""
        alert = Alert(
            stock_code=sample_stock.code,
            alert_type=AlertType.PRICE_ABOVE,
            threshold=Decimal("1500.000"),
            memo="高値アラート"
        )
        session.add(alert)
        session.commit()

        assert alert.id is not None
        assert alert.stock_code == "7203"
        assert alert.alert_type == AlertType.PRICE_ABOVE
        assert alert.threshold == Decimal("1500.000")
        assert alert.is_active is True  # デフォルト値
        assert alert.memo == "高値アラート"

    def test_alert_default_active(self, session, sample_stock):
        """アラートのデフォルト有効状態テスト"""
        alert = Alert(
            stock_code=sample_stock.code,
            alert_type=AlertType.PRICE_BELOW,
            threshold=Decimal("800.000")
        )
        session.add(alert)
        session.commit()

        assert alert.is_active is True


class TestRelationships:
    """リレーションシップのテスト"""

    def test_stock_price_data_relationship(self, session, sample_stock, sample_price_data):
        """銘柄-価格データのリレーションシップテスト"""
        # 銘柄から価格データにアクセス
        assert len(sample_stock.price_data) == 10
        assert all(pd.stock_code == sample_stock.code for pd in sample_stock.price_data)

        # 価格データから銘柄にアクセス
        price = sample_price_data[0]
        assert price.stock.code == sample_stock.code
        assert price.stock.name == "トヨタ自動車"

    def test_stock_trade_relationship(self, session, sample_stock):
        """銘柄-取引のリレーションシップテスト"""
        trade = Trade.create_buy_trade(session, sample_stock.code, 100, 1000.0)
        session.commit()

        # 銘柄から取引にアクセス
        assert len(sample_stock.trades) == 1
        assert sample_stock.trades[0].id == trade.id

        # 取引から銘柄にアクセス
        assert trade.stock.code == sample_stock.code

    def test_stock_watchlist_relationship(self, session, sample_stock):
        """銘柄-ウォッチリストのリレーションシップテスト"""
        item = WatchlistItem(stock_code=sample_stock.code, memo="テスト")
        session.add(item)
        session.commit()

        # 銘柄からウォッチリストアイテムにアクセス
        assert len(sample_stock.watchlist_items) == 1
        assert sample_stock.watchlist_items[0].memo == "テスト"

        # ウォッチリストアイテムから銘柄にアクセス
        assert item.stock.code == sample_stock.code

    def test_stock_alert_relationship(self, session, sample_stock):
        """銘柄-アラートのリレーションシップテスト"""
        alert = Alert(
            stock_code=sample_stock.code,
            alert_type=AlertType.PRICE_ABOVE,
            threshold=Decimal("1500.000")
        )
        session.add(alert)
        session.commit()

        # 銘柄からアラートにアクセス
        assert len(sample_stock.alerts) == 1
        assert sample_stock.alerts[0].threshold == Decimal("1500.000")

        # アラートから銘柄にアクセス
        assert alert.stock.code == sample_stock.code


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_search_query(self, session, sample_stock):
        """空の検索クエリのテスト"""
        result = Stock.search_by_name_or_code(session, "")
        # 空文字は全件にマッチするため、1件以上返る
        assert len(result) >= 1

    def test_get_latest_prices_empty_list(self, session):
        """空の銘柄リストでの最新価格取得テスト"""
        result = PriceData.get_latest_prices(session, [])
        assert result == {}

    def test_get_latest_prices_nonexistent_stocks(self, session):
        """存在しない銘柄での最新価格取得テスト"""
        result = PriceData.get_latest_prices(session, ["9999"])
        assert result == {}

    def test_volume_spike_no_data(self, session):
        """データなしでの出来高急増検出テスト"""
        result = PriceData.get_volume_spike_candidates(session)
        assert result == []

    def test_portfolio_summary_no_trades(self, session):
        """取引なしでのポートフォリオサマリーテスト"""
        result = Trade.get_portfolio_summary(session)
        assert result["portfolio"] == {}
        assert result["total_cost"] == 0
        assert result["total_proceeds"] == 0
        assert result["net_position"] == 0
