"""
高度な注文管理システムのテスト
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.day_trade.automation.advanced_order_manager import (
    AdvancedOrderManager,
    Order,
    OrderCondition,
    OrderFill,
    OrderStatus,
    OrderType,
)
from src.day_trade.core.trade_manager import TradeType


class TestAdvancedOrderManager:
    """高度な注文管理システムのテストクラス"""

    @pytest.fixture
    def order_manager(self):
        """注文管理システムインスタンス"""
        return AdvancedOrderManager()

    @pytest.fixture
    def sample_order(self):
        """サンプル注文"""
        return Order(
            symbol="7203",
            order_type=OrderType.LIMIT,
            side=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
        )

    def test_order_manager_initialization(self, order_manager):
        """注文管理システム初期化のテスト"""
        assert len(order_manager.orders) == 0
        assert len(order_manager.active_orders) == 0
        assert len(order_manager.order_history) == 0
        assert len(order_manager.fills) == 0
        assert order_manager.execution_stats["orders_submitted"] == 0

    @pytest.mark.asyncio
    async def test_submit_order_basic(self, order_manager, sample_order):
        """基本的な注文提出のテスト"""
        order_id = await order_manager.submit_order(sample_order)

        assert order_id == sample_order.order_id
        assert order_id in order_manager.orders
        assert order_id in order_manager.active_orders
        assert order_manager.orders[order_id].status == OrderStatus.WORKING
        assert order_manager.execution_stats["orders_submitted"] == 1

    @pytest.mark.asyncio
    async def test_cancel_order(self, order_manager, sample_order):
        """注文キャンセルのテスト"""
        order_id = await order_manager.submit_order(sample_order)

        success = await order_manager.cancel_order(order_id)

        assert success is True
        assert order_manager.orders[order_id].status == OrderStatus.CANCELLED
        assert order_id not in order_manager.active_orders
        assert order_manager.execution_stats["orders_cancelled"] == 1

    @pytest.mark.asyncio
    async def test_market_order_execution(self, order_manager):
        """成行注文実行のテスト"""
        market_order = Order(
            symbol="7203",
            order_type=OrderType.MARKET,
            side=TradeType.BUY,
            quantity=100,
        )

        order_id = await order_manager.submit_order(market_order)
        current_price = Decimal("2500.0")

        # 市場更新処理
        fills = await order_manager.process_market_update("7203", current_price)

        assert len(fills) == 1
        fill = fills[0]
        assert fill.symbol == "7203"
        assert fill.quantity == 100
        assert fill.price == current_price
        assert order_manager.orders[order_id].status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_limit_order_execution(self, order_manager):
        """指値注文実行のテスト"""
        limit_order = Order(
            symbol="7203",
            order_type=OrderType.LIMIT,
            side=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
        )

        await order_manager.submit_order(limit_order)

        # 指値に達しない価格では実行されない
        fills = await order_manager.process_market_update("7203", Decimal("2600.0"))
        assert len(fills) == 0

        # 指値に達する価格で実行される
        fills = await order_manager.process_market_update("7203", Decimal("2450.0"))
        assert len(fills) == 1
        assert fills[0].quantity == 100

    @pytest.mark.asyncio
    async def test_stop_order_execution(self, order_manager):
        """ストップ注文実行のテスト"""
        stop_order = Order(
            symbol="7203",
            order_type=OrderType.STOP,
            side=TradeType.SELL,
            quantity=100,
            stop_price=Decimal("2400.0"),
        )

        await order_manager.submit_order(stop_order)

        # ストップ価格に達しない場合は実行されない
        fills = await order_manager.process_market_update("7203", Decimal("2450.0"))
        assert len(fills) == 0

        # ストップ価格に達すると実行される
        fills = await order_manager.process_market_update("7203", Decimal("2350.0"))
        assert len(fills) == 1
        assert fills[0].side == TradeType.SELL

    def test_order_fill_processing(self):
        """約定処理のテスト"""
        order = Order(
            symbol="7203",
            order_type=OrderType.MARKET,
            side=TradeType.BUY,
            quantity=100,
        )

        fill = OrderFill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol="7203",
            side=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            commission=Decimal("25.0"),
        )

        order.add_fill(fill)

        assert order.filled_quantity == 100
        assert order.remaining_quantity == 0
        assert order.status == OrderStatus.FILLED
        assert order.average_fill_price == Decimal("2500.0")
        assert len(order.fills) == 1

    def test_partial_fill_processing(self):
        """部分約定処理のテスト"""
        order = Order(
            symbol="7203",
            order_type=OrderType.LIMIT,
            side=TradeType.BUY,
            quantity=200,
            price=Decimal("2500.0"),
        )

        # 第1回約定
        fill1 = OrderFill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol="7203",
            side=TradeType.BUY,
            quantity=80,
            price=Decimal("2500.0"),
            commission=Decimal("20.0"),
        )
        order.add_fill(fill1)

        assert order.filled_quantity == 80
        assert order.remaining_quantity == 120
        assert order.status == OrderStatus.PARTIALLY_FILLED

        # 第2回約定
        fill2 = OrderFill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol="7203",
            side=TradeType.BUY,
            quantity=120,
            price=Decimal("2480.0"),
            commission=Decimal("30.0"),
        )
        order.add_fill(fill2)

        assert order.filled_quantity == 200
        assert order.remaining_quantity == 0
        assert order.status == OrderStatus.FILLED

        # 平均約定価格の確認
        expected_avg = (Decimal("2500.0") * 80 + Decimal("2480.0") * 120) / 200
        assert order.average_fill_price == expected_avg

    @pytest.mark.asyncio
    async def test_oco_order(self, order_manager):
        """OCO注文のテスト"""
        primary_order = Order(
            symbol="7203",
            order_type=OrderType.LIMIT,
            side=TradeType.SELL,
            quantity=100,
            price=Decimal("2600.0"),
        )

        secondary_order = Order(
            symbol="7203",
            order_type=OrderType.STOP,
            side=TradeType.SELL,
            quantity=100,
            stop_price=Decimal("2400.0"),
        )

        primary_id, secondary_id = await order_manager.submit_oco_order(
            primary_order, secondary_order
        )

        # 両方の注文が提出されることを確認
        assert primary_id in order_manager.active_orders
        assert secondary_id in order_manager.active_orders

        # プライマリ注文が約定すると、セカンダリ注文がキャンセルされる
        fills = await order_manager.process_market_update("7203", Decimal("2600.0"))

        assert len(fills) == 1  # プライマリのみ約定
        assert order_manager.orders[primary_id].status == OrderStatus.FILLED
        assert order_manager.orders[secondary_id].status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_if_done_order(self, order_manager):
        """IFD注文のテスト"""
        parent_order = Order(
            symbol="7203",
            order_type=OrderType.LIMIT,
            side=TradeType.BUY,
            quantity=100,
            price=Decimal("2450.0"),
        )

        child_order = Order(
            symbol="7203",
            order_type=OrderType.LIMIT,
            side=TradeType.SELL,
            quantity=100,
            price=Decimal("2550.0"),
        )

        parent_id, child_id = await order_manager.submit_if_done_order(
            parent_order, child_order
        )

        # 親注文のみアクティブ、子注文は待機
        assert parent_id in order_manager.active_orders
        assert child_id not in order_manager.active_orders
        assert order_manager.orders[child_id].status == OrderStatus.PENDING

        # 親注文が約定すると子注文が有効化される
        await order_manager.process_market_update("7203", Decimal("2450.0"))

        assert order_manager.orders[parent_id].status == OrderStatus.FILLED
        assert order_manager.orders[child_id].status == OrderStatus.WORKING
        assert child_id in order_manager.active_orders

    def test_order_validation(self, order_manager):
        """注文バリデーションのテスト"""
        # 無効な銘柄
        invalid_order = Order(
            symbol="",
            order_type=OrderType.MARKET,
            side=TradeType.BUY,
            quantity=100,
        )
        assert order_manager._validate_order(invalid_order) is False

        # 無効な数量
        invalid_order.symbol = "7203"
        invalid_order.quantity = -100
        assert order_manager._validate_order(invalid_order) is False

        # 指値価格なし
        invalid_order.quantity = 100
        invalid_order.order_type = OrderType.LIMIT
        invalid_order.price = None
        assert order_manager._validate_order(invalid_order) is False

        # 有効な注文
        invalid_order.price = Decimal("2500.0")
        assert order_manager._validate_order(invalid_order) is True

    def test_get_order_status(self, order_manager, sample_order):
        """注文ステータス取得のテスト"""
        order_manager.orders[sample_order.order_id] = sample_order

        status = order_manager.get_order_status(sample_order.order_id)

        assert status is not None
        assert status["symbol"] == "7203"
        assert status["order_type"] == "limit"
        assert status["side"] == "buy"
        assert status["quantity"] == 100
        assert status["price"] == 2500.0

    def test_get_active_orders(self, order_manager, sample_order):
        """アクティブ注文リスト取得のテスト"""
        sample_order.status = OrderStatus.WORKING
        order_manager.orders[sample_order.order_id] = sample_order
        order_manager.active_orders[sample_order.order_id] = sample_order

        active_orders = order_manager.get_active_orders()

        assert len(active_orders) == 1
        assert active_orders[0]["symbol"] == "7203"

        # シンボルフィルタ
        filtered_orders = order_manager.get_active_orders("7203")
        assert len(filtered_orders) == 1

        filtered_orders = order_manager.get_active_orders("6758")
        assert len(filtered_orders) == 0

    def test_execution_statistics(self, order_manager):
        """実行統計のテスト"""
        order_manager.execution_stats["orders_submitted"] = 10
        order_manager.execution_stats["orders_filled"] = 8
        order_manager.execution_stats["orders_cancelled"] = 2

        stats = order_manager.get_execution_statistics()

        assert stats["orders_submitted"] == 10
        assert stats["orders_filled"] == 8
        assert stats["orders_cancelled"] == 2
        assert stats["fill_rate"] == 80.0  # 8/10 * 100
        assert "active_orders" in stats
        assert "total_orders" in stats

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, order_manager):
        """全注文キャンセルのテスト"""
        # 複数の注文を提出
        orders = []
        for i in range(3):
            order = Order(
                symbol=f"700{i}",
                order_type=OrderType.LIMIT,
                side=TradeType.BUY,
                quantity=100,
                price=Decimal("2500.0"),
            )
            await order_manager.submit_order(order)
            orders.append(order)

        assert len(order_manager.active_orders) == 3

        # 全キャンセル
        cancelled_count = await order_manager.cancel_all_orders()

        assert cancelled_count == 3
        assert len(order_manager.active_orders) == 0

        # 全注文がキャンセル状態
        for order in orders:
            assert order_manager.orders[order.order_id].status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_symbol_specific_cancel(self, order_manager):
        """銘柄別注文キャンセルのテスト"""
        # 異なる銘柄の注文を提出
        order1 = Order(
            symbol="7203", order_type=OrderType.MARKET, side=TradeType.BUY, quantity=100
        )
        order2 = Order(
            symbol="6758", order_type=OrderType.MARKET, side=TradeType.BUY, quantity=100
        )

        await order_manager.submit_order(order1)
        await order_manager.submit_order(order2)

        # 特定銘柄のみキャンセル
        cancelled_count = await order_manager.cancel_all_orders("7203")

        assert cancelled_count == 1
        assert order_manager.orders[order1.order_id].status == OrderStatus.CANCELLED
        assert order_manager.orders[order2.order_id].status == OrderStatus.WORKING


class TestTrailingStop:
    """トレーリングストップのテスト"""

    @pytest.fixture
    def order_manager(self):
        return AdvancedOrderManager()

    @pytest.mark.asyncio
    async def test_trailing_stop_sell(self, order_manager):
        """売りトレーリングストップのテスト"""
        trail_order = Order(
            symbol="7203",
            order_type=OrderType.TRAIL,
            side=TradeType.SELL,
            quantity=100,
            trail_amount=Decimal("100.0"),  # 100円のトレール
        )

        await order_manager.submit_order(trail_order)

        # 価格上昇時にストップ価格が更新される
        await order_manager.process_market_update("7203", Decimal("2600.0"))
        expected_stop = Decimal("2600.0") - Decimal("100.0")
        assert trail_order.stop_price == expected_stop

        # さらに価格上昇
        await order_manager.process_market_update("7203", Decimal("2700.0"))
        expected_stop = Decimal("2700.0") - Decimal("100.0")
        assert trail_order.stop_price == expected_stop

        # 価格下落時はストップ価格は変更されない
        await order_manager.process_market_update("7203", Decimal("2650.0"))
        assert trail_order.stop_price == Decimal("2600.0")  # 変更されない

    @pytest.mark.asyncio
    async def test_trailing_stop_execution(self, order_manager):
        """トレーリングストップ実行のテスト"""
        trail_order = Order(
            symbol="7203",
            order_type=OrderType.TRAIL,
            side=TradeType.SELL,
            quantity=100,
            trail_amount=Decimal("100.0"),
            stop_price=Decimal("2500.0"),  # 初期ストップ価格
        )

        await order_manager.submit_order(trail_order)

        # ストップ価格以下に下落すると実行される
        fills = await order_manager.process_market_update("7203", Decimal("2450.0"))

        assert len(fills) == 1
        assert trail_order.status == OrderStatus.FILLED
        assert fills[0].side == TradeType.SELL


class TestOrderConditions:
    """注文条件のテスト"""

    @pytest.fixture
    def order_manager(self):
        return AdvancedOrderManager()

    def test_order_expiry(self, order_manager):
        """注文期限のテスト"""
        expired_order = Order(
            symbol="7203",
            order_type=OrderType.LIMIT,
            side=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            condition=OrderCondition.GTD,
            expire_time=datetime.now() - timedelta(minutes=1),  # 既に期限切れ
        )

        assert order_manager._check_order_expiry(expired_order) is True
        assert expired_order.status == OrderStatus.EXPIRED

    def test_order_not_expired(self, order_manager):
        """注文未期限のテスト"""
        valid_order = Order(
            symbol="7203",
            order_type=OrderType.LIMIT,
            side=TradeType.BUY,
            quantity=100,
            price=Decimal("2500.0"),
            condition=OrderCondition.GTD,
            expire_time=datetime.now() + timedelta(hours=1),  # まだ有効
        )

        assert order_manager._check_order_expiry(valid_order) is False
        assert valid_order.status != OrderStatus.EXPIRED
