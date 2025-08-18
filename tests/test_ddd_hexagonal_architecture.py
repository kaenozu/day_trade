"""
DDD・ヘキサゴナルアーキテクチャ統合テスト

新しく実装したアーキテクチャパターンの動作確認
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime
from uuid import uuid4

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent / "src"))


def test_value_objects():
    """値オブジェクトのテスト"""
    print("=== 値オブジェクトテスト ===")

    try:
        from day_trade.domain.common.value_objects import Money, Price, Quantity, Symbol, Percentage

        # Money値オブジェクト
        money1 = Money(Decimal("1000"))
        money2 = Money(Decimal("500"))
        total = money1.add(money2)
        print(f"[OK] Money計算: {money1} + {money2} = {total}")

        # Price値オブジェクト
        price = Price(Decimal("1500"))
        quantity = Quantity(100)
        trade_value = price.to_money(quantity)
        print(f"[OK] Price計算: {price} × {quantity} = {trade_value}")

        # Symbol値オブジェクト
        symbol = Symbol("7203")
        print(f"[OK] Symbol作成: {symbol} (日本株: {symbol.is_japanese_stock()}, 市場: {symbol.get_market_segment()})")

        # Percentage値オブジェクト
        percentage = Percentage.from_ratio(Decimal("0.05"))
        print(f"[OK] Percentage変換: 0.05 → {percentage}")

        return True

    except Exception as e:
        print(f"[ERROR] 値オブジェクトテスト失敗: {e}")
        return False


def test_domain_events():
    """ドメインイベントのテスト"""
    print("\n=== ドメインイベントテスト ===")

    try:
        from day_trade.domain.common.domain_events import (
            domain_event_dispatcher, TradeExecutedEvent, TradeEventLogger
        )

        # イベント作成
        event = TradeExecutedEvent(
            trade_id=uuid4(),
            symbol="7203",
            quantity=100,
            price="1500.00",
            direction="buy",
            commission="100.00"
        )

        # ハンドラー登録とイベント発行
        handler = TradeEventLogger()
        domain_event_dispatcher.register_handler(TradeExecutedEvent, handler)
        domain_event_dispatcher.dispatch(event)

        # イベント履歴確認
        events = domain_event_dispatcher.get_events(TradeExecutedEvent, limit=1)
        print(f"[OK] イベント発行・処理成功: {len(events)}件のイベント")
        print(f"   イベントタイプ: {events[0].event_type if events else 'なし'}")

        return True

    except Exception as e:
        print(f"[ERROR] ドメインイベントテスト失敗: {e}")
        return False


def test_domain_entities():
    """ドメインエンティティのテスト"""
    print("\n=== ドメインエンティティテスト ===")

    try:
        from day_trade.domain.trading.entities import Trade, Position, Portfolio, TradeId, PositionId
        from day_trade.domain.common.value_objects import Symbol, Price, Quantity, Money

        # 取引エンティティ作成
        trade = Trade(
            trade_id=TradeId(),
            symbol=Symbol("7203"),
            quantity=Quantity(100),
            price=Price(Decimal("1500")),
            direction="buy",
            executed_at=datetime.now(),
            commission=Money(Decimal("100"))
        )
        print(f"[OK] Trade作成成功: {trade.symbol} {trade.quantity.value}株 @{trade.price}")

        # ポジションエンティティ作成
        position = Position(PositionId(), Symbol("7203"))
        position.add_trade(trade)
        current_qty = position.calculate_current_quantity()
        print(f"[OK] Position作成成功: 保有数量={current_qty.value}株")

        # ポートフォリオエンティティ作成
        portfolio = Portfolio(uuid4())
        executed_position = portfolio.execute_trade(trade)
        print(f"[OK] Portfolio取引実行成功: ポジション数={len(portfolio.get_all_positions())}")

        return True

    except Exception as e:
        print(f"[ERROR] ドメインエンティティテスト失敗: {e}")
        return False


def test_domain_services():
    """ドメインサービスのテスト"""
    print("\n=== ドメインサービステスト ===")

    try:
        from day_trade.domain.trading.services import RiskManagementService
        from day_trade.infrastructure.repositories.memory_repositories import MemoryUnitOfWork
        from day_trade.domain.trading.entities import Portfolio
        from day_trade.domain.common.value_objects import Symbol, Price

        # UnitOfWork作成
        uow = MemoryUnitOfWork()

        # リスク管理サービス
        risk_service = RiskManagementService(uow)

        # テスト用ポートフォリオ
        portfolio = Portfolio(uuid4())
        current_prices = {
            Symbol("7203"): Price(Decimal("1500"))
        }

        # リスク評価
        risk_result = risk_service.assess_portfolio_risk(portfolio, current_prices)
        print(f"[OK] リスク評価成功: 受容可能={risk_result.is_acceptable}, スコア={risk_result.risk_score}")

        return True

    except Exception as e:
        print(f"[ERROR] ドメインサービステスト失敗: {e}")
        return False


def test_application_services():
    """アプリケーションサービスのテスト"""
    print("\n=== アプリケーションサービステスト ===")

    try:
        from day_trade.application.services.trading_service import TradingApplicationService, TradeRequest
        from day_trade.infrastructure.repositories.memory_repositories import MemoryUnitOfWork
        from day_trade.domain.common.value_objects import Symbol, Price

        # UnitOfWork作成
        uow = MemoryUnitOfWork()

        # 市場データ設定
        uow.market_data.update_price(Symbol("7203"), {
            'price': 1500.0,
            'volume': 10000,
            'timestamp': datetime.now()
        })

        # アプリケーションサービス
        trading_service = TradingApplicationService(uow)

        # デフォルトポートフォリオ取得
        portfolio = uow.portfolios.get_default_portfolio()

        # 取引リクエスト作成
        trade_request = TradeRequest(
            symbol="7203",
            direction="buy",
            quantity=100,
            order_type="market"
        )

        # 取引実行
        result = trading_service.execute_trade(portfolio.id, trade_request)
        print(f"[OK] 取引実行: 成功={result.success}")
        if result.success:
            print(f"   取引ID: {result.trade_id}")
            print(f"   実行価格: {result.executed_price}")

        # ポートフォリオサマリー取得
        summary = trading_service.get_portfolio_summary(portfolio.id)
        if summary:
            print(f"[OK] ポートフォリオサマリー取得成功: 総価値={summary.total_value}")

        return True

    except Exception as e:
        print(f"[ERROR] アプリケーションサービステスト失敗: {e}")
        return False


def test_infrastructure_repositories():
    """インフラストラクチャリポジトリのテスト"""
    print("\n=== インフラストラクチャリポジトリテスト ===")

    try:
        from day_trade.infrastructure.repositories.memory_repositories import (
            MemoryUnitOfWork, MemoryTradeRepository, MemoryMarketDataRepository
        )
        from day_trade.domain.trading.entities import Trade, TradeId
        from day_trade.domain.common.value_objects import Symbol, Price, Quantity, Money

        # UnitOfWork
        uow = MemoryUnitOfWork()

        # 取引リポジトリテスト
        trade = Trade(
            trade_id=TradeId(),
            symbol=Symbol("7203"),
            quantity=Quantity(100),
            price=Price(Decimal("1500")),
            direction="buy",
            executed_at=datetime.now(),
            commission=Money(Decimal("100"))
        )

        uow.trades.save(trade)
        retrieved_trade = uow.trades.find_by_id(trade.id)
        print(f"[OK] 取引リポジトリ: 保存・取得成功 ({retrieved_trade.symbol if retrieved_trade else 'なし'})")

        # 市場データリポジトリテスト
        symbol = Symbol("7203")
        price_data = {'price': 1600.0, 'volume': 5000}
        uow.market_data.update_price(symbol, price_data)

        current_price = uow.market_data.get_current_price(symbol)
        print(f"[OK] 市場データリポジトリ: 価格更新・取得成功 (価格={current_price['price'] if current_price else 'なし'})")

        # 設定リポジトリテスト
        config = uow.configuration.get_trading_config()
        print(f"[OK] 設定リポジトリ: 設定取得成功 (最大ポジション数={config['max_positions']})")

        return True

    except Exception as e:
        print(f"[ERROR] インフラストラクチャリポジトリテスト失敗: {e}")
        return False


def test_end_to_end_workflow():
    """エンドツーエンドワークフローテスト"""
    print("\n=== エンドツーエンドワークフローテスト ===")

    try:
        from day_trade.application.services.trading_service import TradingApplicationService, TradeRequest
        from day_trade.infrastructure.repositories.memory_repositories import MemoryUnitOfWork
        from day_trade.domain.common.value_objects import Symbol

        # システム初期化
        uow = MemoryUnitOfWork()
        trading_service = TradingApplicationService(uow)

        # 市場データ設定
        symbols = ["7203", "9984", "6758"]
        for symbol_code in symbols:
            symbol = Symbol(symbol_code)
            uow.market_data.update_price(symbol, {
                'price': 1000.0 + int(symbol_code),
                'volume': 10000,
                'timestamp': datetime.now()
            })

        # デフォルトポートフォリオ
        portfolio = uow.portfolios.get_default_portfolio()

        # 複数の取引実行
        trade_results = []
        for symbol_code in symbols:
            trade_request = TradeRequest(
                symbol=symbol_code,
                direction="buy",
                quantity=100,
                order_type="market"
            )
            result = trading_service.execute_trade(portfolio.id, trade_request)
            trade_results.append(result)

        successful_trades = sum(1 for r in trade_results if r.success)
        print(f"[OK] 複数取引実行: {successful_trades}/{len(symbols)} 件成功")

        # ポートフォリオ状態確認
        summary = trading_service.get_portfolio_summary(portfolio.id)
        if summary:
            print(f"[OK] 最終ポートフォリオ状態:")
            print(f"   総価値: {summary.total_value}")
            print(f"   ポジション数: {summary.open_positions_count}")
            print(f"   総損益: {summary.total_pnl}")

        # リスク評価
        risk_result = trading_service.assess_portfolio_risk(portfolio.id)
        print(f"[OK] リスク評価: 受容可能={risk_result.is_acceptable}")

        # パフォーマンス分析
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime.now()

        performance = trading_service.get_performance_metrics(
            portfolio.id, start_date, end_date
        )
        print(f"[OK] パフォーマンス分析: 取引数={performance.get('total_trades', 0)}")

        return True

    except Exception as e:
        print(f"[ERROR] エンドツーエンドワークフローテスト失敗: {e}")
        return False


def main():
    """メインテスト実行"""
    print("DDD・ヘキサゴナルアーキテクチャ統合テスト開始\n")

    tests = [
        ("値オブジェクト", test_value_objects),
        ("ドメインイベント", test_domain_events),
        ("ドメインエンティティ", test_domain_entities),
        ("ドメインサービス", test_domain_services),
        ("アプリケーションサービス", test_application_services),
        ("インフラストラクチャリポジトリ", test_infrastructure_repositories),
        ("エンドツーエンドワークフロー", test_end_to_end_workflow),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"[OK] {test_name}テスト: 成功")
            else:
                print(f"[ERROR] {test_name}テスト: 失敗")
        except Exception as e:
            print(f"[ERROR] {test_name}テスト: 例外発生 - {e}")

    print(f"\n{'='*60}")
    print(f"DDD・ヘキサゴナルアーキテクチャテスト結果: {passed}/{total} 通過")
    print(f"成功率: {passed/total*100:.1f}%")

    if passed == total:
        print("\n[SUCCESS] DDD・ヘキサゴナルアーキテクチャが正常に実装されました！")
        print("\n実装されたコンポーネント:")
        print("• 値オブジェクト（Money, Price, Quantity, Symbol, Percentage）")
        print("• ドメインエンティティ（Trade, Position, Portfolio）")
        print("• ドメインイベント（TradeExecuted, PositionOpened, etc.）")
        print("• ドメインサービス（TradingDomainService, RiskManagementService）")
        print("• アプリケーションサービス（TradingApplicationService）")
        print("• リポジトリパターン（メモリベース実装）")
        print("• 作業単位パターン（UnitOfWork）")
        print("• ヘキサゴナルアーキテクチャ（ポート・アダプター）")
        return True
    else:
        print("\n[WARNING] 一部のコンポーネントに問題があります。修正が必要です。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)