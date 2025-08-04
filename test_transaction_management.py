"""
トランザクション管理テスト
Issue 187: DB操作のトランザクション管理の徹底
"""

import os
import sys
from decimal import Decimal

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.day_trade.core.trade_manager import TradeManager


def test_buy_stock_transaction():
    """株式買い注文のトランザクション管理テスト"""

    print("=== 株式買い注文トランザクションテスト ===")

    # TradeManagerインスタンス作成（DBは使わずメモリ内テスト）
    tm = TradeManager(load_from_db=False)

    # 買い注文実行
    try:
        result = tm.buy_stock(
            symbol="7203",  # トヨタ
            quantity=100,
            price=Decimal("2500"),
            current_market_price=Decimal("2520"),
            notes="テスト買い注文",
            persist_to_db=False,  # メモリ内テスト
        )

        print("買い注文成功:")
        print(f"  取引ID: {result['trade_id']}")
        print(f"  銘柄: {result['symbol']}")
        print(f"  数量: {result['quantity']}")
        print(f"  価格: {result['price']}円")
        print(f"  手数料: {result['commission']}円")
        print(f"  総コスト: {result['total_cost']}円")

        if result["position"]:
            pos = result["position"]
            print("  更新後ポジション:")
            print(f"    保有数量: {pos['quantity']}")
            print(f"    平均単価: {pos['average_price']}円")
            print(f"    時価総額: {pos['market_value']}円")
            print(f"    含み損益: {pos['unrealized_pnl']}円")

        return True

    except Exception as e:
        print(f"買い注文エラー: {e}")
        return False


def test_sell_stock_transaction():
    """株式売り注文のトランザクション管理テスト"""

    print("\n=== 株式売り注文トランザクションテスト ===")

    # TradeManagerインスタンス作成
    tm = TradeManager(load_from_db=False)

    # まず買い注文でポジション作成
    tm.buy_stock(
        symbol="7203", quantity=200, price=Decimal("2500"), persist_to_db=False
    )

    # 売り注文実行
    try:
        result = tm.sell_stock(
            symbol="7203",
            quantity=100,
            price=Decimal("2650"),
            current_market_price=Decimal("2650"),
            notes="テスト売り注文",
            persist_to_db=False,
        )

        print("売り注文成功:")
        print(f"  取引ID: {result['trade_id']}")
        print(f"  銘柄: {result['symbol']}")
        print(f"  数量: {result['quantity']}")
        print(f"  価格: {result['price']}円")
        print(f"  手数料: {result['commission']}円")
        print(f"  売却収入: {result['gross_proceeds']}円")
        print(f"  ポジション完全クローズ: {result['position_closed']}")

        if result["realized_pnl"]:
            pnl = result["realized_pnl"]
            print("  実現損益:")
            print(f"    損益: {pnl['pnl']}円")
            print(f"    損益率: {pnl['pnl_percent']}%")

        if result["position"]:
            pos = result["position"]
            print("  残りポジション:")
            print(f"    保有数量: {pos['quantity']}")
            print(f"    平均単価: {pos['average_price']}円")

        return True

    except Exception as e:
        print(f"売り注文エラー: {e}")
        return False


def test_transaction_rollback():
    """トランザクションロールバックテスト"""

    print("\n=== トランザクションロールバックテスト ===")

    tm = TradeManager(load_from_db=False)

    # 正常な買い注文でポジション作成
    tm.buy_stock(
        symbol="7203", quantity=100, price=Decimal("2500"), persist_to_db=False
    )

    initial_position = tm.get_position("7203")
    initial_trades_count = len(tm.trades)

    print("初期状態:")
    print(f"  ポジション数量: {initial_position.quantity}")
    print(f"  取引履歴数: {initial_trades_count}")

    # エラーを発生させる売り注文（保有数量を超過）
    try:
        tm.sell_stock(
            symbol="7203",
            quantity=200,  # 保有数量100を超過
            price=Decimal("2600"),
            persist_to_db=False,
        )
        print("エラー：想定していたエラーが発生しませんでした")
        return False

    except ValueError as e:
        print(f"期待されたエラー発生: {e}")

        # データがロールバックされていることを確認
        rollback_position = tm.get_position("7203")
        rollback_trades_count = len(tm.trades)

        print("ロールバック後:")
        print(f"  ポジション数量: {rollback_position.quantity}")
        print(f"  取引履歴数: {rollback_trades_count}")

        # 初期状態と同じであることを確認
        if (
            rollback_position.quantity == initial_position.quantity
            and rollback_trades_count == initial_trades_count
        ):
            print("✓ トランザクションロールバック成功")
            return True
        else:
            print("✗ トランザクションロールバック失敗")
            return False


def test_execute_trade_order():
    """統一取引注文インターフェーステスト"""

    print("\n=== 統一取引注文インターフェーステスト ===")

    tm = TradeManager(load_from_db=False)

    # 買い注文
    buy_order = {
        "action": "buy",
        "symbol": "8411",  # みずほFG
        "quantity": 1000,
        "price": Decimal("150"),
        "current_market_price": Decimal("152"),
        "notes": "統一インターフェース買いテスト",
    }

    try:
        buy_result = tm.execute_trade_order(buy_order, persist_to_db=False)
        print(f"買い注文成功: {buy_result['trade_id']}")

        # 売り注文
        sell_order = {
            "action": "sell",
            "symbol": "8411",
            "quantity": 500,
            "price": Decimal("155"),
            "current_market_price": Decimal("155"),
            "notes": "統一インターフェース売りテスト",
        }

        sell_result = tm.execute_trade_order(sell_order, persist_to_db=False)
        print(f"売り注文成功: {sell_result['trade_id']}")

        # 無効な注文
        invalid_order = {
            "action": "invalid",
            "symbol": "8411",
            "quantity": 100,
            "price": Decimal("150"),
        }

        try:
            tm.execute_trade_order(invalid_order, persist_to_db=False)
            print("エラー：無効な注文が受け入れられました")
            return False
        except ValueError as e:
            print(f"期待されたエラー発生: {e}")

        return True

    except Exception as e:
        print(f"統一インターフェーステストエラー: {e}")
        return False


def test_data_consistency():
    """データ整合性テスト"""

    print("\n=== データ整合性テスト ===")

    tm = TradeManager(load_from_db=False)

    # 複数の取引実行
    transactions = [
        ("buy", "7203", 100, Decimal("2500")),
        ("buy", "7203", 200, Decimal("2480")),
        ("sell", "7203", 150, Decimal("2520")),
        ("buy", "8411", 1000, Decimal("150")),
        ("sell", "8411", 300, Decimal("155")),
    ]

    for action, symbol, quantity, price in transactions:
        if action == "buy":
            tm.buy_stock(symbol, quantity, price, persist_to_db=False)
        else:
            tm.sell_stock(symbol, quantity, price, persist_to_db=False)

    # データ整合性チェック
    summary = tm.get_portfolio_summary()

    print("ポートフォリオサマリー:")
    print(f"  総ポジション数: {summary['total_positions']}")
    print(f"  総コスト: {summary['total_cost']}円")
    print(f"  総取引数: {summary['total_trades']}")
    print(f"  実現損益: {summary['total_realized_pnl']}円")
    print(f"  勝率: {summary['win_rate']}")

    # 個別ポジション確認
    print("\n個別ポジション:")
    for symbol, position in tm.get_all_positions().items():
        print(
            f"  {symbol}: {position.quantity}株 (平均単価: {position.average_price}円)"
        )

    return True


if __name__ == "__main__":
    print("トランザクション管理テスト開始")

    tests = [
        ("株式買い注文トランザクション", test_buy_stock_transaction),
        ("株式売り注文トランザクション", test_sell_stock_transaction),
        ("トランザクションロールバック", test_transaction_rollback),
        ("統一取引注文インターフェース", test_execute_trade_order),
        ("データ整合性", test_data_consistency),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name}: 成功")
                passed += 1
            else:
                print(f"✗ {test_name}: 失敗")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name}: エラー - {e}")
            failed += 1

    print("\n=== テスト結果 ===")
    print(f"成功: {passed}")
    print(f"失敗: {failed}")
    print(
        f"成功率: {passed}/{passed + failed} ({passed / (passed + failed) * 100:.1f}%)"
    )

    if failed == 0:
        print("✓ すべてのトランザクション管理テストが成功しました")
    else:
        print("✗ 一部のテストが失敗しました")
