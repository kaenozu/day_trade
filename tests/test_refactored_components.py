"""
リファクタリング済みコンポーネントのテスト

新しく作成した各モジュールが正常に動作するかテスト
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent / "src"))

def test_trade_models():
    """取引モデルのテスト"""
    print("=== 取引モデルテスト ===")

    try:
        from day_trade.core.models.trade_models import Trade, BuyLot, Position, TradeStatus

        # Trade作成テスト
        trade = Trade(
            symbol="7203",
            trade_type="buy",
            quantity=100,
            price=Decimal("1000"),
            timestamp=datetime.now(),
            commission=Decimal("100"),
            status=TradeStatus.FILLED
        )

        print(f"✅ Trade作成成功: {trade.symbol} {trade.quantity}株 @{trade.price}")
        print(f"   総コスト: {trade.total_cost}")
        print(f"   買い注文: {trade.is_buy}")

        # BuyLot作成テスト
        buy_lot = BuyLot(
            symbol="7203",
            quantity=100,
            price=Decimal("1000"),
            timestamp=datetime.now(),
            remaining_quantity=100,
            commission=Decimal("100")
        )

        print(f"✅ BuyLot作成成功: 残数量={buy_lot.remaining_quantity}")
        print(f"   手数料込み単価: {buy_lot.average_price_with_commission}")

        # Position作成テスト
        position = Position(symbol="7203")
        position.buy_lots = [buy_lot]
        position.total_quantity = 100
        position.update_metrics(Decimal("1100"))

        print(f"✅ Position作成成功: {position.symbol} {position.total_quantity}株保有")
        print(f"   未実現損益: {position.unrealized_pnl}")

    except Exception as e:
        print(f"❌ 取引モデルテスト失敗: {e}")
        return False

    return True


def test_trade_calculator():
    """取引計算機のテスト"""
    print("\n=== 取引計算機テスト ===")

    try:
        from day_trade.core.calculators.trade_calculator import (
            DecimalCalculator, CommissionCalculator, PnLCalculator
        )
        from day_trade.core.models.trade_models import BuyLot

        # Decimal変換テスト
        calc = DecimalCalculator()
        price = calc.safe_decimal_conversion("1000.50", "価格")
        print(f"✅ Decimal変換成功: {price} (型: {type(price)})")

        # 手数料計算テスト
        commission_calc = CommissionCalculator()
        trade_amount = Decimal("100000")
        commission = commission_calc.calculate_commission(trade_amount)
        print(f"✅ 手数料計算成功: {trade_amount}円の取引で手数料{commission}円")

        # 損益計算テスト
        pnl_calc = PnLCalculator()
        buy_lots = [
            BuyLot("7203", 100, Decimal("1000"), datetime.now(), 100, Decimal("100"))
        ]

        realized_pnl_list, updated_lots = pnl_calc.calculate_fifo_pnl(
            sell_quantity=50,
            sell_price=Decimal("1100"),
            sell_commission=Decimal("100"),
            buy_lots=buy_lots
        )

        print(f"✅ FIFO損益計算成功: {len(realized_pnl_list)}件の実現損益")
        if realized_pnl_list:
            pnl = realized_pnl_list[0]
            print(f"   損益: {pnl.net_pnl}円 (収益率: {pnl.return_rate:.2f}%)")

    except Exception as e:
        print(f"❌ 取引計算機テスト失敗: {e}")
        return False

    return True


def test_gpu_common():
    """GPU共通機能のテスト"""
    print("\n=== GPU共通機能テスト ===")

    try:
        from day_trade.acceleration.gpu_common import (
            TechnicalIndicatorCalculator, create_indicator_calculator
        )
        import numpy as np

        # 計算機作成
        calc = create_indicator_calculator(prefer_gpu=False)  # CPUモード
        print(f"✅ 計算機作成成功: GPU使用={calc.use_gpu}")

        # テストデータ
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])

        # SMA計算テスト
        sma = calc.calculate_sma(prices, period=5)
        print(f"✅ SMA計算成功: 最新値={sma[-1]:.2f}")

        # EMA計算テスト
        ema = calc.calculate_ema(prices, period=5)
        print(f"✅ EMA計算成功: 最新値={ema[-1]:.2f}")

        # RSI計算テスト（期間を短くする）
        if len(prices) >= 5:
            rsi = calc.calculate_rsi(prices, period=3)
            print(f"✅ RSI計算成功: 最新値={rsi[-1]:.2f}")

        # 結果をNumPy形式で変換
        sma_np = calc.to_numpy(sma)
        print(f"✅ NumPy変換成功: {type(sma_np)}")

    except Exception as e:
        print(f"❌ GPU共通機能テスト失敗: {e}")
        return False

    return True


def test_optimized_cache():
    """最適化キャッシュのテスト"""
    print("\n=== 最適化キャッシュテスト ===")

    try:
        from day_trade.performance.optimized_cache_v2 import (
            OptimizedCacheManager, cached, calculate_indicator_cached
        )

        # キャッシュマネージャーテスト
        manager = OptimizedCacheManager()
        cache = manager.create_cache("test_cache", "hierarchical", l1_size=10, l2_ttl=60.0)
        print(f"✅ キャッシュ作成成功: {type(cache).__name__}")

        # 基本操作テスト
        cache.put("key1", "value1")
        value = cache.get("key1")
        print(f"✅ キャッシュ基本操作成功: {value}")

        # 統計情報テスト
        if hasattr(cache, 'get_stats'):
            stats = cache.get_stats()
            print(f"✅ キャッシュ統計取得成功: L1ヒット={stats.get('l1_hits', 0)}")

        # デコレーターテスト
        result1 = calculate_indicator_cached("7203", "sma", 20)
        result2 = calculate_indicator_cached("7203", "sma", 20)  # キャッシュヒット
        print(f"✅ キャッシュデコレーター成功: {result1} = {result2}")

    except Exception as e:
        print(f"❌ 最適化キャッシュテスト失敗: {e}")
        return False

    return True


def test_error_handling():
    """エラーハンドリングのテスト"""
    print("\n=== エラーハンドリングテスト ===")

    try:
        from day_trade.core.error_handling.enhanced_error_system import (
            BaseApplicationError, ErrorCategory, ErrorSeverity,
            NetworkError, ValidationError, global_error_handler
        )

        # カスタム例外テスト
        error = NetworkError(
            "テスト用ネットワークエラー",
            details={"url": "https://example.com", "timeout": 30}
        )
        print(f"✅ カスタム例外作成成功: {error.error_id}")
        print(f"   カテゴリ: {error.category.value}, 重要度: {error.severity.value}")

        # バリデーションエラーテスト
        validation_error = ValidationError(
            "無効な入力値",
            user_message="入力された値が正しくありません"
        )
        print(f"✅ バリデーションエラー作成成功: {validation_error.user_message}")

        # エラーハンドラー統計テスト
        stats = global_error_handler.get_error_stats()
        print(f"✅ エラー統計取得成功: 総エラー数={stats['total_errors']}")

    except Exception as e:
        print(f"❌ エラーハンドリングテスト失敗: {e}")
        return False

    return True


def test_trading_types():
    """取引型定義のテスト"""
    print("\n=== 取引型定義テスト ===")

    try:
        from day_trade.core.types.trading_types import (
            MarketDataDict, OrderDict, TradingResult,
            NumericRange, ensure_decimal, ensure_symbol,
            validate_trading_data
        )

        # MarketDataDict作成テスト
        market_data: MarketDataDict = {
            "symbol": "7203",
            "timestamp": datetime.now(),
            "open": 1000.0,
            "high": 1100.0,
            "low": 950.0,
            "close": 1050.0,
            "volume": 10000
        }
        print(f"✅ MarketDataDict作成成功: {market_data['symbol']}")

        # TradingResult テスト
        success_result = TradingResult(success=True, data="test_data")
        print(f"✅ TradingResult成功ケース: {success_result.get_data()}")

        error_result = TradingResult(success=False, error="テストエラー")
        default_data = error_result.get_data_or_default("デフォルト値")
        print(f"✅ TradingResultエラーケース: {default_data}")

        # NumericRange テスト
        price_range = NumericRange(Decimal("100"), Decimal("2000"))
        is_valid = price_range.contains(Decimal("1500"))
        clamped = price_range.clamp(Decimal("3000"))
        print(f"✅ NumericRange テスト: 有効={is_valid}, クランプ={clamped}")

        # ユーティリティ関数テスト
        decimal_val = ensure_decimal(1000.50)
        symbol_val = ensure_symbol("  7203  ")
        print(f"✅ ユーティリティ関数: {decimal_val} ({type(decimal_val)}), '{symbol_val}'")

        # データバリデーションテスト
        test_data = {
            "symbol": "7203",
            "direction": "buy",
            "quantity": 100,
            "timestamp": datetime.now()
        }
        is_valid_data = validate_trading_data(test_data)
        print(f"✅ データバリデーション: {is_valid_data}")

    except Exception as e:
        print(f"❌ 取引型定義テスト失敗: {e}")
        return False

    return True


def main():
    """メインテスト実行"""
    print("リファクタリング済みコンポーネントテスト開始\n")

    tests = [
        ("取引モデル", test_trade_models),
        ("取引計算機", test_trade_calculator),
        ("GPU共通機能", test_gpu_common),
        ("最適化キャッシュ", test_optimized_cache),
        ("エラーハンドリング", test_error_handling),
        ("取引型定義", test_trading_types),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}テスト: 成功")
            else:
                print(f"❌ {test_name}テスト: 失敗")
        except Exception as e:
            print(f"❌ {test_name}テスト: 例外発生 - {e}")

    print(f"\nテスト結果: {passed}/{total} 通過")
    print(f"成功率: {passed/total*100:.1f}%")

    if passed == total:
        print("全てのリファクタリングコンポーネントが正常に動作しています！")
        return True
    else:
        print("一部のコンポーネントに問題があります。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)