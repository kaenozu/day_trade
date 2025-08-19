"""
リファクタリング済みコンポーネントの簡単テスト

Unicode文字を避けた基本的な動作確認
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """基本機能のテスト"""
    print("=== 基本機能テスト開始 ===")

    success_count = 0
    total_tests = 0

    # 1. 取引モデルテスト
    total_tests += 1
    try:
        from day_trade.core.models.trade_models import Trade, TradeStatus

        trade = Trade(
            symbol="7203",
            trade_type="buy",
            quantity=100,
            price=Decimal("1000"),
            timestamp=datetime.now(),
            status=TradeStatus.FILLED
        )

        assert trade.symbol == "7203"
        assert trade.is_buy == True
        assert trade.quantity == 100
        print("[OK] 取引モデル作成・操作成功")
        success_count += 1

    except Exception as e:
        print(f"[ERROR] 取引モデルテスト失敗: {e}")

    # 2. 計算機テスト
    total_tests += 1
    try:
        from day_trade.core.calculators.trade_calculator import DecimalCalculator

        calc = DecimalCalculator()
        result = calc.safe_decimal_conversion("1000.50", "テスト価格")

        assert isinstance(result, Decimal)
        assert result == Decimal("1000.50")
        print("[OK] 計算機 Decimal変換成功")
        success_count += 1

    except Exception as e:
        print(f"[ERROR] 計算機テスト失敗: {e}")

    # 3. GPU共通機能テスト
    total_tests += 1
    try:
        from day_trade.acceleration.gpu_common import create_indicator_calculator
        import numpy as np

        calc = create_indicator_calculator(prefer_gpu=False)
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        sma = calc.calculate_sma(prices, period=3)

        assert len(sma) > 0
        print("[OK] GPU共通機能 SMA計算成功")
        success_count += 1

    except Exception as e:
        print(f"[ERROR] GPU共通機能テスト失敗: {e}")

    # 4. キャッシュテスト
    total_tests += 1
    try:
        from day_trade.performance.optimized_cache_v2 import OptimizedCacheManager

        manager = OptimizedCacheManager()
        cache = manager.create_cache("test", "lru", max_size=100)

        cache.put("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"
        print("[OK] 最適化キャッシュ動作成功")
        success_count += 1

    except Exception as e:
        print(f"[ERROR] キャッシュテスト失敗: {e}")

    # 5. エラーハンドリングテスト
    total_tests += 1
    try:
        from day_trade.core.error_handling.enhanced_error_system import NetworkError, ErrorCategory

        error = NetworkError("テストエラー")

        assert error.category == ErrorCategory.NETWORK
        assert "テストエラー" in str(error)
        print("[OK] エラーハンドリング作成成功")
        success_count += 1

    except Exception as e:
        print(f"[ERROR] エラーハンドリングテスト失敗: {e}")

    # 6. 型定義テスト
    total_tests += 1
    try:
        from day_trade.core.types.trading_types import TradingResult, ensure_decimal

        result = TradingResult(success=True, data="test")
        decimal_val = ensure_decimal(100.5)

        assert result.is_success == True
        assert isinstance(decimal_val, Decimal)
        print("[OK] 型定義・ユーティリティ成功")
        success_count += 1

    except Exception as e:
        print(f"[ERROR] 型定義テスト失敗: {e}")

    # 結果表示
    print(f"\n=== テスト結果 ===")
    print(f"成功: {success_count}/{total_tests}")
    print(f"成功率: {success_count/total_tests*100:.1f}%")

    if success_count == total_tests:
        print("全ての基本機能が正常に動作しています")
        return True
    else:
        print("一部の機能に問題があります")
        return False


def test_application_startup():
    """アプリケーション起動テスト"""
    print("\n=== アプリケーション起動テスト ===")

    try:
        # 基本的なモジュールがインポートできるかテスト
        import day_trade.core.models.trade_models
        import day_trade.core.calculators.trade_calculator
        import day_trade.acceleration.gpu_common
        import day_trade.performance.optimized_cache_v2
        import day_trade.core.error_handling.enhanced_error_system
        import day_trade.core.types.trading_types

        print("[OK] 全ての新規モジュールのインポート成功")
        return True

    except Exception as e:
        print(f"[ERROR] モジュールインポート失敗: {e}")
        return False


def main():
    """メインテスト"""
    print("リファクタリング済みコンポーネント簡単テスト開始")
    print("=" * 50)

    # 基本機能テスト
    basic_success = test_basic_functionality()

    # アプリケーション起動テスト
    startup_success = test_application_startup()

    # 総合結果
    print(f"\n{'='*50}")
    print("総合テスト結果:")
    print(f"基本機能: {'成功' if basic_success else '失敗'}")
    print(f"起動テスト: {'成功' if startup_success else '失敗'}")

    overall_success = basic_success and startup_success
    print(f"総合判定: {'成功' if overall_success else '失敗'}")

    if overall_success:
        print("\nリファクタリング完了！新しいアーキテクチャが正常に動作しています。")
    else:
        print("\n一部に問題があります。修正が必要です。")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)