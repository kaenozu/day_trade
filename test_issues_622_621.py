#!/usr/bin/env python3
"""
Issues #622, #621 テストケース

日本株シンボル処理とフォールバック戦略の改善をテスト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.data.real_market_data import RealMarketDataManager

def test_issue_622_japanese_symbol_formatting():
    """Issue #622: 日本株シンボル処理の改善テスト"""
    print("=== Issue #622: 日本株シンボル処理テスト ===")

    manager = RealMarketDataManager()

    test_cases = [
        ("7203", "7203.T"),      # 日本株（4桁数字）
        ("7203.T", "7203.T"),    # 既に.T付き日本株
        ("AAPL", "AAPL"),        # 米国株（英字）
        ("BRK.B", "BRK.B"),      # 米国株（ドット付き）
        ("123", "123.T"),        # 3桁数字（日本株として扱う）
        ("12345", "12345.T"),    # 5桁数字（日本株として扱う）
        ("TSE:7203", "TSE:7203"), # 取引所コード付き
    ]

    for input_symbol, expected in test_cases:
        result = manager._format_japanese_symbol(input_symbol)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"  {status} {input_symbol} -> {result} (期待値: {expected})")

    print()

def test_issue_621_fallback_strategy():
    """Issue #621: フォールバック戦略の改善テスト"""
    print("=== Issue #621: フォールバック戦略テスト ===")

    manager = RealMarketDataManager()

    # 存在しない銘柄コードでフォールバック戦略をテスト
    test_symbols = [
        "9999",  # 存在しない日本株
        "7203",  # 実在する日本株（キャッシュがあるかもしれない）
        "8306",  # 実在する日本株（金融業）
        "FAKE",  # 存在しない米国株
    ]

    for symbol in test_symbols:
        print(f"  テスト銘柄: {symbol}")

        # キャッシュから価格取得テスト
        cached_price = manager._get_last_cached_price(symbol)
        print(f"    キャッシュ価格: {cached_price}")

        # セクター推定テスト
        estimated_price = manager._estimate_price_by_sector(symbol)
        print(f"    セクター推定価格: {estimated_price}")

        # 総合フォールバック価格テスト
        fallback_price = manager._get_fallback_price(symbol)
        print(f"    フォールバック価格: {fallback_price}円")

        print()

def test_integration():
    """統合テスト"""
    print("=== 統合テスト ===")

    manager = RealMarketDataManager()

    # 現在価格取得（改善されたフォールバック付き）
    test_symbols = ["7203", "9999", "AAPL"]

    for symbol in test_symbols:
        try:
            price = manager.get_current_price(symbol)
            print(f"  {symbol}: {price}円")
        except Exception as e:
            print(f"  {symbol}: エラー - {e}")

    print()

if __name__ == "__main__":
    print("Issues #622, #621 修正テスト\n")

    test_issue_622_japanese_symbol_formatting()
    test_issue_621_fallback_strategy()
    test_integration()

    print("テスト完了")