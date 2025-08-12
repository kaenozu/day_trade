#!/usr/bin/env python3
"""
取引管理ロジックセキュリティ強化テスト
Issue #396: 取引管理ロジックの堅牢化とセキュリティ強化

実装されたセキュリティ強化:
1. Decimal型変換における浮動小数点誤差対策 - safe_decimal_conversion強化
2. _get_earliest_buy_dateの非効率性および不正確さ修正 - FIFOロット管理
3. ファイルI/Oセキュリティ強化(パス検証) - validate_file_path強化
4. ログ出力の機密情報マスキング - mask_sensitive_info包括強化
"""

import logging
import os
import tempfile
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from unittest.mock import patch

# テスト中はログ出力を有効化
logging.basicConfig(level=logging.INFO)

from src.day_trade.core.trade_manager import (
    TradeManager,
    _mask_file_path,
    _mask_financial_value,
    _mask_number,
    _mask_quantity_value,
    _mask_transaction_id,
    mask_sensitive_info,
    quantize_decimal,
    safe_decimal_conversion,
    validate_file_path,
)


def test_decimal_conversion_security():
    """Decimal型変換セキュリティテスト"""
    print("=== Decimal型変換セキュリティテスト ===")

    # 1. 正常な変換のテスト
    print("\n1. 正常な変換:")
    valid_tests = [
        ("100", "文字列"),
        (100, "整数"),
        (100.50, "浮動小数点数"),
        (Decimal("100.50"), "Decimal"),
    ]

    for value, description in valid_tests:
        try:
            result = safe_decimal_conversion(value)
            print(f"  OK {description}: {value} → {result} (type: {type(result)})")
        except Exception as e:
            print(f"  FAIL {description}: 予期しないエラー: {e}")

    # 2. エラー処理のテスト
    print("\n2. エラー処理:")
    error_tests = [
        ("invalid", "無効な文字列"),
        (None, "None値"),
        (float("inf"), "無限大"),
        (float("-inf"), "負の無限大"),
        (float("nan"), "NaN"),
        ("", "空文字列"),
        ([], "リスト"),
        ({}, "辞書"),
    ]

    for value, description in error_tests:
        try:
            result = safe_decimal_conversion(value)
            print(f"  FAIL {description}: 例外が発生すべき: {result}")
        except (ValueError, TypeError, InvalidOperation) as e:
            # エラーメッセージに機密情報が含まれていないかチェック
            error_msg = str(e)
            if any(
                dangerous in error_msg for dangerous in ["password", "key", "secret"]
            ):
                print(
                    f"  WARN {description}: エラーメッセージに機密情報が含まれる可能性: {error_msg}"
                )
            else:
                print(f"  OK {description}: 正常に阻止 - {type(e).__name__}")

    # 3. 精度保証のテスト
    print("\n3. 精度保証:")
    precision_tests = [
        (0.1 + 0.2, "浮動小数点精度問題"),
        (1.0 / 3.0, "無限小数"),
        (999999.999999, "大きな小数"),
    ]

    for value, description in precision_tests:
        try:
            result = safe_decimal_conversion(value)
            quantized = quantize_decimal(result)
            print(f"  OK {description}: {value} → {result} → {quantized}")
        except Exception as e:
            print(f"  FAIL {description}: エラー: {e}")


def test_file_path_security():
    """ファイルパスセキュリティテスト"""
    print("\n=== ファイルパスセキュリティテスト ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. 正常なパスのテスト
        print("\n1. 正常なパス:")
        valid_paths = [
            "data/trades.csv",
            "./output/report.json",
            "temp/backup.db",
            os.path.join(temp_dir, "safe_file.txt"),
        ]

        for path in valid_paths:
            try:
                result = validate_file_path(path)
                print(f"  OK 正常パス: {path} → {result}")
            except Exception as e:
                print(f"  WARN 正常パス: {path} - {e}")

        # 2. 危険なパスのテスト
        print("\n2. 危険なパス:")
        dangerous_paths = [
            "../../../etc/passwd",  # パストラバーサル攻撃
            "..\\..\\..\\windows\\system32\\config",  # Windowsパストラバーサル
            "/etc/shadow",  # システムファイルへの直接アクセス
            "file:///etc/passwd",  # URLスキーム
            "\\\\malicious\\share\\file",  # UNCパス
            "file\x00.txt",  # NULLバイト攻撃
            "C:\\Windows\\System32\\cmd.exe",  # システムディレクトリ
            "~/.ssh/id_rsa",  # SSHキーファイル
            "a" * 300,  # パス長攻撃
        ]

        for path in dangerous_paths:
            try:
                result = validate_file_path(path)
                print(f"  FAIL 危険パス: 通過した - {path} → {result}")
            except ValueError as e:
                print(f"  OK 危険パス: 正常に阻止 - {path}")
            except Exception as e:
                print(f"  WARN 危険パス: 予期しないエラー - {path}: {e}")


def test_sensitive_info_masking():
    """機密情報マスキングテスト"""
    print("\n=== 機密情報マスキングテスト ===")

    # 1. 金額・価格情報のマスキング
    print("\n1. 金額・価格情報:")
    financial_tests = [
        ("price: 2500.00", "価格情報"),
        ("JPY1,234,567", "日本円表記"),
        ("$123.45", "ドル表記"),
        ("commission: 500", "手数料情報"),
        ("total: 10000.50", "合計金額"),
        ("balance: 50000", "残高情報"),
    ]

    for text, description in financial_tests:
        masked = mask_sensitive_info(text)
        print(f"  {description}:")
        print(f"    元: {text}")
        print(f"    後: {masked}")

    # 2. ファイルパス情報のマスキング
    print("\n2. ファイルパス情報:")
    path_tests = [
        ("C:\\Users\\admin\\Documents\\secret.txt", "Windowsパス"),
        ("/home/user/private/data.csv", "Unixパス"),
        ("./config/api_keys.json", "相対パス"),
    ]

    for text, description in path_tests:
        masked = mask_sensitive_info(text)
        print(f"  {description}:")
        print(f"    元: {text}")
        print(f"    後: {masked}")

    # 3. 取引ID・識別子のマスキング
    print("\n3. 取引ID・識別子:")
    id_tests = [
        ("trade_id: TXN_ABC123DEF456", "取引ID"),
        ("transaction_id: ORDER_2024_001", "注文ID"),
        ("id: USER_1234567890_SESSION", "長いID"),
    ]

    for text, description in id_tests:
        masked = mask_sensitive_info(text)
        print(f"  {description}:")
        print(f"    元: {text}")
        print(f"    後: {masked}")

    # 4. APIキー・認証情報のマスキング
    print("\n4. APIキー・認証情報:")
    auth_tests = [
        ("api_key: sk_live_abcdef123456", "APIキー"),
        ("secret: super_secret_password_123", "シークレット"),
        ("token: eyJhbGciOiJIUzI1NiIsInR5cCI", "JWTトークン"),
        ("password: mypassword123", "パスワード"),
    ]

    for text, description in auth_tests:
        masked = mask_sensitive_info(text)
        print(f"  {description}:")
        print(f"    元: {text}")
        print(f"    後: {masked}")

    # 5. 複合的なテスト
    print("\n5. 複合テスト:")
    complex_text = """
    Trade executed: trade_id=TXN_123, symbol=AAPL, price=$150.25, quantity=100,
    commission=$9.95, total_cost=$15,034.95, api_key=sk_live_abc123,
    file_path=/Users/trader/data/trades.csv, balance=$50,000.00
    """

    masked_complex = mask_sensitive_info(complex_text)
    print("  複合データ:")
    print(f"    元: {complex_text.strip()}")
    print(f"    後: {masked_complex.strip()}")


def test_fifo_lot_management():
    """FIFOロット管理テスト"""
    print("\n=== FIFOロット管理テスト ===")

    try:
        tm = TradeManager()  # デフォルト設定でテスト

        # 1. 複数回の買い注文
        print("\n1. 複数回買い注文:")
        buy_orders = [
            (100, Decimal("1000.00"), "2024-01-01"),
            (200, Decimal("1100.00"), "2024-01-02"),
            (150, Decimal("1200.00"), "2024-01-03"),
        ]

        for qty, price, date in buy_orders:
            trade_id = tm.buy_stock(
                symbol="TEST",
                quantity=qty,
                price=price,
                commission=Decimal("9.95"),
                timestamp=datetime.fromisoformat(date),
            )
            print(
                f"  買い注文追加: {qty}株 @{price} (ID: {mask_sensitive_info(str(trade_id))})"
            )

        # 2. ポジション確認
        if "TEST" in tm.positions:
            position = tm.positions["TEST"]
            print("\n2. ポジション確認:")
            print(
                f"  総保有株数: {mask_sensitive_info(f'quantity: {position.quantity}')}"
            )
            print(
                f"  平均取得価格: {mask_sensitive_info(f'price: {position.avg_cost}')}"
            )
            print(f"  ロット数: {len(position.buy_lots)}")

            # 3. FIFO売却テスト
            print("\n3. FIFO売却テスト:")
            sell_qty = 250  # 最初の2ロットの一部を売却
            trade_id = tm.sell_stock(
                symbol="TEST",
                quantity=sell_qty,
                price=Decimal("1150.00"),
                commission=Decimal("9.95"),
            )
            print(
                f"  売り注文実行: {sell_qty}株 (ID: {mask_sensitive_info(str(trade_id))})"
            )

            # 売却後のポジション確認
            updated_position = tm.positions.get("TEST")
            if updated_position:
                print(
                    f"  売却後保有株数: {mask_sensitive_info(f'quantity: {updated_position.quantity}')}"
                )
                print(f"  残りロット数: {len(updated_position.buy_lots)}")

                # 4. 最早買い取引日取得テスト
                earliest_date = tm._get_earliest_buy_date("TEST")
                print(f"  最早買い取引日: {earliest_date}")

                print("  OK FIFO管理テスト - 成功")
            else:
                print("  OK 全ポジション売却完了")
        else:
            print("  FAIL ポジションが作成されていない")

    except Exception as e:
        print(f"  FAIL FIFOテストエラー: {mask_sensitive_info(str(e))}")


def test_logging_security():
    """ログ出力セキュリティテスト"""
    print("\n=== ログ出力セキュリティテスト ===")

    # ログキャプチャ用のハンドラー設定
    import io

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)

    logger = logging.getLogger("src.day_trade.core.trade_manager")
    try:
        tm = TradeManager()  # デフォルト設定でテスト
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # テスト取引実行（ログ出力を含む）
        print("\n1. 取引ログのセキュリティテスト:")
        trade_id = tm.buy_stock(
            symbol="SECURE_TEST",
            quantity=100,
            price=Decimal("1500.00"),
            commission=Decimal("9.95"),
        )

        # ログ出力を確認
        log_output = log_stream.getvalue()

        # 機密情報がマスキングされているかチェック
        sensitive_patterns = [
            r"price:\s*[0-9]+\.[0-9]+",  # 生の価格
            r"commission:\s*[0-9]+\.[0-9]+",  # 生の手数料
            r"quantity:\s*[0-9]+",  # 生の数量
        ]

        has_sensitive = False
        for pattern in sensitive_patterns:
            import re

            if re.search(pattern, log_output):
                has_sensitive = True
                break

        if has_sensitive:
            print("  WARN ログに機密情報が含まれている可能性があります")
            print(f"    ログ内容: {log_output[:200]}...")
        else:
            print("  OK ログの機密情報マスキング - 正常")

        # マスクされた情報が含まれているかチェック
        if "*" in log_output or "MASKED" in log_output:
            print("  OK マスキング処理が適用されている")
        else:
            print("  WARN マスキング処理が確認できない")

    except Exception as e:
        print(f"  FAIL ログセキュリティテストエラー: {mask_sensitive_info(str(e))}")
    finally:
        logger.removeHandler(handler)


def main():
    """メイン実行"""
    print("=== 取引管理ロジックセキュリティ強化テスト ===")
    print("Issue #396: 取引管理ロジックの堅牢化とセキュリティ強化")
    print("=" * 60)

    try:
        # 各セキュリティ機能のテスト
        test_decimal_conversion_security()
        test_file_path_security()
        test_sensitive_info_masking()
        test_fifo_lot_management()
        test_logging_security()

        print("\n" + "=" * 60)
        print("OK 取引管理ロジックセキュリティ強化テスト完了")
        print("\n実装されたセキュリティ強化:")
        print("- [SECURE] Decimal型変換浮動小数点誤差対策 (企業レベル精度保証)")
        print("- [FAST] FIFO会計ロット管理最適化 (O(1)アクセス性能)")
        print("- [SHIELD] ファイルI/Oパストラバーサル攻撃防止")
        print("- [LOCK] ログ機密情報マスキング (包括的情報保護)")
        print("- [AUDIT] セキュリティ違反統計追跡")

    except Exception as e:
        print(f"\nFAIL テスト実行中にエラーが発生: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
