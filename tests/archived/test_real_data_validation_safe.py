#!/usr/bin/env python3
"""
実市場データ検証テスト - Windows対応版

Issue #321: 実データでの最終動作確認テスト
実際の市場データの取得・品質検証の実行テスト（ASCII安全）
"""

import sys
import time
from pathlib import Path

# プロジェクトルート設定
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

print("REAL MARKET DATA VALIDATION TEST - Windows Safe")
print("Issue #321: 実データでの最終動作確認テスト")
print("=" * 60)


def test_network_connectivity():
    """ネットワーク接続テスト"""
    print("\n=== Network Connectivity Test ===")

    try:
        import urllib.request

        # 基本ネットワーク接続テスト
        try:
            urllib.request.urlopen("https://www.google.com", timeout=10)
            print("[OK] Basic network connection: Success")
            network_ok = True
        except Exception as e:
            print(f"[NG] Basic network connection: Failed - {e}")
            network_ok = False

        # Yahoo Finance API接続テスト
        try:
            import yfinance as yf

            ticker = yf.Ticker("AAPL")  # 米国株で確実にテスト
            info = ticker.info
            if info:
                print("[OK] Yahoo Finance API: Success")
                api_ok = True
            else:
                print("[NG] Yahoo Finance API: No data returned")
                api_ok = False
        except Exception as e:
            print(f"[NG] Yahoo Finance API: Failed - {e}")
            api_ok = False

        # 日本株データ取得テスト
        try:
            ticker_jp = yf.Ticker("7203.T")  # トヨタ
            hist = ticker_jp.history(period="5d")
            if not hist.empty and len(hist) > 0:
                print(f"[OK] Japanese stock data: Success ({len(hist)} records)")
                data_ok = True
            else:
                print("[NG] Japanese stock data: No data returned")
                data_ok = False
        except Exception as e:
            print(f"[NG] Japanese stock data: Failed - {e}")
            data_ok = False

        success = network_ok and api_ok and data_ok

        if success:
            print("[OK] Network connectivity test: ALL PASSED")
        else:
            print("[NG] Network connectivity test: SOME FAILED")
            print("  - Check internet connection")
            print("  - Check API access restrictions")

        return success

    except Exception as e:
        print(f"Network connectivity test error: {e}")
        return False


def test_simple_data_quality():
    """簡単なデータ品質テスト"""
    print("\n=== Simple Data Quality Test ===")

    try:
        import yfinance as yf

        # テスト対象銘柄（確実にデータがある銘柄）
        test_symbols = ["7203.T", "8306.T", "9984.T", "6758.T", "9432.T"]

        print(f"Testing {len(test_symbols)} symbols...")

        successful_fetches = 0
        total_records = 0
        quality_issues = 0

        for _i, symbol in enumerate(test_symbols):
            try:
                print(f"  Testing {symbol}...")

                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="30d")

                if hist.empty:
                    print(f"    [NG] {symbol}: No data")
                    continue

                records = len(hist)
                total_records += records

                # 簡単な品質チェック
                issues = 0

                # 負の価格チェック
                for col in ["Open", "High", "Low", "Close"]:
                    if col in hist.columns and (hist[col] <= 0).any():
                        issues += 1

                # High >= Low チェック
                if "High" in hist.columns and "Low" in hist.columns:
                    if (hist["High"] < hist["Low"]).any():
                        issues += 1

                # データ欠損チェック
                if hist.isna().any().any():
                    issues += 1

                quality_issues += issues
                successful_fetches += 1

                print(f"    [OK] {symbol}: {records} records, {issues} quality issues")

                time.sleep(0.1)  # API制限対策

            except Exception as e:
                print(f"    [NG] {symbol}: Error - {e}")

        print("\nData Quality Test Results:")
        print(f"  Successful fetches: {successful_fetches}/{len(test_symbols)}")
        print(f"  Total records: {total_records}")
        print(f"  Quality issues: {quality_issues}")
        print(f"  Success rate: {successful_fetches/len(test_symbols)*100:.1f}%")

        # 成功条件: 80%以上成功
        success = successful_fetches / len(test_symbols) >= 0.8

        if success:
            print("[OK] Data quality test: PASSED")
        else:
            print("[NG] Data quality test: FAILED")

        return success

    except Exception as e:
        print(f"Data quality test error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_large_scale_data_fetch():
    """大規模データ取得テスト"""
    print("\n=== Large Scale Data Fetch Test ===")

    try:
        import yfinance as yf

        # TOPIX Core30の一部（20銘柄）でテスト
        test_symbols = [
            "7203.T",
            "8306.T",
            "9984.T",
            "6758.T",
            "9432.T",
            "8001.T",
            "6861.T",
            "8058.T",
            "4502.T",
            "7974.T",
            "8411.T",
            "8316.T",
            "8031.T",
            "8053.T",
            "7751.T",
            "6981.T",
            "9983.T",
            "4568.T",
            "6367.T",
            "6954.T",
        ]

        print(f"Testing large scale fetch: {len(test_symbols)} symbols")

        start_time = time.time()
        successful_count = 0
        total_records = 0

        for i, symbol in enumerate(test_symbols):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="7d")  # 1週間データ

                if not hist.empty:
                    successful_count += 1
                    total_records += len(hist)

                # プログレス表示
                if (i + 1) % 5 == 0:
                    progress = (i + 1) / len(test_symbols) * 100
                    print(f"  Progress: {progress:.1f}% ({i + 1}/{len(test_symbols)})")

                time.sleep(0.2)  # API制限対策

            except Exception as e:
                print(f"  Error with {symbol}: {e}")

        fetch_time = time.time() - start_time

        print("\nLarge Scale Fetch Results:")
        print(f"  Successful symbols: {successful_count}/{len(test_symbols)}")
        print(f"  Total records: {total_records}")
        print(f"  Fetch time: {fetch_time:.2f} seconds")
        print(f"  Average time per symbol: {fetch_time/len(test_symbols):.2f} seconds")
        print(f"  Success rate: {successful_count/len(test_symbols)*100:.1f}%")

        # 成功条件
        success_conditions = [
            successful_count >= len(test_symbols) * 0.8,  # 80%以上成功
            fetch_time <= 60,  # 60秒以内
            total_records >= 100,  # 100レコード以上取得
        ]

        success = all(success_conditions)

        if success:
            print("[OK] Large scale data fetch test: PASSED")
            print("  - System ready for 85-symbol ML analysis")
        else:
            print("[NG] Large scale data fetch test: FAILED")
            if successful_count < len(test_symbols) * 0.8:
                print(
                    f"  - Success rate too low: {successful_count/len(test_symbols)*100:.1f}%"
                )
            if fetch_time > 60:
                print(f"  - Fetch time too long: {fetch_time:.1f}s")
            if total_records < 100:
                print(f"  - Insufficient data: {total_records} records")

        return success

    except Exception as e:
        print(f"Large scale data fetch test error: {e}")
        return False


def test_data_consistency():
    """データ整合性テスト"""
    print("\n=== Data Consistency Test ===")

    try:
        import yfinance as yf

        # 信頼性の高い銘柄でテスト
        symbol = "7203.T"  # トヨタ
        print(f"Testing data consistency for {symbol}...")

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="30d")

        if hist.empty:
            print(f"[NG] No data available for {symbol}")
            return False

        print(f"Analyzing {len(hist)} records...")

        consistency_score = 100.0
        issues = []

        # 1. OHLC関係の整合性
        if all(col in hist.columns for col in ["Open", "High", "Low", "Close"]):
            # High >= Low
            invalid_hl = (hist["High"] < hist["Low"]).sum()
            if invalid_hl > 0:
                consistency_score -= 20
                issues.append(f"Invalid High/Low relationship: {invalid_hl} cases")

            # Open, Close が High, Low の範囲内
            ohlc_violations = (
                (hist["Open"] > hist["High"])
                | (hist["Open"] < hist["Low"])
                | (hist["Close"] > hist["High"])
                | (hist["Close"] < hist["Low"])
            ).sum()

            if ohlc_violations > 0:
                consistency_score -= 15
                issues.append(f"OHLC range violations: {ohlc_violations} cases")

        # 2. 価格の妥当性
        for col in ["Open", "High", "Low", "Close"]:
            if col in hist.columns:
                negative_prices = (hist[col] <= 0).sum()
                if negative_prices > 0:
                    consistency_score -= 25
                    issues.append(f"Negative prices in {col}: {negative_prices} cases")

        # 3. 出来高の妥当性
        if "Volume" in hist.columns:
            negative_volume = (hist["Volume"] < 0).sum()
            if negative_volume > 0:
                consistency_score -= 10
                issues.append(f"Negative volume: {negative_volume} cases")

        # 4. 時系列順序
        if not hist.index.is_monotonic_increasing:
            consistency_score -= 20
            issues.append("Non-monotonic time series")

        # 5. データ欠損
        missing_data = hist.isna().sum().sum()
        if missing_data > 0:
            consistency_score -= min(20, missing_data)
            issues.append(f"Missing data points: {missing_data}")

        print("Consistency Analysis Results:")
        print(f"  Consistency Score: {consistency_score:.1f}/100")
        print(f"  Issues Found: {len(issues)}")

        for issue in issues:
            print(f"    - {issue}")

        # 成功条件: 80点以上
        success = consistency_score >= 80.0

        if success:
            print("[OK] Data consistency test: PASSED")
            print("  - Data quality is suitable for ML analysis")
        else:
            print("[NG] Data consistency test: FAILED")
            print("  - Data quality issues may affect ML analysis")

        return success

    except Exception as e:
        print(f"Data consistency test error: {e}")
        return False


def main():
    """メイン実行"""
    print("Real Market Data Validation Test starting...")

    test_results = []

    # テスト実行
    tests = [
        ("Network Connectivity", test_network_connectivity),
        ("Simple Data Quality", test_simple_data_quality),
        ("Data Consistency", test_data_consistency),
        ("Large Scale Data Fetch", test_large_scale_data_fetch),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            success = test_func()
            test_results.append((test_name, success))
        except Exception as e:
            print(f"{test_name} exception: {e}")
            test_results.append((test_name, False))

    # 最終結果
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")

    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)

    for test_name, success in test_results:
        status = "[OK]" if success else "[NG]"
        print(f"  {status} {test_name}")

    success_rate = passed / total
    print(f"\nSuccess Rate: {passed}/{total} ({success_rate:.1%})")

    if success_rate >= 0.75:  # 75%以上で合格
        print("\n[SUCCESS] Real Market Data Validation System: READY")
        print("System is prepared for real data ML analysis testing")

        # 次のステップの推奨
        print("\nNext Recommended Actions:")
        print("  1. Execute real data ML analysis test (85 symbols)")
        print("  2. Run real market portfolio optimization")
        print("  3. Perform real data backtesting")
        print("  4. Conduct 24-hour stability test")

        return True
    else:
        print("\n[FAILED] Some tests failed")
        print("Please resolve issues before proceeding to next steps")

        # 問題解決のヒント
        print("\nTroubleshooting Tips:")
        print("  - Ensure stable internet connection")
        print("  - Check API access permissions")
        print("  - Verify yfinance library installation")
        print("  - Consider using VPN if access is restricted")

        return False


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n{'='*60}")
        print("READY FOR REAL DATA ML ANALYSIS")
        print("Issue #321 - Real data validation completed successfully")
        print(f"{'='*60}")

    exit(0 if success else 1)
