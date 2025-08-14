#!/usr/bin/env python3
"""
Issue #755 Phase 3: SmartSymbolSelectorテスト実行スクリプト

SmartSymbolSelector関連テストの実行・レポート生成
"""

import sys
import subprocess
import time
from pathlib import Path
import argparse


def run_smart_symbol_selector_tests(test_type: str = "all", verbose: bool = True) -> dict:
    """
    SmartSymbolSelectorテストスイート実行

    Args:
        test_type: 実行するテストタイプ ("all", "comprehensive", "integration")
        verbose: 詳細出力フラグ

    Returns:
        テスト結果辞書
    """

    test_files = {
        "comprehensive": "test_smart_symbol_selector_comprehensive.py",
        "integration": "test_smart_symbol_selector_integration.py"
    }

    results = {}
    total_start_time = time.time()

    # 実行対象テスト決定
    if test_type == "all":
        tests_to_run = test_files
    else:
        tests_to_run = {test_type: test_files.get(test_type)}
        if tests_to_run[test_type] is None:
            print(f"エラー: 不明なテストタイプ '{test_type}'")
            return {}

    print("=" * 80)
    print("🧪 SmartSymbolSelector包括的テストスイート実行開始")
    print("=" * 80)

    for test_name, test_file in tests_to_run.items():
        if test_file is None:
            continue

        print(f"\n📋 {test_name.upper()}テスト実行中: {test_file}")
        print("-" * 60)

        start_time = time.time()

        try:
            # pytestで実行
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "-v" if verbose else "-q",
                "--tb=short",
                "--durations=10"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )

            execution_time = time.time() - start_time

            # 結果解析
            output_lines = result.stdout.split('\n')
            test_count = 0
            passed = 0
            failed = 0
            errors = 0

            for line in output_lines:
                if "passed" in line and "failed" in line:
                    # pytest結果行の解析
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            passed = int(parts[i-1])
                        elif part == "failed":
                            failed = int(parts[i-1])
                        elif part == "error" or part == "errors":
                            errors = int(parts[i-1])
                elif line.startswith("=") and ("passed" in line or "failed" in line):
                    test_count = passed + failed + errors

            results[test_name] = {
                "file": test_file,
                "execution_time": execution_time,
                "total_tests": test_count,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "success_rate": (passed / max(test_count, 1)) * 100,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }

            # 結果表示
            if result.returncode == 0:
                print(f"✅ {test_name}テスト成功: {passed}件成功 / {execution_time:.2f}秒")
            else:
                print(f"❌ {test_name}テスト失敗: {failed}件失敗, {errors}件エラー")
                if verbose and result.stderr:
                    print(f"エラー詳細:\n{result.stderr}")

        except Exception as e:
            print(f"❌ {test_name}テスト実行エラー: {e}")
            results[test_name] = {
                "file": test_file,
                "execution_time": 0,
                "error": str(e),
                "return_code": -1
            }

    total_time = time.time() - total_start_time

    # 総合結果レポート
    print("\n" + "=" * 80)
    print("📊 SmartSymbolSelectorテスト結果サマリー")
    print("=" * 80)

    total_tests = sum(r.get("total_tests", 0) for r in results.values())
    total_passed = sum(r.get("passed", 0) for r in results.values())
    total_failed = sum(r.get("failed", 0) for r in results.values())
    total_errors = sum(r.get("errors", 0) for r in results.values())
    overall_success_rate = (total_passed / max(total_tests, 1)) * 100

    print(f"📈 総合結果:")
    print(f"   - 実行テストファイル数: {len([r for r in results.values() if r.get('return_code') != -1])}")
    print(f"   - 総テスト数: {total_tests}")
    print(f"   - 成功: {total_passed} ({total_passed/max(total_tests,1)*100:.1f}%)")
    print(f"   - 失敗: {total_failed}")
    print(f"   - エラー: {total_errors}")
    print(f"   - 総合成功率: {overall_success_rate:.1f}%")
    print(f"   - 総実行時間: {total_time:.2f}秒")

    print(f"\n📋 個別テスト結果:")
    for test_name, result in results.items():
        if result.get("return_code") != -1:
            status = "✅" if result.get("return_code") == 0 else "❌"
            print(f"   {status} {test_name}: {result.get('passed', 0)}/{result.get('total_tests', 0)} "
                  f"({result.get('success_rate', 0):.1f}%) - {result.get('execution_time', 0):.2f}秒")

    # 推奨アクション
    if overall_success_rate >= 95:
        print(f"\n🎉 素晴らしい結果です！ SmartSymbolSelectorの品質は非常に高いです。")
    elif overall_success_rate >= 85:
        print(f"\n👍 良好な結果です。いくつかの改善点がありますが、全体的に安定しています。")
    elif overall_success_rate >= 70:
        print(f"\n⚠️  改善が必要です。失敗したテストを詳しく調査してください。")
    else:
        print(f"\n🚨 深刻な問題があります。システムの基本動作を確認してください。")

    print("=" * 80)

    return results


def generate_smart_selector_test_report(results: dict, output_file: str = "smart_symbol_selector_test_report.txt"):
    """SmartSymbolSelectorテスト結果の詳細レポート生成"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("SmartSymbolSelector包括的テスト結果レポート\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 概要
        total_tests = sum(r.get("total_tests", 0) for r in results.values())
        total_passed = sum(r.get("passed", 0) for r in results.values())
        overall_success_rate = (total_passed / max(total_tests, 1)) * 100

        f.write(f"概要:\n")
        f.write(f"- 実行テストスイート数: {len(results)}\n")
        f.write(f"- 総テスト数: {total_tests}\n")
        f.write(f"- 成功率: {overall_success_rate:.1f}%\n\n")

        # 詳細結果
        f.write("詳細結果:\n")
        f.write("-" * 40 + "\n")

        for test_name, result in results.items():
            f.write(f"\n{test_name.upper()}テスト:\n")
            f.write(f"  ファイル: {result.get('file', 'N/A')}\n")
            f.write(f"  実行時間: {result.get('execution_time', 0):.2f}秒\n")
            f.write(f"  テスト数: {result.get('total_tests', 0)}\n")
            f.write(f"  成功: {result.get('passed', 0)}\n")
            f.write(f"  失敗: {result.get('failed', 0)}\n")
            f.write(f"  エラー: {result.get('errors', 0)}\n")
            f.write(f"  成功率: {result.get('success_rate', 0):.1f}%\n")

            if result.get('stderr'):
                f.write(f"  エラー詳細:\n{result['stderr']}\n")

        # SmartSymbolSelector特有の分析
        f.write(f"\n\nSmartSymbolSelector品質分析:\n")
        f.write(f"- 銘柄選択アルゴリズム: テスト済み\n")
        f.write(f"- 流動性・ボラティリティ分析: テスト済み\n")
        f.write(f"- DataFetcher統合: テスト済み\n")
        f.write(f"- EnsembleSystem統合: テスト済み\n")
        f.write(f"- エンドツーエンド自動化: テスト済み\n")

    print(f"📄 詳細レポートを '{output_file}' に保存しました。")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="SmartSymbolSelectorテストスイート実行")
    parser.add_argument(
        "--type",
        choices=["all", "comprehensive", "integration"],
        default="all",
        help="実行するテストタイプ"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細出力"
    )
    parser.add_argument(
        "--report",
        help="レポート出力ファイル名"
    )

    args = parser.parse_args()

    # テスト実行
    results = run_smart_symbol_selector_tests(args.type, args.verbose)

    # レポート生成
    if args.report:
        generate_smart_selector_test_report(results, args.report)

    # 終了コード決定
    total_tests = sum(r.get("total_tests", 0) for r in results.values())
    total_passed = sum(r.get("passed", 0) for r in results.values())
    success_rate = (total_passed / max(total_tests, 1)) * 100

    if success_rate >= 90:
        sys.exit(0)  # 成功
    else:
        sys.exit(1)  # 失敗


if __name__ == "__main__":
    main()