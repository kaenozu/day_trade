#!/usr/bin/env python3
"""
品質ゲートシステムの基本テスト
Issue #415対応
"""

import asyncio
import subprocess
import sys
from pathlib import Path


async def test_basic_quality_gates():
    """基本品質ゲートテスト"""
    print("=== CI/CD品質ゲートシステム基本テスト ===")

    test_results = {
        "mypy_type_check": False,
        "bandit_security_check": False,
        "test_coverage": False,
        "dependency_check": False,
        "overall_status": False
    }

    # 1. MyPy型チェック
    print("\n1. MyPy型チェック実行中...")
    try:
        result = subprocess.run(
            ["mypy", "src/day_trade/core", "--ignore-missing-imports", "--show-error-codes"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("   [完了] MyPy型チェック: PASS")
            test_results["mypy_type_check"] = True
        else:
            print("   [警告] MyPy型チェック: FAIL (部分的)")
            error_count = result.stdout.count(": error:")
            print(f"      型エラー数: {error_count}")

    except Exception as e:
        print(f"   [エラー] MyPy実行エラー: {e}")

    # 2. Banditセキュリティチェック
    print("\n2. Banditセキュリティチェック実行中...")
    try:
        result = subprocess.run(
            ["bandit", "-r", "src/day_trade/core", "-q"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("   [完了] Banditセキュリティチェック: PASS")
            test_results["bandit_security_check"] = True
        else:
            print("   [警告] Banditセキュリティチェック: 問題検出")

    except Exception as e:
        print(f"   [エラー] Bandit実行エラー: {e}")

    # 3. テストカバレッジチェック
    print("\n3. テストカバレッジチェック実行中...")
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--cov=src/day_trade/core", "--cov-report=term", "-q", "tests/", "--maxfail=1"],
            capture_output=True,
            text=True,
            timeout=90
        )

        # カバレッジパーセンテージを抽出
        output = result.stdout
        for line in output.split('\n'):
            if 'TOTAL' in line and '%' in line:
                coverage_str = line.split()[-1].replace('%', '')
                try:
                    coverage_pct = float(coverage_str)
                    print(f"   [統計] テストカバレッジ: {coverage_pct}%")

                    if coverage_pct >= 30.0:  # 30%閾値
                        print("   [完了] カバレッジ閾値: PASS (≥30%)")
                        test_results["test_coverage"] = True
                    else:
                        print("   [警告] カバレッジ閾値: FAIL (<30%)")
                    break
                except ValueError:
                    pass

    except Exception as e:
        print(f"   [エラー] テストカバレッジチェック実行エラー: {e}")

    # 4. 依存関係チェック
    print("\n4. 依存関係健全性チェック実行中...")
    try:
        # pip-auditで脆弱性チェック
        result = subprocess.run(
            ["pip-audit", "--desc", "--format=json"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("   [完了] 依存関係脆弱性: 問題なし")
            test_results["dependency_check"] = True
        else:
            print("   [警告] 依存関係に脆弱性検出")

    except Exception as e:
        print(f"   [エラー] 依存関係チェック実行エラー: {e}")

    # 5. 総合評価
    print("\n5. 総合品質評価...")

    passed_tests = sum(test_results.values())
    total_tests = len(test_results) - 1  # overall_statusを除く

    quality_score = (passed_tests / total_tests) * 100

    print(f"   品質テスト通過: {passed_tests}/{total_tests}")
    print(f"   品質スコア: {quality_score:.1f}/100")

    if quality_score >= 75:
        print("   [合格] 総合評価: EXCELLENT")
        test_results["overall_status"] = True
    elif quality_score >= 50:
        print("   [可] 総合評価: ACCEPTABLE")
        test_results["overall_status"] = True
    else:
        print("   [不可] 総合評価: NEEDS IMPROVEMENT")

    # 6. 推奨事項
    print("\n6. 改善推奨事項:")
    recommendations = []

    if not test_results["mypy_type_check"]:
        recommendations.append("- 型注釈の追加とMyPyエラーの修正")

    if not test_results["bandit_security_check"]:
        recommendations.append("- セキュリティ脆弱性の修正")

    if not test_results["test_coverage"]:
        recommendations.append("- テストカバレッジを30%以上に向上")

    if not test_results["dependency_check"]:
        recommendations.append("- 依存関係の脆弱性対応")

    if recommendations:
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("   - 現在、すべての品質基準を満たしています")

    print("\n=== テスト完了 ===")

    return test_results

# メイン実行
if __name__ == "__main__":
    asyncio.run(test_basic_quality_gates())
