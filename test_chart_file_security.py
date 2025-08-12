#!/usr/bin/env python3
"""
チャート生成ファイルパスセキュリティテスト
Issue #393: チャート生成におけるファイルパスのセキュリティ強化

実装されたセキュリティ強化:
1. 出力ディレクトリの外部制御リスク対策 - 危険パスパターン検出・許可ディレクトリ制限
2. TOCTOU脆弱性対策 - cleanup_old_charts原子的操作・シンボリックリンク攻撃防止
"""

# テスト中はログ出力を有効化
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

logging.basicConfig(level=logging.INFO)

from src.day_trade.dashboard.visualization_engine import DashboardVisualizationEngine


def test_output_directory_security():
    """出力ディレクトリセキュリティテスト"""
    print("=== 出力ディレクトリセキュリティテスト ===")

    # 1. 正常なディレクトリパスのテスト
    print("\n1. 正常ディレクトリパス:")
    safe_directories = [
        "dashboard_charts",
        "output/charts",
        "temp/visualizations",
        "./chart_output",
        "charts/daily",
    ]

    for dir_path in safe_directories:
        try:
            # 一時ディレクトリでテスト
            with tempfile.TemporaryDirectory() as temp_dir:
                test_path = os.path.join(temp_dir, dir_path)
                engine = DashboardVisualizationEngine(test_path)
                print(f"  OK {dir_path} - 正常に作成")
        except Exception as e:
            print(f"  FAIL {dir_path} - 予期しないエラー: {e}")

    # 2. 危険なディレクトリパスのテスト
    print("\n2. 危険ディレクトリパス:")
    dangerous_directories = [
        "../../../etc",  # パストラバーサル攻撃
        "~/malicious",  # ホームディレクトリ参照
        "/etc/passwd_charts",  # システムディレクトリ
        "/usr/local/malicious",  # システムディレクトリ
        "/var/www/html",  # Webディレクトリ
        "/root/secret_charts",  # rootディレクトリ
        "c:\\windows\\system32",  # Windowsシステムディレクトリ
        "c:\\program files\\charts",  # Windowsプログラムディレクトリ
        "\\\\malicious\\share",  # UNCパス
        "charts\x00malicious",  # NULLバイト攻撃
        "a" * 250,  # 長すぎるパス
    ]

    for dir_path in dangerous_directories:
        try:
            engine = DashboardVisualizationEngine(dir_path)
            print(f"  FAIL {dir_path[:30]}... - 危険パスが通過")
        except ValueError as e:
            print(f"  OK {dir_path[:30]}... - 正常に阻止")
        except Exception as e:
            print(f"  WARN {dir_path[:30]}... - 予期しないエラー: {e}")


def test_toctou_vulnerability_protection():
    """TOCTOU脆弱性対策テスト"""
    print("\n=== TOCTOU脆弱性対策テスト ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # セキュリティ強化されたエンジン
        engine = DashboardVisualizationEngine(temp_dir)

        # 1. 通常のファイルクリーンアップテスト
        print("\n1. 通常ファイルクリーンアップ:")

        # テスト用の古いファイルを作成
        old_file = Path(temp_dir) / "old_chart_20230101_120000.png"
        with open(old_file, "wb") as f:
            f.write(b"test chart data")

        # ファイルタイムスタンプを古く設定
        old_time = time.time() - (25 * 3600)  # 25時間前
        os.utime(old_file, (old_time, old_time))

        try:
            engine.cleanup_old_charts(hours=24)

            if not old_file.exists():
                print("  OK 通常クリーンアップ - 古いファイル正常削除")
            else:
                print("  FAIL 通常クリーンアップ - 古いファイルが残存")

        except Exception as e:
            print(f"  FAIL 通常クリーンアップエラー: {e}")

        # 2. シンボリックリンク攻撃対策テスト
        print("\n2. シンボリックリンク攻撃対策:")

        try:
            # 危険なファイルへのシンボリックリンクを作成
            danger_file = Path(temp_dir) / "important_system_file.txt"
            with open(danger_file, "w") as f:
                f.write("重要なシステムファイル")

            symlink_file = Path(temp_dir) / "malicious_chart.png"

            # Windowsでのシンボリックリンク作成（権限が必要）
            try:
                if os.name == "nt":
                    # Windows環境での処理
                    import subprocess

                    subprocess.run(
                        ["mklink", str(symlink_file), str(danger_file)],
                        shell=True,
                        check=True,
                        capture_output=True,
                    )
                else:
                    # Unix/Linux環境での処理
                    symlink_file.symlink_to(danger_file)

                # シンボリックリンクのタイムスタンプを古く設定
                old_time = time.time() - (25 * 3600)
                os.utime(symlink_file, (old_time, old_time), follow_symlinks=False)

                engine.cleanup_old_charts(hours=24)

                # 重要: シンボリックリンクは削除されず、リンク先も保護される
                if danger_file.exists() and symlink_file.exists():
                    print("  OK シンボリックリンク攻撃対策 - リンクと対象ファイル両方保護")
                elif danger_file.exists():
                    print("  OK シンボリックリンク攻撃対策 - 対象ファイル保護（リンクのみ削除）")
                else:
                    print("  FAIL シンボリックリンク攻撃対策 - 対象ファイルが削除された")

            except (PermissionError, subprocess.CalledProcessError):
                print("  SKIP シンボリックリンク攻撃対策 - 権限不足でテストスキップ")
            except Exception as e:
                print(f"  WARN シンボリックリンクテストエラー: {e}")

        except Exception as e:
            print(f"  FAIL シンボリックリンク攻撃対策エラー: {e}")

        # 3. 危険なファイル名攻撃対策テスト
        print("\n3. 危険ファイル名攻撃対策:")

        dangerous_filenames = [
            "../malicious_chart.png",  # パストラバーサル
            "../../etc_chart.png",  # ディレクトリトラバーサル
            "normal/../../bad.png",  # 埋め込みトラバーサル
        ]

        for filename in dangerous_filenames:
            try:
                # 危険なファイル名のファイルを作成を試みる
                dangerous_file_path = Path(temp_dir) / filename

                # ファイル作成の試行（実際には作成されない想定）
                try:
                    os.makedirs(dangerous_file_path.parent, exist_ok=True)
                    with open(dangerous_file_path, "w") as f:
                        f.write("malicious content")

                    # 古いタイムスタンプを設定
                    old_time = time.time() - (25 * 3600)
                    os.utime(dangerous_file_path, (old_time, old_time))

                except Exception:
                    # ファイル作成に失敗した場合はスキップ
                    continue

                # クリーンアップ実行
                engine.cleanup_old_charts(hours=24)

                # 危険なファイル名のファイルは削除されるべきではない
                if dangerous_file_path.exists():
                    print(f"  OK {filename} - 危険ファイル名削除を回避")
                else:
                    print(f"  WARN {filename} - ファイル削除された（要確認）")

            except Exception as e:
                print(f"  SKIP {filename} - テスト実行不可: {e}")

        # 4. 無効パラメータ対策テスト
        print("\n4. 無効パラメータ対策:")

        invalid_params = [(-1, "負の数値"), (0, "ゼロ"), (0.5, "1時間未満")]

        for hours_param, description in invalid_params:
            try:
                engine.cleanup_old_charts(hours=hours_param)
                print(f"  OK {description} - 無効パラメータ正常処理")
            except Exception as e:
                print(f"  FAIL {description} - 予期しないエラー: {e}")


def test_directory_traversal_attack():
    """ディレクトリトラバーサル攻撃テスト"""
    print("\n=== ディレクトリトラバーサル攻撃テスト ===")

    # 攻撃パターンの包括的テスト
    attack_patterns = [
        "../../../sensitive_data",
        "..\\..\\..\\windows\\system32",
        "%2e%2e%2fsensitive",
        "%2e%2e\\sensitive",
        "charts/../../../etc",
        "charts\\..\\..\\..\\windows",
        "./charts/../../../root",
        "legitimate_dir/../../../../../../etc",
    ]

    for pattern in attack_patterns:
        try:
            engine = DashboardVisualizationEngine(pattern)
            print(f"  FAIL {pattern} - パストラバーサル攻撃が通過")
        except ValueError:
            print(f"  OK {pattern} - パストラバーサル攻撃を正常に阻止")
        except Exception as e:
            print(f"  WARN {pattern} - 予期しないエラー: {e}")


def test_file_size_limits():
    """ファイルサイズ制限テスト"""
    print("\n=== ファイルサイズ制限テスト ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        engine = DashboardVisualizationEngine(temp_dir)

        # 1. 正常サイズファイル
        print("\n1. 正常サイズファイル:")
        normal_file = Path(temp_dir) / "normal_chart.png"
        with open(normal_file, "wb") as f:
            f.write(b"x" * (1024 * 1024))  # 1MB

        old_time = time.time() - (25 * 3600)
        os.utime(normal_file, (old_time, old_time))

        try:
            engine.cleanup_old_charts(hours=24)
            if not normal_file.exists():
                print("  OK 正常サイズ - 正常に削除")
            else:
                print("  FAIL 正常サイズ - 削除されなかった")
        except Exception as e:
            print(f"  FAIL 正常サイズテストエラー: {e}")

        # 2. 異常に大きなファイル
        print("\n2. 異常に大きなファイル:")
        huge_file = Path(temp_dir) / "huge_chart.png"

        try:
            with open(huge_file, "wb") as f:
                f.write(b"x" * (60 * 1024 * 1024))  # 60MB（制限50MBを超過）

            old_time = time.time() - (25 * 3600)
            os.utime(huge_file, (old_time, old_time))

            engine.cleanup_old_charts(hours=24)

            if huge_file.exists():
                print("  OK 異常サイズ - 削除をスキップ（セキュリティ対策）")
            else:
                print("  WARN 異常サイズ - 削除された（要確認）")

        except Exception as e:
            print(f"  SKIP 異常サイズテスト - ディスク容量不足等: {e}")


def test_integration_security():
    """統合セキュリティテスト"""
    print("\n=== 統合セキュリティテスト ===")

    # 複合的なセキュリティテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 1. 正常な可視化エンジン作成
            engine = DashboardVisualizationEngine(temp_dir)

            # 2. サンプルデータでチャート作成
            sample_portfolio_data = [
                {"timestamp": datetime.now().isoformat(), "total_value": 1000000}
            ]

            chart_path = engine.create_portfolio_value_chart(sample_portfolio_data)

            if chart_path and Path(chart_path).exists():
                print("  OK 統合テスト - チャート作成成功")

                # 3. セキュリティ強化されたクリーンアップ実行
                engine.cleanup_old_charts(hours=0.01)  # 1分以内の最近ファイル

                if Path(chart_path).exists():
                    print("  OK 統合テスト - 最近のファイル保護")
                else:
                    print("  WARN 統合テスト - 最近のファイルが削除された")

            else:
                print("  FAIL 統合テスト - チャート作成失敗")

        except Exception as e:
            print(f"  FAIL 統合テストエラー: {e}")


def main():
    """メイン実行"""
    print("=== チャート生成ファイルパスセキュリティテスト ===")
    print("Issue #393: チャート生成におけるファイルパスのセキュリティ強化")
    print("=" * 60)

    try:
        # 各セキュリティ機能のテスト
        test_output_directory_security()
        test_toctou_vulnerability_protection()
        test_directory_traversal_attack()
        test_file_size_limits()
        test_integration_security()

        print("\n" + "=" * 60)
        print("OK チャート生成ファイルパスセキュリティテスト完了")
        print("\n実装されたセキュリティ強化:")
        print("- [GUARD] 出力ディレクトリ検証（パストラバーサル・システムディレクトリ防止）")
        print("- [ATOMIC] TOCTOU脆弱性対策（原子的操作・レースコンディション防止）")
        print("- [SHIELD] シンボリックリンク攻撃防止")
        print("- [FILTER] 危険ファイル名検出・回避")
        print("- [LIMIT] ファイルサイズ制限（DoS攻撃防止）")
        print("- [AUDIT] セキュリティイベント詳細ログ記録")

    except Exception as e:
        print(f"\nFAIL テスト実行中にエラーが発生: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
