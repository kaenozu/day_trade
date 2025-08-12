#!/usr/bin/env python3
"""
システム停止スクリプト

Issue #320: システム本番稼働準備・設定最適化
安全なシステム停止・クリーンアップ処理
"""

import contextlib
import os
import signal
import sys
import time
from pathlib import Path

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent


class SystemStopper:
    """システム停止管理"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.pid_file = self.project_root / "daytrade.pid"

    def stop_production_system(self, timeout: int = 30) -> bool:
        """本番システム停止"""
        print("本番システム停止開始...")

        # PIDファイル確認
        if not self.pid_file.exists():
            print(
                "PIDファイルが見つかりません - システムは稼働していない可能性があります"
            )
            return True

        try:
            # PID読み込み
            with open(self.pid_file) as f:
                pid = int(f.read().strip())

            print(f"プロセス {pid} に停止シグナル送信...")

            # プロセス存在確認
            if not self._process_exists(pid):
                print("プロセスは既に停止しています")
                self._cleanup_pid_file()
                return True

            # 正常停止シグナル送信（SIGTERM）
            try:
                os.kill(pid, signal.SIGTERM)
                print("SIGTERM シグナル送信完了")
            except OSError as e:
                print(f"シグナル送信エラー: {e}")
                return False

            # 停止待機
            print(f"プロセス停止を待機中... (最大 {timeout} 秒)")

            for i in range(timeout):
                if not self._process_exists(pid):
                    print(f"プロセス正常停止完了 ({i + 1} 秒)")
                    self._cleanup_pid_file()
                    return True

                if i % 5 == 0:  # 5秒ごとに進捗表示
                    print(f"待機中... ({i + 1}/{timeout} 秒)")

                time.sleep(1)

            # タイムアウト - 強制終了
            print("正常停止タイムアウト - 強制終了を実行")
            return self._force_kill(pid)

        except FileNotFoundError:
            print("PIDファイルが読み込めません")
            return False
        except ValueError:
            print("PIDファイルの形式が無効です")
            return False
        except Exception as e:
            print(f"システム停止エラー: {e}")
            return False

    def _process_exists(self, pid: int) -> bool:
        """プロセス存在確認"""
        try:
            # プロセス存在確認（Windowsでも動作）
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _force_kill(self, pid: int) -> bool:
        """強制終了"""
        try:
            print(f"プロセス {pid} を強制終了...")

            if os.name == "nt":  # Windows
                import subprocess

                result = subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True
                )
                if result.returncode == 0:
                    print("強制終了成功 (Windows)")
                    self._cleanup_pid_file()
                    return True
                else:
                    print(f"強制終了失敗: {result.stderr}")
                    return False
            else:  # Unix系
                os.kill(pid, signal.SIGKILL)
                time.sleep(2)  # 少し待機

                if not self._process_exists(pid):
                    print("強制終了成功 (Unix)")
                    self._cleanup_pid_file()
                    return True
                else:
                    print("強制終了失敗")
                    return False

        except Exception as e:
            print(f"強制終了エラー: {e}")
            return False

    def _cleanup_pid_file(self):
        """PIDファイルクリーンアップ"""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                print("PIDファイル削除完了")
        except Exception as e:
            print(f"PIDファイル削除エラー: {e}")

    def stop_all_related_processes(self):
        """関連プロセス全停止"""
        print("関連プロセス検索・停止開始...")

        try:
            if os.name == "nt":  # Windows
                import subprocess

                # Pythonプロセスでdaytradeを実行しているものを検索
                result = subprocess.run(
                    [
                        "wmic",
                        "process",
                        "where",
                        'name="python.exe" AND CommandLine LIKE "%daytrade%"',
                        "get",
                        "ProcessId,CommandLine",
                        "/format:csv",
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    lines = [
                        line.strip()
                        for line in result.stdout.split("\n")
                        if line.strip()
                    ]
                    for line in lines[1:]:  # ヘッダーをスキップ
                        if line and "daytrade" in line.lower():
                            parts = line.split(",")
                            if len(parts) >= 3:
                                try:
                                    pid = int(parts[2])  # ProcessId
                                    print(f"関連プロセス発見: PID {pid}")
                                    subprocess.run(
                                        ["taskkill", "/F", "/PID", str(pid)],
                                        capture_output=True,
                                    )
                                except ValueError:
                                    continue

            else:  # Unix系
                import subprocess

                # pgrep で daytrade 関連プロセス検索
                result = subprocess.run(
                    ["pgrep", "-f", "daytrade"], capture_output=True, text=True
                )

                if result.returncode == 0:
                    pids = result.stdout.strip().split("\n")
                    for pid_str in pids:
                        if pid_str:
                            try:
                                pid = int(pid_str)
                                print(f"関連プロセス発見: PID {pid}")
                                os.kill(pid, signal.SIGTERM)
                                time.sleep(1)

                                # まだ残っていれば強制終了
                                if self._process_exists(pid):
                                    os.kill(pid, signal.SIGKILL)

                            except (ValueError, OSError):
                                continue

            print("関連プロセス停止処理完了")

        except Exception as e:
            print(f"関連プロセス停止エラー: {e}")

    def cleanup_resources(self):
        """リソースクリーンアップ"""
        print("システムリソースクリーンアップ開始...")

        cleanup_tasks = [
            ("PIDファイル", self._cleanup_pid_file),
            ("一時ファイル", self._cleanup_temp_files),
            ("ログファイル整理", self._cleanup_log_files),
            ("キャッシュクリア", self._cleanup_cache_files),
        ]

        for task_name, task_func in cleanup_tasks:
            try:
                task_func()
                print(f"{task_name}: 完了")
            except Exception as e:
                print(f"{task_name}: エラー - {e}")

        print("リソースクリーンアップ完了")

    def _cleanup_temp_files(self):
        """一時ファイルクリーンアップ"""
        temp_patterns = ["*.tmp", "*.temp", "*~", ".DS_Store"]

        for pattern in temp_patterns:
            for temp_file in self.project_root.glob(f"**/{pattern}"):
                if temp_file.is_file():
                    temp_file.unlink()

    def _cleanup_log_files(self):
        """ログファイル整理"""
        log_dir = self.project_root / "logs"
        if not log_dir.exists():
            return

        # 7日以上古いログファイルを削除
        import time

        current_time = time.time()
        week_ago = current_time - (7 * 24 * 3600)

        for log_file in log_dir.glob("**/*.log"):
            if log_file.stat().st_mtime < week_ago:
                with contextlib.suppress(Exception):
                    log_file.unlink()

    def _cleanup_cache_files(self):
        """キャッシュファイルクリーンアップ"""
        cache_dirs = [
            self.project_root / "__pycache__",
            self.project_root / "src" / "day_trade" / "__pycache__",
            self.project_root / ".pytest_cache",
        ]

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                import shutil

                with contextlib.suppress(Exception):
                    shutil.rmtree(cache_dir)


def main():
    """メイン実行関数"""
    print("Day Trade System Stopper")
    print("=" * 40)

    stopper = SystemStopper()

    # コマンドライン引数確認
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "force":
            print("強制停止モード")
            stopper.stop_all_related_processes()
            stopper.cleanup_resources()
            return 0

        elif command == "clean":
            print("クリーンアップのみ実行")
            stopper.cleanup_resources()
            return 0

        elif command == "help":
            print("使用方法:")
            print("  python stop_system.py        - 正常停止")
            print("  python stop_system.py force  - 強制停止")
            print("  python stop_system.py clean  - クリーンアップのみ")
            print("  python stop_system.py help   - このヘルプを表示")
            return 0

    # 正常停止処理
    try:
        success = stopper.stop_production_system(timeout=30)

        if success:
            print("システム正常停止完了")
            stopper.cleanup_resources()
            return 0
        else:
            print("正常停止に失敗しました")

            # 強制停止を提案
            user_input = input("強制停止を実行しますか? (y/N): ").lower().strip()
            if user_input in ["y", "yes"]:
                stopper.stop_all_related_processes()
                stopper.cleanup_resources()
                print("強制停止完了")
                return 0
            else:
                print("停止処理を中断しました")
                return 1

    except KeyboardInterrupt:
        print("\\n停止処理が中断されました")
        return 1
    except Exception as e:
        print(f"停止処理エラー: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
