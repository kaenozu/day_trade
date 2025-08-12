"""
Windows環境でのコンソール表示問題修正ユーティリティ
Issue #205: Windows環境でのdaytrade.py実行時エラー対応
"""

import os
import sys
from contextlib import contextmanager

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="windows_console_fix")


def setup_windows_console():
    """
    Windows環境でのコンソール設定を初期化

    - UTF-8エンコーディングの設定
    - ANSIエスケープシーケンスの有効化
    - cp932コーデックエラーの回避
    """
    if sys.platform != "win32":
        return

    try:
        # 標準出力・標準エラー出力をUTF-8に再設定
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

        # 環境変数でPythonのエンコーディングを強制設定
        os.environ["PYTHONIOENCODING"] = "utf-8:replace"

        # Windows ANSIエスケープシーケンス有効化
        try:
            import colorama

            colorama.init(autoreset=True)
        except ImportError:
            # coloramaがない場合はWindows APIで直接設定
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                mode = ctypes.c_ulong()
                kernel32.GetConsoleMode(handle, ctypes.byref(mode))
                mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(handle, mode)
            except Exception:
                pass  # ANSIカラー無効でも動作継続

        logger.info(
            "Windows コンソール環境を初期化しました",
            extra={
                "platform": "win32",
                "encoding": "utf-8",
                "ansi_enabled": True,
            },
        )

    except Exception as e:
        logger.warning(
            "Windows コンソール初期化で警告",
            extra={
                "error": str(e),
                "fallback": "デフォルト設定で継続",
            },
        )


@contextmanager
def safe_console_context():
    """
    安全なコンソール出力コンテキストマネージャー

    Windows環境でのエンコーディングエラーを回避して
    コンソール出力を実行する
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        if sys.platform == "win32":
            # Windows環境では安全なエンコーディングでラップ
            import io

            # UTF-8でエラー時は置換するTextIOWrapperを作成
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "buffer"):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

        yield

    except UnicodeEncodeError as e:
        # エンコーディングエラーの場合はASCII版に置換
        error_msg = str(e).encode("ascii", errors="replace").decode("ascii")
        print(f"Encoding error (replaced): {error_msg}")

    finally:
        # 元の設定に復元
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def create_safe_live_context():
    """
    Rich Live表示の競合を回避する安全なコンテキストマネージャーを作成

    Returns:
        安全なLiveコンテキストマネージャー
    """

    @contextmanager
    def safe_live_context(*args, **kwargs):
        # 活動中のLiveインスタンス数を追跡
        _active_live_count = getattr(safe_live_context, "_active_count", 0)

        if _active_live_count > 0:
            # 既にLiveが活動中の場合は何もしない（競合回避）
            logger.debug("Rich Live競合回避: 既存のLiveインスタンスが活動中")
            yield None
            return

        try:
            import os

            from rich.live import Live

            # テスト環境では無効化
            if os.environ.get("PYTEST_CURRENT_TEST"):
                # テスト中はダミーLiveオブジェクトを返す
                class DummyLive:
                    def update(self, *args, **kwargs):
                        pass

                    def refresh(self):
                        pass

                yield DummyLive()
                return

            # Liveインスタンスカウンターを増加
            safe_live_context._active_count = getattr(safe_live_context, "_active_count", 0) + 1

            # Liveコンテキストを開始
            with Live(*args, **kwargs) as live:
                yield live

        except Exception as e:
            logger.warning("Rich Live表示エラー", error=str(e), fallback="通常出力で継続")
            yield None

        finally:
            # Liveインスタンスカウンターを減少
            safe_live_context._active_count = max(
                0, getattr(safe_live_context, "_active_count", 1) - 1
            )

    return safe_live_context


def safe_rich_print(content, **kwargs):
    """
    Rich printの安全な実行

    Args:
        content: 出力内容
        **kwargs: rich.printのオプション
    """
    try:
        from rich import print as rich_print

        if sys.platform == "win32":
            # Windows環境では文字列をASCII互換に変換
            if isinstance(content, str):
                # Unicode文字をASCII互換文字に置換
                content = content.replace("✓", "OK").replace("✗", "NG").replace("●", "*")
                # その他の特殊文字も置換
                content = content.encode("ascii", errors="replace").decode("ascii")

        rich_print(content, **kwargs)

    except UnicodeEncodeError:
        # Rich printが失敗した場合は通常のprintにフォールバック
        print(content)
    except ImportError:
        # Richがインストールされていない場合
        print(content)
    except Exception as e:
        logger.debug(f"Rich print エラー: {e}")
        print(content)


def ensure_ascii_output(text: str) -> str:
    """
    テキストをWindows環境で安全に出力できるASCII互換形式に変換

    Args:
        text: 変換対象テキスト

    Returns:
        ASCII互換テキスト
    """
    if sys.platform != "win32":
        return text

    # よく使用されるUnicode文字を置換
    replacements = {
        "✓": "OK",
        "✗": "NG",
        "●": "*",
        "○": "o",
        "◆": "*",
        "◇": "o",
        "→": "->",
        "←": "<-",
        "↑": "^",
        "↓": "v",
        "■": "#",
        "□": "_",
        "※": "*",
        "…": "...",
    }

    result = text
    for unicode_char, ascii_char in replacements.items():
        result = result.replace(unicode_char, ascii_char)

    # 残りの非ASCII文字を置換
    result = result.encode("ascii", errors="replace").decode("ascii")

    return result


# Windows環境での初期化をモジュールインポート時に実行
if __name__ != "__main__":
    setup_windows_console()
