#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Personal - リファクタリング版メインエントリーポイント

使用方法:
  python main.py           # デイトレード推奨（デフォルト）
  python main.py --quick   # 基本分析
  python main.py --web     # Webダッシュボード
  python main.py --help    # 詳細オプション
"""

import os
import sys
from pathlib import Path
from typing import NoReturn


def setup_environment() -> None:
    """環境設定とパス設定を行う"""
    # プロジェクトルート設定
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # Windows環境での文字化け対策
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    if sys.platform == 'win32':
        _configure_windows_encoding()


def _configure_windows_encoding() -> None:
    """Windows環境での文字エンコーディング設定"""
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6以下の場合のフォールバック
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


def _initialize_logging() -> None:
    """ログシステムを初期化する"""
    from daytrade_logging import setup_logging
    debug_mode = '--debug' in sys.argv
    setup_logging(debug=debug_mode)


def _execute_cli() -> int:
    """CLIを実行する"""
    from daytrade_cli import DayTradeCLI
    cli = DayTradeCLI()
    return cli.execute()


def _handle_keyboard_interrupt() -> int:
    """キーボード割り込みを処理する"""
    print("\n🛑 処理を中断しました")
    return 1


def _handle_import_error(error: ImportError) -> int:
    """インポートエラーを処理する"""
    print(f"❌ 必要なモジュールが見つかりません: {error}")
    print("📥 pip install -r requirements.txt を実行してください")
    return 1


def _handle_general_error(error: Exception) -> int:
    """一般的なエラーを処理する"""
    print(f"❌ システムエラーが発生しました: {error}")
    if '--debug' in sys.argv:
        import traceback
        traceback.print_exc()
    return 1


def main() -> int:
    """リファクタリング版メイン関数"""
    try:
        _initialize_logging()
        return _execute_cli()
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt()
    except ImportError as e:
        return _handle_import_error(e)
    except Exception as e:
        return _handle_general_error(e)


def _display_banner() -> None:
    """アプリケーションバナーを表示する"""
    print("🚀 Day Trade Personal - 93%精度AIシステム")
    print("📊 リファクタリング版 v2.0")
    print("=" * 50)


def run() -> NoReturn:
    """アプリケーションを実行する"""
    setup_environment()
    _display_banner()
    sys.exit(main())


if __name__ == "__main__":
    run()