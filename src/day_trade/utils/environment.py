# -*- coding: utf-8 -*-
"""
実行環境設定モジュール
"""

import os
import sys
from pathlib import Path

def _configure_windows_encoding() -> None:
    """Windows環境での文字エンコーディングをUTF-8に設定する"""
    if sys.platform != 'win32':
        return
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (TypeError, AttributeError):
        # Python 3.6以前や一部の非対話的環境向けのフォールバック
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

def setup_environment() -> None:
    """
    プロジェクトの実行環境を設定します。

    - プロジェクトルートをPythonパスに追加
    - Windows環境でのコンソールの文字エンコーディングをUTF-8に設定
    """
    # このファイルの場所 (src/day_trade/utils) からプロジェクトルートを取得
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    _configure_windows_encoding()
    os.environ['PYTHONIOENCODING'] = 'utf-8'
