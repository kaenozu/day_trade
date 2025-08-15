#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encoding Utilities - エンコーディング関連ユーティリティ

Windows環境での文字化け対策などの共通機能
"""

import sys
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def setup_windows_encoding():
    """
    Windows環境での文字化け対策を設定する

    Returns:
        bool: 設定が成功したかどうか
    """
    if sys.platform != 'win32':
        return True  # Windows以外では何もしない

    try:
        # 環境変数設定
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # stdout, stderrの再設定
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        else:
            # 古いPythonバージョンの場合のフォールバック
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

        logger.debug("Windows encoding setup completed successfully")
        return True

    except Exception as e:
        logger.warning(f"Windows encoding setup failed: {e}")
        return False

def ensure_utf8_encoding():
    """
    UTF-8エンコーディングを確保する

    Returns:
        str: 現在のエンコーディング情報
    """
    encoding_info = {
        'stdout': getattr(sys.stdout, 'encoding', 'unknown'),
        'stderr': getattr(sys.stderr, 'encoding', 'unknown'),
        'default': sys.getdefaultencoding(),
        'filesystem': sys.getfilesystemencoding()
    }

    if sys.platform == 'win32':
        setup_windows_encoding()

    return encoding_info

def safe_print(message: str, encoding: Optional[str] = None):
    """
    エンコーディングを考慮した安全な印刷

    Args:
        message: 印刷するメッセージ
        encoding: 使用するエンコーディング（Noneの場合は自動）
    """
    if encoding is None:
        encoding = 'utf-8' if sys.platform == 'win32' else sys.stdout.encoding

    try:
        print(message)
    except UnicodeEncodeError:
        # エンコーディングエラーの場合はログに記録
        logger.error(f"Encoding error printing message: {repr(message)}")
        print(message.encode(encoding, errors='replace').decode(encoding))

def get_system_encoding_info() -> dict:
    """
    システムのエンコーディング情報を取得

    Returns:
        dict: エンコーディング情報
    """
    return {
        'platform': sys.platform,
        'stdout_encoding': getattr(sys.stdout, 'encoding', None),
        'stderr_encoding': getattr(sys.stderr, 'encoding', None),
        'default_encoding': sys.getdefaultencoding(),
        'filesystem_encoding': sys.getfilesystemencoding(),
        'locale_encoding': getattr(sys.stdout, 'encoding', None),
        'python_io_encoding': os.environ.get('PYTHONIOENCODING', 'not_set')
    }

# モジュール読み込み時に自動でWindows対応を実行
if __name__ != "__main__":
    setup_windows_encoding()