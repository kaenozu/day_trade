#!/usr/bin/env python3
"""
Day Trade Personal - 初期化モジュール

システム初期化とWindows環境対応
"""

import os
import sys
import locale

class SystemInitializer:
    """システム初期化クラス"""

    @staticmethod
    def initialize_environment():
        """環境初期化"""
        # 環境変数設定
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # Windows Console API対応
        if sys.platform == 'win32':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except:
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

    @staticmethod
    def setup_logging():
        """ログ設定"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
