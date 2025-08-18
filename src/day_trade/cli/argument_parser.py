#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Personal - コマンドライン引数パーサー

リファクタリング後のメインアプリケーション用引数パーサー
"""

import argparse
from typing import List, Optional


class ArgumentParser:
    """コマンドライン引数パーサー"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Day Trade Personal - 個人利用専用版",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="93%精度AIシステムでデイトレード支援"
        )
        self._setup_arguments()
    
    def _setup_arguments(self):
        """引数を設定"""
        # メインモード（排他的）
        mode_group = self.parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            '--quick', '-q',
            action='store_true',
            help='基本分析モード（高速）'
        )
        mode_group.add_argument(
            '--multi', '-m',
            action='store_true',
            help='複数銘柄分析モード'
        )
        mode_group.add_argument(
            '--web', '-w',
            action='store_true',
            help='Webダッシュボード起動'
        )
        mode_group.add_argument(
            '--validate', '-v',
            action='store_true',
            help='予測精度検証モード'
        )
        
        # 共通オプション
        self.parser.add_argument(
            '--symbols', '-s',
            nargs='+',
            help='対象銘柄コード（デフォルト: トヨタ, 三菱UFJ, SBG, ソニー）'
        )
        self.parser.add_argument(
            '--port', '-p',
            type=int,
            default=8000,
            help='Webサーバーポート（デフォルト: 8000）'
        )
        self.parser.add_argument(
            '--debug', '-d',
            action='store_true',
            help='デバッグモード'
        )
        self.parser.add_argument(
            '--no-cache',
            action='store_true',
            help='キャッシュを使用しない'
        )
        self.parser.add_argument(
            '--config',
            type=str,
            help='設定ファイルパス'
        )
    
    def parse_args(self, args: Optional[List[str]] = None):
        """引数を解析"""
        return self.parser.parse_args(args)
    
    def print_help(self):
        """ヘルプを表示"""
        self.parser.print_help()