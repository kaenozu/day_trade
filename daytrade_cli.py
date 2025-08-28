#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade CLI Module - CLIインターフェース分離
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any


class DayTradeCLI:
    """デイトレードシステムのCLIインターフェース"""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """コマンドライン引数パーサーを作成"""
        parser = argparse.ArgumentParser(
            description='Day Trade Personal - 個人利用専用版',
            epilog='93%精度AIシステムでデイトレード支援'
        )

        # 基本モード選択
        mode_group = parser.add_mutually_exclusive_group()
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

        # 銘柄指定
        parser.add_argument(
            '--symbols', '-s',
            nargs='+',
            default=None,
            help='対象銘柄コード（デフォルト: 設定ファイルから高・中優先度銘柄を自動選択）'
        )

        # オプション設定
        parser.add_argument(
            '--port', '-p',
            type=int,
            default=8000,
            help='Webサーバーポート（デフォルト: 8000）'
        )
        parser.add_argument(
            '--debug', '-d',
            action='store_true',
            help='デバッグモード'
        )
        parser.add_argument(
            '--no-cache',
            action='store_true',
            help='キャッシュを使用しない'
        )
        parser.add_argument(
            '--config',
            type=str,
            help='設定ファイルパス'
        )

        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """コマンドライン引数を解析"""
        return self.parser.parse_args(args)

    def execute(self, args: Optional[List[str]] = None) -> int:
        """CLIコマンドを実行"""
        parsed_args = self.parse_args(args)

        try:
            # 適切なハンドラーを実行
            if parsed_args.web:
                return self._run_web_mode(parsed_args)
            elif parsed_args.quick:
                return self._run_quick_mode(parsed_args)
            elif parsed_args.multi:
                return self._run_multi_mode(parsed_args)
            elif parsed_args.validate:
                return self._run_validate_mode(parsed_args)
            else:
                return self._run_default_mode(parsed_args)

        except KeyboardInterrupt:
            print("\\n処理を中断しました")
            return 1
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            if parsed_args.debug:
                import traceback
                traceback.print_exc()
            return 1

    def _run_web_mode(self, args: argparse.Namespace) -> int:
        """Webダッシュボードモード実行"""
        import subprocess
        import sys

        # CLIスクリプトと同じディレクトリにあるdaytrade_web.pyのパスを構築
        web_script_path = Path(__file__).parent / "daytrade_web.py"
        
        command = [
            sys.executable, 
            str(web_script_path),
            f"--port={args.port}"
        ]
        if args.debug:
            command.append("--debug")

        print(f"Webサーバーを起動します...")
        print(f"コマンド: {' '.join(command)}")
        print(f"ブラウザで http://localhost:{args.port} を開いてください。")
        print("サーバーを停止するには Ctrl+C を押してください。")

        try:
            # subprocess.runはプロセスが終了するまで待機する
            result = subprocess.run(command, check=False)
            return result.returncode
        except FileNotFoundError:
            print(f"エラー: {web_script_path} が見つかりません。", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Webサーバーの起動に失敗しました: {e}", file=sys.stderr)
            return 1

    def _run_quick_mode(self, args: argparse.Namespace) -> int:
        """基本分析モード実行"""
        from daytrade_core import DayTradeCore

        core = DayTradeCore(debug=args.debug, use_cache=not args.no_cache)
        return asyncio.run(core.run_quick_analysis(args.symbols))

    def _run_multi_mode(self, args: argparse.Namespace) -> int:
        """複数銘柄分析モード実行"""
        from daytrade_core import DayTradeCore

        core = DayTradeCore(debug=args.debug, use_cache=not args.no_cache)
        return asyncio.run(core.run_multi_analysis(args.symbols))

    def _run_validate_mode(self, args: argparse.Namespace) -> int:
        """予測精度検証モード実行"""
        from daytrade_core import DayTradeCore

        core = DayTradeCore(debug=args.debug, use_cache=not args.no_cache)
        return asyncio.run(core.run_validation(args.symbols))

    def _run_default_mode(self, args: argparse.Namespace) -> int:
        """デフォルトモード（デイトレード推奨）実行"""
        from daytrade_core import DayTradeCore

        core = DayTradeCore(debug=args.debug, use_cache=not args.no_cache)
        return asyncio.run(core.run_daytrading_analysis(args.symbols))


def main():
    """CLIメイン関数"""
    cli = DayTradeCLI()
    return cli.execute()


if __name__ == "__main__":
    sys.exit(main())