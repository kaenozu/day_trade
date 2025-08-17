#!/usr/bin/env python3
"""
Day Trade Personal - アプリケーションクラス

リファクタリング後のメインアプリケーション
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from .system_initializer import SystemInitializer
from ..cli.argument_parser import ArgumentParser
from ..analysis.trading_analyzer import TradingAnalyzer
from ..dashboard.web_dashboard import WebDashboard


class DayTradeApplication:
    """Day Trade メインアプリケーション"""

    def __init__(self):
        """初期化"""
        SystemInitializer.initialize_environment()
        SystemInitializer.setup_logging()

        self.analyzer = None
        self.web_dashboard = None

    def run(self) -> int:
        """アプリケーション実行"""
        try:
            # 引数解析
            parser = ArgumentParser()
            args = parser.parse_args()

            # モード別実行
            if args.web:
                return self._run_web_mode(args)
            elif args.quick:
                return self._run_quick_analysis(args)
            elif args.multi:
                return self._run_multi_analysis(args)
            else:
                return self._run_default_analysis(args)

        except KeyboardInterrupt:
            print("\n操作が中断されました")
            return 0
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return 1

    def _run_web_mode(self, args) -> int:
        """Webモード実行"""
        print("🌐 Webダッシュボード起動中...")
        self.web_dashboard = WebDashboard(port=args.port, debug=args.debug)
        self.web_dashboard.run()
        return 0

    def _run_quick_analysis(self, args) -> int:
        """クイック分析実行"""
        print("⚡ クイック分析モード")
        self.analyzer = TradingAnalyzer(quick_mode=True)
        results = self.analyzer.analyze(args.symbols)
        self._display_results(results)
        return 0

    def _run_multi_analysis(self, args) -> int:
        """マルチ分析実行"""
        print("📊 マルチ銘柄分析モード")
        self.analyzer = TradingAnalyzer(multi_mode=True)
        results = self.analyzer.analyze(args.symbols)
        self._display_results(results)
        return 0

    def _run_default_analysis(self, args) -> int:
        """デフォルト分析実行"""
        print("🎯 デフォルト分析モード")
        self.analyzer = TradingAnalyzer()
        results = self.analyzer.analyze(args.symbols)
        self._display_results(results)
        return 0

    def _display_results(self, results):
        """結果表示"""
        print("\n" + "="*50)
        print("📈 分析結果")
        print("="*50)

        for result in results:
            print(f"銘柄: {result.get('symbol', 'N/A')}")
            print(f"推奨: {result.get('recommendation', 'N/A')}")
            print(f"信頼度: {result.get('confidence', 0):.1%}")
            print("-" * 30)
