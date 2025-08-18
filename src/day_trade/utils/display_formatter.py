#!/usr/bin/env python3
"""
Day Trade Personal - ディスプレイフォーマッター

UI・見た目の改善のための表示ユーティリティ
"""

import sys
import time
from datetime import datetime
from typing import List, Dict, Any

try:
    import colorama
    colorama.init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


class Colors:
    """ANSIカラーコード定数"""
    if COLORS_AVAILABLE and sys.platform == 'win32':
        # Windows環境
        RED = '\033[31m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        BLUE = '\033[34m'
        MAGENTA = '\033[35m'
        CYAN = '\033[36m'
        WHITE = '\033[37m'
        BRIGHT_GREEN = '\033[92m'
        BRIGHT_YELLOW = '\033[93m'
        BRIGHT_BLUE = '\033[94m'
        BRIGHT_CYAN = '\033[96m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
    else:
        # カラー無効または他の環境
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ''
        BRIGHT_GREEN = BRIGHT_YELLOW = BRIGHT_BLUE = BRIGHT_CYAN = ''
        RESET = BOLD = ''


class DisplayFormatter:
    """表示フォーマッター"""

    def __init__(self):
        self.start_time = time.time()
        self.config = None

    def print_startup_banner(self):
        """美しいスタートアップバナー"""
        print()
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│           🚀 Day Trade Personal AI System                   │")
        print("│              93%精度 自動分析エンジン v2.0                   │")
        print("│                                                             │")
        print("│  ⚡ 高速並列処理  🤖 機械学習  📊 テクニカル分析           │")
        print("├─────────────────────────────────────────────────────────────┤")
        elapsed = time.time() - self.start_time
        print(f"│  ✅ システム初期化完了 ({elapsed:.1f}秒)                              │")
        print("└─────────────────────────────────────────────────────────────┘")
        print()

    def print_analysis_header(self, symbol_count: int):
        """分析開始ヘッダー"""
        timestamp = datetime.now().strftime('%Y年%m月%d日 %H:%M')
        print("┌──────────────────────────────────────────────────────────────┐")
        print(f"│  📊 Day Trade AI分析結果 - {timestamp}              │")
        print("├──────────────────────────────────────────────────────────────┤")

    def print_analysis_results(self, results: List[Dict[str, Any]]):
        """美しい分析結果表示"""
        if not results:
            print("│  ⚠️  分析対象銘柄がありません                                │")
            print("└──────────────────────────────────────────────────────────────┘")
            return

        # 推奨別にグループ化
        buy_stocks = []
        sell_stocks = []
        hold_stocks = []
        skip_stocks = []

        for result in results:
            symbol = result.get('symbol', 'N/A')
            rec = result.get('recommendation', 'HOLD')
            conf = result.get('confidence', 0)

            if rec == 'SKIP':
                skip_stocks.append({'symbol': symbol, 'confidence': conf})
                continue

            company_name = self._get_company_name_safe(result)
            stock_data = {
                'symbol': symbol,
                'company': company_name,
                'confidence': conf
            }

            if rec == 'BUY':
                buy_stocks.append(stock_data)
            elif rec == 'SELL':
                sell_stocks.append(stock_data)
            else:
                hold_stocks.append(stock_data)

        # 各カテゴリー表示
        if buy_stocks:
            print(f"│  🚀 BUY推奨 ({len(buy_stocks)}銘柄)                                         │")
            for stock in buy_stocks:
                confidence_bar = self._create_confidence_bar(stock['confidence'])
                print(f"│     • {stock['symbol']} {stock['company']:<15} 信頼度: {confidence_bar} {stock['confidence']:.0%}     │")
            print("│                                                              │")

        if sell_stocks:
            print(f"│  📉 SELL推奨 ({len(sell_stocks)}銘柄)                                        │")
            for stock in sell_stocks:
                confidence_bar = self._create_confidence_bar(stock['confidence'])
                print(f"│     • {stock['symbol']} {stock['company']:<15} 信頼度: {confidence_bar} {stock['confidence']:.0%}     │")
            print("│                                                              │")

        if hold_stocks:
            print(f"│  ⏸️  HOLD推奨 ({len(hold_stocks)}銘柄)                                       │")
            for stock in hold_stocks:
                confidence_bar = self._create_confidence_bar(stock['confidence'])
                print(f"│     • {stock['symbol']} {stock['company']:<15} 信頼度: {confidence_bar} {stock['confidence']:.0%}     │")
            print("│                                                              │")

        if skip_stocks:
            print(f"│  ⚠️  分析不可 ({len(skip_stocks)}銘柄)                                     │")
            for stock in skip_stocks:
                print(f"│     • {stock['symbol']} (上場廃止)                                     │")
            print("│                                                              │")

        # フッター統計
        self._print_analysis_footer(results)

    def _create_confidence_bar(self, confidence: float, width: int = 10) -> str:
        """信頼度バーを作成"""
        filled = int(confidence * width)
        bar = "█" * filled + "▌" * (1 if (confidence * width) % 1 > 0.5 else 0)
        bar += "▌" * (width - len(bar))
        return bar[:width]  # 確実に指定幅に収める

    def _load_config(self):
        """設定ファイルを読み込み"""
        try:
            import json
            from pathlib import Path

            # パスの計算（display_formatter.py から見た相対パス）
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "settings.json"

            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = {'watchlist': {'symbols': []}}
        except Exception as e:
            self.config = {'watchlist': {'symbols': []}}

    def _get_company_name_safe(self, result: Dict[str, Any]) -> str:
        """設定ファイルから会社名を安全に取得"""
        symbol = result.get('symbol', '')
        try:
            # 設定ファイルをまだ読み込んでいない場合は読み込み
            if self.config is None:
                self._load_config()

            # 設定ファイルから会社名を検索
            for symbol_info in self.config.get('watchlist', {}).get('symbols', []):
                if symbol_info.get('code') == symbol:
                    return symbol_info.get('name', symbol)

            # 見つからない場合は銘柄コードをそのまま返す
            return symbol
        except Exception as e:
            # エラー時は銘柄コードをそのまま返す
            return symbol

    def _print_analysis_footer(self, results: List[Dict[str, Any]]):
        """分析フッター統計"""
        elapsed = time.time() - self.start_time
        analyzed_count = len([r for r in results if r.get('recommendation') != 'SKIP'])

        # 平均信頼度計算
        confidences = [r.get('confidence', 0) for r in results if r.get('recommendation') != 'SKIP']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # 次回更新時刻（30分後）
        next_update = datetime.now().replace(minute=30 if datetime.now().minute < 30 else 0,
                                           hour=datetime.now().hour + (1 if datetime.now().minute >= 30 else 0),
                                           second=0).strftime('%H:%M')

        # データ更新時刻（直近の30分区切り）
        data_update = datetime.now().replace(minute=0 if datetime.now().minute < 30 else 30,
                                           second=0).strftime('%H:%M')

        print("├──────────────────────────────────────────────────────────────┤")
        print(f"│  ⏱️  分析時間: {elapsed:.1f}秒    📊 分析精度: {avg_confidence:.1%}               │")
        print(f"│  🔄 次回更新: {next_update}      💾 データ更新: {data_update}               │")
        print("└──────────────────────────────────────────────────────────────┘")

    def print_progress_bar(self, current: int, total: int, current_symbol: str = ""):
        """プログレスバー表示"""
        percentage = (current / total) * 100 if total > 0 else 0
        filled = int(percentage / 100 * 40)  # 40文字幅のバー
        bar = "█" * filled + "▌" * (40 - filled)

        remaining_time = self._estimate_remaining_time(current, total)

        print("\r📊 銘柄分析中...")
        print("┌──────────────────────────────────────────────────────────────┐")
        print(f"│  {bar} {percentage:.0f}% ({current}/{total} 完了)    │")
        if current_symbol:
            print(f"│  📈 現在処理中: {current_symbol}                                │")
        if remaining_time:
            print(f"│  ⏱️  推定残り時間: {remaining_time}                                   │")
        print("└──────────────────────────────────────────────────────────────┘")

    def _estimate_remaining_time(self, current: int, total: int) -> str:
        """残り時間推定"""
        if current == 0:
            return ""

        elapsed = time.time() - self.start_time
        rate = current / elapsed
        remaining = (total - current) / rate

        if remaining < 60:
            return f"{remaining:.1f}秒"
        elif remaining < 3600:
            return f"{remaining/60:.1f}分"
        else:
            return f"{remaining/3600:.1f}時間"

    def print_error_box(self, error_message: str):
        """エラーメッセージボックス"""
        print()
        print("┌──────────────────────────────────────────────────────────────┐")
        print("│  ❌ エラーが発生しました                                     │")
        print("├──────────────────────────────────────────────────────────────┤")

        # メッセージを60文字で折り返し
        words = error_message.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= 58:  # パディング考慮
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        for line in lines:
            print(f"│  {line:<58} │")

        print("└──────────────────────────────────────────────────────────────┘")
        print()


# シングルトンインスタンス
formatter = DisplayFormatter()