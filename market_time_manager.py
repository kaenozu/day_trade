#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Time Manager - 市場時間管理システム

正確な東京証券取引所の開場・休場判定
祝日・年末年始・臨時休場対応
"""

import jpholiday
from datetime import datetime, time, timedelta
from typing import Dict, Tuple, Optional
import logging
from enum import Enum

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class MarketSession(Enum):
    """市場セッション"""
    PRE_MARKET = "寄り前"          # 〜9:00
    MORNING_SESSION = "前場"       # 9:00-11:30
    LUNCH_BREAK = "昼休み"         # 11:30-12:30
    AFTERNOON_SESSION = "後場"     # 12:30-15:00
    AFTER_MARKET = "大引け後"      # 15:00〜
    MARKET_CLOSED = "休場"         # 土日祝日

class MarketStatus(Enum):
    """市場状況"""
    OPEN = "開場中"
    CLOSED = "休場中"
    PRE_OPEN = "開場前"
    POST_CLOSE = "終了後"

class MarketTimeManager:
    """
    東京証券取引所の正確な時間管理システム
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 東証の営業時間
        self.morning_open = time(9, 0)      # 前場開始
        self.morning_close = time(11, 30)   # 前場終了
        self.afternoon_open = time(12, 30)  # 後場開始
        self.afternoon_close = time(15, 0)  # 後場終了

        # 特別営業日・休業日の管理
        self.special_holidays = {
            # 年末年始休場（通常12/31-1/3）
            "2024-12-31": "大納会翌日",
            "2025-01-01": "元日",
            "2025-01-02": "年始休場",
            "2025-01-03": "年始休場",
            # 臨時休場など（必要に応じて追加）
        }

        self.special_trading_days = {
            # 振替営業日など（通常はなし）
        }

    def is_market_day(self, date: datetime = None) -> bool:
        """
        指定日が市場営業日かどうか判定

        Args:
            date: 判定したい日付（未指定時は今日）

        Returns:
            bool: 営業日かどうか
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime('%Y-%m-%d')

        # 特別休場日チェック
        if date_str in self.special_holidays:
            return False

        # 特別営業日チェック
        if date_str in self.special_trading_days:
            return True

        # 土日チェック
        if date.weekday() >= 5:  # 土曜=5, 日曜=6
            return False

        # 祝日チェック（jpholiday使用）
        if jpholiday.is_holiday(date.date()):
            return False

        return True

    def get_current_session(self, now: datetime = None) -> MarketSession:
        """
        現在の市場セッション取得

        Args:
            now: 現在時刻（未指定時はシステム時刻）

        Returns:
            MarketSession: 現在のセッション
        """
        if now is None:
            now = datetime.now()

        # 休場日判定
        if not self.is_market_day(now):
            return MarketSession.MARKET_CLOSED

        current_time = now.time()

        # 時間帯による判定
        if current_time < self.morning_open:
            return MarketSession.PRE_MARKET
        elif self.morning_open <= current_time < self.morning_close:
            return MarketSession.MORNING_SESSION
        elif self.morning_close <= current_time < self.afternoon_open:
            return MarketSession.LUNCH_BREAK
        elif self.afternoon_open <= current_time < self.afternoon_close:
            return MarketSession.AFTERNOON_SESSION
        else:
            return MarketSession.AFTER_MARKET

    def get_market_status(self, now: datetime = None) -> MarketStatus:
        """
        現在の市場状況取得

        Args:
            now: 現在時刻

        Returns:
            MarketStatus: 市場状況
        """
        session = self.get_current_session(now)

        if session == MarketSession.MARKET_CLOSED:
            return MarketStatus.CLOSED
        elif session in [MarketSession.MORNING_SESSION, MarketSession.AFTERNOON_SESSION]:
            return MarketStatus.OPEN
        elif session == MarketSession.PRE_MARKET:
            return MarketStatus.PRE_OPEN
        else:
            return MarketStatus.POST_CLOSE

    def is_market_open(self, now: datetime = None) -> bool:
        """
        現在市場が開場中かどうか

        Returns:
            bool: 開場中かどうか
        """
        return self.get_market_status(now) == MarketStatus.OPEN

    def get_next_market_open(self, now: datetime = None) -> datetime:
        """
        次の市場開場時刻取得

        Args:
            now: 基準時刻

        Returns:
            datetime: 次の開場時刻
        """
        if now is None:
            now = datetime.now()

        # 現在のセッション確認
        session = self.get_current_session(now)

        if session == MarketSession.PRE_MARKET:
            # 今日の前場開始
            return datetime.combine(now.date(), self.morning_open)

        elif session == MarketSession.MORNING_SESSION:
            # 既に開場中なので後場開始時刻を返す
            return datetime.combine(now.date(), self.afternoon_open)

        elif session == MarketSession.LUNCH_BREAK:
            # 今日の後場開始
            return datetime.combine(now.date(), self.afternoon_open)

        elif session in [MarketSession.AFTERNOON_SESSION, MarketSession.AFTER_MARKET]:
            # 翌営業日の前場開始
            next_day = now + timedelta(days=1)
            while not self.is_market_day(next_day):
                next_day += timedelta(days=1)
            return datetime.combine(next_day.date(), self.morning_open)

        else:  # MARKET_CLOSED
            # 次の営業日の前場開始
            next_day = now + timedelta(days=1)
            while not self.is_market_day(next_day):
                next_day += timedelta(days=1)
            return datetime.combine(next_day.date(), self.morning_open)

    def get_next_market_close(self, now: datetime = None) -> datetime:
        """
        次の市場終了時刻取得

        Args:
            now: 基準時刻

        Returns:
            datetime: 次の終了時刻
        """
        if now is None:
            now = datetime.now()

        session = self.get_current_session(now)

        if session == MarketSession.MORNING_SESSION:
            # 前場終了時刻
            return datetime.combine(now.date(), self.morning_close)

        elif session in [MarketSession.LUNCH_BREAK, MarketSession.AFTERNOON_SESSION]:
            # 後場終了時刻
            return datetime.combine(now.date(), self.afternoon_close)

        else:
            # 次の営業日の後場終了
            next_open = self.get_next_market_open(now)
            return datetime.combine(next_open.date(), self.afternoon_close)

    def get_time_until_next_event(self, now: datetime = None) -> Tuple[str, int]:
        """
        次のイベントまでの時間取得

        Args:
            now: 基準時刻

        Returns:
            Tuple[str, int]: (イベント名, 秒数)
        """
        if now is None:
            now = datetime.now()

        session = self.get_current_session(now)

        if session == MarketSession.PRE_MARKET:
            next_event = self.get_next_market_open(now)
            event_name = "前場開始"

        elif session == MarketSession.MORNING_SESSION:
            next_event = datetime.combine(now.date(), self.morning_close)
            event_name = "前場終了"

        elif session == MarketSession.LUNCH_BREAK:
            next_event = datetime.combine(now.date(), self.afternoon_open)
            event_name = "後場開始"

        elif session == MarketSession.AFTERNOON_SESSION:
            next_event = datetime.combine(now.date(), self.afternoon_close)
            event_name = "大引け"

        else:
            next_event = self.get_next_market_open(now)
            event_name = "次回開場"

        seconds = int((next_event - now).total_seconds())
        return event_name, max(0, seconds)

    def get_market_summary(self, now: datetime = None) -> Dict[str, any]:
        """
        市場状況の総合情報取得

        Returns:
            Dict: 市場状況サマリー
        """
        if now is None:
            now = datetime.now()

        session = self.get_current_session(now)
        status = self.get_market_status(now)
        event_name, seconds = self.get_time_until_next_event(now)

        # 秒数を時分秒に変換
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        if hours > 0:
            time_str = f"{hours}時間{minutes}分{remaining_seconds}秒"
        elif minutes > 0:
            time_str = f"{minutes}分{remaining_seconds}秒"
        else:
            time_str = f"{remaining_seconds}秒"

        return {
            "current_time": now.strftime('%Y-%m-%d %H:%M:%S'),
            "is_market_day": self.is_market_day(now),
            "session": session.value,
            "status": status.value,
            "is_open": self.is_market_open(now),
            "next_event": event_name,
            "time_until_next": time_str,
            "next_open": self.get_next_market_open(now).strftime('%Y-%m-%d %H:%M:%S'),
            "next_close": self.get_next_market_close(now).strftime('%Y-%m-%d %H:%M:%S')
        }

    def get_session_advice(self, now: datetime = None) -> str:
        """
        現在のセッションに応じたアドバイス生成

        Returns:
            str: アドバイスメッセージ
        """
        if now is None:
            now = datetime.now()

        session = self.get_current_session(now)
        event_name, seconds = self.get_time_until_next_event(now)

        # 時間表示を簡潔に
        if seconds > 3600:
            time_display = f"約{seconds//3600}時間後"
        elif seconds > 60:
            time_display = f"約{seconds//60}分後"
        else:
            time_display = f"{seconds}秒後"

        if session == MarketSession.PRE_MARKET:
            return f"[{now.strftime('%H:%M')}] 寄り前準備時間\n・{time_display}に前場開始\n・寄り付き戦略の最終確認時間\n・ニュース・材料の再チェック推奨"

        elif session == MarketSession.MORNING_SESSION:
            return f"[{now.strftime('%H:%M')}] 前場取引中\n・活発な値動きが期待される時間帯\n・{time_display}に前場終了\n・デイトレード主力時間"

        elif session == MarketSession.LUNCH_BREAK:
            return f"[{now.strftime('%H:%M')}] 昼休み中\n・{time_display}に後場開始\n・午前の結果分析と午後戦略調整\n・海外市況・ニュースチェック時間"

        elif session == MarketSession.AFTERNOON_SESSION:
            return f"[{now.strftime('%H:%M')}] 後場取引中\n・機関投資家の動きが活発化\n・{time_display}で大引け\n・ポジション決済準備時間"

        elif session == MarketSession.AFTER_MARKET:
            return f"[{now.strftime('%H:%M')}] 翌日前場予想モード\n・{event_name}まで{time_display}\n・オーバーナイトギャップ・寄り付き戦略検討\n・翌日のエントリー銘柄選定"

        else:  # MARKET_CLOSED
            return f"[{now.strftime('%H:%M')}] 休場日\n・{event_name}まで{time_display}\n・市場分析・学習時間\n・戦略見直しと準備期間"


def main():
    """市場時間管理システムのテスト"""
    print("=== 市場時間管理システム テスト ===")

    manager = MarketTimeManager()
    summary = manager.get_market_summary()

    print("\n[ 現在の市場状況 ]")
    print(f"現在時刻: {summary['current_time']}")
    print(f"営業日判定: {'営業日' if summary['is_market_day'] else '休場日'}")
    print(f"セッション: {summary['session']}")
    print(f"市場状況: {summary['status']}")
    print(f"開場中: {'Yes' if summary['is_open'] else 'No'}")
    print()

    print("[ 次回イベント ]")
    print(f"次回: {summary['next_event']}")
    print(f"残り時間: {summary['time_until_next']}")
    print()

    print("[ セッションアドバイス ]")
    advice = manager.get_session_advice()
    print(advice)

    print("\n" + "="*50)
    print("市場時間管理システム 正常動作確認")

if __name__ == "__main__":
    main()