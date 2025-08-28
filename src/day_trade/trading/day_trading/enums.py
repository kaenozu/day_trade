#!/usr/bin/env python3
"""
デイトレードエンジン用の列挙型定義

取引シグナルと取引セッションの定義を提供します。
"""

from enum import Enum


class DayTradingSignal(Enum):
    """デイトレードシグナル"""
    STRONG_BUY = "強い買い"      # 即座に買い
    BUY = "買い"               # 押し目で買い
    HOLD = "ホールド"          # 既存ポジション維持
    SELL = "売り"              # 利確・損切り売り
    STRONG_SELL = "強い売り"    # 即座に売り
    WAIT = "待機"              # エントリーチャンス待ち


class TradingSession(Enum):
    """取引時間帯"""
    PRE_MARKET = "寄り前"      # 9:00前
    MORNING_SESSION = "前場"    # 9:00-11:30
    LUNCH_BREAK = "昼休み"     # 11:30-12:30
    AFTERNOON_SESSION = "後場"  # 12:30-15:00
    AFTER_MARKET = "大引け後"   # 15:00後