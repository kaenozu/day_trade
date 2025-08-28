#!/usr/bin/env python3
"""
シグナル分析モジュール

デイトレードシグナルの判定、分析、計算ロジックを提供します。
"""

import logging
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Any, Optional, Tuple
from .enums import DayTradingSignal, TradingSession
from .recommendation import DayTradingRecommendation


logger = logging.getLogger(__name__)


class TradingSessionAnalyzer:
    """取引セッション分析クラス"""

    def __init__(self, market_manager=None):
        self.market_manager = market_manager

    def get_current_trading_session(self) -> TradingSession:
        """現在の取引時間帯を取得（正確な市場時間管理）"""
        # 正確な市場時間管理システム使用
        if self.market_manager:
            try:
                from market_time_manager import MarketSession
                market_session = self.market_manager.get_current_session(datetime.now())

                # MarketSessionをTradingSessionに変換
                session_map = {
                    MarketSession.PRE_MARKET: TradingSession.PRE_MARKET,
                    MarketSession.MORNING_SESSION: TradingSession.MORNING_SESSION,
                    MarketSession.LUNCH_BREAK: TradingSession.LUNCH_BREAK,
                    MarketSession.AFTERNOON_SESSION: TradingSession.AFTERNOON_SESSION,
                    MarketSession.AFTER_MARKET: TradingSession.AFTER_MARKET,
                    MarketSession.MARKET_CLOSED: TradingSession.AFTER_MARKET  # 休場日は大引け後として扱う
                }

                return session_map.get(market_session, TradingSession.AFTER_MARKET)
            except ImportError:
                pass

        # フォールバック: 従来の簡易判定
        now = datetime.now()
        current_time = now.time()

        if current_time < time(9, 0):
            return TradingSession.PRE_MARKET
        elif time(9, 0) <= current_time < time(11, 30):
            return TradingSession.MORNING_SESSION
        elif time(11, 30) <= current_time < time(12, 30):
            return TradingSession.LUNCH_BREAK
        elif time(12, 30) <= current_time < time(15, 0):
            return TradingSession.AFTERNOON_SESSION
        else:
            return TradingSession.AFTER_MARKET

    def get_session_multiplier(self, session: TradingSession) -> float:
        """時間帯別の動きやすさ係数"""
        multipliers = {
            TradingSession.PRE_MARKET: 0.5,        # 寄り前は動き小さい
            TradingSession.MORNING_SESSION: 1.3,    # 前場は活発
            TradingSession.LUNCH_BREAK: 0.3,       # 昼休みは動かない
            TradingSession.AFTERNOON_SESSION: 1.0,  # 後場は標準
            TradingSession.AFTER_MARKET: 0.4       # 引け後は限定的
        }
        return multipliers.get(session, 1.0)

    def get_session_advice(self) -> str:
        """現在の時間帯に応じたアドバイス"""
        # 正確な市場時間管理システム使用
        if self.market_manager and hasattr(self.market_manager, 'get_session_advice'):
            return self.market_manager.get_session_advice()

        # フォールバック: 従来のアドバイス
        session = self.get_current_trading_session()
        current_time = datetime.now().strftime("%H:%M")
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%m/%d")

        advice_map = {
            TradingSession.PRE_MARKET: f"[{current_time}] 寄り前時間帯\n・寄り付き前の情報収集時間\n・前日の米国市況・為替確認\n・寄り成行注文の準備",

            TradingSession.MORNING_SESSION: f"[{current_time}] 前場時間帯\n・最も活発な取引時間\n・デイトレードのメイン時間帯\n・急な値動きに注意",

            TradingSession.LUNCH_BREAK: f"[{current_time}] 昼休み時間帯\n・取引量減少・値動き限定的\n・後場に向けた戦略見直し時間\n・新規エントリーは慎重に",

            TradingSession.AFTERNOON_SESSION: f"[{current_time}] 後場時間帯\n・決済中心の時間帯\n・大引けに向けポジション整理\n・15:00までに決済完了推奨",

            TradingSession.AFTER_MARKET: f"[{current_time}] 翌日前場予想モード\n・{tomorrow_str}（明日）の前場戦略立案時間\n・オーバーナイトギャップ・寄り付き戦略検討\n・翌日のエントリー銘柄選定"
        }

        return advice_map.get(session, "取引時間外")


class SignalAnalyzer:
    """シグナル分析クラス"""

    def determine_daytrading_signal(self, volatility: float, momentum: float,
                                    volume: float, session: TradingSession) -> DayTradingSignal:
        """デイトレードシグナル判定"""

        # 昼休みや引け後は基本的に待機
        if session in [TradingSession.LUNCH_BREAK, TradingSession.AFTER_MARKET]:
            return DayTradingSignal.WAIT

        # 強いモメンタム + 高出来高 = 強いシグナル
        if momentum >= 2.0 and volume >= 1.8 and volatility >= 4.0:
            return DayTradingSignal.STRONG_BUY
        elif momentum <= -2.0 and volume >= 1.8 and volatility >= 4.0:
            return DayTradingSignal.STRONG_SELL

        # 中程度のシグナル
        elif momentum >= 1.0 and volume >= 1.2:
            return DayTradingSignal.BUY
        elif momentum <= -1.0 and volume >= 1.2:
            return DayTradingSignal.SELL

        # 横ばい・保留
        elif abs(momentum) < 0.5 or volatility < 2.0:
            return DayTradingSignal.WAIT

        # その他はホールド
        else:
            return DayTradingSignal.HOLD

    def get_entry_timing(self, signal: DayTradingSignal, session: TradingSession) -> str:
        """エントリータイミング推奨"""

        timing_map = {
            DayTradingSignal.STRONG_BUY: {
                TradingSession.PRE_MARKET: "寄り成行で即エントリー",
                TradingSession.MORNING_SESSION: "今すぐ成行エントリー",
                TradingSession.AFTERNOON_SESSION: "押し目があれば即エントリー",
            },
            DayTradingSignal.BUY: {
                TradingSession.PRE_MARKET: "寄り後の値動き確認後",
                TradingSession.MORNING_SESSION: "5-10分値動き見て押し目エントリー",
                TradingSession.AFTERNOON_SESSION: "後場開始後の流れ確認",
            },
            DayTradingSignal.SELL: {
                TradingSession.MORNING_SESSION: "利確または損切り実行",
                TradingSession.AFTERNOON_SESSION: "大引け前に決済完了",
            },
            DayTradingSignal.STRONG_SELL: {
                TradingSession.MORNING_SESSION: "即座に決済",
                TradingSession.AFTERNOON_SESSION: "成行で即決済",
            },
            DayTradingSignal.WAIT: {
                TradingSession.PRE_MARKET: "寄り後の動向待ち",
                TradingSession.LUNCH_BREAK: "後場開始まで待機",
                TradingSession.AFTER_MARKET: "明日の寄り付き待ち",
            },
            DayTradingSignal.HOLD: {
                TradingSession.MORNING_SESSION: "ポジション維持・様子見",
                TradingSession.AFTERNOON_SESSION: "大引けまでホールド検討",
            }
        }

        return timing_map.get(signal, {}).get(session, "市況確認後に判断")

    def calculate_profit_stop_levels(self, signal: DayTradingSignal,
                                     volatility: float) -> Tuple[float, float]:
        """利確・損切りライン計算"""

        # ボラティリティベースの利確・損切り設定
        base_profit = min(volatility * 0.6, 3.0)  # 最大3%利確
        base_stop = min(volatility * 0.4, 2.0)    # 最大2%損切り

        signal_adjustments = {
            DayTradingSignal.STRONG_BUY: (base_profit * 1.2, base_stop * 0.8),
            DayTradingSignal.BUY: (base_profit, base_stop),
            DayTradingSignal.STRONG_SELL: (base_profit * 1.2, base_stop * 0.8),
            DayTradingSignal.SELL: (base_profit, base_stop),
            DayTradingSignal.HOLD: (base_profit * 0.7, base_stop * 1.2),
            DayTradingSignal.WAIT: (0.5, 0.5)  # 待機時は小幅
        }

        profit, stop = signal_adjustments.get(signal, (base_profit, base_stop))
        return round(profit, 1), round(stop, 1)

    def get_recommended_holding_time(self, signal: DayTradingSignal,
                                     session: TradingSession) -> str:
        """推奨保有時間"""

        if signal == DayTradingSignal.STRONG_BUY:
            return "30分〜2時間（短期集中）"
        elif signal == DayTradingSignal.BUY:
            return "1〜4時間（場中完結）"
        elif signal in [DayTradingSignal.SELL, DayTradingSignal.STRONG_SELL]:
            return "即座に決済"
        elif signal == DayTradingSignal.HOLD:
            if session == TradingSession.MORNING_SESSION:
                return "大引けまで保有検討"
            else:
                return "1〜2時間程度"
        else:  # WAIT
            return "エントリーチャンス待ち"

    def calculate_daytrading_confidence(self, signal: DayTradingSignal,
                                        volatility: float, volume: float,
                                        session: TradingSession) -> float:
        """デイトレード信頼度計算"""

        base_confidence = 70.0

        # シグナル強度による調整
        signal_boost = {
            DayTradingSignal.STRONG_BUY: 15,
            DayTradingSignal.BUY: 10,
            DayTradingSignal.STRONG_SELL: 15,
            DayTradingSignal.SELL: 10,
            DayTradingSignal.HOLD: 5,
            DayTradingSignal.WAIT: -10
        }.get(signal, 0)

        # 出来高による調整
        volume_boost = min((volume - 1.0) * 10, 15)

        # ボラティリティによる調整（適度が最適）
        if 3.0 <= volatility <= 6.0:
            volatility_boost = 10  # 適度なボラティリティ
        elif volatility > 6.0:
            volatility_boost = 5   # 高すぎるボラティリティ
        else:
            volatility_boost = 0   # 低ボラティリティ

        # 時間帯による調整
        session_boost = {
            TradingSession.MORNING_SESSION: 10,
            TradingSession.AFTERNOON_SESSION: 5,
            TradingSession.PRE_MARKET: -5,
            TradingSession.LUNCH_BREAK: -15,
            TradingSession.AFTER_MARKET: -10
        }.get(session, 0)

        final_confidence = base_confidence + signal_boost + volume_boost + volatility_boost + session_boost
        return max(50, min(95, final_confidence))

    def assess_daytrading_risk(self, volatility: float, signal: DayTradingSignal) -> str:
        """デイトレードリスク評価"""

        if volatility >= 6.0:
            return "高"
        elif volatility >= 4.0:
            return "中"
        else:
            return "低"

    def describe_volume_trend(self, volume_ratio: float) -> str:
        """出来高動向の説明"""
        if volume_ratio >= 2.0:
            return "出来高急増"
        elif volume_ratio >= 1.5:
            return "出来高増加"
        elif volume_ratio >= 1.2:
            return "出来高やや多い"
        elif volume_ratio >= 0.8:
            return "出来高普通"
        else:
            return "出来高少ない"

    def describe_price_momentum(self, momentum: float) -> str:
        """価格モメンタムの説明"""
        if momentum >= 2.0:
            return "強い上昇"
        elif momentum >= 1.0:
            return "上昇"
        elif momentum >= 0.5:
            return "やや上昇"
        elif momentum <= -2.0:
            return "強い下落"
        elif momentum <= -1.0:
            return "下落"
        elif momentum <= -0.5:
            return "やや下落"
        else:
            return "横ばい"

    def calculate_market_timing_score(self, signal: DayTradingSignal,
                                      volatility: float, volume: float,
                                      session: TradingSession) -> float:
        """市場タイミングスコア算出"""

        # 基本スコア
        score = 50.0

        # シグナル強度
        signal_scores = {
            DayTradingSignal.STRONG_BUY: 90,
            DayTradingSignal.BUY: 75,
            DayTradingSignal.STRONG_SELL: 85,
            DayTradingSignal.SELL: 70,
            DayTradingSignal.HOLD: 60,
            DayTradingSignal.WAIT: 40
        }
        score = signal_scores.get(signal, 50)

        # ボラティリティ調整（デイトレードには適度が最適）
        if 3.0 <= volatility <= 6.0:
            score += 15
        elif volatility > 6.0:
            score += 5  # 高すぎは危険
        else:
            score -= 10  # 低すぎは面白くない

        # 出来高調整
        score += min((volume - 1.0) * 20, 15)

        # 時間帯調整
        session_adj = {
            TradingSession.MORNING_SESSION: 10,
            TradingSession.AFTERNOON_SESSION: 5,
            TradingSession.PRE_MARKET: -5,
            TradingSession.LUNCH_BREAK: -20,
            TradingSession.AFTER_MARKET: -15
        }.get(session, 0)
        score += session_adj

        return max(0, min(100, score))


class RealDataConverter:
    """実データ変換クラス"""

    def convert_to_daytrading_recommendation(self, real_rec: Dict[str, any], 
                                             session: TradingSession) -> DayTradingRecommendation:
        """実データ推奨をDayTradingRecommendationに変換"""

        # シグナル変換
        signal_map = {
            "強い買い": DayTradingSignal.STRONG_BUY,
            "●買い●": DayTradingSignal.BUY,
            "△やや買い△": DayTradingSignal.BUY,
            "…待機…": DayTradingSignal.WAIT,
            "▽売り▽": DayTradingSignal.SELL,
            "▼強い売り▼": DayTradingSignal.STRONG_SELL,
        }

        signal = signal_map.get(real_rec.get("signal", "…待機…"), DayTradingSignal.WAIT)

        # エントリータイミング生成
        entry_timing = self._generate_entry_timing(real_rec, session)

        # 保有時間推奨
        holding_time = self._generate_holding_time(real_rec, session)

        return DayTradingRecommendation(
            symbol=real_rec["symbol"],
            name=real_rec["name"],
            signal=signal,
            entry_timing=entry_timing,
            target_profit=real_rec.get("target_profit", 2.0),
            stop_loss=real_rec.get("stop_loss", 1.5),
            holding_time=holding_time,
            confidence=real_rec.get("confidence", 80),
            risk_level=self._calculate_risk_level(real_rec),
            volume_trend="実データベース",
            price_momentum=self._calculate_momentum(real_rec),
            intraday_volatility=real_rec.get("volatility", 3.0),
            market_timing_score=real_rec.get("trading_score", 70)
        )

    def _generate_entry_timing(self, real_rec: Dict[str, any], session: TradingSession) -> str:
        """エントリータイミング生成"""
        signal = real_rec.get("signal", "")
        change_pct = real_rec.get("change_percent", 0)

        if "強い買い" in signal:
            return "即座に成り行きで積極エントリー（実データ推奨）"
        elif "買い" in signal:
            if change_pct > 0:
                return "押し目5-10分待ってからエントリー（実ギャップ）"
            else:
                return "現在値近辺でエントリー（実下値）"
        elif "待機" in signal:
            return "待機期間中エントリーチャンス探し"
        else:
            return "利確・損切りタイミング判定（実データ）"

    def _generate_holding_time(self, real_rec: Dict[str, any], session: TradingSession) -> str:
        """保有時間推奨"""
        volatility = real_rec.get("volatility", 3.0)

        if volatility > 5.0:
            return "短時間1〜2時間程度"
        elif volatility > 3.0:
            return "30分〜1時間での決済"
        else:
            return "当日中決済完了推奨"

    def _calculate_risk_level(self, real_rec: Dict[str, any]) -> str:
        """リスクレベル計算"""
        volatility = real_rec.get("volatility", 3.0)
        trading_score = real_rec.get("trading_score", 70)

        if volatility > 6.0:
            return "[高リスク]"
        elif volatility > 3.0 and trading_score > 80:
            return "[中リスク]"
        else:
            return "[低リスク]"

    def _calculate_momentum(self, real_rec: Dict[str, any]) -> str:
        """価格モメンタム計算"""
        change_pct = real_rec.get("change_pct", 0)
        rsi = real_rec.get("rsi", 50)

        if change_pct > 2.0 and rsi < 70:
            return "強い上昇モメンタム"
        elif change_pct > 0 and rsi < 60:
            return "上昇基調"
        elif change_pct < -2.0 and rsi > 30:
            return "下落からの反転期待"
        else:
            return "モメンタム中立"