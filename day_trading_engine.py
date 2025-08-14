#!/usr/bin/env python3
"""
Day Trade Personal - 1日単位デイトレード推奨エンジン

デイトレードに特化した1日単位の売買タイミング推奨システム
"""

import asyncio
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from src.day_trade.ml.base_models.random_forest_model import RandomForestModel
from src.day_trade.data.fetchers.yfinance_fetcher import YFinanceFetcher

# 実データプロバイダー統合
try:
    from real_data_provider import RealDataProvider, RealDataAnalysisEngine
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

# 市場時間管理システム統合
try:
    from market_time_manager import MarketTimeManager, MarketSession
    MARKET_TIME_AVAILABLE = True
except ImportError:
    MARKET_TIME_AVAILABLE = False

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

@dataclass
class DayTradingRecommendation:
    """デイトレード推奨"""
    symbol: str
    name: str
    signal: DayTradingSignal
    entry_timing: str          # エントリータイミング
    target_profit: float       # 目標利益率(%)
    stop_loss: float          # 損切りライン(%)
    holding_time: str         # 推奨保有時間
    confidence: float         # 信頼度
    risk_level: str          # リスクレベル
    volume_trend: str        # 出来高動向
    price_momentum: str      # 価格モメンタム
    intraday_volatility: float  # 日中ボラティリティ
    market_timing_score: float  # 市場タイミングスコア

class PersonalDayTradingEngine:
    """個人向けデイトレードエンジン"""

    def __init__(self):
        # 動的銘柄取得システム
        self.daytrading_symbols = self._load_dynamic_symbols()

        # 市場時間管理システム初期化
        if MARKET_TIME_AVAILABLE:
            self.market_manager = MarketTimeManager()
            self.time_mode = "ACCURATE"
        else:
            self.market_manager = None
            self.time_mode = "SIMPLE"

        self.current_session = self._get_current_trading_session()

        # 実データプロバイダー初期化
        if REAL_DATA_AVAILABLE:
            self.real_data_engine = RealDataAnalysisEngine()
            self.data_mode = "REAL"
        else:
            self.real_data_engine = None
            self.data_mode = "DEMO"
            
    def _load_dynamic_symbols(self) -> dict:
        """動的銘柄取得"""
        try:
            from src.day_trade.data.symbol_selector import DynamicSymbolSelector
            selector = DynamicSymbolSelector()
            
            # デイトレード向け高流動性銘柄を取得
            symbols = selector.get_liquid_symbols(limit=20)
            
            # 辞書形式に変換（名前は簡易版）
            symbol_dict = {}
            for symbol in symbols:
                # 簡易的な名前マッピング（本来はDBから取得すべき）
                name_map = {
                    "7203": "トヨタ自動車", "8306": "三菱UFJ", "6758": "ソニーG",
                    "9984": "ソフトバンクG", "4751": "サイバーエージェント", "6861": "キーエンス",
                    "4689": "LINEヤフー", "7974": "任天堂", "8058": "三菱商事",
                    "1605": "INPEX", "6098": "リクルート", "8001": "伊藤忠商事"
                }
                symbol_dict[symbol] = name_map.get(symbol, f"銘柄{symbol}")
            
            print(f"✅ 動的銘柄取得成功: {len(symbol_dict)}銘柄")
            return symbol_dict
            
        except Exception as e:
            print(f"❌ 動的銘柄取得失敗: {e}")
            raise RuntimeError(f"デイトレード銘柄の取得に失敗しました: {e}") from e

        # MLモデルの初期化とロード (仮実装)
        try:
            self.ml_model = RandomForestModel() # configは後で追加
            self.data_fetcher = YFinanceFetcher() # YFinanceFetcherの初期化
        except ImportError:
            self.ml_model = None
            self.data_fetcher = None

    def _get_current_trading_session(self) -> TradingSession:
        """現在の取引時間帯を取得（正確な市場時間管理）"""
        # 正確な市場時間管理システム使用
        if self.market_manager:
            market_session = self.market_manager.get_current_session()

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

    async def get_today_daytrading_recommendations(self, limit: int = 20) -> List[DayTradingRecommendation]:
        """今日のデイトレード推奨取得（実データ対応）"""
        current_session = self._get_current_trading_session()

        # 大引け後は翌日前場予想モードに切り替え
        if current_session == TradingSession.AFTER_MARKET:
            return await self.get_tomorrow_premarket_forecast(limit)

        # 実データエンジン使用可能な場合
        if self.real_data_engine:
            real_recommendations = await self.real_data_engine.analyze_daytrading_opportunities(limit)
            return [self._convert_to_daytrading_recommendation(rec, current_session) for rec in real_recommendations]

        # フォールバック: ダミーデータ使用
        recommendations = []
        symbols = list(self.daytrading_symbols.keys())[:limit]

        for symbol in symbols:
            rec = await self._analyze_daytrading_opportunity(symbol, current_session)
            recommendations.append(rec)

        # 市場タイミングスコア順でソート
        recommendations.sort(key=lambda x: x.market_timing_score, reverse=True)

        return recommendations

    def _convert_to_daytrading_recommendation(self, real_rec: Dict[str, any], session: TradingSession) -> DayTradingRecommendation:
        """実データ推奨をDayTradingRecommendationに変換"""

        # シグナル変換
        signal_map = {
            "★強い買い★": DayTradingSignal.STRONG_BUY,
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
        change_pct = real_rec.get("change_percent", 0)
        rsi = real_rec.get("rsi", 50)

        if change_pct > 2.0 and rsi < 70:
            return "強い上昇モメンタム"
        elif change_pct > 0 and rsi < 60:
            return "上昇基調"
        elif change_pct < -2.0 and rsi > 30:
            return "下落からの反転期待"
        else:
            return "モメンタム中立"

    async def get_tomorrow_premarket_forecast(self, limit: int = 20) -> List[DayTradingRecommendation]:
        """翌日前場予想取得（大引け後専用）"""
        recommendations = []
        symbols = list(self.daytrading_symbols.keys())[:limit]

        # 翌日の日付を生成
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime('%Y%m%d')

        for symbol in symbols:
            rec = await self._analyze_tomorrow_premarket_opportunity(symbol, tomorrow_str)
            recommendations.append(rec)

        # 市場タイミングスコア順でソート
        recommendations.sort(key=lambda x: x.market_timing_score, reverse=True)
        return recommendations

    async def _analyze_tomorrow_premarket_opportunity(self, symbol: str, tomorrow_str: str) -> DayTradingRecommendation:
        """翌日前場機会分析"""
        symbol_name = self.daytrading_symbols[symbol]

        # AIモデルによる予測に置き換え
        # 1. 予測に必要な市場データを取得
        market_data = self._fetch_mock_market_data(symbol, tomorrow_str)

        # 2. 取得したデータから特徴量を生成
        features = self._prepare_features_for_prediction(market_data)

        # 3. AIモデルで予測
        # predictions配列の各要素が、オーバーナイトギャップ、プレマーケットモメンタムなどの予測値に対応すると仮定
        # この部分のインデックスと意味は、実際のMLモデルの実装に依存
        # 例: predictions[0] = overnight_gap, predictions[1] = premarket_momentum など
        # 今回は暫定的に、予測結果の最初のいくつかの要素を割り当てる
        prediction_results = self.ml_model.predict(features)

        # predictionsとconfidenceの形状を確認し、適切に割り当てる
        # ここでは単一の予測値を想定
        predicted_values = prediction_results.predictions.flatten()
        confidence_values = prediction_results.confidence.flatten() if prediction_results.confidence is not None else np.array([0.0])

        overnight_gap = predicted_values[0] if len(predicted_values) > 0 else 0.0
        premarket_momentum = predicted_values[1] if len(predicted_values) > 1 else 0.0
        volume_expectation = predicted_values[2] if len(predicted_values) > 2 else 0.0
        volatility_forecast = predicted_values[3] if len(predicted_values) > 3 else 0.0

        # 信頼度を考慮した上で、ランダム性を残すか、完全にAI予測に置き換えるかを検討する
        # ここではAI予測を優先
        confidence = 75.0 # AIモデルの信頼度をここに反映させる

        # 翌日前場シグナル判定
        signal = self._determine_tomorrow_premarket_signal(
            overnight_gap, premarket_momentum, volume_expectation, volatility_forecast
        )

        # 寄り付き戦略
        entry_timing = self._get_tomorrow_entry_strategy(signal, overnight_gap)

        # 翌日用利確・損切り設定
        target_profit, stop_loss = self._calculate_premarket_profit_stop_levels(
            signal, volatility_forecast
        )

        # 前場推奨保有時間
        holding_time = self._get_premarket_holding_time(signal)

        # 翌日予想信頼度 (AIモデルの信頼度を使用)
        # prediction_results.confidenceが存在し、かつ値が1つ以上ある場合、その平均値を使用
        # 存在しない場合、または値がない場合は、以前の計算ロジックをフォールバックとして使用するか、固定値を割り当てる
        if confidence_values.size > 0:
            confidence = np.mean(confidence_values) * 100 # %表示にするため
        else:
            # フォールバックとして以前のロジックを使用
            confidence = self._calculate_premarket_confidence(
                signal, volatility_forecast, volume_expectation
            )

        # リスク評価（オーバーナイト考慮）
        risk_level = self._assess_premarket_risk(volatility_forecast, overnight_gap)

        # 市場動向説明
        volume_trend = self._describe_volume_trend(volume_expectation)
        momentum_desc = self._describe_overnight_momentum(overnight_gap, premarket_momentum)

        # 翌日タイミングスコア
        market_timing_score = self._calculate_premarket_timing_score(
            signal, volatility_forecast, volume_expectation
        )

        return DayTradingRecommendation(
            symbol=symbol,
            name=symbol_name,
            signal=signal,
            entry_timing=entry_timing,
            target_profit=target_profit,
            stop_loss=stop_loss,
            holding_time=holding_time,
            confidence=confidence,
            risk_level=risk_level,
            volume_trend=volume_trend,
            price_momentum=momentum_desc,
            intraday_volatility=volatility_forecast,
            market_timing_score=market_timing_score
        )

    def _determine_tomorrow_premarket_signal(self, overnight_gap: float, momentum: float,
                                           volume: float, volatility: float) -> DayTradingSignal:
        """翌日前場シグナル判定"""
        # ポジティブギャップ + 強いモメンタム = 強い買い
        if overnight_gap >= 1.0 and momentum >= 1.0 and volume >= 1.5:
            return DayTradingSignal.STRONG_BUY

        # ネガティブギャップ + 弱いモメンタム = 強い売り
        elif overnight_gap <= -1.0 and momentum <= -1.0 and volume >= 1.5:
            return DayTradingSignal.STRONG_SELL

        # 中程度のポジティブ要因
        elif overnight_gap >= 0.5 or momentum >= 0.5:
            return DayTradingSignal.BUY

        # 中程度のネガティブ要因
        elif overnight_gap <= -0.5 or momentum <= -0.5:
            return DayTradingSignal.SELL

        # 方向感不明
        elif abs(overnight_gap) < 0.3 and abs(momentum) < 0.3:
            return DayTradingSignal.WAIT

        else:
            return DayTradingSignal.HOLD

    def _get_tomorrow_entry_strategy(self, signal: DayTradingSignal, overnight_gap: float) -> str:
        """翌日エントリー戦略"""
        gap_direction = "上ギャップ" if overnight_gap > 0 else "下ギャップ" if overnight_gap < -0.5 else "フラット"

        strategy_map = {
            DayTradingSignal.STRONG_BUY: f"寄り成行で積極エントリー（{gap_direction}想定）",
            DayTradingSignal.BUY: f"寄り後5-10分様子見してエントリー（{gap_direction}）",
            DayTradingSignal.STRONG_SELL: f"寄り成行で売りエントリー（{gap_direction}想定）",
            DayTradingSignal.SELL: f"戻り売りタイミング狙い（{gap_direction}）",
            DayTradingSignal.HOLD: f"寄り後の流れ確認してから判断（{gap_direction}）",
            DayTradingSignal.WAIT: f"前場中盤まで様子見推奨（{gap_direction}）"
        }

        return strategy_map.get(signal, "寄り付き後の値動き確認")

    def _calculate_premarket_profit_stop_levels(self, signal: DayTradingSignal,
                                              volatility: float) -> tuple[float, float]:
        """前場用利確・損切りライン"""
        # ボラティリティベース + オーバーナイトリスク考慮
        base_profit = min(volatility * 0.7, 3.5)  # 少し大きめの利確目標
        base_stop = min(volatility * 0.5, 2.5)    # オーバーナイトリスク考慮

        signal_adjustments = {
            DayTradingSignal.STRONG_BUY: (base_profit * 1.3, base_stop * 0.9),
            DayTradingSignal.BUY: (base_profit * 1.1, base_stop),
            DayTradingSignal.STRONG_SELL: (base_profit * 1.3, base_stop * 0.9),
            DayTradingSignal.SELL: (base_profit * 1.1, base_stop),
            DayTradingSignal.HOLD: (base_profit * 0.8, base_stop * 1.1),
            DayTradingSignal.WAIT: (base_profit * 0.6, base_stop * 0.8)
        }

        profit, stop = signal_adjustments.get(signal, (base_profit, base_stop))
        return round(profit, 1), round(stop, 1)

    def _get_premarket_holding_time(self, signal: DayTradingSignal) -> str:
        """前場推奨保有時間"""
        time_map = {
            DayTradingSignal.STRONG_BUY: "寄り〜前場終了まで（最大2時間30分）",
            DayTradingSignal.BUY: "寄り後1〜2時間程度",
            DayTradingSignal.STRONG_SELL: "即座決済〜30分以内",
            DayTradingSignal.SELL: "30分〜1時間での決済",
            DayTradingSignal.HOLD: "前場中の流れ次第",
            DayTradingSignal.WAIT: "エントリー機会待ち"
        }
        return time_map.get(signal, "前場中の判断")

    def _calculate_premarket_confidence(self, signal: DayTradingSignal,
                                      volatility: float, volume: float) -> float:
        """前場予想信頼度"""
        base_confidence = 65.0  # 予想なので基本信頼度を下げる

        signal_boost = {
            DayTradingSignal.STRONG_BUY: 20,
            DayTradingSignal.BUY: 15,
            DayTradingSignal.STRONG_SELL: 20,
            DayTradingSignal.SELL: 15,
            DayTradingSignal.HOLD: 5,
            DayTradingSignal.WAIT: 0
        }.get(signal, 0)

        volume_boost = min((volume - 1.0) * 8, 12)
        volatility_boost = 8 if 3.0 <= volatility <= 5.0 else 3

        final_confidence = base_confidence + signal_boost + volume_boost + volatility_boost
        return max(50, min(85, final_confidence))  # 予想なので最大85%

    def _assess_premarket_risk(self, volatility: float, overnight_gap: float) -> str:
        """前場リスク評価（オーバーナイト考慮）"""
        if volatility >= 5.0 or abs(overnight_gap) >= 1.5:
            return "高"
        elif volatility >= 3.5 or abs(overnight_gap) >= 0.8:
            return "中"
        else:
            return "低"

    def _describe_overnight_momentum(self, gap: float, momentum: float) -> str:
        """オーバーナイト・前場モメンタム説明"""
        gap_desc = "上ギャップ期待" if gap > 0.5 else "下ギャップ警戒" if gap < -0.5 else "ギャップなし"
        momentum_desc = "強い上昇期待" if momentum > 1.0 else "上昇期待" if momentum > 0.5 else \
                       "強い下落警戒" if momentum < -1.0 else "下落警戒" if momentum < -0.5 else "横ばい"
        return f"{gap_desc}・{momentum_desc}"

    def _calculate_premarket_timing_score(self, signal: DayTradingSignal,
                                        volatility: float, volume: float) -> float:
        """翌日前場タイミングスコア"""
        base_score = {
            DayTradingSignal.STRONG_BUY: 85,
            DayTradingSignal.BUY: 75,
            DayTradingSignal.STRONG_SELL: 80,
            DayTradingSignal.SELL: 70,
            DayTradingSignal.HOLD: 50,
            DayTradingSignal.WAIT: 30
        }.get(signal, 40)

        # ボラティリティ・出来高調整
        vol_adj = 10 if 3.0 <= volatility <= 5.0 else 0
        volume_adj = min((volume - 1.0) * 15, 10)

        final_score = base_score + vol_adj + volume_adj
        return max(0, min(100, final_score))

    async def _analyze_daytrading_opportunity(self, symbol: str, session: TradingSession) -> DayTradingRecommendation:
        """デイトレード機会分析"""
        symbol_name = self.daytrading_symbols[symbol]

        # AIモデルによる予測に置き換え
        # 1. 予測に必要な市場データを取得
        market_data = self._fetch_mock_market_data(symbol, str(datetime.now().date()).replace('-', '')) # YYYYMMDD形式に変換

        # 2. 取得したデータから特徴量を生成
        features = self._prepare_features_for_prediction(market_data)

        # 3. AIモデルで予測
        # predictions配列の各要素が、日中ボラティリティ、出来高比率、価格モメンタムなどの予測値に対応すると仮定
        prediction_results = self.ml_model.predict(features)

        predicted_values = prediction_results.predictions.flatten()
        confidence_values = prediction_results.confidence.flatten() if prediction_results.confidence is not None else np.array([0.0])

        intraday_volatility = predicted_values[0] if len(predicted_values) > 0 else 0.0
        volume_ratio = predicted_values[1] if len(predicted_values) > 1 else 0.0
        price_momentum = predicted_values[2] if len(predicted_values) > 2 else 0.0

        # 時間帯別補正
        session_multiplier = self._get_session_multiplier(session)
        adjusted_volatility = intraday_volatility * session_multiplier

        # シグナル判定
        signal = self._determine_daytrading_signal(
            volatility=adjusted_volatility,
            momentum=price_momentum,
            volume=volume_ratio,
            session=session
        )

        # エントリータイミング
        entry_timing = self._get_entry_timing(signal, session)

        # 利確・損切りライン設定
        target_profit, stop_loss = self._calculate_profit_stop_levels(
            signal, adjusted_volatility
        )

        # 推奨保有時間
        holding_time = self._get_recommended_holding_time(signal, session)

        # 信頼度計算（AIモデルの信頼度を使用）
        if confidence_values.size > 0:
            confidence = np.mean(confidence_values) * 100 # %表示にするため
        else:
            # フォールバックとして以前のロジックを使用
            confidence = self._calculate_daytrading_confidence(
                signal, adjusted_volatility, volume_ratio, session
            )

        # リスクレベル
        risk_level = self._assess_daytrading_risk(adjusted_volatility, signal)

        # 出来高・価格動向の文字化
        volume_trend = self._describe_volume_trend(volume_ratio)
        momentum_desc = self._describe_price_momentum(price_momentum)

        # 市場タイミングスコア（デイトレード適性度）
        market_timing_score = self._calculate_market_timing_score(
            signal, adjusted_volatility, volume_ratio, session
        )

        return DayTradingRecommendation(
            symbol=symbol,
            name=symbol_name,
            signal=signal,
            entry_timing=entry_timing,
            target_profit=target_profit,
            stop_loss=stop_loss,
            holding_time=holding_time,
            confidence=confidence,
            risk_level=risk_level,
            volume_trend=volume_trend,
            price_momentum=momentum_desc,
            intraday_volatility=adjusted_volatility,
            market_timing_score=market_timing_score
        )

    def _get_session_multiplier(self, session: TradingSession) -> float:
        """時間帯別の動きやすさ係数"""
        multipliers = {
            TradingSession.PRE_MARKET: 0.5,        # 寄り前は動き小さい
            TradingSession.MORNING_SESSION: 1.3,    # 前場は活発
            TradingSession.LUNCH_BREAK: 0.3,       # 昼休みは動かない
            TradingSession.AFTERNOON_SESSION: 1.0,  # 後場は標準
            TradingSession.AFTER_MARKET: 0.4       # 引け後は限定的
        }
        return multipliers.get(session, 1.0)

    def _fetch_mock_market_data(self, symbol: str, date_str: str) -> Dict[str, Any]:
        """
        市場データ取得メソッド
        YFinanceFetcherを使用して過去の市場データを取得する
        """
        # YFinanceFetcherを使用して履歴データを取得
        # 'date_str'はYYYYMMDD形式なので、YYYY-MM-DDに変換
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        # 過去数日間のデータが必要な場合を考慮し、期間と間隔を設定
        # ここでは過去1日分のデータを取得する例 (日中ボラティリティなどを計算するため)
        # 実際には、AIモデルが必要とする期間のデータを取得するように調整

        # for simplicity, let's fetch one day data
        # To get overnight gap, we might need previous day's closing price
        # For now, let's just get 5 days of data to simulate some history
        historical_data = self.data_fetcher.get_historical_data(
            code=symbol, period="5d", interval="1d"
        )

        if historical_data is not None and not historical_data.empty:
            # 最新のデータを取得
            latest_data = historical_data.iloc[-1]
            # 昨日の終値 (一つ前のデータ)
            prev_close = historical_data.iloc[-2]['終値'] if len(historical_data) >= 2 else latest_data['終値'] # Fallback if not enough data

            return {
                "Open": latest_data.get("始値", 0),
                "High": latest_data.get("高値", 0),
                "Low": latest_data.get("安値", 0),
                "Close": latest_data.get("終値", 0),
                "Volume": latest_data.get("出来高", 0),
                "PrevClose": prev_close,
                "DateTime": latest_data.name # Index contains datetime
            }
        else:
            # データ取得失敗時、またはデータがない場合のフォールバック
            # ここでは依然としてモックデータを返す
            np.random.seed(hash(symbol + date_str + "market_data") % 100000)
            return {
                "Open": np.random.uniform(1000, 2000),
                "High": np.random.uniform(2000, 2100),
                "Low": np.random.uniform(900, 1000),
                "Close": np.random.uniform(1000, 2000),
                "Volume": int(np.random.uniform(1_000_000, 10_000_000)),
                "PrevClose": np.random.uniform(1000, 2000),
            }

    def _prepare_features_for_prediction(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        予測用特徴量準備メソッド
        RandomForestモデルが必要とする形式にデータを変換する。
        ここでは、簡易的なテクニカル指標と価格・出来高特徴量を生成。
        """
        # 必要なデータポイントが存在するか確認
        if not all(k in market_data for k in ["Open", "High", "Low", "Close", "Volume", "PrevClose"]):
            # データ不足の場合、ゼロ配列を返すか、エラーハンドリングを行う
            # ここでは暫定的に、num_featuresに合うようにゼロ埋めされた配列を返す
            num_features = 10  # モデルが期待する特徴量の数
            return np.zeros((1, num_features))

        open_p = market_data["Open"]
        high_p = market_data["High"]
        low_p = market_data["Low"]
        close_p = market_data["Close"]
        volume = market_data["Volume"]
        prev_close = market_data["PrevClose"]

        # 簡易的な特徴量エンジニアリング
        # 価格特徴量
        price_change = close_p - prev_close
        daily_range = high_p - low_p

        # 出来高特徴量 (ここでは単純な出来高を特徴量として使う)

        # テクニカル指標 (ここでは簡易的なRSIとMACDをモックとして生成)
        # 実際には src/day_trade/analysis/signals.py などからインポートして利用
        # RSIを模倣
        rsi_mock = 50 + (price_change / max(1, daily_range)) * 10
        rsi_mock = np.clip(rsi_mock, 0, 100) # RSIは0-100の範囲

        # MACDを模倣
        macd_mock = (close_p - prev_close) * 10
        macd_mock = np.clip(macd_mock, -100, 100) # 適当な範囲

        # モデルが期待する特徴量の順序に合わせて配列を作成
        # RandomForestModelは (n_samples, n_features) 形式を期待
        # 例: [price_change, daily_range, volume, rsi_mock, macd_mock, ...]
        # 現状は5つの特徴量
        features_array = np.array([
            price_change,
            daily_range,
            volume,
            rsi_mock,
            macd_mock,
            open_p, high_p, low_p, close_p, prev_close # 他の価格情報も特徴量として含める
        ]).reshape(1, -1) # 1サンプル、n_features

        # RandomForestModelの_hyperparameter_optimizationでnum_features=10としていたので、ここで合わせる
        # 実際には、config/ml.json の features に基づいて、適切な特徴量エンジニアリングパイプラインを構築する必要がある

        # もし特徴量の数が足りない場合、ゼロ埋めするなどの対応が必要
        num_expected_features = 10 # RandomForestModelが期待する特徴量の数
        if features_array.shape[1] < num_expected_features:
            padding = np.zeros((1, num_expected_features - features_array.shape[1]))
            features_array = np.hstack((features_array, padding))
        elif features_array.shape[1] > num_expected_features:
            features_array = features_array[:, :num_expected_features] # 多すぎる場合は切り詰める

        return features_array

    def _determine_daytrading_signal(self, volatility: float, momentum: float,
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

    def _get_entry_timing(self, signal: DayTradingSignal, session: TradingSession) -> str:
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

    def _calculate_profit_stop_levels(self, signal: DayTradingSignal,
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

    def _get_recommended_holding_time(self, signal: DayTradingSignal,
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

    def _calculate_daytrading_confidence(self, signal: DayTradingSignal,
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

    def _assess_daytrading_risk(self, volatility: float, signal: DayTradingSignal) -> str:
        """デイトレードリスク評価"""

        if volatility >= 6.0:
            return "高"
        elif volatility >= 4.0:
            return "中"
        else:
            return "低"

    def _describe_volume_trend(self, volume_ratio: float) -> str:
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

    def _describe_price_momentum(self, momentum: float) -> str:
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

    def _calculate_market_timing_score(self, signal: DayTradingSignal,
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

    def get_session_advice(self) -> str:
        """現在の時間帯に応じたアドバイス（正確な市場時間管理）"""
        # 正確な市場時間管理システム使用
        if self.market_manager:
            return self.market_manager.get_session_advice()

        # フォールバック: 従来のアドバイス
        session = self._get_current_trading_session()
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

# デイトレード機能のデモ
async def demo_daytrading_engine():
    """デイトレードエンジンデモ"""
    print("=== Day Trade Personal - デイトレード推奨エンジン ===")

    engine = PersonalDayTradingEngine()

    # 現在の時間帯アドバイス
    print(engine.get_session_advice())
    print()

    # 今日のデイトレード推奨
    print("今日のデイトレード推奨 TOP5:")
    print("-" * 50)

    recommendations = await engine.get_today_daytrading_recommendations(limit=20)

    for i, rec in enumerate(recommendations, 1):
        signal_icon = {
            DayTradingSignal.STRONG_BUY: "[★強い買い★]",
            DayTradingSignal.BUY: "[●買い●]",
            DayTradingSignal.STRONG_SELL: "[▼強い売り▼]",
            DayTradingSignal.SELL: "[▽売り▽]",
            DayTradingSignal.HOLD: "[■ホールド■]",
            DayTradingSignal.WAIT: "[…待機…]"
        }.get(rec.signal, "[?]")

        print(f"{i}. {rec.symbol} ({rec.name})")
        print(f"   シグナル: {signal_icon}")
        print(f"   エントリー: {rec.entry_timing}")
        print(f"   目標利確: +{rec.target_profit}% / 損切り: -{rec.stop_loss}%")
        print(f"   保有時間: {rec.holding_time}")
        print(f"   信頼度: {rec.confidence:.0f}% | リスク: {rec.risk_level}")
        print(f"   出来高: {rec.volume_trend} | 値動き: {rec.price_momentum}")
        print(f"   タイミングスコア: {rec.market_timing_score:.0f}/100")
        print()

if __name__ == "__main__":
    asyncio.run(demo_daytrading_engine())