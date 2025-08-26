#!/usr/bin/env python3
"""
翌日予測モジュール

翌日前場の予測分析機能を提供します。
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .enums import DayTradingSignal, TradingSession
from .recommendation import DayTradingRecommendation


logger = logging.getLogger(__name__)


class TomorrowForecaster:
    """翌日予測クラス"""

    def __init__(self, model_loader=None):
        self.model_loader = model_loader

    async def get_tomorrow_premarket_forecast(self, symbols: dict, limit: int = 20) -> List[DayTradingRecommendation]:
        """翌日前場予想取得（大引け後専用）"""
        recommendations = []
        symbol_list = list(symbols.keys())[:limit]

        # 翌日の日付を生成
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime('%Y%m%d')

        for symbol in symbol_list:
            rec = await self._analyze_tomorrow_premarket_opportunity(symbol, symbols[symbol], tomorrow_str)
            recommendations.append(rec)

        # 市場タイミングスコア順でソート
        recommendations.sort(key=lambda x: x.market_timing_score, reverse=True)
        return recommendations

    async def _analyze_tomorrow_premarket_opportunity(self, symbol: str, symbol_name: str, tomorrow_str: str) -> DayTradingRecommendation:
        """翌日前場機会分析"""
        from .data_handlers import MarketDataHandler, FeaturePreparator

        # モックデータ取得（実際の実装では実データを使用）
        data_handler = MarketDataHandler({})
        market_data = data_handler._generate_intelligent_mock_data(symbol, tomorrow_str)
        
        # 特徴量準備
        feature_preparator = FeaturePreparator()
        features = feature_preparator.prepare_features_for_prediction(market_data)

        # AIモデルによる予測
        if self.model_loader:
            try:
                prediction_results, _ = await self.model_loader.predict(symbol, features)
                return self._build_tomorrow_recommendation_from_prediction(symbol, symbol_name, prediction_results)
            except Exception as e:
                logger.error(f"AI予測失敗 ({symbol}): {e}", exc_info=True)

        # フォールバック予測
        logger.info(f"フォールバック予測を実行します ({symbol})")
        return self._build_fallback_tomorrow_recommendation(symbol, symbol_name, market_data)

    def _build_tomorrow_recommendation_from_prediction(self, symbol: str, symbol_name: str, prediction_results: Any) -> DayTradingRecommendation:
        """AIの予測結果から翌日の推奨を構築する"""
        predicted_values = prediction_results.predictions.flatten()
        confidence_values = prediction_results.confidence.flatten() if prediction_results.confidence is not None else np.array([0.0])

        overnight_gap = predicted_values[0] if len(predicted_values) > 0 else 0.0
        premarket_momentum = predicted_values[1] if len(predicted_values) > 1 else 0.0
        volume_expectation = predicted_values[2] if len(predicted_values) > 2 else 0.0
        volatility_forecast = predicted_values[3] if len(predicted_values) > 3 else 0.0

        signal = self._determine_tomorrow_premarket_signal(
            overnight_gap, premarket_momentum, volume_expectation, volatility_forecast
        )
        entry_timing = self._get_tomorrow_entry_strategy(signal, overnight_gap)
        target_profit, stop_loss = self._calculate_premarket_profit_stop_levels(
            signal, volatility_forecast
        )
        holding_time = self._get_premarket_holding_time(signal)

        if confidence_values.size > 0:
            confidence = np.mean(confidence_values) * 100
        else:
            confidence = self._calculate_premarket_confidence(
                signal, volatility_forecast, volume_expectation
            )

        risk_level = self._assess_premarket_risk(volatility_forecast, overnight_gap)
        volume_trend = self._describe_volume_trend(volume_expectation)
        momentum_desc = self._describe_overnight_momentum(overnight_gap, premarket_momentum)
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

    def _build_fallback_tomorrow_recommendation(self, symbol: str, symbol_name: str, market_data: Dict[str, Any]) -> DayTradingRecommendation:
        """フォールバック翌日推奨"""
        # 簡易的な予測ロジック
        price_change = market_data["Close"] - market_data["PrevClose"]
        volatility = abs(market_data["High"] - market_data["Low"]) / market_data["Close"] * 100 if market_data["Close"] > 0 else 3.0
        
        overnight_gap = (market_data["Open"] - market_data["PrevClose"]) / market_data["PrevClose"] * 100 if market_data["PrevClose"] > 0 else 0.0
        momentum = price_change / market_data["PrevClose"] * 100 if market_data["PrevClose"] > 0 else 0
        avg_volume = market_data.get("AvgVolume", market_data["Volume"])
        volume_ratio = market_data["Volume"] / avg_volume if avg_volume > 0 else 1.0

        signal = self._determine_tomorrow_premarket_signal(overnight_gap, momentum, volume_ratio, volatility)
        
        return DayTradingRecommendation(
            symbol=symbol,
            name=symbol_name,
            signal=signal,
            entry_timing=self._get_tomorrow_entry_strategy(signal, overnight_gap),
            target_profit=round(min(volatility * 0.7, 3.5), 1),
            stop_loss=round(min(volatility * 0.5, 2.5), 1),
            holding_time=self._get_premarket_holding_time(signal),
            confidence=self._calculate_premarket_confidence(signal, volatility, volume_ratio),
            risk_level=self._assess_premarket_risk(volatility, overnight_gap),
            volume_trend=self._describe_volume_trend(volume_ratio),
            price_momentum=self._describe_overnight_momentum(overnight_gap, momentum),
            intraday_volatility=volatility,
            market_timing_score=self._calculate_premarket_timing_score(signal, volatility, volume_ratio)
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

    def _describe_volume_trend(self, volume_ratio: float) -> str:
        """出来高動向説明"""
        if volume_ratio >= 2.0:
            return "出来高急増期待"
        elif volume_ratio >= 1.5:
            return "出来高増加期待"
        elif volume_ratio >= 1.2:
            return "出来高やや多め期待"
        elif volume_ratio >= 0.8:
            return "出来高普通予想"
        else:
            return "出来高少なめ予想"

    def _describe_overnight_momentum(self, gap: float, momentum: float) -> str:
        """オーバーナイト・前場モメンタム説明"""
        gap_desc = "上ギャップ期待" if gap > 0.5 else "下ギャップ警戒" if gap < -0.5 else "フラット"
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