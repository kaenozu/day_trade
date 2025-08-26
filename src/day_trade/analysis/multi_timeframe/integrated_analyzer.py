#!/usr/bin/env python3
"""
マルチタイムフレーム分析システム - 統合分析モジュール
Issue #315: 高度テクニカル指標・ML機能拡張

複数時間軸の統合分析とシグナル生成機能を提供
"""

import warnings
from typing import Dict

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class IntegratedAnalyzer:
    """
    統合分析クラス
    
    複数時間軸の分析結果を統合してシグナルを生成する機能を提供
    """
    
    def __init__(self):
        """初期化"""
        self.timeframe_weights = {
            "daily": 0.4,
            "weekly": 0.35,
            "monthly": 0.25,
        }
        
        logger.info("統合分析システム初期化完了")
    
    def perform_integrated_analysis(
        self, timeframe_results: Dict[str, Dict]
    ) -> Dict[str, any]:
        """
        統合分析実行
        
        Args:
            timeframe_results: 各時間軸の分析結果
            
        Returns:
            統合分析結果辞書
        """
        try:
            # トレンド方向の統合判定
            trend_votes = {}
            trend_weights = {}

            for tf_key, tf_data in timeframe_results.items():
                trend = tf_data.get("trend_direction", "sideways")
                strength = tf_data.get("trend_strength", 50)
                weight = self.timeframe_weights.get(tf_key, 0.33)

                # トレンド投票
                if trend not in trend_votes:
                    trend_votes[trend] = 0
                    trend_weights[trend] = 0

                trend_votes[trend] += weight
                trend_weights[trend] += weight * (strength / 100)

            # 最有力トレンドの特定
            if trend_votes:
                dominant_trend = max(trend_votes.keys(), key=lambda x: trend_votes[x])
                trend_confidence = trend_votes[dominant_trend] * 100

                # 強度重み付け
                weighted_confidence = trend_weights.get(dominant_trend, 0) * 100
            else:
                dominant_trend = "sideways"
                trend_confidence = 0
                weighted_confidence = 0

            # 時間軸間の整合性チェック
            consistency_score = self._calculate_timeframe_consistency(timeframe_results)

            # 統合シグナル生成
            integrated_signal = self._generate_integrated_signal(
                dominant_trend,
                weighted_confidence,
                consistency_score,
                timeframe_results,
            )

            return {
                "overall_trend": dominant_trend,
                "trend_confidence": float(weighted_confidence),
                "consistency_score": float(consistency_score),
                "integrated_signal": integrated_signal,
                "timeframe_agreement": self._analyze_timeframe_agreement(
                    timeframe_results
                ),
            }

        except Exception as e:
            logger.error(f"統合分析エラー: {e}")
            return {"overall_trend": "error", "error": str(e)}
    
    def _calculate_timeframe_consistency(
        self, timeframe_results: Dict[str, Dict]
    ) -> float:
        """
        時間軸間の整合性スコア計算（0-100）
        
        Args:
            timeframe_results: 時間軸別分析結果
            
        Returns:
            整合性スコア
        """
        try:
            if len(timeframe_results) < 2:
                return 0

            # トレンド方向の一致度
            trends = [tf["trend_direction"] for tf in timeframe_results.values()]
            unique_trends = set(trends)

            if len(unique_trends) == 1:
                trend_consistency = 100
            elif len(unique_trends) == 2:
                trend_consistency = 50
            else:
                trend_consistency = 0

            # テクニカル指標の一致度
            indicator_consistency = 0
            indicator_count = 0

            # RSIの一致度（全て過買われ、過売われ、中立で一致するか）
            rsi_values = []
            for tf_data in timeframe_results.values():
                rsi = tf_data.get("technical_indicators", {}).get("rsi")
                if rsi is not None:
                    if rsi > 70:
                        rsi_values.append("overbought")
                    elif rsi < 30:
                        rsi_values.append("oversold")
                    else:
                        rsi_values.append("neutral")

            if len(rsi_values) >= 2:
                if len(set(rsi_values)) == 1:
                    indicator_consistency += 30
                elif len(set(rsi_values)) == 2:
                    indicator_consistency += 10
                indicator_count += 1

            # MACDシグナルの一致度
            macd_signals = []
            for tf_data in timeframe_results.values():
                macd = tf_data.get("technical_indicators", {}).get("macd")
                if macd is not None:
                    macd_signals.append("positive" if macd > 0 else "negative")

            if len(macd_signals) >= 2:
                if len(set(macd_signals)) == 1:
                    indicator_consistency += 20
                indicator_count += 1

            # 一目均衡表の一致度
            ichimoku_signals = []
            for tf_data in timeframe_results.values():
                ichimoku = tf_data.get("ichimoku_signal")
                if ichimoku:
                    if ichimoku in ["buy", "strong_buy"]:
                        ichimoku_signals.append("bullish")
                    elif ichimoku in ["sell", "strong_sell"]:
                        ichimoku_signals.append("bearish")
                    else:
                        ichimoku_signals.append("neutral")

            if len(ichimoku_signals) >= 2:
                if len(set(ichimoku_signals)) == 1:
                    indicator_consistency += 30
                elif len(set(ichimoku_signals)) == 2:
                    indicator_consistency += 10
                indicator_count += 1

            # 平均化
            if indicator_count > 0:
                indicator_consistency = indicator_consistency / indicator_count

            # 全体の整合性スコア
            overall_consistency = trend_consistency * 0.6 + indicator_consistency * 0.4

            return max(0, min(100, overall_consistency))

        except Exception as e:
            logger.error(f"整合性計算エラー: {e}")
            return 0
    
    def _generate_integrated_signal(
        self,
        dominant_trend: str,
        confidence: float,
        consistency: float,
        timeframe_results: Dict[str, Dict],
    ) -> Dict[str, any]:
        """
        統合シグナル生成
        
        Args:
            dominant_trend: 主要トレンド
            confidence: 信頼度
            consistency: 整合性
            timeframe_results: 時間軸別結果
            
        Returns:
            統合シグナル辞書
        """
        try:
            # 基本シグナル判定
            if (
                dominant_trend in ["strong_uptrend", "uptrend"]
                and confidence >= 60
                and consistency >= 70
            ):
                signal_action = "BUY"
                signal_strength = "STRONG"
            elif dominant_trend in ["strong_uptrend", "uptrend"] and confidence >= 40:
                signal_action = "BUY"
                signal_strength = "MODERATE"
            elif (
                dominant_trend in ["strong_downtrend", "downtrend"]
                and confidence >= 60
                and consistency >= 70
            ):
                signal_action = "SELL"
                signal_strength = "STRONG"
            elif (
                dominant_trend in ["strong_downtrend", "downtrend"] and confidence >= 40
            ):
                signal_action = "SELL"
                signal_strength = "MODERATE"
            else:
                signal_action = "HOLD"
                signal_strength = "WEAK"

            # 調整要因チェック
            adjustment_factors = []

            # 短期と長期の不一致チェック
            if "daily" in timeframe_results and "monthly" in timeframe_results:
                daily_trend = timeframe_results["daily"]["trend_direction"]
                monthly_trend = timeframe_results["monthly"]["trend_direction"]

                if daily_trend != monthly_trend:
                    if daily_trend in [
                        "strong_downtrend",
                        "downtrend",
                    ] and monthly_trend in ["uptrend", "strong_uptrend"]:
                        adjustment_factors.append("短期下落・長期上昇の調整局面")
                        if signal_action == "SELL":
                            signal_strength = "WEAK"
                    elif daily_trend in [
                        "uptrend",
                        "strong_uptrend",
                    ] and monthly_trend in ["downtrend", "strong_downtrend"]:
                        adjustment_factors.append("短期上昇・長期下落の調整局面")
                        if signal_action == "BUY":
                            signal_strength = "WEAK"

            # オーバーボート・オーバーソールドチェック
            extreme_rsi_count = 0
            for tf_data in timeframe_results.values():
                rsi = tf_data.get("technical_indicators", {}).get("rsi")
                if rsi is not None:
                    if rsi > 80:
                        extreme_rsi_count += 1
                        adjustment_factors.append("RSI過買われ水準")
                    elif rsi < 20:
                        extreme_rsi_count += 1
                        adjustment_factors.append("RSI過売られ水準")

            if extreme_rsi_count >= 2:  # 複数時間軸で極端
                if signal_action in ["BUY", "SELL"]:
                    signal_strength = (
                        "MODERATE" if signal_strength == "STRONG" else "WEAK"
                    )

            return {
                "action": signal_action,
                "strength": signal_strength,
                "confidence": float(confidence),
                "consistency": float(consistency),
                "dominant_trend": dominant_trend,
                "adjustment_factors": adjustment_factors,
                "signal_score": self._calculate_signal_score(
                    signal_action, signal_strength, confidence, consistency
                ),
            }

        except Exception as e:
            logger.error(f"統合シグナル生成エラー: {e}")
            return {
                "action": "HOLD",
                "strength": "WEAK",
                "confidence": 0,
                "consistency": 0,
                "error": str(e),
            }
    
    def _calculate_signal_score(
        self, action: str, strength: str, confidence: float, consistency: float
    ) -> float:
        """
        シグナルスコア計算（-100 to +100）
        
        Args:
            action: アクション
            strength: 強度
            confidence: 信頼度
            consistency: 整合性
            
        Returns:
            シグナルスコア
        """
        try:
            base_score = 0

            # アクションベーススコア
            if action == "BUY":
                base_score = 50
            elif action == "SELL":
                base_score = -50
            else:  # HOLD
                base_score = 0

            # 強度による調整
            strength_multiplier = {"STRONG": 1.0, "MODERATE": 0.7, "WEAK": 0.4}.get(
                strength, 0.4
            )

            base_score *= strength_multiplier

            # 信頼度による調整
            confidence_adjustment = (confidence / 100) * 0.3
            consistency_adjustment = (consistency / 100) * 0.2

            final_score = base_score * (
                1 + confidence_adjustment + consistency_adjustment
            )

            return max(-100, min(100, final_score))

        except Exception as e:
            logger.error(f"シグナルスコア計算エラー: {e}")
            return 0
    
    def _analyze_timeframe_agreement(
        self, timeframe_results: Dict[str, Dict]
    ) -> Dict[str, any]:
        """
        時間軸合意分析
        
        Args:
            timeframe_results: 時間軸別分析結果
            
        Returns:
            時間軸合意分析結果
        """
        try:
            agreements = {
                "trend_agreement": [],
                "technical_agreement": [],
                "signal_agreement": [],
            }

            # トレンド合意
            trends = [
                (tf, data["trend_direction"]) for tf, data in timeframe_results.items()
            ]
            for i, (tf1, trend1) in enumerate(trends):
                for tf2, trend2 in trends[i + 1 :]:
                    if trend1 == trend2:
                        agreements["trend_agreement"].append(f"{tf1}-{tf2}: {trend1}")

            # テクニカル指標合意
            for tf, data in timeframe_results.items():
                rsi = data.get("technical_indicators", {}).get("rsi")
                if rsi:
                    if rsi > 70:
                        agreements["technical_agreement"].append(f"{tf}: RSI過買われ")
                    elif rsi < 30:
                        agreements["technical_agreement"].append(f"{tf}: RSI過売られ")

            return agreements

        except Exception as e:
            logger.error(f"時間軸合意分析エラー: {e}")
            return {}