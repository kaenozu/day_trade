#!/usr/bin/env python3
"""
マルチタイムフレーム分析システム - リスク評価モジュール
Issue #315: 高度テクニカル指標・ML機能拡張

リスク評価と投資推奨機能を提供
"""

import warnings
from typing import Dict, Optional

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class RiskEvaluator:
    """
    リスク評価クラス
    
    マルチタイムフレームリスク評価と投資推奨機能を提供
    """
    
    def __init__(self):
        """初期化"""
        logger.info("リスク評価システム初期化完了")
    
    def assess_multi_timeframe_risk(
        self, timeframe_results: Dict[str, Dict]
    ) -> Dict[str, any]:
        """
        マルチタイムフレームリスク評価
        
        Args:
            timeframe_results: 時間軸別分析結果
            
        Returns:
            リスク評価結果辞書
        """
        try:
            risk_factors = []
            risk_score = 0  # 0-100, 高いほど危険

            # ボラティリティリスク
            high_vol_count = 0
            for tf_key, tf_data in timeframe_results.items():
                strength = tf_data.get("trend_strength", 50)
                if strength > 80:  # 高強度トレンド = 高ボラティリティ
                    high_vol_count += 1
                    risk_factors.append(f"{tf_data['timeframe']}高ボラティリティ")

            risk_score += high_vol_count * 15

            # トレンド不整合リスク
            trends = [tf["trend_direction"] for tf in timeframe_results.values()]
            unique_trends = len(set(trends))
            if unique_trends >= 3:
                risk_score += 30
                risk_factors.append("時間軸間トレンド不整合")
            elif unique_trends == 2:
                risk_score += 15
                risk_factors.append("一部時間軸トレンド相違")

            # 極端なテクニカル指標リスク
            extreme_indicators = 0
            for tf_data in timeframe_results.values():
                indicators = tf_data.get("technical_indicators", {})

                # 極端なRSI
                rsi = indicators.get("rsi")
                if rsi is not None and (rsi > 85 or rsi < 15):
                    extreme_indicators += 1
                    risk_factors.append(f"極端なRSI({rsi:.1f})")

                # ボリンジャーバンド極端位置
                bb_pos = indicators.get("bb_position")
                if bb_pos is not None and (bb_pos > 0.95 or bb_pos < 0.05):
                    extreme_indicators += 1
                    risk_factors.append("ボリンジャーバンド極端位置")

            risk_score += extreme_indicators * 10

            # サポート・レジスタンス近接リスク
            sr_risk = 0
            for tf_data in timeframe_results.values():
                current_price = tf_data.get("current_price", 0)
                support = tf_data.get("support_level")
                resistance = tf_data.get("resistance_level")

                if support and current_price > 0:
                    support_distance = abs(current_price - support) / current_price
                    if support_distance < 0.02:  # 2%以内
                        sr_risk += 1
                        risk_factors.append(f"{tf_data['timeframe']}サポート近接")

                if resistance and current_price > 0:
                    resistance_distance = (
                        abs(current_price - resistance) / current_price
                    )
                    if resistance_distance < 0.02:  # 2%以内
                        sr_risk += 1
                        risk_factors.append(f"{tf_data['timeframe']}レジスタンス近接")

            risk_score += sr_risk * 8

            # リスクレベル分類
            if risk_score >= 70:
                risk_level = "HIGH"
            elif risk_score >= 40:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                "risk_level": risk_level,
                "risk_score": max(0, min(100, risk_score)),
                "risk_factors": risk_factors,
                "total_risk_factors": len(risk_factors),
            }

        except Exception as e:
            logger.error(f"リスク評価エラー: {e}")
            return {"risk_level": "UNKNOWN", "risk_score": 50, "error": str(e)}
    
    def generate_investment_recommendation(
        self,
        integrated_signal: Dict,
        risk_assessment: Dict,
        timeframe_results: Dict[str, Dict],
    ) -> Dict[str, any]:
        """
        投資推奨生成
        
        Args:
            integrated_signal: 統合シグナル
            risk_assessment: リスク評価
            timeframe_results: 時間軸別結果
            
        Returns:
            投資推奨辞書
        """
        try:
            action = integrated_signal.get("action", "HOLD")
            strength = integrated_signal.get("strength", "WEAK")
            risk_level = risk_assessment.get("risk_level", "MEDIUM")

            # 基本推奨
            if action == "BUY" and strength == "STRONG" and risk_level == "LOW":
                recommendation = "STRONG_BUY"
                position_size = "FULL"
            elif action == "BUY" and (strength == "STRONG" or risk_level == "LOW"):
                recommendation = "BUY"
                position_size = "LARGE"
            elif action == "BUY" and strength == "MODERATE":
                recommendation = "WEAK_BUY"
                position_size = "SMALL"
            elif action == "SELL" and strength == "STRONG" and risk_level == "LOW":
                recommendation = "STRONG_SELL"
                position_size = "FULL"
            elif action == "SELL" and (strength == "STRONG" or risk_level == "LOW"):
                recommendation = "SELL"
                position_size = "LARGE"
            elif action == "SELL" and strength == "MODERATE":
                recommendation = "WEAK_SELL"
                position_size = "SMALL"
            else:
                recommendation = "HOLD"
                position_size = "NEUTRAL"

            # リスク調整
            if risk_level == "HIGH":
                if position_size in ["FULL", "LARGE"]:
                    position_size = "SMALL"
                elif position_size == "SMALL":
                    position_size = "MINIMAL"
                    recommendation = (
                        f"CAUTIOUS_{action}" if action != "HOLD" else "HOLD"
                    )

            # 推奨理由生成
            reasons = []

            # トレンド理由
            dominant_trend = integrated_signal.get("dominant_trend", "sideways")
            if dominant_trend != "sideways":
                reasons.append(f"複数時間軸で{dominant_trend}を確認")

            # 整合性理由
            consistency = integrated_signal.get("consistency", 0)
            if consistency >= 70:
                reasons.append("時間軸間の高い整合性")
            elif consistency < 40:
                reasons.append("時間軸間の整合性に懸念")

            # リスク理由
            risk_factors = risk_assessment.get("risk_factors", [])
            if len(risk_factors) == 0:
                reasons.append("明確なリスク要因なし")
            elif len(risk_factors) >= 3:
                reasons.append("複数のリスク要因を確認")

            # 価格位置理由
            support_breaks = 0
            resistance_breaks = 0
            for tf_data in timeframe_results.values():
                current_price = tf_data.get("current_price", 0)
                support = tf_data.get("support_level")
                resistance = tf_data.get("resistance_level")

                if support and current_price < support:
                    support_breaks += 1
                if resistance and current_price > resistance:
                    resistance_breaks += 1

            if resistance_breaks >= 2:
                reasons.append("複数時間軸でレジスタンス突破")
            elif support_breaks >= 2:
                reasons.append("複数時間軸でサポート割れ")

            return {
                "recommendation": recommendation,
                "position_size": position_size,
                "confidence": integrated_signal.get("confidence", 0),
                "reasons": reasons,
                "holding_period": self._suggest_holding_period(
                    timeframe_results, dominant_trend
                ),
                "stop_loss_suggestion": self._calculate_stop_loss(
                    timeframe_results, action
                ),
                "take_profit_suggestion": self._calculate_take_profit(
                    timeframe_results, action
                ),
            }

        except Exception as e:
            logger.error(f"投資推奨生成エラー: {e}")
            return {
                "recommendation": "HOLD",
                "position_size": "NEUTRAL",
                "error": str(e),
            }
    
    def _suggest_holding_period(
        self, timeframe_results: Dict[str, Dict], dominant_trend: str
    ) -> str:
        """
        保有期間推奨
        
        Args:
            timeframe_results: 時間軸別結果
            dominant_trend: 主要トレンド
            
        Returns:
            推奨保有期間
        """
        try:
            if dominant_trend in ["strong_uptrend", "strong_downtrend"]:
                if "monthly" in timeframe_results:
                    return "LONG_TERM"  # 3-6ヶ月
                elif "weekly" in timeframe_results:
                    return "MEDIUM_TERM"  # 1-3ヶ月
                else:
                    return "SHORT_TERM"  # 1-4週間
            elif dominant_trend in ["uptrend", "downtrend"]:
                return "MEDIUM_TERM"
            else:
                return "SHORT_TERM"

        except Exception:
            return "SHORT_TERM"
    
    def _calculate_stop_loss(
        self, timeframe_results: Dict[str, Dict], action: str
    ) -> Optional[float]:
        """
        ストップロス計算
        
        Args:
            timeframe_results: 時間軸別結果
            action: アクション
            
        Returns:
            ストップロス価格
        """
        try:
            if action not in ["BUY", "SELL"]:
                return None

            current_price = None
            support_level = None
            resistance_level = None

            # 日足データを優先使用
            if "daily" in timeframe_results:
                tf_data = timeframe_results["daily"]
                current_price = tf_data.get("current_price")
                support_level = tf_data.get("support_level")
                resistance_level = tf_data.get("resistance_level")

            if not current_price:
                return None

            if action == "BUY" and support_level:
                # 買いポジション: サポートレベルの少し下
                stop_loss = support_level * 0.98
            elif action == "SELL" and resistance_level:
                # 売りポジション: レジスタンスレベルの少し上
                stop_loss = resistance_level * 1.02
            else:
                # デフォルト: 現在価格の±5%
                multiplier = 0.95 if action == "BUY" else 1.05
                stop_loss = current_price * multiplier

            return float(stop_loss)

        except Exception as e:
            logger.error(f"ストップロス計算エラー: {e}")
            return None
    
    def _calculate_take_profit(
        self, timeframe_results: Dict[str, Dict], action: str
    ) -> Optional[float]:
        """
        利益確定価格計算
        
        Args:
            timeframe_results: 時間軸別結果
            action: アクション
            
        Returns:
            利益確定価格
        """
        try:
            if action not in ["BUY", "SELL"]:
                return None

            current_price = None
            resistance_level = None
            support_level = None

            # 日足データを優先使用
            if "daily" in timeframe_results:
                tf_data = timeframe_results["daily"]
                current_price = tf_data.get("current_price")
                resistance_level = tf_data.get("resistance_level")
                support_level = tf_data.get("support_level")

            if not current_price:
                return None

            if action == "BUY" and resistance_level:
                # 買いポジション: レジスタンスレベルの少し下
                take_profit = resistance_level * 0.98
            elif action == "SELL" and support_level:
                # 売りポジション: サポートレベルの少し上
                take_profit = support_level * 1.02
            else:
                # デフォルト: 現在価格の±10%
                multiplier = 1.10 if action == "BUY" else 0.90
                take_profit = current_price * multiplier

            return float(take_profit)

        except Exception as e:
            logger.error(f"利益確定計算エラー: {e}")
            return None