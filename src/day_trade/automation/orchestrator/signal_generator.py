"""
シグナル・アラート生成モジュール
AI分析結果からトレーディングシグナルとスマートアラートを生成
"""

from datetime import datetime
from typing import Any, Dict, List

from ...utils.logging_config import get_context_logger
from .config import AIAnalysisResult, OrchestrationConfig

logger = get_context_logger(__name__)


class SignalGenerator:
    """
    シグナル・アラート生成エンジン
    
    AI分析結果を基にトレーディングシグナルと
    スマートアラートを生成します。
    """

    def __init__(self, config: OrchestrationConfig):
        """
        初期化
        
        Args:
            config: オーケストレーション設定
        """
        self.config = config

    def generate_ai_signals(self, analysis: AIAnalysisResult) -> List[Dict[str, Any]]:
        """
        AIシグナル生成
        
        Args:
            analysis: AI分析結果
            
        Returns:
            List[Dict[str, Any]]: 生成されたシグナルリスト
        """
        signals = []

        try:
            # 基本シグナル
            base_signal = {
                "symbol": analysis.symbol,
                "type": "AI_ANALYSIS",
                "timestamp": analysis.timestamp.isoformat(),
                "source": "next_gen_ai_engine",
                "confidence": analysis.confidence_scores.get("overall", 0.5),
                "recommendation": analysis.recommendation,
                "safe_mode": True,
                "trading_disabled": True,
            }

            # 予測シグナル
            if "predicted_change" in analysis.predictions:
                prediction_signal = base_signal.copy()
                prediction_signal.update(
                    {
                        "type": "PRICE_PREDICTION",
                        "predicted_change": analysis.predictions["predicted_change"],
                        "prediction_horizon": self.config.prediction_horizon,
                        "data_quality": analysis.data_quality,
                    }
                )
                signals.append(prediction_signal)

            # テクニカルシグナル
            if "moving_average" in analysis.technical_signals:
                ma_signal = base_signal.copy()
                ma_signal.update(
                    {
                        "type": "TECHNICAL_SIGNAL",
                        "indicator": "moving_average",
                        "signals": analysis.technical_signals["moving_average"],
                    }
                )
                signals.append(ma_signal)

            # RSIシグナル
            if "rsi" in analysis.technical_signals:
                rsi_signal = base_signal.copy()
                rsi_signal.update(
                    {
                        "type": "TECHNICAL_SIGNAL",
                        "indicator": "rsi",
                        "signals": analysis.technical_signals["rsi"],
                    }
                )
                signals.append(rsi_signal)

            # ボラティリティシグナル
            if "volatility" in analysis.technical_signals:
                vol_signal = base_signal.copy()
                vol_signal.update(
                    {
                        "type": "VOLATILITY_SIGNAL",
                        "indicator": "volatility",
                        "signals": analysis.technical_signals["volatility"],
                    }
                )
                signals.append(vol_signal)

            # リスクアラート
            if analysis.risk_assessment.get("overall_risk_score", 0) > 0.7:
                risk_signal = base_signal.copy()
                risk_signal.update(
                    {
                        "type": "RISK_ALERT",
                        "risk_level": analysis.risk_assessment.get(
                            "risk_level", "unknown"
                        ),
                        "risk_factors": analysis.risk_assessment,
                    }
                )
                signals.append(risk_signal)

            # 高信頼度シグナル
            overall_confidence = analysis.confidence_scores.get("overall", 0)
            if overall_confidence > self.config.confidence_threshold:
                confidence_signal = base_signal.copy()
                confidence_signal.update(
                    {
                        "type": "HIGH_CONFIDENCE_SIGNAL",
                        "confidence_level": overall_confidence,
                        "threshold": self.config.confidence_threshold,
                        "action_required": True,
                    }
                )
                signals.append(confidence_signal)

        except Exception as e:
            logger.error(f"AIシグナル生成エラー {analysis.symbol}: {e}")
            # エラーシグナルを生成
            error_signal = {
                "symbol": analysis.symbol,
                "type": "SIGNAL_GENERATION_ERROR",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "safe_mode": True,
                "trading_disabled": True,
            }
            signals.append(error_signal)

        return signals

    def generate_smart_alerts(
        self, analysis: AIAnalysisResult
    ) -> List[Dict[str, Any]]:
        """
        スマートアラート生成
        
        Args:
            analysis: AI分析結果
            
        Returns:
            List[Dict[str, Any]]: 生成されたアラートリスト
        """
        alerts = []

        try:
            # データ品質アラート
            if analysis.data_quality < self.config.data_quality_threshold:
                alerts.append(
                    {
                        "symbol": analysis.symbol,
                        "type": "DATA_QUALITY_WARNING",
                        "message": f"データ品質低下: {analysis.data_quality:.1f}%",
                        "severity": "medium",
                        "timestamp": datetime.now().isoformat(),
                        "threshold": self.config.data_quality_threshold,
                        "current_value": analysis.data_quality,
                    }
                )

            # 高信頼度予測アラート
            overall_confidence = analysis.confidence_scores.get("overall", 0)
            if overall_confidence > self.config.confidence_threshold:
                alerts.append(
                    {
                        "symbol": analysis.symbol,
                        "type": "HIGH_CONFIDENCE_PREDICTION",
                        "message": f"高信頼度予測: {analysis.recommendation} "
                                   f"(信頼度: {overall_confidence:.2f})",
                        "severity": "high",
                        "timestamp": datetime.now().isoformat(),
                        "action_required": False,
                        "confidence": overall_confidence,
                        "recommendation": analysis.recommendation,
                    }
                )

            # パフォーマンス異常アラート
            analysis_time = analysis.performance_metrics.get("analysis_time", 0)
            if analysis_time > 30:  # 30秒以上
                alerts.append(
                    {
                        "symbol": analysis.symbol,
                        "type": "PERFORMANCE_DEGRADATION",
                        "message": f"分析時間異常: {analysis_time:.1f}秒",
                        "severity": "low",
                        "timestamp": datetime.now().isoformat(),
                        "analysis_time": analysis_time,
                        "threshold": 30,
                    }
                )

            # メモリ使用量アラート
            memory_usage = analysis.performance_metrics.get("memory_usage", 0)
            if memory_usage > 100:  # 100MB以上
                alerts.append(
                    {
                        "symbol": analysis.symbol,
                        "type": "MEMORY_USAGE_WARNING",
                        "message": f"メモリ使用量高: {memory_usage:.1f}MB",
                        "severity": "medium",
                        "timestamp": datetime.now().isoformat(),
                        "memory_usage": memory_usage,
                        "threshold": 100,
                    }
                )

            # 極端な価格変動予測アラート
            if "predicted_change" in analysis.predictions:
                predicted_change = analysis.predictions["predicted_change"]
                if abs(predicted_change) > 0.1:  # 10%以上の変動予測
                    alerts.append(
                        {
                            "symbol": analysis.symbol,
                            "type": "EXTREME_PREDICTION_ALERT",
                            "message": f"極端な価格変動予測: {predicted_change:.1%}",
                            "severity": "high",
                            "timestamp": datetime.now().isoformat(),
                            "predicted_change": predicted_change,
                            "direction": "up" if predicted_change > 0 else "down",
                            "action_required": True,
                        }
                    )

            # 高リスクアラート
            risk_level = analysis.risk_assessment.get("risk_level", "unknown")
            if risk_level == "high":
                overall_risk = analysis.risk_assessment.get("overall_risk_score", 0)
                alerts.append(
                    {
                        "symbol": analysis.symbol,
                        "type": "HIGH_RISK_ALERT",
                        "message": f"高リスク検出: {risk_level} "
                                   f"(スコア: {overall_risk:.2f})",
                        "severity": "high",
                        "timestamp": datetime.now().isoformat(),
                        "risk_score": overall_risk,
                        "risk_factors": analysis.risk_assessment,
                        "action_required": True,
                    }
                )

            # テクニカル分析アラート
            self._generate_technical_alerts(analysis, alerts)

        except Exception as e:
            logger.error(f"スマートアラート生成エラー {analysis.symbol}: {e}")
            # エラーアラートを追加
            alerts.append(
                {
                    "symbol": analysis.symbol,
                    "type": "ALERT_GENERATION_ERROR",
                    "message": f"アラート生成エラー: {str(e)}",
                    "severity": "low",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
            )

        return alerts

    def _generate_technical_alerts(
        self, analysis: AIAnalysisResult, alerts: List[Dict[str, Any]]
    ) -> None:
        """
        テクニカル分析アラート生成
        
        Args:
            analysis: AI分析結果
            alerts: アラートリスト（参照渡し）
        """
        try:
            # RSI極端値アラート
            if "rsi" in analysis.technical_signals:
                rsi_data = analysis.technical_signals["rsi"]
                rsi_value = rsi_data.get("value", 50)
                
                if rsi_value < 20:  # 極度の売られ過ぎ
                    alerts.append(
                        {
                            "symbol": analysis.symbol,
                            "type": "RSI_OVERSOLD_EXTREME",
                            "message": f"RSI極度の売られ過ぎ: {rsi_value:.1f}",
                            "severity": "high",
                            "timestamp": datetime.now().isoformat(),
                            "rsi_value": rsi_value,
                            "signal": "oversold_extreme",
                        }
                    )
                elif rsi_value > 80:  # 極度の買われ過ぎ
                    alerts.append(
                        {
                            "symbol": analysis.symbol,
                            "type": "RSI_OVERBOUGHT_EXTREME",
                            "message": f"RSI極度の買われ過ぎ: {rsi_value:.1f}",
                            "severity": "high",
                            "timestamp": datetime.now().isoformat(),
                            "rsi_value": rsi_value,
                            "signal": "overbought_extreme",
                        }
                    )

            # 移動平均クロスアラート
            if "moving_average" in analysis.technical_signals:
                ma_data = analysis.technical_signals["moving_average"]
                
                if ma_data.get("golden_cross"):
                    alerts.append(
                        {
                            "symbol": analysis.symbol,
                            "type": "GOLDEN_CROSS_ALERT",
                            "message": "ゴールデンクロス発生",
                            "severity": "high",
                            "timestamp": datetime.now().isoformat(),
                            "signal": "bullish",
                            "action_required": True,
                        }
                    )
                elif ma_data.get("death_cross"):
                    alerts.append(
                        {
                            "symbol": analysis.symbol,
                            "type": "DEATH_CROSS_ALERT",
                            "message": "デッドクロス発生",
                            "severity": "high",
                            "timestamp": datetime.now().isoformat(),
                            "signal": "bearish",
                            "action_required": True,
                        }
                    )

            # ボラティリティ異常アラート
            if "volatility" in analysis.technical_signals:
                vol_data = analysis.technical_signals["volatility"]
                vol_regime = vol_data.get("regime", "normal")
                
                if vol_regime == "high":
                    alerts.append(
                        {
                            "symbol": analysis.symbol,
                            "type": "HIGH_VOLATILITY_ALERT",
                            "message": "高ボラティリティ環境検出",
                            "severity": "medium",
                            "timestamp": datetime.now().isoformat(),
                            "volatility_percentile": vol_data.get("percentile", 0),
                            "regime": vol_regime,
                        }
                    )
                elif vol_regime == "low":
                    alerts.append(
                        {
                            "symbol": analysis.symbol,
                            "type": "LOW_VOLATILITY_ALERT",
                            "message": "低ボラティリティ環境検出",
                            "severity": "low",
                            "timestamp": datetime.now().isoformat(),
                            "volatility_percentile": vol_data.get("percentile", 0),
                            "regime": vol_regime,
                        }
                    )

        except Exception as e:
            logger.error(f"テクニカルアラート生成エラー {analysis.symbol}: {e}")

    def generate_batch_signals(
        self, analyses: List[AIAnalysisResult]
    ) -> List[Dict[str, Any]]:
        """
        バッチシグナル生成
        
        Args:
            analyses: AI分析結果リスト
            
        Returns:
            List[Dict[str, Any]]: バッチ生成されたシグナルリスト
        """
        all_signals = []

        for analysis in analyses:
            try:
                signals = self.generate_ai_signals(analysis)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"バッチシグナル生成エラー {analysis.symbol}: {e}")

        return all_signals

    def generate_batch_alerts(
        self, analyses: List[AIAnalysisResult]
    ) -> List[Dict[str, Any]]:
        """
        バッチアラート生成
        
        Args:
            analyses: AI分析結果リスト
            
        Returns:
            List[Dict[str, Any]]: バッチ生成されたアラートリスト
        """
        all_alerts = []

        for analysis in analyses:
            try:
                alerts = self.generate_smart_alerts(analysis)
                all_alerts.extend(alerts)
            except Exception as e:
                logger.error(f"バッチアラート生成エラー {analysis.symbol}: {e}")

        return all_alerts

    def get_status(self) -> Dict[str, Any]:
        """
        シグナルジェネレーターのステータス取得
        
        Returns:
            Dict[str, Any]: ステータス情報
        """
        return {
            "confidence_threshold": self.config.confidence_threshold,
            "data_quality_threshold": self.config.data_quality_threshold,
            "prediction_horizon": self.config.prediction_horizon,
            "safe_mode": True,
            "trading_disabled": True,
        }