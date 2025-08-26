"""
レポート管理システム - シグナル・統計分析モジュール

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません
"""

from typing import Any, Dict, List, Optional

from ...automation.analysis_only_engine import AnalysisReport, MarketAnalysis
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SignalAnalyzers:
    """シグナル・統計分析機能を提供するクラス"""

    @staticmethod
    def calculate_signal_statistics(
        analyses: Dict[str, MarketAnalysis],
        latest_report: Optional[AnalysisReport],
    ) -> Dict[str, Any]:
        """シグナル統計計算"""
        signal_stats = {
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "strong_signals": 0,
            "medium_signals": 0,
            "weak_signals": 0,
            "average_confidence": 0,
            "signal_distribution": {},
            "insights": [],
        }

        try:
            confidences = []
            buy_count = 0
            sell_count = 0

            for _, analysis in analyses.items():
                if analysis.signal:
                    signal_stats["total_signals"] += 1
                    confidences.append(analysis.signal.confidence)

                    if analysis.signal.signal_type.value == "buy":
                        buy_count += 1
                    else:
                        sell_count += 1

                    # 信頼度による分類
                    if analysis.signal.confidence >= 80:
                        signal_stats["strong_signals"] += 1
                    elif analysis.signal.confidence >= 60:
                        signal_stats["medium_signals"] += 1
                    else:
                        signal_stats["weak_signals"] += 1

            signal_stats["buy_signals"] = buy_count
            signal_stats["sell_signals"] = sell_count

            if confidences:
                signal_stats["average_confidence"] = sum(confidences) / len(confidences)

            # 最新レポートからの情報
            if latest_report:
                signal_stats["latest_report"] = {
                    "strong_signals": latest_report.strong_signals,
                    "medium_signals": latest_report.medium_signals,
                    "weak_signals": latest_report.weak_signals,
                    "market_sentiment": latest_report.market_sentiment,
                }

            # インサイト生成
            total = signal_stats["total_signals"]
            if total > 0:
                signal_stats["insights"] = [
                    f"総シグナル数: {total}個",
                    f"買いシグナル: {buy_count}個 ({buy_count / total * 100:.1f}%)",
                    f"売りシグナル: {sell_count}個 ({sell_count / total * 100:.1f}%)",
                    f"平均信頼度: {signal_stats['average_confidence']:.1f}%",
                    f"強いシグナル: {signal_stats['strong_signals']}個",
                    "※ 高い信頼度のシグナルほど分析の確実性が高いことを示します",
                ]
            else:
                signal_stats["insights"] = ["現在、有効なシグナルは検出されていません"]

        except Exception as e:
            logger.error(f"シグナル統計計算エラー: {e}")
            signal_stats["error"] = str(e)

        return signal_stats

    @staticmethod
    def assess_individual_risk(analysis: MarketAnalysis) -> Dict[str, Any]:
        """個別銘柄リスク評価"""
        risk_assessment = {
            "risk_level": "中",
            "factors": [],
            "score": 50,  # 0-100スケール
        }

        try:
            score = 50  # ベース
            factors = []

            # ボラティリティベースの評価
            if analysis.volatility:
                if analysis.volatility > 0.4:
                    score += 20
                    factors.append("高ボラティリティ（高リスク）")
                elif analysis.volatility < 0.15:
                    score -= 15
                    factors.append("低ボラティリティ（低リスク）")
                else:
                    factors.append("中程度のボラティリティ")

            # トレンドベースの評価
            if analysis.price_trend == "下降":
                score += 15
                factors.append("下降トレンド（下落リスク）")
            elif analysis.price_trend == "上昇":
                score -= 10
                factors.append("上昇トレンド（上昇余地）")

            # 出来高トレンド
            if analysis.volume_trend == "減少":
                score += 5
                factors.append("出来高減少（流動性懸念）")

            # リスクレベル判定
            if score >= 70:
                risk_level = "高"
            elif score >= 55:
                risk_level = "やや高"
            elif score <= 35:
                risk_level = "低"
            elif score <= 45:
                risk_level = "やや低"
            else:
                risk_level = "中"

            risk_assessment.update(
                {
                    "risk_level": risk_level,
                    "factors": factors,
                    "score": min(100, max(0, score)),
                }
            )

        except Exception as e:
            logger.error(f"リスク評価エラー: {e}")
            risk_assessment["error"] = str(e)

        return risk_assessment

    @staticmethod
    def interpret_signal(signal) -> str:
        """シグナル解釈"""
        try:
            confidence = signal.confidence
            signal_type = signal.signal_type.value

            if confidence >= 80:
                strength_desc = "非常に強い"
            elif confidence >= 70:
                strength_desc = "強い"
            elif confidence >= 60:
                strength_desc = "中程度の"
            else:
                strength_desc = "弱い"

            action = "買い" if signal_type == "buy" else "売り"

            return f"{strength_desc}{action}シグナル（信頼度: {confidence:.1f}%）"

        except Exception as e:
            return f"シグナル解釈エラー: {e}"

    @staticmethod
    def create_detailed_individual_analyses(
        analyses: Dict[str, MarketAnalysis], symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """詳細個別銘柄分析"""
        detailed_analyses = {}

        for symbol in symbols:
            if symbol in analyses:
                analysis = analyses[symbol]

                # 基本情報
                detailed_analyses[symbol] = {
                    "basic_info": {
                        "symbol": symbol,
                        "current_price": float(analysis.current_price),
                        "analysis_time": analysis.analysis_timestamp.isoformat(),
                    },
                    "trend_analysis": {
                        "price_trend": analysis.price_trend,
                        "volume_trend": analysis.volume_trend,
                        "volatility": analysis.volatility,
                    },
                    "signal_analysis": {},
                    "recommendations": analysis.recommendations,
                    "risk_assessment": SignalAnalyzers.assess_individual_risk(analysis),
                }

                # シグナル情報
                if analysis.signal:
                    detailed_analyses[symbol]["signal_analysis"] = {
                        "signal_type": analysis.signal.signal_type.value,
                        "confidence": analysis.signal.confidence,
                        "strength": (
                            analysis.signal.strength.value
                            if hasattr(analysis.signal, "strength")
                            else "N/A"
                        ),
                        "interpretation": SignalAnalyzers.interpret_signal(analysis.signal),
                    }

        return detailed_analyses