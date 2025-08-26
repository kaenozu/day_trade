"""
レポート管理システム - 市場分析モジュール

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません
"""

from datetime import datetime
from typing import Any, Dict, List

from ...automation.analysis_only_engine import MarketAnalysis
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MarketAnalyzers:
    """市場分析機能を提供するクラス"""

    @staticmethod
    def analyze_market_trends(
        analyses: Dict[str, MarketAnalysis], symbols: List[str]
    ) -> Dict[str, Any]:
        """市場トレンド分析"""
        trend_data = {
            "overall_sentiment": "neutral",
            "trending_up": [],
            "trending_down": [],
            "sideways": [],
            "trend_strength": {},
            "sector_trends": {},
            "key_observations": [],
        }

        try:
            up_count = 0
            down_count = 0
            sideways_count = 0

            for symbol in symbols:
                if symbol in analyses:
                    analysis = analyses[symbol]

                    # 価格トレンド分類
                    if analysis.price_trend == "上昇":
                        trend_data["trending_up"].append(symbol)
                        up_count += 1
                    elif analysis.price_trend == "下降":
                        trend_data["trending_down"].append(symbol)
                        down_count += 1
                    else:
                        trend_data["sideways"].append(symbol)
                        sideways_count += 1

                    # トレンド強度（ボラティリティベース）
                    if analysis.volatility:
                        trend_data["trend_strength"][symbol] = {
                            "volatility": analysis.volatility,
                            "classification": (
                                "高"
                                if analysis.volatility > 0.3
                                else "中" if analysis.volatility > 0.2 else "低"
                            ),
                        }

            # 全体的センチメント判定
            total = len(symbols)
            if total > 0:
                up_ratio = up_count / total
                down_ratio = down_count / total

                if up_ratio > 0.6:
                    trend_data["overall_sentiment"] = "強気"
                elif down_ratio > 0.6:
                    trend_data["overall_sentiment"] = "弱気"
                elif up_ratio > 0.4:
                    trend_data["overall_sentiment"] = "やや強気"
                elif down_ratio > 0.4:
                    trend_data["overall_sentiment"] = "やや弱気"

            # キー観察事項
            trend_data["key_observations"] = [
                (
                    f"上昇トレンド銘柄: {up_count}銘柄 ({up_count / total * 100:.1f}%)"
                    if total > 0
                    else "データ不足"
                ),
                (
                    f"下降トレンド銘柄: {down_count}銘柄 ({down_count / total * 100:.1f}%)"
                    if total > 0
                    else "データ不足"
                ),
                (
                    f"横ばい銘柄: {sideways_count}銘柄 ({sideways_count / total * 100:.1f}%)"
                    if total > 0
                    else "データ不足"
                ),
                f"全体的市場センチメント: {trend_data['overall_sentiment']}",
            ]

        except Exception as e:
            logger.error(f"トレンド分析エラー: {e}")
            trend_data["error"] = str(e)

        return trend_data

    @staticmethod
    def analyze_correlations(
        analyses: Dict[str, MarketAnalysis], symbols: List[str]
    ) -> Dict[str, Any]:
        """相関分析"""
        correlation_data = {
            "price_correlations": {},
            "trend_correlations": {},
            "volatility_correlations": {},
            "insights": [],
        }

        try:
            # 価格データ収集
            prices = {}
            trends = {}
            volatilities = {}

            for symbol in symbols:
                if symbol in analyses:
                    analysis = analyses[symbol]
                    prices[symbol] = float(analysis.current_price)
                    trends[symbol] = (
                        1
                        if analysis.price_trend == "上昇"
                        else -1 if analysis.price_trend == "下降" else 0
                    )
                    volatilities[symbol] = analysis.volatility or 0

            # 簡易相関分析（より高度な分析は将来実装）
            if len(prices) >= 2:
                correlation_data["insights"] = [
                    f"分析対象銘柄数: {len(prices)}銘柄",
                    "※ より詳細な相関分析は履歴データが蓄積されると実行可能になります",
                    "※ 現在は同時点でのトレンド一致性を分析しています",
                ]

                # トレンド一致性分析
                trend_values = list(trends.values())
                positive_trends = sum(1 for t in trend_values if t > 0)
                negative_trends = sum(1 for t in trend_values if t < 0)

                if positive_trends > len(trend_values) * 0.7:
                    correlation_data["insights"].append(
                        "多くの銘柄が同方向（上昇）にトレンドしています"
                    )
                elif negative_trends > len(trend_values) * 0.7:
                    correlation_data["insights"].append(
                        "多くの銘柄が同方向（下降）にトレンドしています"
                    )
                else:
                    correlation_data["insights"].append(
                        "銘柄間でトレンド方向が分散しています"
                    )

        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            correlation_data["error"] = str(e)

        return correlation_data

    @staticmethod
    def analyze_volatility_patterns(
        analyses: Dict[str, MarketAnalysis], symbols: List[str]
    ) -> Dict[str, Any]:
        """ボラティリティパターン分析"""
        volatility_data = {
            "average_volatility": 0,
            "high_volatility_stocks": [],
            "low_volatility_stocks": [],
            "volatility_distribution": {},
            "insights": [],
        }

        try:
            volatilities = []
            vol_by_symbol = {}

            for symbol in symbols:
                if symbol in analyses and analyses[symbol].volatility is not None:
                    vol = analyses[symbol].volatility
                    volatilities.append(vol)
                    vol_by_symbol[symbol] = vol

            if volatilities:
                avg_vol = sum(volatilities) / len(volatilities)
                volatility_data["average_volatility"] = avg_vol

                # 高/低ボラティリティ銘柄分類
                for symbol, vol in vol_by_symbol.items():
                    if vol > avg_vol * 1.5:
                        volatility_data["high_volatility_stocks"].append(
                            {"symbol": symbol, "volatility": vol, "category": "高"}
                        )
                    elif vol < avg_vol * 0.5:
                        volatility_data["low_volatility_stocks"].append(
                            {"symbol": symbol, "volatility": vol, "category": "低"}
                        )

                # インサイト生成
                volatility_data["insights"] = [
                    f"平均ボラティリティ: {avg_vol:.3f}",
                    f"高ボラティリティ銘柄: {len(volatility_data['high_volatility_stocks'])}銘柄",
                    f"低ボラティリティ銘柄: {len(volatility_data['low_volatility_stocks'])}銘柄",
                    "※ 高ボラティリティ = 価格変動が大きい（リスク・リターン共に高い可能性）",
                    "※ 低ボラティリティ = 価格変動が小さい（安定性が高い可能性）",
                ]

        except Exception as e:
            logger.error(f"ボラティリティ分析エラー: {e}")
            volatility_data["error"] = str(e)

        return volatility_data

    @staticmethod
    def calculate_data_freshness(
        analyses: Dict[str, MarketAnalysis]
    ) -> Dict[str, Any]:
        """データ鮮度計算"""
        if not analyses:
            return {"status": "no_data"}

        now = datetime.now()
        timestamps = [analysis.analysis_timestamp for analysis in analyses.values()]

        if timestamps:
            latest = max(timestamps)
            oldest = min(timestamps)
            avg_age = sum([(now - ts).seconds for ts in timestamps]) / len(timestamps)

            return {
                "latest_analysis": latest.isoformat(),
                "oldest_analysis": oldest.isoformat(),
                "average_age_seconds": avg_age,
                "total_analyses": len(timestamps),
                "freshness_score": max(0, 100 - avg_age / 60),  # 分単位で減点
            }

        return {"status": "calculation_error"}