"""
強化された分析レポート管理システム

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..automation.analysis_only_engine import (
    AnalysisOnlyEngine,
    AnalysisReport,
    MarketAnalysis,
)
from ..config.trading_mode_config import is_safe_mode
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ReportFormat(Enum):
    """レポート形式"""

    JSON = "json"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "markdown"


class ReportType(Enum):
    """レポートタイプ"""

    DAILY_SUMMARY = "daily_summary"
    WEEKLY_ANALYSIS = "weekly_analysis"
    MARKET_OVERVIEW = "market_overview"
    SIGNAL_PERFORMANCE = "signal_performance"
    EDUCATIONAL_REPORT = "educational_report"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


@dataclass
class DetailedMarketReport:
    """詳細市場分析レポート"""

    report_id: str
    report_type: ReportType
    generated_at: datetime
    symbols_analyzed: List[str]

    # 市場概要
    market_summary: Dict[str, Any]

    # 個別銘柄分析
    individual_analyses: Dict[str, Dict[str, Any]]

    # トレンド分析
    trend_analysis: Dict[str, Any]

    # 相関分析
    correlation_analysis: Dict[str, Any]

    # ボラティリティ分析
    volatility_analysis: Dict[str, Any]

    # シグナル統計
    signal_statistics: Dict[str, Any]

    # 教育的洞察
    educational_insights: List[str]

    # 推奨事項
    recommendations: List[str]

    # メタデータ
    metadata: Dict[str, Any]


class EnhancedReportManager:
    """強化された分析レポート管理システム"""

    def __init__(self, analysis_engine: Optional[AnalysisOnlyEngine] = None):
        # セーフモードチェック
        if not is_safe_mode():
            raise ValueError("セーフモードでない場合は使用できません")

        self.analysis_engine = analysis_engine
        self.report_history: List[DetailedMarketReport] = []
        self.export_directory = Path("reports")
        self.export_directory.mkdir(exist_ok=True)

        logger.info("強化された分析レポート管理システム初期化完了")
        logger.info("※ 完全セーフモード - 分析・教育・研究専用")

    def generate_comprehensive_report(
        self,
        report_type: ReportType = ReportType.MARKET_OVERVIEW,
        symbols: Optional[List[str]] = None,
    ) -> DetailedMarketReport:
        """包括的分析レポート生成"""

        if not self.analysis_engine:
            raise ValueError("分析エンジンが設定されていません")

        # 分析対象銘柄
        target_symbols = symbols or self.analysis_engine.symbols

        logger.info(f"包括的分析レポート生成開始: {report_type.value}")

        # 基本データ収集
        all_analyses = self.analysis_engine.get_all_analyses()
        latest_report = self.analysis_engine.get_latest_report()
        market_summary = self.analysis_engine.get_market_summary()

        # レポートID生成
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 詳細分析実行
        trend_analysis = self._analyze_market_trends(all_analyses, target_symbols)
        correlation_analysis = self._analyze_correlations(all_analyses, target_symbols)
        volatility_analysis = self._analyze_volatility_patterns(all_analyses, target_symbols)
        signal_stats = self._calculate_signal_statistics(all_analyses, latest_report)
        educational_insights = self._generate_educational_insights(
            all_analyses, trend_analysis, signal_stats
        )
        recommendations = self._generate_smart_recommendations(
            all_analyses, trend_analysis, signal_stats
        )

        # 個別銘柄詳細分析
        detailed_individual = self._create_detailed_individual_analyses(
            all_analyses, target_symbols
        )

        # メタデータ
        metadata = {
            "analysis_engine_version": "1.0",
            "safe_mode": True,
            "trading_disabled": True,
            "generation_time_seconds": 0,  # 実装時に計測
            "data_freshness": self._calculate_data_freshness(all_analyses),
            "disclaimer": "このレポートは分析情報です。投資判断は自己責任で行ってください。",
        }

        # レポート作成
        report = DetailedMarketReport(
            report_id=report_id,
            report_type=report_type,
            generated_at=datetime.now(),
            symbols_analyzed=target_symbols,
            market_summary=market_summary,
            individual_analyses=detailed_individual,
            trend_analysis=trend_analysis,
            correlation_analysis=correlation_analysis,
            volatility_analysis=volatility_analysis,
            signal_statistics=signal_stats,
            educational_insights=educational_insights,
            recommendations=recommendations,
            metadata=metadata,
        )

        # 履歴に追加
        self.report_history.append(report)

        logger.info(f"包括的分析レポート生成完了: {report_id}")
        return report

    def _analyze_market_trends(
        self, analyses: Dict[str, MarketAnalysis], symbols: List[str]
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
                    f"上昇トレンド銘柄: {up_count}銘柄 ({up_count/total*100:.1f}%)"
                    if total > 0
                    else "データ不足"
                ),
                (
                    f"下降トレンド銘柄: {down_count}銘柄 ({down_count/total*100:.1f}%)"
                    if total > 0
                    else "データ不足"
                ),
                (
                    f"横ばい銘柄: {sideways_count}銘柄 ({sideways_count/total*100:.1f}%)"
                    if total > 0
                    else "データ不足"
                ),
                f"全体的市場センチメント: {trend_data['overall_sentiment']}",
            ]

        except Exception as e:
            logger.error(f"トレンド分析エラー: {e}")
            trend_data["error"] = str(e)

        return trend_data

    def _analyze_correlations(
        self, analyses: Dict[str, MarketAnalysis], symbols: List[str]
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
                    correlation_data["insights"].append("銘柄間でトレンド方向が分散しています")

        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            correlation_data["error"] = str(e)

        return correlation_data

    def _analyze_volatility_patterns(
        self, analyses: Dict[str, MarketAnalysis], symbols: List[str]
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

    def _calculate_signal_statistics(
        self,
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
                    f"買いシグナル: {buy_count}個 ({buy_count/total*100:.1f}%)",
                    f"売りシグナル: {sell_count}個 ({sell_count/total*100:.1f}%)",
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

    def _generate_educational_insights(
        self,
        analyses: Dict[str, MarketAnalysis],
        trend_analysis: Dict[str, Any],
        signal_stats: Dict[str, Any],
    ) -> List[str]:
        """教育的洞察の生成"""
        insights = [
            "🎓 教育的洞察とポイント",
            "",
            "📊 テクニカル分析について:",
            "- 価格トレンド分析: 株価の方向性を把握する基本的手法",
            "- ボラティリティ分析: 価格変動の大きさを測定し、リスクを評価",
            "- シグナル分析: 複数の指標を組み合わせて売買タイミングを判断",
            "",
            "🔍 市場分析の解釈:",
        ]

        try:
            # トレンド分析からの教育的ポイント
            if "overall_sentiment" in trend_analysis:
                sentiment = trend_analysis["overall_sentiment"]
                insights.extend(
                    [
                        f"- 現在の市場センチメント: {sentiment}",
                        "- センチメントは複数銘柄のトレンド方向から判断されます",
                    ]
                )

            # シグナル統計からの教育的ポイント
            if signal_stats.get("total_signals", 0) > 0:
                avg_confidence = signal_stats.get("average_confidence", 0)
                insights.extend(
                    [
                        f"- 平均シグナル信頼度: {avg_confidence:.1f}%",
                        "- 信頼度60%以上のシグナルが一般的に参考になります",
                        "- 複数の指標が同じ方向を示す時により信頼性が高まります",
                    ]
                )

            insights.extend(
                [
                    "",
                    "⚠️ 重要な注意事項:",
                    "- このシステムは教育・研究目的です",
                    "- 実際の投資判断は十分な検討と自己責任で行ってください",
                    "- 市場は常に変動し、予測には限界があります",
                    "- リスク管理と分散投資の重要性を理解してください",
                    "",
                    "📚 学習推奨事項:",
                    "- ファンダメンタル分析との組み合わせ",
                    "- 長期的な投資戦略の検討",
                    "- 市場心理学と行動経済学の理解",
                    "- リスク許容度の適切な評価",
                ]
            )

        except Exception as e:
            logger.error(f"教育的洞察生成エラー: {e}")
            insights.append(f"洞察生成エラー: {e}")

        return insights

    def _generate_smart_recommendations(
        self,
        analyses: Dict[str, MarketAnalysis],
        trend_analysis: Dict[str, Any],
        signal_stats: Dict[str, Any],
    ) -> List[str]:
        """スマート推奨事項生成"""
        recommendations = [
            "💡 分析ベース推奨事項",
            "",
            "🔍 注目すべき銘柄分析:",
        ]

        try:
            # 高信頼度シグナルの銘柄を推奨
            high_confidence_stocks = []
            for symbol, analysis in analyses.items():
                if analysis.signal and analysis.signal.confidence >= 75:
                    high_confidence_stocks.append(
                        {
                            "symbol": symbol,
                            "confidence": analysis.signal.confidence,
                            "signal_type": analysis.signal.signal_type.value,
                        }
                    )

            if high_confidence_stocks:
                high_confidence_stocks.sort(key=lambda x: x["confidence"], reverse=True)
                recommendations.extend(
                    [
                        "高信頼度シグナル検出銘柄:",
                    ]
                )

                for stock in high_confidence_stocks[:5]:  # 上位5銘柄
                    action = "買い注目" if stock["signal_type"] == "buy" else "売り注目"
                    recommendations.append(
                        f"- {stock['symbol']}: {action} (信頼度: {stock['confidence']:.1f}%)"
                    )

            # トレンド分析からの推奨
            if trend_analysis.get("overall_sentiment") in ["強気", "やや強気"]:
                recommendations.extend(
                    [
                        "",
                        "📈 市場環境分析:",
                        "- 市場全体が強気傾向を示しています",
                        "- 上昇トレンド銘柄への注目が推奨されます",
                        "- ただし、過熱感にも注意が必要です",
                    ]
                )
            elif trend_analysis.get("overall_sentiment") in ["弱気", "やや弱気"]:
                recommendations.extend(
                    [
                        "",
                        "📉 市場環境分析:",
                        "- 市場全体が弱気傾向を示しています",
                        "- 慎重な姿勢と下落リスクの評価が重要です",
                        "- 質の高い銘柄の押し目買い機会を探すことも検討",
                    ]
                )

            recommendations.extend(
                [
                    "",
                    "🎯 分析手法の改善提案:",
                    "- 複数の時間軸での分析を併用",
                    "- ファンダメンタル要因との照合",
                    "- リスク管理ルールの設定",
                    "- 定期的な分析結果の見直し",
                    "",
                    "⚠️ 免責事項:",
                    "- これらは分析結果に基づく情報提供です",
                    "- 投資の最終判断は必ず自己責任で行ってください",
                    "- 市場環境の変化により分析結果は変動します",
                    "- リスク管理を最優先に考慮してください",
                ]
            )

        except Exception as e:
            logger.error(f"推奨事項生成エラー: {e}")
            recommendations.append(f"推奨事項生成エラー: {e}")

        return recommendations

    def _create_detailed_individual_analyses(
        self, analyses: Dict[str, MarketAnalysis], symbols: List[str]
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
                    "risk_assessment": self._assess_individual_risk(analysis),
                    "educational_notes": self._generate_individual_educational_notes(analysis),
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
                        "interpretation": self._interpret_signal(analysis.signal),
                    }

        return detailed_analyses

    def _assess_individual_risk(self, analysis: MarketAnalysis) -> Dict[str, Any]:
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

    def _interpret_signal(self, signal) -> str:
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

    def _generate_individual_educational_notes(self, analysis: MarketAnalysis) -> List[str]:
        """個別銘柄の教育的ノート"""
        notes = [
            "📚 この銘柄の分析ポイント:",
        ]

        try:
            # 価格トレンドに基づく教育ノート
            if analysis.price_trend == "上昇":
                notes.extend(
                    [
                        "- 上昇トレンド: 価格が上向きに推移",
                        "- 継続性と過熱感のバランスに注目",
                        "- 押し目での買い場を探す戦略が考えられます",
                    ]
                )
            elif analysis.price_trend == "下降":
                notes.extend(
                    [
                        "- 下降トレンド: 価格が下向きに推移",
                        "- 下落の原因と底値の見極めが重要",
                        "- 反発ポイントの特定が鍵となります",
                    ]
                )
            else:
                notes.extend(
                    [
                        "- 横ばいトレンド: 価格がレンジ内で推移",
                        "- ブレイクアウトの方向性に注目",
                        "- レンジの上下限での売買戦略が有効な場合があります",
                    ]
                )

            # ボラティリティに基づく教育ノート
            if analysis.volatility:
                if analysis.volatility > 0.3:
                    notes.append("- 高いボラティリティ: 短期的な価格変動が大きい")
                elif analysis.volatility < 0.15:
                    notes.append("- 低いボラティリティ: 安定した価格推移")
                else:
                    notes.append("- 中程度のボラティリティ: バランスの取れた変動")

            notes.extend(
                [
                    "",
                    "🎯 学習のポイント:",
                    "- トレンドと出来高の関係性",
                    "- ボラティリティと投資期間の適合性",
                    "- 複数の分析手法の組み合わせの重要性",
                ]
            )

        except Exception as e:
            logger.error(f"教育ノート生成エラー: {e}")
            notes.append(f"ノート生成エラー: {e}")

        return notes

    def _calculate_data_freshness(self, analyses: Dict[str, MarketAnalysis]) -> Dict[str, Any]:
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

    def export_report(
        self,
        report: DetailedMarketReport,
        format: ReportFormat = ReportFormat.JSON,
        filename: Optional[str] = None,
    ) -> Path:
        """レポートエクスポート"""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_analysis_report_{timestamp}.{format.value}"

        filepath = self.export_directory / filename

        try:
            if format == ReportFormat.JSON:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)

            elif format == ReportFormat.MARKDOWN:
                self._export_as_markdown(report, filepath)

            elif format == ReportFormat.HTML:
                self._export_as_html(report, filepath)

            elif format == ReportFormat.CSV:
                self._export_as_csv(report, filepath)

            logger.info(f"レポートエクスポート完了: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"レポートエクスポートエラー: {e}")
            raise

    def _export_as_markdown(self, report: DetailedMarketReport, filepath: Path):
        """Markdown形式でエクスポート"""
        md_content = [
            "# 市場分析レポート",
            f"**レポートID**: {report.report_id}",
            f"**生成日時**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**レポートタイプ**: {report.report_type.value}",
            f"**分析銘柄数**: {len(report.symbols_analyzed)}",
            "",
            "## 🔒 セーフモード通知",
            "**このシステムは完全にセーフモードで動作しています**",
            "- 自動取引: 完全無効",
            "- 注文実行: 完全無効",
            "- 用途: 分析・教育・研究のみ",
            "",
            "## 📊 市場概要",
        ]

        # 市場サマリー追加
        if report.market_summary:
            for key, value in report.market_summary.items():
                if key != "注意":
                    md_content.append(f"- **{key}**: {value}")

        # トレンド分析追加
        md_content.extend(
            [
                "",
                "## 📈 トレンド分析",
                f"**全体的センチメント**: {report.trend_analysis.get('overall_sentiment', 'N/A')}",
            ]
        )

        if "key_observations" in report.trend_analysis:
            md_content.append("\n### 主要観察事項")
            for obs in report.trend_analysis["key_observations"]:
                md_content.append(f"- {obs}")

        # 教育的洞察追加
        md_content.extend(["", "## 🎓 教育的洞察"])
        for insight in report.educational_insights:
            md_content.append(insight)

        # 推奨事項追加
        md_content.extend(["", "## 💡 推奨事項"])
        for rec in report.recommendations:
            md_content.append(rec)

        # ファイル書き込み
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))

    def _export_as_html(self, report: DetailedMarketReport, filepath: Path):
        """HTML形式でエクスポート"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>市場分析レポート - {report.report_id}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .safe-mode {{
                    background: #28a745;
                    color: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .section {{
                    background: #f8f9fa;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .insight-item {{
                    background: #e3f2fd;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                }}
                .recommendation {{
                    background: #f3e5f5;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 市場分析レポート</h1>
                <p>レポートID: {report.report_id}</p>
                <p>生成日時: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="safe-mode">
                <h3>🔒 セーフモード動作中</h3>
                <p>このシステムは分析・教育・研究専用です。自動取引は完全に無効化されています。</p>
            </div>

            <div class="section">
                <h2>📈 トレンド分析</h2>
                <p><strong>全体的センチメント:</strong> {report.trend_analysis.get('overall_sentiment', 'N/A')}</p>
            </div>

            <div class="section">
                <h2>🎓 教育的洞察</h2>
        """

        for insight in report.educational_insights:
            html_content += f'<div class="insight-item">{insight}</div>'

        html_content += """
            </div>

            <div class="section">
                <h2>💡 推奨事項</h2>
        """

        for rec in report.recommendations:
            html_content += f'<div class="recommendation">{rec}</div>'

        html_content += f"""
            </div>

            <div class="section">
                <h2>ℹ️ メタデータ</h2>
                <p><strong>分析銘柄数:</strong> {len(report.symbols_analyzed)}</p>
                <p><strong>セーフモード:</strong> 有効</p>
                <p><strong>免責事項:</strong> {report.metadata.get('disclaimer', 'N/A')}</p>
            </div>
        </body>
        </html>
        """

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _export_as_csv(self, report: DetailedMarketReport, filepath: Path):
        """CSV形式でエクスポート（簡易版）"""
        try:
            # 個別銘柄データをCSV形式で出力
            csv_data = []

            for symbol, analysis in report.individual_analyses.items():
                row = {
                    "レポートID": report.report_id,
                    "生成日時": report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "銘柄": symbol,
                    "現在価格": analysis["basic_info"]["current_price"],
                    "価格トレンド": analysis["trend_analysis"]["price_trend"],
                    "出来高トレンド": analysis["trend_analysis"]["volume_trend"],
                    "ボラティリティ": analysis["trend_analysis"]["volatility"],
                    "リスクレベル": analysis["risk_assessment"]["risk_level"],
                    "リスクスコア": analysis["risk_assessment"]["score"],
                }

                if "signal_analysis" in analysis and analysis["signal_analysis"]:
                    row.update(
                        {
                            "シグナルタイプ": analysis["signal_analysis"].get("signal_type", "N/A"),
                            "シグナル信頼度": analysis["signal_analysis"].get("confidence", "N/A"),
                        }
                    )

                csv_data.append(row)

            # pandas使用してCSV出力
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False, encoding="utf-8-sig")

        except Exception as e:
            logger.error(f"CSV出力エラー: {e}")
            # フォールバック: 基本的なCSV出力
            with open(filepath, "w", encoding="utf-8-sig") as f:
                f.write("レポートID,生成日時,銘柄数,セーフモード\n")
                f.write(
                    f"{report.report_id},{report.generated_at},{len(report.symbols_analyzed)},有効\n"
                )

    def get_report_history(self, limit: int = 10) -> List[DetailedMarketReport]:
        """レポート履歴取得"""
        return self.report_history[-limit:]

    def clear_history(self):
        """履歴クリア"""
        self.report_history.clear()
        logger.info("レポート履歴をクリアしました")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """要約統計取得"""
        return {
            "total_reports_generated": len(self.report_history),
            "safe_mode": is_safe_mode(),
            "export_directory": str(self.export_directory),
            "last_report_time": (
                self.report_history[-1].generated_at.isoformat() if self.report_history else None
            ),
        }
