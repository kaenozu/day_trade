"""
レポート管理システム - 洞察・推奨生成モジュール

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません
"""

from typing import Any, Dict, List

from ...automation.analysis_only_engine import MarketAnalysis
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class InsightGenerators:
    """洞察・推奨生成機能を提供するクラス"""

    @staticmethod
    def generate_educational_insights(
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

    @staticmethod
    def generate_smart_recommendations(
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

    @staticmethod
    def generate_individual_educational_notes(analysis: MarketAnalysis) -> List[str]:
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