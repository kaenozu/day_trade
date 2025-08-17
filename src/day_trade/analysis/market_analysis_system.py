"""
市場分析・情報提供システム

自動取引機能の無効化により新たに構築された分析専用システム。
包括的な市場分析、情報提供、手動取引支援機能を提供。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config.trading_mode_config import get_current_trading_config, is_safe_mode
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MarketAnalysisSystem:
    """
    市場分析・情報提供システム

    提供機能:
    1. リアルタイム市場データ分析
    2. 取引シグナル分析・情報提供
    3. リスク分析レポート
    4. 手動取引支援情報
    5. ポートフォリオ分析
    6. 市場トレンド分析

    ※ 注文実行機能は含まれていません
    """

    def __init__(self, symbols: List[str]) -> None:
        # 安全確認
        if not is_safe_mode():
            raise RuntimeError("安全性エラー: 自動取引が無効化されていません")

        self.symbols = symbols
        self.trading_config = get_current_trading_config()

        # 分析データストレージ
        self.market_analysis_history: List[Dict[str, Any]] = []
        self.signal_analysis_history: List[Dict[str, Any]] = []
        self.risk_analysis_history: List[Dict[str, Any]] = []

        # 分析統計
        self.analysis_stats = {
            "market_analyses_performed": 0,
            "signals_analyzed": 0,
            "risk_reports_generated": 0,
            "recommendations_provided": 0,
            "last_analysis_time": None,
        }

        logger.info(
            f"市場分析システム初期化完了 - "
            f"監視銘柄数: {len(symbols)} "
            f"モード: {self.trading_config.current_mode.value}"
        )

    async def perform_comprehensive_market_analysis(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """包括的市場分析"""
        try:
            analysis_time = datetime.now()

            analysis_result = {
                "timestamp": analysis_time,
                "market_overview": await self._analyze_market_overview(market_data),
                "symbol_analysis": await self._analyze_individual_symbols(market_data),
                "trend_analysis": await self._analyze_market_trends(market_data),
                "volatility_analysis": await self._analyze_volatility(market_data),
                "correlation_analysis": await self._analyze_correlations(market_data),
                "recommendation_summary": await self._generate_recommendations(
                    market_data
                ),
            }

            # 履歴に保存
            self.market_analysis_history.append(analysis_result)

            # 古い履歴をクリーンアップ（最新100件のみ保持）
            if len(self.market_analysis_history) > 100:
                self.market_analysis_history = self.market_analysis_history[-100:]

            # 統計更新
            self.analysis_stats["market_analyses_performed"] += 1
            self.analysis_stats["last_analysis_time"] = analysis_time

            logger.info(f"包括的市場分析完了 - {len(self.symbols)}銘柄")

            return analysis_result

        except Exception as e:
            logger.error(f"市場分析エラー: {e}")
            return {
                "timestamp": datetime.now(),
                "error": str(e),
                "status": "分析失敗",
            }

    async def _analyze_market_overview(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """市場全体の概要分析"""
        try:
            overview = {
                "total_symbols": len(self.symbols),
                "market_status": "分析中",
                "overall_sentiment": "中立",
                "major_movements": [],
                "market_volatility": "低",
            }

            if market_data:
                # 価格変動の分析
                price_changes = []
                for _symbol, data in market_data.items():
                    if isinstance(data, dict) and "price_change_pct" in data:
                        price_changes.append(data["price_change_pct"])

                if price_changes:
                    avg_change = sum(price_changes) / len(price_changes)
                    overview["average_price_change"] = f"{avg_change:.2f}%"

                    if avg_change > 1.0:
                        overview["overall_sentiment"] = "強気"
                    elif avg_change < -1.0:
                        overview["overall_sentiment"] = "弱気"

                    # ボラティリティ判定
                    volatility = sum(abs(change) for change in price_changes) / len(
                        price_changes
                    )
                    if volatility > 2.0:
                        overview["market_volatility"] = "高"
                    elif volatility > 1.0:
                        overview["market_volatility"] = "中"

            return overview

        except Exception as e:
            logger.error(f"市場概要分析エラー: {e}")
            return {"error": str(e)}

    async def _analyze_individual_symbols(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """個別銘柄分析"""
        try:
            symbol_analyses = {}

            for symbol in self.symbols:
                if symbol in market_data:
                    data = market_data[symbol]

                    analysis = {
                        "current_price": data.get("current_price", "N/A"),
                        "price_change": data.get("price_change", "N/A"),
                        "price_change_pct": data.get("price_change_pct", "N/A"),
                        "volume": data.get("volume", "N/A"),
                        "analysis_summary": self._generate_symbol_summary(symbol, data),
                        "support_resistance": self._identify_support_resistance(
                            symbol, data
                        ),
                        "trend_direction": self._determine_trend(symbol, data),
                    }

                    symbol_analyses[symbol] = analysis
                else:
                    symbol_analyses[symbol] = {
                        "status": "データなし",
                        "note": "市場データが利用できません",
                    }

            return symbol_analyses

        except Exception as e:
            logger.error(f"個別銘柄分析エラー: {e}")
            return {}

    def _generate_symbol_summary(self, symbol: str, data: Dict[str, Any]) -> str:
        """銘柄サマリー生成"""
        try:
            price_change_pct = data.get("price_change_pct", 0)

            if price_change_pct > 2.0:
                return f"{symbol}: 強い上昇トレンド (+{price_change_pct:.1f}%)"
            elif price_change_pct > 0.5:
                return f"{symbol}: 上昇傾向 (+{price_change_pct:.1f}%)"
            elif price_change_pct < -2.0:
                return f"{symbol}: 強い下落トレンド ({price_change_pct:.1f}%)"
            elif price_change_pct < -0.5:
                return f"{symbol}: 下落傾向 ({price_change_pct:.1f}%)"
            else:
                return f"{symbol}: 横ばい ({price_change_pct:.1f}%)"

        except Exception:
            return f"{symbol}: 分析エラー"

    def _identify_support_resistance(
        self, symbol: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """サポート・レジスタンス分析"""
        try:
            current_price = data.get("current_price", 0)

            # 簡易的なサポート・レジスタンス計算
            # 実際の実装では履歴データを使用
            support = current_price * 0.95  # 5%下
            resistance = current_price * 1.05  # 5%上

            return {
                "support_level": support,
                "resistance_level": resistance,
                "confidence": "低（履歴データ不足）",
            }

        except Exception as e:
            return {"error": str(e)}

    def _determine_trend(self, symbol: str, data: Dict[str, Any]) -> str:
        """トレンド方向判定"""
        try:
            price_change_pct = data.get("price_change_pct", 0)

            if price_change_pct > 1.0:
                return "上昇トレンド"
            elif price_change_pct < -1.0:
                return "下降トレンド"
            else:
                return "横ばい"

        except Exception:
            return "判定不可"

    async def _analyze_market_trends(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """市場トレンド分析"""
        try:
            trend_analysis = {
                "short_term_trend": "分析中",
                "medium_term_trend": "分析中",
                "long_term_trend": "分析中",
                "trend_strength": "中",
                "reversal_signals": [],
            }

            # 簡易トレンド分析
            if market_data:
                price_changes = [
                    data.get("price_change_pct", 0)
                    for data in market_data.values()
                    if isinstance(data, dict)
                ]

                if price_changes:
                    avg_change = sum(price_changes) / len(price_changes)

                    if avg_change > 1.0:
                        trend_analysis["short_term_trend"] = "上昇"
                    elif avg_change < -1.0:
                        trend_analysis["short_term_trend"] = "下降"
                    else:
                        trend_analysis["short_term_trend"] = "横ばい"

            return trend_analysis

        except Exception as e:
            logger.error(f"トレンド分析エラー: {e}")
            return {"error": str(e)}

    async def _analyze_volatility(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """ボラティリティ分析"""
        try:
            volatility_analysis = {
                "overall_volatility": "中",
                "high_volatility_symbols": [],
                "low_volatility_symbols": [],
                "volatility_trend": "安定",
            }

            if market_data:
                volatilities = []

                for symbol, data in market_data.items():
                    if isinstance(data, dict) and "price_change_pct" in data:
                        vol = abs(data["price_change_pct"])
                        volatilities.append((symbol, vol))

                        if vol > 3.0:
                            volatility_analysis["high_volatility_symbols"].append(
                                f"{symbol} ({vol:.1f}%)"
                            )
                        elif vol < 0.5:
                            volatility_analysis["low_volatility_symbols"].append(
                                f"{symbol} ({vol:.1f}%)"
                            )

                if volatilities:
                    avg_vol = sum(vol for _, vol in volatilities) / len(volatilities)

                    if avg_vol > 2.0:
                        volatility_analysis["overall_volatility"] = "高"
                    elif avg_vol < 1.0:
                        volatility_analysis["overall_volatility"] = "低"

            return volatility_analysis

        except Exception as e:
            logger.error(f"ボラティリティ分析エラー: {e}")
            return {"error": str(e)}

    async def _analyze_correlations(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """相関分析"""
        try:
            correlation_analysis = {
                "strong_correlations": [],
                "negative_correlations": [],
                "independent_movements": [],
                "note": "相関分析には長期データが必要です",
            }

            # 実際の実装では履歴データを使用した相関計算を行う
            # ここでは簡易的な分析のみ

            return correlation_analysis

        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            return {"error": str(e)}

    async def _generate_recommendations(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """推奨事項生成"""
        try:
            recommendations = {
                "buy_candidates": [],
                "sell_candidates": [],
                "hold_recommendations": [],
                "risk_warnings": [],
                "general_advice": [],
            }

            for symbol, data in market_data.items():
                if isinstance(data, dict):
                    price_change_pct = data.get("price_change_pct", 0)

                    if price_change_pct > 3.0:
                        recommendations["risk_warnings"].append(
                            f"{symbol}: 急激な上昇 - 利確検討"
                        )
                    elif price_change_pct < -3.0:
                        recommendations["risk_warnings"].append(
                            f"{symbol}: 急激な下落 - 損切り検討"
                        )
                    elif 1.0 < price_change_pct <= 2.0:
                        recommendations["buy_candidates"].append(
                            f"{symbol}: 安定上昇 - 買い検討"
                        )
                    elif -2.0 <= price_change_pct < -1.0:
                        recommendations["sell_candidates"].append(
                            f"{symbol}: 下落傾向 - 売り検討"
                        )
                    else:
                        recommendations["hold_recommendations"].append(
                            f"{symbol}: 様子見"
                        )

            # 一般的なアドバイス
            recommendations["general_advice"] = [
                "※これらは分析情報であり、投資判断は自己責任で行ってください",
                "※実際の取引前には詳細な分析と確認を行ってください",
                "※市場状況は常に変化するため、定期的な見直しが重要です",
            ]

            self.analysis_stats["recommendations_provided"] += len(
                recommendations["buy_candidates"]
                + recommendations["sell_candidates"]
                + recommendations["risk_warnings"]
            )

            return recommendations

        except Exception as e:
            logger.error(f"推奨事項生成エラー: {e}")
            return {"error": str(e)}

    def get_analysis_summary(self) -> Dict[str, Any]:
        """分析サマリー取得"""
        try:
            return {
                "system_info": {
                    "system_name": "市場分析・情報提供システム",
                    "mode": self.trading_config.current_mode.value,
                    "safe_mode": is_safe_mode(),
                    "monitored_symbols": len(self.symbols),
                },
                "statistics": self.analysis_stats.copy(),
                "last_analysis": (
                    self.market_analysis_history[-1]
                    if self.market_analysis_history
                    else None
                ),
                "enabled_features": self.trading_config.get_enabled_features(),
                "disabled_features": self.trading_config.get_disabled_features(),
            }

        except Exception as e:
            logger.error(f"分析サマリー取得エラー: {e}")
            return {"error": str(e)}

    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最近の分析履歴取得"""
        try:
            return (
                self.market_analysis_history[-limit:]
                if self.market_analysis_history
                else []
            )

        except Exception as e:
            logger.error(f"分析履歴取得エラー: {e}")
            return []


class ManualTradingSupport:
    """
    手動取引支援システム

    自動取引に代わる手動取引支援機能を提供
    """

    def __init__(self) -> None:
        if not is_safe_mode():
            raise RuntimeError("安全性エラー: 自動取引が無効化されていません")

        self.support_history: List[Dict[str, Any]] = []

        logger.info("手動取引支援システム初期化完了")

    def generate_trading_suggestion(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        portfolio_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """取引提案生成"""
        try:
            suggestion_time = datetime.now()

            suggestion = {
                "timestamp": suggestion_time,
                "symbol": symbol,
                "market_analysis": self._analyze_current_situation(symbol, market_data),
                "trading_suggestions": self._generate_specific_suggestions(
                    symbol, market_data
                ),
                "risk_assessment": self._assess_risk(symbol, market_data),
                "position_sizing": self._suggest_position_size(
                    symbol, market_data, portfolio_info
                ),
                "timing_advice": self._provide_timing_advice(symbol, market_data),
                "important_notes": [
                    "これは情報提供のみです",
                    "実際の取引判断は自己責任で行ってください",
                    "市場状況の変化にご注意ください",
                ],
            }

            # 履歴に保存
            self.support_history.append(suggestion)

            logger.info(f"取引提案生成完了: {symbol}")

            return suggestion

        except Exception as e:
            logger.error(f"取引提案生成エラー: {e}")
            return {"error": str(e)}

    def _analyze_current_situation(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """現在の状況分析"""
        try:
            if symbol not in market_data:
                return {"error": "市場データなし"}

            data = market_data[symbol]

            return {
                "current_price": data.get("current_price", "N/A"),
                "price_change": data.get("price_change", "N/A"),
                "price_change_pct": data.get("price_change_pct", "N/A"),
                "volume": data.get("volume", "N/A"),
                "market_condition": self._determine_market_condition(data),
            }

        except Exception as e:
            return {"error": str(e)}

    def _determine_market_condition(self, data: Dict[str, Any]) -> str:
        """市場状況判定"""
        try:
            price_change_pct = data.get("price_change_pct", 0)

            if abs(price_change_pct) > 3.0:
                return "高ボラティリティ"
            elif abs(price_change_pct) > 1.0:
                return "活発"
            else:
                return "安定"

        except Exception:
            return "不明"

    def _generate_specific_suggestions(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> List[str]:
        """具体的な取引提案"""
        try:
            suggestions = []

            if symbol in market_data:
                data = market_data[symbol]
                price_change_pct = data.get("price_change_pct", 0)

                if price_change_pct > 2.0:
                    suggestions.extend(
                        [
                            f"上昇トレンド継続中 (+{price_change_pct:.1f}%)",
                            "押し目待ちでの買い参入を検討",
                            "利確ポイントの設定を推奨",
                        ]
                    )
                elif price_change_pct < -2.0:
                    suggestions.extend(
                        [
                            f"下落トレンド継続中 ({price_change_pct:.1f}%)",
                            "反発ポイントでの買い参入を検討",
                            "損切りラインの設定が重要",
                        ]
                    )
                else:
                    suggestions.extend(
                        [
                            "横ばい相場",
                            "レンジブレイクを待つ",
                            "様子見も有効な戦略",
                        ]
                    )

            return suggestions

        except Exception as e:
            return [f"分析エラー: {e}"]

    def _assess_risk(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """リスク評価"""
        try:
            if symbol not in market_data:
                return {"risk_level": "不明", "reason": "データなし"}

            data = market_data[symbol]
            price_change_pct = abs(data.get("price_change_pct", 0))

            if price_change_pct > 5.0:
                return {
                    "risk_level": "非常に高い",
                    "reason": f"価格変動が大きい ({price_change_pct:.1f}%)",
                    "recommendation": "ポジションサイズを小さくする",
                }
            elif price_change_pct > 2.0:
                return {
                    "risk_level": "高い",
                    "reason": f"価格変動がやや大きい ({price_change_pct:.1f}%)",
                    "recommendation": "慎重な取引を推奨",
                }
            elif price_change_pct > 1.0:
                return {
                    "risk_level": "中",
                    "reason": f"通常の価格変動範囲 ({price_change_pct:.1f}%)",
                    "recommendation": "標準的なリスク管理",
                }
            else:
                return {
                    "risk_level": "低い",
                    "reason": f"価格変動が小さい ({price_change_pct:.1f}%)",
                    "recommendation": "相対的に安全",
                }

        except Exception as e:
            return {"risk_level": "不明", "reason": str(e)}

    def _suggest_position_size(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        portfolio_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """ポジションサイズ提案"""
        try:
            suggestion = {
                "recommended_size": "標準",
                "percentage_of_portfolio": "2-5%",
                "reasoning": "一般的な推奨値",
            }

            if symbol in market_data:
                data = market_data[symbol]
                price_change_pct = abs(data.get("price_change_pct", 0))

                if price_change_pct > 3.0:
                    suggestion = {
                        "recommended_size": "小さめ",
                        "percentage_of_portfolio": "1-2%",
                        "reasoning": "高ボラティリティのためリスクを抑制",
                    }
                elif price_change_pct < 1.0:
                    suggestion = {
                        "recommended_size": "標準〜やや大きめ",
                        "percentage_of_portfolio": "3-7%",
                        "reasoning": "安定的な動きのため",
                    }

            return suggestion

        except Exception as e:
            return {"error": str(e)}

    def _provide_timing_advice(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """タイミングアドバイス"""
        try:
            advice = {
                "entry_timing": "様子見",
                "exit_timing": "保有なし",
                "general_advice": "市場の動向を注視",
            }

            if symbol in market_data:
                data = market_data[symbol]
                price_change_pct = data.get("price_change_pct", 0)

                if price_change_pct > 1.0:
                    advice["entry_timing"] = "押し目を待つ"
                    advice["general_advice"] = "上昇トレンド中 - 慎重に参入"
                elif price_change_pct < -1.0:
                    advice["entry_timing"] = "反発を待つ"
                    advice["general_advice"] = "下落トレンド中 - 底値確認後"
                else:
                    advice["entry_timing"] = "ブレイクアウトを待つ"
                    advice["general_advice"] = "レンジ相場 - 方向性待ち"

            return advice

        except Exception as e:
            return {"error": str(e)}

    def get_support_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """支援履歴取得"""
        return self.support_history[-limit:] if self.support_history else []
