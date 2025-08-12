"""
統合分析システム

自動取引機能の無効化により構築された新しいメインシステム。
市場分析・情報提供・手動取引支援を統合的に提供。
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List

from ..analysis.market_analysis_system import ManualTradingSupport, MarketAnalysisSystem
from ..automation.risk_aware_trading_engine import MarketAnalysisEngine
from ..config.trading_mode_config import (
    get_current_trading_config,
    is_safe_mode,
    log_current_configuration,
)
from ..data.stock_fetcher import StockFetcher
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class IntegratedAnalysisSystem:
    """
    統合分析システム

    【重要】自動取引機能は完全に無効化されています

    統合提供機能:
    1. 市場データの収集・分析
    2. リアルタイム監視
    3. 取引シグナル分析
    4. リスク分析・警告
    5. 手動取引支援
    6. 包括的レポート生成

    ※ 注文実行機能は含まれていません
    """

    def __init__(self, symbols: List[str]):
        # 安全確認
        if not is_safe_mode():
            raise RuntimeError(
                "安全性エラー: 自動取引が無効化されていません。"
                "システムを安全に使用するために、まず trading_mode_config で "
                "自動取引を無効化してください。"
            )

        self.symbols = symbols
        self.trading_config = get_current_trading_config()

        # サブシステム初期化
        try:
            self.market_analysis = MarketAnalysisSystem(symbols)
            self.manual_trading_support = ManualTradingSupport()
            self.analysis_engine = MarketAnalysisEngine(
                symbols=symbols,
                emergency_stop_enabled=False,  # 分析専用のため無効
            )
            self.stock_fetcher = StockFetcher()
        except Exception as e:
            logger.error(f"サブシステム初期化エラー: {e}")
            raise

        # 統合システム状態
        self.is_running = False
        self.last_analysis_time = None

        # 統計情報
        self.system_stats = {
            "total_analysis_cycles": 0,
            "market_data_updates": 0,
            "recommendations_generated": 0,
            "alerts_issued": 0,
            "system_start_time": datetime.now(),
        }

        logger.info(
            f"統合分析システム初期化完了 - "
            f"監視銘柄: {len(symbols)}銘柄 "
            f"モード: {self.trading_config.current_mode.value}"
        )

        # 設定状況をログ出力
        log_current_configuration()

    async def start_comprehensive_analysis(
        self, analysis_interval: float = 60.0
    ) -> None:
        """包括的分析システム開始"""
        if self.is_running:
            logger.warning("統合分析システムは既に実行中です")
            return

        # 最終安全確認
        if not is_safe_mode():
            logger.error("安全性確認失敗: 分析システムを開始できません")
            raise RuntimeError("セーフモードが無効化されています")

        logger.info("統合分析システムを開始します...")

        self.is_running = True

        try:
            # 分析エンジン開始
            await self.analysis_engine.start()

            # メイン分析ループ
            await self._main_analysis_loop(analysis_interval)

        except Exception as e:
            logger.error(f"分析システム実行エラー: {e}")
            raise
        finally:
            self.is_running = False

    async def stop_analysis_system(self) -> None:
        """分析システム停止"""
        logger.info("統合分析システム停止要求受信")

        self.is_running = False

        try:
            # 分析エンジン停止
            await self.analysis_engine.stop()

            logger.info("統合分析システム停止完了")

        except Exception as e:
            logger.error(f"分析システム停止エラー: {e}")

    async def _main_analysis_loop(self, interval: float) -> None:
        """メイン分析ループ"""
        cycle_count = 0

        logger.info(f"分析ループ開始 (間隔: {interval}秒)")

        while self.is_running:
            try:
                cycle_start = datetime.now()

                # 1. 市場データ取得
                market_data = await self._fetch_market_data()
                if market_data:
                    self.system_stats["market_data_updates"] += 1

                # 2. 包括的市場分析
                market_analysis = (
                    await self.market_analysis.perform_comprehensive_market_analysis(
                        market_data
                    )
                )

                # 3. 手動取引支援情報生成
                trading_suggestions = await self._generate_trading_suggestions(
                    market_data
                )

                # 4. リスク分析・警告
                risk_alerts = await self._perform_risk_analysis(
                    market_data, market_analysis
                )

                # 5. 統合レポート生成
                integrated_report = await self._generate_integrated_report(
                    market_analysis, trading_suggestions, risk_alerts
                )

                # 6. 結果のログ出力
                await self._log_analysis_results(integrated_report)

                # 7. 統計更新
                cycle_count += 1
                self.system_stats["total_analysis_cycles"] = cycle_count
                self.last_analysis_time = datetime.now()

                cycle_duration = (datetime.now() - cycle_start).total_seconds()

                if cycle_count % 10 == 0:  # 10サイクルごとに詳細ログ
                    logger.info(
                        f"分析サイクル {cycle_count} 完了 "
                        f"(実行時間: {cycle_duration:.2f}秒)"
                    )

                # インターバル待機
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"分析ループエラー: {e}")

                # エラー時は短時間待機してリトライ
                await asyncio.sleep(10.0)

        logger.info(f"分析ループ終了 (総サイクル数: {cycle_count})")

    async def _fetch_market_data(self) -> Dict[str, Any]:
        """市場データ取得"""
        try:
            market_data = {}

            for symbol in self.symbols:
                try:
                    data = await self.stock_fetcher.get_current_price_async(symbol)
                    if data:
                        market_data[symbol] = data
                except Exception as e:
                    logger.debug(f"市場データ取得エラー {symbol}: {e}")
                    market_data[symbol] = {"error": str(e)}

            logger.debug(f"市場データ取得完了: {len(market_data)}銘柄")

            return market_data

        except Exception as e:
            logger.error(f"市場データ取得エラー: {e}")
            return {}

    async def _generate_trading_suggestions(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """手動取引支援情報生成"""
        try:
            suggestions = {}

            for symbol in self.symbols:
                if symbol in market_data and market_data[symbol]:
                    suggestion = (
                        self.manual_trading_support.generate_trading_suggestion(
                            symbol, market_data
                        )
                    )
                    suggestions[symbol] = suggestion

            if suggestions:
                self.system_stats["recommendations_generated"] += len(suggestions)

            return suggestions

        except Exception as e:
            logger.error(f"取引提案生成エラー: {e}")
            return {}

    async def _perform_risk_analysis(
        self, market_data: Dict[str, Any], market_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """リスク分析・警告"""
        try:
            risk_alerts = []

            # 市場データベースのリスクチェック
            for symbol, data in market_data.items():
                if isinstance(data, dict) and "price_change_pct" in data:
                    price_change = data["price_change_pct"]

                    if abs(price_change) > 5.0:
                        risk_alerts.append(
                            {
                                "symbol": symbol,
                                "alert_type": "高ボラティリティ警告",
                                "severity": "高",
                                "message": f"{symbol}: 価格変動が大きい ({price_change:.1f}%)",
                                "recommendation": "注意深い監視が必要",
                                "timestamp": datetime.now(),
                            }
                        )
                        self.system_stats["alerts_issued"] += 1

            # 市場全体のリスクチェック
            if market_analysis and "market_overview" in market_analysis:
                market_overview = market_analysis["market_overview"]
                if market_overview.get("market_volatility") == "高":
                    risk_alerts.append(
                        {
                            "symbol": "全体",
                            "alert_type": "市場ボラティリティ警告",
                            "severity": "中",
                            "message": "市場全体のボラティリティが高くなっています",
                            "recommendation": "慎重な取引を推奨",
                            "timestamp": datetime.now(),
                        }
                    )
                    self.system_stats["alerts_issued"] += 1

            return risk_alerts

        except Exception as e:
            logger.error(f"リスク分析エラー: {e}")
            return []

    async def _generate_integrated_report(
        self,
        market_analysis: Dict[str, Any],
        trading_suggestions: Dict[str, Any],
        risk_alerts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """統合レポート生成"""
        try:
            report = {
                "timestamp": datetime.now(),
                "system_status": {
                    "running": self.is_running,
                    "safe_mode": is_safe_mode(),
                    "trading_mode": self.trading_config.current_mode.value,
                    "monitored_symbols": len(self.symbols),
                },
                "market_analysis": market_analysis,
                "trading_suggestions": trading_suggestions,
                "risk_alerts": risk_alerts,
                "system_statistics": self.system_stats.copy(),
                "summary": self._generate_executive_summary(
                    market_analysis, trading_suggestions, risk_alerts
                ),
            }

            return report

        except Exception as e:
            logger.error(f"統合レポート生成エラー: {e}")
            return {"error": str(e)}

    def _generate_executive_summary(
        self,
        market_analysis: Dict[str, Any],
        trading_suggestions: Dict[str, Any],
        risk_alerts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """エグゼクティブサマリー生成"""
        try:
            summary = {
                "market_sentiment": "中立",
                "active_opportunities": 0,
                "risk_level": "低",
                "key_recommendations": [],
                "important_alerts": [],
            }

            # 市場センチメント判定
            if market_analysis and "market_overview" in market_analysis:
                market_overview = market_analysis["market_overview"]
                summary["market_sentiment"] = market_overview.get(
                    "overall_sentiment", "中立"
                )

            # 機会数算出
            if trading_suggestions:
                buy_candidates = 0
                for suggestion in trading_suggestions.values():
                    if (
                        isinstance(suggestion, dict)
                        and "trading_suggestions" in suggestion
                    ):
                        suggestions_text = " ".join(suggestion["trading_suggestions"])
                        if "買い検討" in suggestions_text:
                            buy_candidates += 1
                summary["active_opportunities"] = buy_candidates

            # リスクレベル判定
            if risk_alerts:
                high_severity_alerts = [
                    alert for alert in risk_alerts if alert.get("severity") == "高"
                ]
                if high_severity_alerts:
                    summary["risk_level"] = "高"
                elif len(risk_alerts) > 3:
                    summary["risk_level"] = "中"

            # 重要な推奨事項
            if market_analysis and "recommendation_summary" in market_analysis:
                recommendations = market_analysis["recommendation_summary"]
                summary["key_recommendations"] = recommendations.get(
                    "general_advice", []
                )

            # 重要アラート
            summary["important_alerts"] = [
                alert["message"] for alert in risk_alerts[-3:]  # 最新3件
            ]

            return summary

        except Exception as e:
            logger.error(f"エグゼクティブサマリー生成エラー: {e}")
            return {"error": str(e)}

    async def _log_analysis_results(self, integrated_report: Dict[str, Any]) -> None:
        """分析結果のログ出力"""
        try:
            if "summary" in integrated_report:
                summary = integrated_report["summary"]

                logger.info("=== 分析結果サマリー ===")
                logger.info(
                    f"市場センチメント: {summary.get('market_sentiment', 'N/A')}"
                )
                logger.info(f"投資機会: {summary.get('active_opportunities', 0)}件")
                logger.info(f"リスクレベル: {summary.get('risk_level', 'N/A')}")

                if summary.get("important_alerts"):
                    logger.warning("重要アラート:")
                    for alert in summary["important_alerts"]:
                        logger.warning(f"  ⚠️ {alert}")

            # リスクアラートの詳細ログ
            if "risk_alerts" in integrated_report:
                risk_alerts = integrated_report["risk_alerts"]
                if risk_alerts:
                    logger.warning(f"リスクアラート {len(risk_alerts)}件発生:")
                    for alert in risk_alerts:
                        logger.warning(f"  🚨 {alert.get('message', 'N/A')}")

        except Exception as e:
            logger.error(f"分析結果ログ出力エラー: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        try:
            return {
                "system_info": {
                    "name": "統合分析システム",
                    "version": "1.0",
                    "mode": self.trading_config.current_mode.value,
                    "safe_mode": is_safe_mode(),
                    "automatic_trading": "完全無効",
                },
                "runtime_status": {
                    "running": self.is_running,
                    "monitored_symbols": len(self.symbols),
                    "last_analysis": (
                        self.last_analysis_time.isoformat()
                        if self.last_analysis_time
                        else None
                    ),
                },
                "statistics": self.system_stats.copy(),
                "subsystem_status": {
                    "market_analysis": self.market_analysis.get_analysis_summary(),
                    "analysis_engine": self.analysis_engine.get_comprehensive_status(),
                },
                "enabled_features": self.trading_config.get_enabled_features(),
                "disabled_features": self.trading_config.get_disabled_features(),
            }

        except Exception as e:
            logger.error(f"システム状態取得エラー: {e}")
            return {"error": str(e)}

    async def generate_manual_analysis(
        self, symbol: str, analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """手動分析要求"""
        try:
            logger.info(f"手動分析実行: {symbol} (タイプ: {analysis_type})")

            # 市場データ取得
            market_data = await self._fetch_market_data()

            if symbol not in market_data:
                return {
                    "error": f"銘柄 {symbol} の市場データが取得できません",
                    "timestamp": datetime.now(),
                }

            # 分析実行
            analysis_result = {
                "symbol": symbol,
                "analysis_type": analysis_type,
                "timestamp": datetime.now(),
                "market_data": market_data[symbol],
                "analysis": await self.market_analysis.perform_comprehensive_market_analysis(
                    {symbol: market_data[symbol]}
                ),
                "trading_suggestion": self.manual_trading_support.generate_trading_suggestion(
                    symbol, market_data
                ),
            }

            logger.info(f"手動分析完了: {symbol}")

            return analysis_result

        except Exception as e:
            logger.error(f"手動分析エラー: {e}")
            return {"error": str(e)}


# 使用例とシステム起動用の便利関数
async def start_analysis_system_example():
    """統合分析システム起動例"""
    symbols = ["7203", "6758", "9984"]  # トヨタ、ソニー、ソフトバンク

    try:
        # システム初期化
        system = IntegratedAnalysisSystem(symbols)

        # 包括的分析開始
        await system.start_comprehensive_analysis(analysis_interval=30.0)

    except KeyboardInterrupt:
        logger.info("ユーザーによる停止要求")
        await system.stop_analysis_system()
    except Exception as e:
        logger.error(f"システム実行エラー: {e}")


if __name__ == "__main__":
    # システム設定確認
    log_current_configuration()

    # 分析システム起動
    asyncio.run(start_analysis_system_example())
