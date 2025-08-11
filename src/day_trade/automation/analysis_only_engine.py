"""
分析専用エンジン

【重要】自動取引機能は一切含まれていません
市場データの取得、分析、情報提供のみを行います

※ 実際の取引実行は一切行いません
※ 完全にセーフモードで動作します
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from ..analysis.signals import TradingSignal, TradingSignalGenerator
from ..config.trading_mode_config import get_current_trading_config, is_safe_mode
from ..data.stock_fetcher import StockFetcher
from ..utils.enhanced_error_handler import get_default_error_handler
from ..utils.enhanced_performance_monitor import get_performance_monitor
from ..utils.exception_handler import ExceptionContext, log_exception
from ..utils.exceptions import (
    AnalysisError,
    DataError,
    NetworkError,
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()
performance_monitor = get_performance_monitor()


class AnalysisStatus(Enum):
    """分析エンジンステータス"""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class MarketAnalysis:
    """市場分析結果"""

    symbol: str
    current_price: Decimal
    analysis_timestamp: datetime
    signal: Optional[TradingSignal] = None
    volatility: Optional[float] = None
    volume_trend: Optional[str] = None
    price_trend: Optional[str] = None
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class AnalysisReport:
    """分析レポート"""

    timestamp: datetime
    total_symbols: int
    analyzed_symbols: int
    strong_signals: int
    medium_signals: int
    weak_signals: int
    market_sentiment: str
    top_recommendations: List[Dict[str, Any]]
    analysis_time_ms: float


class AnalysisOnlyEngine:
    """
    分析専用エンジン

    【重要】自動取引機能は一切含まれていません

    機能:
    1. 市場データの取得と監視
    2. テクニカル分析の実行
    3. トレーディングシグナルの生成
    4. 市場分析レポートの作成
    5. 投資情報の提供

    ※ 取引の実行や注文の送信は一切行いません
    """

    def __init__(
        self,
        symbols: List[str],
        signal_generator: Optional[TradingSignalGenerator] = None,
        stock_fetcher: Optional[StockFetcher] = None,
        update_interval: float = 30.0,  # 30秒間隔（分析のみなので長め）
    ):
        # セーフモードチェック
        if not is_safe_mode():
            raise ValueError("セーフモードでない場合は、このエンジンは使用できません")

        self.symbols = symbols
        self.signal_generator = signal_generator or TradingSignalGenerator()
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.update_interval = update_interval

        # 状態管理
        self.status = AnalysisStatus.STOPPED
        self.market_analyses: Dict[str, MarketAnalysis] = {}
        self.analysis_history: List[AnalysisReport] = []

        # 統計情報
        self.stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_analysis_time": 0.0,
            "last_analysis": None,
        }

        # 非同期処理用
        self._stop_event = asyncio.Event()

        # 設定確認
        self.trading_config = get_current_trading_config()

        logger.info(f"AnalysisOnlyEngine初期化完了 - 監視銘柄: {len(symbols)}")
        logger.info("※ 完全にセーフモード - 分析・情報提供のみ実行します")
        logger.info("※ 自動取引機能は一切含まれていません")

        # システム監視開始
        performance_monitor.start_system_monitoring(interval=30.0)
        logger.info("パフォーマンス監視システムを開始しました")

    async def start(self) -> None:
        """分析エンジン開始"""
        if self.status == AnalysisStatus.RUNNING:
            logger.warning("分析エンジンは既に実行中です")
            return

        logger.info("分析専用エンジンを開始します...")
        logger.info("※ 市場データの分析・情報提供のみを行います")
        logger.info("※ 取引の実行は一切行いません")

        self.status = AnalysisStatus.RUNNING
        self._stop_event.clear()

        # メイン分析ループ開始
        await self._analysis_loop()

    async def stop(self) -> None:
        """分析エンジン停止"""
        logger.info("分析エンジン停止要求受信")
        self.status = AnalysisStatus.STOPPED
        self._stop_event.set()

        # システム監視停止
        performance_monitor.stop_system_monitoring()
        logger.info("パフォーマンス監視システムを停止しました")

        logger.info("分析エンジンが停止しました")

    async def pause(self) -> None:
        """分析エンジン一時停止"""
        if self.status == AnalysisStatus.RUNNING:
            self.status = AnalysisStatus.PAUSED
            logger.info("分析エンジンを一時停止しました")

    async def resume(self) -> None:
        """分析エンジン再開"""
        if self.status == AnalysisStatus.PAUSED:
            self.status = AnalysisStatus.RUNNING
            logger.info("分析エンジンを再開しました")

    async def _analysis_loop(self) -> None:
        """メイン分析ループ"""
        try:
            while (
                not self._stop_event.is_set() and self.status != AnalysisStatus.STOPPED
            ):
                if self.status == AnalysisStatus.PAUSED:
                    await asyncio.sleep(5.0)
                    continue

                analysis_start = time.time()

                context = ExceptionContext("AnalysisEngine", "analysis_cycle")

                try:
                    # 市場データ分析の実行（パフォーマンス監視付き）
                    with performance_monitor.monitor("analysis_cycle", "analysis"):
                        await self._perform_market_analysis()

                    # レポート生成
                    report = self._generate_analysis_report(
                        time.time() - analysis_start
                    )
                    self.analysis_history.append(report)

                    # 統計更新
                    self._update_statistics(time.time() - analysis_start, success=True)

                    logger.info(
                        f"分析完了 - {report.analyzed_symbols}/{report.total_symbols}銘柄 "
                        f"({report.analysis_time_ms:.1f}ms)"
                    )

                except (DataError, NetworkError) as e:
                    # 回復可能なエラー - 統計更新してログ出力、処理継続
                    self._update_statistics(time.time() - analysis_start, success=False)
                    log_exception(
                        logger,
                        e,
                        {
                            "component": "AnalysisEngine",
                            "operation": "analysis_cycle",
                            "cycle_time": time.time() - analysis_start,
                        },
                        level="warning",
                    )

                except AnalysisError as e:
                    # 分析エラー - 統計更新してログ出力、処理継続
                    self._update_statistics(time.time() - analysis_start, success=False)
                    log_exception(
                        logger,
                        e,
                        {"component": "AnalysisEngine", "operation": "analysis_cycle"},
                    )

                except Exception as e:
                    # 予期しないエラー - AnalysisErrorに変換
                    self._update_statistics(time.time() - analysis_start, success=False)
                    analysis_error = context.handle(e)
                    log_exception(
                        logger,
                        analysis_error,
                        {"component": "AnalysisEngine", "operation": "analysis_cycle"},
                    )

                # インターバル待機
                await asyncio.sleep(self.update_interval)

        except AnalysisError as e:
            # 分析関連の重大エラー
            self.status = AnalysisStatus.ERROR
            log_exception(
                logger,
                e,
                {
                    "component": "AnalysisEngine",
                    "operation": "main_loop",
                    "engine_status": self.status.value,
                },
            )
            error_handler.handle_error(e, context={"engine_status": self.status.value})

        except Exception as e:
            # 予期しない重大エラー
            self.status = AnalysisStatus.ERROR
            main_context = ExceptionContext("AnalysisEngine", "main_loop")
            critical_error = main_context.handle(e)
            log_exception(
                logger,
                critical_error,
                {
                    "component": "AnalysisEngine",
                    "operation": "main_loop",
                    "engine_status": self.status.value,
                },
                level="critical",
            )
            error_handler.handle_error(
                critical_error, context={"engine_status": self.status.value}
            )

    async def _perform_market_analysis(self) -> None:
        """市場データ分析の実行"""
        successful_analyses = 0

        for symbol in self.symbols:
            try:
                # 現在価格取得（パフォーマンス監視付き）
                with performance_monitor.monitor(f"price_fetch_{symbol}", "data_fetch"):
                    current_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.stock_fetcher.get_current_price, symbol
                    )

                if not current_data:
                    logger.warning(f"価格データ取得失敗: {symbol}")
                    continue

                current_price = Decimal(str(current_data["current_price"]))

                # 履歴データ取得（パフォーマンス監視付き）
                with performance_monitor.monitor(
                    f"historical_data_{symbol}", "data_fetch"
                ):
                    historical_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.stock_fetcher.get_historical_data, symbol, "30d"
                    )

                # シグナル生成（パフォーマンス監視付き）
                signal = None
                if historical_data is not None and not historical_data.empty:
                    with performance_monitor.monitor(
                        f"signal_generation_{symbol}", "ml_analysis"
                    ):
                        signal = self.signal_generator.generate_signal(historical_data)

                # 分析結果作成（パフォーマンス監視付き）
                with performance_monitor.monitor(
                    f"analysis_creation_{symbol}", "analysis"
                ):
                    analysis = MarketAnalysis(
                        symbol=symbol,
                        current_price=current_price,
                        analysis_timestamp=datetime.now(),
                        signal=signal,
                        volatility=self._calculate_volatility(historical_data)
                        if historical_data is not None
                        else None,
                        volume_trend=self._analyze_volume_trend(historical_data)
                        if historical_data is not None
                        else None,
                        price_trend=self._analyze_price_trend(historical_data)
                        if historical_data is not None
                        else None,
                        recommendations=self._generate_recommendations(
                            symbol, current_price, signal
                        ),
                    )

                self.market_analyses[symbol] = analysis
                successful_analyses += 1

                # シグナルが強い場合は詳細ログ出力
                if signal and signal.confidence >= 80.0:
                    logger.info(
                        f"【強いシグナル検出】{symbol}: {signal.signal_type.value} "
                        f"(信頼度: {signal.confidence:.1f}%)"
                    )

            except (KeyError, ValueError) as e:
                # データ形式エラー - 特定銘柄のみスキップ
                data_error = DataError(
                    message=f"銘柄データ処理エラー: {symbol}",
                    error_code="SYMBOL_DATA_ERROR",
                    details={"symbol": symbol, "original_error": str(e)},
                )
                log_exception(
                    logger,
                    data_error,
                    {
                        "component": "AnalysisEngine",
                        "operation": "symbol_analysis",
                        "symbol": symbol,
                    },
                    level="warning",
                )

            except Exception as e:
                # 予期しないエラー - 銘柄をスキップして継続
                symbol_context = ExceptionContext(
                    "AnalysisEngine", f"symbol_analysis_{symbol}"
                )
                symbol_error = symbol_context.handle(e)
                log_exception(
                    logger,
                    symbol_error,
                    {
                        "component": "AnalysisEngine",
                        "operation": "symbol_analysis",
                        "symbol": symbol,
                    },
                    level="warning",
                )

        logger.info(f"市場分析完了 - {successful_analyses}/{len(self.symbols)}銘柄")

    def _calculate_volatility(self, data) -> Optional[float]:
        """ボラティリティ計算"""
        try:
            if data is None or data.empty or len(data) < 2:
                return None

            returns = data["Close"].pct_change().dropna()
            if len(returns) > 0:
                return float(returns.std() * (252**0.5))  # 年率ボラティリティ
            return None
        except (KeyError, AttributeError) as e:
            # データ構造エラー - 予期されるエラー
            logger.debug(f"ボラティリティ計算: データ構造不正 - {e}")
            return None
        except Exception as e:
            # 予期しないエラー
            calc_context = ExceptionContext("AnalysisEngine", "volatility_calculation")
            calc_error = calc_context.handle(e)
            log_exception(
                logger,
                calc_error,
                {
                    "component": "AnalysisEngine",
                    "operation": "volatility_calculation",
                    "data_type": type(data).__name__ if data is not None else "None",
                },
                level="warning",
            )
            return None

    def _analyze_volume_trend(self, data) -> Optional[str]:
        """出来高トレンド分析"""
        try:
            if data is None or data.empty or len(data) < 5:
                return None

            recent_volume = data["Volume"].tail(3).mean()
            older_volume = data["Volume"].head(3).mean()

            if recent_volume > older_volume * 1.2:
                return "増加"
            elif recent_volume < older_volume * 0.8:
                return "減少"
            else:
                return "安定"
        except (KeyError, AttributeError) as e:
            logger.debug(f"出来高トレンド分析: データ構造不正 - {e}")
            return None
        except Exception as e:
            volume_context = ExceptionContext("AnalysisEngine", "volume_trend_analysis")
            volume_error = volume_context.handle(e)
            log_exception(
                logger,
                volume_error,
                {"component": "AnalysisEngine", "operation": "volume_trend_analysis"},
                level="warning",
            )
            return None

    def _analyze_price_trend(self, data) -> Optional[str]:
        """価格トレンド分析"""
        try:
            if data is None or data.empty or len(data) < 5:
                return None

            recent_price = data["Close"].tail(1).iloc[0]
            ma_5 = data["Close"].tail(5).mean()
            ma_10 = data["Close"].tail(10).mean() if len(data) >= 10 else ma_5

            if recent_price > ma_5 > ma_10:
                return "上昇"
            elif recent_price < ma_5 < ma_10:
                return "下降"
            else:
                return "横ばい"
        except (KeyError, AttributeError, IndexError) as e:
            logger.debug(f"価格トレンド分析: データ構造不正 - {e}")
            return None
        except Exception as e:
            price_context = ExceptionContext("AnalysisEngine", "price_trend_analysis")
            price_error = price_context.handle(e)
            log_exception(
                logger,
                price_error,
                {"component": "AnalysisEngine", "operation": "price_trend_analysis"},
                level="warning",
            )
            return None

    def _generate_recommendations(
        self, symbol: str, price: Decimal, signal: Optional[TradingSignal]
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        try:
            recommendations.append(f"現在価格: {price:,.0f}円")

            if signal:
                confidence_level = (
                    "高"
                    if signal.confidence >= 80
                    else "中"
                    if signal.confidence >= 60
                    else "低"
                )
                action = "買い注目" if signal.signal_type.value == "buy" else "売り注目"
                recommendations.append(
                    f"シグナル: {action} (信頼度: {confidence_level})"
                )

                if hasattr(signal, "reasoning") and signal.reasoning:
                    recommendations.append(f"根拠: {signal.reasoning}")
            else:
                recommendations.append("シグナル: なし")

            recommendations.append("※ これは分析情報です")
            recommendations.append("※ 投資判断は自己責任で行ってください")
            recommendations.append("※ 自動取引は実行されません")

        except (AttributeError, ValueError) as e:
            # シグナル関連のデータエラー - デフォルト推奨事項を返す
            logger.debug(f"推奨事項生成: シグナルデータ不正 - {symbol}: {e}")
            recommendations = [
                f"現在価格: {price:,.0f}円",
                "シグナル: データ不足",
                "※ これは分析情報です",
                "※ 投資判断は自己責任で行ってください",
                "※ 自動取引は実行されません",
            ]
        except Exception as e:
            # 予期しないエラー
            rec_context = ExceptionContext(
                "AnalysisEngine", f"recommendations_{symbol}"
            )
            rec_error = rec_context.handle(e)
            log_exception(
                logger,
                rec_error,
                {
                    "component": "AnalysisEngine",
                    "operation": "recommendations_generation",
                    "symbol": symbol,
                },
            )
            recommendations = ["分析情報の生成に失敗しました"]

        return recommendations

    def _generate_analysis_report(self, analysis_time: float) -> AnalysisReport:
        """分析レポート生成"""
        strong_signals = 0
        medium_signals = 0
        weak_signals = 0
        analyzed_count = 0

        top_recommendations = []

        for symbol, analysis in self.market_analyses.items():
            analyzed_count += 1

            if analysis.signal:
                if analysis.signal.confidence >= 80:
                    strong_signals += 1
                elif analysis.signal.confidence >= 60:
                    medium_signals += 1
                else:
                    weak_signals += 1

                # トップ推奨に追加（高信頼度のシグナルのみ）
                if analysis.signal.confidence >= 70:
                    top_recommendations.append(
                        {
                            "symbol": symbol,
                            "action": analysis.signal.signal_type.value,
                            "confidence": analysis.signal.confidence,
                            "price": float(analysis.current_price),
                        }
                    )

        # トップ推奨を信頼度順にソート
        top_recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        top_recommendations = top_recommendations[:5]  # 上位5つ

        # 市場センチメント判定
        total_signals = strong_signals + medium_signals + weak_signals
        if total_signals == 0:
            sentiment = "中性"
        elif strong_signals > total_signals * 0.3:
            sentiment = "強気"
        elif weak_signals > total_signals * 0.5:
            sentiment = "弱気"
        else:
            sentiment = "中性"

        return AnalysisReport(
            timestamp=datetime.now(),
            total_symbols=len(self.symbols),
            analyzed_symbols=analyzed_count,
            strong_signals=strong_signals,
            medium_signals=medium_signals,
            weak_signals=weak_signals,
            market_sentiment=sentiment,
            top_recommendations=top_recommendations,
            analysis_time_ms=analysis_time * 1000,
        )

    def _update_statistics(self, analysis_time: float, success: bool) -> None:
        """統計情報更新"""
        self.stats["total_analyses"] += 1

        if success:
            self.stats["successful_analyses"] += 1
        else:
            self.stats["failed_analyses"] += 1

        # 移動平均で分析時間を更新
        if self.stats["avg_analysis_time"] == 0:
            self.stats["avg_analysis_time"] = analysis_time
        else:
            alpha = 0.1
            self.stats["avg_analysis_time"] = (
                alpha * analysis_time + (1 - alpha) * self.stats["avg_analysis_time"]
            )

        self.stats["last_analysis"] = datetime.now()

    def get_latest_analysis(self, symbol: str) -> Optional[MarketAnalysis]:
        """最新の分析結果を取得"""
        return self.market_analyses.get(symbol)

    def get_all_analyses(self) -> Dict[str, MarketAnalysis]:
        """全ての分析結果を取得"""
        return self.market_analyses.copy()

    def get_latest_report(self) -> Optional[AnalysisReport]:
        """最新の分析レポートを取得"""
        return self.analysis_history[-1] if self.analysis_history else None

    def get_analysis_history(self, limit: int = 10) -> List[AnalysisReport]:
        """分析履歴を取得"""
        return self.analysis_history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """エンジンの状態を取得"""
        return {
            "status": self.status.value,
            "monitored_symbols": len(self.symbols),
            "analyzed_symbols": len(self.market_analyses),
            "safe_mode": is_safe_mode(),
            "trading_disabled": True,  # 常にTrue
            "stats": self.stats.copy(),
            "latest_report": self.get_latest_report().__dict__
            if self.get_latest_report()
            else None,
        }

    def get_symbol_recommendations(self, symbol: str) -> List[str]:
        """特定銘柄の推奨事項を取得"""
        analysis = self.get_latest_analysis(symbol)
        if analysis:
            return analysis.recommendations
        return [f"{symbol}の分析データがありません"]

    def get_market_summary(self) -> Dict[str, Any]:
        """市場サマリーを取得"""
        latest_report = self.get_latest_report()
        if not latest_report:
            return {"error": "分析データがありません"}

        return {
            "総銘柄数": latest_report.total_symbols,
            "分析済み銘柄数": latest_report.analyzed_symbols,
            "強いシグナル": latest_report.strong_signals,
            "中程度シグナル": latest_report.medium_signals,
            "弱いシグナル": latest_report.weak_signals,
            "市場センチメント": latest_report.market_sentiment,
            "トップ推奨": latest_report.top_recommendations,
            "最終分析時刻": latest_report.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "注意": "これは分析情報です。実際の投資は自己責任で行ってください。",
        }
