"""
市場分析・監視エンジン（旧：リスク認識取引エンジン）

【重要】自動取引機能は無効化済み
分析・情報提供・手動取引支援に特化したシステム。
包括的な市場監視と分析機能を提供。

注意：このエンジンは取引を実行しません
"""

import asyncio
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..analysis.signals import TradingSignal, TradingSignalGenerator
from ..config.trading_mode_config import get_current_trading_config, is_safe_mode
from ..core.trade_manager import TradeManager
from ..data.stock_fetcher import StockFetcher
from ..utils.enhanced_error_handler import get_default_error_handler
from ..utils.logging_config import get_context_logger
from .enhanced_trading_engine import EnhancedTradingEngine, ExecutionMode
from .risk_manager import (
    EmergencyReason,
    RiskAlert,
    RiskLevel,
    RiskLimits,
    RiskManager,
)
from .trading_engine import EngineStatus, RiskParameters

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


class MarketAnalysisEngine(EnhancedTradingEngine):
    """
    市場分析・監視エンジン（旧：RiskAwareTradingEngine）

    【重要】自動取引機能は完全に無効化済み

    提供機能:
    1. 市場データのリアルタイム監視
    2. シグナル分析と情報提供
    3. ポートフォリオ状況の追跡
    4. リスク分析とアラート
    5. 手動取引支援情報の生成

    ※ 注文実行・ポジション操作は行いません
    """

    def __init__(
        self,
        symbols: List[str],
        trade_manager: Optional[TradeManager] = None,
        signal_generator: Optional[TradingSignalGenerator] = None,
        stock_fetcher: Optional[StockFetcher] = None,
        risk_params: Optional[RiskParameters] = None,
        risk_limits: Optional[RiskLimits] = None,
        initial_cash: Decimal = Decimal("1000000"),
        execution_mode: ExecutionMode = ExecutionMode.BALANCED,
        update_interval: float = 1.0,
        emergency_stop_enabled: bool = True,
    ):
        # 安全確認：自動取引が無効化されているかチェック
        if not is_safe_mode():
            raise RuntimeError("安全性エラー: 自動取引が無効化されていません")
        # 親クラス初期化
        super().__init__(
            symbols=symbols,
            trade_manager=trade_manager,
            signal_generator=signal_generator,
            stock_fetcher=stock_fetcher,
            risk_params=risk_params,
            initial_cash=initial_cash,
            execution_mode=execution_mode,
            update_interval=update_interval,
        )

        # リスク管理システム初期化
        self.risk_manager = RiskManager(
            risk_limits=risk_limits,
            alert_callback=self._handle_risk_alert,
            emergency_callback=self._handle_emergency_stop,
        )

        self.emergency_stop_enabled = emergency_stop_enabled

        # 分析関連統計
        self.analysis_stats = {
            "signals_analyzed": 0,
            "alerts_generated": 0,
            "analysis_cycles": 0,
            "market_updates": 0,
            "last_analysis": None,
        }

        # 取引設定確認
        self.trading_config = get_current_trading_config()

        logger.info(
            f"市場分析エンジン初期化完了 - "
            f"モード: {self.trading_config.current_mode.value} "
            f"(自動取引: 無効)"
        )

    async def start(self) -> None:
        """分析エンジン開始（取引機能無効）"""
        if self.status == EngineStatus.RUNNING:
            logger.warning("分析エンジンは既に実行中です")
            return

        # 安全確認
        if not is_safe_mode():
            logger.error("安全性エラー: 自動取引が無効化されていません")
            return

        logger.info("市場分析エンジンを開始します... (取引実行なし)")

        # リスク監視開始（分析目的のみ）
        await self.risk_manager.start_monitoring()

        # 分析専用モードで開始
        self.status = EngineStatus.RUNNING
        self._stop_event.clear()

        # メインループを非同期で実行
        self._main_task = asyncio.create_task(self._main_analysis_loop())

    async def stop(self) -> None:
        """分析エンジン停止"""
        logger.info("市場分析エンジン停止要求受信")

        # リスク監視停止
        await self.risk_manager.stop_monitoring()

        # エンジン停止
        self.status = EngineStatus.STOPPED
        self._stop_event.set()

        # メインタスクが存在する場合は停止を待機
        if hasattr(self, "_main_task") and self._main_task:
            try:
                await asyncio.wait_for(self._main_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("メインタスクの停止がタイムアウトしました")
                self._main_task.cancel()

        logger.info("市場分析エンジン停止完了")

    async def _main_analysis_loop(self) -> None:
        """分析専用メインループ（取引実行なし）"""
        cycle_count = 0

        logger.info("分析専用メインループ開始")

        while not self._stop_event.is_set() and self.status != EngineStatus.STOPPED:
            if self.status == EngineStatus.PAUSED:
                await asyncio.sleep(1.0)
                continue

            cycle_start = time.time()

            try:
                # 1. 市場データ更新
                await self._update_market_data_analysis_only()
                self.analysis_stats["market_updates"] += 1

                # 2. ポジション監視（分析目的のみ）
                await self._analyze_positions_only()

                # 3. シグナル生成と分析（実行なし）
                signals = await self._generate_analysis_signals()

                # 4. 市場状況の包括的分析
                market_analysis = await self._perform_market_analysis()

                # 5. 分析結果のログ出力
                await self._log_analysis_results(signals, market_analysis)

                # 6. 統計更新
                cycle_time = time.time() - cycle_start
                self._update_analysis_stats(cycle_time)

                cycle_count += 1
                self.analysis_stats["analysis_cycles"] = cycle_count
                self.analysis_stats["last_analysis"] = datetime.now()

                # インターバル調整
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"分析ループエラー: {e}")
                error_handler.handle_error(
                    e,
                    context={
                        "cycle_count": cycle_count,
                        "engine_status": self.status.value,
                        "analysis_mode": True,
                    },
                )

                # エラー時は短時間待機してリトライ
                await asyncio.sleep(5.0)

        logger.info("分析専用メインループ終了")

    async def _update_market_data_analysis_only(self) -> None:
        """市場データ更新（分析専用）"""
        try:
            # 親クラスの市場データ更新機能を利用（取引実行なし）
            await self._update_market_data()
        except Exception as e:
            logger.error(f"市場データ更新エラー: {e}")

    async def _analyze_positions_only(self) -> None:
        """ポジション分析（実行なし）"""
        try:
            # 現在のポジション状況を分析するが取引は実行しない
            positions = self.portfolio_manager.get_all_positions()

            if positions:
                logger.info(f"現在のポジション数: {len(positions)}")
                for symbol, position in positions.items():
                    logger.info(
                        f"  {symbol}: {position.quantity}株 "
                        f"平均価格:{position.average_price} "
                        f"現在価格:{position.current_price} "
                        f"損益:{position.unrealized_pnl}"
                    )
        except Exception as e:
            logger.error(f"ポジション分析エラー: {e}")

    async def _generate_analysis_signals(self) -> List[Tuple[str, TradingSignal]]:
        """分析用シグナル生成（取引実行なし）"""
        try:
            signals = await self._generate_trading_signals()
            self.analysis_stats["signals_analyzed"] += len(signals)
            return signals
        except Exception as e:
            logger.error(f"分析シグナル生成エラー: {e}")
            return []

    async def _perform_market_analysis(self) -> Dict[str, Any]:
        """包括的市場分析"""
        try:
            analysis = {
                "timestamp": datetime.now(),
                "market_status": "分析中",
                "active_symbols": len(self.market_data),
                "trading_mode": self.trading_config.current_mode.value,
                "safe_mode": is_safe_mode(),
            }
            return analysis
        except Exception as e:
            logger.error(f"市場分析エラー: {e}")
            return {}

    async def _log_analysis_results(
        self, signals: List[Tuple[str, TradingSignal]], market_analysis: Dict[str, Any]
    ) -> None:
        """分析結果のログ出力"""
        try:
            if signals:
                logger.info(f"生成されたシグナル数: {len(signals)}")
                for symbol, signal in signals:
                    logger.info(
                        f"  {symbol}: {signal.signal_type.value} "
                        f"強度:{signal.strength.value} "
                        f"信頼度:{signal.confidence:.1f}% "
                        f"理由:{', '.join(signal.reasons)}"
                    )

            if market_analysis:
                logger.info(f"市場分析: {market_analysis}")
        except Exception as e:
            logger.error(f"分析結果ログ出力エラー: {e}")

    def _update_analysis_stats(self, cycle_time: float) -> None:
        """分析統計更新"""
        try:
            # 実行時間の統計など
            if not hasattr(self, "_cycle_times"):
                self._cycle_times = []

            self._cycle_times.append(cycle_time)

            # 直近100サイクルの平均時間を保持
            if len(self._cycle_times) > 100:
                self._cycle_times = self._cycle_times[-100:]

            avg_cycle_time = sum(self._cycle_times) / len(self._cycle_times)

            if self.analysis_stats["analysis_cycles"] % 10 == 0:  # 10サイクルごとに出力
                logger.debug(
                    f"分析統計: サイクル数={self.analysis_stats['analysis_cycles']} "
                    f"平均実行時間={avg_cycle_time:.3f}秒"
                )
        except Exception as e:
            logger.error(f"分析統計更新エラー: {e}")

    async def _check_emergency_conditions(self) -> None:
        """緊急停止条件チェック"""
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()

            # ポートフォリオサマリーを辞書形式に変換
            summary_dict = {
                "daily_pnl": portfolio_summary.daily_pnl,
                "current_drawdown": getattr(
                    portfolio_summary, "current_drawdown", Decimal("0")
                ),
                "total_positions": portfolio_summary.total_positions,
                "total_equity": portfolio_summary.total_equity,
            }

            emergency_reason = self.risk_manager.check_emergency_conditions(
                summary_dict
            )

            if emergency_reason:
                await self.risk_manager.trigger_emergency_stop(
                    emergency_reason, f"自動検知 - {summary_dict}"
                )

        except Exception as e:
            logger.error(f"緊急停止条件チェックエラー: {e}")
            # エラー自体も緊急事態として扱う
            await self.risk_manager.trigger_emergency_stop(
                EmergencyReason.SYSTEM_ERROR, f"緊急停止条件チェック失敗: {e}"
            )

    async def _monitor_positions_with_risk(self) -> None:
        """リスク管理統合ポジション監視"""
        try:
            # 現在の市場価格を取得
            market_prices = {
                symbol: data.price for symbol, data in self.market_data.items()
            }

            # ポートフォリオの現在ポジション
            positions = {}
            for symbol, position in self.portfolio_manager.get_all_positions().items():
                positions[symbol] = {
                    "quantity": position.quantity,
                    "average_price": position.average_price,
                    "current_price": position.current_price,
                    "unrealized_pnl": position.unrealized_pnl,
                }

            # リスク監視実行
            alerts = self.risk_manager.monitor_positions(positions, market_prices)

            # 重要なアラートに対する自動対応
            for alert in alerts:
                if alert.risk_level == RiskLevel.CRITICAL:
                    await self._handle_critical_position_alert(alert)

        except Exception as e:
            logger.error(f"リスク統合ポジション監視エラー: {e}")

    async def _comprehensive_risk_check_async(self) -> Dict[str, Any]:
        """非同期対応包括的リスクチェック"""
        try:
            # 親クラスのリスクチェック
            basic_risk = self._comprehensive_risk_check()

            # RiskManagerによる追加チェック（ダミーデータで）
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            positions = {
                symbol: {
                    "quantity": position.quantity,
                    "average_price": position.average_price,
                    "current_price": position.current_price,
                }
                for symbol, position in self.portfolio_manager.get_all_positions().items()
            }

            enhanced_risk = {
                "approved": basic_risk["approved"],
                "reason": basic_risk["reason"],
                "risk_manager_active": not self.risk_manager.is_emergency_stopped,
                "portfolio_risk": basic_risk.get("portfolio_risk", {}),
                "risk_score": basic_risk.get("risk_score", 0.0),
            }

            # 緊急停止状態では全て拒否
            if self.risk_manager.is_emergency_stopped:
                enhanced_risk["approved"] = False
                enhanced_risk["reason"] = "緊急停止状態"

            self.risk_stats["last_risk_check"] = datetime.now()

            return enhanced_risk

        except Exception as e:
            logger.error(f"包括的リスクチェックエラー: {e}")
            return {
                "approved": False,
                "reason": f"リスクチェックエラー: {e}",
                "error": True,
            }

    async def _process_trading_signals_with_risk(
        self, signals: List[Tuple[str, TradingSignal]]
    ) -> None:
        """【無効化済み】取引シグナル処理（分析のみ）"""
        # 自動取引機能は完全に無効化
        logger.warning("取引実行は無効化されています - 分析情報のみ提供")

        try:
            for symbol, signal in signals:
                # 分析情報のみログ出力
                logger.info(
                    f"【分析のみ】シグナル検出: {symbol} {signal.signal_type.value} "
                    f"信頼度:{signal.confidence:.1f}% "
                    f"(実行されません)"
                )

        except Exception as e:
            logger.error(f"シグナル分析エラー: {e}")

    async def _calculate_risk_aware_position_size(
        self, symbol: str, signal: TradingSignal
    ) -> int:
        """【無効化済み】ポジションサイズ計算（分析のみ）"""
        # 実際の取引は行わないが、分析情報として計算結果を提供
        try:
            if symbol not in self.market_data:
                return 0

            current_price = self.market_data[symbol].price
            portfolio_equity = (
                self.portfolio_manager.get_portfolio_summary().total_equity
            )

            # 分析用の推奨サイズ計算（実際の注文には使用されない）
            estimated_volatility = Decimal("0.02")

            # 理論的な推奨サイズ（参考情報のみ）
            optimal_size = self.risk_manager.calculate_optimal_position_size(
                symbol=symbol,
                signal_confidence=signal.confidence,
                current_price=current_price,
                portfolio_equity=portfolio_equity,
                volatility=estimated_volatility,
            )

            logger.info(
                f"【分析情報】推奨ポジションサイズ: {symbol} {optimal_size}株 "
                f"@{current_price} (参考情報・実行されません)"
            )

            return 0  # 常に0を返す（取引実行なし）

        except Exception as e:
            logger.error(f"ポジションサイズ分析エラー: {e}")
            return 0

    async def _create_and_submit_risk_aware_orders(
        self, symbol: str, signal: TradingSignal, quantity: int
    ) -> None:
        """【無効化済み】注文作成・提出（分析のみ）"""
        # 注文実行機能は完全に無効化
        logger.warning("注文実行機能は無効化されています")

        try:
            if symbol not in self.market_data:
                return

            current_price = self.market_data[symbol].price

            # 分析情報のみ出力
            logger.info(
                f"【分析情報】理論的注文内容: {symbol} {signal.signal_type.value} "
                f"{quantity}株 @{current_price} (信頼度: {signal.confidence:.1f}%) "
                f"※実際の注文は行われません"
            )

        except Exception as e:
            logger.error(f"注文分析エラー: {e}")

    async def _handle_critical_position_alert(self, alert: RiskAlert) -> None:
        """【無効化済み】重要ポジションアラート（通知のみ）"""
        try:
            logger.critical(f"【重要アラート】{alert.message}")

            if alert.symbol and "ストップロス" in alert.message:
                if alert.symbol in self.market_data:
                    current_price = self.market_data[alert.symbol].price
                    position = self.portfolio_manager.get_position(alert.symbol)

                    if position and not position.is_flat():
                        logger.critical(
                            f"【手動対応必要】ストップロス到達: {alert.symbol} @{current_price} "
                            f"現在数量:{position.quantity}株 "
                            f"※自動決済は無効化されています - 手動で確認してください"
                        )

                        # 統計のみ更新
                        self.analysis_stats["alerts_generated"] += 1

        except Exception as e:
            logger.error(f"アラート処理エラー: {e}")

    def _handle_risk_alert(self, alert: RiskAlert) -> None:
        """リスクアラートハンドラー（通知のみ）"""
        try:
            self.analysis_stats["alerts_generated"] += 1

            # アラートレベル別の通知（自動対応はなし）
            if alert.risk_level == RiskLevel.CRITICAL:
                logger.critical(f"【重要リスクアラート】{alert.message} ※手動確認必要")

                # 通知のみ実行（自動対応はスキップ）
                if alert.symbol:
                    asyncio.create_task(self._handle_critical_position_alert(alert))

            elif alert.risk_level == RiskLevel.HIGH:
                logger.error(f"【高リスクアラート】{alert.message}")

            elif alert.risk_level == RiskLevel.MEDIUM:
                logger.warning(f"【中リスクアラート】{alert.message}")

            else:
                logger.info(f"【低リスクアラート】{alert.message}")

        except Exception as e:
            logger.error(f"リスクアラートハンドラーエラー: {e}")

    async def _handle_emergency_stop(self, reason: EmergencyReason, info: str) -> None:
        """【更新済み】緊急停止ハンドラー（通知・監視停止のみ）"""
        try:
            self.analysis_stats["alerts_generated"] += 1

            logger.critical(f"【緊急アラート】{reason.value} - {info}")
            logger.critical("※自動取引は既に無効化されています")

            # 現在のポジション状況を報告（決済は行わない）
            try:
                positions = self.portfolio_manager.get_all_positions()
                if positions:
                    logger.critical("【手動確認必要】現在のポジション:")
                    for symbol, position in positions.items():
                        logger.critical(
                            f"  {symbol}: {position.quantity}株 "
                            f"損益:{position.unrealized_pnl} "
                            f"※手動で確認・対応してください"
                        )
                else:
                    logger.info("現在ポジションはありません")

            except Exception as e:
                logger.error(f"ポジション確認エラー: {e}")

            # 分析エンジン停止
            self.status = EngineStatus.STOPPED
            self._stop_event.set()

            logger.critical("分析エンジン緊急停止完了")

        except Exception as e:
            logger.critical(f"緊急停止ハンドラーエラー: {e}")

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """包括的ステータス（分析モード）"""
        try:
            # 基本ステータス
            status = {
                "engine_type": "MarketAnalysisEngine",
                "engine_status": self.status.value,
                "trading_mode": self.trading_config.current_mode.value,
                "safe_mode": is_safe_mode(),
                "automatic_trading": "無効",
                "timestamp": datetime.now().isoformat(),
            }

            # 分析統計
            status["analysis_stats"] = self.analysis_stats.copy()
            if status["analysis_stats"]["last_analysis"]:
                status["analysis_stats"]["last_analysis"] = status["analysis_stats"][
                    "last_analysis"
                ].isoformat()

            # 市場データ状況
            status["market_data"] = {
                "symbols_monitored": len(self.market_data),
                "symbols_list": list(self.market_data.keys()),
            }

            # リスク管理情報
            try:
                risk_report = self.risk_manager.get_risk_report()
                status["risk_monitoring"] = {
                    "monitoring_active": self.risk_manager._monitoring_task is not None,
                    "emergency_stopped": self.risk_manager.is_emergency_stopped,
                    "active_alerts": len(self.risk_manager.active_alerts),
                    "risk_metrics": risk_report.get("risk_metrics", {}),
                }
            except Exception:
                status["risk_monitoring"] = {"status": "エラー"}

            # 有効機能一覧
            status["enabled_features"] = self.trading_config.get_enabled_features()
            status["disabled_features"] = self.trading_config.get_disabled_features()

            return status

        except Exception as e:
            logger.error(f"ステータス取得エラー: {e}")
            return {
                "error": str(e),
                "engine_type": "MarketAnalysisEngine",
                "safe_mode": True,
                "automatic_trading": "無効",
            }

    def emergency_stop(self) -> None:
        """分析エンジン緊急停止"""
        logger.critical("分析エンジン緊急停止要求")

        # 分析エンジンの緊急停止
        asyncio.create_task(
            self.risk_manager.trigger_emergency_stop(
                EmergencyReason.MANUAL, "手動分析エンジン停止要求"
            )
        )

    def reset_emergency_stop(self, operator: str = "operator") -> bool:
        """分析エンジン緊急停止リセット"""
        try:
            if not self.risk_manager.is_emergency_stopped:
                logger.info("緊急停止状態ではありません")
                return False

            # 安全確認
            if not is_safe_mode():
                logger.error("安全性エラー: セーフモードが無効化されています")
                return False

            self.risk_manager.reset_emergency_stop(operator)

            # 分析エンジン状態をリセット
            if self.status == EngineStatus.STOPPED:
                self.status = EngineStatus.PAUSED

            logger.warning(
                f"分析エンジン緊急停止リセット完了 - 操作者: {operator} "
                f"※自動取引は引き続き無効です"
            )
            return True

        except Exception as e:
            logger.error(f"緊急停止リセットエラー: {e}")
            return False


# 下位互換性のためのエイリアス（警告付き）
class RiskAwareTradingEngine(MarketAnalysisEngine):
    """
    【非推奨】RiskAwareTradingEngine は MarketAnalysisEngine に名前変更されました

    このクラスは下位互換性のためのエイリアスです。
    新しいコードでは MarketAnalysisEngine を使用してください。
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "RiskAwareTradingEngine は非推奨です。"
            "MarketAnalysisEngine を使用してください。"
        )
        super().__init__(*args, **kwargs)
