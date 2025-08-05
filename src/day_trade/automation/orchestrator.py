"""
全自動化オーケストレーター
デイトレードの全工程を統合実行
"""

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..analysis.backtest import BacktestEngine
from ..analysis.enhanced_ensemble import (
    EnhancedEnsembleStrategy,
    PredictionHorizon,
)
from ..analysis.ensemble import (
    EnsembleStrategy,
    EnsembleTradingStrategy,
    EnsembleVotingType,
)
from ..analysis.indicators import TechnicalIndicators
from ..analysis.patterns import ChartPatternRecognizer
from ..analysis.signals import TradingSignalGenerator
from ..config.config_manager import ConfigManager
from ..core.alerts import AlertManager
from ..core.portfolio import PortfolioAnalyzer
from ..core.trade_manager import TradeManager
from ..core.watchlist import WatchlistManager
from ..data.stock_fetcher import StockFetcher
from ..utils.logging_config import (
    get_context_logger,
    log_error_with_context,
)
from ..utils.progress import ProgressType, multi_step_progress, progress_context

logger = get_context_logger(__name__)

# スクリーニング機能のインポート
try:
    from ..analysis.screener import StockScreener

    SCREENER_AVAILABLE = True
except ImportError:
    SCREENER_AVAILABLE = False
    logger.warning("スクリーニング機能は利用できません")


@dataclass
class ExecutionResult:
    """実行結果"""

    success: bool
    symbol: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class AutomationReport:
    """自動化実行レポート"""

    start_time: datetime
    end_time: datetime
    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    execution_results: List[ExecutionResult]
    generated_signals: List[Dict[str, Any]]
    triggered_alerts: List[Dict[str, Any]]
    portfolio_summary: Dict[str, Any]
    errors: List[str]


class DayTradeOrchestrator:
    """デイトレード全自動化オーケストレーター"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_manager = ConfigManager(config_path)
        self.execution_settings = self.config_manager.get_execution_settings()

        # ログレベル設定
        logging.getLogger().setLevel(
            getattr(logging, self.execution_settings.log_level)
        )

        # 各コンポーネントの初期化
        self.stock_fetcher = StockFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.pattern_recognizer = ChartPatternRecognizer()
        self.signal_generator = TradingSignalGenerator()

        # アンサンブル戦略の初期化
        ensemble_settings = self.config_manager.get_ensemble_settings()
        if ensemble_settings.enabled:
            strategy_type = EnsembleStrategy(ensemble_settings.strategy_type)
            voting_type = EnsembleVotingType(ensemble_settings.voting_type)

            # 従来のアンサンブル戦略
            self.ensemble_strategy = EnsembleTradingStrategy(
                ensemble_strategy=strategy_type,
                voting_type=voting_type,
                performance_file=ensemble_settings.performance_file_path,
            )

            # 強化アンサンブル戦略（MLモデル統合）
            self.enhanced_ensemble = EnhancedEnsembleStrategy(
                ensemble_strategy=strategy_type,
                voting_type=voting_type,
                enable_ml_models=True,
                performance_file=ensemble_settings.performance_file_path,
            )
            # 初期化時にMLモデルの訓練を実行
            self.train_ml_models(symbols=self.config_manager.get_symbol_codes())

            logger.info(
                f"アンサンブル戦略を有効化: {strategy_type.value}, 投票方式: {voting_type.value}"
            )
            logger.info("強化アンサンブル戦略（MLモデル統合）を有効化")
        else:
            self.ensemble_strategy = None
            self.enhanced_ensemble = None
            logger.info("アンサンブル戦略は無効化されています")

        self.trade_manager = TradeManager()
        self.portfolio_analyzer = PortfolioAnalyzer(
            self.trade_manager, self.stock_fetcher
        )
        self.watchlist_manager = WatchlistManager()
        self.alert_manager = AlertManager(self.stock_fetcher)

        # スクリーニング機能（利用可能な場合のみ）
        if SCREENER_AVAILABLE:
            self.stock_screener = StockScreener(self.stock_fetcher)
            logger.info("スクリーニング機能を有効化")
        else:
            self.stock_screener = None
            logger.info("スクリーニング機能は無効です")

        # バックテストエンジン（設定で有効な場合のみ）
        self.backtest_engine = None
        self.advanced_backtest_engine = None

        if self.config_manager.get_backtest_settings().enabled:
            self.backtest_engine = BacktestEngine(
                self.stock_fetcher, self.signal_generator
            )

            # 高度なバックテストエンジンも初期化
            try:
                from ..analysis.advanced_backtest import (
                    AdvancedBacktestEngine,
                    TradingCosts,
                )

                trading_costs = TradingCosts(
                    commission_rate=0.001,
                    bid_ask_spread_rate=0.001,
                    slippage_rate=0.0005,
                    market_impact_rate=0.0002,
                )

                self.advanced_backtest_engine = AdvancedBacktestEngine(
                    initial_capital=1000000.0,
                    trading_costs=trading_costs,
                    position_sizing="percent",
                    max_position_size=0.2,
                    realistic_execution=True,
                )

                logger.info("高度なバックテストエンジンを初期化")

            except ImportError as e:
                logger.warning(f"高度なバックテストエンジンの初期化に失敗: {e}")
                self.advanced_backtest_engine = None

        # 実行状態
        self.current_report: Optional[AutomationReport] = None
        self.is_running = False

    def run_full_automation(
        self,
        symbols: Optional[List[str]] = None,
        report_only: bool = False,
        show_progress: bool = True,
    ) -> AutomationReport:
        """
        全自動化処理を実行

        Args:
            symbols: 対象銘柄（未指定時は設定ファイルから取得）
            report_only: レポート生成のみ実行
            show_progress: 進捗表示フラグ

        Returns:
            実行レポート
        """
        start_time = datetime.now()
        self.is_running = True

        logger.info("=== デイトレード全自動化処理を開始 ===")

        try:
            # 対象銘柄の決定
            if symbols is None:
                symbols = self.config_manager.get_symbol_codes()

            logger.info(f"対象銘柄数: {len(symbols)}")

            # レポート初期化
            self.current_report = AutomationReport(
                start_time=start_time,
                end_time=start_time,
                total_symbols=len(symbols),
                successful_symbols=0,
                failed_symbols=0,
                execution_results=[],
                generated_signals=[],
                triggered_alerts=[],
                portfolio_summary={},
                errors=[],
            )

            if show_progress:
                # 進捗表示付きで実行
                steps = [
                    ("data_fetch", "株価データ取得", 3.0),
                    ("technical_analysis", "テクニカル分析", 4.0),
                    ("pattern_recognition", "パターン認識", 2.0),
                    ("signal_generation", "シグナル生成", 3.0),
                    ("ensemble_strategy", "アンサンブル戦略", 2.0),
                    ("portfolio_update", "ポートフォリオ更新", 1.0),
                    ("alerts_check", "アラート確認", 1.0),
                    ("report_generation", "レポート生成", 1.0),
                ]

                with multi_step_progress("デイトレード自動化実行", steps) as progress:
                    if not report_only:
                        # メイン処理実行
                        stock_data, analysis_results, pattern_results, signals, alerts = self._execute_main_pipeline(symbols)
                        # 結果を保存
                        self.current_report.generated_signals = signals
                        self.current_report.triggered_alerts = alerts

                        for _ in range(len(steps) - 1): #レポート生成以外のステップを完了
                            progress.complete_step()

                    else:
                        # レポートのみの場合は全ステップをスキップしてレポート生成
                        for _i in range(len(steps) - 1):
                            progress.complete_step()

                    # レポート生成
                    self._generate_reports()
                    progress.complete_step()
            else:
                # 進捗表示なしで実行（従来通り）
                if not report_only:
                    # メイン処理実行
                    stock_data, analysis_results, pattern_results, signals, alerts = self._execute_main_pipeline(symbols)
                    self.current_report.generated_signals = signals
                    self.current_report.triggered_alerts = alerts

                # レポート生成
                self._generate_reports()

            # 最終化
            self.current_report.end_time = datetime.now()
            execution_time = (self.current_report.end_time - start_time).total_seconds()

            logger.info(f"=== 全自動化処理完了 (実行時間: {execution_time:.2f}秒) ===")
            logger.info(
                f"成功: {self.current_report.successful_symbols}/{self.current_report.total_symbols}"
            )

            return self.current_report

        except Exception as e:
            logger.error(f"全自動化処理エラー: {e}")
            logger.error(traceback.format_exc())

            if self.current_report:
                self.current_report.errors.append(str(e))
                self.current_report.end_time = datetime.now()

            raise
        finally:
            self.is_running = False

    def _execute_main_pipeline_with_progress(self, symbols: List[str], progress):
        """進捗表示付きメイン処理パイプラインを実行"""
        logger.info("Step 1: 株価データ取得開始")
        stock_data = self._fetch_stock_data_batch(symbols, show_progress=False)
        progress.complete_step()

        logger.info("Step 2: テクニカル分析実行")
        analysis_results = self._run_technical_analysis_batch(
            stock_data, show_progress=False
        )
        progress.complete_step()

        logger.info("Step 3: パターン認識実行")
        pattern_results = self._run_pattern_recognition_batch(
            stock_data, show_progress=False
        )
        progress.complete_step()

        logger.info("Step 4: シグナル生成実行")
        signals = self._generate_signals_batch(
            analysis_results, pattern_results, stock_data, show_progress=False
        )
        progress.complete_step()

        logger.info("Step 5: アンサンブル戦略実行")
        self._run_ensemble_strategy(signals, stock_data, analysis_results, pattern_results)
        progress.complete_step()

        logger.info("Step 6: ポートフォリオ更新")
        self._update_portfolio_data()
        progress.complete_step()

        if self.backtest_engine:
            logger.info("Step 8: バックテスト実行")
            self._run_backtest_analysis(symbols)

        return stock_data, analysis_results, pattern_results, signals, alerts

    def _run_ensemble_strategy(
        self, signals: List[Dict[str, Any]], stock_data: Dict[str, Any], analysis_results: Dict[str, Any], pattern_results: Dict[str, Any]
    ):
        """アンサンブル戦略を実行"""
        if not self.ensemble_strategy:
            logger.info("アンサンブル戦略は設定されていません")
            return

        try:
            # シグナルをアンサンブル戦略で統合
            for symbol, data in stock_data.items():
                if data and data.get("historical") is not None:
                    historical_df = data["historical"]
                    analysis = analysis_results.get(symbol, {})
                    patterns = pattern_results.get(symbol, {})

                    # Convert the analysis dict to a proper indicators DataFrame
                    indicators_df = self._convert_analysis_to_indicators(analysis, historical_df)

                    if indicators_df.empty:
                        continue

                    # アンサンブル戦略でシグナル生成
                    ensemble_signal_result = self.ensemble_strategy.generate_ensemble_signal(
                        historical_df, indicators_df, patterns
                    )

                    if ensemble_signal_result:
                        # generate_ensemble_signalは単一のEnsembleSignalオブジェクトを返す
                        # ここでは元の信号リストに追加するために変換する
                        signals.append(
                            {
                                "symbol": symbol,
                                "type": ensemble_signal_result.ensemble_signal.signal_type.value.upper(),
                                "reason": f"Ensemble: {', '.join(ensemble_signal_result.ensemble_signal.reasons[:2])}",
                                "confidence": ensemble_signal_result.ensemble_confidence / 100.0,
                                "timestamp": datetime.now(),
                                "ensemble_details": {
                                    "strategy_type": self.ensemble_strategy.ensemble_strategy.value,
                                    "voting_type": self.ensemble_strategy.voting_type.value,
                                    "strategy_weights": ensemble_signal_result.strategy_weights,
                                    "voting_scores": ensemble_signal_result.voting_scores,
                                    "meta_features": ensemble_signal_result.meta_features,
                                },
                            }
                        )
                        logger.debug(
                            f"アンサンブル戦略シグナル生成: {symbol} (1個)"
                        )

        except Exception as e:
            error_msg = f"アンサンブル戦略エラー: {e}"
            logger.error(error_msg)
            self.current_report.errors.append(error_msg)

    def _execute_main_pipeline(self, symbols: List[str]):
        """メイン処理パイプラインを実行"""
        logger.info("Step 1: 株価データ取得開始")
        stock_data = self._fetch_stock_data_batch(symbols)

        logger.info("Step 2: テクニカル分析実行")
        analysis_results = self._run_technical_analysis_batch(stock_data)

        logger.info("Step 3: パターン認識実行")
        pattern_results = self._run_pattern_recognition_batch(stock_data)

        logger.info("Step 4: シグナル生成実行")
        signals = self._generate_signals_batch(
            analysis_results, pattern_results, stock_data
        )

        logger.info("Step 5: アンサンブル戦略実行")
        self._run_ensemble_strategy(signals, stock_data, analysis_results, pattern_results)

        logger.info("Step 6: ポートフォリオ更新")
        self._update_portfolio_data()

        logger.info("Step 7: アラートチェック実行")
        alerts = self._check_alerts_batch(stock_data)

        if self.backtest_engine:
            logger.info("Step 8: バックテスト実行")
            self._run_backtest_analysis(symbols)

        return stock_data, analysis_results, pattern_results, signals, alerts

    def _fetch_stock_data_batch(
        self, symbols: List[str], show_progress: bool = False
    ) -> Dict[str, Any]:
        """株価データを並列取得"""
        stock_data = {}

        def fetch_single_stock(symbol: str) -> Tuple[str, Any]:
            try:
                start_time = time.time()

                # 現在価格取得
                current_data = self.stock_fetcher.get_current_price(symbol)

                # 履歴データ取得（アンサンブル戦略のため3ヶ月分取得）
                historical_data = self.stock_fetcher.get_historical_data(
                    symbol, period="5y", interval="1d"
                )

                execution_time = time.time() - start_time

                result = ExecutionResult(
                    success=True,
                    symbol=symbol,
                    data={"current": current_data, "historical": historical_data},
                    execution_time=execution_time,
                )

                if self.current_report:
                    self.current_report.execution_results.append(result)
                    self.current_report.successful_symbols += 1

                return symbol, {"current": current_data, "historical": historical_data}

            except Exception as e:
                error_msg = f"データ取得エラー ({symbol}): {e}"
                logger.error(error_msg)

                result = ExecutionResult(
                    success=False, symbol=symbol, error=error_msg, execution_time=0.0
                )

                if self.current_report:
                    self.current_report.execution_results.append(result)
                    self.current_report.failed_symbols += 1
                    self.current_report.errors.append(error_msg)

                return symbol, None

        # 並列実行
        max_workers = min(self.execution_settings.max_concurrent_requests, len(symbols))

        if show_progress:
            with progress_context(
                f"株価データ取得 ({len(symbols)}銘柄)",
                total=len(symbols),
                progress_type=ProgressType.DETERMINATE,
            ) as progress, ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(fetch_single_stock, symbol): symbol
                    for symbol in symbols
                }

                for future in as_completed(future_to_symbol):
                    symbol, data = future.result()
                    if data:
                        stock_data[symbol] = data
                        progress.set_description(f"データ取得完了: {symbol}")
                    progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(fetch_single_stock, symbol): symbol
                    for symbol in symbols
                }

                for future in as_completed(future_to_symbol):
                    symbol, data = future.result()
                    if data:
                        stock_data[symbol] = data

        logger.info(f"株価データ取得完了: {len(stock_data)}/{len(symbols)} 銘柄")
        return stock_data

    def _run_technical_analysis_batch(
        self, stock_data: Dict[str, Any], show_progress: bool = False
    ) -> Dict[str, Dict]:
        """テクニカル分析を並列実行"""
        if not self.config_manager.get_technical_indicator_settings().enabled:
            logger.info("テクニカル分析は無効化されています")
            return {}

        analysis_results = {}

        if show_progress:
            with progress_context(
                f"テクニカル分析 ({len(stock_data)}銘柄)",
                total=len(stock_data),
                progress_type=ProgressType.DETERMINATE,
            ) as progress:
                for symbol, data in stock_data.items():
                    progress.set_description(f"テクニカル分析: {symbol}")
                    analysis_results[symbol] = self._analyze_single_symbol(symbol, data)
                    progress.update(1)
        else:
            for symbol, data in stock_data.items():
                analysis_results[symbol] = self._analyze_single_symbol(symbol, data)

        logger.info(f"テクニカル分析完了: {len(analysis_results)} 銘柄")
        return analysis_results

    def _analyze_single_symbol(
        self, symbol: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """単一銘柄のテクニカル分析"""
        try:
            if data and data.get("historical") is not None:
                historical = data["historical"]

                # 各種指標を計算
                indicators = {}

                # 移動平均
                settings = self.config_manager.get_technical_indicator_settings()
                for period in settings.sma_periods:
                    indicators[f"sma_{period}"] = self.technical_indicators.sma(
                        historical, period
                    )

                # RSI
                indicators["rsi"] = self.technical_indicators.rsi(
                    historical, settings.rsi_period
                )

                # MACD
                macd_data = self.technical_indicators.macd(
                    historical,
                    settings.macd_params["fast"],
                    settings.macd_params["slow"],
                    settings.macd_params["signal"],
                )
                indicators["macd"] = macd_data

                # ボリンジャーバンド
                bb_data = self.technical_indicators.bollinger_bands(
                    historical,
                    settings.bollinger_params["period"],
                    settings.bollinger_params["std_dev"],
                )
                indicators["bollinger"] = bb_data

                return indicators

        except Exception as e:
            error_msg = f"テクニカル分析エラー ({symbol}): {e}"
            logger.error(error_msg)
            self.current_report.errors.append(error_msg)

        return {}

    def _run_pattern_recognition_batch(
        self, stock_data: Dict[str, Any], show_progress: bool = False
    ) -> Dict[str, Dict]:
        """パターン認識を並列実行"""
        if not self.config_manager.get_pattern_recognition_settings().enabled:
            logger.info("パターン認識は無効化されています")
            return {}

        pattern_results = {}

        for symbol, data in stock_data.items():
            try:
                if data and data.get("historical") is not None:
                    historical = data["historical"]

                    # パターン認識実行
                    patterns = self.pattern_recognizer.detect_all_patterns(historical)
                    pattern_results[symbol] = patterns

            except Exception as e:
                error_msg = f"パターン認識エラー ({symbol}): {e}"
                logger.error(error_msg)
                self.current_report.errors.append(error_msg)

        pattern_results[symbol] = patterns

        logger.info(f"パターン認識完了: {len(pattern_results)} 銘柄")
        return pattern_results

    def _generate_signals_batch(
        self,
        analysis_results: Dict[str, Dict],
        pattern_results: Dict[str, Dict],
        stock_data: Dict[str, Any] = None,
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """シグナル生成を並列実行"""
        if not self.config_manager.get_signal_generation_settings().enabled:
            logger.info("シグナル生成は無効化されています")
            return []

        all_signals = []
        settings = self.config_manager.get_signal_generation_settings()

        for symbol in analysis_results:
            try:
                analysis = analysis_results.get(symbol, {})
                patterns = pattern_results.get(symbol, {})

                # historicalデータを取得
                symbol_stock_data = stock_data.get(symbol) if stock_data else None
                historical_df = (
                    symbol_stock_data.get("historical")
                    if symbol_stock_data
                    else pd.DataFrame()
                )

                # analysis辞書をDataFrameに変換
                indicators_df = self._convert_analysis_to_indicators(
                    analysis, historical_df
                )

                if not indicators_df.empty:
                    # 強化アンサンブル戦略が有効な場合は優先使用
                    if self.enhanced_ensemble:
                        enhanced_signals = self._generate_enhanced_ensemble_signals(
                            symbol, indicators_df, patterns, symbol_stock_data
                        )
                        all_signals.extend(enhanced_signals)
                    elif self.ensemble_strategy:
                        # 従来のアンサンブル戦略
                        # _generate_ensemble_signalsはindicators_dfを期待しているので、ここで渡す
                        ensemble_signals = self._generate_ensemble_signals(
                            symbol, indicators_df, patterns, symbol_stock_data
                        )
                        all_signals.extend(ensemble_signals)
                    else:
                        # 従来のシグナル生成
                        # _evaluate_trading_signalsはindicators_dfを期待しているので、ここで渡す
                        signals = self._evaluate_trading_signals(
                            symbol,
                            indicators_df,
                            patterns,
                            settings,  # analysisの代わりにindicators_dfを渡す
                        )
                        all_signals.extend(signals)

            except Exception as e:
                error_msg = f"シグナル生成エラー ({symbol}): {e}"
                logger.error(error_msg)
                self.current_report.errors.append(error_msg)

        logger.info(f"シグナル生成完了: {len(all_signals)} 個のシグナル")
        return all_signals

    def _generate_ensemble_signals(
        self, symbol: str, analysis: Dict, patterns: Dict, stock_data: Dict = None
    ) -> List[Dict[str, Any]]:
        """アンサンブル戦略によるシグナル生成"""
        signals = []

        try:
            from datetime import datetime

            # 実際の株価データを取得
            if stock_data and "historical" in stock_data:
                price_df = stock_data["historical"]
            else:
                # フォールバック：最小限のデータでも動作するようにする
                logger.warning(f"実際の価格データがありません ({symbol})")
                return []

            # データ量チェック
            if len(price_df) < 20:
                logger.warning(
                    f"データが不足しています ({symbol}): {len(price_df)}日分"
                )
                return []

            # analysis辞書からindicators_dfを生成
            indicators_df = self._convert_analysis_to_indicators(analysis, price_df)

            # アンサンブルシグナル生成
            ensemble_signal = self.ensemble_strategy.generate_ensemble_signal(
                price_df, indicators_df, patterns
            )

            if (
                ensemble_signal
                and ensemble_signal.ensemble_signal.signal_type.value != "hold"
            ):
                signal_data = {
                    "symbol": symbol,
                    "type": ensemble_signal.ensemble_signal.signal_type.value.upper(),
                    "reason": f"Ensemble: {', '.join(ensemble_signal.ensemble_signal.reasons[:2])}",
                    "confidence": ensemble_signal.ensemble_confidence / 100.0,
                    "timestamp": datetime.now(),
                    "ensemble_details": {
                        "strategy_type": self.ensemble_strategy.ensemble_strategy.value,
                        "voting_type": self.ensemble_strategy.voting_type.value,
                        "strategy_weights": ensemble_signal.strategy_weights,
                        "voting_scores": ensemble_signal.voting_scores,
                        "meta_features": ensemble_signal.meta_features,
                    },
                }
                signals.append(signal_data)

                logger.debug(
                    f"アンサンブルシグナル生成 ({symbol}): {signal_data['type']}, 信頼度: {signal_data['confidence']:.2f}"
                )

            return signals

        except Exception as e:
            logger.error(f"アンサンブルシグナル生成エラー ({symbol}): {e}")
            return []

    def _generate_enhanced_ensemble_signals(
        self, symbol: str, indicators: pd.DataFrame, patterns: Dict, stock_data: Dict = None
    ) -> List[Dict[str, Any]]:
        """強化アンサンブル戦略によるシグナル生成"""
        signals = []

        try:
            from datetime import datetime

            # 実際の株価データを取得
            if stock_data and "historical" in stock_data:
                price_df = stock_data["historical"]
            else:
                logger.warning(f"実際の価格データがありません ({symbol})")
                return []

            # データ量チェック
            if len(price_df) < 20:
                logger.warning(
                    f"データが不足しています ({symbol}): {len(price_df)}日分"
                )
                return []

            # 市場データ（利用可能な場合）
            market_data = None  # 今後の実装で市場データを追加予定

            # 強化アンサンブルシグナル生成
            enhanced_signal = self.enhanced_ensemble.generate_enhanced_signal(
                symbol, price_df, indicators, market_data, PredictionHorizon.SHORT_TERM
            )

            # HOLDシグナルもレポートに含める
            signal_data = {
                "symbol": symbol,
                "type": enhanced_signal.signal_type.value.upper(),
                "reason": f"Enhanced Ensemble: ML+Rules (confidence: {enhanced_signal.ensemble_confidence:.1f}%)",
                "confidence": enhanced_signal.ensemble_confidence / 100.0,
                "timestamp": datetime.now(),
                "enhanced_details": {
                    "prediction_horizon": enhanced_signal.prediction_horizon.value,
                    "price_target": enhanced_signal.price_target,
                    "uncertainty": enhanced_signal.uncertainty,
                    "risk_score": enhanced_signal.risk_score,
                    "market_context": {
                        "volatility_regime": enhanced_signal.market_context.volatility_regime,
                        "trend_direction": enhanced_signal.market_context.trend_direction,
                        "market_sentiment": enhanced_signal.market_context.market_sentiment,
                    },
                    "strategy_weights": enhanced_signal.strategy_weights,
                    "ml_predictions": {
                        name: {
                            "prediction": pred.prediction,
                            "confidence": pred.confidence,
                            "model_name": pred.model_name,
                        }
                        for name, pred in enhanced_signal.ml_predictions.items()
                    },
                    "rule_signals": {
                        name: {
                            "signal_type": signal.signal_type.value,
                            "confidence": signal.confidence,
                            "reasons": signal.reasons[:2],  # 上位2つの理由
                        }
                        for name, signal in enhanced_signal.rule_based_signals.items()
                    },
                },
            }
            signals.append(signal_data)

            logger.debug(
                f"強化アンサンブルシグナル生成 ({symbol}): {signal_data['type']}, "
                f"信頼度: {signal_data['confidence']:.2f}, "
                f"リスクスコア: {enhanced_signal.risk_score:.1f}"
            )

            return signals

        except Exception as e:
            logger.error(f"強化アンサンブルシグナル生成エラー ({symbol}): {e}")
            logger.error(f"エラー詳細: {str(e)}")
            return []

    def _convert_analysis_to_indicators(
        self, analysis: Dict, price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """分析結果を指標DataFrameに変換"""
        all_indicator_series = []
        index = price_df.index  # 基準となるインデックス

        try:
            # RSI変換
            if "rsi" in analysis and isinstance(analysis["rsi"], pd.Series):
                all_indicator_series.append(analysis["rsi"].rename("RSI"))

            # MACD変換
            if "macd" in analysis and isinstance(analysis["macd"], pd.DataFrame):
                macd_df = analysis["macd"]
                if "MACD" in macd_df.columns:
                    all_indicator_series.append(macd_df["MACD"].rename("MACD"))
                if "Signal" in macd_df.columns:
                    all_indicator_series.append(macd_df["Signal"].rename("MACD_Signal"))

            # ボリンジャーバンド変換
            if "bollinger" in analysis and isinstance(analysis["bollinger"], pd.DataFrame):
                bb_df = analysis["bollinger"]
                if "Upper" in bb_df.columns:
                    all_indicator_series.append(bb_df["Upper"].rename("BB_Upper"))
                if "Lower" in bb_df.columns:
                    all_indicator_series.append(bb_df["Lower"].rename("BB_Lower"))
                if "Middle" in bb_df.columns:
                    all_indicator_series.append(bb_df["Middle"].rename("BB_Middle"))

            # 移動平均変換 (sma_X)
            for key, value in analysis.items():
                if key.startswith("sma_") and isinstance(value, pd.Series):
                    all_indicator_series.append(value.rename(key.upper()))

            if not all_indicator_series:
                return pd.DataFrame(index=index)

            # すべてのSeriesを一つのDataFrameに結合
            # indexを統一し、不足している値を前方向/後方向で埋める
            indicators_df = (
                pd.concat(all_indicator_series, axis=1)
                .reindex(index)
                .ffill()
                .bfill()
            )
            # 全てNaNのカラムは削除
            indicators_df.dropna(axis=1, how="all", inplace=True)


            logger.debug(f"指標DataFrame変換完了: {len(indicators_df.columns)}列")
            return indicators_df

        except Exception as e:
            logger.error(f"指標DataFrame変換エラー: {e}")
            return pd.DataFrame(index=index)

    def _evaluate_trading_signals(
        self, symbol: str, indicators: pd.DataFrame, patterns: Dict, settings
    ) -> List[Dict[str, Any]]:
        """個別銘柄のシグナル評価"""
        signals = []

        try:
            # RSIシグナル
            if "RSI" in indicators.columns:
                rsi_values = indicators["RSI"]
                if not rsi_values.empty:
                    current_rsi = rsi_values.iloc[-1]

                    if current_rsi > 70:
                        signals.append(
                            {
                                "symbol": symbol,
                                "type": "SELL",
                                "reason": "RSI Overbought",
                                "confidence": 0.7,
                                "value": current_rsi,
                                "timestamp": datetime.now(),
                            }
                        )
                    elif current_rsi < 30:
                        signals.append(
                            {
                                "symbol": symbol,
                                "type": "BUY",
                                "reason": "RSI Oversold",
                                "confidence": 0.7,
                                "value": current_rsi,
                                "timestamp": datetime.now(),
                            }
                        )

            # 移動平均クロスオーバー
            if "SMA_5" in indicators.columns and "SMA_20" in indicators.columns:
                sma_5 = indicators["SMA_5"]
                sma_20 = indicators["SMA_20"]

                if len(sma_5) >= 2 and len(sma_20) >= 2:
                    if (
                        sma_5.iloc[-1] > sma_20.iloc[-1]
                        and sma_5.iloc[-2] <= sma_20.iloc[-2]
                    ):
                        signals.append(
                            {
                                "symbol": symbol,
                                "type": "BUY",
                                "reason": "Golden Cross (SMA)",
                                "confidence": 0.8,
                                "timestamp": datetime.now(),
                            }
                        )
                    elif (
                        sma_5.iloc[-1] < sma_20.iloc[-1]
                        and sma_5.iloc[-2] >= sma_20.iloc[-2]
                    ):
                        signals.append(
                            {
                                "symbol": symbol,
                                "type": "SELL",
                                "reason": "Dead Cross (SMA)",
                                "confidence": 0.8,
                                "timestamp": datetime.now(),
                            }
                        )

            # 信頼度フィルタリング
            filtered_signals = [
                signal
                for signal in signals
                if signal["confidence"] >= settings.confidence_threshold
            ]

            return filtered_signals

        except Exception as e:
            logger.error(f"シグナル評価エラー ({symbol}): {e}")
            return []

    def _check_alerts_batch(
        self, stock_data: Dict[str, Any], show_progress: bool = False
    ) -> List[Dict[str, Any]]:
        """アラートチェックを実行"""
        if not self.config_manager.get_alert_settings().enabled:
            logger.info("アラート機能は無効化されています")
            return []

        try:
            # アラートマネージャーでチェック実行
            self.alert_manager.check_all_alerts()

            # 最近のアラート履歴を取得
            recent_alerts = self.alert_manager.get_alert_history(hours=1)

            alert_list = []
            for alert in recent_alerts:
                alert_list.append(
                    {
                        "symbol": alert.symbol,
                        "type": alert.alert_type.value,
                        "message": alert.message,
                        "priority": alert.priority.value,
                        "timestamp": alert.trigger_time,
                        "current_value": alert.current_value,
                    }
                )

            logger.info(f"アラートチェック完了: {len(alert_list)} 個のアラート")
            return alert_list

        except Exception as e:
            error_msg = f"アラートチェックエラー: {e}"
            logger.error(error_msg)
            self.current_report.errors.append(error_msg)
            return []

    def _update_portfolio_data(self):
        """ポートフォリオデータを更新"""
        try:
            # ポートフォリオサマリーを取得
            summary = self.trade_manager.get_portfolio_summary()

            # ポートフォリオ分析を実行
            metrics = self.portfolio_analyzer.get_portfolio_metrics()

            self.current_report.portfolio_summary = {
                "summary": summary,
                "metrics": {
                    "total_value": str(metrics.total_value),
                    "total_cost": str(metrics.total_cost),
                    "total_pnl": str(metrics.total_pnl),
                    "total_pnl_percent": str(metrics.total_pnl_percent),
                    "volatility": metrics.volatility,
                    "sharpe_ratio": metrics.sharpe_ratio,
                },
                "timestamp": datetime.now().isoformat(),
            }

            logger.info("ポートフォリオデータ更新完了")

        except Exception as e:
            error_msg = f"ポートフォリオ更新エラー: {e}"
            logger.error(error_msg)
            self.current_report.errors.append(error_msg)

    def _run_backtest_analysis(self, symbols: List[str]):
        """バックテスト分析を実行"""
        try:
            settings = self.config_manager.get_backtest_settings()

            # バックテスト設定を作成
            from decimal import Decimal

            from ..analysis.backtest import BacktestConfig, BacktestMode

            config = BacktestConfig(
                start_date=datetime.now() - timedelta(days=settings.period_days),
                end_date=datetime.now(),
                initial_capital=Decimal(str(settings.initial_capital)),
                mode=BacktestMode.MULTI_SYMBOL,
                position_size_percent=settings.position_size_percent,
                max_positions=settings.max_positions,
                transaction_cost=Decimal("0.001"),
            )

            # バックテスト実行
            result = self.backtest_engine.run_backtest(symbols, config)

            logger.info(f"バックテスト完了: 最終資本 {result.final_capital}")

        except Exception as e:
            error_msg = f"バックテストエラー: {e}"
            logger.error(error_msg)
            self.current_report.errors.append(error_msg)

    def _generate_reports(self):
        """レポートを生成"""
        try:
            report_settings = self.config_manager.get_report_settings()

            if not report_settings.enabled:
                logger.info("レポート生成は無効化されています")
                return

            # レポートディレクトリを作成
            report_dir = Path(report_settings.output_directory)
            report_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # JSON形式でレポート保存
            if "json" in report_settings.formats:
                self._save_json_report(report_dir, timestamp)

            # CSV形式でレポート保存
            if "csv" in report_settings.formats:
                self._save_csv_report(report_dir, timestamp)

            # HTML形式でレポート保存
            if "html" in report_settings.formats:
                self._save_html_report(report_dir, timestamp)

            logger.info(f"レポート生成完了: {report_dir}")

        except Exception as e:
            error_msg = f"レポート生成エラー: {e}"
            logger.error(error_msg)
            self.current_report.errors.append(error_msg)

    def _save_json_report(self, report_dir: Path, timestamp: str):
        """JSONレポートを保存"""
        import json

        report_data = {
            "execution_summary": {
                "start_time": self.current_report.start_time.isoformat(),
                "end_time": self.current_report.end_time.isoformat(),
                "total_symbols": self.current_report.total_symbols,
                "successful_symbols": self.current_report.successful_symbols,
                "failed_symbols": self.current_report.failed_symbols,
                "execution_time": (
                    self.current_report.end_time - self.current_report.start_time
                ).total_seconds(),
            },
            "signals": [
                {
                    **signal,
                    "timestamp": (
                        signal["timestamp"].isoformat()
                        if "timestamp" in signal
                        else None
                    ),
                }
                for signal in self.current_report.generated_signals
            ],
            "alerts": [
                {
                    **alert,
                    "timestamp": (
                        alert["timestamp"].isoformat() if "timestamp" in alert else None
                    ),
                }
                for alert in self.current_report.triggered_alerts
            ],
            "portfolio": (
                {
                    **{
                        k: v
                        for k, v in self.current_report.portfolio_summary.items()
                        if k != "timestamp"
                    },
                    "timestamp": (
                        self.current_report.portfolio_summary.get(
                            "timestamp", datetime.now()
                        ).isoformat()
                        if isinstance(
                            self.current_report.portfolio_summary.get("timestamp"),
                            datetime,
                        )
                        else self.current_report.portfolio_summary.get("timestamp")
                    ),
                }
                if self.current_report.portfolio_summary
                else {}
            ),
            "errors": self.current_report.errors,
        }

        report_path = report_dir / f"daytrade_report_{timestamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

    def _save_csv_report(self, report_dir: Path, timestamp: str):
        """CSVレポートを保存"""
        import pandas as pd

        # シグナルCSV
        if self.current_report.generated_signals:
            signals_df = pd.DataFrame(self.current_report.generated_signals)
            signals_path = report_dir / f"signals_{timestamp}.csv"
            signals_df.to_csv(signals_path, index=False, encoding="utf-8-sig")

        # アラートCSV
        if self.current_report.triggered_alerts:
            alerts_df = pd.DataFrame(self.current_report.triggered_alerts)
            alerts_path = report_dir / f"alerts_{timestamp}.csv"
            alerts_df.to_csv(alerts_path, index=False, encoding="utf-8-sig")

    def run_stock_screening(
        self,
        symbols: Optional[List[str]] = None,
        screener_type: str = "default",
        min_score: float = 0.1,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        銘柄スクリーニングを実行

        Args:
            symbols: 対象銘柄（未指定時は設定から取得）
            screener_type: スクリーナータイプ（default, growth, value, momentum）
            min_score: 最小スコア閾値
            max_results: 最大結果数

        Returns:
            スクリーニング結果リスト
        """
        if not self.stock_screener:
            logger.error("スクリーニング機能が利用できません")
            return []

        if symbols is None:
            symbols = self.config_manager.get_symbol_codes()

        logger.info(
            f"銘柄スクリーニング開始: {len(symbols)}銘柄, タイプ: {screener_type}"
        )

        try:
            if screener_type == "default":
                results = self.stock_screener.screen_stocks(
                    symbols, min_score=min_score, max_results=max_results
                )
            else:
                # 事前定義されたスクリーナーを使用
                predefined_screeners = self.stock_screener.get_predefined_screeners()
                if screener_type in predefined_screeners:
                    screener_func = predefined_screeners[screener_type]
                    results = screener_func(
                        symbols, min_score=min_score, max_results=max_results
                    )
                else:
                    logger.error(f"不明なスクリーナータイプ: {screener_type}")
                    return []

            # 結果を辞書形式に変換
            screening_results = []
            for result in results:
                screening_results.append(
                    {
                        "symbol": result.symbol,
                        "score": result.score,
                        "matched_conditions": [
                            c.value for c in result.matched_conditions
                        ],
                        "last_price": result.last_price,
                        "volume": result.volume,
                        "technical_data": result.technical_data,
                    }
                )

            logger.info(f"スクリーニング完了: {len(screening_results)}銘柄")
            return screening_results

        except Exception as e:
            logger.error(f"スクリーニング実行エラー: {e}")
            return []

    def generate_screening_report(self, screening_results: List[Dict[str, Any]]) -> str:
        """
        スクリーニング結果のレポート生成

        Args:
            screening_results: スクリーニング結果

        Returns:
            レポート文字列
        """
        if not screening_results:
            return "スクリーニング条件を満たす銘柄が見つかりませんでした。"

        report_lines = [
            f"=== 銘柄スクリーニング結果 ({len(screening_results)}銘柄) ===",
            "",
        ]

        for i, result in enumerate(screening_results, 1):
            report_lines.extend(
                [
                    f"{i}. 銘柄コード: {result['symbol']}",
                    f"   スコア: {result['score']:.2f}",
                    (
                        f"   現在価格: ¥{result['last_price']:,.0f}"
                        if result["last_price"]
                        else "   現在価格: N/A"
                    ),
                    (
                        f"   出来高: {result['volume']:,}"
                        if result["volume"]
                        else "   出来高: N/A"
                    ),
                    f"   マッチした条件: {', '.join(result['matched_conditions'])}",
                ]
            )

            # テクニカルデータの一部を表示
            if result.get("technical_data"):
                tech_data = result["technical_data"]
                if "rsi" in tech_data:
                    report_lines.append(f"   RSI: {tech_data['rsi']:.1f}")
                if "price_change_1d" in tech_data:
                    change = tech_data["price_change_1d"]
                    report_lines.append(f"   1日変化率: {change:+.2f}%")

            report_lines.append("")

        return "\n".join(report_lines)

    def _save_html_report(self, report_dir: Path, timestamp: str):
        """HTMLレポートを保存"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>DayTrade自動化レポート</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 DayTrade自動化レポート</h1>
                <p>実行日時: {self.current_report.start_time.strftime("%Y年%m月%d日 %H:%M:%S")}</p>
            </div>

            <div class="section">
                <h2>📊 実行サマリー</h2>
                <p>対象銘柄数: {self.current_report.total_symbols}</p>
                <p>成功: {self.current_report.successful_symbols}</p>
                <p>失敗: {self.current_report.failed_symbols}</p>
                <p>シグナル数: {len(self.current_report.generated_signals)}</p>
                <p>アラート数: {len(self.current_report.triggered_alerts)}</p>
            </div>

            <div class="section">
                <h2>📈 生成シグナル</h2>
                <table>
                    <tr><th>銘柄</th><th>タイプ</th><th>理由</th><th>信頼度</th></tr>
        """

        for signal in self.current_report.generated_signals:
            html_content += f"""
                    <tr>
                        <td>{signal.get("symbol", "N/A")}</td>
                        <td>{signal.get("type", "N/A")}</td>
                        <td>{signal.get("reason", "N/A")}</td>
                        <td>{signal.get("confidence", "N/A")}</td>
                    </tr>
            """

        html_content += """
                </table>
            </div>
        </body>
        </html>
        """

        report_path = report_dir / f"daytrade_report_{timestamp}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def train_ml_models(
        self,
        symbols: Optional[List[str]] = None,
        training_period_months: int = 6,
        retrain_interval_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        機械学習モデルの訓練

        Args:
            symbols: 対象銘柄（未指定時は設定から取得）
            training_period_months: 訓練データ期間（月）
            retrain_interval_hours: 再訓練間隔（時間）

        Returns:
            訓練結果レポート
        """
        if not self.enhanced_ensemble:
            logger.error("強化アンサンブル戦略が利用できません")
            return {"success": False, "error": "Enhanced ensemble not available"}

        if symbols is None:
            symbols = self.config_manager.get_symbol_codes()

        logger.info(f"機械学習モデル訓練開始: {len(symbols)}銘柄")

        training_results = {
            "success": False,
            "trained_symbols": [],
            "failed_symbols": [],
            "models_trained": 0,
            "errors": [],
        }

        try:
            # 各銘柄の訓練データを準備・訓練
            for symbol in symbols:
                try:
                    logger.info(f"MLモデル訓練中: {symbol}")

                    # 訓練用履歴データ取得
                    training_data = self.stock_fetcher.get_historical_data(
                        symbol, period=f"{training_period_months}mo", interval="1d"
                    )

                    if training_data is None or len(training_data) < 60:
                        logger.warning(
                            f"訓練データが不足: {symbol} ({len(training_data) if training_data is not None else 0}日分)"
                        )
                        training_results["failed_symbols"].append(symbol)
                        continue

                    # 機械学習モデル訓練
                    success = self.enhanced_ensemble.train_ml_models(
                        symbol,
                        training_data,
                        retrain_interval_hours=retrain_interval_hours,
                    )

                    if success:
                        training_results["trained_symbols"].append(symbol)
                        training_results["models_trained"] += 1
                        logger.info(f"MLモデル訓練完了: {symbol}")
                    else:
                        training_results["failed_symbols"].append(symbol)
                        logger.warning(f"MLモデル訓練失敗: {symbol}")

                except Exception as e:
                    error_msg = f"MLモデル訓練エラー ({symbol}): {e}"
                    logger.error(error_msg)
                    training_results["errors"].append(error_msg)
                    training_results["failed_symbols"].append(symbol)

            # 結果サマリー
            total_symbols = len(symbols)
            successful_symbols = len(training_results["trained_symbols"])
            training_results["success"] = successful_symbols > 0

            logger.info(
                f"機械学習モデル訓練完了: {successful_symbols}/{total_symbols} 銘柄成功, "
                f"{training_results['models_trained']}個のモデル訓練"
            )

            return training_results

        except Exception as e:
            error_msg = f"機械学習モデル訓練全体エラー: {e}"
            logger.error(error_msg)
            training_results["errors"].append(error_msg)
            return training_results

    def run_advanced_backtest(
        self,
        symbols: Optional[List[str]] = None,
        strategy_signals: Optional[pd.DataFrame] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        高度なバックテスト実行

        Args:
            symbols: 対象銘柄
            strategy_signals: 戦略シグナル
            start_date: 開始日
            end_date: 終了日

        Returns:
            バックテスト結果
        """
        if not self.advanced_backtest_engine:
            logger.error("高度なバックテストエンジンが利用できません")
            return {"success": False, "error": "Advanced backtest engine not available"}

        if symbols is None:
            symbols = self.config_manager.get_symbol_codes()[
                :5
            ]  # テスト用に最初の5銘柄

        logger.info(f"高度なバックテスト開始: {len(symbols)}銘柄")

        try:
            backtest_results = {}

            for symbol in symbols:
                try:
                    # 履歴データ取得
                    historical_data = self.stock_fetcher.get_historical_data(
                        symbol, period="1y", interval="1d"
                    )

                    if historical_data is None or len(historical_data) < 100:
                        logger.warning(f"バックテスト用データが不足: {symbol}")
                        continue

                    # 戦略シグナル生成（提供されない場合）
                    if strategy_signals is None:
                        # 簡易なシグナル生成（移動平均クロス）
                        ma_short = historical_data["Close"].rolling(20).mean()
                        ma_long = historical_data["Close"].rolling(50).mean()

                        signals_df = pd.DataFrame(index=historical_data.index)
                        signals_df["signal"] = "hold"
                        signals_df["confidence"] = 50.0

                        buy_signals = (ma_short > ma_long) & (
                            ma_short.shift(1) <= ma_long.shift(1)
                        )
                        sell_signals = (ma_short < ma_long) & (
                            ma_short.shift(1) >= ma_long.shift(1)
                        )

                        signals_df.loc[buy_signals, "signal"] = "buy"
                        signals_df.loc[buy_signals, "confidence"] = 70.0
                        signals_df.loc[sell_signals, "signal"] = "sell"
                        signals_df.loc[sell_signals, "confidence"] = 70.0
                    else:
                        signals_df = strategy_signals

                    # バックテスト実行
                    performance = self.advanced_backtest_engine.run_backtest(
                        historical_data, signals_df, start_date, end_date
                    )

                    backtest_results[symbol] = {
                        "total_return": performance.total_return,
                        "annual_return": performance.annual_return,
                        "volatility": performance.volatility,
                        "sharpe_ratio": performance.sharpe_ratio,
                        "sortino_ratio": performance.sortino_ratio,
                        "max_drawdown": performance.max_drawdown,
                        "calmar_ratio": performance.calmar_ratio,
                        "win_rate": performance.win_rate,
                        "profit_factor": performance.profit_factor,
                        "total_trades": performance.total_trades,
                        "total_commission": performance.total_commission,
                        "total_slippage": performance.total_slippage,
                    }

                    logger.info(
                        f"バックテスト完了 ({symbol}): リターン {performance.total_return:.2%}, "
                        f"シャープレシオ {performance.sharpe_ratio:.2f}"
                    )

                except Exception as e:
                    logger.error(f"個別銘柄バックテストエラー ({symbol}): {e}")
                    continue

            # 全体結果サマリー
            if backtest_results:
                avg_return = sum(
                    r["total_return"] for r in backtest_results.values()
                ) / len(backtest_results)
                avg_sharpe = sum(
                    r["sharpe_ratio"] for r in backtest_results.values()
                ) / len(backtest_results)

                summary = {
                    "success": True,
                    "symbols_tested": len(backtest_results),
                    "average_return": avg_return,
                    "average_sharpe_ratio": avg_sharpe,
                    "individual_results": backtest_results,
                }

                logger.info(
                    f"高度なバックテスト完了: {len(backtest_results)}銘柄, "
                    f"平均リターン {avg_return:.2%}, 平均シャープレシオ {avg_sharpe:.2f}"
                )

                return summary
            else:
                return {"success": False, "error": "No successful backtests"}

        except Exception as e:
            error_msg = f"高度なバックテスト全体エラー: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}


# 使用例とテスト
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        # オーケストレーターのテスト実行
        orchestrator = DayTradeOrchestrator()

        # 小規模テスト（3銘柄）
        test_symbols = ["7203", "8306", "9984"]

        logger.info("=== DayTrade自動化テスト実行 ===")
        report = orchestrator.run_full_automation(symbols=test_symbols)

        logger.info(
            "Automation test completed",
            successful_symbols=report.successful_symbols,
            total_symbols=report.total_symbols,
            signals_count=len(report.generated_signals),
            alerts_count=len(report.triggered_alerts),
            errors_count=len(report.errors),
        )

        if report.errors:
            logger.error("Errors occurred during automation test", errors=report.errors)

        logger.info("テスト完了")

    except Exception as e:
        log_error_with_context(
            e, {"operation": "automation_test", "test_symbols": test_symbols}
        )
