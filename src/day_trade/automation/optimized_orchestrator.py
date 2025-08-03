"""
パフォーマンス最適化済み自動化オーケストレーター

並列処理、キャッシュ、メモリ最適化を適用した高速版オーケストレーター。
大量銘柄の同時処理とリアルタイム性能を向上。
"""

import gc
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import threading
from pathlib import Path

import pandas as pd
import numpy as np

from ..analysis.optimized_ml_models import OptimizedEnsemblePredictor, create_optimized_model_ensemble
from ..analysis.optimized_feature_engineering import OptimizedAdvancedFeatureEngineer, OptimizedDataQualityEnhancer
from ..analysis.ensemble import EnsembleStrategy, EnsembleTradingStrategy, EnsembleVotingType
from ..analysis.enhanced_ensemble import EnhancedEnsembleStrategy, PredictionHorizon
from ..analysis.indicators import TechnicalIndicators
from ..analysis.patterns import ChartPatternRecognizer
from ..analysis.signals import TradingSignalGenerator
from ..config.config_manager import ConfigManager
from ..core.alerts import AlertManager
from ..core.portfolio import PortfolioAnalyzer
from ..core.trade_manager import TradeManager
from ..core.watchlist import WatchlistManager
from ..data.stock_fetcher import StockFetcher
from ..utils.logging_config import get_context_logger, log_performance_metric, log_business_event
from ..utils.performance_analyzer import profile_performance, global_profiler
from ..utils.progress import ProgressType, multi_step_progress, progress_context

logger = get_context_logger(__name__)


@dataclass
class OptimizedExecutionResult:
    """最適化済み実行結果"""

    success: bool
    symbol: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    cache_hit: bool = False
    memory_usage_mb: float = 0.0

    # パフォーマンス統計
    processing_stages: Dict[str, float] = field(default_factory=dict)
    features_generated: int = 0
    signals_generated: int = 0


@dataclass
class OptimizedAutomationReport:
    """最適化済み自動化実行レポート"""

    start_time: datetime
    end_time: datetime
    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    execution_results: List[OptimizedExecutionResult]
    generated_signals: List[Dict[str, Any]]
    triggered_alerts: List[Dict[str, Any]]
    portfolio_summary: Dict[str, Any]
    errors: List[str]

    # パフォーマンス統計
    total_execution_time: float = 0.0
    avg_execution_time_per_symbol: float = 0.0
    cache_hit_rate: float = 0.0
    total_memory_usage_mb: float = 0.0
    parallel_efficiency: float = 0.0
    throughput_symbols_per_second: float = 0.0


class SymbolCache:
    """銘柄データキャッシュシステム"""

    def __init__(self, max_size: int = 100, ttl_minutes: int = 5):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.lock = threading.RLock()

    def get(self, symbol: str, data_type: str) -> Optional[pd.DataFrame]:
        """キャッシュからデータ取得"""
        with self.lock:
            key = f"{symbol}_{data_type}"

            if key in self.cache:
                cached_item = self.cache[key]

                # TTLチェック
                if datetime.now() - cached_item['timestamp'] < self.ttl:
                    self.access_times[key] = datetime.now()
                    return cached_item['data']
                else:
                    # 期限切れのため削除
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]

            return None

    def put(self, symbol: str, data_type: str, data: pd.DataFrame):
        """キャッシュにデータ保存"""
        with self.lock:
            key = f"{symbol}_{data_type}"

            # キャッシュサイズ制限
            if len(self.cache) >= self.max_size:
                # 最も古いアクセスのキーを削除
                oldest_key = min(self.access_times.keys(),
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            self.cache[key] = {
                'data': data.copy(),
                'timestamp': datetime.now()
            }
            self.access_times[key] = datetime.now()

    def clear(self):
        """キャッシュクリア"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        with self.lock:
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'ttl_minutes': self.ttl.total_seconds() / 60
            }


class OptimizedDayTradeOrchestrator:
    """パフォーマンス最適化済みデイトレード自動化オーケストレーター"""

    def __init__(self, config_path: Optional[str] = None, enable_optimizations: bool = True):
        """
        Args:
            config_path: 設定ファイルのパス
            enable_optimizations: 最適化機能の有効化フラグ
        """
        self.config_manager = ConfigManager(config_path)
        self.execution_settings = self.config_manager.get_execution_settings()
        self.enable_optimizations = enable_optimizations

        # パフォーマンス設定
        self.max_workers = min(8, (self.execution_settings.parallel_threads or 4))
        self.batch_size = 10  # 一度に処理する銘柄数
        self.memory_limit_mb = 2000  # メモリ使用量制限

        # キャッシュシステム
        self.symbol_cache = SymbolCache(max_size=200, ttl_minutes=5)
        self.feature_cache = {}

        # パフォーマンス統計
        self.performance_stats = {
            'symbols_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'memory_peaks': []
        }

        # 各コンポーネントの初期化
        self._initialize_components()

        # 実行状態
        self.current_report: Optional[OptimizedAutomationReport] = None
        self.is_running = False

        logger.info(
            "最適化済みオーケストレーター初期化完了",
            section="orchestrator_init",
            optimizations_enabled=enable_optimizations,
            max_workers=self.max_workers,
            batch_size=self.batch_size
        )

    def _initialize_components(self):
        """コンポーネント初期化（最適化済み）"""

        # 基本コンポーネント
        self.stock_fetcher = StockFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.pattern_recognizer = ChartPatternRecognizer()
        self.signal_generator = TradingSignalGenerator()

        # 最適化済みコンポーネント
        if self.enable_optimizations:
            self.optimized_feature_engineer = OptimizedAdvancedFeatureEngineer()
            self.optimized_data_enhancer = OptimizedDataQualityEnhancer()
            self.optimized_ml_ensemble = create_optimized_model_ensemble()

            logger.info("最適化済みコンポーネントを初期化")

        # アンサンブル戦略
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

            # 強化アンサンブル戦略（最適化版）
            self.enhanced_ensemble = EnhancedEnsembleStrategy(
                ensemble_strategy=strategy_type,
                voting_type=voting_type,
                enable_ml_models=self.enable_optimizations,
                performance_file=ensemble_settings.performance_file_path
            )
        else:
            self.ensemble_strategy = None
            self.enhanced_ensemble = None

        # その他のコンポーネント
        self.trade_manager = TradeManager()
        self.portfolio_analyzer = PortfolioAnalyzer(self.trade_manager, self.stock_fetcher)
        self.watchlist_manager = WatchlistManager()
        self.alert_manager = AlertManager(self.stock_fetcher, self.watchlist_manager)

        # スレッドプール
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max(2, self.max_workers // 2))

    @profile_performance
    def run_optimized_automation(
        self,
        symbols: Optional[List[str]] = None,
        enable_parallel: bool = True,
        enable_caching: bool = True,
        show_progress: bool = True
    ) -> OptimizedAutomationReport:
        """
        最適化済み全自動化処理を実行

        Args:
            symbols: 対象銘柄
            enable_parallel: 並列処理有効化
            enable_caching: キャッシュ有効化
            show_progress: 進捗表示
        """
        start_time = datetime.now()
        self.is_running = True

        logger.info(
            "最適化済み自動化処理開始",
            section="optimized_automation",
            symbols_count=len(symbols) if symbols else 0,
            parallel_enabled=enable_parallel,
            caching_enabled=enable_caching
        )

        try:
            # 対象銘柄の決定
            if symbols is None:
                symbols = self.config_manager.get_watchlist_symbols()

            # レポート初期化
            report = OptimizedAutomationReport(
                start_time=start_time,
                end_time=start_time,
                total_symbols=len(symbols),
                successful_symbols=0,
                failed_symbols=0,
                execution_results=[],
                generated_signals=[],
                triggered_alerts=[],
                portfolio_summary={},
                errors=[]
            )

            # バッチ処理実行
            if enable_parallel and len(symbols) > self.batch_size:
                execution_results = self._run_parallel_processing(symbols, enable_caching, show_progress)
            else:
                execution_results = self._run_sequential_processing(symbols, enable_caching, show_progress)

            # 結果統合
            report.execution_results = execution_results
            report.successful_symbols = sum(1 for r in execution_results if r.success)
            report.failed_symbols = len(execution_results) - report.successful_symbols

            # シグナル統合
            report.generated_signals = self._aggregate_signals(execution_results)

            # アラート処理
            report.triggered_alerts = self._process_alerts(execution_results)

            # ポートフォリオ分析
            report.portfolio_summary = self._analyze_portfolio_optimized()

            # パフォーマンス統計計算
            end_time = datetime.now()
            report.end_time = end_time
            report.total_execution_time = (end_time - start_time).total_seconds()

            if len(symbols) > 0:
                report.avg_execution_time_per_symbol = report.total_execution_time / len(symbols)
                report.throughput_symbols_per_second = len(symbols) / report.total_execution_time

            # キャッシュ統計
            total_requests = self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
            if total_requests > 0:
                report.cache_hit_rate = self.performance_stats['cache_hits'] / total_requests * 100

            # メモリ統計
            report.total_memory_usage_mb = sum(r.memory_usage_mb for r in execution_results)

            self.current_report = report

            logger.info(
                "最適化済み自動化処理完了",
                section="optimized_automation",
                execution_time=report.total_execution_time,
                successful_symbols=report.successful_symbols,
                failed_symbols=report.failed_symbols,
                cache_hit_rate=report.cache_hit_rate,
                throughput=report.throughput_symbols_per_second
            )

            return report

        except Exception as e:
            logger.error(
                "最適化済み自動化処理エラー",
                section="optimized_automation",
                error=str(e)
            )
            raise
        finally:
            self.is_running = False

    def _run_parallel_processing(
        self,
        symbols: List[str],
        enable_caching: bool,
        show_progress: bool
    ) -> List[OptimizedExecutionResult]:
        """並列処理実行"""

        execution_results = []

        # シンボルをバッチに分割
        batches = [symbols[i:i + self.batch_size]
                  for i in range(0, len(symbols), self.batch_size)]

        logger.info(
            f"並列処理開始: {len(batches)}バッチ",
            section="parallel_processing",
            total_symbols=len(symbols),
            batch_size=self.batch_size
        )

        # バッチ並列実行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_symbol_batch, batch, enable_caching): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    execution_results.extend(batch_results)

                    logger.debug(
                        f"バッチ処理完了: {len(batch)}銘柄",
                        section="parallel_processing",
                        successful=sum(1 for r in batch_results if r.success)
                    )

                except Exception as e:
                    logger.error(
                        f"バッチ処理エラー: {batch}",
                        section="parallel_processing",
                        error=str(e)
                    )

                    # エラーバッチの個別処理
                    for symbol in batch:
                        error_result = OptimizedExecutionResult(
                            success=False,
                            symbol=symbol,
                            error=str(e)
                        )
                        execution_results.append(error_result)

        return execution_results

    def _process_symbol_batch(
        self,
        symbols: List[str],
        enable_caching: bool
    ) -> List[OptimizedExecutionResult]:
        """銘柄バッチ処理"""

        batch_results = []

        for symbol in symbols:
            try:
                result = self._process_single_symbol_optimized(symbol, enable_caching)
                batch_results.append(result)

                # メモリ使用量監視
                self._monitor_memory_usage()

            except Exception as e:
                error_result = OptimizedExecutionResult(
                    success=False,
                    symbol=symbol,
                    error=str(e)
                )
                batch_results.append(error_result)

        return batch_results

    def _run_sequential_processing(
        self,
        symbols: List[str],
        enable_caching: bool,
        show_progress: bool
    ) -> List[OptimizedExecutionResult]:
        """逐次処理実行"""

        execution_results = []

        logger.info(
            "逐次処理開始",
            section="sequential_processing",
            total_symbols=len(symbols)
        )

        for i, symbol in enumerate(symbols):
            try:
                result = self._process_single_symbol_optimized(symbol, enable_caching)
                execution_results.append(result)

                if show_progress and (i + 1) % 10 == 0:
                    logger.info(
                        f"進捗: {i + 1}/{len(symbols)} 銘柄処理完了",
                        section="sequential_processing"
                    )

            except Exception as e:
                error_result = OptimizedExecutionResult(
                    success=False,
                    symbol=symbol,
                    error=str(e)
                )
                execution_results.append(error_result)

        return execution_results

    @profile_performance
    def _process_single_symbol_optimized(
        self,
        symbol: str,
        enable_caching: bool
    ) -> OptimizedExecutionResult:
        """単一銘柄の最適化処理"""

        start_time = time.time()
        start_memory = self._get_memory_usage()

        processing_stages = {}

        try:
            # 1. データ取得（キャッシュ対応）
            stage_start = time.time()
            data = self._get_symbol_data_cached(symbol, enable_caching)
            processing_stages['data_fetch'] = time.time() - stage_start

            if data is None or len(data) < 50:
                return OptimizedExecutionResult(
                    success=False,
                    symbol=symbol,
                    error="データ不足",
                    execution_time=time.time() - start_time
                )

            # 2. テクニカル指標計算（最適化版）
            stage_start = time.time()
            indicators = self._calculate_indicators_optimized(data, symbol, enable_caching)
            processing_stages['indicators'] = time.time() - stage_start

            # 3. 特徴量エンジニアリング（最適化版）
            features_generated = 0
            if self.enable_optimizations:
                stage_start = time.time()
                enhanced_data = self.optimized_data_enhancer.clean_ohlcv_data(data)
                feature_data = self.optimized_feature_engineer.generate_composite_features(
                    enhanced_data, indicators
                )
                features_generated = feature_data.shape[1] - data.shape[1]
                processing_stages['feature_engineering'] = time.time() - stage_start
            else:
                feature_data = data

            # 4. シグナル生成（アンサンブル）
            stage_start = time.time()
            signals = self._generate_signals_optimized(symbol, feature_data, indicators)
            processing_stages['signal_generation'] = time.time() - stage_start

            # 5. 機械学習予測（最適化版）
            ml_predictions = {}
            if self.enable_optimizations and len(feature_data) > 100:
                stage_start = time.time()
                ml_predictions = self._generate_ml_predictions_optimized(feature_data)
                processing_stages['ml_prediction'] = time.time() - stage_start

            # 結果データ構築
            result_data = {
                'data': data.tail(10).to_dict(),  # 最新10行のみ保存
                'indicators': {k: v.tail(5).to_dict() for k, v in indicators.items()},
                'signals': signals,
                'ml_predictions': ml_predictions,
                'features_count': features_generated
            }

            # パフォーマンス統計更新
            self.performance_stats['symbols_processed'] += 1

            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory

            return OptimizedExecutionResult(
                success=True,
                symbol=symbol,
                data=result_data,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                processing_stages=processing_stages,
                features_generated=features_generated,
                signals_generated=len(signals)
            )

        except Exception as e:
            logger.error(
                f"銘柄処理エラー: {symbol}",
                section="symbol_processing",
                error=str(e)
            )

            return OptimizedExecutionResult(
                success=False,
                symbol=symbol,
                error=str(e),
                execution_time=time.time() - start_time,
                processing_stages=processing_stages
            )

    def _get_symbol_data_cached(self, symbol: str, enable_caching: bool) -> Optional[pd.DataFrame]:
        """キャッシュ対応データ取得"""

        if enable_caching:
            cached_data = self.symbol_cache.get(symbol, 'stock_data')
            if cached_data is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_data
            else:
                self.performance_stats['cache_misses'] += 1

        # データフェッチ
        try:
            data = self.stock_fetcher.get_stock_data(symbol, period="3mo")

            if enable_caching and data is not None:
                self.symbol_cache.put(symbol, 'stock_data', data)

            return data

        except Exception as e:
            logger.warning(f"データ取得エラー: {symbol}", error=str(e))
            return None

    def _calculate_indicators_optimized(
        self,
        data: pd.DataFrame,
        symbol: str,
        enable_caching: bool
    ) -> Dict[str, pd.Series]:
        """最適化済み指標計算"""

        cache_key = f"{symbol}_indicators"

        if enable_caching and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # 高速指標計算
        indicators = {}

        try:
            # 基本指標（ベクトル化）
            indicators['sma_20'] = data['Close'].rolling(20).mean()
            indicators['sma_50'] = data['Close'].rolling(50).mean()
            indicators['ema_12'] = data['Close'].ewm(span=12).mean()
            indicators['ema_26'] = data['Close'].ewm(span=26).mean()

            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()

            # RSI（簡易版）
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))

            # ボリンジャーバンド
            bb_sma = data['Close'].rolling(20).mean()
            bb_std = data['Close'].rolling(20).std()
            indicators['bb_upper'] = bb_sma + (bb_std * 2)
            indicators['bb_lower'] = bb_sma - (bb_std * 2)

            if enable_caching:
                self.feature_cache[cache_key] = indicators

        except Exception as e:
            logger.warning(f"指標計算エラー: {symbol}", error=str(e))
            indicators = {
                'sma_20': pd.Series(dtype=float),
                'sma_50': pd.Series(dtype=float),
                'rsi': pd.Series(dtype=float),
                'macd': pd.Series(dtype=float)
            }

        return indicators

    def _generate_signals_optimized(
        self,
        symbol: str,
        data: pd.DataFrame,
        indicators: Dict[str, pd.Series]
    ) -> List[Dict[str, Any]]:
        """最適化済みシグナル生成"""

        signals = []

        try:
            # 強化アンサンブルシグナル
            if self.enhanced_ensemble:
                enhanced_signal = self.enhanced_ensemble.generate_enhanced_signal(
                    data, indicators, prediction_horizon=PredictionHorizon.SHORT_TERM
                )

                if enhanced_signal:
                    signals.append({
                        'type': 'enhanced_ensemble',
                        'signal': enhanced_signal.signal_type.value,
                        'confidence': enhanced_signal.ensemble_confidence,
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'risk_score': enhanced_signal.risk_score,
                        'uncertainty': enhanced_signal.uncertainty
                    })

            # 従来のアンサンブルシグナル（補完用）
            if self.ensemble_strategy:
                traditional_signal = self.ensemble_strategy.generate_signal(data, indicators)

                if traditional_signal:
                    signals.append({
                        'type': 'traditional_ensemble',
                        'signal': traditional_signal.signal_type.value,
                        'confidence': traditional_signal.confidence,
                        'timestamp': datetime.now(),
                        'symbol': symbol
                    })

        except Exception as e:
            logger.warning(f"シグナル生成エラー: {symbol}", error=str(e))

        return signals

    def _generate_ml_predictions_optimized(self, feature_data: pd.DataFrame) -> Dict[str, Any]:
        """最適化済みML予測"""

        ml_predictions = {}

        try:
            if self.optimized_ml_ensemble and len(feature_data) > 50:
                # 特徴量準備
                feature_cols = [col for col in feature_data.columns
                              if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

                if len(feature_cols) > 5:
                    X = feature_data[feature_cols].fillna(0).tail(1)
                    predictions = self.optimized_ml_ensemble.predict(X)

                    if predictions:
                        ml_predictions = {
                            'prediction': predictions[0].prediction,
                            'confidence': predictions[0].confidence,
                            'model_name': predictions[0].model_name,
                            'cache_hit': predictions[0].cache_hit
                        }

        except Exception as e:
            logger.debug(f"ML予測エラー", error=str(e))

        return ml_predictions

    def _aggregate_signals(self, execution_results: List[OptimizedExecutionResult]) -> List[Dict[str, Any]]:
        """シグナル統合"""
        all_signals = []

        for result in execution_results:
            if result.success and result.data and 'signals' in result.data:
                all_signals.extend(result.data['signals'])

        return all_signals

    def _process_alerts(self, execution_results: List[OptimizedExecutionResult]) -> List[Dict[str, Any]]:
        """アラート処理（軽量版）"""
        alerts = []

        # 高信頼度シグナルのみアラート対象
        for result in execution_results:
            if result.success and result.data and 'signals' in result.data:
                for signal in result.data['signals']:
                    if signal.get('confidence', 0) > 80:
                        alerts.append({
                            'symbol': result.symbol,
                            'signal_type': signal.get('signal'),
                            'confidence': signal.get('confidence'),
                            'timestamp': signal.get('timestamp'),
                            'alert_level': 'high'
                        })

        return alerts

    def _analyze_portfolio_optimized(self) -> Dict[str, Any]:
        """最適化済みポートフォリオ分析"""
        try:
            return self.portfolio_analyzer.get_current_analysis()
        except Exception as e:
            logger.warning("ポートフォリオ分析エラー", error=str(e))
            return {'error': str(e)}

    def _monitor_memory_usage(self):
        """メモリ使用量監視"""
        current_memory = self._get_memory_usage()
        self.performance_stats['memory_peaks'].append(current_memory)

        # メモリ制限チェック
        if current_memory > self.memory_limit_mb:
            logger.warning(
                "メモリ使用量が制限を超過",
                section="memory_monitoring",
                current_mb=current_memory,
                limit_mb=self.memory_limit_mb
            )

            # メモリクリーンアップ
            self._cleanup_memory()

    def _get_memory_usage(self) -> float:
        """現在のメモリ使用量取得（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def _cleanup_memory(self):
        """メモリクリーンアップ"""
        # キャッシュクリア
        if len(self.feature_cache) > 50:
            self.feature_cache.clear()

        # ガベージコレクション
        gc.collect()

        logger.debug("メモリクリーンアップ実行", section="memory_management")

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス統計サマリー"""
        return {
            'symbols_processed': self.performance_stats['symbols_processed'],
            'cache_stats': {
                'hits': self.performance_stats['cache_hits'],
                'misses': self.performance_stats['cache_misses'],
                'hit_rate': (self.performance_stats['cache_hits'] /
                           max(1, self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) * 100)
            },
            'memory_stats': {
                'current_usage_mb': self._get_memory_usage(),
                'peak_usage_mb': max(self.performance_stats['memory_peaks']) if self.performance_stats['memory_peaks'] else 0,
                'avg_usage_mb': np.mean(self.performance_stats['memory_peaks']) if self.performance_stats['memory_peaks'] else 0
            },
            'cache_system_stats': self.symbol_cache.get_stats(),
            'profiler_summary': global_profiler.get_metrics_summary()
        }

    def __del__(self):
        """リソースクリーンアップ"""
        if hasattr(self, 'thread_executor'):
            self.thread_executor.shutdown(wait=False)
        if hasattr(self, 'process_executor'):
            self.process_executor.shutdown(wait=False)


# 使用例とデモ
if __name__ == "__main__":
    logger.info("最適化済みオーケストレーターデモ開始", section="demo")

    try:
        # 最適化済みオーケストレーター作成
        orchestrator = OptimizedDayTradeOrchestrator(enable_optimizations=True)

        # テスト銘柄
        test_symbols = ["7203", "8306", "9984", "6758", "8035"]

        # 最適化済み自動化実行
        report = orchestrator.run_optimized_automation(
            symbols=test_symbols,
            enable_parallel=True,
            enable_caching=True,
            show_progress=True
        )

        # パフォーマンス統計
        perf_summary = orchestrator.get_performance_summary()

        logger.info(
            "最適化済みオーケストレーターデモ完了",
            section="demo",
            execution_time=report.total_execution_time,
            throughput=report.throughput_symbols_per_second,
            cache_hit_rate=report.cache_hit_rate,
            performance_summary=perf_summary
        )

    except Exception as e:
        logger.error(f"デモ実行エラー: {e}", section="demo")

    finally:
        # リソースクリーンアップ
        if 'orchestrator' in locals():
            del orchestrator
        gc.collect()
