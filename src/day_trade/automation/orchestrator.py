"""
Next-Gen AI Trading Engine Orchestrator
高度AI駆動市場分析・統合オーケストレーション

完全セーフモード - 自動取引機能は一切含まれていません
"""

import asyncio

# 重いML系インポートは遅延（CI環境でメモリ削減）
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# プロジェクト内モジュール
from ..automation.analysis_only_engine import AnalysisOnlyEngine
from ..config.trading_mode_config import get_current_trading_config, is_safe_mode
from ..core.portfolio import PortfolioManager
from ..data.stock_fetcher import StockFetcher
from ..utils.fault_tolerance import FaultTolerantExecutor
from ..utils.logging_config import get_context_logger
from ..utils.performance_monitor import PerformanceMonitor
from ..models.database import get_default_database_manager

CI_MODE = os.getenv("CI", "false").lower() == "true"

if not CI_MODE:
    from ..data.advanced_ml_engine import (
        AdvancedMLEngine,
        ModelConfig,
        create_advanced_ml_engine,
    )
    from ..data.batch_data_fetcher import (
        AdvancedBatchDataFetcher,
        DataRequest,
        DataResponse,
    )
else:
    # CI環境では軽量ダミークラス使用
    AdvancedMLEngine = None
    ModelConfig = None
    create_advanced_ml_engine = None
    AdvancedBatchDataFetcher = None
    DataRequest = None
    DataResponse = None

# 並列処理システム
try:
    from ..utils.parallel_executor_manager import (
        ExecutionResult,
        ParallelExecutorManager,
        TaskType,
    )

    PARALLEL_EXECUTOR_AVAILABLE = True
except ImportError:
    # フォールバック用レガシー並列処理
    from concurrent.futures import ThreadPoolExecutor, as_completed

    PARALLEL_EXECUTOR_AVAILABLE = False
    CONCURRENT_AVAILABLE = True

logger = get_context_logger(__name__)


@dataclass
class AIAnalysisResult:
    """AI分析結果"""

    symbol: str
    timestamp: datetime
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    technical_signals: Dict[str, Any]
    ml_features: Dict[str, Any]
    performance_metrics: Dict[str, float]
    data_quality: float
    recommendation: str
    risk_assessment: Dict[str, Any]


@dataclass
class ExecutionReport:
    """実行レポート（拡張版）"""

    start_time: datetime
    end_time: datetime
    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    generated_signals: List[Dict[str, Any]]
    triggered_alerts: List[Dict[str, Any]]
    ai_analysis_results: List[AIAnalysisResult] = None
    portfolio_summary: Optional[Dict[str, Any]] = None
    performance_stats: Dict[str, Any] = None
    system_health: Dict[str, Any] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.ai_analysis_results is None:
            self.ai_analysis_results = []


@dataclass
class OrchestrationConfig:
    """オーケストレーション設定"""

    max_workers: int = 8
    max_thread_workers: int = 12  # I/Oバウンド用
    max_process_workers: int = 4  # CPUバウンド用
    enable_ml_engine: bool = True
    enable_advanced_batch: bool = True
    enable_performance_monitoring: bool = True
    enable_fault_tolerance: bool = True
    enable_parallel_optimization: bool = True  # 新並列システム
    prediction_horizon: int = 5  # 予測期間（日）
    confidence_threshold: float = 0.7
    data_quality_threshold: float = 80.0
    timeout_seconds: int = 300
    cache_enabled: bool = True
    retry_attempts: int = 3


class NextGenAIOrchestrator:
    """
    次世代AI取引エンジン オーケストレーター

    【重要】完全セーフモード - 自動取引機能は一切含まれていません

    高度機能：
    1. LSTM-Transformer ハイブリッドモデル統合
    2. 大規模並列AI分析パイプライン
    3. リアルタイムデータ処理・品質管理
    4. 高度リスク評価・ポートフォリオ最適化
    5. 包括的システム監視・フォールトトレラント

    ※ 実際の取引実行は一切行いません（分析・教育目的のみ）
    """

    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        ml_config: Optional[ModelConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        初期化

        Args:
            config: オーケストレーション設定
            ml_config: MLモデル設定
            config_path: 設定ファイルパス（オプション）
        """
        # セーフモードチェック
        if not is_safe_mode():
            raise ValueError(
                "セーフモードでない場合は、このオーケストレーターは使用できません"
            )

        self.config = config or OrchestrationConfig()

        # CI環境では軽量化
        if CI_MODE:
            self.config.enable_ml_engine = False
            self.config.enable_advanced_batch = False
            self.config.enable_realtime_predictions = False
            self.config.batch_size = min(self.config.batch_size, 10)
            self.ml_config = None
            logger.info("CI軽量モード: ML機能を無効化")
        else:
            self.ml_config = ml_config or ModelConfig()

        self.config_path = config_path
        self.trading_config = get_current_trading_config()

        # コアコンポーネント初期化
        self.stock_fetcher = StockFetcher()
        self.analysis_engines: Dict[str, AnalysisOnlyEngine] = {}
        self.db_manager = get_default_database_manager()  # DatabaseManagerを取得

        # 高度AIコンポーネント初期化（CI環境では無効化）
        if self.config.enable_ml_engine and not CI_MODE:
            self.ml_engine = create_advanced_ml_engine(asdict(self.ml_config))
        else:
            self.ml_engine = None

        if self.config.enable_advanced_batch and not CI_MODE:
            self.batch_fetcher = AdvancedBatchDataFetcher(
                max_workers=self.config.max_workers,
                enable_kafka=False,  # セーフモードではKafka無効
                enable_redis=False,  # セーフモードではRedis無効
            )
        else:
            self.batch_fetcher = None

        # パフォーマンス監視
        if self.config.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None

        # フォールトトレラント実行
        if self.config.enable_fault_tolerance:
            self.fault_executor = FaultTolerantExecutor(
                max_retries=self.config.retry_attempts,
                timeout_seconds=self.config.timeout_seconds,
            )
        else:
            self.fault_executor = None

        # 並列実行マネージャー (Issue #383)
        if self.config.enable_parallel_optimization and PARALLEL_EXECUTOR_AVAILABLE:
            self.parallel_manager = ParallelExecutorManager(
                max_thread_workers=self.config.max_thread_workers,
                max_process_workers=self.config.max_process_workers,
                enable_adaptive_sizing=True,
                performance_monitoring=self.config.enable_performance_monitoring,
            )
            logger.info(
                f"並列実行最適化有効: Thread={self.config.max_thread_workers}, "
                f"Process={self.config.max_process_workers}"
            )
        else:
            self.parallel_manager = None
            if not PARALLEL_EXECUTOR_AVAILABLE:
                logger.warning(
                    "ParallelExecutorManagerが利用できません。レガシー並列処理を使用"
                )

        # 実行統計
        self.execution_history = []
        self.performance_metrics = {}

        logger.info("Next-Gen AI Orchestrator 初期化完了 - 完全セーフモード")
        logger.info("※ 自動取引機能は一切含まれていません")
        logger.info(
            f"設定: ML={self.config.enable_ml_engine}, "
            f"Batch={self.config.enable_advanced_batch}"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parallel_manager:
            self.parallel_manager.shutdown()

    def _execute_parallel_analysis(
        self,
        symbols: List[str],
        analysis_functions: List[Tuple[Callable, Dict[str, Any]]],
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, List[ExecutionResult]]:
        """
        並列分析実行 (Issue #383対応)

        CPU/I/Oバウンドタスクを適切に分離して効率的に並列実行

        Args:
            symbols: 分析対象銘柄
            analysis_functions: (関数, 引数辞書) のリスト
            max_concurrent: 最大同時実行数

        Returns:
            シンボル別実行結果辞書
        """
        if not self.parallel_manager:
            logger.warning(
                "並列マネージャーが無効です。シーケンシャル実行にフォールバック"
            )
            return self._execute_sequential_fallback(symbols, analysis_functions)

        results = {}
        all_tasks = []

        # 各銘柄×各分析関数の組み合わせでタスクを生成
        for symbol in symbols:
            symbol_tasks = []

            for analysis_func, kwargs in analysis_functions:
                # シンボル固有の引数を設定
                task_kwargs = kwargs.copy()
                task_kwargs["symbol"] = symbol

                # タスクタイプをヒント
                if (
                    "fetch" in analysis_func.__name__
                    or "download" in analysis_func.__name__
                ):
                    task_type = TaskType.IO_BOUND
                elif (
                    "compute" in analysis_func.__name__
                    or "calculate" in analysis_func.__name__
                ):
                    task_type = TaskType.CPU_BOUND
                else:
                    task_type = TaskType.MIXED

                task = (analysis_func, (), task_kwargs)
                all_tasks.append((symbol, analysis_func.__name__, task))
                symbol_tasks.append(task)

            results[symbol] = []

        # バッチ実行
        batch_tasks = [task for _, _, task in all_tasks]
        execution_results = self.parallel_manager.execute_batch(
            batch_tasks, max_concurrent=max_concurrent or self.config.max_workers
        )

        # 結果を銘柄別に整理
        for (symbol, func_name, _), exec_result in zip(all_tasks, execution_results):
            results[symbol].append(exec_result)

        # 統計情報をログ出力
        successful_tasks = sum(1 for r in execution_results if r.success)
        total_tasks = len(execution_results)

        logger.info(f"並列分析完了: {successful_tasks}/{total_tasks} 成功")
        if self.parallel_manager:
            perf_stats = self.parallel_manager.get_performance_stats()
            for executor_name, stats in perf_stats.items():
                logger.info(
                    f"{executor_name}: 平均時間={stats['average_time_ms']:.1f}ms, "
                    f"成功率={stats['success_rate']:.1%}"
                )

        return results

    def _execute_sequential_fallback(
        self,
        symbols: List[str],
        analysis_functions: List[Tuple[Callable, Dict[str, Any]]],
    ) -> Dict[str, List[ExecutionResult]]:
        """
        シーケンシャル実行フォールバック"""
        results = {}

        for symbol in symbols:
            symbol_results = []

            for analysis_func, kwargs in analysis_functions:
                task_kwargs = kwargs.copy()
                task_kwargs["symbol"] = symbol

                start_time = time.perf_counter()
                try:
                    result = analysis_func(**task_kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = e
                    logger.error(f"Sequential execution failed for {symbol}: {e}")

                execution_time = (time.perf_counter() - start_time) * 1000

                exec_result = ExecutionResult(
                    task_id=f"{symbol}_{analysis_func.__name__}",
                    result=result,
                    execution_time_ms=execution_time,
                    # executor_type=ExecutorType.THREAD_POOL,  # フォールバック
                    success=success,
                    error=error,
                )

                symbol_results.append(exec_result)

            results[symbol] = symbol_results

        return results

    def run_advanced_analysis(
        self,
        symbols: Optional[List[str]] = None,
        analysis_type: str = "comprehensive",
        include_predictions: bool = True,
    ) -> ExecutionReport:
        """
        高度AI分析実行

        Args:
            symbols: 分析対象銘柄リスト
            analysis_type: 分析タイプ ("basic", "comprehensive", "ml_focus")
            include_predictions: 予測分析を含むか

        Returns:
            ExecutionReport: 詳細実行結果レポート
        """
        start_time = datetime.now()

        logger.info(f"Next-Gen AI分析開始 - タイプ: {analysis_type}")
        logger.info(f"対象銘柄: {len(symbols) if symbols else 0}")

        if not symbols:
            symbols = ["7203", "8306", "9984", "6758", "4689"]  # デフォルト銘柄

        # 分析実行
        generated_signals = []
        triggered_alerts = []
        ai_analysis_results = []
        successful_symbols = 0
        failed_symbols = 0
        errors = []
        actual_portfolio_summary = None

        try:
            # ポートフォリオ情報取得
            try:
                from ..database.database import get_default_database_manager

                db_manager = get_default_database_manager()
                with db_manager.session_scope() as session:
                    portfolio_manager = PortfolioManager(session)
                    actual_portfolio_summary = portfolio_manager.get_portfolio_summary()
            except ImportError:
                logger.warning(
                    "Database manager not available, skipping portfolio summary"
                )
                actual_portfolio_summary = None

            # 高度バッチデータ取得
            if self.batch_fetcher:
                batch_results = self._execute_batch_data_collection(symbols)
            else:
                batch_results = {}

            # 分析エンジン初期化
            for symbol in symbols:
                try:
                    # 分析エンジン作成
                    if symbol not in self.analysis_engines:
                        self.analysis_engines[symbol] = AnalysisOnlyEngine([symbol])
                except Exception as e:
                    logger.warning(
                        f"Failed to create analysis engine for {symbol}: {e}"
                    )

            # 並列AI分析実行
            if CONCURRENT_AVAILABLE and len(symbols) > 1:
                results = self._execute_parallel_ai_analysis(
                    symbols, batch_data, analysis_type, include_predictions
                )
            else:
                results = self._execute_sequential_ai_analysis(
                    symbols, batch_data, analysis_type, include_predictions
                )

            # 結果集計
            for symbol, result in results.items():
                if result["success"]:
                    successful_symbols += 1
                    ai_analysis_results.append(result["analysis"])
                    generated_signals.extend(result["signals"])
                    triggered_alerts.extend(result["alerts"])
                else:
                    failed_symbols += 1
                    errors.extend(result["errors"])

            # ポートフォリオ分析統合
            if actual_portfolio_summary:
                portfolio_summary = actual_portfolio_summary
            else:
                portfolio_summary = self._generate_portfolio_analysis(
                    ai_analysis_results
                )

            # システムヘルス分析
            system_health = self._analyze_system_health()

            # パフォーマンス統計
            performance_stats = self._calculate_performance_stats(start_time)

        except Exception as e:
            error_msg = f"高度AI分析実行エラー: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)

        end_time = datetime.now()

        # 実行レポート作成
        report = ExecutionReport(
            start_time=start_time,
            end_time=end_time,
            total_symbols=len(symbols),
            successful_symbols=successful_symbols,
            failed_symbols=failed_symbols,
            generated_signals=generated_signals,
            triggered_alerts=triggered_alerts,
            ai_analysis_results=ai_analysis_results,
            portfolio_summary=portfolio_summary,
            performance_stats=performance_stats,
            system_health=system_health,
            errors=errors,
        )

        # 実行履歴に保存
        self.execution_history.append(report)
        self.execution_history = self.execution_history[-50:]  # 最新50件のみ保持

        logger.info(
            f"Next-Gen AI分析完了 - 成功: {successful_symbols}, 失敗: {failed_symbols}"
        )
        return report

    def _execute_batch_data_collection(
        self, symbols: List[str]
    ) -> Dict[str, DataResponse]:
        """
        高度バッチデータ収集"""

        logger.info(f"バッチデータ収集開始: {len(symbols)} 銘柄")

        try:
            # データリクエスト作成
            requests = [
                DataRequest(
                    symbol=symbol,
                    period="1y",  # より長期間のデータ
                    preprocessing=True,
                    features=[
                        "trend_strength",
                        "momentum",
                        "price_channel",
                        "gap_analysis",
                    ],
                    priority=5 if symbol in ["7203", "8306"] else 3,
                    cache_ttl=3600,
                )
                for symbol in symbols
            ]

            # バッチ実行
            return self.batch_fetcher.fetch_batch(requests, use_parallel=True)

        except Exception as e:
            logger.error(f"バッチデータ収集エラー: {e}")
            return {}

    def _execute_parallel_ai_analysis(
        self,
        symbols: List[str],
        batch_data: Dict[str, DataResponse],
        analysis_type: str,
        include_predictions: bool,
    ) -> Dict[str, Dict]:
        """
        並列AI分析実行"""

        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 分析タスク投入
            future_to_symbol = {
                executor.submit(
                    self._analyze_single_symbol,
                    symbol,
                    batch_data.get(symbol),
                    analysis_type,
                    include_predictions,
                ): symbol
                for symbol in symbols
            }

            # 結果収集
            for future in as_completed(
                future_to_symbol, timeout=self.config.timeout_seconds
            ):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=60)
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"並列AI分析エラー {symbol}: {e}")
                    results[symbol] = {
                        "success": False,
                        "errors": [str(e)],
                        "analysis": None,
                        "signals": [],
                        "alerts": [],
                    }

        return results

    def _execute_sequential_ai_analysis(
        self,
        symbols: List[str],
        batch_data: Dict[str, DataResponse],
        analysis_type: str,
        include_predictions: bool,
    ) -> Dict[str, Dict]:
        """
        逐次AI分析実行"""

        results = {}

        for symbol in symbols:
            try:
                result = self._analyze_single_symbol(
                    symbol, batch_data.get(symbol), analysis_type, include_predictions
                )
                results[symbol] = result
            except Exception as e:
                logger.error(f"逐次AI分析エラー {symbol}: {e}")
                results[symbol] = {
                    "success": False,
                    "errors": [str(e)],
                    "analysis": None,
                    "signals": [],
                    "alerts": [],
                }

        return results

    def _analyze_single_symbol(
        self,
        symbol: str,
        data_response: Optional[DataResponse],
        analysis_type: str,
        include_predictions: bool,
    ) -> Dict:
        """
        単一銘柄AI分析"""

        start_time = time.time()

        try:
            # データ品質チェック
            if not data_response or not data_response.success:
                return {
                    "success": False,
                    "errors": [f"{symbol}: データ取得失敗"],
                    "analysis": None,
                    "signals": [],
                    "alerts": [],
                }

            market_data = data_response.data
            data_quality = data_response.data_quality_score

            # データ品質閾値チェック
            if data_quality < self.config.data_quality_threshold:
                logger.warning(f"{symbol}: データ品質不足 ({data_quality:.1f})")

            # 基本分析エンジン実行
            if symbol not in self.analysis_engines:
                self.analysis_engines[symbol] = AnalysisOnlyEngine([symbol])

            engine = self.analysis_engines[symbol]
            # basic_status = engine.get_status()
            # market_summary = engine.get_market_summary()

            # 高度AI分析実行
            ai_predictions = {}
            confidence_scores = {}

            if include_predictions and self.ml_engine and len(market_data) > 100:
                try:
                    # ML予測実行
                    X_sequences, y_sequences = self.ml_engine.prepare_data(market_data)

                    if len(X_sequences) > 0:
                        # 最新データで予測
                        latest_sequence = X_sequences[-1:]
                        prediction_result = self.ml_engine.predict(latest_sequence)

                        ai_predictions = {
                            "price_direction": "up"
                            if prediction_result.predictions[0] > 0
                            else "down",
                            "predicted_change": float(prediction_result.predictions[0]),
                            "confidence": float(prediction_result.confidence[0])
                            if prediction_result.confidence is not None
                            else 0.5,
                        }

                        confidence_scores = {
                            "ml_model": float(prediction_result.confidence[0])
                            if prediction_result.confidence is not None
                            else 0.5,
                            "data_quality": data_quality / 100.0,
                            "overall": (
                                float(prediction_result.confidence[0])
                                if prediction_result.confidence is not None
                                else 0.5
                            )
                            * (data_quality / 100.0),
                        }

                except Exception as e:
                    logger.warning(f"{symbol}: ML予測エラー - {e}")
                    ai_predictions = {"error": str(e)}
                    confidence_scores = {"overall": 0.3}

            # テクニカル分析シグナル
            technical_signals = self._generate_technical_signals(market_data, symbol)

            # ML特徴量サマリー
            ml_features = self._extract_ml_features_summary(market_data)

            # パフォーマンス指標
            performance_metrics = {
                "analysis_time": time.time() - start_time,
                "data_points": len(market_data),
                "feature_count": len(market_data.columns),
                "memory_usage": self._estimate_memory_usage(market_data),
            }

            # リスク評価
            risk_assessment = self._calculate_risk_assessment(
                market_data, ai_predictions, confidence_scores
            )

            # 推奨アクション生成
            recommendation = self._generate_recommendation(
                ai_predictions, confidence_scores, technical_signals, risk_assessment
            )

            # AI分析結果作成
            ai_analysis = AIAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                predictions=ai_predictions,
                confidence_scores=confidence_scores,
                technical_signals=technical_signals,
                ml_features=ml_features,
                performance_metrics=performance_metrics,
                data_quality=data_quality,
                recommendation=recommendation,
                risk_assessment=risk_assessment,
            )

            # シグナル生成
            signals = self._generate_ai_signals(ai_analysis)

            # アラート生成
            alerts = self._generate_smart_alerts(ai_analysis)

            return {
                "success": True,
                "errors": [],
                "analysis": ai_analysis,
                "signals": signals,
                "alerts": alerts,
            }

        except Exception as e:
            return {
                "success": False,
                "errors": [f"{symbol}: 分析エラー - {str(e)}"],
                "analysis": None,
                "signals": [],
                "alerts": [],
            }

    def _generate_technical_signals(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """
        テクニカル分析シグナル生成"""

        signals = {}

        try:
            if "終値" in data.columns and len(data) >= 50:
                current_price = data["終値"].iloc[-1]

                # 移動平均シグナル
                sma_20 = data["終値"].rolling(20).mean().iloc[-1]
                sma_50 = data["終値"].rolling(50).mean().iloc[-1]

                signals["moving_average"] = {
                    "sma_20_signal": "bullish" if current_price > sma_20 else "bearish",
                    "sma_50_signal": "bullish" if current_price > sma_50 else "bearish",
                    "golden_cross": sma_20 > sma_50,
                    "death_cross": sma_20 < sma_50,
                }

                # RSIシグナル
                if "RSI_14" in data.columns:
                    rsi = data["RSI_14"].iloc[-1]
                    signals["rsi"] = {
                        "value": rsi,
                        "signal": "oversold"
                        if rsi < 30
                        else "overbought"
                        if rsi > 70
                        else "neutral",
                    }

                # ボラティリティシグナル
                if "volatility_20d" in data.columns:
                    volatility = data["volatility_20d"].iloc[-1]
                    vol_percentile = data["volatility_20d"].rank(pct=True).iloc[-1]

                    signals["volatility"] = {
                        "current": volatility,
                        "percentile": vol_percentile,
                        "regime": "high"
                        if vol_percentile > 0.8
                        else "low"
                        if vol_percentile < 0.2
                        else "normal",
                    }

        except Exception as e:
            logger.error(f"テクニカルシグナル生成エラー {symbol}: {e}")
            signals = {"error": str(e)}

        return signals

    def _extract_ml_features_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        ML特徴量サマリー抽出"""

        features = {}

        try:
            # 基本統計
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) > 0:
                features["basic_stats"] = {
                    "feature_count": len(numeric_columns),
                    "data_completeness": 1.0
                    - data[numeric_columns].isnull().sum().sum()
                    / (len(data) * len(numeric_columns)),
                    "value_ranges": {
                        col: {
                            "min": float(data[col].min()),
                            "max": float(data[col].max()),
                        }
                        for col in numeric_columns[:5]  # 最初の5列のみ
                    },
                }

            # 時系列特性
            if "終値" in data.columns:
                returns = data["終値"].pct_change()

                features["time_series"] = {
                    "trend": "upward"
                    if data["終値"].iloc[-1] > data["終値"].iloc[0]
                    else "downward",
                    "volatility": float(returns.std()),
                    "sharpe_estimate": float(returns.mean() / returns.std())
                    if returns.std() > 0
                    else 0,
                    "max_drawdown": float(
                        (data["終値"] / data["終値"].expanding().max() - 1).min()
                    ),
                }

        except Exception as e:
            logger.error(f"ML特徴量サマリーエラー: {e}")
            features = {"error": str(e)}

        return features

    def _calculate_risk_assessment(
        self,
        data: pd.DataFrame,
        predictions: Dict[str, Any],
        confidence_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        リスク評価計算"""

        risk_assessment = {}

        try:
            # データ品質リスク
            data_completeness = 1.0 - data.isnull().sum().sum() / (
                len(data) * len(data.columns)
            )

            # 予測不確実性リスク
            prediction_risk = 1.0 - confidence_scores.get("overall", 0.5)

            # ボラティリティリスク
            if "終値" in data.columns:
                returns = data["終値"].pct_change()
                volatility_risk = min(returns.std() * 10, 1.0)  # 正規化
            else:
                volatility_risk = 0.5

            # 流動性リスク（出来高ベース）
            if "出来高" in data.columns:
                volume_trend = data["出来高"].rolling(20).mean().pct_change().iloc[-1]
                liquidity_risk = max(0, -volume_trend)  # 出来高減少時にリスク増
            else:
                liquidity_risk = 0.3

            # 総合リスクスコア
            overall_risk = np.mean(
                [
                    data_completeness * 0.2,
                    prediction_risk * 0.3,
                    volatility_risk * 0.3,
                    liquidity_risk * 0.2,
                ]
            )

            risk_assessment = {
                "data_quality_risk": 1.0 - data_completeness,
                "prediction_uncertainty": prediction_risk,
                "volatility_risk": volatility_risk,
                "liquidity_risk": liquidity_risk,
                "overall_risk_score": overall_risk,
                "risk_level": "high"
                if overall_risk > 0.7
                else "medium"
                if overall_risk > 0.4
                else "low",
            }

        except Exception as e:
            logger.error(f"リスク評価エラー: {e}")
            risk_assessment = {
                "overall_risk_score": 0.5,
                "risk_level": "unknown",
                "error": str(e),
            }

        return risk_assessment

    def _generate_recommendation(
        self,
        predictions: Dict[str, Any],
        confidence_scores: Dict[str, float],
        technical_signals: Dict[str, Any],
        risk_assessment: Dict[str, Any],
    ) -> str:
        """推奨アクション生成"""

        try:
            overall_confidence = confidence_scores.get("overall", 0.5)
            overall_risk = risk_assessment.get("overall_risk_score", 0.5)

            # 信頼度とリスクに基づく推奨
            if (
                overall_confidence > self.config.confidence_threshold
                and overall_risk < 0.4
            ):
                if predictions.get("predicted_change", 0) > 0.02:  # 2%以上の上昇予測
                    return "STRONG_BUY_SIGNAL"
                elif predictions.get("predicted_change", 0) < -0.02:  # 2%以上の下落予測
                    return "STRONG_SELL_SIGNAL"
                else:
                    return "HOLD"
            elif overall_confidence > 0.5 and overall_risk < 0.6:
                if predictions.get("predicted_change", 0) > 0:
                    return "WEAK_BUY_SIGNAL"
                else:
                    return "WEAK_SELL_SIGNAL"
            else:
                return "INSUFFICIENT_CONFIDENCE"

        except Exception as e:
            logger.error(f"推奨生成エラー: {e}")
            return "ANALYSIS_ERROR"

    def _generate_ai_signals(self, analysis: AIAnalysisResult) -> List[Dict[str, Any]]:
        """
        AIシグナル生成"""

        signals = {}

        try:
            # 基本シグナル
            base_signal = {
                "symbol": analysis.symbol,
                "type": "AI_ANALYSIS",
                "timestamp": analysis.timestamp.isoformat(),
                "source": "next_gen_ai_engine",
                "confidence": analysis.confidence_scores.get("overall", 0.5),
                "recommendation": analysis.recommendation,
                "safe_mode": True,
                "trading_disabled": True,
            }

            # 予測シグナル
            if "predicted_change" in analysis.predictions:
                prediction_signal = base_signal.copy()
                prediction_signal.update(
                    {
                        "type": "PRICE_PREDICTION",
                        "predicted_change": analysis.predictions["predicted_change"],
                        "prediction_horizon": self.config.prediction_horizon,
                        "data_quality": analysis.data_quality,
                    }
                )
                signals.append(prediction_signal)

            # テクニカルシグナル
            if "moving_average" in analysis.technical_signals:
                ma_signal = base_signal.copy()
                ma_signal.update(
                    {
                        "type": "TECHNICAL_SIGNAL",
                        "indicator": "moving_average",
                        "signals": analysis.technical_signals["moving_average"],
                    }
                )
                signals.append(ma_signal)

            # リスクアラート
            if analysis.risk_assessment.get("overall_risk_score", 0) > 0.7:
                risk_signal = base_signal.copy()
                risk_signal.update(
                    {
                        "type": "RISK_ALERT",
                        "risk_level": analysis.risk_assessment.get(
                            "risk_level", "unknown"
                        ),
                        "risk_factors": analysis.risk_assessment,
                    }
                )
                signals.append(risk_signal)

        except Exception as e:
            logger.error(f"AIシグナル生成エラー {analysis.symbol}: {e}")

        return signals

    def _generate_smart_alerts(
        self, analysis: AIAnalysisResult
    ) -> List[Dict[str, Any]]:
        """
        スマートアラート生成"""

        alerts = []

        try:
            # データ品質アラート
            if analysis.data_quality < self.config.data_quality_threshold:
                alerts.append(
                    {
                        "symbol": analysis.symbol,
                        "type": "DATA_QUALITY_WARNING",
                        "message": f"データ品質低下: {analysis.data_quality:.1f}%",
                        "severity": "medium",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # 高信頼度予測アラート
            overall_confidence = analysis.confidence_scores.get("overall", 0)
            if overall_confidence > self.config.confidence_threshold:
                alerts.append(
                    {
                        "symbol": analysis.symbol,
                        "type": "HIGH_CONFIDENCE_PREDICTION",
                        "message": f"高信頼度予測: {analysis.recommendation} (信頼度: {overall_confidence:.2f})",
                        "severity": "high",
                        "timestamp": datetime.now().isoformat(),
                        "action_required": False,
                    }
                )

            # パフォーマンス異常アラート
            analysis_time = analysis.performance_metrics.get("analysis_time", 0)
            if analysis_time > 30:  # 30秒以上
                alerts.append(
                    {
                        "symbol": analysis.symbol,
                        "type": "PERFORMANCE_DEGRADATION",
                        "message": f"分析時間異常: {analysis_time:.1f}秒",
                        "severity": "low",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        except Exception as e:
            logger.error(f"スマートアラート生成エラー {analysis.symbol}: {e}")

        return alerts

    def _generate_portfolio_analysis(
        self, ai_results: List[AIAnalysisResult]
    ) -> Dict[str, Any]:
        """ポートフォリオ分析生成"""

        if not ai_results:
            return {
                "status": "analysis_only",
                "trading_disabled": True,
                "analyzed_symbols": 0,
            }

        try:
            # 総合統計
            total_symbols = len(ai_results)
            high_confidence_count = sum(
                1
                for r in ai_results
                if r.confidence_scores.get("overall", 0)
                > self.config.confidence_threshold
            )

            # 推奨分布
            recommendations = [r.recommendation for r in ai_results]
            recommendation_counts = {}
            for rec in set(recommendations):
                recommendation_counts[rec] = recommendations.count(rec)

            # 平均データ品質
            avg_data_quality = np.mean([r.data_quality for r in ai_results])

            # リスク分布
            risk_levels = [
                r.risk_assessment.get("risk_level", "unknown") for r in ai_results
            ]
            risk_distribution = {}
            for risk in set(risk_levels):
                risk_distribution[risk] = risk_levels.count(risk)

            return {
                "status": "analysis_only",
                "trading_disabled": True,
                "analyzed_symbols": total_symbols,
                "high_confidence_predictions": high_confidence_count,
                "recommendation_distribution": recommendation_counts,
                "average_data_quality": avg_data_quality,
                "risk_distribution": risk_distribution,
                "portfolio_metrics": {
                    "total_analysis_value": "N/A (分析専用)",
                    "confidence_weighted_score": np.mean(
                        [
                            r.confidence_scores.get("overall", 0)
                            for r in ai_results
                        ]
                    ),
                    "risk_weighted_score": np.mean(
                        [
                            r.risk_assessment.get("overall_risk_score", 0.5)
                            for r in ai_results
                        ]
                    ),
                },
            }

        except Exception as e:
            logger.error(f"ポートフォリオ分析エラー: {e}")
            return {
                "status": "analysis_only",
                "trading_disabled": True,
                "error": str(e),
            }

    def _analyze_system_health(self) -> Dict[str, Any]:
        """
        システムヘルス分析"""

        try:
            health = {"overall_status": "healthy", "components": {}}

            # MLエンジンヘルス
            if self.ml_engine:
                try:
                    ml_summary = self.ml_engine.get_model_summary()
                    health["components"]["ml_engine"] = {
                        "status": "operational",
                        "model_loaded": ml_summary.get("status") != "モデル未初期化",
                        "device": ml_summary.get("device", "unknown"),
                    }
                except Exception as e:
                    health["components"]["ml_engine"] = {
                        "status": "error",
                        "error": str(e),
                    }

            # バッチフェッチャーヘルス
            if self.batch_fetcher:
                try:
                    batch_stats = self.batch_fetcher.get_pipeline_stats()
                    health["components"]["batch_fetcher"] = {
                        "status": "operational",
                        "throughput": batch_stats.throughput_rps,
                        "success_rate": batch_stats.successful_requests
                        / batch_stats.total_requests
                        if batch_stats.total_requests > 0
                        else 1.0,
                    }
                except Exception as e:
                    health["components"]["batch_fetcher"] = {
                        "status": "error",
                        "error": str(e),
                    }

            # パフォーマンス監視ヘルス
            if self.performance_monitor:
                try:
                    health["components"]["performance_monitor"] = {
                        "status": "operational",
                        "monitoring_active": True,
                    }
                except Exception as e:
                    health["components"]["performance_monitor"] = {
                        "status": "error",
                        "error": str(e),
                    }

            # 全体ステータス判定
            component_statuses = [
                comp.get("status") for comp in health["components"].values()
            ]
            if "error" in component_statuses:
                health["overall_status"] = "degraded"

            return health

        except Exception as e:
            return {"overall_status": "error", "error": str(e)}

    def _calculate_performance_stats(self, start_time: datetime) -> Dict[str, Any]:
        """
        パフォーマンス統計計算"""

        try:
            execution_time = (datetime.now() - start_time).total_seconds()

            stats = {
                "execution_time_seconds": execution_time,
                "timestamp": datetime.now().isoformat(),
            }

            # バッチフェッチャー統計
            if self.batch_fetcher:
                batch_stats = self.batch_fetcher.get_pipeline_stats()
                stats["batch_fetcher"] = {
                    "total_requests": batch_stats.total_requests,
                    "success_rate": batch_stats.successful_requests
                    / batch_stats.total_requests
                    if batch_stats.total_requests > 0
                    else 0,
                    "avg_fetch_time": batch_stats.avg_fetch_time,
                    "throughput_rps": batch_stats.throughput_rps,
                }

            # MLエンジン統計
            if self.ml_engine and self.ml_engine.performance_history:
                avg_inference_time = np.mean(
                    [p["inference_time"] for p in self.ml_engine.performance_history]
                )
                stats["ml_engine"] = {
                    "predictions_made": len(self.ml_engine.performance_history),
                    "avg_inference_time": avg_inference_time,
                    "model_version": self.ml_engine.model_metadata["version"],
                }

            return stats

        except Exception as e:
            return {"error": str(e), "execution_time_seconds": 0}

    def _estimate_memory_usage(self, data: pd.DataFrame) -> float:
        """
        メモリ使用量推定"""
        try:
            return data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        except Exception:
            return 0.0

    async def run_async_advanced_analysis(
        self, symbols: List[str], analysis_type: str = "comprehensive"
    ) -> ExecutionReport:
        """
        非同期高度分析実行"""

        logger.info("非同期Next-Gen AI分析開始")

        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            None, self.run_advanced_analysis, symbols, analysis_type, True
        )

        logger.info("非同期Next-Gen AI分析完了")
        return report

    def get_execution_history(self, limit: int = 10) -> List[ExecutionReport]:
        """
        実行履歴取得"""
        return self.execution_history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """
        オーケストレーターステータス取得"""

        return {
            "safe_mode": is_safe_mode(),
            "trading_disabled": True,
            "automatic_trading": False,
            "analysis_engines": len(self.analysis_engines),
            "config_path": self.config_path,
            "mode": "next_gen_ai_analysis",
            "components": {
                "ml_engine_enabled": self.config.enable_ml_engine,
                "advanced_batch_enabled": self.config.enable_advanced_batch,
                "performance_monitoring": self.config.enable_performance_monitoring,
                "fault_tolerance": self.config.enable_fault_tolerance,
            },
            "execution_count": len(self.execution_history),
            "last_execution": self.execution_history[-1].start_time.isoformat()
            if self.execution_history
            else None,
        }

    def cleanup(self):
        """
        リソースクリーンアップ"""

        logger.info("Next-Gen AI Orchestrator クリーンアップ開始")

        try:
            # 分析エンジンクリーンアップ
            for symbol, engine in self.analysis_engines.items():
                try:
                    if hasattr(engine, "stop"):
                        engine.stop()
                    logger.debug(f"エンジン {symbol} クリーンアップ完了")
                except Exception as e:
                    logger.warning(f"エンジン {symbol} クリーンアップエラー: {e}")

            self.analysis_engines.clear()

            # バッチフェッチャークリーンアップ
            if self.batch_fetcher:
                try:
                    self.batch_fetcher.close()
                    logger.debug("バッチフェッチャー クリーンアップ完了")
                except Exception as e:
                    logger.warning(f"バッチフェッチャー クリーンアップエラー: {e}")

            logger.info("Next-Gen AI Orchestrator クリーンアップ完了")

        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")


# 後方互換性のためのエイリアス
DayTradeOrchestrator = NextGenAIOrchestrator
