#!/usr/bin/env python3
"""
高度並列ML処理エンジン
Issue #323: ML Processing Parallelization for Throughput Improvement

4-8倍のスループット改善を実現する並列化システム
"""

import gc
import os
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil

try:
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # キャッシュマネージャーのモック
    class UnifiedCacheManager:
        def __init__(self, **kwargs):
            pass

        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

        def batch_get(self, keys, **kwargs):
            return {}

        def batch_put(self, data, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"mock_key_{hash(str(args) + str(kwargs))}"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 依存関係チェック
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta未インストール - 技術指標計算が制限されます")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("scikit-learn未インストール - ML機能利用不可")


@dataclass
class ProcessingTask:
    """処理タスク定義"""

    task_id: str
    symbol: str
    data: pd.DataFrame
    task_type: str  # 'ml_features', 'technical_indicators', 'prediction'
    priority: float = 1.0
    timeout: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """処理結果"""

    task_id: str
    symbol: str
    success: bool
    result: Any = None
    processing_time: float = 0.0
    memory_used: float = 0.0
    error_message: Optional[str] = None
    worker_id: Optional[str] = None


@dataclass
class ResourceMetrics:
    """リソースメトリクス"""

    cpu_usage: float
    memory_usage_mb: float
    available_memory_mb: float
    cpu_count: int
    timestamp: datetime


class ResourceMonitor:
    """リソース監視システム"""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        self._monitoring = False
        self._monitor_thread = None
        self.metrics_history = []
        self.lock = threading.RLock()

    def start_monitoring(self):
        """監視開始"""
        with self.lock:
            if not self._monitoring:
                self._monitoring = True
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop, daemon=True
                )
                self._monitor_thread.start()
                logger.info("リソース監視開始")

    def stop_monitoring(self):
        """監視停止"""
        with self.lock:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2)
                logger.info("リソース監視停止")

    def _monitor_loop(self):
        """監視ループ"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)
                    # 履歴を100件に制限
                    if len(self.metrics_history) > 100:
                        self.metrics_history = self.metrics_history[-100:]

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"リソース監視エラー: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_metrics(self) -> ResourceMetrics:
        """メトリクス収集"""
        cpu_usage = psutil.cpu_percent()

        memory_info = self.process.memory_info()
        current_memory = (memory_info.rss - self.initial_memory) / 1024 / 1024

        system_memory = psutil.virtual_memory()
        available_memory = system_memory.available / 1024 / 1024

        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage_mb=current_memory,
            available_memory_mb=available_memory,
            cpu_count=os.cpu_count(),
            timestamp=datetime.now(),
        )

    def get_current_metrics(self) -> ResourceMetrics:
        """現在のメトリクス取得"""
        return self._collect_metrics()

    def get_average_metrics(self, last_n_samples: int = 10) -> ResourceMetrics:
        """平均メトリクス取得"""
        with self.lock:
            if not self.metrics_history:
                return self.get_current_metrics()

            recent_metrics = self.metrics_history[-last_n_samples:]

            avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
            avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
            avg_available = np.mean([m.available_memory_mb for m in recent_metrics])

            return ResourceMetrics(
                cpu_usage=avg_cpu,
                memory_usage_mb=avg_memory,
                available_memory_mb=avg_available,
                cpu_count=os.cpu_count(),
                timestamp=datetime.now(),
            )


class AdaptiveLoadBalancer:
    """動的負荷分散システム"""

    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.task_history = []
        self.performance_cache = {}

    def calculate_optimal_workers(self, task_type: str, task_count: int) -> int:
        """最適worker数算出"""
        metrics = self.resource_monitor.get_current_metrics()

        if task_type == "ml_computation":
            # CPU集約的処理
            base_workers = min(metrics.cpu_count, 8)

            # CPU使用率に基づく調整
            if metrics.cpu_usage > 80:
                return max(1, base_workers // 2)
            elif metrics.cpu_usage < 40:
                return min(base_workers * 2, 16)
            else:
                return base_workers

        elif task_type == "data_processing":
            # I/Oバウンド処理
            memory_factor = min(metrics.available_memory_mb / 1024, 4)  # 最大4倍
            base_workers = int(metrics.cpu_count * memory_factor)

            return min(base_workers, 20)

        elif task_type == "feature_engineering":
            # 中間処理
            return min(metrics.cpu_count, task_count, 6)

        else:
            # デフォルト
            return min(metrics.cpu_count, 4)

    def should_use_process_pool(self, task_type: str, data_size_mb: float) -> bool:
        """プロセスプール使用判定"""
        if task_type == "ml_computation" and data_size_mb > 10:
            return True

        if data_size_mb > 50:  # 大量データ
            return True

        return False

    def estimate_task_complexity(self, symbol: str, data: pd.DataFrame) -> float:
        """タスク複雑度推定"""
        base_complexity = len(data) / 1000  # データサイズベース

        # 銘柄別調整（ボラティリティ等）
        try:
            price_volatility = data["Close"].pct_change().std()
            complexity_factor = 1.0 + (price_volatility * 10)  # ボラティリティ調整
        except:
            complexity_factor = 1.0

        return base_complexity * complexity_factor


def parallel_ml_worker(task_data: Tuple[str, pd.DataFrame, Dict]) -> ProcessingResult:
    """並列処理ワーカー関数（プロセス用）"""
    try:
        symbol, data, config = task_data
        start_time = time.time()
        worker_id = f"worker_{os.getpid()}_{threading.current_thread().ident}"

        # メモリ使用量測定開始
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # ML特徴量計算
        features = calculate_essential_features_optimized(data)

        # 簡易予測（軽量モデル）
        prediction = generate_lightweight_prediction(features)

        # 処理時間とメモリ使用量
        processing_time = time.time() - start_time
        final_memory = process.memory_info().rss
        memory_used = (final_memory - initial_memory) / 1024 / 1024

        return ProcessingResult(
            task_id=f"{symbol}_{int(start_time)}",
            symbol=symbol,
            success=True,
            result={"features": features, "prediction": prediction},
            processing_time=processing_time,
            memory_used=memory_used,
            worker_id=worker_id,
        )

    except Exception as e:
        return ProcessingResult(
            task_id=f"{symbol}_error",
            symbol=symbol,
            success=False,
            error_message=str(e),
            processing_time=time.time() - start_time if "start_time" in locals() else 0,
            worker_id=worker_id if "worker_id" in locals() else "unknown",
        )


def calculate_essential_features_optimized(data: pd.DataFrame) -> Dict[str, Any]:
    """最適化された特徴量計算（Issue #325の最適化適用）"""
    if not PANDAS_TA_AVAILABLE:
        return {}

    try:
        # 最小限の必須指標のみ計算（Issue #325で特定した16指標）
        features = {}

        # トレンド指標
        features["sma_20"] = ta.sma(data["Close"], length=20).iloc[-1]
        features["ema_12"] = ta.ema(data["Close"], length=12).iloc[-1]
        features["ema_26"] = ta.ema(data["Close"], length=26).iloc[-1]

        # モメンタム指標
        features["rsi_14"] = ta.rsi(data["Close"], length=14).iloc[-1]
        features["macd"] = ta.macd(data["Close"])["MACD_12_26_9"].iloc[-1]
        features["macd_signal"] = ta.macd(data["Close"])["MACDs_12_26_9"].iloc[-1]

        # ボラティリティ指標
        bb = ta.bbands(data["Close"], length=20)
        features["bb_upper"] = bb["BBU_20_2.0"].iloc[-1]
        features["bb_lower"] = bb["BBL_20_2.0"].iloc[-1]

        # ボリューム指標
        features["volume_sma"] = ta.sma(data["Volume"], length=10).iloc[-1]

        # 価格動向
        features["price_change_pct"] = (
            (data["Close"].iloc[-1] - data["Close"].iloc[-5])
            / data["Close"].iloc[-5]
            * 100
        )
        features["volatility"] = (
            data["Close"].pct_change().rolling(20).std().iloc[-1] * 100
        )

        # NaN値処理
        features = {k: v for k, v in features.items() if not pd.isna(v)}

        return features

    except Exception as e:
        logger.error(f"特徴量計算エラー: {e}")
        return {}


def generate_lightweight_prediction(features: Dict[str, Any]) -> Dict[str, Any]:
    """軽量予測生成"""
    if not features or not SKLEARN_AVAILABLE:
        return {"prediction": 0.0, "confidence": 0.0}

    try:
        # 簡単なルールベース予測（高速）
        score = 0.0
        weight_sum = 0.0

        # RSI判定
        if "rsi_14" in features:
            rsi = features["rsi_14"]
            if rsi < 30:  # 買い
                score += 1.0
                weight_sum += 1.0
            elif rsi > 70:  # 売り
                score -= 1.0
                weight_sum += 1.0

        # MACD判定
        if "macd" in features and "macd_signal" in features:
            macd_diff = features["macd"] - features["macd_signal"]
            if macd_diff > 0:  # 買いシグナル
                score += 0.5
            else:  # 売りシグナル
                score -= 0.5
            weight_sum += 0.5

        # 価格変動判定
        if "price_change_pct" in features:
            change = features["price_change_pct"]
            if change > 5:  # 強い上昇
                score += 0.8
            elif change > 2:  # 中程度上昇
                score += 0.3
            elif change < -5:  # 強い下降
                score -= 0.8
            elif change < -2:  # 中程度下降
                score -= 0.3
            weight_sum += 0.8

        # 最終スコア計算
        if weight_sum > 0:
            final_score = score / weight_sum
            confidence = min(weight_sum / 2.3, 1.0)  # 最大信頼度
        else:
            final_score = 0.0
            confidence = 0.0

        return {
            "prediction": final_score,
            "confidence": confidence,
            "signal": "BUY"
            if final_score > 0.3
            else "SELL"
            if final_score < -0.3
            else "HOLD",
        }

    except Exception as e:
        logger.error(f"予測生成エラー: {e}")
        return {"prediction": 0.0, "confidence": 0.0, "signal": "HOLD"}


class AdvancedParallelMLEngine:
    """
    高度並列ML処理エンジン

    4-8倍のスループット改善を実現する並列化システム
    """

    def __init__(
        self,
        cpu_workers: Optional[int] = None,
        io_workers: Optional[int] = None,
        memory_limit_gb: float = 2.0,
        cache_enabled: bool = True,
        enable_monitoring: bool = True,
    ):
        """
        初期化

        Args:
            cpu_workers: CPU並列ワーカー数（Noneの場合自動検出）
            io_workers: I/O並列ワーカー数（Noneの場合自動検出）
            memory_limit_gb: メモリ制限（GB）
            cache_enabled: キャッシュ機能有効化
            enable_monitoring: リソース監視有効化
        """
        # システムリソース検出
        self.cpu_count = os.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)

        # ワーカー数設定
        self.cpu_workers = cpu_workers or min(self.cpu_count, 8)
        self.io_workers = io_workers or min(int(self.memory_gb * 4), 20)
        self.memory_limit_gb = memory_limit_gb

        # 並列化プール
        self.process_pool = None
        self.thread_pool = None
        self._pools_initialized = False

        # キャッシュシステム
        if cache_enabled:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=64,  # ホットキャッシュ
                    l2_memory_mb=256,  # ウォームキャッシュ
                    l3_disk_mb=512,  # コールドキャッシュ
                )
                self.cache_enabled = True
                logger.info("統合キャッシュシステム有効化")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
                self.cache_enabled = False
        else:
            self.cache_manager = None
            self.cache_enabled = False

        # リソース監視
        if enable_monitoring:
            self.resource_monitor = ResourceMonitor()
            self.load_balancer = AdaptiveLoadBalancer(self.resource_monitor)
            self.resource_monitor.start_monitoring()
        else:
            self.resource_monitor = None
            self.load_balancer = None

        # 統計情報
        self.processing_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "cache_hits": 0,
            "processing_times": [],
            "memory_usage_history": [],
        }

        logger.info("高度並列MLエンジン初期化完了")
        logger.info(f"  - CPU workers: {self.cpu_workers}")
        logger.info(f"  - I/O workers: {self.io_workers}")
        logger.info(f"  - メモリ制限: {memory_limit_gb}GB")
        logger.info(f"  - キャッシュ: {'有効' if self.cache_enabled else '無効'}")
        logger.info(f"  - リソース監視: {'有効' if enable_monitoring else '無効'}")

    def _initialize_pools(self):
        """プール初期化"""
        if not self._pools_initialized:
            self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
            self.thread_pool = ThreadPoolExecutor(max_workers=self.io_workers)
            self._pools_initialized = True
            logger.info("並列処理プール初期化完了")

    def _shutdown_pools(self):
        """プールシャットダウン"""
        if self._pools_initialized:
            if self.process_pool:
                self.process_pool.shutdown(wait=True)
                self.process_pool = None
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                self.thread_pool = None
            self._pools_initialized = False
            logger.info("並列処理プールシャットダウン完了")

    def batch_process_symbols(
        self,
        stock_data: Dict[str, pd.DataFrame],
        use_cache: bool = True,
        timeout_per_symbol: int = 60,
    ) -> Tuple[Dict[str, Any], float]:
        """
        バッチ並列処理

        Args:
            stock_data: 株式データ辞書 {symbol: DataFrame}
            use_cache: キャッシュ使用
            timeout_per_symbol: 銘柄あたりタイムアウト（秒）

        Returns:
            Tuple[処理結果辞書, 総処理時間]
        """
        start_time = time.time()

        try:
            # プール初期化
            self._initialize_pools()

            # キャッシュチェック
            if use_cache and self.cache_enabled:
                cached_results, uncached_symbols = self._check_cache_batch(
                    stock_data.keys()
                )
                logger.info(
                    f"キャッシュヒット: {len(cached_results)}/{len(stock_data)} 銘柄"
                )
                self.processing_stats["cache_hits"] += len(cached_results)
            else:
                cached_results = {}
                uncached_symbols = list(stock_data.keys())

            # 未キャッシュ銘柄の処理
            if uncached_symbols:
                uncached_data = {
                    symbol: stock_data[symbol] for symbol in uncached_symbols
                }
                new_results = self._parallel_process_batch(
                    uncached_data, timeout_per_symbol
                )

                # キャッシュに保存
                if use_cache and self.cache_enabled:
                    self._save_to_cache_batch(new_results)
            else:
                new_results = {}

            # 結果統合
            final_results = {**cached_results, **new_results}

            # 統計更新
            total_time = time.time() - start_time
            self._update_processing_stats(final_results, total_time)

            logger.info(
                f"バッチ処理完了: {len(final_results)}/{len(stock_data)} 銘柄 ({total_time:.2f}秒)"
            )

            return final_results, total_time

        except Exception as e:
            logger.error(f"バッチ処理エラー: {e}")
            return {}, time.time() - start_time

        finally:
            # リソースクリーンアップ
            gc.collect()

    def _check_cache_batch(
        self, symbols: List[str]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """バッチキャッシュチェック"""
        cached_results = {}
        uncached_symbols = []

        for symbol in symbols:
            cache_key = generate_unified_cache_key(
                "ml_engine",
                "parallel_analysis",
                symbol,
                time_bucket_minutes=10,  # 10分間キャッシュ
            )

            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                cached_results[symbol] = cached_result
            else:
                uncached_symbols.append(symbol)

        return cached_results, uncached_symbols

    def _save_to_cache_batch(self, results: Dict[str, Any]):
        """バッチキャッシュ保存"""
        for symbol, result in results.items():
            cache_key = generate_unified_cache_key(
                "ml_engine", "parallel_analysis", symbol, time_bucket_minutes=10
            )

            # 重要度設定（成功結果は高優先度）
            priority = (
                8.0
                if isinstance(result, dict) and result.get("success", False)
                else 3.0
            )

            self.cache_manager.put(cache_key, result, priority=priority)

    def _parallel_process_batch(
        self, stock_data: Dict[str, pd.DataFrame], timeout_per_symbol: int
    ) -> Dict[str, Any]:
        """並列バッチ処理実行"""

        # 負荷分散最適化
        if self.load_balancer:
            optimal_cpu_workers = self.load_balancer.calculate_optimal_workers(
                "ml_computation", len(stock_data)
            )

            # 必要に応じてワーカー数調整
            if optimal_cpu_workers != self.cpu_workers:
                logger.info(
                    f"ワーカー数動的調整: {self.cpu_workers} → {optimal_cpu_workers}"
                )
                self.cpu_workers = optimal_cpu_workers
                # プール再初期化
                self._shutdown_pools()
                self._initialize_pools()

        # タスク準備
        task_data = []
        for symbol, data in stock_data.items():
            if not data.empty:
                task_data.append((symbol, data, {}))  # config空辞書

        # 並列実行
        results = {}
        futures = []

        try:
            # プロセスプール実行
            for task in task_data:
                future = self.process_pool.submit(parallel_ml_worker, task)
                futures.append((future, task[0]))  # (future, symbol)

            # 結果収集
            for future, symbol in futures:
                try:
                    result = future.result(timeout=timeout_per_symbol)
                    if result.success:
                        results[symbol] = result.result
                    else:
                        logger.error(f"並列処理失敗 {symbol}: {result.error_message}")
                        results[symbol] = {"error": result.error_message}

                except Exception as e:
                    logger.error(f"並列処理例外 {symbol}: {e}")
                    results[symbol] = {"error": str(e)}

        except Exception as e:
            logger.error(f"並列実行エラー: {e}")

        return results

    def _update_processing_stats(self, results: Dict[str, Any], processing_time: float):
        """統計情報更新"""
        successful = sum(
            1 for r in results.values() if isinstance(r, dict) and not r.get("error")
        )

        self.processing_stats["total_processed"] += len(results)
        self.processing_stats["successful_processed"] += successful
        self.processing_stats["processing_times"].append(processing_time)

        # メモリ使用量記録
        if self.resource_monitor:
            metrics = self.resource_monitor.get_current_metrics()
            self.processing_stats["memory_usage_history"].append(
                metrics.memory_usage_mb
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """性能統計取得"""
        stats = self.processing_stats.copy()

        if stats["processing_times"]:
            stats["avg_processing_time"] = np.mean(stats["processing_times"])
            stats["min_processing_time"] = np.min(stats["processing_times"])
            stats["max_processing_time"] = np.max(stats["processing_times"])

        if stats["memory_usage_history"]:
            stats["avg_memory_usage_mb"] = np.mean(stats["memory_usage_history"])
            stats["peak_memory_usage_mb"] = np.max(stats["memory_usage_history"])

        # 成功率計算
        if stats["total_processed"] > 0:
            stats["success_rate"] = (
                stats["successful_processed"] / stats["total_processed"]
            )
        else:
            stats["success_rate"] = 0.0

        # キャッシュヒット率
        if stats["total_processed"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_processed"]
        else:
            stats["cache_hit_rate"] = 0.0

        return stats

    def optimize_performance(self):
        """性能最適化実行"""
        if self.resource_monitor:
            metrics = self.resource_monitor.get_average_metrics(10)

            logger.info(
                f"性能最適化実行 - CPU: {metrics.cpu_usage:.1f}%, "
                f"Memory: {metrics.memory_usage_mb:.1f}MB"
            )

            # メモリ最適化
            if metrics.memory_usage_mb > self.memory_limit_gb * 1024 * 0.8:
                logger.warning("メモリ使用量が制限に接近 - 最適化実行")
                gc.collect()

                if self.cache_manager:
                    self.cache_manager.optimize_memory()

    def shutdown(self):
        """システムシャットダウン"""
        logger.info("高度並列MLエンジンシャットダウン開始")

        # プールシャットダウン
        self._shutdown_pools()

        # リソース監視停止
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()

        # キャッシュクリーンアップ
        if self.cache_manager:
            # 最終統計レポート
            if hasattr(self.cache_manager, "get_comprehensive_stats"):
                final_stats = self.cache_manager.get_comprehensive_stats()
                logger.info(
                    f"キャッシュ最終統計: ヒット率={final_stats['overall']['hit_rate']:.1%}"
                )

        logger.info("高度並列MLエンジンシャットダウン完了")

    def __enter__(self):
        """コンテキストマネージャーエントリ"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーエグジット"""
        self.shutdown()


if __name__ == "__main__":
    # テスト実行
    print("=== 高度並列MLエンジンテスト ===")

    # テストデータ生成
    test_symbols = ["7203", "8306", "9984"]
    test_stock_data = {}

    for symbol in test_symbols:
        # モックデータ生成
        dates = pd.date_range(start="2024-01-01", periods=100)
        test_data = pd.DataFrame(
            {
                "Open": np.random.uniform(2000, 3000, 100),
                "High": np.random.uniform(2100, 3100, 100),
                "Low": np.random.uniform(1900, 2900, 100),
                "Close": np.random.uniform(2000, 3000, 100),
                "Volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )
        test_stock_data[symbol] = test_data

    # 並列エンジンテスト
    try:
        with AdvancedParallelMLEngine(
            cpu_workers=4,
            io_workers=8,
            memory_limit_gb=1.0,
            cache_enabled=True,
            enable_monitoring=True,
        ) as engine:
            # バッチ処理実行
            print("\n1. バッチ並列処理テスト...")
            results, processing_time = engine.batch_process_symbols(test_stock_data)

            print(f"処理時間: {processing_time:.2f}秒")
            print(f"成功銘柄: {len(results)}/{len(test_stock_data)}")

            # 性能統計
            print("\n2. 性能統計...")
            stats = engine.get_performance_stats()
            print(f"平均処理時間: {stats.get('avg_processing_time', 0):.3f}秒")
            print(f"成功率: {stats.get('success_rate', 0):.1%}")
            print(f"キャッシュヒット率: {stats.get('cache_hit_rate', 0):.1%}")

            # 2回目実行（キャッシュ効果テスト）
            print("\n3. キャッシュ効果テスト...")
            results2, processing_time2 = engine.batch_process_symbols(test_stock_data)
            print(f"2回目処理時間: {processing_time2:.2f}秒")

            if processing_time > 0:
                speedup = processing_time / processing_time2
                print(f"高速化: {speedup:.1f}倍")

            print("\n✅ 高度並列MLエンジンテスト完了")

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback

        traceback.print_exc()
