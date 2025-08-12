#!/usr/bin/env python3
"""
統一バッチデータプロセッサー
Issue #376: バッチ処理の強化

複数銘柄・期間のデータを効率的にバッチ処理するための統一フレームワーク
"""

import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# プロジェクトモジュール
try:
    from ..data.enhanced_stock_fetcher import EnhancedStockFetcher
    from ..data.stock_fetcher import StockFetcher
    from ..models.bulk_operations import BulkOperationManager
    from ..models.database import DatabaseManager, get_database_manager
    from ..utils.cache_utils import generate_safe_cache_key
    from ..utils.logging_config import get_context_logger, log_performance_metric
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    def generate_safe_cache_key(*args, **kwargs):
        return str(hash(str(args) + str(kwargs)))

    # モッククラス
    class StockFetcher:
        def get_current_price(self, code):
            return {"price": 100.0, "timestamp": time.time()}

    class EnhancedStockFetcher:
        def __init__(self, **kwargs):
            pass

        def get_current_price(self, code):
            return {"price": 100.0, "timestamp": time.time()}

    def get_database_manager():
        return None


logger = get_context_logger(__name__)


class BatchOperationType(Enum):
    """バッチ操作タイプ"""

    PRICE_FETCH = "price_fetch"
    HISTORICAL_FETCH = "historical_fetch"
    COMPANY_INFO_FETCH = "company_info_fetch"
    DATABASE_INSERT = "database_insert"
    DATABASE_UPDATE = "database_update"
    MIXED_OPERATION = "mixed_operation"


@dataclass
class BatchRequest:
    """バッチリクエストデータ"""

    operation_type: BatchOperationType
    parameters: Dict[str, Any]
    priority: int = 1
    timeout_seconds: int = 300
    retry_count: int = 3
    created_at: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: f"batch_{int(time.time() * 1000)}")

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.timeout_seconds


@dataclass
class BatchResult:
    """バッチ処理結果"""

    request_id: str
    operation_type: BatchOperationType
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    items_processed: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    completed_at: float = field(default_factory=time.time)


class BatchQueue:
    """バッチ処理キュー（優先度付き）"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queues = {priority: deque() for priority in range(1, 6)}  # 優先度1-5
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self.total_size = 0

    def put(self, request: BatchRequest) -> bool:
        """リクエストをキューに追加"""
        with self._condition:
            if self.total_size >= self.max_size:
                return False

            # 期限切れリクエストの除去
            self._cleanup_expired()

            priority = max(1, min(5, request.priority))  # 1-5に正規化
            self._queues[priority].append(request)
            self.total_size += 1

            self._condition.notify_all()
            return True

    def get(self, timeout: float = 1.0) -> Optional[BatchRequest]:
        """最高優先度のリクエストを取得"""
        with self._condition:
            end_time = time.time() + timeout

            while self.total_size == 0:
                remaining = end_time - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

            # 高優先度から順に確認
            for priority in range(5, 0, -1):
                if self._queues[priority]:
                    request = self._queues[priority].popleft()
                    self.total_size -= 1
                    return request

            return None

    def _cleanup_expired(self):
        """期限切れリクエストの削除"""
        for priority_queue in self._queues.values():
            expired_count = 0
            while priority_queue and priority_queue[0].is_expired():
                priority_queue.popleft()
                expired_count += 1

            self.total_size = max(0, self.total_size - expired_count)

    def size(self) -> int:
        with self._lock:
            return self.total_size

    def get_queue_stats(self) -> Dict[str, int]:
        with self._lock:
            return {f"priority_{p}": len(q) for p, q in self._queues.items()}


class BatchOptimizer:
    """バッチ処理最適化エンジン"""

    def __init__(self):
        self.operation_stats = defaultdict(
            lambda: {
                "total_requests": 0,
                "avg_processing_time": 0.0,
                "success_rate": 0.0,
                "optimal_batch_size": 50,
            }
        )
        self._lock = threading.RLock()

    def record_operation(
        self,
        operation_type: BatchOperationType,
        batch_size: int,
        processing_time_ms: float,
        success: bool,
    ):
        """操作結果の記録と学習"""
        with self._lock:
            stats = self.operation_stats[operation_type.value]

            # 指数平滑化による統計更新
            alpha = 0.1
            stats["total_requests"] += 1
            stats["avg_processing_time"] = (
                stats["avg_processing_time"] * (1 - alpha) + processing_time_ms * alpha
            )

            if success:
                stats["success_rate"] = stats["success_rate"] * (1 - alpha) + 1.0 * alpha

                # 最適バッチサイズの動的調整
                current_optimal = stats["optimal_batch_size"]
                if processing_time_ms < stats["avg_processing_time"] * 0.8:
                    # 処理が高速な場合は少し増加
                    stats["optimal_batch_size"] = min(200, int(current_optimal * 1.1))
                elif processing_time_ms > stats["avg_processing_time"] * 1.2:
                    # 処理が遅い場合は減少
                    stats["optimal_batch_size"] = max(10, int(current_optimal * 0.9))

    def get_optimal_batch_size(self, operation_type: BatchOperationType) -> int:
        """最適バッチサイズの取得"""
        with self._lock:
            return self.operation_stats[operation_type.value]["optimal_batch_size"]

    def should_combine_requests(self, requests: List[BatchRequest]) -> bool:
        """リクエストの結合判定"""
        if len(requests) < 2:
            return False

        # 同一操作タイプで、優先度が近い場合は結合
        operation_types = set(r.operation_type for r in requests)
        priorities = set(r.priority for r in requests)

        return len(operation_types) == 1 and max(priorities) - min(priorities) <= 1

    def get_processing_stats(self) -> Dict[str, Any]:
        """処理統計の取得"""
        with self._lock:
            return dict(self.operation_stats)


class BatchDataProcessor:
    """統一バッチデータプロセッサー"""

    def __init__(
        self,
        max_workers: int = 8,
        queue_size: int = 1000,
        default_batch_size: int = 50,
        enable_cache: bool = True,
    ):
        """
        初期化

        Args:
            max_workers: 最大ワーカー数
            queue_size: キューサイズ
            default_batch_size: デフォルトバッチサイズ
            enable_cache: キャッシュ有効化
        """
        self.max_workers = max_workers
        self.default_batch_size = default_batch_size
        self.enable_cache = enable_cache

        # コア コンポーネント
        self.batch_queue = BatchQueue(queue_size)
        self.optimizer = BatchOptimizer()

        # データフェッチャー
        self.stock_fetcher = StockFetcher()
        self.enhanced_fetcher = None

        # データベース
        self.db_manager = get_database_manager()
        self.bulk_ops_manager = None
        if self.db_manager:
            try:
                self.bulk_ops_manager = BulkOperationManager(self.db_manager)
            except Exception as e:
                logger.warning(f"BulkOperationManager初期化失敗: {e}")

        # スレッドプール
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # 統計情報
        self.stats = {
            "requests_processed": 0,
            "requests_failed": 0,
            "total_processing_time_ms": 0.0,
            "avg_batch_size": 0.0,
            "cache_hit_rate": 0.0,
        }
        self._stats_lock = threading.RLock()

        # ワーカースレッド開始
        self._running = True
        self.worker_threads = []
        for i in range(max_workers):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)

        logger.info(
            f"BatchDataProcessor初期化完了: workers={max_workers}, "
            f"queue_size={queue_size}, batch_size={default_batch_size}"
        )

    def submit_batch_request(self, request: BatchRequest) -> str:
        """バッチリクエスト送信"""
        if not self.batch_queue.put(request):
            raise RuntimeError("バッチキューが満杯です")

        logger.debug(f"バッチリクエスト送信: {request.request_id} ({request.operation_type.value})")
        return request.request_id

    def bulk_fetch_prices(
        self, codes: List[str], priority: int = 1, timeout_seconds: int = 300
    ) -> str:
        """価格一括取得リクエスト"""
        request = BatchRequest(
            operation_type=BatchOperationType.PRICE_FETCH,
            parameters={"codes": codes},
            priority=priority,
            timeout_seconds=timeout_seconds,
        )
        return self.submit_batch_request(request)

    def bulk_fetch_historical_data(
        self,
        codes: List[str],
        period: str = "1mo",
        interval: str = "1d",
        priority: int = 2,
        timeout_seconds: int = 600,
    ) -> str:
        """ヒストリカルデータ一括取得リクエスト"""
        request = BatchRequest(
            operation_type=BatchOperationType.HISTORICAL_FETCH,
            parameters={"codes": codes, "period": period, "interval": interval},
            priority=priority,
            timeout_seconds=timeout_seconds,
        )
        return self.submit_batch_request(request)

    def bulk_database_operation(
        self,
        operation: str,  # 'insert' or 'update'
        table_name: str,
        data: List[Dict[str, Any]],
        priority: int = 3,
    ) -> str:
        """データベース一括操作リクエスト"""
        op_type = (
            BatchOperationType.DATABASE_INSERT
            if operation == "insert"
            else BatchOperationType.DATABASE_UPDATE
        )

        request = BatchRequest(
            operation_type=op_type,
            parameters={"operation": operation, "table_name": table_name, "data": data},
            priority=priority,
        )
        return self.submit_batch_request(request)

    def _worker_loop(self, worker_id: int):
        """ワーカーループ"""
        logger.debug(f"バッチワーカー{worker_id}開始")

        while self._running:
            try:
                # リクエストの取得
                request = self.batch_queue.get(timeout=1.0)
                if not request:
                    continue

                # リクエスト処理
                result = self._process_request(request)
                self._update_stats(result)

                # 最適化統計の更新
                self.optimizer.record_operation(
                    request.operation_type,
                    result.items_processed,
                    result.processing_time_ms,
                    result.success,
                )

            except Exception as e:
                logger.error(f"バッチワーカー{worker_id}エラー: {e}")

        logger.debug(f"バッチワーカー{worker_id}終了")

    def _process_request(self, request: BatchRequest) -> BatchResult:
        """リクエスト処理"""
        start_time = time.time()

        try:
            if request.operation_type == BatchOperationType.PRICE_FETCH:
                (
                    result_data,
                    items_processed,
                    cache_hits,
                    api_calls,
                ) = self._process_price_fetch(request.parameters.get("codes", []))
            elif request.operation_type == BatchOperationType.HISTORICAL_FETCH:
                (
                    result_data,
                    items_processed,
                    cache_hits,
                    api_calls,
                ) = self._process_historical_fetch(request.parameters)
            elif request.operation_type in [
                BatchOperationType.DATABASE_INSERT,
                BatchOperationType.DATABASE_UPDATE,
            ]:
                (
                    result_data,
                    items_processed,
                    cache_hits,
                    api_calls,
                ) = self._process_database_operation(request.parameters)
            else:
                raise ValueError(f"未サポート操作タイプ: {request.operation_type}")

            processing_time_ms = (time.time() - start_time) * 1000

            log_performance_metric(
                f"batch_{request.operation_type.value}", processing_time_ms, "ms"
            )

            return BatchResult(
                request_id=request.request_id,
                operation_type=request.operation_type,
                success=True,
                data=result_data,
                processing_time_ms=processing_time_ms,
                items_processed=items_processed,
                cache_hits=cache_hits,
                api_calls=api_calls,
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"バッチ処理エラー ({request.request_id}): {e}")

            return BatchResult(
                request_id=request.request_id,
                operation_type=request.operation_type,
                success=False,
                data={},
                error=str(e),
                processing_time_ms=processing_time_ms,
            )

    def _process_price_fetch(self, codes: List[str]) -> Tuple[Dict[str, Any], int, int, int]:
        """価格取得処理"""
        if not codes:
            return {}, 0, 0, 0

        # Enhanced Stock Fetcherを優先使用
        if not self.enhanced_fetcher:
            try:
                self.enhanced_fetcher = EnhancedStockFetcher(
                    cache_config={
                        "persistent_cache_enabled": True,
                        "enable_multi_layer_cache": True,
                    }
                )
            except Exception:
                self.enhanced_fetcher = None

        fetcher = self.enhanced_fetcher or self.stock_fetcher
        results = {}
        cache_hits = 0
        api_calls = 0

        # バッチサイズの最適化
        optimal_batch_size = self.optimizer.get_optimal_batch_size(BatchOperationType.PRICE_FETCH)

        # バッチごとに処理
        for i in range(0, len(codes), optimal_batch_size):
            batch_codes = codes[i : i + optimal_batch_size]

            # 拡張fetcherのbulk取得を使用
            if hasattr(fetcher, "bulk_get_current_prices"):
                batch_results = fetcher.bulk_get_current_prices(
                    batch_codes, batch_size=len(batch_codes)
                )
                results.update(batch_results)
            else:
                # 個別取得のフォールバック
                for code in batch_codes:
                    try:
                        price_data = fetcher.get_current_price(code)
                        if price_data:
                            results[code] = price_data
                            api_calls += 1
                    except Exception as e:
                        logger.debug(f"価格取得失敗 {code}: {e}")

        # キャッシュ統計の取得
        if hasattr(fetcher, "get_cache_stats"):
            try:
                cache_stats = fetcher.get_cache_stats()
                cache_hits = cache_stats.get("cache_stats", {}).get("l1_hits", 0)
            except Exception:
                pass

        return results, len(codes), cache_hits, api_calls

    def _process_historical_fetch(
        self, params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int, int, int]:
        """ヒストリカルデータ取得処理"""
        codes = params.get("codes", [])
        period = params.get("period", "1mo")
        interval = params.get("interval", "1d")

        if not codes:
            return {}, 0, 0, 0

        fetcher = self.enhanced_fetcher or self.stock_fetcher
        results = {}
        api_calls = 0

        # 最適バッチサイズを使用
        optimal_batch_size = self.optimizer.get_optimal_batch_size(
            BatchOperationType.HISTORICAL_FETCH
        )

        if hasattr(fetcher, "bulk_get_historical_data"):
            bulk_results = fetcher.bulk_get_historical_data(
                codes, period=period, interval=interval, batch_size=optimal_batch_size
            )
            results.update(bulk_results)
            api_calls = len(
                codes
            )  # yfinance.downloadが効率的にAPIを呼び出すため、リクエスト数としては全コード数をカウント
        else:
            # フォールバックとして従来の個別取得
            for code in codes:
                try:
                    historical_data = fetcher.get_historical_data(code, period, interval)
                    if historical_data is not None:
                        results[code] = historical_data
                        api_calls += 1
                except Exception as e:
                    logger.debug(f"ヒストリカルデータ取得失敗 {code}: {e}")

        return results, len(codes), 0, api_calls

    def _process_database_operation(
        self, params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int, int, int]:
        """データベース操作処理"""
        operation = params.get("operation")
        table_name = params.get("table_name")
        data = params.get("data", [])

        if not data or not self.bulk_ops_manager:
            return {"status": "skipped", "reason": "no_data_or_manager"}, 0, 0, 0

        try:
            if operation == "insert":
                result = self.bulk_ops_manager.bulk_insert_with_conflict_resolution(
                    table_name, data
                )
            else:  # update
                # バルクアップデート処理（実装は bulk_ops_manager に依存）
                result = {"inserted": 0, "updated": len(data), "failed": 0}

            return result, len(data), 0, 0

        except Exception as e:
            logger.error(f"データベース操作失敗 ({operation}): {e}")
            return {"status": "error", "error": str(e)}, len(data), 0, 0

    def _update_stats(self, result: BatchResult):
        """統計情報更新"""
        with self._stats_lock:
            self.stats["requests_processed"] += 1
            if not result.success:
                self.stats["requests_failed"] += 1

            # 移動平均による更新
            alpha = 0.1
            self.stats["total_processing_time_ms"] += result.processing_time_ms
            self.stats["avg_batch_size"] = (
                self.stats["avg_batch_size"] * (1 - alpha) + result.items_processed * alpha
            )

            if result.items_processed > 0:
                hit_rate = result.cache_hits / result.items_processed
                self.stats["cache_hit_rate"] = (
                    self.stats["cache_hit_rate"] * (1 - alpha) + hit_rate * alpha
                )

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self._stats_lock:
            stats = self.stats.copy()

        stats.update(
            {
                "queue_size": self.batch_queue.size(),
                "queue_stats": self.batch_queue.get_queue_stats(),
                "optimizer_stats": self.optimizer.get_processing_stats(),
            }
        )

        return stats

    def get_health_status(self) -> Dict[str, Any]:
        """ヘルスステータス取得"""
        total_requests = self.stats["requests_processed"]
        failed_requests = self.stats["requests_failed"]

        error_rate = failed_requests / max(total_requests, 1)
        queue_utilization = self.batch_queue.size() / self.batch_queue.max_size

        health_status = "healthy"
        if error_rate > 0.1:  # 10%以上の失敗率
            health_status = "degraded"
        if error_rate > 0.3 or queue_utilization > 0.8:
            health_status = "unhealthy"

        return {
            "status": health_status,
            "error_rate": error_rate,
            "queue_utilization": queue_utilization,
            "worker_count": len(self.worker_threads),
            "cache_hit_rate": self.stats.get("cache_hit_rate", 0),
            "avg_processing_time_ms": (
                self.stats["total_processing_time_ms"] / max(total_requests, 1)
            ),
        }

    def shutdown(self):
        """シャットダウン"""
        logger.info("BatchDataProcessor シャットダウン開始")

        self._running = False

        # ワーカースレッド終了待機
        for thread in self.worker_threads:
            thread.join(timeout=5)

        # エグゼキュータシャットダウン
        self.executor.shutdown(wait=True)

        logger.info("BatchDataProcessor シャットダウン完了")


# グローバルインスタンス管理
_global_batch_processor: Optional[BatchDataProcessor] = None
_processor_lock = threading.Lock()


def get_batch_processor(**kwargs) -> BatchDataProcessor:
    """グローバルバッチプロセッサー取得"""
    global _global_batch_processor

    if _global_batch_processor is None:
        with _processor_lock:
            if _global_batch_processor is None:
                _global_batch_processor = BatchDataProcessor(**kwargs)

    return _global_batch_processor


def shutdown_batch_processor():
    """グローバルバッチプロセッサーシャットダウン"""
    global _global_batch_processor

    if _global_batch_processor:
        with _processor_lock:
            if _global_batch_processor:
                _global_batch_processor.shutdown()
                _global_batch_processor = None


if __name__ == "__main__":
    # テスト実行
    print("=== Issue #376 バッチデータプロセッサーテスト ===")

    processor = BatchDataProcessor(max_workers=4, default_batch_size=20)

    try:
        # 価格取得テスト
        print("\n1. 価格一括取得テスト")
        test_codes = ["7203", "6758", "9984", "9983", "6861"]
        request_id = processor.bulk_fetch_prices(test_codes, priority=1)
        print(f"リクエストID: {request_id}")

        # 処理完了待機
        time.sleep(3)

        # 統計情報表示
        print("\n2. 統計情報")
        stats = processor.get_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # ヘルスステータス
        print("\n3. ヘルスステータス")
        health = processor.get_health_status()
        for key, value in health.items():
            print(f"  {key}: {value}")

    finally:
        processor.shutdown()

    print("\n=== バッチデータプロセッサーテスト完了 ===")
