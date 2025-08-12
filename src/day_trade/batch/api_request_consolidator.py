#!/usr/bin/env python3
"""
APIリクエスト統合システム
複数のAPIリクエストを効率的にバッチ処理するための統合システム

主要機能:
- APIリクエスト統合とバッチ処理
- 適応的バッチサイズ調整
- レート制限対応
- 失敗時の自動リトライ
- リクエスト優先度管理
"""

import asyncio
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..cache.enhanced_persistent_cache import EnhancedPersistentCache
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class RequestPriority(Enum):
    """リクエスト優先度"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    REALTIME = 5


@dataclass
class APIRequest:
    """APIリクエスト定義"""

    request_id: str
    endpoint: str
    symbols: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: RequestPriority = RequestPriority.NORMAL
    timeout: float = 30.0
    retry_count: int = 3
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        """優先度による比較（PriorityQueueで使用）"""
        return self.priority.value > other.priority.value


@dataclass
class BatchRequest:
    """バッチリクエスト"""

    batch_id: str
    requests: List[APIRequest]
    consolidated_symbols: List[str]
    batch_size: int
    estimated_time: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class APIResponse:
    """APIレスポンス"""

    request_id: str
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    response_time: float = 0.0
    symbols_processed: List[str] = field(default_factory=list)
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationStats:
    """統合統計"""

    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    consolidation_ratio: float = 0.0  # リクエスト数削減率
    total_processing_time: float = 0.0
    success_rate: float = 0.0
    cache_hit_rate: float = 0.0
    avg_response_time: float = 0.0
    adaptive_adjustments: int = 0


class APIRequestConsolidator:
    """
    APIリクエスト統合システム

    複数のAPIリクエストを効率的にバッチ処理し、
    外部API呼び出し数を最適化する
    """

    def __init__(
        self,
        base_batch_size: int = 50,
        max_batch_size: int = 200,
        min_batch_size: int = 10,
        batch_timeout: float = 2.0,
        max_workers: int = 6,
        enable_adaptive_sizing: bool = True,
        enable_caching: bool = True,
        rate_limit_calls_per_minute: int = 300,
        cache_ttl_seconds: int = 300,
    ):
        self.base_batch_size = base_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.batch_timeout = batch_timeout
        self.max_workers = max_workers
        self.enable_adaptive_sizing = enable_adaptive_sizing
        self.enable_caching = enable_caching
        self.rate_limit_calls_per_minute = rate_limit_calls_per_minute

        # 内部状態
        self.request_queue = PriorityQueue()
        self.pending_requests = {}  # request_id -> APIRequest
        self.active_batches = {}  # batch_id -> BatchRequest
        self.response_callbacks = {}  # request_id -> callback

        # パフォーマンス統計
        self.stats = ConsolidationStats()
        self.response_times_history = deque(maxlen=100)
        self.success_rates_history = deque(maxlen=50)
        self.batch_sizes_history = deque(maxlen=100)

        # レート制限管理
        self.api_call_times = deque(maxlen=rate_limit_calls_per_minute)
        self.rate_limit_lock = threading.Lock()

        # 適応的サイズ調整
        self.current_batch_size = base_batch_size
        self.last_adjustment_time = time.time()
        self.adjustment_cooldown = 30.0  # 30秒間隔

        # キャッシュシステム
        self.cache = None
        if enable_caching:
            try:
                self.cache = EnhancedPersistentCache(
                    cache_name="api_request_consolidator",
                    max_size=10000,
                    ttl_seconds=cache_ttl_seconds,
                )
                logger.info("APIリクエストキャッシュ有効化")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")

        # バックグラウンド処理
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_processor_thread = None
        self.running = False

        logger.info(
            f"APIリクエスト統合システム初期化完了 - "
            f"batch_size={base_batch_size}, workers={max_workers}, "
            f"caching={enable_caching}, adaptive={enable_adaptive_sizing}"
        )

    def start(self):
        """バックグラウンド処理開始"""
        if self.running:
            return

        self.running = True
        self.batch_processor_thread = threading.Thread(
            target=self._batch_processor_loop, daemon=True
        )
        self.batch_processor_thread.start()
        logger.info("APIリクエスト統合システム開始")

    def stop(self):
        """システム停止"""
        self.running = False
        if self.batch_processor_thread:
            self.batch_processor_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        if self.cache:
            self.cache.close()
        logger.info("APIリクエスト統合システム停止")

    def submit_request(
        self,
        endpoint: str,
        symbols: List[str],
        parameters: Dict[str, Any] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        APIリクエスト投入

        Args:
            endpoint: API エンドポイント
            symbols: 銘柄コードリスト
            parameters: リクエストパラメータ
            priority: 優先度
            timeout: タイムアウト
            callback: 完了時コールバック

        Returns:
            リクエストID
        """
        request_id = f"req_{int(time.time() * 1000)}_{len(self.pending_requests)}"

        # キャッシュチェック
        if self.cache and self._check_cache_for_request(endpoint, symbols, parameters):
            # キャッシュヒット - 即座にレスポンス
            cached_data = self._get_cached_response(endpoint, symbols, parameters)
            response = APIResponse(
                request_id=request_id,
                success=True,
                data=cached_data,
                response_time=0.001,
                symbols_processed=symbols,
                cache_hit=True,
            )

            if callback:
                try:
                    callback(response)
                except Exception as e:
                    logger.error(f"コールバックエラー {request_id}: {e}")

            self.stats.cache_hit_rate = self._update_cache_hit_rate(True)
            return request_id

        # 新しいリクエスト作成
        api_request = APIRequest(
            request_id=request_id,
            endpoint=endpoint,
            symbols=symbols,
            parameters=parameters or {},
            priority=priority,
            timeout=timeout,
            callback=callback,
        )

        # キューに投入
        self.pending_requests[request_id] = api_request
        if callback:
            self.response_callbacks[request_id] = callback

        self.request_queue.put(api_request)
        self.stats.total_requests += 1

        logger.debug(
            f"リクエスト投入: {request_id} - {len(symbols)} symbols, priority={priority.name}"
        )
        return request_id

    def _batch_processor_loop(self):
        """バッチ処理メインループ"""
        logger.info("バッチ処理ループ開始")

        while self.running:
            try:
                # バッチ作成
                batch = self._create_batch()
                if batch:
                    # バッチ処理実行
                    self._process_batch_async(batch)
                else:
                    # バッチが作成されない場合は短時間待機
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"バッチ処理ループエラー: {e}")
                time.sleep(1.0)

        logger.info("バッチ処理ループ終了")

    def _create_batch(self) -> Optional[BatchRequest]:
        """バッチ作成"""
        if self.request_queue.empty():
            return None

        batch_requests = []
        all_symbols = set()
        batch_start_time = time.time()

        # 現在のバッチサイズを取得
        current_size = self._get_adaptive_batch_size()

        # 優先度順にリクエスト取得
        while (
            len(batch_requests) < current_size
            and not self.request_queue.empty()
            and (time.time() - batch_start_time) < self.batch_timeout
        ):
            try:
                request = self.request_queue.get_nowait()
                batch_requests.append(request)
                all_symbols.update(request.symbols)
            except:
                break

        if not batch_requests:
            return None

        # 重複銘柄削除
        consolidated_symbols = list(all_symbols)

        batch_id = f"batch_{int(time.time() * 1000)}_{len(self.active_batches)}"
        batch = BatchRequest(
            batch_id=batch_id,
            requests=batch_requests,
            consolidated_symbols=consolidated_symbols,
            batch_size=len(batch_requests),
        )

        self.active_batches[batch_id] = batch
        self.stats.total_batches += 1

        # 統合率計算
        original_symbol_count = sum(len(req.symbols) for req in batch_requests)
        if original_symbol_count > 0:
            consolidation_ratio = 1.0 - (
                len(consolidated_symbols) / original_symbol_count
            )
            self.stats.consolidation_ratio = (
                self.stats.consolidation_ratio * (self.stats.total_batches - 1)
                + consolidation_ratio
            ) / self.stats.total_batches

        logger.debug(
            f"バッチ作成: {batch_id} - {len(batch_requests)} requests, "
            f"{len(consolidated_symbols)} unique symbols"
        )

        return batch

    def _process_batch_async(self, batch: BatchRequest):
        """バッチ処理非同期実行"""
        future = self.executor.submit(self._process_batch, batch)
        future.add_done_callback(lambda f: self._handle_batch_completion(batch, f))

    def _process_batch(self, batch: BatchRequest) -> Dict[str, APIResponse]:
        """バッチ処理実行"""
        start_time = time.time()
        responses = {}

        try:
            # レート制限チェック
            self._wait_for_rate_limit()

            # 統合APIリクエスト実行（実際のAPI呼び出しは各リクエストのエンドポイントに依存）
            batch_data = self._execute_consolidated_api_call(batch)

            # レスポンス時間記録
            response_time = time.time() - start_time
            self.response_times_history.append(response_time)

            # 各リクエストへのレスポンス分割
            for request in batch.requests:
                relevant_data = self._extract_relevant_data(batch_data, request)

                response = APIResponse(
                    request_id=request.request_id,
                    success=relevant_data is not None,
                    data=relevant_data,
                    response_time=response_time,
                    symbols_processed=request.symbols,
                    cache_hit=False,
                )

                responses[request.request_id] = response

                # キャッシュに保存
                if self.cache and response.success:
                    self._cache_response(request, relevant_data)

            # 成功率更新
            success_count = sum(1 for r in responses.values() if r.success)
            batch_success_rate = success_count / len(responses) if responses else 0.0
            self.success_rates_history.append(batch_success_rate)

            logger.info(
                f"バッチ処理完了: {batch.batch_id} - {len(responses)} responses, "
                f"success_rate={batch_success_rate:.1%}, time={response_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"バッチ処理エラー {batch.batch_id}: {e}")

            # エラーレスポンス作成
            for request in batch.requests:
                responses[request.request_id] = APIResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message=str(e),
                    response_time=time.time() - start_time,
                )

        return responses

    def _execute_consolidated_api_call(self, batch: BatchRequest) -> Dict[str, Any]:
        """統合APIコール実行（ダミー実装）"""
        # 実際の実装では、各エンドポイントに応じたAPI呼び出しを行う
        # ここではシミュレーション
        time.sleep(0.1)  # API呼び出しシミュレーション

        # API呼び出し時刻記録（レート制限用）
        with self.rate_limit_lock:
            self.api_call_times.append(time.time())

        # ダミーデータ生成
        return {
            symbol: {
                "price": 100.0 + hash(symbol) % 1000,
                "volume": 1000000 + hash(symbol) % 500000,
                "timestamp": time.time(),
            }
            for symbol in batch.consolidated_symbols
        }

    def _extract_relevant_data(
        self, batch_data: Dict[str, Any], request: APIRequest
    ) -> Any:
        """リクエストに関連するデータ抽出"""
        if not batch_data:
            return None

        # リクエストの銘柄に関連するデータのみ抽出
        relevant_data = {}
        for symbol in request.symbols:
            if symbol in batch_data:
                relevant_data[symbol] = batch_data[symbol]

        return relevant_data if relevant_data else None

    def _handle_batch_completion(self, batch: BatchRequest, future):
        """バッチ完了処理"""
        try:
            responses = future.result()

            # コールバック実行
            for request_id, response in responses.items():
                if request_id in self.response_callbacks:
                    callback = self.response_callbacks[request_id]
                    try:
                        callback(response)
                    except Exception as e:
                        logger.error(f"コールバックエラー {request_id}: {e}")

                    # クリーンアップ
                    del self.response_callbacks[request_id]

                if request_id in self.pending_requests:
                    del self.pending_requests[request_id]

            # バッチクリーンアップ
            if batch.batch_id in self.active_batches:
                del self.active_batches[batch.batch_id]

            # 適応的サイズ調整
            if self.enable_adaptive_sizing:
                self._adjust_batch_size_adaptive(responses)

            # 統計更新
            self._update_stats(batch, responses)

        except Exception as e:
            logger.error(f"バッチ完了処理エラー {batch.batch_id}: {e}")

    def _get_adaptive_batch_size(self) -> int:
        """適応的バッチサイズ取得"""
        if not self.enable_adaptive_sizing:
            return self.current_batch_size

        # 最近のパフォーマンスに基づいてサイズ調整
        if len(self.response_times_history) < 10:
            return self.current_batch_size

        recent_avg_time = statistics.mean(list(self.response_times_history)[-10:])
        recent_success_rate = (
            statistics.mean(list(self.success_rates_history)[-5:])
            if self.success_rates_history
            else 1.0
        )

        # 調整ロジック
        if recent_avg_time > 5.0 and recent_success_rate > 0.95:
            # レスポンス時間が長い場合はサイズ縮小
            target_size = max(self.min_batch_size, self.current_batch_size - 10)
        elif recent_avg_time < 2.0 and recent_success_rate > 0.98:
            # レスポンス時間が短く成功率が高い場合はサイズ拡大
            target_size = min(self.max_batch_size, self.current_batch_size + 5)
        else:
            target_size = self.current_batch_size

        return target_size

    def _adjust_batch_size_adaptive(self, responses: Dict[str, APIResponse]):
        """適応的バッチサイズ調整"""
        if time.time() - self.last_adjustment_time < self.adjustment_cooldown:
            return

        if not responses:
            return

        avg_response_time = statistics.mean(
            [r.response_time for r in responses.values()]
        )
        success_rate = sum(1 for r in responses.values() if r.success) / len(responses)

        old_size = self.current_batch_size

        # 調整ロジック
        if avg_response_time > 3.0 or success_rate < 0.9:
            # パフォーマンス悪化時はサイズ縮小
            self.current_batch_size = max(
                self.min_batch_size, self.current_batch_size - 5
            )
        elif avg_response_time < 1.0 and success_rate > 0.95:
            # パフォーマンス良好時はサイズ拡大
            self.current_batch_size = min(
                self.max_batch_size, self.current_batch_size + 3
            )

        if self.current_batch_size != old_size:
            self.stats.adaptive_adjustments += 1
            self.last_adjustment_time = time.time()
            logger.info(
                f"適応的バッチサイズ調整: {old_size} -> {self.current_batch_size}"
            )

    def _wait_for_rate_limit(self):
        """レート制限待機"""
        if not self.api_call_times:
            return

        with self.rate_limit_lock:
            current_time = time.time()

            # 古いレコード削除
            while self.api_call_times and current_time - self.api_call_times[0] > 60.0:
                self.api_call_times.popleft()

            # レート制限チェック
            if len(self.api_call_times) >= self.rate_limit_calls_per_minute:
                oldest_call = self.api_call_times[0]
                wait_time = 60.0 - (current_time - oldest_call)
                if wait_time > 0:
                    logger.info(f"レート制限待機: {wait_time:.1f}秒")
                    time.sleep(wait_time)

    def _check_cache_for_request(
        self, endpoint: str, symbols: List[str], parameters: Dict[str, Any]
    ) -> bool:
        """キャッシュチェック"""
        if not self.cache:
            return False

        cache_key = self._generate_cache_key(endpoint, symbols, parameters)
        return self.cache.exists(cache_key)

    def _get_cached_response(
        self, endpoint: str, symbols: List[str], parameters: Dict[str, Any]
    ) -> Any:
        """キャッシュからレスポンス取得"""
        if not self.cache:
            return None

        cache_key = self._generate_cache_key(endpoint, symbols, parameters)
        return self.cache.get(cache_key)

    def _cache_response(self, request: APIRequest, data: Any):
        """レスポンスキャッシュ"""
        if not self.cache or not data:
            return

        cache_key = self._generate_cache_key(
            request.endpoint, request.symbols, request.parameters
        )
        self.cache.put(cache_key, data)

    def _generate_cache_key(
        self, endpoint: str, symbols: List[str], parameters: Dict[str, Any]
    ) -> str:
        """キャッシュキー生成"""
        symbols_str = "|".join(sorted(symbols))
        params_str = "|".join(f"{k}={v}" for k, v in sorted(parameters.items()))
        return f"{endpoint}:{symbols_str}:{params_str}"

    def _update_cache_hit_rate(self, cache_hit: bool) -> float:
        """キャッシュヒット率更新"""
        # 簡略化された実装
        return self.stats.cache_hit_rate

    def _update_stats(self, batch: BatchRequest, responses: Dict[str, APIResponse]):
        """統計更新"""
        self.batch_sizes_history.append(batch.batch_size)

        # 平均値更新
        if self.batch_sizes_history:
            self.stats.avg_batch_size = statistics.mean(self.batch_sizes_history)

        if responses:
            success_count = sum(1 for r in responses.values() if r.success)
            batch_success_rate = success_count / len(responses)

            # 全体成功率更新
            total_processed = self.stats.total_requests
            if total_processed > 0:
                current_successes = self.stats.success_rate * (
                    total_processed - len(responses)
                )
                self.stats.success_rate = (
                    current_successes + success_count
                ) / total_processed

            # 平均レスポンス時間更新
            if self.response_times_history:
                self.stats.avg_response_time = statistics.mean(
                    self.response_times_history
                )

    def get_stats(self) -> ConsolidationStats:
        """統計取得"""
        # 現在の状態を反映
        self.stats.total_requests = (
            len(self.pending_requests) + self.stats.total_requests
        )
        return self.stats

    def get_status(self) -> Dict[str, Any]:
        """システムステータス取得"""
        return {
            "running": self.running,
            "pending_requests": len(self.pending_requests),
            "active_batches": len(self.active_batches),
            "current_batch_size": self.current_batch_size,
            "queue_size": self.request_queue.qsize(),
            "stats": self.get_stats().__dict__,
            "recent_performance": {
                "avg_response_time": (
                    statistics.mean(self.response_times_history)
                    if self.response_times_history
                    else 0.0
                ),
                "avg_success_rate": (
                    statistics.mean(self.success_rates_history)
                    if self.success_rates_history
                    else 0.0
                ),
                "avg_batch_size": (
                    statistics.mean(self.batch_sizes_history)
                    if self.batch_sizes_history
                    else 0.0
                ),
            },
        }


# 便利関数
def create_consolidator(
    batch_size: int = 50, max_workers: int = 6, enable_caching: bool = True, **kwargs
) -> APIRequestConsolidator:
    """統合システム作成ヘルパー"""
    return APIRequestConsolidator(
        base_batch_size=batch_size,
        max_workers=max_workers,
        enable_caching=enable_caching,
        **kwargs,
    )


if __name__ == "__main__":
    # テスト実行
    print("=== APIリクエスト統合システム テスト ===")

    consolidator = create_consolidator(batch_size=20, max_workers=4)
    consolidator.start()

    # テストリクエスト投入
    test_symbols = [
        ["7203", "8306", "9984"],  # リクエスト1
        ["6758", "4689", "2914"],  # リクエスト2
        ["8035", "7203", "8306"],  # リクエスト3（重複あり）
        ["1234", "5678", "9999"],  # リクエスト4
    ]

    responses_received = []

    def response_callback(response: APIResponse):
        responses_received.append(response)
        print(
            f"Response: {response.request_id} - Success: {response.success}, "
            f"Symbols: {len(response.symbols_processed)}, Cache: {response.cache_hit}"
        )

    # リクエスト投入
    request_ids = []
    for i, symbols in enumerate(test_symbols):
        request_id = consolidator.submit_request(
            endpoint="stock_data",
            symbols=symbols,
            priority=RequestPriority.HIGH if i % 2 == 0 else RequestPriority.NORMAL,
            callback=response_callback,
        )
        request_ids.append(request_id)
        print(f"Submitted: {request_id} with {len(symbols)} symbols")

    # 処理完了待機
    print("\nProcessing...")
    time.sleep(5.0)

    # 結果表示
    print(f"\nReceived {len(responses_received)} responses")

    stats = consolidator.get_stats()
    print("\nStats:")
    print(f"  Total Requests: {stats.total_requests}")
    print(f"  Total Batches: {stats.total_batches}")
    print(f"  Avg Batch Size: {stats.avg_batch_size:.1f}")
    print(f"  Consolidation Ratio: {stats.consolidation_ratio:.1%}")
    print(f"  Success Rate: {stats.success_rate:.1%}")
    print(f"  Avg Response Time: {stats.avg_response_time:.3f}s")

    status = consolidator.get_status()
    print(f"\nStatus: {status}")

    # クリーンアップ
    consolidator.stop()
    print("\n=== テスト完了 ===")
