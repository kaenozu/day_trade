"""
バルクデータフェッチャー

複数銘柄の並列・一括データ取得機能
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ...utils.logging_config import get_context_logger, log_performance_metric
from .yfinance_fetcher import YFinanceFetcher

logger = get_context_logger(__name__)


class BulkFetcher:
    """バルク（一括）データフェッチャー"""

    def __init__(
        self,
        base_fetcher: YFinanceFetcher = None,
        max_workers: int = 10,
        batch_size: int = 50,
        rate_limit_delay: float = 0.1,
    ):
        """
        初期化

        Args:
            base_fetcher: ベースとなるフェッチャー
            max_workers: 最大並行ワーカー数
            batch_size: バッチサイズ
            rate_limit_delay: レート制限対応の遅延（秒）
        """
        self.base_fetcher = base_fetcher or YFinanceFetcher()
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay

        # 統計情報
        self.bulk_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_symbols": 0,
            "successful_symbols": 0,
            "average_batch_time": 0.0,
            "last_request_time": None,
        }

        logger.info(
            f"BulkFetcher初期化完了: workers={max_workers}, "
            f"batch_size={batch_size}, delay={rate_limit_delay}s"
        )

    def bulk_get_current_prices(
        self,
        codes: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        複数銘柄の現在価格を一括取得

        Args:
            codes: 銘柄コードリスト
            progress_callback: 進捗コールバック関数(processed, total)

        Returns:
            銘柄別価格データ辞書
        """
        start_time = time.time()

        try:
            self.bulk_stats["total_requests"] += 1
            self.bulk_stats["total_symbols"] += len(codes)

            results = {}

            # バッチに分割
            batches = [
                codes[i : i + self.batch_size]
                for i in range(0, len(codes), self.batch_size)
            ]
            processed_count = 0

            logger.info(
                f"価格一括取得開始: {len(codes)}銘柄を{len(batches)}バッチで処理"
            )

            for batch_idx, batch in enumerate(batches):
                batch_start_time = time.time()

                # バッチ内並列処理
                batch_results = self._process_price_batch(batch)
                results.update(batch_results)

                processed_count += len(batch)

                # 進捗通知
                if progress_callback:
                    progress_callback(processed_count, len(codes))

                # レート制限対応
                if batch_idx < len(batches) - 1:  # 最後のバッチ以外
                    time.sleep(self.rate_limit_delay)

                batch_time = time.time() - batch_start_time
                logger.debug(
                    f"バッチ{batch_idx + 1}/{len(batches)}完了: {len(batch)}銘柄, {batch_time:.2f}秒"
                )

            # 統計更新
            self.bulk_stats["successful_symbols"] += len(
                [r for r in results.values() if r]
            )
            self.bulk_stats["successful_requests"] += 1
            self.bulk_stats["last_request_time"] = datetime.now().isoformat()

            total_time = time.time() - start_time
            self.bulk_stats["average_batch_time"] = (
                self.bulk_stats["average_batch_time"]
                * (self.bulk_stats["successful_requests"] - 1)
                + total_time
            ) / self.bulk_stats["successful_requests"]

            successful_count = len([r for r in results.values() if r])

            log_performance_metric(
                "bulk_current_prices",
                {
                    "total_symbols": len(codes),
                    "successful_symbols": successful_count,
                    "processing_time": total_time,
                    "symbols_per_second": (
                        len(codes) / total_time if total_time > 0 else 0
                    ),
                },
            )

            logger.info(
                f"価格一括取得完了: {successful_count}/{len(codes)}銘柄成功, "
                f"{total_time:.2f}秒"
            )

            return results

        except Exception as e:
            self.bulk_stats["failed_requests"] += 1
            logger.error(f"価格一括取得エラー: {e}")
            return {}

    def _process_price_batch(self, batch: List[str]) -> Dict[str, Dict[str, float]]:
        """価格データバッチ処理"""
        results = {}

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(batch))
        ) as executor:
            # 並列タスク投入
            future_to_code = {
                executor.submit(self.base_fetcher.get_current_price, code): code
                for code in batch
            }

            # 結果収集
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    result = future.result(timeout=30)  # 30秒タイムアウト
                    results[code] = result if result else {}
                except Exception as e:
                    logger.warning(f"価格取得失敗: {code} - {e}")
                    results[code] = {}

        return results

    def bulk_get_company_info(
        self,
        codes: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        複数銘柄の企業情報を一括取得

        Args:
            codes: 銘柄コードリスト
            progress_callback: 進捗コールバック関数

        Returns:
            銘柄別企業情報辞書
        """
        start_time = time.time()

        try:
            results = {}
            batches = [
                codes[i : i + self.batch_size]
                for i in range(0, len(codes), self.batch_size)
            ]
            processed_count = 0

            logger.info(
                f"企業情報一括取得開始: {len(codes)}銘柄を{len(batches)}バッチで処理"
            )

            for batch_idx, batch in enumerate(batches):
                batch_results = self._process_company_info_batch(batch)
                results.update(batch_results)

                processed_count += len(batch)

                if progress_callback:
                    progress_callback(processed_count, len(codes))

                # より長い遅延（企業情報取得はより重い処理のため）
                if batch_idx < len(batches) - 1:
                    time.sleep(self.rate_limit_delay * 2)

            successful_count = len([r for r in results.values() if r])
            total_time = time.time() - start_time

            log_performance_metric(
                "bulk_company_info",
                {
                    "total_symbols": len(codes),
                    "successful_symbols": successful_count,
                    "processing_time": total_time,
                },
            )

            logger.info(
                f"企業情報一括取得完了: {successful_count}/{len(codes)}銘柄成功, "
                f"{total_time:.2f}秒"
            )

            return results

        except Exception as e:
            logger.error(f"企業情報一括取得エラー: {e}")
            return {}

    def _process_company_info_batch(
        self, batch: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """企業情報バッチ処理"""
        results = {}

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers // 2, len(batch))
        ) as executor:
            # 企業情報取得は重い処理なので並列数を抑制
            future_to_code = {
                executor.submit(self.base_fetcher.get_company_info, code): code
                for code in batch
            }

            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    result = future.result(
                        timeout=60
                    )  # 60秒タイムアウト（企業情報は時間がかかる）
                    results[code] = result if result else {}
                except Exception as e:
                    logger.warning(f"企業情報取得失敗: {code} - {e}")
                    results[code] = {}

        return results

    def bulk_get_current_prices_optimized(
        self, codes: List[str], chunk_size: int = 20, max_concurrent_chunks: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        最適化された価格一括取得（大量銘柄対応）

        Args:
            codes: 銘柄コードリスト
            chunk_size: チャンクサイズ
            max_concurrent_chunks: 最大同時チャンク数

        Returns:
            銘柄別価格データ辞書
        """
        start_time = time.time()

        try:
            # チャンクに分割
            chunks = [
                codes[i : i + chunk_size] for i in range(0, len(codes), chunk_size)
            ]
            results = {}

            logger.info(
                f"最適化価格取得開始: {len(codes)}銘柄を{len(chunks)}チャンク"
                f"（サイズ{chunk_size}、同時{max_concurrent_chunks}）で処理"
            )

            # セマフォで同時実行数を制御
            semaphore = asyncio.Semaphore(max_concurrent_chunks)

            async def process_chunk_async(
                chunk: List[str],
            ) -> Dict[str, Dict[str, float]]:
                async with semaphore:
                    # 非同期でチャンク処理
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, self._process_price_batch, chunk
                    )

            # 非同期実行
            async def run_all_chunks():
                tasks = [process_chunk_async(chunk) for chunk in chunks]
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

                for chunk_result in chunk_results:
                    if isinstance(chunk_result, dict):
                        results.update(chunk_result)
                    else:
                        logger.error(f"チャンク処理エラー: {chunk_result}")

            # 非同期実行
            try:
                asyncio.run(run_all_chunks())
            except RuntimeError:
                # 既存のループがある場合の処理
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 同期的にフォールバック
                    for chunk in chunks:
                        chunk_result = self._process_price_batch(chunk)
                        results.update(chunk_result)
                else:
                    loop.run_until_complete(run_all_chunks())

            successful_count = len([r for r in results.values() if r])
            total_time = time.time() - start_time

            log_performance_metric(
                "bulk_optimized_prices",
                {
                    "total_symbols": len(codes),
                    "successful_symbols": successful_count,
                    "processing_time": total_time,
                    "symbols_per_second": (
                        len(codes) / total_time if total_time > 0 else 0
                    ),
                    "chunk_size": chunk_size,
                    "concurrent_chunks": max_concurrent_chunks,
                },
            )

            logger.info(
                f"最適化価格取得完了: {successful_count}/{len(codes)}銘柄成功, "
                f"{total_time:.2f}秒 ({len(codes) / total_time:.1f}銘柄/秒)"
            )

            return results

        except Exception as e:
            logger.error(f"最適化価格取得エラー: {e}")
            return {}

    def get_bulk_stats(self) -> Dict[str, Any]:
        """バルク処理統計を取得"""
        stats = self.bulk_stats.copy()

        # 追加の計算済み統計
        if stats["total_symbols"] > 0:
            stats["symbol_success_rate"] = (
                stats["successful_symbols"] / stats["total_symbols"]
            )

        if stats["total_requests"] > 0:
            stats["request_success_rate"] = (
                stats["successful_requests"] / stats["total_requests"]
            )

        stats["configuration"] = {
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "rate_limit_delay": self.rate_limit_delay,
        }

        return stats

    def reset_bulk_stats(self) -> None:
        """バルク処理統計をリセット"""
        self.bulk_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_symbols": 0,
            "successful_symbols": 0,
            "average_batch_time": 0.0,
            "last_request_time": None,
        }
        logger.info("バルク処理統計リセット完了")

    def optimize_batch_settings(self, target_symbols: int) -> Dict[str, Any]:
        """
        対象銘柄数に基づいてバッチ設定を最適化

        Args:
            target_symbols: 処理予定銘柄数

        Returns:
            最適化されたバッチ設定
        """
        try:
            # 基本設定
            optimal_batch_size = self.batch_size
            optimal_workers = self.max_workers
            optimal_delay = self.rate_limit_delay

            # 銘柄数に基づく調整
            if target_symbols > 500:
                # 大量銘柄の場合
                optimal_batch_size = min(100, target_symbols // 10)
                optimal_workers = min(20, self.max_workers * 2)
                optimal_delay = max(0.05, self.rate_limit_delay * 0.5)

            elif target_symbols > 100:
                # 中量銘柄の場合
                optimal_batch_size = min(50, target_symbols // 5)
                optimal_workers = min(15, int(self.max_workers * 1.5))
                optimal_delay = self.rate_limit_delay

            elif target_symbols < 20:
                # 少量銘柄の場合
                optimal_batch_size = target_symbols
                optimal_workers = min(target_symbols, self.max_workers)
                optimal_delay = self.rate_limit_delay * 2

            # 予想処理時間を計算
            estimated_time = (
                (target_symbols / optimal_batch_size)
                * (optimal_batch_size / optimal_workers)
                * (1.0 + optimal_delay)
            )

            recommendations = {
                "current_settings": {
                    "batch_size": self.batch_size,
                    "max_workers": self.max_workers,
                    "rate_limit_delay": self.rate_limit_delay,
                },
                "optimal_settings": {
                    "batch_size": optimal_batch_size,
                    "max_workers": optimal_workers,
                    "rate_limit_delay": optimal_delay,
                },
                "estimated_processing_time": estimated_time,
                "target_symbols": target_symbols,
                "recommendations": [],
            }

            # 推奨事項の生成
            if optimal_batch_size != self.batch_size:
                recommendations["recommendations"].append(
                    f"バッチサイズを{self.batch_size}から{optimal_batch_size}に変更"
                )

            if optimal_workers != self.max_workers:
                recommendations["recommendations"].append(
                    f"最大ワーカー数を{self.max_workers}から{optimal_workers}に変更"
                )

            if abs(optimal_delay - self.rate_limit_delay) > 0.01:
                recommendations["recommendations"].append(
                    f"レート制限遅延を{self.rate_limit_delay}sから{optimal_delay}sに変更"
                )

            return recommendations

        except Exception as e:
            logger.error(f"バッチ設定最適化エラー: {e}")
            return {"error": str(e)}

    def apply_optimized_settings(self, settings: Dict[str, Any]) -> None:
        """最適化されたバッチ設定を適用"""
        try:
            optimal_settings = settings.get("optimal_settings", {})

            if "batch_size" in optimal_settings:
                self.batch_size = optimal_settings["batch_size"]

            if "max_workers" in optimal_settings:
                self.max_workers = optimal_settings["max_workers"]

            if "rate_limit_delay" in optimal_settings:
                self.rate_limit_delay = optimal_settings["rate_limit_delay"]

            logger.info(
                f"バッチ設定更新完了: batch_size={self.batch_size}, "
                f"max_workers={self.max_workers}, delay={self.rate_limit_delay}s"
            )

        except Exception as e:
            logger.error(f"バッチ設定適用エラー: {e}")

    def health_check(self) -> Dict[str, Any]:
        """バルクフェッチャーのヘルスチェック"""
        try:
            stats = self.get_bulk_stats()
            base_health = self.base_fetcher.health_check()

            # バルク固有の健康評価
            bulk_health_score = 100
            bulk_issues = []

            # 成功率チェック
            if stats.get("symbol_success_rate", 1.0) < 0.8:
                bulk_health_score -= 20
                bulk_issues.append(
                    f"低いシンボル成功率: {stats.get('symbol_success_rate', 0):.2%}"
                )

            # リクエスト成功率チェック
            if stats.get("request_success_rate", 1.0) < 0.9:
                bulk_health_score -= 15
                bulk_issues.append(
                    f"低いリクエスト成功率: {stats.get('request_success_rate', 0):.2%}"
                )

            # 処理時間チェック
            if stats.get("average_batch_time", 0) > 60:  # 1分超
                bulk_health_score -= 10
                bulk_issues.append(
                    f"長い平均処理時間: {stats.get('average_batch_time', 0):.1f}秒"
                )

            # 総合評価
            combined_score = (base_health["score"] + bulk_health_score) / 2
            combined_issues = base_health["issues"] + bulk_issues

            health_status = "healthy"
            if combined_score < 60:
                health_status = "critical"
            elif combined_score < 80:
                health_status = "warning"

            return {
                "status": health_status,
                "score": combined_score,
                "issues": combined_issues,
                "bulk_stats": stats,
                "base_fetcher_health": base_health,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"バルクヘルスチェックエラー: {e}")
            return {
                "status": "error",
                "score": 0,
                "issues": [f"ヘルスチェック実行エラー: {e}"],
                "timestamp": time.time(),
            }
