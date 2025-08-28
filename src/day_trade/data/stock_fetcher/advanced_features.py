"""
StockFetcher高度な機能
最適化されたキャッシュ機能、パフォーマンス監視、適応的調整機能
"""

import logging
import os
import time
from typing import Dict, Any, Optional

from ...utils.logging_config import (
    get_context_logger,
    log_performance_metric,
)
from .cache_core import DataCache


class AdvancedStockFetcherMixin:
    """StockFetcher用の高度な機能ミックスイン"""
    
    def __init_advanced_features__(self):
        """高度な機能の初期化"""
        # 適応的キャッシュ調整用の統計とタイマー
        self.cache_adjustment_interval = int(
            os.getenv("CACHE_ADJUSTMENT_INTERVAL", "3600")
        )  # 1時間
        self.last_cache_adjustment = time.time()
        self.auto_cache_tuning_enabled = (
            os.getenv("AUTO_CACHE_TUNING", "true").lower() == "true"
        )
        
        # データキャッシュインスタンス
        self._data_cache = DataCache()

    def _maybe_adjust_cache_settings(self):
        """
        定期的なキャッシュ設定の自動調整チェック
        各API呼び出し時に呼ばれるパフォーマンス最適化機能
        """
        if not self.auto_cache_tuning_enabled:
            return

        current_time = time.time()
        if current_time - self.last_cache_adjustment < self.cache_adjustment_interval:
            return

        try:
            # 現在のキャッシュ統計を取得して調整
            if hasattr(self, "_data_cache") and self._data_cache:
                adjustment_result = self._data_cache.auto_tune_cache_settings()

                if adjustment_result.get("adjusted"):
                    self.logger.info(
                        "Cache settings automatically adjusted",
                        adjustments=adjustment_result.get("adjustments"),
                        expected_improvement=adjustment_result.get(
                            "improvement_expected"
                        ),
                        stats=adjustment_result.get("stats"),
                    )

                self.last_cache_adjustment = current_time

        except Exception as e:
            # キャッシュ調整でエラーが発生しても本来の処理は継続
            self.logger.warning(f"Cache auto-tuning failed: {e}")

    def get_cache_performance_report(self) -> Dict[str, Any]:
        """
        キャッシュパフォーマンスレポートを取得

        Returns:
            キャッシュ統計とパフォーマンス指標
        """
        report = {
            "cache_enabled": hasattr(self, "_data_cache"),
            "auto_tuning_enabled": self.auto_cache_tuning_enabled,
            "last_adjustment": self.last_cache_adjustment,
            "adjustment_interval": self.cache_adjustment_interval,
        }

        if hasattr(self, "_data_cache") and self._data_cache:
            cache_stats = self._data_cache.get_cache_stats()
            cache_info = self._data_cache.get_cache_info()
            optimization_suggestions = self._data_cache.optimize_cache_settings()

            report.update(
                {
                    "cache_stats": cache_stats,
                    "cache_info": cache_info,
                    "optimization_suggestions": optimization_suggestions,
                    "performance_metrics": {
                        "effective_hit_rate": cache_stats.get("hit_rate", 0)
                        + cache_stats.get("stale_hit_rate", 0) * 0.7,  # stale hitは70%の価値
                        "memory_efficiency": cache_stats.get("cache_utilization", 0),
                        "eviction_pressure": cache_stats.get("eviction_count", 0)
                        / max(cache_stats.get("hit_count", 1), 1),
                    },
                }
            )

        return report

    def _record_cache_performance(
        self, operation: str, execution_time_ms: float, cache_hit: bool = False
    ):
        """
        キャッシュパフォーマンスメトリクスを記録
        
        Args:
            operation: 実行した操作名
            execution_time_ms: 実行時間（ミリ秒）
            cache_hit: キャッシュヒットかどうか
        """
        try:
            log_performance_metric(
                f"cache_{operation}",
                {
                    "execution_time_ms": execution_time_ms,
                    "cache_hit": cache_hit,
                    "operation": operation,
                },
            )
            
            # 適応的キャッシュ調整のチェック
            self._maybe_adjust_cache_settings()
            
        except Exception as e:
            # パフォーマンス記録のエラーは無視（本来の処理に影響しない）
            if hasattr(self, "logger"):
                self.logger.debug(f"Performance recording failed: {e}")

    def clear_all_performance_caches(self) -> None:
        """すべてのパフォーマンスキャッシュをクリア"""
        try:
            # メインキャッシュをクリア
            if hasattr(self, "_data_cache") and self._data_cache:
                self._data_cache.clear()
            
            # LRUキャッシュもクリア
            if hasattr(self, "_get_ticker"):
                self._get_ticker.cache_clear()

            # メソッドレベルのキャッシュもクリア
            cache_methods = [
                "get_current_price",
                "get_historical_data", 
                "get_historical_data_range",
                "get_company_info"
            ]
            
            for method_name in cache_methods:
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    if hasattr(method, "clear_cache"):
                        method.clear_cache()

            if hasattr(self, "logger"):
                self.logger.info("すべてのパフォーマンスキャッシュをクリアしました")
                
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"キャッシュクリア中にエラーが発生: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        パフォーマンス統計のサマリーを取得
        
        Returns:
            パフォーマンス統計情報
        """
        summary = {
            "timestamp": time.time(),
            "cache_report": self.get_cache_performance_report(),
        }
        
        # リトライ統計があれば追加
        if hasattr(self, "get_retry_stats"):
            summary["retry_stats"] = self.get_retry_stats()
            
        return summary