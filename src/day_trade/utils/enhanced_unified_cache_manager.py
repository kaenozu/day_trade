#!/usr/bin/env python3
"""
Enhanced Unified Cache Manager with L4 Archive Layer
Issue #377: Advanced Caching Strategy Extension

4層キャッシュ（L1/L2/L3/L4）+ 予測的プリワーミング機能
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    from ..utils.logging_config import get_context_logger
    from .advanced_cache_layers import L4ArchiveCache
    from .unified_cache_manager import (
        CacheEntry,
        CacheStats,
        L1HotCache,
        L2WarmCache,
        L3ColdCache,
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


@dataclass
class ExtendedCacheStats:
    """拡張キャッシュ統計"""

    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    l4_hits: int = 0  # L4追加
    misses: int = 0
    evictions: int = 0
    prewarming_hits: int = 0  # プリワーミング統計
    prewarming_accuracy: float = 0.0
    total_compression_savings_mb: float = 0.0  # 圧縮による節約
    memory_usage_mb: float = 0
    disk_usage_mb: float = 0
    archive_usage_gb: float = 0  # L4使用量
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0


class PredictiveWarming:
    """予測的キャッシュウォーミング"""

    def __init__(self, cache_manager, prediction_window_hours: int = 24):
        self.cache_manager = cache_manager
        self.prediction_window_hours = prediction_window_hours
        self.access_history = deque(maxlen=10000)  # (timestamp, key, layer)
        self.predictions = {}
        self.prediction_accuracy = deque(maxlen=1000)
        self.enabled = True
        self.last_analysis = 0
        self.analysis_interval = 3600  # 1時間毎に分析

        # バックグラウンド予測スレッド
        self.prediction_thread = threading.Thread(
            target=self._prediction_worker, daemon=True
        )
        self.prediction_thread.start()

        logger.info(f"予測的ウォーミング初期化: {prediction_window_hours}時間窓")

    def record_access(self, key: str, layer: str):
        """アクセス記録"""
        if self.enabled:
            self.access_history.append((time.time(), key, layer))

    def _prediction_worker(self):
        """予測ワーカー（バックグラウンド）"""
        while True:
            try:
                if time.time() - self.last_analysis > self.analysis_interval:
                    self._analyze_and_predict()
                    self.last_analysis = time.time()

                time.sleep(60)  # 1分間隔でチェック

            except Exception as e:
                logger.error(f"予測ワーカーエラー: {e}")
                time.sleep(300)  # エラー時は5分待機

    def _analyze_and_predict(self):
        """アクセスパターン分析と予測"""
        if len(self.access_history) < 100:
            return

        try:
            current_time = time.time()

            # 過去24時間のアクセスパターン分析
            recent_accesses = [
                (timestamp, key, layer)
                for timestamp, key, layer in self.access_history
                if current_time - timestamp < self.prediction_window_hours * 3600
            ]

            if not recent_accesses:
                return

            # 時間別アクセス頻度分析
            hourly_patterns = {}
            for timestamp, key, layer in recent_accesses:
                hour = int((timestamp % 86400) / 3600)  # 時間（0-23）
                if hour not in hourly_patterns:
                    hourly_patterns[hour] = {}
                if key not in hourly_patterns[hour]:
                    hourly_patterns[hour][key] = 0
                hourly_patterns[hour][key] += 1

            # 現在時刻の予測
            current_hour = int((current_time % 86400) / 3600)
            next_hour = (current_hour + 1) % 24

            # 次の1時間で必要になりそうなキーを予測
            predictions = []
            if next_hour in hourly_patterns:
                for key, frequency in hourly_patterns[next_hour].items():
                    if frequency > 1:  # 複数回アクセスされているキー
                        confidence = min(frequency / 10.0, 1.0)  # 信頼度計算
                        predictions.append((key, confidence))

            # 予測に基づくプリワーミング
            predictions.sort(key=lambda x: x[1], reverse=True)
            for key, confidence in predictions[:20]:  # 上位20件
                if confidence > 0.5:  # 信頼度50%以上
                    self._prewarm_key(key, confidence)

            logger.debug(
                f"予測完了: {len(predictions)}件, 高信頼度: {sum(1 for _, c in predictions if c > 0.5)}件"
            )

        except Exception as e:
            logger.error(f"予測分析エラー: {e}")

    def _prewarm_key(self, key: str, confidence: float):
        """キーのプリワーミング"""
        try:
            # L3またはL4からL1/L2への昇格を試行
            entry = self.cache_manager.l3_cache.get(key)
            if not entry and hasattr(self.cache_manager, "l4_cache"):
                entry = self.cache_manager.l4_cache.get(key)

            if entry:
                # 信頼度に応じてレイヤー選択
                if confidence > 0.8:
                    self.cache_manager.l1_cache.put(key, entry)
                elif confidence > 0.6:
                    self.cache_manager.l2_cache.put(key, entry)

                logger.debug(f"プリワーミング成功: {key}, 信頼度={confidence:.2f}")

        except Exception as e:
            logger.debug(f"プリワーミングエラー ({key}): {e}")


class EnhancedUnifiedCacheManager(UnifiedCacheManager):
    """L4アーカイブレイヤー付き統合キャッシュマネージャー"""

    def __init__(
        self,
        # 既存パラメータ
        l1_memory_mb: int = 64,
        l2_memory_mb: int = 256,
        l3_disk_mb: int = 1024,
        l1_ttl: int = 30,
        l2_ttl: int = 300,
        l3_ttl: int = 86400,
        cache_db_path: str = "data/unified_cache.db",
        # L4パラメータ
        l4_archive_gb: int = 10,
        l4_ttl_days: int = 365,
        l4_compression: str = "auto",
        # プリワーミング設定
        enable_prewarming: bool = True,
        prewarming_window_hours: int = 24,
    ):
        """拡張統合キャッシュマネージャー初期化"""

        # 基底クラス初期化
        super().__init__(
            l1_memory_mb,
            l2_memory_mb,
            l3_disk_mb,
            l1_ttl,
            l2_ttl,
            l3_ttl,
            cache_db_path,
        )

        # L4アーカイブキャッシュ追加
        self.l4_cache = L4ArchiveCache(
            db_path=cache_db_path.replace(".db", "_archive.db"),
            max_size_gb=l4_archive_gb,
            ttl_days=l4_ttl_days,
            compression_algorithm=l4_compression,
        )

        # 拡張統計
        self.extended_stats = ExtendedCacheStats()

        # 予測的ウォーミング
        if enable_prewarming:
            self.predictive_warming = PredictiveWarming(self, prewarming_window_hours)
        else:
            self.predictive_warming = None

        logger.info(
            f"拡張統合キャッシュマネージャー初期化完了 "
            f"(L1:{l1_memory_mb}MB, L2:{l2_memory_mb}MB, L3:{l3_disk_mb}MB, L4:{l4_archive_gb}GB)"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """4層階層キャッシュからデータ取得"""
        start_time = time.time()
        found_layer = None

        try:
            # L1 (Hot) キャッシュ
            entry = self.l1_cache.get(key)
            if entry:
                self.extended_stats.l1_hits += 1
                found_layer = "L1"
                if self.predictive_warming:
                    self.predictive_warming.record_access(key, "L1")
                self._record_access_time(start_time)
                return entry.value

            # L2 (Warm) キャッシュ
            entry = self.l2_cache.get(key)
            if entry:
                self.extended_stats.l2_hits += 1
                found_layer = "L2"
                # L1に昇格
                self.l1_cache.put(key, entry)
                if self.predictive_warming:
                    self.predictive_warming.record_access(key, "L2")
                self._record_access_time(start_time)
                return entry.value

            # L3 (Cold) キャッシュ
            entry = self.l3_cache.get(key)
            if entry:
                self.extended_stats.l3_hits += 1
                found_layer = "L3"
                # L2に昇格 (高頻度アクセスならL1にも)
                self.l2_cache.put(key, entry)
                if entry.access_count > 5:
                    self.l1_cache.put(key, entry)
                if self.predictive_warming:
                    self.predictive_warming.record_access(key, "L3")
                self._record_access_time(start_time)
                return entry.value

            # L4 (Archive) キャッシュ
            entry = self.l4_cache.get(key)
            if entry:
                self.extended_stats.l4_hits += 1
                found_layer = "L4"
                # アクセス頻度に応じて上位レイヤーに昇格
                if entry.access_count > 10:
                    self.l2_cache.put(key, entry)
                    if entry.access_count > 20:
                        self.l1_cache.put(key, entry)
                elif entry.priority > 3.0:
                    self.l3_cache.put(key, entry)

                if self.predictive_warming:
                    self.predictive_warming.record_access(key, "L4")
                self._record_access_time(start_time)
                return entry.value

            # 全レイヤーでミス
            self.extended_stats.misses += 1
            self._record_access_time(start_time)
            return default

        except Exception as e:
            logger.error(f"拡張キャッシュ取得エラー: {e}")
            self.extended_stats.misses += 1
            self._record_access_time(start_time)
            return default

    def put(
        self, key: str, value: Any, priority: float = 1.0, target_layer: str = "auto"
    ) -> bool:
        """4層階層キャッシュにデータ保存"""
        try:
            # エントリ作成
            import pickle

            current_time = time.time()
            value_size = len(pickle.dumps(value))

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                size_bytes=value_size,
                priority=priority,
            )

            # 自動レイヤー選択の拡張
            if target_layer == "auto":
                target_layer = self._select_optimal_layer_enhanced(entry)

            success = False

            # レイヤー別保存ロジック
            if target_layer in ["l1", "L1", "hot"]:
                success = self.l1_cache.put(key, entry)

            elif target_layer in ["l2", "L2", "warm"]:
                success = self.l2_cache.put(key, entry)
                # 小さくて重要なデータはL1にも
                if value_size < 10 * 1024 and priority > 5.0:
                    self.l1_cache.put(key, entry)

            elif target_layer in ["l3", "L3", "cold"]:
                success = self.l3_cache.put(key, entry)
                # 中程度の重要度ならL2にも
                if priority > 2.0:
                    self.l2_cache.put(key, entry)

            else:  # L4 archive
                success = self.l4_cache.put(key, entry)
                # 重要度に応じて上位レイヤーにも保存
                if priority > 3.0:
                    self.l3_cache.put(key, entry)
                if priority > 6.0:
                    self.l2_cache.put(key, entry)

            return success

        except Exception as e:
            logger.error(f"拡張キャッシュ保存エラー: {e}")
            return False

    def _select_optimal_layer_enhanced(self, entry: CacheEntry) -> str:
        """拡張最適レイヤー選択"""
        size_kb = entry.size_bytes / 1024
        priority = entry.priority

        # 超高速アクセス要求（小サイズ・超高優先度） → L1
        if size_kb < 5 and priority > 8.0:
            return "l1"

        # 高速アクセス要求（小〜中サイズ・高優先度） → L2
        elif size_kb < 50 and priority > 5.0:
            return "l2"

        # 中程度アクセス（中サイズ・中優先度） → L3
        elif size_kb < 500 and priority > 2.0:
            return "l3"

        # 低頻度・大容量・長期保存 → L4
        else:
            return "l4"

    def delete(self, key: str) -> bool:
        """全レイヤーから削除"""
        success = False
        success |= self.l1_cache.delete(key)
        success |= self.l2_cache.delete(key)
        success |= self.l3_cache.delete(key)
        success |= self.l4_cache.delete(key)
        return success

    def clear_all(self):
        """全キャッシュクリア（L4含む）"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()
        self.l4_cache.clear()
        self.extended_stats = ExtendedCacheStats()
        logger.info("全キャッシュクリア完了（L4含む）")

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """拡張統計情報取得"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()
        l4_stats = self.l4_cache.get_stats()

        total_hits = (
            self.extended_stats.l1_hits
            + self.extended_stats.l2_hits
            + self.extended_stats.l3_hits
            + self.extended_stats.l4_hits
        )
        total_requests = total_hits + self.extended_stats.misses

        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        avg_access_time = np.mean(self.access_times) if self.access_times else 0

        # 予測統計
        prediction_stats = {}
        if self.predictive_warming:
            prediction_stats = {
                "prewarming_accuracy": (
                    np.mean(self.predictive_warming.prediction_accuracy)
                    if self.predictive_warming.prediction_accuracy
                    else 0.0
                ),
                "access_history_size": len(self.predictive_warming.access_history),
            }

        return {
            "overall": {
                "hit_rate": overall_hit_rate,
                "total_requests": total_requests,
                "avg_access_time_ms": avg_access_time,
                "l1_hit_ratio": self.extended_stats.l1_hits / total_requests
                if total_requests > 0
                else 0,
                "l2_hit_ratio": self.extended_stats.l2_hits / total_requests
                if total_requests > 0
                else 0,
                "l3_hit_ratio": self.extended_stats.l3_hits / total_requests
                if total_requests > 0
                else 0,
                "l4_hit_ratio": self.extended_stats.l4_hits / total_requests
                if total_requests > 0
                else 0,
            },
            "layers": {"L1": l1_stats, "L2": l2_stats, "L3": l3_stats, "L4": l4_stats},
            "memory_usage_total_mb": (
                l1_stats.get("memory_usage_mb", 0) + l2_stats.get("memory_usage_mb", 0)
            ),
            "disk_usage_mb": l3_stats.get("disk_usage_mb", 0),
            "archive_usage_gb": l4_stats.get("disk_usage_gb", 0),
            "compression_savings": {
                "avg_compression_ratio": l4_stats.get("avg_compression_ratio", 1.0),
                "compression_time_avg_ms": l4_stats.get("compression_time_avg", 0)
                * 1000,
                "decompression_time_avg_ms": l4_stats.get("decompression_time_avg", 0)
                * 1000,
            },
            "predictive_warming": prediction_stats,
        }

    def optimize_all_layers(self) -> Dict[str, Any]:
        """全レイヤー最適化"""
        optimizations = {}

        try:
            # L4圧縮最適化
            l4_optimization = self.l4_cache.optimize_compression()
            optimizations["l4_compression"] = l4_optimization

            # メモリ最適化（既存）
            self.optimize_memory()
            optimizations["memory_optimized"] = True

            # 統計ベース推奨事項
            stats = self.get_enhanced_stats()
            recommendations = self._generate_optimization_recommendations(stats)
            optimizations["recommendations"] = recommendations

            logger.info(f"全レイヤー最適化完了: {len(recommendations)}件の推奨事項")

        except Exception as e:
            logger.error(f"最適化エラー: {e}")
            optimizations["error"] = str(e)

        return optimizations

    def _generate_optimization_recommendations(
        self, stats: Dict[str, Any]
    ) -> List[str]:
        """最適化推奨事項生成"""
        recommendations = []
        overall = stats["overall"]

        # ヒット率分析
        if overall["hit_rate"] < 0.8:
            recommendations.append(
                "全体ヒット率が80%未満です。TTL延長またはキャッシュサイズ増加を検討してください。"
            )

        # レイヤー比率分析
        l4_ratio = overall["l4_hit_ratio"]
        if l4_ratio > 0.3:
            recommendations.append(
                "L4アクセスが30%超です。上位レイヤーのサイズ増加を検討してください。"
            )

        # 応答時間分析
        if overall["avg_access_time_ms"] > 5:
            recommendations.append(
                "平均応答時間が5ms超です。インデックスまたはメモリキャッシュの最適化が必要です。"
            )

        # 圧縮効率分析
        compression_ratio = stats["compression_savings"]["avg_compression_ratio"]
        if compression_ratio > 0.8:
            recommendations.append(
                "L4の圧縮効率が低いです。圧縮アルゴリズムの変更を検討してください。"
            )

        # 容量使用率分析
        if stats["archive_usage_gb"] > 8:  # 10GB中8GB使用
            recommendations.append(
                "L4アーカイブ使用率が80%超です。容量拡張または古いデータの削除を検討してください。"
            )

        return recommendations


# 統合キャッシュキー生成（既存関数をそのまま使用）
generate_enhanced_cache_key = generate_unified_cache_key


if __name__ == "__main__":
    # 拡張統合キャッシュマネージャーテスト
    print("=== 拡張統合キャッシュマネージャーテスト ===")

    try:
        # テスト用設定
        cache_manager = EnhancedUnifiedCacheManager(
            l1_memory_mb=32,
            l2_memory_mb=128,
            l3_disk_mb=512,
            l4_archive_gb=2,
            enable_prewarming=True,
        )

        # テストデータ
        test_data = {
            "ultra_hot": ("A" * 50, 9.0),  # 超高優先度・小サイズ → L1
            "hot_data": ("B" * 500, 7.0),  # 高優先度・小〜中サイズ → L2
            "warm_data": ("C" * 5000, 4.0),  # 中優先度・中サイズ → L3
            "cold_data": ("D" * 50000, 2.0),  # 低優先度・大サイズ → L4
            "archive_data": ("E" * 500000, 0.5),  # 極低優先度・超大サイズ → L4
        }

        # データ保存テスト
        print("\n1. 4層キャッシュ保存テスト...")
        for key, (value, priority) in test_data.items():
            success = cache_manager.put(key, value, priority=priority)
            print(f"  {key}: {'成功' if success else '失敗'} (優先度={priority})")

        # データ取得テスト
        print("\n2. 4層キャッシュ取得テスト...")
        for key in test_data:
            value = cache_manager.get(key)
            print(f"  {key}: {'ヒット' if value else 'ミス'}")

        # 拡張統計情報
        print("\n3. 拡張統計情報...")
        stats = cache_manager.get_enhanced_stats()
        overall = stats["overall"]
        print(f"  全体ヒット率: {overall['hit_rate']:.2%}")
        print(f"  平均アクセス時間: {overall['avg_access_time_ms']:.2f}ms")
        print(f"  L1ヒット率: {overall['l1_hit_ratio']:.1%}")
        print(f"  L2ヒット率: {overall['l2_hit_ratio']:.1%}")
        print(f"  L3ヒット率: {overall['l3_hit_ratio']:.1%}")
        print(f"  L4ヒット率: {overall['l4_hit_ratio']:.1%}")
        print(f"  メモリ使用量: {stats['memory_usage_total_mb']:.1f}MB")
        print(f"  アーカイブ使用量: {stats['archive_usage_gb']:.3f}GB")

        # 最適化テスト
        print("\n4. 全レイヤー最適化...")
        optimizations = cache_manager.optimize_all_layers()
        recommendations = optimizations.get("recommendations", [])
        print(f"  推奨事項数: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:3]):  # 最初の3件のみ表示
            print(f"    {i+1}. {rec}")

        print("\n✅ 拡張統合キャッシュマネージャーテスト完了")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
