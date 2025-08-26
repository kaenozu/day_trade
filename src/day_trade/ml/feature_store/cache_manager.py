#!/usr/bin/env python3
"""
キャッシュマネージャー

特徴量キャッシュのライフサイクル管理機能を提供します。
"""

from datetime import datetime, timedelta
from typing import Dict

from .base import FeatureMetadata, FeatureStoreConfig
from .feature_operations import FeatureOperations
from .metadata_manager import MetadataManager
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class CacheManager:
    """
    キャッシュマネージャークラス
    
    特徴量キャッシュのクリーンアップとサイズ管理を行います。
    """

    def __init__(
        self,
        config: FeatureStoreConfig,
        feature_operations: FeatureOperations,
        metadata_manager: MetadataManager,
    ):
        """
        初期化
        
        Args:
            config: 特徴量ストア設定
            feature_operations: 特徴量操作クラス
            metadata_manager: メタデータマネージャー
        """
        self.config = config
        self.feature_operations = feature_operations
        self.metadata_manager = metadata_manager

    def cleanup_old_features(self, metadata_index: Dict[str, FeatureMetadata]):
        """
        古い特徴量のクリーンアップ
        
        Args:
            metadata_index: メタデータインデックス
        """
        if not self.config.max_cache_age_days:
            return

        cutoff_date = datetime.now() - timedelta(days=self.config.max_cache_age_days)
        removed_count = 0

        feature_ids_to_remove = []

        for feature_id, metadata in metadata_index.items():
            if metadata.created_at < cutoff_date:
                feature_ids_to_remove.append(feature_id)

        for feature_id in feature_ids_to_remove:
            try:
                self.feature_operations.remove_feature(metadata_index, feature_id)
                removed_count += 1
            except Exception as e:
                logger.warning(f"特徴量削除エラー: {feature_id} - {e}")

        if removed_count > 0:
            # メタデータインデックスを保存
            self.metadata_manager.save_metadata_index(metadata_index)
            logger.info(f"古い特徴量をクリーンアップ: {removed_count}件")

    def cleanup_cache_by_size(self, metadata_index: Dict[str, FeatureMetadata]):
        """
        サイズによるキャッシュクリーンアップ
        
        Args:
            metadata_index: メタデータインデックス
        """
        current_size_mb = self.feature_operations.get_cache_size_mb(metadata_index)
        
        if current_size_mb <= self.config.max_cache_size_mb:
            return

        # 最も古い特徴量から削除
        sorted_metadata = sorted(
            metadata_index.items(), key=lambda x: x[1].created_at
        )

        removed_count = 0
        target_size_mb = self.config.max_cache_size_mb * 0.8  # 80%まで削減

        for feature_id, metadata in sorted_metadata:
            current_size_mb = self.feature_operations.get_cache_size_mb(metadata_index)
            if current_size_mb <= target_size_mb:
                break

            try:
                self.feature_operations.remove_feature(metadata_index, feature_id)
                removed_count += 1
            except Exception as e:
                logger.warning(f"特徴量削除エラー: {feature_id} - {e}")

        if removed_count > 0:
            # メタデータインデックスを保存
            self.metadata_manager.save_metadata_index(metadata_index)
            logger.info(f"サイズによるキャッシュクリーンアップ: {removed_count}件削除")

    def cleanup_cache(
        self, 
        metadata_index: Dict[str, FeatureMetadata], 
        force: bool = False
    ):
        """
        包括的なキャッシュクリーンアップ
        
        Args:
            metadata_index: メタデータインデックス
            force: 強制クリーンアップフラグ
        """
        if force:
            logger.info("強制キャッシュクリーンアップ実行")

        # 古い特徴量の削除
        self.cleanup_old_features(metadata_index)

        # サイズによるクリーンアップ
        current_size_mb = self.feature_operations.get_cache_size_mb(metadata_index)
        if force or current_size_mb > self.config.max_cache_size_mb:
            self.cleanup_cache_by_size(metadata_index)

        # データベース最適化
        self.metadata_manager.vacuum_database()

    def cleanup_broken_features(self, metadata_index: Dict[str, FeatureMetadata]):
        """
        破損した特徴量の削除
        
        Args:
            metadata_index: メタデータインデックス
        """
        broken_feature_ids = []

        for feature_id, metadata in metadata_index.items():
            file_path = metadata.file_path
            
            # ファイル存在確認
            from pathlib import Path
            if not Path(file_path).exists():
                broken_feature_ids.append(feature_id)
                continue

            # ファイルサイズ確認
            try:
                actual_size = Path(file_path).stat().st_size
                if actual_size != metadata.file_size_bytes:
                    logger.warning(
                        f"ファイルサイズ不一致: {feature_id} "
                        f"(期待: {metadata.file_size_bytes}, 実際: {actual_size})"
                    )
                    broken_feature_ids.append(feature_id)
            except Exception as e:
                logger.warning(f"ファイル確認エラー: {feature_id} - {e}")
                broken_feature_ids.append(feature_id)

        # 破損したファイルを削除
        removed_count = 0
        for feature_id in broken_feature_ids:
            try:
                self.feature_operations.remove_feature(metadata_index, feature_id)
                removed_count += 1
            except Exception as e:
                logger.error(f"破損ファイル削除エラー: {feature_id} - {e}")

        if removed_count > 0:
            # メタデータインデックスを保存
            self.metadata_manager.save_metadata_index(metadata_index)
            logger.info(f"破損した特徴量を削除: {removed_count}件")

    def get_cache_statistics(self, metadata_index: Dict[str, FeatureMetadata]) -> Dict:
        """
        キャッシュ統計情報の取得
        
        Args:
            metadata_index: メタデータインデックス
            
        Returns:
            キャッシュ統計情報
        """
        if not metadata_index:
            return {
                "total_features": 0,
                "total_size_mb": 0.0,
                "oldest_feature": None,
                "newest_feature": None,
                "avg_file_size_mb": 0.0,
                "symbols_count": 0,
            }

        total_size_bytes = sum(m.file_size_bytes for m in metadata_index.values())
        total_size_mb = total_size_bytes / (1024 * 1024)

        created_times = [m.created_at for m in metadata_index.values()]
        oldest_feature = min(created_times).isoformat()
        newest_feature = max(created_times).isoformat()

        avg_file_size_mb = total_size_mb / len(metadata_index) if metadata_index else 0

        unique_symbols = set(m.symbol for m in metadata_index.values())

        return {
            "total_features": len(metadata_index),
            "total_size_mb": round(total_size_mb, 2),
            "oldest_feature": oldest_feature,
            "newest_feature": newest_feature,
            "avg_file_size_mb": round(avg_file_size_mb, 3),
            "symbols_count": len(unique_symbols),
        }

    def optimize_cache(self, metadata_index: Dict[str, FeatureMetadata]):
        """
        キャッシュ最適化
        
        Args:
            metadata_index: メタデータインデックス
        """
        logger.info("キャッシュ最適化開始")

        # 破損ファイルの削除
        self.cleanup_broken_features(metadata_index)

        # 包括的なクリーンアップ
        self.cleanup_cache(metadata_index, force=False)

        # 統計情報取得
        stats = self.get_cache_statistics(metadata_index)
        
        logger.info(
            "キャッシュ最適化完了",
            extra=stats
        )


class CacheAnalyzer:
    """
    キャッシュアナライザークラス
    
    キャッシュの利用状況を分析します。
    """

    def __init__(self):
        """初期化"""
        pass

    def analyze_cache_usage(
        self, metadata_index: Dict[str, FeatureMetadata]
    ) -> Dict:
        """
        キャッシュ使用状況の分析
        
        Args:
            metadata_index: メタデータインデックス
            
        Returns:
            分析結果
        """
        if not metadata_index:
            return {"error": "メタデータが空です"}

        # 銘柄別統計
        symbol_stats = {}
        for metadata in metadata_index.values():
            symbol = metadata.symbol
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    "count": 0,
                    "total_size_bytes": 0,
                    "avg_generation_time": 0,
                }
            
            symbol_stats[symbol]["count"] += 1
            symbol_stats[symbol]["total_size_bytes"] += metadata.file_size_bytes
            symbol_stats[symbol]["avg_generation_time"] += metadata.generation_time_seconds

        # 平均生成時間の計算
        for symbol in symbol_stats:
            count = symbol_stats[symbol]["count"]
            symbol_stats[symbol]["avg_generation_time"] /= count
            symbol_stats[symbol]["total_size_mb"] = symbol_stats[symbol]["total_size_bytes"] / (1024 * 1024)

        # 上位統計
        top_symbols_by_count = sorted(
            symbol_stats.items(), 
            key=lambda x: x[1]["count"], 
            reverse=True
        )[:10]

        top_symbols_by_size = sorted(
            symbol_stats.items(), 
            key=lambda x: x[1]["total_size_mb"], 
            reverse=True
        )[:10]

        return {
            "symbol_stats": symbol_stats,
            "top_symbols_by_count": top_symbols_by_count,
            "top_symbols_by_size": top_symbols_by_size,
            "total_symbols": len(symbol_stats),
        }

    def suggest_cleanup_targets(
        self, metadata_index: Dict[str, FeatureMetadata]
    ) -> Dict:
        """
        クリーンアップ対象の提案
        
        Args:
            metadata_index: メタデータインデックス
            
        Returns:
            クリーンアップ提案
        """
        suggestions = {
            "old_features": [],
            "large_features": [],
            "duplicate_configs": [],
        }

        # 古い特徴量（30日以上）
        cutoff_date = datetime.now() - timedelta(days=30)
        for feature_id, metadata in metadata_index.items():
            if metadata.created_at < cutoff_date:
                suggestions["old_features"].append({
                    "feature_id": feature_id,
                    "symbol": metadata.symbol,
                    "age_days": (datetime.now() - metadata.created_at).days,
                    "size_mb": metadata.file_size_bytes / (1024 * 1024),
                })

        # 大きな特徴量（10MB以上）
        for feature_id, metadata in metadata_index.items():
            size_mb = metadata.file_size_bytes / (1024 * 1024)
            if size_mb > 10:
                suggestions["large_features"].append({
                    "feature_id": feature_id,
                    "symbol": metadata.symbol,
                    "size_mb": size_mb,
                })

        # 重複する設定ハッシュ
        config_hash_counts = {}
        for metadata in metadata_index.values():
            config_hash = metadata.config_hash
            if config_hash not in config_hash_counts:
                config_hash_counts[config_hash] = []
            config_hash_counts[config_hash].append(metadata)

        for config_hash, metadata_list in config_hash_counts.items():
            if len(metadata_list) > 5:  # 5つ以上で重複とみなす
                suggestions["duplicate_configs"].append({
                    "config_hash": config_hash,
                    "count": len(metadata_list),
                    "symbols": [m.symbol for m in metadata_list],
                })

        return suggestions