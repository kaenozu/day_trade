#!/usr/bin/env python3
"""
特徴量操作モジュール

特徴量の存在確認、読み込み、保存などの基本操作を提供します。
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import (
    FeatureMetadata,
    FeatureStoreConfig,
    StatisticsTracker,
    generate_config_hash,
    generate_feature_id,
)
from .metadata_manager import MetadataManager
from .storage_backends import StorageManager
from ...analysis.feature_engineering_unified import (
    FeatureConfig,
    FeatureEngineeringManager,
    FeatureResult,
)
from ...core.optimization_strategy import OptimizationConfig
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class FeatureOperations:
    """
    特徴量操作クラス
    
    特徴量の基本的な CRUD 操作を管理します。
    """

    def __init__(
        self,
        base_path: Path,
        config: FeatureStoreConfig,
        metadata_manager: MetadataManager,
        storage_manager: StorageManager,
        stats_tracker: StatisticsTracker,
    ):
        """
        初期化
        
        Args:
            base_path: 特徴量ベースパス
            config: 特徴量ストア設定
            metadata_manager: メタデータマネージャー
            storage_manager: ストレージマネージャー
            stats_tracker: 統計トラッカー
        """
        self.base_path = base_path
        self.config = config
        self.metadata_manager = metadata_manager
        self.storage_manager = storage_manager
        self.stats_tracker = stats_tracker

    def has_feature(
        self, 
        metadata_index: dict, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        feature_config: FeatureConfig
    ) -> bool:
        """
        特徴量の存在確認
        
        Args:
            metadata_index: メタデータインデックス
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            
        Returns:
            特徴量が存在する場合True
        """
        config_hash = generate_config_hash(feature_config)
        feature_id = generate_feature_id(symbol, start_date, end_date, config_hash)

        if feature_id not in metadata_index:
            return False

        # ファイルの実際の存在確認
        metadata = metadata_index[feature_id]
        file_path = Path(metadata.file_path)

        if not file_path.exists():
            # メタデータから削除
            del metadata_index[feature_id]
            return False

        return True

    def load_feature(
        self, 
        metadata_index: dict,
        symbol: str, 
        start_date: str, 
        end_date: str, 
        feature_config: FeatureConfig
    ) -> Optional[FeatureResult]:
        """
        特徴量の読み込み
        
        Args:
            metadata_index: メタデータインデックス
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            
        Returns:
            読み込まれた特徴量結果、存在しない場合はNone
        """
        config_hash = generate_config_hash(feature_config)
        feature_id = generate_feature_id(symbol, start_date, end_date, config_hash)

        if not self.has_feature(metadata_index, symbol, start_date, end_date, feature_config):
            self.stats_tracker.increment_cache_misses()
            return None

        start_time = time.time()

        try:
            metadata = metadata_index[feature_id]
            file_path = Path(metadata.file_path)

            # 保存形式の判定と読み込み
            storage_format = self.storage_manager.detect_storage_format(file_path)
            feature_data = self.storage_manager.load_feature_data(file_path, storage_format)

            # FeatureResultの復元
            feature_result = FeatureResult(
                features=feature_data["features"],
                feature_names=metadata.feature_names,
                metadata={
                    **feature_data["metadata"],
                    "loaded_from_cache": True,
                    "cache_feature_id": feature_id,
                },
                generation_time=metadata.generation_time_seconds,
                strategy_used=metadata.strategy_used,
            )

            loading_time = time.time() - start_time

            # 統計更新
            self.stats_tracker.increment_cache_hits()
            self.stats_tracker.increment_features_loaded()
            self.stats_tracker.add_loading_time(loading_time)

            logger.info(
                "特徴量キャッシュから読み込み完了",
                extra={
                    "feature_id": feature_id,
                    "symbol": symbol,
                    "loading_time_ms": round(loading_time * 1000, 2),
                    "feature_count": len(metadata.feature_names),
                    "data_shape": feature_data["features"].shape,
                },
            )

            return feature_result

        except Exception as e:
            logger.error(f"特徴量読み込みエラー: {feature_id} - {e}")
            # 破損したキャッシュの削除
            self.remove_feature(metadata_index, feature_id)
            self.stats_tracker.increment_cache_misses()
            return None

    def save_feature(
        self,
        metadata_index: dict,
        symbol: str,
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        feature_result: FeatureResult,
    ) -> str:
        """
        特徴量の保存
        
        Args:
            metadata_index: メタデータインデックス
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            feature_result: 特徴量結果
            
        Returns:
            生成された特徴量ID
        """
        config_hash = generate_config_hash(feature_config)
        feature_id = generate_feature_id(symbol, start_date, end_date, config_hash)

        start_time = time.time()

        try:
            # 保存データの準備
            feature_data = {
                "features": feature_result.features,
                "metadata": feature_result.metadata,
            }

            # 保存形式の決定
            storage_format = self.storage_manager.determine_storage_format(feature_data)

            # ファイルパス取得（形式に応じた拡張子）
            file_path = self.storage_manager.get_file_path(
                self.base_path, feature_id, storage_format
            )

            # データ保存
            self.storage_manager.save_feature_data(feature_data, file_path, storage_format)

            # ファイルサイズ取得
            file_size = file_path.stat().st_size

            # メタデータ作成
            metadata = FeatureMetadata(
                feature_id=feature_id,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                config_hash=config_hash,
                feature_names=feature_result.feature_names,
                created_at=datetime.now(),
                file_path=str(file_path),
                file_size_bytes=file_size,
                row_count=len(feature_result.features),
                column_count=len(feature_result.feature_names),
                generation_time_seconds=feature_result.generation_time,
                strategy_used=feature_result.strategy_used,
            )

            # メタデータインデックスに追加
            metadata_index[feature_id] = metadata

            # インデックス保存
            self.metadata_manager.save_metadata_index(metadata_index)

            saving_time = time.time() - start_time

            # 統計更新
            self.stats_tracker.increment_features_generated()

            logger.info(
                "特徴量保存完了",
                extra={
                    "feature_id": feature_id,
                    "symbol": symbol,
                    "file_size_mb": round(file_size / 1024 / 1024, 3),
                    "saving_time_ms": round(saving_time * 1000, 2),
                    "feature_count": len(feature_result.feature_names),
                    "data_shape": feature_result.features.shape,
                    "compression": self.config.enable_compression,
                    "storage_format": storage_format,
                },
            )

            return feature_id

        except Exception as e:
            logger.error(f"特徴量保存エラー: {feature_id} - {e}")
            raise

    def remove_feature(self, metadata_index: dict, feature_id: str):
        """
        特徴量の削除
        
        Args:
            metadata_index: メタデータインデックス
            feature_id: 特徴量ID
        """
        if feature_id in metadata_index:
            metadata = metadata_index[feature_id]
            file_path = Path(metadata.file_path)

            # ファイル削除
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"ファイル削除エラー: {file_path} - {e}")

            # メタデータから削除
            del metadata_index[feature_id]

            logger.debug(f"特徴量削除完了: {feature_id}")

    def get_or_generate_feature(
        self,
        metadata_index: dict,
        symbol: str,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> FeatureResult:
        """
        特徴量の取得または生成（メインインターフェース）
        
        Args:
            metadata_index: メタデータインデックス
            symbol: 銘柄コード
            data: 価格データ
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            optimization_config: 最適化設定
            
        Returns:
            特徴量結果
        """
        # キャッシュから読み込み試行
        cached_feature = self.load_feature(
            metadata_index, symbol, start_date, end_date, feature_config
        )
        if cached_feature:
            return cached_feature

        # 特徴量生成
        logger.info(
            "特徴量生成開始",
            extra={
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "data_shape": data.shape,
            },
        )

        manager = FeatureEngineeringManager(optimization_config)
        feature_result = manager.generate_features(data, feature_config)

        # 保存
        feature_id = self.save_feature(
            metadata_index, symbol, start_date, end_date, feature_config, feature_result
        )

        # メタデータに特徴量IDを追加
        feature_result.metadata["feature_id"] = feature_id

        return feature_result

    def get_cache_size_mb(self, metadata_index: dict) -> float:
        """
        キャッシュサイズの取得
        
        Args:
            metadata_index: メタデータインデックス
            
        Returns:
            キャッシュサイズ（MB）
        """
        total_size = 0
        for metadata in metadata_index.values():
            total_size += metadata.file_size_bytes
        return total_size / 1024 / 1024