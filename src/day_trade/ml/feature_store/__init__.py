#!/usr/bin/env python3
"""
特徴量ストアモジュール

特徴量の効率的保存・読み込み・再利用システムを提供します。

主要クラス:
- FeatureStore: メインの特徴量ストア
- FeatureStoreConfig: 設定クラス
- FeatureMetadata: メタデータクラス

使用例:
    ```python
    from day_trade.ml.feature_store import FeatureStore, FeatureStoreConfig
    
    # 設定作成
    config = FeatureStoreConfig(
        base_path="data/features",
        max_cache_size_mb=1024,
        enable_parallel_batch_processing=True
    )
    
    # 特徴量ストア作成
    feature_store = FeatureStore(config)
    
    # 特徴量取得または生成
    result = feature_store.get_or_generate_feature(
        symbol="7203",
        data=stock_data,
        start_date="2024-01-01",
        end_date="2024-12-31",
        feature_config=feature_config
    )
    ```
"""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# 基本クラス・設定のインポート
from .base import (
    FeatureMetadata,
    FeatureStoreConfig,
    StatisticsTracker,
)

# 各種マネージャーのインポート
from .batch_processing import BatchProcessor, ChunkedBatchProcessor
from .cache_manager import CacheManager, CacheAnalyzer
from .feature_operations import FeatureOperations
from .metadata_manager import MetadataManager
from .storage_backends import StorageManager

from ..analysis.feature_engineering_unified import FeatureConfig, FeatureResult
from ..core.optimization_strategy import OptimizationConfig
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class FeatureStore:
    """
    特徴量ストア - 特徴量の保存・再利用システム
    
    主要機能:
    1. 特徴量の効率的保存・読み込み
    2. 重複計算の排除と再利用機能
    3. バッチ特徴量生成
    4. バージョン管理とメタデータ追跡
    """

    def __init__(self, config: FeatureStoreConfig = None):
        """
        初期化
        
        Args:
            config: 特徴量ストア設定
        """
        self.config = config or FeatureStoreConfig()
        self.base_path = Path(self.config.base_path)
        self.metadata_path = self.base_path / "metadata"

        # ディレクトリ作成
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)

        # コンポーネントの初期化
        self.stats_tracker = StatisticsTracker()
        self.metadata_manager = MetadataManager(self.metadata_path, self.config)
        self.storage_manager = StorageManager(self.config)
        self.feature_operations = FeatureOperations(
            self.base_path,
            self.config,
            self.metadata_manager,
            self.storage_manager,
            self.stats_tracker,
        )
        self.cache_manager = CacheManager(
            self.config, self.feature_operations, self.metadata_manager
        )
        self.batch_processor = BatchProcessor(self.config, self.feature_operations)
        self.chunked_batch_processor = ChunkedBatchProcessor(
            self.config, self.feature_operations
        )

        # メタデータインデックス
        self.metadata_index: Dict[str, FeatureMetadata] = {}

        # 初期化処理
        self._initialize()

    def _initialize(self):
        """初期化処理"""
        logger.info(
            "特徴量ストア初期化開始",
            extra={"base_path": str(self.base_path), "config": asdict(self.config)},
        )

        # メタデータの読み込み
        self.metadata_index = self.metadata_manager.load_metadata_index()

        # クリーンアップ（設定による）
        if self.config.cleanup_on_startup:
            self.cache_manager.cleanup_old_features(self.metadata_index)

        logger.info(
            "特徴量ストア初期化完了",
            extra={
                "features_count": len(self.metadata_index),
                "cache_size_mb": self.feature_operations.get_cache_size_mb(
                    self.metadata_index
                ),
            },
        )

    def has_feature(
        self, symbol: str, start_date: str, end_date: str, feature_config: FeatureConfig
    ) -> bool:
        """
        特徴量の存在確認
        
        Args:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            
        Returns:
            特徴量が存在する場合True
        """
        return self.feature_operations.has_feature(
            self.metadata_index, symbol, start_date, end_date, feature_config
        )

    def load_feature(
        self, symbol: str, start_date: str, end_date: str, feature_config: FeatureConfig
    ) -> Optional[FeatureResult]:
        """
        特徴量の読み込み
        
        Args:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            
        Returns:
            読み込まれた特徴量結果、存在しない場合はNone
        """
        return self.feature_operations.load_feature(
            self.metadata_index, symbol, start_date, end_date, feature_config
        )

    def save_feature(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        feature_result: FeatureResult,
    ) -> str:
        """
        特徴量の保存
        
        Args:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            feature_result: 特徴量結果
            
        Returns:
            生成された特徴量ID
        """
        return self.feature_operations.save_feature(
            self.metadata_index,
            symbol,
            start_date,
            end_date,
            feature_config,
            feature_result,
        )

    def get_or_generate_feature(
        self,
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
            symbol: 銘柄コード
            data: 価格データ
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            optimization_config: 最適化設定
            
        Returns:
            特徴量結果
        """
        return self.feature_operations.get_or_generate_feature(
            self.metadata_index,
            symbol,
            data,
            start_date,
            end_date,
            feature_config,
            optimization_config,
        )

    def batch_generate_features(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """
        バッチ特徴量生成（並列化対応）
        
        Args:
            symbols: 銘柄コードのリスト
            data_dict: 銘柄別データ辞書
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            optimization_config: 最適化設定
            
        Returns:
            銘柄別特徴量結果の辞書
        """
        return self.batch_processor.batch_generate_features(
            self.metadata_index,
            symbols,
            data_dict,
            start_date,
            end_date,
            feature_config,
            optimization_config,
        )

    def batch_generate_features_chunked(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """
        チャンク化バッチ特徴量生成
        
        Args:
            symbols: 銘柄コードのリスト
            data_dict: 銘柄別データ辞書
            start_date: 開始日
            end_date: 終了日
            feature_config: 特徴量設定
            optimization_config: 最適化設定
            
        Returns:
            銘柄別特徴量結果の辞書
        """
        return self.chunked_batch_processor.process_in_chunks(
            self.metadata_index,
            symbols,
            data_dict,
            start_date,
            end_date,
            feature_config,
            optimization_config,
        )

    def get_stats(self) -> Dict:
        """
        統計情報の取得
        
        Returns:
            統計情報の辞書
        """
        cache_size_mb = self.feature_operations.get_cache_size_mb(self.metadata_index)
        features_in_cache = len(self.metadata_index)
        
        return self.stats_tracker.get_stats(cache_size_mb, features_in_cache)

    def cleanup_cache(self, force: bool = False):
        """
        キャッシュのクリーンアップ
        
        Args:
            force: 強制クリーンアップフラグ
        """
        self.cache_manager.cleanup_cache(self.metadata_index, force)

    def optimize_cache(self):
        """キャッシュ最適化"""
        self.cache_manager.optimize_cache(self.metadata_index)

    def analyze_cache(self) -> Dict:
        """
        キャッシュ使用状況の分析
        
        Returns:
            分析結果
        """
        analyzer = CacheAnalyzer()
        return analyzer.analyze_cache_usage(self.metadata_index)

    def get_cache_statistics(self) -> Dict:
        """
        キャッシュ統計情報の取得
        
        Returns:
            キャッシュ統計情報
        """
        return self.cache_manager.get_cache_statistics(self.metadata_index)

    def suggest_cleanup_targets(self) -> Dict:
        """
        クリーンアップ対象の提案
        
        Returns:
            クリーンアップ提案
        """
        analyzer = CacheAnalyzer()
        return analyzer.suggest_cleanup_targets(self.metadata_index)

    @contextmanager
    def batch_context(self):
        """バッチ処理用コンテキストマネージャー"""
        try:
            logger.info("バッチ処理開始")
            yield self
        finally:
            # バッチ処理後のクリーンアップ
            self.metadata_manager.save_metadata_index(self.metadata_index)
            logger.info("バッチ処理完了")


# 便利関数
def create_feature_store(config: FeatureStoreConfig = None) -> FeatureStore:
    """
    特徴量ストアの作成
    
    Args:
        config: 特徴量ストア設定
        
    Returns:
        FeatureStoreインスタンス
    """
    return FeatureStore(config)


# バックワード互換性のためのエクスポート
__all__ = [
    # メインクラス
    "FeatureStore",
    "create_feature_store",
    
    # 設定・データクラス
    "FeatureStoreConfig",
    "FeatureMetadata",
    
    # マネージャークラス（高度な使用法向け）
    "MetadataManager",
    "StorageManager",
    "FeatureOperations",
    "CacheManager",
    "CacheAnalyzer",
    "BatchProcessor",
    "ChunkedBatchProcessor",
    
    # ユーティリティ
    "StatisticsTracker",
]