#!/usr/bin/env python3
"""
特徴量ストア（Feature Store）システム
Issue #380: 機械学習モデルの最適化 - 特徴量生成の効率化

主要機能:
1. 特徴量の効率的保存・読み込み
2. 重複計算の排除と再利用機能
3. バッチ特徴量生成
4. バージョン管理とメタデータ追跡
"""

import hashlib
import json
import pickle
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..analysis.feature_engineering_unified import (
    FeatureConfig,
    FeatureEngineeringManager,
    FeatureResult,
)
from ..core.optimization_strategy import OptimizationConfig
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class FeatureMetadata:
    """特徴量メタデータ"""

    feature_id: str
    symbol: str
    start_date: str
    end_date: str
    config_hash: str
    feature_names: List[str]
    created_at: datetime
    file_path: str
    file_size_bytes: int
    row_count: int
    column_count: int
    generation_time_seconds: float
    strategy_used: str

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMetadata":
        """辞書から復元"""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class FeatureStoreConfig:
    """特徴量ストア設定"""

    base_path: str = "data/features"
    max_cache_age_days: int = 30
    max_cache_size_mb: int = 1024  # 1GB
    enable_compression: bool = True
    enable_metadata_indexing: bool = True
    cleanup_on_startup: bool = True
    auto_versioning: bool = True


class FeatureStore:
    """特徴量ストア - 特徴量の保存・再利用システム"""

    def __init__(self, config: FeatureStoreConfig = None):
        """初期化"""
        self.config = config or FeatureStoreConfig()
        self.base_path = Path(self.config.base_path)
        self.metadata_path = self.base_path / "metadata"

        # ディレクトリ作成
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)

        # メタデータインデックス
        self.metadata_index: Dict[str, FeatureMetadata] = {}

        # 統計情報
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "features_generated": 0,
            "features_loaded": 0,
            "total_generation_time": 0.0,
            "total_loading_time": 0.0,
        }

        # 初期化処理
        self._initialize()

    def _initialize(self):
        """初期化処理"""
        logger.info(
            "特徴量ストア初期化開始",
            extra={"base_path": str(self.base_path), "config": asdict(self.config)},
        )

        # メタデータの読み込み
        self._load_metadata_index()

        # クリーンアップ（設定による）
        if self.config.cleanup_on_startup:
            self._cleanup_old_features()

        logger.info(
            "特徴量ストア初期化完了",
            extra={
                "features_count": len(self.metadata_index),
                "cache_size_mb": self._get_cache_size_mb(),
            },
        )

    def _load_metadata_index(self):
        """メタデータインデックスの読み込み"""
        index_file = self.metadata_path / "index.json"

        if index_file.exists():
            try:
                with open(index_file, encoding="utf-8") as f:
                    index_data = json.load(f)

                for feature_id, metadata_dict in index_data.items():
                    self.metadata_index[feature_id] = FeatureMetadata.from_dict(
                        metadata_dict
                    )

                logger.info(
                    f"メタデータインデックス読み込み完了: {len(self.metadata_index)}件"
                )

            except Exception as e:
                logger.error(f"メタデータインデックス読み込みエラー: {e}")
                self.metadata_index = {}
        else:
            logger.info("新しいメタデータインデックスを作成")

    def _save_metadata_index(self):
        """メタデータインデックスの保存"""
        index_file = self.metadata_path / "index.json"

        try:
            index_data = {
                feature_id: metadata.to_dict()
                for feature_id, metadata in self.metadata_index.items()
            }

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)

            logger.debug(
                f"メタデータインデックス保存完了: {len(self.metadata_index)}件"
            )

        except Exception as e:
            logger.error(f"メタデータインデックス保存エラー: {e}")

    def _generate_feature_id(
        self, symbol: str, start_date: str, end_date: str, config_hash: str
    ) -> str:
        """特徴量IDの生成"""
        # 一意な識別子の生成
        id_string = f"{symbol}_{start_date}_{end_date}_{config_hash}"
        feature_id = hashlib.md5(id_string.encode()).hexdigest()[:16]
        return feature_id

    def _generate_config_hash(self, feature_config: FeatureConfig) -> str:
        """特徴量設定のハッシュ値生成"""
        # 設定を正規化してハッシュ化
        config_dict = asdict(feature_config)
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return config_hash

    def _get_feature_file_path(self, feature_id: str) -> Path:
        """特徴量ファイルパスの取得"""
        # 2レベルディレクトリ構造（パフォーマンス向上）
        dir1 = feature_id[:2]
        dir2 = feature_id[2:4]

        feature_dir = self.base_path / dir1 / dir2
        feature_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{feature_id}.pkl"
        if self.config.enable_compression:
            filename += ".gz"

        return feature_dir / filename

    def _cleanup_old_features(self):
        """古い特徴量のクリーンアップ"""
        if not self.config.max_cache_age_days:
            return

        cutoff_date = datetime.now() - timedelta(days=self.config.max_cache_age_days)
        removed_count = 0

        feature_ids_to_remove = []

        for feature_id, metadata in self.metadata_index.items():
            if metadata.created_at < cutoff_date:
                feature_ids_to_remove.append(feature_id)

        for feature_id in feature_ids_to_remove:
            try:
                self._remove_feature(feature_id)
                removed_count += 1
            except Exception as e:
                logger.warning(f"特徴量削除エラー: {feature_id} - {e}")

        if removed_count > 0:
            logger.info(f"古い特徴量をクリーンアップ: {removed_count}件")

    def _remove_feature(self, feature_id: str):
        """特徴量の削除"""
        if feature_id in self.metadata_index:
            metadata = self.metadata_index[feature_id]
            file_path = Path(metadata.file_path)

            # ファイル削除
            if file_path.exists():
                file_path.unlink()

            # メタデータから削除
            del self.metadata_index[feature_id]

            logger.debug(f"特徴量削除完了: {feature_id}")

    def _get_cache_size_mb(self) -> float:
        """キャッシュサイズの取得"""
        total_size = 0
        for metadata in self.metadata_index.values():
            total_size += metadata.file_size_bytes
        return total_size / 1024 / 1024

    def has_feature(
        self, symbol: str, start_date: str, end_date: str, feature_config: FeatureConfig
    ) -> bool:
        """特徴量の存在確認"""
        config_hash = self._generate_config_hash(feature_config)
        feature_id = self._generate_feature_id(
            symbol, start_date, end_date, config_hash
        )

        if feature_id not in self.metadata_index:
            return False

        # ファイルの実際の存在確認
        metadata = self.metadata_index[feature_id]
        file_path = Path(metadata.file_path)

        if not file_path.exists():
            # メタデータから削除
            del self.metadata_index[feature_id]
            return False

        return True

    def load_feature(
        self, symbol: str, start_date: str, end_date: str, feature_config: FeatureConfig
    ) -> Optional[FeatureResult]:
        """特徴量の読み込み"""
        config_hash = self._generate_config_hash(feature_config)
        feature_id = self._generate_feature_id(
            symbol, start_date, end_date, config_hash
        )

        if not self.has_feature(symbol, start_date, end_date, feature_config):
            self.stats["cache_misses"] += 1
            return None

        start_time = time.time()

        try:
            metadata = self.metadata_index[feature_id]
            file_path = Path(metadata.file_path)

            # 特徴量データの読み込み
            if self.config.enable_compression and file_path.suffix == ".gz":
                import gzip

                with gzip.open(file_path, "rb") as f:
                    feature_data = pickle.load(f)
            else:
                with open(file_path, "rb") as f:
                    feature_data = pickle.load(f)

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
            self.stats["cache_hits"] += 1
            self.stats["features_loaded"] += 1
            self.stats["total_loading_time"] += loading_time

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
            self._remove_feature(feature_id)
            self.stats["cache_misses"] += 1
            return None

    def save_feature(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        feature_result: FeatureResult,
    ) -> str:
        """特徴量の保存"""
        config_hash = self._generate_config_hash(feature_config)
        feature_id = self._generate_feature_id(
            symbol, start_date, end_date, config_hash
        )

        start_time = time.time()

        try:
            # 保存データの準備
            feature_data = {
                "features": feature_result.features,
                "metadata": feature_result.metadata,
            }

            # ファイルパス取得
            file_path = self._get_feature_file_path(feature_id)

            # データ保存
            if self.config.enable_compression:
                import gzip

                with gzip.open(file_path, "wb") as f:
                    pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_path, "wb") as f:
                    pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)

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
            self.metadata_index[feature_id] = metadata

            # インデックス保存
            self._save_metadata_index()

            saving_time = time.time() - start_time

            # 統計更新
            self.stats["features_generated"] += 1

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
                },
            )

            return feature_id

        except Exception as e:
            logger.error(f"特徴量保存エラー: {feature_id} - {e}")
            raise

    def get_or_generate_feature(
        self,
        symbol: str,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> FeatureResult:
        """特徴量の取得または生成（メインインターフェース）"""
        # キャッシュから読み込み試行
        cached_feature = self.load_feature(symbol, start_date, end_date, feature_config)
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
            symbol, start_date, end_date, feature_config, feature_result
        )

        # メタデータに特徴量IDを追加
        feature_result.metadata["feature_id"] = feature_id

        return feature_result

    def batch_generate_features(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """バッチ特徴量生成"""
        logger.info(
            "バッチ特徴量生成開始",
            extra={
                "symbols_count": len(symbols),
                "start_date": start_date,
                "end_date": end_date,
            },
        )

        results = {}
        cache_hits = 0
        cache_misses = 0

        for symbol in symbols:
            if symbol not in data_dict:
                logger.warning(f"データが見つかりません: {symbol}")
                continue

            try:
                # キャッシュ確認
                if self.has_feature(symbol, start_date, end_date, feature_config):
                    cached_result = self.load_feature(
                        symbol, start_date, end_date, feature_config
                    )
                    if cached_result:
                        results[symbol] = cached_result
                        cache_hits += 1
                        continue

                # 生成
                data = data_dict[symbol]
                feature_result = self.get_or_generate_feature(
                    symbol,
                    data,
                    start_date,
                    end_date,
                    feature_config,
                    optimization_config,
                )
                results[symbol] = feature_result
                cache_misses += 1

            except Exception as e:
                logger.error(f"バッチ特徴量生成エラー: {symbol} - {e}")
                continue

        logger.info(
            "バッチ特徴量生成完了",
            extra={
                "processed_symbols": len(results),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": f"{cache_hits/(cache_hits+cache_misses)*100:.1f}%"
                if (cache_hits + cache_misses) > 0
                else "0%",
            },
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """統計情報の取得"""
        cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            (self.stats["cache_hits"] / cache_total * 100) if cache_total > 0 else 0
        )

        return {
            **self.stats,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "features_in_cache": len(self.metadata_index),
            "cache_size_mb": round(self._get_cache_size_mb(), 2),
            "avg_generation_time_ms": round(
                (
                    self.stats["total_generation_time"]
                    / max(1, self.stats["features_generated"])
                )
                * 1000,
                2,
            ),
            "avg_loading_time_ms": round(
                (
                    self.stats["total_loading_time"]
                    / max(1, self.stats["features_loaded"])
                )
                * 1000,
                2,
            ),
        }

    def cleanup_cache(self, force: bool = False):
        """キャッシュのクリーンアップ"""
        if force or self._get_cache_size_mb() > self.config.max_cache_size_mb:
            # 最も古い特徴量から削除
            sorted_metadata = sorted(
                self.metadata_index.items(), key=lambda x: x[1].created_at
            )

            removed_count = 0
            target_size_mb = self.config.max_cache_size_mb * 0.8  # 80%まで削減

            for feature_id, metadata in sorted_metadata:
                if self._get_cache_size_mb() <= target_size_mb:
                    break

                self._remove_feature(feature_id)
                removed_count += 1

            self._save_metadata_index()

            logger.info(f"キャッシュクリーンアップ完了: {removed_count}件削除")

    @contextmanager
    def batch_context(self):
        """バッチ処理用コンテキストマネージャー"""
        try:
            logger.info("バッチ処理開始")
            yield self
        finally:
            # バッチ処理後のクリーンアップ
            self._save_metadata_index()
            logger.info("バッチ処理完了")


# 便利関数
def create_feature_store(config: FeatureStoreConfig = None) -> FeatureStore:
    """特徴量ストアの作成"""
    return FeatureStore(config)
