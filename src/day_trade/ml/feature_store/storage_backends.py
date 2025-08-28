#!/usr/bin/env python3
"""
特徴量ストレージバックエンド

様々な形式での特徴量データ保存・読み込み機能を提供します。
Issue #718対応: 最適化された保存形式をサポート
"""

import gzip
import pickle
from pathlib import Path
from typing import Any, Dict

from .base import (
    JOBLIB_AVAILABLE, 
    NUMPY_AVAILABLE, 
    PARQUET_AVAILABLE, 
    FeatureStoreConfig,
)
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 条件付きインポート
if JOBLIB_AVAILABLE:
    import joblib

if NUMPY_AVAILABLE:
    import numpy as np

if PARQUET_AVAILABLE:
    import pyarrow.parquet as pq
    import pyarrow as pa


class StorageBackend:
    """
    ストレージバックエンド基底クラス
    
    特徴量データの保存・読み込みインターフェースを定義します。
    """

    def __init__(self, config: FeatureStoreConfig):
        """
        初期化
        
        Args:
            config: 特徴量ストア設定
        """
        self.config = config

    def save(self, data: Dict[str, Any], file_path: Path):
        """
        データ保存（サブクラスで実装）
        
        Args:
            data: 保存するデータ
            file_path: 保存先ファイルパス
        """
        raise NotImplementedError

    def load(self, file_path: Path) -> Dict[str, Any]:
        """
        データ読み込み（サブクラスで実装）
        
        Args:
            file_path: 読み込み元ファイルパス
            
        Returns:
            読み込まれたデータ
        """
        raise NotImplementedError


class PickleBackend(StorageBackend):
    """
    Pickle形式のストレージバックエンド
    
    標準的なPython pickleを使用した保存・読み込み。
    圧縮もサポートします。
    """

    def save(self, data: Dict[str, Any], file_path: Path):
        """
        Pickle形式でデータを保存
        
        Args:
            data: 保存するデータ
            file_path: 保存先ファイルパス
        """
        if self.config.enable_compression:
            with gzip.open(file_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path: Path) -> Dict[str, Any]:
        """
        Pickle形式からデータを読み込み
        
        Args:
            file_path: 読み込み元ファイルパス
            
        Returns:
            読み込まれたデータ
        """
        if self.config.enable_compression and file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            with open(file_path, "rb") as f:
                return pickle.load(f)


class JoblibBackend(StorageBackend):
    """
    Joblib形式のストレージバックエンド
    
    NumPyデータの効率的な保存・読み込みに最適化されています。
    """

    def save(self, data: Dict[str, Any], file_path: Path):
        """
        Joblib形式でデータを保存
        
        Args:
            data: 保存するデータ
            file_path: 保存先ファイルパス
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is not available")
            
        if self.config.enable_compression:
            joblib.dump(data, file_path, compress=3)  # 圧縮レベル3
        else:
            joblib.dump(data, file_path)

    def load(self, file_path: Path) -> Dict[str, Any]:
        """
        Joblib形式からデータを読み込み
        
        Args:
            file_path: 読み込み元ファイルパス
            
        Returns:
            読み込まれたデータ
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is not available")
            
        return joblib.load(file_path)


class NumpyBackend(StorageBackend):
    """
    NumPy形式のストレージバックエンド
    
    NumPy配列の効率的な保存・読み込み。
    メタデータは別ファイルに保存されます。
    """

    def save(self, data: Dict[str, Any], file_path: Path):
        """
        NumPy形式でデータを保存
        
        Args:
            data: 保存するデータ
            file_path: 保存先ファイルパス
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is not available")
            
        features = data["features"]
        metadata = data["metadata"]

        # ファイル名から.npyを除去してベースパスを作成
        base_path = str(file_path).rstrip('.npy')

        # NumPy配列を保存
        features_path = f"{base_path}_features.npy"
        np.save(features_path, features)

        # メタデータはPickleで保存
        metadata_path = f"{base_path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, file_path: Path) -> Dict[str, Any]:
        """
        NumPy形式からデータを読み込み
        
        Args:
            file_path: 読み込み元ファイルパス
            
        Returns:
            読み込まれたデータ
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is not available")
            
        # ファイル名から.npyを除去してベースパスを作成
        base_path = str(file_path).rstrip('.npy')

        # NumPy配列を読み込み
        features_path = f"{base_path}_features.npy"
        features = np.load(features_path)

        # メタデータを読み込み
        metadata_path = f"{base_path}_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        return {"features": features, "metadata": metadata}


class ParquetBackend(StorageBackend):
    """
    Parquet形式のストレージバックエンド
    
    DataFrameの効率的な保存・読み込み。
    列指向でクエリ性能に優れています。
    """

    def save(self, data: Dict[str, Any], file_path: Path):
        """
        Parquet形式でデータを保存
        
        Args:
            data: 保存するデータ
            file_path: 保存先ファイルパス
        """
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow is not available")
            
        features = data["features"]
        metadata = data["metadata"]

        if hasattr(features, 'to_parquet'):
            # DataFrameの場合
            features.to_parquet(file_path, compression=self.config.parquet_compression)

            # メタデータは別ファイルで保存
            metadata_path = str(file_path).replace('.parquet', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        else:
            # DataFrameでない場合はfallback
            pickle_backend = PickleBackend(self.config)
            pickle_backend.save(data, file_path.with_suffix('.pkl'))

    def load(self, file_path: Path) -> Dict[str, Any]:
        """
        Parquet形式からデータを読み込み
        
        Args:
            file_path: 読み込み元ファイルパス
            
        Returns:
            読み込まれたデータ
        """
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow is not available")
            
        import pandas as pd

        # Parquetファイルを読み込み
        features = pd.read_parquet(file_path)

        # メタデータを読み込み
        metadata_path = str(file_path).replace('.parquet', '_metadata.pkl')
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        except FileNotFoundError:
            metadata = {}

        return {"features": features, "metadata": metadata}


class StorageManager:
    """
    ストレージマネージャー
    
    適切なストレージバックエンドを選択し、データの保存・読み込みを管理します。
    """

    def __init__(self, config: FeatureStoreConfig):
        """
        初期化
        
        Args:
            config: 特徴量ストア設定
        """
        self.config = config
        self._backends = {
            'pickle': PickleBackend(config),
            'joblib': JoblibBackend(config),
            'numpy': NumpyBackend(config),
            'parquet': ParquetBackend(config),
        }

    def determine_storage_format(self, feature_data: Dict[str, Any]) -> str:
        """
        データ特性に基づく保存形式の決定
        
        Args:
            feature_data: 保存するデータ
            
        Returns:
            保存形式 ('pickle', 'joblib', 'numpy', 'parquet')
        """
        if self.config.storage_format != "auto":
            return self.config.storage_format

        features = feature_data.get("features")

        # データサイズの推定
        data_size_mb = 0
        if hasattr(features, 'nbytes'):
            data_size_mb = features.nbytes / (1024 * 1024)
        elif hasattr(features, '__len__'):
            # 概算
            data_size_mb = len(str(feature_data)) / (1024 * 1024)

        # NumPy配列の場合
        if NUMPY_AVAILABLE and hasattr(features, 'dtype') and hasattr(features, 'shape'):
            if data_size_mb > self.config.auto_format_threshold_mb:
                return 'numpy'
            else:
                return 'joblib' if JOBLIB_AVAILABLE else 'pickle'

        # DataFrame の場合
        if hasattr(features, 'to_parquet') and PARQUET_AVAILABLE and self.config.enable_parquet:
            if data_size_mb > self.config.auto_format_threshold_mb:
                return 'parquet'

        # デフォルト
        if data_size_mb > self.config.auto_format_threshold_mb and JOBLIB_AVAILABLE:
            return 'joblib'
        else:
            return 'pickle'

    def get_file_path(self, base_path: Path, feature_id: str, storage_format: str) -> Path:
        """
        保存形式に応じたファイルパス取得
        
        Args:
            base_path: ベースパス
            feature_id: 特徴ID
            storage_format: 保存形式
            
        Returns:
            ファイルパス
        """
        # 2レベルディレクトリ構造（パフォーマンス向上）
        dir1 = feature_id[:2]
        dir2 = feature_id[2:4]

        feature_dir = base_path / dir1 / dir2
        feature_dir.mkdir(parents=True, exist_ok=True)

        # 拡張子マッピング
        extension_map = {
            'pickle': '.pkl',
            'joblib': '.joblib',
            'numpy': '.npy',
            'parquet': '.parquet'
        }

        extension = extension_map.get(storage_format, '.pkl')
        if self.config.enable_compression and storage_format in ['pickle', 'joblib']:
            extension += '.gz'

        filename = f"{feature_id}{extension}"
        return feature_dir / filename

    def detect_storage_format(self, file_path: Path) -> str:
        """
        ファイル拡張子からの保存形式判定
        
        Args:
            file_path: ファイルパス
            
        Returns:
            保存形式
        """
        suffix = file_path.suffix.lower()
        # 圧縮ファイルの場合は二重拡張子をチェック
        if suffix == '.gz':
            suffix = file_path.suffixes[-2].lower() if len(file_path.suffixes) >= 2 else '.gz'

        format_map = {
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.joblib': 'joblib',
            '.npy': 'numpy',
            '.parquet': 'parquet'
        }

        return format_map.get(suffix, 'pickle')

    def save_feature_data(self, feature_data: Dict[str, Any], file_path: Path, 
                         storage_format: str):
        """
        保存形式に応じたデータ保存
        
        Args:
            feature_data: 保存するデータ
            file_path: 保存先ファイルパス
            storage_format: 保存形式
        """
        if storage_format not in self._backends:
            logger.warning(f"未知の保存形式: {storage_format}, pickle形式で保存")
            storage_format = 'pickle'

        backend = self._backends[storage_format]
        backend.save(feature_data, file_path)

    def load_feature_data(self, file_path: Path, storage_format: str) -> Dict[str, Any]:
        """
        保存形式に応じたデータ読み込み
        
        Args:
            file_path: 読み込み元ファイルパス
            storage_format: 保存形式
            
        Returns:
            読み込まれたデータ
        """
        if storage_format not in self._backends:
            logger.warning(f"未知の保存形式: {storage_format}, pickle形式で読み込み")
            storage_format = 'pickle'

        backend = self._backends[storage_format]
        return backend.load(file_path)