#!/usr/bin/env python3
"""
データバージョン管理システム - データマネージャー

Issue #420: データ管理とデータ品質保証メカニズムの強化

データの保存・ロード・シリアライゼーション機能を提供
"""

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging
    
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DataManager:
    """データ管理クラス"""

    def __init__(self, repository_path: Path):
        """
        初期化
        
        Args:
            repository_path: リポジトリのパス
        """
        self.repository_path = repository_path
        self.data_path = repository_path / "data"

    async def save_version_data(self, version_id: str, data: Any) -> Path:
        """
        バージョンデータ保存
        
        Args:
            version_id: バージョンID
            data: 保存するデータ
            
        Returns:
            保存先のパス
            
        Raises:
            Exception: 保存に失敗した場合
        """
        data_file_path = self.data_path / f"{version_id}.json"

        try:
            # シリアライズ可能な形に変換
            serialized_data = self._serialize_data(data)

            with open(data_file_path, "w", encoding="utf-8") as f:
                json.dump(serialized_data, f, indent=2, ensure_ascii=False, default=str)

            logger.debug(f"データ保存完了: {data_file_path}")
            return data_file_path

        except Exception as e:
            logger.error(f"バージョンデータ保存エラー: {e}")
            raise

    async def load_version_data(self, version_id: str) -> Any:
        """
        バージョンデータロード
        
        Args:
            version_id: バージョンID
            
        Returns:
            ロードしたデータ
            
        Raises:
            FileNotFoundError: データファイルが見つからない場合
            Exception: ロードに失敗した場合
        """
        data_file_path = self.data_path / f"{version_id}.json"

        try:
            if not data_file_path.exists():
                raise FileNotFoundError(f"データファイルが見つかりません: {data_file_path}")

            with open(data_file_path, encoding="utf-8") as f:
                serialized_data = json.load(f)

            # デシリアライズ
            data = self._deserialize_data(serialized_data)
            
            logger.debug(f"データロード完了: {data_file_path}")
            return data

        except Exception as e:
            logger.error(f"バージョンデータロードエラー: {e}")
            raise

    def _serialize_data(self, data: Any) -> dict:
        """
        データのシリアライズ
        
        Args:
            data: シリアライズするデータ
            
        Returns:
            シリアライズされたデータ
        """
        if isinstance(data, pd.DataFrame):
            return {
                "type": "DataFrame",
                "data": data.to_dict(orient="records"),
                "index": data.index.tolist(),
                "columns": data.columns.tolist(),
                "dtypes": data.dtypes.to_dict(),
            }
        else:
            return {"type": type(data).__name__, "data": data}

    def _deserialize_data(self, serialized_data: dict) -> Any:
        """
        データのデシリアライズ
        
        Args:
            serialized_data: シリアライズされたデータ
            
        Returns:
            元のデータ
        """
        data_type = serialized_data.get("type", "unknown")

        if data_type == "DataFrame":
            df = pd.DataFrame(serialized_data["data"])
            if "index" in serialized_data:
                df.index = serialized_data["index"]
            return df
        else:
            return serialized_data["data"]

    def get_data_size(self, data: Any) -> int:
        """
        データサイズ計算（バイト）
        
        Args:
            data: サイズを計算するデータ
            
        Returns:
            データサイズ（バイト）
        """
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data).encode())
            else:
                return len(str(data).encode())
        except Exception:
            return 0

    def calculate_data_hash(self, data: Any) -> str:
        """
        データハッシュ計算
        
        Args:
            data: ハッシュを計算するデータ
            
        Returns:
            データハッシュ
        """
        import hashlib
        import time
        
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame構造とデータのハッシュ
                content = (
                    f"{list(data.columns)}_{data.shape}_{str(data.dtypes.to_dict())}"
                )
                if len(data) > 0:
                    # サンプルデータも含める（大規模データ対応）
                    sample_data = (
                        data.head(100).to_json() if len(data) > 100 else data.to_json()
                    )
                    content += f"_{sample_data}"
            elif isinstance(data, dict):
                content = json.dumps(data, sort_keys=True)
            elif isinstance(data, list):
                content = json.dumps(data)
            else:
                content = str(data)

            return hashlib.sha256(content.encode()).hexdigest()

        except Exception as e:
            logger.error(f"データハッシュ計算エラー: {e}")
            return f"error_hash_{int(time.time())}"

    def calculate_checksum(self, data: Any) -> str:
        """
        チェックサム計算
        
        Args:
            data: チェックサムを計算するデータ
            
        Returns:
            チェックサム値
        """
        import hashlib
        import time
        
        try:
            content = str(data)
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.error(f"チェックサム計算エラー: {e}")
            return f"error_checksum_{int(time.time())}"