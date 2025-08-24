#!/usr/bin/env python3
"""
データ操作モジュール

データのシリアライゼーション、デシリアライゼーション、
ハッシュ計算、チェックサム算出など、データバージョン管理で
必要なデータ操作を提供します。

Classes:
    DataOperations: データ操作メインクラス
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Union

import pandas as pd

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DataOperations:
    """データ操作クラス
    
    データのシリアライゼーション・デシリアライゼーション、
    ハッシュ計算、チェックサム算出、サイズ計算などを提供します。
    """

    def __init__(self, data_storage_path: Path):
        """DataOperationsの初期化
        
        Args:
            data_storage_path: データファイル保存先ディレクトリ
        """
        self.data_storage_path = data_storage_path
        self.data_storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"データ操作マネージャー初期化完了: {data_storage_path}")

    async def save_version_data(self, version_id: str, data: Any) -> Path:
        """バージョンデータをファイルに保存
        
        Args:
            version_id: バージョンID
            data: 保存するデータ
            
        Returns:
            保存されたファイルのパス
            
        Raises:
            Exception: データ保存に失敗した場合
        """
        data_path = self.data_storage_path / f"{version_id}.json"

        try:
            # シリアライズ可能な形に変換
            serialized_data = self._serialize_data(data)

            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(
                    serialized_data, 
                    f, 
                    indent=2, 
                    ensure_ascii=False, 
                    default=str
                )

            logger.debug(f"バージョンデータ保存完了: {data_path}")
            return data_path

        except Exception as e:
            logger.error(f"バージョンデータ保存エラー ({version_id}): {e}")
            raise

    async def load_version_data(self, version_id: str) -> Any:
        """バージョンデータをファイルから読み込み
        
        Args:
            version_id: バージョンID
            
        Returns:
            読み込まれたデータ
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            Exception: データ読み込みに失敗した場合
        """
        data_path = self.data_storage_path / f"{version_id}.json"

        try:
            if not data_path.exists():
                raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")

            with open(data_path, encoding="utf-8") as f:
                serialized_data = json.load(f)

            # デシリアライズしてオリジナル形式に戻す
            data = self._deserialize_data(serialized_data)
            
            logger.debug(f"バージョンデータ読み込み完了: {version_id}")
            return data

        except Exception as e:
            logger.error(f"バージョンデータ読み込みエラー ({version_id}): {e}")
            raise

    def _serialize_data(self, data: Any) -> dict:
        """データをシリアライズ可能な形式に変換
        
        Args:
            data: シリアライズするデータ
            
        Returns:
            シリアライズされたデータ辞書
        """
        if isinstance(data, pd.DataFrame):
            return {
                "type": "DataFrame",
                "data": data.to_dict(orient="records"),
                "index": data.index.tolist(),
                "columns": data.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()},
                "shape": data.shape,
            }
        elif isinstance(data, pd.Series):
            return {
                "type": "Series",
                "data": data.to_dict(),
                "index": data.index.tolist(),
                "name": data.name,
                "dtype": str(data.dtype),
            }
        elif isinstance(data, (dict, list, str, int, float, bool)):
            return {
                "type": type(data).__name__,
                "data": data,
            }
        else:
            # その他の型は文字列として保存
            return {
                "type": type(data).__name__,
                "data": str(data),
                "is_string_representation": True,
            }

    def _deserialize_data(self, serialized_data: dict) -> Any:
        """シリアライズされたデータを元の形式に復元
        
        Args:
            serialized_data: シリアライズされたデータ
            
        Returns:
            復元されたデータ
        """
        data_type = serialized_data.get("type", "unknown")

        if data_type == "DataFrame":
            df = pd.DataFrame(serialized_data["data"])
            
            # インデックスを復元
            if "index" in serialized_data:
                df.index = serialized_data["index"]
                
            # データ型を復元（可能な場合）
            if "dtypes" in serialized_data:
                for col, dtype_str in serialized_data["dtypes"].items():
                    try:
                        if col in df.columns:
                            df[col] = df[col].astype(dtype_str)
                    except (ValueError, TypeError):
                        logger.warning(f"データ型復元失敗: {col} ({dtype_str})")
                        
            return df
            
        elif data_type == "Series":
            series = pd.Series(
                data=list(serialized_data["data"].values()),
                index=serialized_data["index"],
                name=serialized_data.get("name")
            )
            
            # データ型を復元
            if "dtype" in serialized_data:
                try:
                    series = series.astype(serialized_data["dtype"])
                except (ValueError, TypeError):
                    logger.warning(f"Series データ型復元失敗: {serialized_data['dtype']}")
                    
            return series
            
        elif data_type in ("dict", "list", "str", "int", "float", "bool"):
            return serialized_data["data"]
            
        else:
            # 文字列表現として保存されたデータ
            return serialized_data["data"]

    def calculate_data_hash(self, data: Any) -> str:
        """データのハッシュ値を計算
        
        大規模データに対応するため、サンプリングを使用します。
        
        Args:
            data: ハッシュ化するデータ
            
        Returns:
            SHA256ハッシュ値（16進数文字列）
        """
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame構造とデータのハッシュ
                content = (
                    f"{list(data.columns)}_{data.shape}_{str(data.dtypes.to_dict())}"
                )
                
                # 大規模データ対応：サンプリング
                if len(data) > 1000:
                    # 先頭、中間、末尾からサンプル
                    sample_indices = [
                        *range(min(100, len(data))),
                        *range(len(data) // 2 - 50, len(data) // 2 + 50),
                        *range(max(0, len(data) - 100), len(data))
                    ]
                    sample_data = data.iloc[sample_indices].to_json()
                    content += f"_sample_{sample_data}"
                else:
                    content += f"_full_{data.to_json()}"
                    
            elif isinstance(data, pd.Series):
                content = f"{data.name}_{data.dtype}_{len(data)}"
                if len(data) > 1000:
                    sample_data = data.head(100).to_json()
                    content += f"_sample_{sample_data}"
                else:
                    content += f"_full_{data.to_json()}"
                    
            elif isinstance(data, dict):
                content = json.dumps(data, sort_keys=True, default=str)
                
            elif isinstance(data, list):
                # 大規模リスト対応
                if len(data) > 1000:
                    sample_data = data[:100] + data[-100:]
                    content = json.dumps(sample_data, default=str) + f"_len_{len(data)}"
                else:
                    content = json.dumps(data, default=str)
                    
            else:
                content = str(data)

            return hashlib.sha256(content.encode()).hexdigest()

        except Exception as e:
            logger.error(f"データハッシュ計算エラー: {e}")
            return f"error_hash_{int(time.time())}"

    def calculate_checksum(self, data: Any) -> str:
        """データのチェックサムを計算
        
        Args:
            data: チェックサム計算対象データ
            
        Returns:
            MD5チェックサム（16進数文字列）
        """
        try:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                content = str(data.values.tobytes() if hasattr(data, 'values') else str(data))
            else:
                content = str(data)
                
            return hashlib.md5(content.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"チェックサム計算エラー: {e}")
            return f"error_checksum_{int(time.time())}"

    def get_data_size(self, data: Any) -> int:
        """データサイズを計算（バイト単位）
        
        Args:
            data: サイズ計算対象データ
            
        Returns:
            データサイズ（バイト）
        """
        try:
            if isinstance(data, pd.DataFrame):
                return int(data.memory_usage(deep=True).sum())
                
            elif isinstance(data, pd.Series):
                return int(data.memory_usage(deep=True))
                
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data, default=str).encode('utf-8'))
                
            else:
                return len(str(data).encode('utf-8'))
                
        except Exception as e:
            logger.warning(f"データサイズ計算エラー: {e}")
            return 0

    def get_file_count(self, data: Any) -> int:
        """データに関連するファイル数を計算
        
        現在の実装では常に1を返しますが、
        将来的に複数ファイルに分割保存する場合に拡張可能です。
        
        Args:
            data: ファイル数計算対象データ
            
        Returns:
            ファイル数（現在は常に1）
        """
        return 1

    def cleanup_version_data(self, version_id: str) -> bool:
        """バージョンデータファイルを削除
        
        Args:
            version_id: 削除対象のバージョンID
            
        Returns:
            削除成功した場合True
        """
        try:
            data_path = self.data_storage_path / f"{version_id}.json"
            
            if data_path.exists():
                data_path.unlink()
                logger.debug(f"バージョンデータファイル削除完了: {version_id}")
                return True
            else:
                logger.warning(f"削除対象ファイルが見つかりません: {version_id}")
                return False
                
        except Exception as e:
            logger.error(f"バージョンデータファイル削除エラー ({version_id}): {e}")
            return False

    def move_to_archive(self, version_id: str, archive_path: Path) -> bool:
        """バージョンデータをアーカイブディレクトリに移動
        
        Args:
            version_id: 移動対象のバージョンID  
            archive_path: アーカイブディレクトリのパス
            
        Returns:
            移動成功した場合True
        """
        try:
            source_path = self.data_storage_path / f"{version_id}.json"
            archive_path.mkdir(parents=True, exist_ok=True)
            target_path = archive_path / f"archived_{version_id}.json"
            
            if source_path.exists():
                source_path.rename(target_path)
                logger.debug(f"バージョンデータアーカイブ完了: {version_id}")
                return True
            else:
                logger.warning(f"アーカイブ対象ファイルが見つかりません: {version_id}")
                return False
                
        except Exception as e:
            logger.error(f"バージョンデータアーカイブエラー ({version_id}): {e}")
            return False

    def validate_data_integrity(self, version_id: str, expected_checksum: str) -> bool:
        """データの整合性を検証
        
        Args:
            version_id: 検証対象のバージョンID
            expected_checksum: 期待されるチェックサム
            
        Returns:
            整合性チェックが成功した場合True
        """
        try:
            data = await self.load_version_data(version_id)
            actual_checksum = self.calculate_checksum(data)
            
            is_valid = actual_checksum == expected_checksum
            
            if not is_valid:
                logger.warning(
                    f"データ整合性エラー ({version_id}): "
                    f"期待値={expected_checksum}, 実際値={actual_checksum}"
                )
                
            return is_valid
            
        except Exception as e:
            logger.error(f"データ整合性検証エラー ({version_id}): {e}")
            return False