#!/usr/bin/env python3
"""
差分計算モジュール

データバージョン管理システムでのバージョン間差分計算機能を提供します。
DataFrameやメタデータの詳細な差分分析が可能です。

Classes:
    DiffCalculator: 差分計算メインクラス
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from .data_operations import DataOperations
from .types import DataVersion

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DiffCalculator:
    """差分計算クラス
    
    バージョン間のデータ差分、メタデータ差分、統計差分を計算します。
    特にDataFrameに対しては詳細な分析機能を提供します。
    """

    def __init__(self, data_operations: DataOperations):
        """DiffCalculatorの初期化
        
        Args:
            data_operations: データ操作インスタンス
        """
        self.data_ops = data_operations
        logger.info("差分計算マネージャー初期化完了")

    async def calculate_version_diff(
        self, version1: DataVersion, version2: DataVersion
    ) -> Dict[str, Any]:
        """バージョン間の包括的な差分を計算
        
        Args:
            version1: 比較元バージョン
            version2: 比較先バージョン
            
        Returns:
            詳細な差分情報を含む辞書
        """
        logger.info(
            f"バージョン差分計算開始: {version1.version_id} <-> {version2.version_id}"
        )

        try:
            # バージョン基本情報
            diff_result = {
                "version1": version1.version_id,
                "version2": version2.version_id,
                "timestamp1": version1.timestamp.isoformat(),
                "timestamp2": version2.timestamp.isoformat(),
                "time_diff_seconds": (
                    version2.timestamp - version1.timestamp
                ).total_seconds(),
                "hash_changed": version1.data_hash != version2.data_hash,
                "size_diff_bytes": version2.size_bytes - version1.size_bytes,
                "author1": version1.author,
                "author2": version2.author,
                "branch1": version1.branch,
                "branch2": version2.branch,
            }

            # メタデータ差分
            diff_result["metadata_diff"] = self.calculate_metadata_diff(
                version1.metadata, version2.metadata
            )

            # データ差分（データを読み込んで計算）
            try:
                data1 = await self.data_ops.load_version_data(version1.version_id)
                data2 = await self.data_ops.load_version_data(version2.version_id)
                
                diff_result["data_diff"] = await self.calculate_data_diff(data1, data2)
                
            except Exception as e:
                logger.warning(f"データ差分計算エラー: {e}")
                diff_result["data_diff"] = {
                    "error": str(e),
                    "calculation_failed": True
                }

            logger.info(f"バージョン差分計算完了: {version1.version_id} <-> {version2.version_id}")
            return diff_result

        except Exception as e:
            logger.error(f"バージョン差分計算エラー: {e}")
            raise

    def calculate_metadata_diff(
        self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """メタデータ差分を計算
        
        Args:
            metadata1: 比較元メタデータ
            metadata2: 比較先メタデータ
            
        Returns:
            メタデータ差分情報
        """
        diff = {
            "added": {},
            "removed": {},
            "changed": {},
            "unchanged": {},
            "summary": {
                "total_keys": len(set(metadata1.keys()) | set(metadata2.keys())),
                "added_count": 0,
                "removed_count": 0,
                "changed_count": 0,
                "unchanged_count": 0,
            }
        }

        all_keys = set(metadata1.keys()) | set(metadata2.keys())

        for key in all_keys:
            if key not in metadata1:
                diff["added"][key] = metadata2[key]
                diff["summary"]["added_count"] += 1
            elif key not in metadata2:
                diff["removed"][key] = metadata1[key]
                diff["summary"]["removed_count"] += 1
            elif metadata1[key] != metadata2[key]:
                diff["changed"][key] = {
                    "old_value": metadata1[key],
                    "new_value": metadata2[key]
                }
                diff["summary"]["changed_count"] += 1
            else:
                diff["unchanged"][key] = metadata1[key]
                diff["summary"]["unchanged_count"] += 1

        return diff

    async def calculate_data_diff(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """データ差分を計算
        
        Args:
            data1: 比較元データ
            data2: 比較先データ
            
        Returns:
            データ差分情報
        """
        diff = {
            "type1": type(data1).__name__,
            "type2": type(data2).__name__,
            "type_changed": type(data1) != type(data2),
        }

        # データ型が異なる場合
        if diff["type_changed"]:
            diff["type_change_details"] = {
                "from_type": type(data1).__name__,
                "to_type": type(data2).__name__,
                "content_summary": "データ型が変更されました"
            }
            return diff

        # DataFrame同士の比較
        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            diff.update(await self._calculate_dataframe_diff(data1, data2))
        
        # Series同士の比較
        elif isinstance(data1, pd.Series) and isinstance(data2, pd.Series):
            diff.update(self._calculate_series_diff(data1, data2))
        
        # 辞書同士の比較
        elif isinstance(data1, dict) and isinstance(data2, dict):
            diff.update(self._calculate_dict_diff(data1, data2))
        
        # リスト同士の比較
        elif isinstance(data1, list) and isinstance(data2, list):
            diff.update(self._calculate_list_diff(data1, data2))
        
        # その他の型
        else:
            diff.update(self._calculate_basic_diff(data1, data2))

        return diff

    async def _calculate_dataframe_diff(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> Dict[str, Any]:
        """DataFrameの詳細差分を計算
        
        Args:
            df1: 比較元DataFrame
            df2: 比較先DataFrame
            
        Returns:
            DataFrame差分情報
        """
        diff = {
            "shape1": df1.shape,
            "shape2": df2.shape,
            "shape_changed": df1.shape != df2.shape,
            "row_count_diff": len(df2) - len(df1),
            "column_count_diff": len(df2.columns) - len(df1.columns),
        }

        # カラム差分
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        diff["columns"] = {
            "added": list(cols2 - cols1),
            "removed": list(cols1 - cols2),
            "common": list(cols1 & cols2),
            "reordered": list(df1.columns) != list(df2.columns) if cols1 == cols2 else None
        }

        # データ型差分（共通カラムのみ）
        common_cols = cols1 & cols2
        dtype_changes = {}
        
        for col in common_cols:
            if str(df1[col].dtype) != str(df2[col].dtype):
                dtype_changes[col] = {
                    "old_dtype": str(df1[col].dtype),
                    "new_dtype": str(df2[col].dtype)
                }
        
        diff["dtype_changes"] = dtype_changes

        # インデックス比較
        diff["index"] = {
            "type1": type(df1.index).__name__,
            "type2": type(df2.index).__name__,
            "length1": len(df1.index),
            "length2": len(df2.index),
            "identical": df1.index.equals(df2.index) if len(df1) == len(df2) else False
        }

        # 統計サマリー（共通カラムの数値列のみ）
        numeric_summaries = {}
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                try:
                    summary1 = df1[col].describe()
                    summary2 = df2[col].describe()
                    
                    numeric_summaries[col] = {
                        "mean_diff": summary2['mean'] - summary1['mean'] if 'mean' in summary1 else None,
                        "std_diff": summary2['std'] - summary1['std'] if 'std' in summary1 else None,
                        "min_diff": summary2['min'] - summary1['min'] if 'min' in summary1 else None,
                        "max_diff": summary2['max'] - summary1['max'] if 'max' in summary1 else None,
                    }
                except Exception as e:
                    logger.warning(f"統計計算エラー ({col}): {e}")

        diff["numeric_summaries"] = numeric_summaries

        # 空値比較
        null_comparison = {}
        for col in common_cols:
            try:
                null1 = df1[col].isnull().sum()
                null2 = df2[col].isnull().sum()
                if null1 != null2:
                    null_comparison[col] = {
                        "null_count1": int(null1),
                        "null_count2": int(null2),
                        "null_diff": int(null2 - null1)
                    }
            except Exception as e:
                logger.warning(f"空値比較エラー ({col}): {e}")

        diff["null_comparison"] = null_comparison

        return diff

    def _calculate_series_diff(self, s1: pd.Series, s2: pd.Series) -> Dict[str, Any]:
        """Seriesの差分を計算
        
        Args:
            s1: 比較元Series
            s2: 比較先Series
            
        Returns:
            Series差分情報
        """
        diff = {
            "length1": len(s1),
            "length2": len(s2),
            "length_diff": len(s2) - len(s1),
            "name1": s1.name,
            "name2": s2.name,
            "name_changed": s1.name != s2.name,
            "dtype1": str(s1.dtype),
            "dtype2": str(s2.dtype),
            "dtype_changed": str(s1.dtype) != str(s2.dtype),
        }

        # 統計比較（数値データの場合）
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            try:
                diff["statistics"] = {
                    "mean_diff": s2.mean() - s1.mean(),
                    "std_diff": s2.std() - s1.std(),
                    "min_diff": s2.min() - s1.min(),
                    "max_diff": s2.max() - s1.max(),
                }
            except Exception as e:
                logger.warning(f"Series統計計算エラー: {e}")

        return diff

    def _calculate_dict_diff(self, dict1: Dict, dict2: Dict) -> Dict[str, Any]:
        """辞書の差分を計算
        
        Args:
            dict1: 比較元辞書
            dict2: 比較先辞書
            
        Returns:
            辞書差分情報
        """
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        diff = {
            "size1": len(dict1),
            "size2": len(dict2),
            "size_diff": len(dict2) - len(dict1),
            "keys": {
                "added": list(set(dict2.keys()) - set(dict1.keys())),
                "removed": list(set(dict1.keys()) - set(dict2.keys())),
                "common": list(set(dict1.keys()) & set(dict2.keys())),
            }
        }

        # 値の変更チェック
        value_changes = {}
        for key in diff["keys"]["common"]:
            if dict1[key] != dict2[key]:
                value_changes[key] = {
                    "old_value": dict1[key],
                    "new_value": dict2[key],
                    "type_changed": type(dict1[key]) != type(dict2[key])
                }

        diff["value_changes"] = value_changes
        diff["value_changes_count"] = len(value_changes)

        return diff

    def _calculate_list_diff(self, list1: List, list2: List) -> Dict[str, Any]:
        """リストの差分を計算
        
        Args:
            list1: 比較元リスト
            list2: 比較先リスト
            
        Returns:
            リスト差分情報
        """
        diff = {
            "length1": len(list1),
            "length2": len(list2),
            "length_diff": len(list2) - len(list1),
        }

        # セットとして比較
        try:
            set1 = set(list1)
            set2 = set(list2)
            
            diff["unique_elements"] = {
                "added": list(set2 - set1),
                "removed": list(set1 - set2),
                "common": list(set1 & set2),
            }
            
            # 順序の比較
            if len(list1) == len(list2):
                diff["order_changed"] = list1 != list2
                diff["identical"] = list1 == list2
                
        except (TypeError, AttributeError):
            # ハッシュできない要素がある場合
            diff["set_comparison_failed"] = True
            diff["identical"] = list1 == list2

        return diff

    def _calculate_basic_diff(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """基本データ型の差分を計算
        
        Args:
            data1: 比較元データ
            data2: 比較先データ
            
        Returns:
            基本差分情報
        """
        diff = {
            "identical": data1 == data2,
            "value1": str(data1),
            "value2": str(data2),
        }

        # 数値の場合は差分を計算
        try:
            if isinstance(data1, (int, float)) and isinstance(data2, (int, float)):
                diff["numeric_diff"] = data2 - data1
                diff["percentage_change"] = ((data2 - data1) / data1 * 100) if data1 != 0 else None
        except Exception:
            pass

        # 文字列の場合は長さを比較
        if isinstance(data1, str) and isinstance(data2, str):
            diff["length1"] = len(data1)
            diff["length2"] = len(data2)
            diff["length_diff"] = len(data2) - len(data1)

        return diff

    def calculate_change_summary(self, diff_result: Dict[str, Any]) -> str:
        """差分結果から変更サマリーを生成
        
        Args:
            diff_result: 差分計算結果
            
        Returns:
            人間が読みやすい変更サマリー
        """
        try:
            summary_parts = []
            
            # 基本情報
            v1 = diff_result.get("version1", "v1")
            v2 = diff_result.get("version2", "v2")
            summary_parts.append(f"バージョン {v1} から {v2} への変更")

            # サイズ変更
            size_diff = diff_result.get("size_diff_bytes", 0)
            if size_diff != 0:
                direction = "増加" if size_diff > 0 else "減少"
                summary_parts.append(f"サイズ{direction}: {abs(size_diff):,} bytes")

            # データ差分サマリー
            data_diff = diff_result.get("data_diff", {})
            if data_diff.get("type_changed"):
                summary_parts.append("データ型が変更されました")
            
            # DataFrame固有の情報
            if "shape1" in data_diff and "shape2" in data_diff:
                shape1 = data_diff["shape1"]
                shape2 = data_diff["shape2"]
                if shape1 != shape2:
                    summary_parts.append(f"形状変更: {shape1} → {shape2}")

            # メタデータ変更
            metadata_diff = diff_result.get("metadata_diff", {})
            metadata_summary = metadata_diff.get("summary", {})
            changed_count = metadata_summary.get("changed_count", 0)
            added_count = metadata_summary.get("added_count", 0)
            removed_count = metadata_summary.get("removed_count", 0)
            
            if changed_count > 0 or added_count > 0 or removed_count > 0:
                meta_changes = []
                if changed_count > 0:
                    meta_changes.append(f"変更{changed_count}件")
                if added_count > 0:
                    meta_changes.append(f"追加{added_count}件")
                if removed_count > 0:
                    meta_changes.append(f"削除{removed_count}件")
                summary_parts.append(f"メタデータ変更: {', '.join(meta_changes)}")

            return "; ".join(summary_parts) if summary_parts else "変更なし"

        except Exception as e:
            logger.error(f"変更サマリー生成エラー: {e}")
            return "変更サマリー生成失敗"