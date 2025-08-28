#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - エンティティ管理

このモジュールは、マスターデータエンティティのCRUD操作と管理を担当します。
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .enums import DataGovernanceLevel, MasterDataType
from .models import MasterDataEntity
from .governance_manager import GovernanceManager
from .quality_manager import QualityManager


class EntityManager:
    """エンティティ管理クラス
    
    マスターデータエンティティの作成、読み取り、更新、削除を担当します。
    """
    
    def __init__(
        self,
        governance_manager: GovernanceManager,
        quality_manager: QualityManager
    ):
        self.logger = logging.getLogger(__name__)
        self.governance_manager = governance_manager
        self.quality_manager = quality_manager
        self._entity_cache: Dict[str, MasterDataEntity] = {}

    async def create_entity(
        self,
        entity_type: MasterDataType,
        primary_key: str,
        attributes: Dict[str, Any],
        source_system: str,
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """マスターデータエンティティ作成
        
        Args:
            entity_type: エンティティタイプ
            primary_key: プライマリキー
            attributes: 属性データ
            source_system: ソースシステム
            created_by: 作成者
            metadata: メタデータ
            
        Returns:
            Tuple[bool, str, Optional[str]]: (成功フラグ, エンティティID, エラーメッセージ)
        """
        try:
            metadata = metadata or {}
            current_time = datetime.now(timezone.utc)

            # エンティティID生成
            entity_id = self._generate_entity_id(entity_type, primary_key)

            # 重複チェック
            if entity_id in self._entity_cache:
                return False, "", f"エンティティID '{entity_id}' は既に存在します"

            # ガバナンスポリシー取得
            policy = self.governance_manager.get_applicable_policy(entity_type)
            governance_level = (
                policy.governance_level if policy else DataGovernanceLevel.STANDARD
            )

            # データバリデーション
            validation_result = await self.quality_manager.validate_entity_attributes(
                attributes, entity_type, policy
            )
            if not validation_result["is_valid"]:
                error_msg = f"データバリデーションエラー: {', '.join(validation_result['errors'])}"
                return False, "", error_msg

            # エンティティ作成
            entity = MasterDataEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                primary_key=primary_key,
                attributes=attributes,
                metadata={**metadata, "source_system": source_system},
                governance_level=governance_level,
                created_by=created_by,
                updated_by=created_by,
                source_systems=[source_system],
            )

            # 品質スコア計算
            quality_score, quality_metrics = await self.quality_manager.calculate_entity_quality(
                entity, policy
            )
            entity.quality_score = quality_score

            # ゴールデンレコード判定
            entity.is_golden_record = await self._determine_golden_record_status(entity, policy)

            # キャッシュに保存
            self._entity_cache[entity_id] = entity

            self.logger.info(f"エンティティ作成完了: {entity_id}")
            return True, entity_id, None

        except Exception as e:
            self.logger.error(f"エンティティ作成エラー: {e}")
            return False, "", str(e)

    async def get_entity(self, entity_id: str) -> Optional[MasterDataEntity]:
        """エンティティ取得
        
        Args:
            entity_id: エンティティID
            
        Returns:
            Optional[MasterDataEntity]: エンティティ、見つからない場合はNone
        """
        try:
            return self._entity_cache.get(entity_id)
        except Exception as e:
            self.logger.error(f"エンティティ取得エラー: {e}")
            return None

    async def update_entity(
        self,
        entity_id: str,
        updated_attributes: Dict[str, Any],
        updated_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """エンティティ更新
        
        Args:
            entity_id: エンティティID
            updated_attributes: 更新する属性
            updated_by: 更新者
            metadata: 追加メタデータ
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            entity = await self.get_entity(entity_id)
            if not entity:
                return False, f"エンティティが見つかりません: {entity_id}"

            # 更新前のバックアップ
            old_attributes = entity.attributes.copy()
            old_version = entity.version

            # 属性更新
            entity.attributes.update(updated_attributes)
            entity.version += 1
            entity.updated_at = datetime.now(timezone.utc)
            entity.updated_by = updated_by

            # メタデータ更新
            if metadata:
                entity.metadata.update(metadata)

            # データバリデーション
            policy = self.governance_manager.get_applicable_policy(entity.entity_type)
            validation_result = await self.quality_manager.validate_entity_attributes(
                entity.attributes, entity.entity_type, policy
            )
            if not validation_result["is_valid"]:
                # バリデーションエラーの場合、元に戻す
                entity.attributes = old_attributes
                entity.version = old_version
                error_msg = f"データバリデーションエラー: {', '.join(validation_result['errors'])}"
                return False, error_msg

            # 品質スコア再計算
            quality_score, quality_metrics = await self.quality_manager.calculate_entity_quality(
                entity, policy
            )
            entity.quality_score = quality_score

            # ゴールデンレコード状態再評価
            entity.is_golden_record = await self._determine_golden_record_status(entity, policy)

            self.logger.info(f"エンティティ更新完了: {entity_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"エンティティ更新エラー: {e}")
            return False, str(e)

    async def delete_entity(
        self, entity_id: str, deleted_by: str
    ) -> Tuple[bool, Optional[str]]:
        """エンティティ削除（論理削除）
        
        Args:
            entity_id: エンティティID
            deleted_by: 削除者
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            entity = await self.get_entity(entity_id)
            if not entity:
                return False, f"エンティティが見つかりません: {entity_id}"

            # 論理削除
            entity.is_active = False
            entity.updated_at = datetime.now(timezone.utc)
            entity.updated_by = deleted_by
            entity.metadata["deleted_at"] = entity.updated_at.isoformat()
            entity.metadata["deleted_by"] = deleted_by

            self.logger.info(f"エンティティ削除完了: {entity_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"エンティティ削除エラー: {e}")
            return False, str(e)

    async def archive_entity(
        self, entity_id: str, archived_by: str
    ) -> Tuple[bool, Optional[str]]:
        """エンティティアーカイブ
        
        Args:
            entity_id: エンティティID
            archived_by: アーカイブ実行者
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            entity = await self.get_entity(entity_id)
            if not entity:
                return False, f"エンティティが見つかりません: {entity_id}"

            # アーカイブ処理
            entity.is_active = False
            entity.updated_at = datetime.now(timezone.utc)
            entity.updated_by = archived_by
            entity.metadata["archived_at"] = entity.updated_at.isoformat()
            entity.metadata["archived_by"] = archived_by
            entity.metadata["archive_reason"] = "Manual archive"

            self.logger.info(f"エンティティアーカイブ完了: {entity_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"エンティティアーカイブエラー: {e}")
            return False, str(e)

    async def restore_entity(
        self, entity_id: str, restored_by: str
    ) -> Tuple[bool, Optional[str]]:
        """エンティティ復元
        
        Args:
            entity_id: エンティティID
            restored_by: 復元実行者
            
        Returns:
            Tuple[bool, Optional[str]]: (成功フラグ, エラーメッセージ)
        """
        try:
            entity = await self.get_entity(entity_id)
            if not entity:
                return False, f"エンティティが見つかりません: {entity_id}"

            if entity.is_active:
                return False, f"エンティティは既にアクティブです: {entity_id}"

            # 復元処理
            entity.is_active = True
            entity.updated_at = datetime.now(timezone.utc)
            entity.updated_by = restored_by
            entity.metadata["restored_at"] = entity.updated_at.isoformat()
            entity.metadata["restored_by"] = restored_by

            # 削除・アーカイブ関連のメタデータをクリア
            for key in ["deleted_at", "deleted_by", "archived_at", "archived_by", "archive_reason"]:
                entity.metadata.pop(key, None)

            self.logger.info(f"エンティティ復元完了: {entity_id}")
            return True, None

        except Exception as e:
            self.logger.error(f"エンティティ復元エラー: {e}")
            return False, str(e)

    async def list_entities(
        self,
        entity_type: Optional[MasterDataType] = None,
        is_active: Optional[bool] = None,
        is_golden_record: Optional[bool] = None,
        min_quality_score: Optional[float] = None
    ) -> List[MasterDataEntity]:
        """エンティティ一覧取得
        
        Args:
            entity_type: フィルタ対象のエンティティタイプ
            is_active: アクティブ状態フィルタ
            is_golden_record: ゴールデンレコードフィルタ
            min_quality_score: 最小品質スコア
            
        Returns:
            List[MasterDataEntity]: エンティティリスト
        """
        try:
            entities = []
            
            for entity in self._entity_cache.values():
                # フィルタ適用
                if entity_type and entity.entity_type != entity_type:
                    continue
                
                if is_active is not None and entity.is_active != is_active:
                    continue
                
                if is_golden_record is not None and entity.is_golden_record != is_golden_record:
                    continue
                
                if min_quality_score is not None and entity.quality_score < min_quality_score:
                    continue
                
                entities.append(entity)
            
            # 更新日時の降順でソート
            entities.sort(key=lambda e: e.updated_at, reverse=True)
            return entities

        except Exception as e:
            self.logger.error(f"エンティティ一覧取得エラー: {e}")
            return []

    async def search_entities(
        self,
        search_criteria: Dict[str, Any],
        entity_type: Optional[MasterDataType] = None
    ) -> List[MasterDataEntity]:
        """エンティティ検索
        
        Args:
            search_criteria: 検索条件（属性名: 検索値のマップ）
            entity_type: エンティティタイプフィルタ
            
        Returns:
            List[MasterDataEntity]: マッチするエンティティリスト
        """
        try:
            matching_entities = []
            
            for entity in self._entity_cache.values():
                if entity_type and entity.entity_type != entity_type:
                    continue
                
                # 検索条件チェック
                matches = True
                for attr_name, search_value in search_criteria.items():
                    entity_value = entity.attributes.get(attr_name)
                    
                    if entity_value is None:
                        matches = False
                        break
                    
                    # 部分一致検索（文字列の場合）
                    if isinstance(search_value, str) and isinstance(entity_value, str):
                        if search_value.lower() not in entity_value.lower():
                            matches = False
                            break
                    # 完全一致検索（その他の型）
                    elif entity_value != search_value:
                        matches = False
                        break
                
                if matches:
                    matching_entities.append(entity)
            
            return matching_entities

        except Exception as e:
            self.logger.error(f"エンティティ検索エラー: {e}")
            return []

    async def get_entity_statistics(self) -> Dict[str, Any]:
        """エンティティ統計情報取得
        
        Returns:
            Dict[str, Any]: 統計情報
        """
        try:
            stats = {
                "total_entities": len(self._entity_cache),
                "active_entities": 0,
                "golden_records": 0,
                "by_entity_type": {},
                "by_governance_level": {},
                "quality_distribution": {
                    "excellent": 0,  # 86-100
                    "good": 0,       # 71-85
                    "fair": 0,       # 51-70
                    "poor": 0        # 0-50
                },
                "average_quality_score": 0.0,
            }

            total_quality_score = 0.0
            
            for entity in self._entity_cache.values():
                # アクティブエンティティ数
                if entity.is_active:
                    stats["active_entities"] += 1
                
                # ゴールデンレコード数
                if entity.is_golden_record:
                    stats["golden_records"] += 1
                
                # エンティティタイプ別集計
                entity_type_key = entity.entity_type.value
                if entity_type_key not in stats["by_entity_type"]:
                    stats["by_entity_type"][entity_type_key] = 0
                stats["by_entity_type"][entity_type_key] += 1
                
                # ガバナンスレベル別集計
                governance_level_key = entity.governance_level.value
                if governance_level_key not in stats["by_governance_level"]:
                    stats["by_governance_level"][governance_level_key] = 0
                stats["by_governance_level"][governance_level_key] += 1
                
                # 品質分布
                quality_score = entity.quality_score
                total_quality_score += quality_score
                
                if quality_score >= 86:
                    stats["quality_distribution"]["excellent"] += 1
                elif quality_score >= 71:
                    stats["quality_distribution"]["good"] += 1
                elif quality_score >= 51:
                    stats["quality_distribution"]["fair"] += 1
                else:
                    stats["quality_distribution"]["poor"] += 1
            
            # 平均品質スコア
            if stats["total_entities"] > 0:
                stats["average_quality_score"] = total_quality_score / stats["total_entities"]
            
            return stats

        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {"error": str(e)}

    def _generate_entity_id(self, entity_type: MasterDataType, primary_key: str) -> str:
        """エンティティID生成
        
        Args:
            entity_type: エンティティタイプ
            primary_key: プライマリキー
            
        Returns:
            str: 生成されたエンティティID
        """
        timestamp = int(time.time())
        hash_part = hashlib.md5(primary_key.encode()).hexdigest()[:8]
        return f"{entity_type.value}_{hash_part}_{timestamp}"

    async def _determine_golden_record_status(
        self,
        entity: MasterDataEntity,
        policy: Optional[Any] = None
    ) -> bool:
        """ゴールデンレコード状態判定
        
        Args:
            entity: 評価対象エンティティ
            policy: ガバナンスポリシー
            
        Returns:
            bool: ゴールデンレコードかどうか
        """
        try:
            # 品質スコアしきい値
            quality_threshold = 90.0
            if policy:
                quality_threshold = policy.quality_threshold

            # 品質スコアチェック
            if entity.quality_score < quality_threshold:
                return False

            # ソースシステムチェック
            if not entity.source_systems:
                return False

            # エンティティタイプ別の特別条件
            if entity.entity_type == MasterDataType.FINANCIAL_INSTRUMENTS:
                # 金融商品の場合、ISINコードが必須
                if "isin" not in entity.attributes or not entity.attributes["isin"]:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"ゴールデンレコード判定エラー: {e}")
            return False

    async def bulk_update_entities(
        self,
        update_operations: List[Dict[str, Any]],
        updated_by: str
    ) -> Dict[str, Any]:
        """一括エンティティ更新
        
        Args:
            update_operations: 更新操作のリスト
            updated_by: 更新者
            
        Returns:
            Dict[str, Any]: 一括更新結果
        """
        try:
            results = {
                "total_operations": len(update_operations),
                "successful": 0,
                "failed": 0,
                "errors": []
            }

            for operation in update_operations:
                entity_id = operation.get("entity_id")
                updated_attributes = operation.get("attributes", {})
                metadata = operation.get("metadata", {})

                success, error_msg = await self.update_entity(
                    entity_id, updated_attributes, updated_by, metadata
                )

                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "entity_id": entity_id,
                        "error": error_msg
                    })

            return results

        except Exception as e:
            self.logger.error(f"一括更新エラー: {e}")
            return {
                "total_operations": len(update_operations),
                "successful": 0,
                "failed": len(update_operations),
                "errors": [{"error": str(e)}]
            }