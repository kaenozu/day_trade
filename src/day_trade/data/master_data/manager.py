#!/usr/bin/env python3
"""
マスターデータ管理（MDM）メインマネージャー
統合されたマスターデータ管理システム
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .catalog_dashboard import CatalogDashboard
from .database_manager import DatabaseManager
from .default_setup import DefaultSetup
from .governance_manager import GovernanceManager
from .integration_rules import get_default_integration_rules
from .quality_assessor import QualityAssessor
from .types import DataDomain, DataLineage, MasterDataEntity, MasterDataStatus

try:
    from ...utils.logging_config import get_context_logger
    from ...utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class UnifiedCacheManager:
        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"mdm_key_{hash(str(args))}"


logger = get_context_logger(__name__)


class MasterDataManager:
    """マスターデータ管理システム"""

    def __init__(
        self,
        storage_path: str = "data/mdm",
        enable_cache: bool = True,
        enable_audit: bool = True,
        data_retention_days: int = 2555,  # 7年（金融業界標準）
    ):
        """初期化"""
        self.storage_path = Path(storage_path)
        self.enable_cache = enable_cache
        self.enable_audit = enable_audit
        self.data_retention_days = data_retention_days

        # ディレクトリ初期化
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # コンポーネント初期化
        self._initialize_components()

        # インメモリキャッシュ
        self.master_entities: Dict[str, MasterDataEntity] = {}
        self.data_lineages: Dict[str, DataLineage] = {}

        logger.info("マスターデータ管理システム初期化完了")
        logger.info(f"  - ストレージパス: {self.storage_path}")
        logger.info(f"  - キャッシュ: {'有効' if enable_cache else '無効'}")
        logger.info(f"  - 監査: {'有効' if enable_audit else '無効'}")
        logger.info(f"  - データ保持期間: {data_retention_days}日")

    def _initialize_components(self):
        """コンポーネント初期化"""
        # データベースマネージャー
        db_path = self.storage_path / "mdm.db"
        self.db_manager = DatabaseManager(db_path, self.enable_audit)

        # デフォルト設定
        self.default_setup = DefaultSetup()

        # 品質評価
        self.quality_assessor = QualityAssessor(self.default_setup.get_data_elements())

        # ガバナンス管理
        self.governance_manager = GovernanceManager()

        # カタログ・ダッシュボード
        self.catalog_dashboard = CatalogDashboard(self.storage_path)

        # データ統合ルール
        self.integration_rules = get_default_integration_rules()

        # キャッシュマネージャー初期化
        if self.enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=128,  # MDMは大量のマスターデータをキャッシュ
                    l2_memory_mb=512,
                    l3_disk_mb=2048,
                )
                logger.info("MDMキャッシュシステム初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None

    async def register_master_entity(
        self,
        entity_type: str,
        primary_key: str,
        attributes: Dict[str, Any],
        domain: DataDomain,
        source_system: str = "mdm",
        steward_id: Optional[str] = None,
    ) -> str:
        """マスターエンティティ登録"""
        entity_id = f"{entity_type}_{primary_key}_{int(time.time())}"

        logger.info(f"マスターエンティティ登録: {entity_id} ({entity_type})")

        try:
            # 既存エンティティチェック
            existing_entity = await self.db_manager.find_existing_entity(entity_type, primary_key)

            # データ統合実行
            if entity_type in self.integration_rules:
                rule = self.integration_rules[entity_type]
                integrated_attributes = await rule.integrate(
                    attributes, existing_entity.attributes if existing_entity else None
                )
            else:
                integrated_attributes = attributes

            # エンティティ作成
            entity = MasterDataEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                primary_key=primary_key,
                attributes=integrated_attributes,
                domain=domain,
                source_system=source_system,
                steward=steward_id,
                version=existing_entity.version + 1 if existing_entity else 1,
            )

            # データ品質評価
            entity.data_quality_score = await self.quality_assessor.assess_entity_quality(entity)

            # ガバナンスポリシー適用
            await self.governance_manager.apply_governance_policies(entity)

            # データベース保存
            await self.db_manager.save_entity(entity)

            # キャッシュ更新
            if self.cache_manager:
                cache_key = generate_unified_cache_key("mdm_entity", entity_id)
                self.cache_manager.put(cache_key, entity, priority=5.0)

            self.master_entities[entity_id] = entity

            # データ系譜記録
            if existing_entity:
                await self._record_data_lineage(
                    source_entity=existing_entity.entity_id,
                    target_entity=entity_id,
                    transformation="data_integration",
                    transformation_type="transform",
                )

            # 監査ログ記録
            await self.db_manager.record_audit_log(
                entity_type="master_entity",
                entity_id=entity_id,
                operation="register",
                new_values=entity.attributes,
                user_id=source_system,
            )

            logger.info(f"マスターエンティティ登録完了: {entity_id}")
            return entity_id

        except Exception as e:
            logger.error(f"マスターエンティティ登録エラー: {e}")
            raise

    async def get_master_entity(self, entity_id: str) -> Optional[MasterDataEntity]:
        """マスターエンティティ取得"""
        # インメモリキャッシュチェック
        if entity_id in self.master_entities:
            return self.master_entities[entity_id]

        # 統合キャッシュチェック
        if self.cache_manager:
            cache_key = generate_unified_cache_key("mdm_entity", entity_id)
            cached_entity = self.cache_manager.get(cache_key)
            if cached_entity:
                self.master_entities[entity_id] = cached_entity
                return cached_entity

        # データベース検索
        entity = await self.db_manager.get_entity(entity_id)
        if entity:
            # キャッシュ更新
            self.master_entities[entity_id] = entity
            if self.cache_manager:
                cache_key = generate_unified_cache_key("mdm_entity", entity_id)
                self.cache_manager.put(cache_key, entity, priority=4.0)

        return entity

    async def search_master_entities(
        self,
        entity_type: Optional[str] = None,
        domain: Optional[DataDomain] = None,
        status: Optional[MasterDataStatus] = None,
        primary_key_pattern: Optional[str] = None,
        limit: int = 100,
    ) -> List[MasterDataEntity]:
        """マスターエンティティ検索"""
        return await self.db_manager.search_entities(
            entity_type=entity_type,
            domain=domain,
            status=status,
            primary_key_pattern=primary_key_pattern,
            limit=limit,
        )

    async def update_master_entity(
        self, entity_id: str, attributes: Dict[str, Any], user_id: str = "system"
    ) -> bool:
        """マスターエンティティ更新"""
        logger.info(f"マスターエンティティ更新: {entity_id}")

        try:
            # 既存エンティティ取得
            existing_entity = await self.get_master_entity(entity_id)
            if not existing_entity:
                logger.warning(f"更新対象エンティティが見つかりません: {entity_id}")
                return False

            # 監査ログ用の変更前データ保存
            old_attributes = existing_entity.attributes.copy()

            # データ統合実行
            if existing_entity.entity_type in self.integration_rules:
                rule = self.integration_rules[existing_entity.entity_type]
                integrated_attributes = await rule.integrate(
                    attributes, existing_entity.attributes
                )
            else:
                integrated_attributes = {**existing_entity.attributes, **attributes}

            # エンティティ更新
            existing_entity.attributes = integrated_attributes
            existing_entity.version += 1
            existing_entity.updated_at = datetime.utcnow()

            # データ品質再評価
            existing_entity.data_quality_score = await self.quality_assessor.assess_entity_quality(
                existing_entity
            )

            # ガバナンスポリシー再適用
            await self.governance_manager.apply_governance_policies(existing_entity)

            # データベース更新
            await self.db_manager.update_entity(existing_entity)

            # キャッシュ更新
            self.master_entities[entity_id] = existing_entity
            if self.cache_manager:
                cache_key = generate_unified_cache_key("mdm_entity", entity_id)
                self.cache_manager.put(cache_key, existing_entity, priority=5.0)

            # 監査ログ記録
            await self.db_manager.record_audit_log(
                entity_type="master_entity",
                entity_id=entity_id,
                operation="update",
                old_values=old_attributes,
                new_values=integrated_attributes,
                user_id=user_id,
            )

            logger.info(f"マスターエンティティ更新完了: {entity_id}")
            return True

        except Exception as e:
            logger.error(f"マスターエンティティ更新エラー: {e}")
            return False

    async def create_data_catalog(self) -> Dict[str, Any]:
        """データカタログ作成"""
        logger.info("データカタログ作成中...")

        try:
            # 必要な情報を収集
            all_entities = await self.search_master_entities(limit=1000)
            quality_metrics = await self.db_manager.get_quality_metrics()
            lineage_summary = await self.db_manager.get_lineage_summary()

            # カタログ作成
            catalog = await self.catalog_dashboard.create_data_catalog(
                data_elements=self.default_setup.get_data_elements(),
                master_entities=all_entities,
                data_stewards=self.governance_manager.data_stewards,
                governance_policies=self.governance_manager.governance_policies,
                quality_metrics=quality_metrics,
                lineage_summary=lineage_summary,
            )

            return catalog

        except Exception as e:
            logger.error(f"データカタログ作成エラー: {e}")
            return {"error": str(e)}

    async def get_mdm_dashboard(self) -> Dict[str, Any]:
        """MDMダッシュボード情報取得"""
        try:
            # 必要な情報を収集
            quality_metrics = await self.db_manager.get_quality_metrics()
            governance_status = self.governance_manager.get_governance_status()
            recent_activities = await self.db_manager.get_recent_activities()
            domain_distribution = await self.db_manager.get_domain_distribution()

            # ダッシュボード作成
            dashboard = await self.catalog_dashboard.get_mdm_dashboard(
                master_entities=self.master_entities,
                data_elements=self.default_setup.get_data_elements(),
                data_stewards=self.governance_manager.data_stewards,
                data_lineages=self.data_lineages,
                quality_metrics=quality_metrics,
                governance_status=governance_status,
                recent_activities=recent_activities,
                domain_distribution=domain_distribution,
            )

            return dashboard

        except Exception as e:
            logger.error(f"MDMダッシュボード情報取得エラー: {e}")
            return {"error": str(e)}

    async def _record_data_lineage(
        self,
        source_entity: str,
        target_entity: str,
        transformation: str,
        transformation_type: str,
        confidence: float = 1.0,
    ):
        """データ系譜記録"""
        lineage_id = f"lineage_{int(time.time())}_{hash(f'{source_entity}_{target_entity}')}"

        lineage = DataLineage(
            lineage_id=lineage_id,
            source_entity=source_entity,
            target_entity=target_entity,
            transformation=transformation,
            transformation_type=transformation_type,
            confidence=confidence,
        )

        self.data_lineages[lineage_id] = lineage
        await self.db_manager.save_lineage(lineage)

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("マスターデータ管理システム クリーンアップ開始")

        # インメモリキャッシュクリア
        self.master_entities.clear()
        self.data_lineages.clear()

        # 古いログデータクリーンアップ（保持期間超過）
        deleted_count = await self.db_manager.cleanup_old_audit_logs(self.data_retention_days)
        if deleted_count > 0:
            logger.info(f"古い監査ログクリーンアップ: {deleted_count}件")

        logger.info("マスターデータ管理システム クリーンアップ完了")


# Factory function
def create_master_data_manager(
    storage_path: str = "data/mdm",
    enable_cache: bool = True,
    enable_audit: bool = True,
    data_retention_days: int = 2555,
) -> MasterDataManager:
    """マスターデータ管理システム作成"""
    return MasterDataManager(
        storage_path=storage_path,
        enable_cache=enable_cache,
        enable_audit=enable_audit,
        data_retention_days=data_retention_days,
    )