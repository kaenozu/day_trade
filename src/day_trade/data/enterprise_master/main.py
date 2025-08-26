#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - メインクラス

このモジュールは、MDMシステムの統合インターフェースを提供します。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .catalog_manager import CatalogManager
from .change_manager import ChangeManager
from .database_operations import DatabaseOperations
from .entity_manager import EntityManager
from .enums import ChangeType, MasterDataType
from .governance_manager import GovernanceManager
from .hierarchy_manager import HierarchyManager
from .quality_manager import QualityManager

# 依存関係のインポートを試行
try:
    from ..monitoring.structured_logging_enhancement import (
        StructuredLoggingEnhancementSystem,
    )
    from ..utils.unified_cache_manager import UnifiedCacheManager
    from .comprehensive_data_quality_system import ComprehensiveDataQualitySystem
    from .enhanced_data_version_control import EnhancedDataVersionControl
    from .master_data_manager import MasterDataManager, MasterDataSet

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

    # Fallback definitions
    class MasterDataSet:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class EnterpriseMasterDataManagement:
    """エンタープライズマスターデータ管理システム - メインクラス
    
    各管理クラスを統合し、統一的なインターフェースを提供します。
    """
    
    def __init__(self, db_path: str = "enterprise_mdm.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path

        # コンポーネント初期化
        self.db_ops = DatabaseOperations(db_path)
        self.governance_manager = GovernanceManager()
        self.quality_manager = QualityManager()
        self.entity_manager = EntityManager(self.governance_manager, self.quality_manager)
        self.change_manager = ChangeManager(self.governance_manager, self.entity_manager)
        self.hierarchy_manager = HierarchyManager()
        self.catalog_manager = CatalogManager(self.db_ops, self.governance_manager)

        # 外部依存関係
        self.cache_manager = UnifiedCacheManager() if DEPENDENCIES_AVAILABLE else None
        self.comprehensive_quality_system = None
        self.version_control = None

        # 統計
        self.stats = {
            "total_entities": 0,
            "golden_records": 0,
            "pending_changes": 0,
            "data_quality_checks": 0,
            "governance_violations": 0,
        }

        self._initialize_external_components()

    def _initialize_external_components(self):
        """外部コンポーネント初期化"""
        if DEPENDENCIES_AVAILABLE:
            try:
                self.comprehensive_quality_system = ComprehensiveDataQualitySystem()
                self.version_control = EnhancedDataVersionControl()
                self.logger.info("外部コンポーネント初期化完了")
            except Exception as e:
                self.logger.warning(f"外部コンポーネント初期化エラー: {e}")

    # エンティティ管理メソッド
    async def register_master_data_entity(
        self,
        entity_type: MasterDataType,
        primary_key: str,
        attributes: Dict[str, Any],
        source_system: str,
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """マスターデータエンティティ登録
        
        Args:
            entity_type: エンティティタイプ
            primary_key: プライマリキー
            attributes: 属性データ
            source_system: ソースシステム
            created_by: 作成者
            metadata: メタデータ
            
        Returns:
            str: エンティティID
            
        Raises:
            ValueError: 登録に失敗した場合
        """
        try:
            success, entity_id, error_msg = await self.entity_manager.create_entity(
                entity_type, primary_key, attributes, source_system, created_by, metadata
            )
            
            if not success:
                raise ValueError(error_msg)

            # データベースに保存
            entity = await self.entity_manager.get_entity(entity_id)
            if entity:
                await self.db_ops.save_entity(entity)
                
                # 変更履歴記録
                await self.db_ops.record_change_history(
                    entity_id,
                    ChangeType.CREATE,
                    None,
                    1,
                    list(attributes.keys()),
                    created_by,
                    datetime.now(timezone.utc),
                    "Initial entity registration",
                    metadata or {}
                )

            # 統計更新
            self.stats["total_entities"] += 1
            if entity and entity.is_golden_record:
                self.stats["golden_records"] += 1

            self.logger.info(f"マスターデータエンティティ登録完了: {entity_id}")
            return entity_id

        except Exception as e:
            self.logger.error(f"エンティティ登録エラー: {e}")
            raise

    async def get_entity(self, entity_id: str) -> Optional[Any]:
        """エンティティ取得"""
        try:
            # まずメモリから取得を試行
            entity = await self.entity_manager.get_entity(entity_id)
            if entity:
                return entity

            # データベースから取得
            entity = await self.db_ops.get_entity(entity_id)
            return entity

        except Exception as e:
            self.logger.error(f"エンティティ取得エラー: {e}")
            return None

    # 変更管理メソッド
    async def request_data_change(
        self,
        entity_id: str,
        change_type: ChangeType,
        proposed_changes: Dict[str, Any],
        business_justification: str,
        requested_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """データ変更リクエスト作成
        
        Args:
            entity_id: 対象エンティティID
            change_type: 変更タイプ
            proposed_changes: 提案する変更内容
            business_justification: ビジネス上の根拠
            requested_by: リクエスト者
            metadata: メタデータ
            
        Returns:
            str: リクエストID
            
        Raises:
            ValueError: リクエスト作成に失敗した場合
        """
        try:
            success, request_id, error_msg = await self.change_manager.create_change_request(
                entity_id, change_type, proposed_changes, business_justification, requested_by, metadata
            )
            
            if not success:
                raise ValueError(error_msg)

            # データベースに保存
            change_request = await self.change_manager.get_change_request(request_id)
            if change_request:
                await self.db_ops.save_change_request(change_request)

            # 統計更新
            if change_request and change_request.approval_status.value == "pending":
                self.stats["pending_changes"] += 1

            self.logger.info(f"データ変更リクエスト作成: {request_id}")
            return request_id

        except Exception as e:
            self.logger.error(f"変更リクエスト作成エラー: {e}")
            raise

    async def approve_change_request(
        self, request_id: str, approved_by: str, approval_notes: str = ""
    ) -> bool:
        """変更リクエスト承認
        
        Args:
            request_id: リクエストID
            approved_by: 承認者
            approval_notes: 承認メモ
            
        Returns:
            bool: 承認成功フラグ
        """
        try:
            success, error_msg = await self.change_manager.approve_change_request(
                request_id, approved_by, approval_notes
            )
            
            if success:
                # データベース更新
                change_request = await self.change_manager.get_change_request(request_id)
                if change_request:
                    await self.db_ops.save_change_request(change_request)
                
                # 統計更新
                self.stats["pending_changes"] -= 1
                
                self.logger.info(f"変更リクエスト承認: {request_id}")
            else:
                self.logger.error(f"変更リクエスト承認エラー: {error_msg}")
                
            return success

        except Exception as e:
            self.logger.error(f"変更リクエスト承認エラー: {e}")
            return False

    async def reject_change_request(
        self, request_id: str, rejected_by: str, rejection_reason: str
    ) -> bool:
        """変更リクエスト却下
        
        Args:
            request_id: リクエストID
            rejected_by: 却下者
            rejection_reason: 却下理由
            
        Returns:
            bool: 却下成功フラグ
        """
        try:
            success, error_msg = await self.change_manager.reject_change_request(
                request_id, rejected_by, rejection_reason
            )
            
            if success:
                # データベース更新
                change_request = await self.change_manager.get_change_request(request_id)
                if change_request:
                    await self.db_ops.save_change_request(change_request)
                
                # 統計更新
                self.stats["pending_changes"] -= 1
                
                self.logger.info(f"変更リクエスト却下: {request_id}")
            else:
                self.logger.error(f"変更リクエスト却下エラー: {error_msg}")
                
            return success

        except Exception as e:
            self.logger.error(f"変更リクエスト却下エラー: {e}")
            return False

    # 階層管理メソッド
    async def create_data_hierarchy(
        self,
        name: str,
        entity_type: MasterDataType,
        root_entity_id: str,
        level_definitions: Dict[int, str],
        created_by: str = "system"
    ) -> str:
        """データ階層作成
        
        Args:
            name: 階層名
            entity_type: エンティティタイプ
            root_entity_id: ルートエンティティID
            level_definitions: レベル定義
            created_by: 作成者
            
        Returns:
            str: 階層ID
            
        Raises:
            ValueError: 階層作成に失敗した場合
        """
        try:
            success, hierarchy_id, error_msg = await self.hierarchy_manager.create_hierarchy(
                name, entity_type, root_entity_id, level_definitions, created_by=created_by
            )
            
            if not success:
                raise ValueError(error_msg)

            # データベース保存
            hierarchies = await self.hierarchy_manager.list_hierarchies()
            hierarchy = next((h for h in hierarchies if h.hierarchy_id == hierarchy_id), None)
            if hierarchy:
                await self.db_ops.save_hierarchy(hierarchy)

            self.logger.info(f"データ階層作成: {hierarchy_id}")
            return hierarchy_id

        except Exception as e:
            self.logger.error(f"階層作成エラー: {e}")
            raise

    # カタログ・リネージュメソッド
    async def get_data_catalog(
        self, entity_type: Optional[MasterDataType] = None
    ) -> Dict[str, Any]:
        """データカタログ取得
        
        Args:
            entity_type: フィルタ対象のエンティティタイプ
            
        Returns:
            Dict[str, Any]: データカタログ情報
        """
        try:
            catalog = await self.catalog_manager.generate_data_catalog(entity_type)
            
            # システム統計更新
            if "system_statistics" in catalog:
                system_stats = catalog["system_statistics"]
                self.stats.update({
                    "total_entities": system_stats.get("total_entities", 0),
                    "golden_records": system_stats.get("golden_records", 0),
                })

            return catalog

        except Exception as e:
            self.logger.error(f"データカタログ取得エラー: {e}")
            return {"error": str(e)}

    async def get_entity_lineage(self, entity_id: str) -> Dict[str, Any]:
        """エンティティデータ系譜取得
        
        Args:
            entity_id: エンティティID
            
        Returns:
            Dict[str, Any]: データ系譜情報
        """
        try:
            return await self.catalog_manager.get_entity_lineage(entity_id)

        except Exception as e:
            self.logger.error(f"データ系譜取得エラー: {e}")
            return {"entity_id": entity_id, "error": str(e)}

    # 品質管理メソッド
    async def run_quality_assessment(
        self, entity_type: Optional[MasterDataType] = None
    ) -> Dict[str, Any]:
        """品質評価実行
        
        Args:
            entity_type: エンティティタイプフィルタ
            
        Returns:
            Dict[str, Any]: 品質レポート
        """
        try:
            return await self.catalog_manager.generate_quality_report(entity_type)

        except Exception as e:
            self.logger.error(f"品質評価エラー: {e}")
            return {"error": str(e)}

    # ユーティリティメソッド
    async def get_system_statistics(self) -> Dict[str, Any]:
        """システム統計情報取得
        
        Returns:
            Dict[str, Any]: システム統計情報
        """
        try:
            # 最新統計情報更新
            entity_stats = await self.entity_manager.get_entity_statistics()
            change_stats = await self.change_manager.get_change_statistics()

            return {
                "entities": entity_stats,
                "changes": change_stats,
                "system_stats": self.stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"システム統計情報取得エラー: {e}")
            return {"error": str(e)}

    async def export_configuration(self) -> Dict[str, Any]:
        """設定エクスポート
        
        Returns:
            Dict[str, Any]: エクスポート設定データ
        """
        try:
            return {
                "governance_policies": self.governance_manager.export_policies(),
                "hierarchies": [h.to_dict() for h in await self.hierarchy_manager.list_hierarchies()],
                "system_configuration": {
                    "db_path": self.db_path,
                    "dependencies_available": DEPENDENCIES_AVAILABLE,
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            }

        except Exception as e:
            self.logger.error(f"設定エクスポートエラー: {e}")
            return {"error": str(e)}

    async def import_configuration(self, config_data: Dict[str, Any]) -> bool:
        """設定インポート
        
        Args:
            config_data: インポート設定データ
            
        Returns:
            bool: インポート成功フラグ
        """
        try:
            success = True
            
            # ガバナンスポリシーインポート
            if "governance_policies" in config_data:
                if not self.governance_manager.import_policies(config_data["governance_policies"]):
                    success = False

            # その他の設定項目をインポート（必要に応じて実装）

            self.logger.info(f"設定インポート: {'成功' if success else '部分的に失敗'}")
            return success

        except Exception as e:
            self.logger.error(f"設定インポートエラー: {e}")
            return False

    async def cleanup_and_optimize(self):
        """システムクリーンアップと最適化"""
        try:
            await self.db_ops.cleanup_database()
            self.logger.info("システムクリーンアップと最適化完了")

        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
            raise


# Factory function
def create_enterprise_mdm_system(
    db_path: str = "enterprise_mdm.db",
) -> EnterpriseMasterDataManagement:
    """エンタープライズMDMシステム作成
    
    Args:
        db_path: データベースファイルパス
        
    Returns:
        EnterpriseMasterDataManagement: MDMシステムインスタンス
    """
    return EnterpriseMasterDataManagement(db_path)


# メイン実行部分（テスト用）
if __name__ == "__main__":
    async def test_enterprise_mdm_system():
        """MDMシステム統合テスト"""
        print("=== エンタープライズマスターデータ管理システムテスト ===")

        try:
            # システム初期化
            mdm_system = create_enterprise_mdm_system("test_enterprise_mdm.db")

            print("\n1. MDMシステム初期化完了")
            print(f"   ガバナンスポリシー数: {len(mdm_system.governance_manager.policies)}")

            # 金融商品マスターデータ登録テスト
            print("\n2. 金融商品マスターデータ登録テスト...")
            
            test_stocks = [
                {
                    "symbol": "7203",
                    "name": "トヨタ自動車",
                    "isin": "JP3633400001",
                    "market": "TSE",
                    "sector": "自動車",
                },
                {
                    "symbol": "9984",
                    "name": "ソフトバンクグループ",
                    "isin": "JP3436100006",
                    "market": "TSE",
                    "sector": "情報通信",
                },
            ]

            entity_ids = []
            for stock in test_stocks:
                entity_id = await mdm_system.register_master_data_entity(
                    entity_type=MasterDataType.FINANCIAL_INSTRUMENTS,
                    primary_key=stock["symbol"],
                    attributes=stock,
                    source_system="topix_master",
                    created_by="test_user"
                )
                entity_ids.append(entity_id)
                print(f"   登録完了: {stock['symbol']} - {entity_id}")

            # データカタログ取得テスト
            print("\n3. データカタログ取得テスト...")
            catalog = await mdm_system.get_data_catalog()
            print(f"   エンティティタイプ数: {catalog.get('total_entity_types', 0)}")

            # 品質評価テスト
            print("\n4. 品質評価テスト...")
            quality_report = await mdm_system.run_quality_assessment()
            if "error" not in quality_report:
                print(f"   総エンティティ数: {quality_report.get('total_entities', 0)}")
                print(f"   平均品質スコア: {quality_report.get('average_quality_score', 0):.2f}")

            # システム統計取得テスト
            print("\n5. システム統計取得テスト...")
            stats = await mdm_system.get_system_statistics()
            if "error" not in stats:
                print(f"   システム統計: {stats.get('system_stats', {})}")

            print("\n[成功] エンタープライズマスターデータ管理システム統合テスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback
            traceback.print_exc()

    # テスト実行
    asyncio.run(test_enterprise_mdm_system())