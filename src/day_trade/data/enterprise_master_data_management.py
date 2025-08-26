#!/usr/bin/env python3
"""
エンタープライズマスターデータ管理（MDM）システム - メインエントリーポイント
Issue #420: データ管理とデータ品質保証メカニズムの強化

企業レベルのマスターデータ管理戦略:
- データ統合・統一化
- ゴールデンレコード管理
- データガバナンス・ポリシー
- 階層・分類管理
- データカタログ・メタデータ管理
- データ品質・整合性保証
- アクセス制御・セキュリティ
- 変更追跡・監査証跡

このファイルは、enterprise_master/ディレクトリに分割されたモジュールを統合します。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from .enterprise_master import (
        EnterpriseMasterDataManagement,
        create_enterprise_mdm_system,
    )
    from .enterprise_master.enums import (
        MasterDataType, 
        DataGovernanceLevel,
        ChangeType,
        ApprovalStatus,
    )
    from .enterprise_master.models import (
        MasterDataEntity,
        DataChangeRequest,
        DataGovernancePolicy,
        MasterDataHierarchy,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

    # Fallback definitions for backward compatibility
    class MasterDataSet:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# 後方互換性のためのエクスポート
__all__ = [
    "EnterpriseMasterDataManagement",
    "create_enterprise_mdm_system",
    "MasterDataType",
    "DataGovernanceLevel", 
    "ChangeType",
    "ApprovalStatus",
    "MasterDataEntity",
    "DataChangeRequest",
    "DataGovernancePolicy",
    "MasterDataHierarchy",
]


# テスト実行機能（元のファイルから移植）
if __name__ == "__main__":
    # テスト実行
    async def test_enterprise_mdm_system():
        print("=== エンタープライズマスターデータ管理システムテスト ===")

        try:
            # システム初期化
            mdm_system = create_enterprise_mdm_system("test_enterprise_mdm.db")

            print("\n1. MDMシステム初期化完了")
            print(f"   ガバナンスポリシー数: {len(mdm_system.governance_policies)}")

            # 金融商品マスターデータ登録
            print("\n2. 金融商品マスターデータ登録...")

            stock_entities = []

            # TOPIX500銘柄サンプル
            sample_stocks = [
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
                {
                    "symbol": "6758",
                    "name": "ソニーグループ",
                    "isin": "JP3435000009",
                    "market": "TSE",
                    "sector": "電気機器",
                },
                {
                    "symbol": "9432",
                    "name": "日本電信電話",
                    "isin": "JP3432600004",
                    "market": "TSE",
                    "sector": "情報通信",
                },
            ]

            for stock in sample_stocks:
                entity_id = await mdm_system.register_master_data_entity(
                    entity_type=MasterDataType.FINANCIAL_INSTRUMENTS,
                    primary_key=stock["symbol"],
                    attributes=stock,
                    source_system="topix_master",
                    created_by="data_admin",
                )
                stock_entities.append(entity_id)
                print(f"   登録完了: {stock['symbol']} - {entity_id}")

            # 市場コードマスター登録
            print("\n3. 市場コードマスター登録...")

            market_codes = [
                {"code": "TSE", "description": "東京証券取引所", "country": "JP"},
                {"code": "OSE", "description": "大阪証券取引所", "country": "JP"},
                {
                    "code": "NYSE",
                    "description": "ニューヨーク証券取引所",
                    "country": "US",
                },
                {"code": "NASDAQ", "description": "ナスダック", "country": "US"},
            ]

            for market in market_codes:
                entity_id = await mdm_system.register_master_data_entity(
                    entity_type=MasterDataType.EXCHANGE_CODES,
                    primary_key=market["code"],
                    attributes=market,
                    source_system="market_reference",
                    created_by="data_admin",
                )
                print(f"   市場コード登録: {market['code']} - {entity_id}")

            # データ変更リクエストテスト
            print("\n4. データ変更リクエストテスト...")

            change_request_id = await mdm_system.request_data_change(
                entity_id=stock_entities[0],
                change_type=ChangeType.UPDATE,
                proposed_changes={
                    "sector": "自動車・輸送機器",
                    "market_cap": 35000000000000,
                },
                business_justification="業界分類の詳細化とマーケットキャップ情報追加",
                requested_by="business_analyst",
            )

            print(f"   変更リクエスト作成: {change_request_id}")

            # 変更リクエスト承認
            approval_result = await mdm_system.approve_change_request(
                change_request_id, "data_steward", "業界分類詳細化は適切。承認します。"
            )

            print(f"   変更リクエスト承認: {'成功' if approval_result else '失敗'}")

            # データ階層作成テスト
            print("\n5. データ階層作成テスト...")

            hierarchy_id = await mdm_system.create_data_hierarchy(
                name="業界分類階層",
                entity_type=MasterDataType.FINANCIAL_INSTRUMENTS,
                root_entity_id="industry_root",
                level_definitions={1: "大分類", 2: "中分類", 3: "小分類"},
                created_by="data_architect",
            )

            print(f"   データ階層作成: {hierarchy_id}")

            # データカタログ取得
            print("\n6. データカタログ取得...")

            catalog = await mdm_system.get_data_catalog()
            print(f"   エンティティタイプ数: {catalog['total_entity_types']}")
            print("   システム統計:")

            for key, value in catalog["system_stats"].items():
                print(f"     {key}: {value}")

            print("   カタログ詳細:")
            for item in catalog["catalog"]:
                print(
                    f"     {item['entity_type']}: {item['total_count']}件 (ゴールデン: {item['golden_records']}件)"
                )
                print(
                    f"       平均品質: {item['average_quality']}, ガバナンス: {item.get('governance_level', 'N/A')}"
                )

            # エンティティ系譜取得
            print("\n7. エンティティデータ系譜取得...")

            lineage = await mdm_system.get_entity_lineage(stock_entities[0])
            print(f"   変更履歴: {len(lineage['change_history'])}件")
            print(f"   品質履歴: {len(lineage['quality_history'])}件")

            if lineage["change_history"]:
                latest_change = lineage["change_history"][0]
                print(
                    f"   最新変更: {latest_change['change_type']} by {latest_change['changed_by']}"
                )
                print(f"   変更フィールド: {latest_change['changed_fields']}")

            print("\n[成功] エンタープライズマスターデータ管理システムテスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_enterprise_mdm_system())