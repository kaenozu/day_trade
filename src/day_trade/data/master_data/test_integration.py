#!/usr/bin/env python3
"""
マスターデータ管理（MDM）統合テスト
分割されたモジュールの統合動作確認
"""

import asyncio
import logging
from datetime import datetime

from . import (
    DataDomain,
    MasterDataManager,
    create_master_data_manager,
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_master_data_manager():
    """MDMシステム統合テスト"""
    print("=== マスターデータ管理（MDM）システム統合テスト ===")

    try:
        # MDMシステム初期化
        mdm = create_master_data_manager(
            storage_path="test_mdm_integrated",
            enable_cache=True,
            enable_audit=True,
            data_retention_days=30,
        )

        print("\n1. システム初期化完了")
        print(f"   ストレージパス: {mdm.storage_path}")
        print(f"   データ要素数: {len(mdm.default_setup.get_data_elements())}")
        print(f"   ガバナンスポリシー数: {len(mdm.governance_manager.governance_policies)}")
        print(f"   データスチュワード数: {len(mdm.governance_manager.data_stewards)}")

        # マスターエンティティ登録テスト
        print("\n2. マスターエンティティ登録テスト...")

        # 株式エンティティ登録
        stock_attributes = {
            "symbol": "7203",
            "company_name": "トヨタ自動車株式会社",
            "industry": "輸送用機器",
            "sector": "製造業",
            "last_price": 2500.0,
            "last_updated": datetime.utcnow().isoformat(),
            "market_cap": 35000000000000,  # 35兆円
            "listing_date": "1949-05-16",
        }

        stock_entity_id = await mdm.register_master_entity(
            entity_type="stock",
            primary_key="7203",
            attributes=stock_attributes,
            domain=DataDomain.SECURITY,
            source_system="market_data",
            steward_id="market_data_steward",
        )

        print(f"   株式エンティティ登録: {stock_entity_id}")

        # 企業エンティティ登録
        company_attributes = {
            "name": "トヨタ自動車株式会社",
            "industry": "輸送用機器",
            "founded": "1937-08-28",
            "headquarters": "愛知県豊田市",
            "employees": 366283,
            "website": "https://toyota.jp",
        }

        company_entity_id = await mdm.register_master_entity(
            entity_type="company",
            primary_key="toyota",
            attributes=company_attributes,
            domain=DataDomain.REFERENCE,
            steward_id="reference_data_steward",
        )

        print(f"   企業エンティティ登録: {company_entity_id}")

        # エンティティ取得テスト
        print("\n3. エンティティ取得テスト...")
        retrieved_stock = await mdm.get_master_entity(stock_entity_id)
        if retrieved_stock:
            print(f"   取得成功: {retrieved_stock.entity_type} - {retrieved_stock.primary_key}")
            print(f"   品質スコア: {retrieved_stock.data_quality_score:.3f}")
            print(f"   バージョン: {retrieved_stock.version}")

        # エンティティ更新テスト
        print("\n4. エンティティ更新テスト...")
        updated_attributes = {
            "last_price": 2520.0,
            "last_updated": datetime.utcnow().isoformat(),
            "price_change": 20.0,
            "price_change_percent": 0.8,
        }

        update_success = await mdm.update_master_entity(
            stock_entity_id, updated_attributes, "test_user"
        )

        print(f"   更新結果: {'成功' if update_success else '失敗'}")

        # エンティティ検索テスト
        print("\n5. エンティティ検索テスト...")

        # 株式エンティティ検索
        stock_entities = await mdm.search_master_entities(
            entity_type="stock", domain=DataDomain.SECURITY, limit=10
        )
        print(f"   株式エンティティ検索結果: {len(stock_entities)}件")

        # 全エンティティ検索
        all_entities = await mdm.search_master_entities(limit=100)
        print(f"   全エンティティ検索結果: {len(all_entities)}件")

        # データカタログ作成テスト
        print("\n6. データカタログ作成テスト...")
        catalog = await mdm.create_data_catalog()

        if "error" not in catalog:
            print(f"   カタログ生成時刻: {catalog['generated_at']}")
            print(f"   データ要素数: {len(catalog['data_elements'])}")
            print("   マスターエンティティ:")
            print(f"     総数: {catalog['master_entities']['total_count']}")
            print(f"     平均品質スコア: {catalog['master_entities']['average_quality_score']:.3f}")
            print(f"   ドメイン数: {len(catalog['domains'])}")
            print(f"   スチュワード数: {len(catalog['stewards'])}")
            print(f"   ガバナンスポリシー数: {len(catalog['governance_policies'])}")

        # MDMダッシュボード確認
        print("\n7. MDMダッシュボード確認...")
        dashboard = await mdm.get_mdm_dashboard()

        if "error" not in dashboard:
            print(f"   システム状態: {dashboard['system_status']}")
            stats = dashboard["statistics"]
            print("   統計情報:")
            print(f"     総エンティティ数: {stats['total_entities']}")
            print(f"     データ要素数: {stats['data_elements']}")
            print(f"     アクティブスチュワード数: {stats['active_stewards']}")

            if "quality_metrics" in dashboard and dashboard["quality_metrics"]:
                quality = dashboard["quality_metrics"]
                print("   品質メトリクス:")
                print(f"     平均品質スコア: {quality.get('average_quality_score', 0):.3f}")

            if "governance_status" in dashboard:
                governance = dashboard["governance_status"]
                print("   ガバナンス状況:")
                print(f"     総ポリシー数: {governance['total_policies']}")
                print(f"     アクティブポリシー数: {governance['active_policies']}")

            # ドメイン分布
            if "domain_distribution" in dashboard:
                print("   ドメイン分布:")
                for domain, count in dashboard["domain_distribution"].items():
                    print(f"     {domain}: {count}件")

            # システムアラート
            if "alerts" in dashboard and dashboard["alerts"]:
                print(f"   システムアラート: {len(dashboard['alerts'])}件")
                for alert in dashboard["alerts"][:3]:
                    print(f"     {alert['type']}: {alert['message']}")

            # 推奨事項
            if "recommendations" in dashboard and dashboard["recommendations"]:
                print(f"   改善推奨事項: {len(dashboard['recommendations'])}件")
                for rec in dashboard["recommendations"][:2]:
                    print(f"     {rec['title']}: {rec['description'][:50]}...")

        # 品質レポート生成テスト
        print("\n8. 品質レポート生成テスト...")
        if retrieved_stock:
            quality_report = mdm.quality_assessor.create_quality_report(retrieved_stock)
            if "error" not in quality_report:
                print(f"   エンティティ: {quality_report['entity_id']}")
                print(f"   総合スコア: {quality_report['overall_score']:.3f}")
                print(f"   完全性スコア: {quality_report['completeness']['score']:.3f}")
                print(f"   鮮度スコア: {quality_report['freshness']['score']:.3f}")
                if quality_report["recommendations"]:
                    print("   推奨事項:")
                    for rec in quality_report["recommendations"]:
                        print(f"     - {rec}")

        # ガバナンス準拠性チェック
        print("\n9. ガバナンス準拠性チェック...")
        if retrieved_stock:
            compliance = mdm.governance_manager.validate_policy_compliance(retrieved_stock)
            if "error" not in compliance:
                print(f"   準拠状況: {'準拠' if compliance['overall_compliant'] else '違反あり'}")
                print(f"   違反件数: {compliance['violation_count']}")
                if compliance["violations"]:
                    print("   違反内容:")
                    for violation in compliance["violations"][:3]:
                        print(f"     - {violation.get('description', 'N/A')}")

        # クリーンアップ
        await mdm.cleanup()

        print("\nマスターデータ管理システム統合テスト完了")

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # テスト実行
    asyncio.run(test_master_data_manager())