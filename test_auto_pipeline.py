#!/usr/bin/env python3
"""
自動パイプラインマネージャーのテスト

Issue #456: 自動データ収集・学習パイプラインの統合
"""

import asyncio
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.day_trade.automation.auto_pipeline_manager import run_auto_pipeline


async def test_pipeline():
    print("=== 自動パイプライン テスト開始 ===")
    print()
    
    try:
        # 限定的なテスト（主要5銘柄のみ）
        test_symbols = ["7203", "8306", "9984", "6758", "4689"]
        
        print(f"テスト対象: {len(test_symbols)} 銘柄")
        print("実行中...")
        print()
        
        result = await run_auto_pipeline(test_symbols)
        
        if result.success:
            print("[OK] 自動パイプライン成功!")
            print(f"   実行時間: {result.execution_time:.2f}秒")
            print(f"   最終段階: {result.final_stage.value}")
            print()
            
            # データ収集結果
            data_result = result.data_collection
            print("[DATA] データ収集結果:")
            print(f"   成功銘柄: {len(data_result.collected_symbols)}")
            print(f"   失敗銘柄: {len(data_result.failed_symbols)}")
            print(f"   総レコード数: {data_result.total_records:,}")
            print(f"   データ品質スコア: {data_result.data_quality_score:.2f}")
            print(f"   収集時間: {data_result.collection_time:.2f}秒")
            print()
            
            # ML学習結果
            ml_result = result.model_update
            print("[ML] ML学習結果:")
            print(f"   更新モデル数: {len(ml_result.models_updated)}")
            print(f"   学習時間: {ml_result.training_time:.2f}秒")
            print(f"   モデルバージョン: {ml_result.model_version}")
            print(f"   性能改善: {ml_result.improvement_percentage:.2f}%")
            if ml_result.performance_metrics:
                print(f"   パフォーマンス指標: {len(ml_result.performance_metrics)} 銘柄")
            print()
            
            # 品質レポート
            quality = result.quality_report
            print("[QUALITY] 品質レポート:")
            print(f"   総合スコア: {quality.overall_score:.2f}")
            print(f"   データ一貫性: {quality.data_consistency_score:.2f}")
            print(f"   鮮度スコア: {quality.freshness_score:.2f}")
            print(f"   推奨アクション: {quality.recommendation}")
            if quality.issues_found:
                print(f"   発見された問題: {len(quality.issues_found)} 件")
                for issue in quality.issues_found[:3]:  # 最初の3件のみ表示
                    print(f"     - {issue}")
            print()
            
            # 段階別結果
            print("[TIME] 段階別実行時間:")
            for stage, stage_result in result.stage_results.items():
                duration = stage_result.get('duration', 0)
                print(f"   {stage.value}: {duration:.2f}秒")
            print()
            
            print(f"[RESULT] 推奨生成数: {result.recommendations_generated}")
            
        else:
            print("[ERROR] 自動パイプライン失敗")
            print(f"   エラー: {result.error_message}")
            print(f"   最終段階: {result.final_stage.value}")
            print(f"   実行時間: {result.execution_time:.2f}秒")
            
    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_pipeline())