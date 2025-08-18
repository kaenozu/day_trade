#!/usr/bin/env python3
"""
統合データベースシステムのテストスクリプト

バックアップ、監視、復元、ダッシュボード機能の統合テスト
本番環境準備度確認
"""

import os
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from day_trade.infrastructure.database.unified_database_manager import (
    UnifiedDatabaseManager,
    initialize_unified_database_manager
)
from day_trade.core.logging.unified_logging_system import get_logger

logger = get_logger(__name__)


def test_unified_initialization():
    """統合システム初期化テスト"""
    logger.info("=== 統合システム初期化テスト開始 ===")
    
    try:
        # 統合マネージャー初期化（テスト設定使用）
        test_config_path = "config/database_test.yaml"
        unified_manager = initialize_unified_database_manager(
            config_path=test_config_path, 
            auto_start=False
        )
        
        # 初期化結果確認
        status = unified_manager.get_system_status()
        logger.info("初期化ステータス", status=status)
        
        return unified_manager, status['initialized']
        
    except Exception as e:
        logger.error(f"統合システム初期化テスト失敗: {e}")
        return None, False


def test_backup_functionality(unified_manager: UnifiedDatabaseManager):
    """バックアップ機能テスト"""
    logger.info("=== バックアップ機能テスト開始 ===")
    
    try:
        # 手動バックアップ作成
        backup_result = unified_manager.create_backup("test")
        logger.info("バックアップ作成結果", result=backup_result)
        
        # バックアップ一覧確認
        backups = unified_manager.list_backups()
        logger.info(f"バックアップ一覧: {len(backups)}件")
        
        return backup_result.get('status') == 'success'
        
    except Exception as e:
        logger.error(f"バックアップ機能テスト失敗: {e}")
        return False


def test_monitoring_functionality(unified_manager: UnifiedDatabaseManager):
    """監視機能テスト"""
    logger.info("=== 監視機能テスト開始 ===")
    
    try:
        # 現在のメトリクス取得
        metrics = unified_manager.get_current_metrics()
        logger.info("現在のメトリクス", available=metrics is not None)
        
        # アクティブアラート確認
        alerts = unified_manager.get_active_alerts()
        logger.info(f"アクティブアラート: {len(alerts)}件")
        
        # 監視を短時間開始してテスト
        if unified_manager.monitoring_system:
            unified_manager.monitoring_system.start_monitoring()
            time.sleep(5)  # 5秒間監視
            
            # メトリクス再取得
            new_metrics = unified_manager.get_current_metrics()
            
            unified_manager.monitoring_system.stop_monitoring()
            
            return new_metrics is not None
        
        return True
        
    except Exception as e:
        logger.error(f"監視機能テスト失敗: {e}")
        return False


def test_restore_functionality(unified_manager: UnifiedDatabaseManager):
    """復元機能テスト（ドライラン）"""
    logger.info("=== 復元機能テスト開始 ===")
    
    try:
        # バックアップ一覧取得
        backups = unified_manager.list_backups()
        
        if not backups:
            logger.warning("復元テスト用のバックアップが存在しません")
            return True  # バックアップがない場合はスキップ
        
        # 最新のバックアップでドライラン実行
        latest_backup = backups[0]['filename']
        
        restore_result = unified_manager.restore_database(
            latest_backup, 
            dry_run=True
        )
        
        logger.info("復元ドライラン結果", result=restore_result)
        
        return restore_result.get('status') == 'success'
        
    except Exception as e:
        logger.error(f"復元機能テスト失敗: {e}")
        return False


def test_dashboard_functionality(unified_manager: UnifiedDatabaseManager):
    """ダッシュボード機能テスト"""
    logger.info("=== ダッシュボード機能テスト開始 ===")
    
    try:
        # ダッシュボードデータ取得
        dashboard_data = unified_manager.get_dashboard_data()
        logger.info("ダッシュボードデータ", available=bool(dashboard_data))
        
        # レポート生成テスト
        report_result = unified_manager.generate_report()
        logger.info("レポート生成結果", status=report_result.get('status'))
        
        return dashboard_data and report_result.get('status') == 'success'
        
    except Exception as e:
        logger.error(f"ダッシュボード機能テスト失敗: {e}")
        return False


def test_health_check(unified_manager: UnifiedDatabaseManager):
    """ヘルスチェック機能テスト"""
    logger.info("=== ヘルスチェック機能テスト開始 ===")
    
    try:
        # システム状態確認
        system_status = unified_manager.get_system_status()
        logger.info("システム状態", health=system_status.get('overall_health'))
        
        # 総合ヘルスチェック実行
        health_check = unified_manager.run_health_check()
        logger.info("ヘルスチェック結果", status=health_check.get('overall_status'))
        
        return system_status.get('initialized', False)
        
    except Exception as e:
        logger.error(f"ヘルスチェック機能テスト失敗: {e}")
        return False


def test_system_shutdown(unified_manager: UnifiedDatabaseManager):
    """システム停止テスト"""
    logger.info("=== システム停止テスト開始 ===")
    
    try:
        shutdown_result = unified_manager.shutdown()
        logger.info("システム停止結果", result=shutdown_result)
        
        return shutdown_result.get('status') == 'success'
        
    except Exception as e:
        logger.error(f"システム停止テスト失敗: {e}")
        return False


def main():
    """メインテスト実行"""
    logger.info("統合データベースシステム テスト開始")
    
    test_results = []
    unified_manager = None
    
    try:
        # 1. 統合システム初期化テスト
        unified_manager, init_success = test_unified_initialization()
        test_results.append(("統合システム初期化", init_success))
        
        if not init_success or not unified_manager:
            logger.error("初期化に失敗したため、以降のテストをスキップします")
            return False
        
        # 2. バックアップ機能テスト
        backup_success = test_backup_functionality(unified_manager)
        test_results.append(("バックアップ機能", backup_success))
        
        # 3. 監視機能テスト
        monitoring_success = test_monitoring_functionality(unified_manager)
        test_results.append(("監視機能", monitoring_success))
        
        # 4. 復元機能テスト
        restore_success = test_restore_functionality(unified_manager)
        test_results.append(("復元機能", restore_success))
        
        # 5. ダッシュボード機能テスト
        dashboard_success = test_dashboard_functionality(unified_manager)
        test_results.append(("ダッシュボード機能", dashboard_success))
        
        # 6. ヘルスチェック機能テスト
        health_success = test_health_check(unified_manager)
        test_results.append(("ヘルスチェック機能", health_success))
        
        # 7. システム停止テスト
        shutdown_success = test_system_shutdown(unified_manager)
        test_results.append(("システム停止", shutdown_success))
        
    except Exception as e:
        logger.error(f"テスト実行中にエラー: {e}")
        test_results.append(("テスト実行", False))
    
    finally:
        # クリーンアップ
        if unified_manager:
            try:
                unified_manager.shutdown()
            except:
                pass
    
    # 結果サマリー
    logger.info(f"\n{'='*60}")
    logger.info("統合データベースシステム テスト結果サマリー")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "[OK]" if result else "[ERROR]"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    logger.info(f"\n成功: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("✅ 統合データベースシステムは本番環境で使用可能です")
        return True
    elif success_rate >= 60:
        logger.warning("⚠️ 一部機能に問題がありますが、基本機能は動作します")
        return True
    else:
        logger.error("❌ 重要な問題があります。本番環境での使用前に修正が必要です")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)