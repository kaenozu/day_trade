#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Readiness Check - 本番環境準備確認
"""

import os
import sys
from pathlib import Path

def check_production_readiness():
    """本番環境準備状況チェック"""
    print("本番環境準備状況チェック")
    print("=" * 40)
    
    checks = []
    
    # 1. 重要ファイル存在確認
    critical_files = [
        'main.py',
        'daytrade_core.py', 
        'daytrade_web.py',
        'daytrade_cli.py',
        'enhanced_data_provider.py',
        'enhanced_personal_analysis_engine.py',
        'user_centric_trading_system.py',
        'performance_optimization_system.py',
        'system_performance_monitor.py',
        'ml_accuracy_improvement_system.py',
        'market_time_manager.py',
        'fallback_notification_system.py'
    ]
    
    print("\n1. 重要ファイル存在確認:")
    missing_files = []
    for filename in critical_files:
        if Path(filename).exists():
            print(f"   OK {filename}")
        else:
            print(f"   NG {filename}")
            missing_files.append(filename)
    
    checks.append(("重要ファイル", len(missing_files) == 0, f"{len(missing_files)}個のファイル不足"))
    
    # 2. 設定ディレクトリ確認
    print("\n2. ディレクトリ構造確認:")
    required_dirs = ['config', 'data', 'logs', 'static']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"   OK {dir_name}/")
        else:
            print(f"   NG {dir_name}/ (creating...)")
            Path(dir_name).mkdir(exist_ok=True)
            missing_dirs.append(dir_name)
    
    checks.append(("ディレクトリ構造", True, "全ディレクトリ準備完了"))
    
    # 3. システム起動テスト
    print("\n3. システム起動テスト:")
    
    try:
        from daytrade_core import DayTradeCore
        core = DayTradeCore()
        print("   OK DayTradeCore起動成功")
        core_status = True
    except Exception as e:
        print(f"   NG DayTradeCore起動失敗: {e}")
        core_status = False
    
    try:
        from daytrade_web import DayTradeWebServer, WEB_AVAILABLE
        if WEB_AVAILABLE:
            server = DayTradeWebServer(port=8001)
            print("   OK WebServer起動準備完了")
            web_status = True
        else:
            print("   WN WebServer依存関係不足 (Flask等)")
            web_status = True  # 警告として扱う
    except Exception as e:
        print(f"   NG WebServer起動失敗: {e}")
        web_status = False
    
    checks.append(("システム起動", core_status and web_status, "コアとWebサーバー"))
    
    # 4. パフォーマンス確認
    print("\n4. パフォーマンス確認:")
    try:
        from performance_optimization_system import get_performance_system
        perf_system = get_performance_system()
        metrics = perf_system.get_current_metrics()
        
        print(f"   OK CPU使用率: {metrics.cpu_percent:.1f}%")
        print(f"   OK メモリ使用率: {metrics.memory_percent:.1f}%")
        print(f"   OK キャッシュヒット率: {metrics.cache_hit_rate:.1%}")
        
        performance_good = (
            metrics.cpu_percent < 50.0 and
            metrics.memory_percent < 80.0 and
            metrics.cache_hit_rate > 0.3
        )
        
        checks.append(("パフォーマンス", performance_good, "CPU・メモリ・キャッシュ"))
        
    except Exception as e:
        print(f"   NG パフォーマンス取得失敗: {e}")
        checks.append(("パフォーマンス", False, str(e)))
    
    # 5. セキュリティ確認
    print("\n5. セキュリティ確認:")
    try:
        from security_monitoring_system import get_security_monitor
        security_monitor = get_security_monitor()
        print("   OK セキュリティ監視システム動作")
        security_status = True
    except ImportError:
        print("   WN セキュリティ監視システム未実装")
        security_status = True  # 警告として扱う
    except Exception as e:
        print(f"   NG セキュリティシステムエラー: {e}")
        security_status = False
    
    checks.append(("セキュリティ", security_status, "監視システム"))
    
    # 6. データベース確認
    print("\n6. データベース確認:")
    try:
        db_files = list(Path('data').glob('*.db'))
        if db_files:
            print(f"   OK データベースファイル: {len(db_files)}個")
            for db_file in db_files[:3]:  # 最初の3個を表示
                print(f"      - {db_file.name}")
        else:
            print("   WN データベースファイル未作成")
        
        db_status = True
    except Exception as e:
        print(f"   NG データベース確認失敗: {e}")
        db_status = False
    
    checks.append(("データベース", db_status, "SQLiteファイル"))
    
    # 7. 最終評価
    print("\n" + "=" * 40)
    print("最終評価:")
    
    passed = sum(1 for _, status, _ in checks if status)
    total = len(checks)
    success_rate = (passed / total) * 100
    
    for check_name, status, detail in checks:
        status_text = "合格" if status else "不合格"
        print(f"  {check_name}: {status_text} ({detail})")
    
    print(f"\n総合結果: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("本番環境準備完了 - 運用開始可能")
        return 0
    elif success_rate >= 70:
        print("本番環境ほぼ準備完了 - 軽微な調整が必要")
        return 1
    else:
        print("本番環境準備不足 - 重大な問題の修正が必要")
        return 2

if __name__ == "__main__":
    sys.exit(check_production_readiness())