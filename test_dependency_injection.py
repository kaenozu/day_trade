#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依存性注入システムのテスト
Issue #918 項目3対応: 依存性注入パターンの導入

簡単な動作確認テスト
"""

import sys
import os
from pathlib import Path

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# パスの設定
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_di_container():
    """DIコンテナの基本動作テスト"""
    print("=== 依存性注入システム テスト ===")
    
    try:
        from src.day_trade.core.dependency_injection import DIContainer, injectable
        
        # テスト用のクラス定義
        @injectable
        class TestService:
            def __init__(self, name: str = "TestService"):
                self.name = name
            
            def get_info(self):
                return f"Service: {self.name}"
        
        class TestClient:
            def __init__(self, service: TestService):
                self.service = service
            
            def get_message(self):
                return f"Client using {self.service.get_info()}"
        
        # DIコンテナテスト
        container = DIContainer()
        
        # 1. シングルトン登録
        container.register_singleton(TestService, TestService)
        print("OK: シングルトン登録成功")
        
        # 2. インスタンス取得
        service1 = container.resolve(TestService)
        service2 = container.resolve(TestService)
        
        # 3. シングルトン確認
        assert service1 is service2, "シングルトンが正常に動作していません"
        print("OK: シングルトン動作確認")
        
        # 4. 依存性注入テスト
        container.register_transient(TestClient, TestClient)
        client = container.resolve(TestClient)
        message = client.get_message()
        print(f"OK: 依存性注入成功: {message}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: DIコンテナテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_services():
    """サービス実装テスト"""
    print("\n=== サービス実装テスト ===")
    
    try:
        from src.day_trade.core.dependency_injection import get_container
        from src.day_trade.core.services import register_default_services
        from src.day_trade.core.dependency_injection import (
            IConfigurationService, ILoggingService, IAnalyzerService
        )
        
        # デフォルトサービス登録
        register_default_services()
        print("OK: デフォルトサービス登録完了")
        
        container = get_container()
        
        # 1. 設定サービステスト
        config_service = container.resolve(IConfigurationService)
        config = config_service.get_config()
        print(f"OK: 設定サービス動作確認: {type(config)}")
        
        # 2. ログサービステスト
        logging_service = container.resolve(ILoggingService)
        logger = logging_service.get_logger("test", "TestContext")
        logger.info("テストログメッセージ")
        print("OK: ログサービス動作確認")
        
        # 3. 分析サービステスト（軽量テスト）
        analyzer_service = container.resolve(IAnalyzerService)
        print("OK: 分析サービス取得確認")
        
        return True
        
    except Exception as e:
        print(f"FAIL: サービステスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_application():
    """アプリケーション初期化テスト"""
    print("\n=== アプリケーション初期化テスト ===")
    
    try:
        from src.day_trade.core.application import StockAnalysisApplication, create_stock_analysis_application
        
        # 1. ファクトリー関数テスト
        app1 = create_stock_analysis_application(debug=True)
        print("OK: ファクトリー関数による作成成功")
        
        # 2. 直接作成テスト
        app2 = StockAnalysisApplication(debug=True)
        print("OK: 直接作成成功")
        
        # 3. サービス注入確認
        assert app2.config_service is not None, "設定サービスが注入されていません"
        assert app2.logging_service is not None, "ログサービスが注入されていません"
        assert app2.analyzer_service is not None, "分析サービスが注入されていません"
        print("OK: 依存性注入確認")
        
        # 4. ログ動作確認
        app2.logger.info("アプリケーション初期化テスト完了")
        print("OK: ロガー動作確認")
        
        return True
        
    except Exception as e:
        print(f"FAIL: アプリケーションテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("依存性注入システムのテストを開始します...\n")
    
    results = []
    
    # 各テストを実行
    results.append(("DIコンテナ", test_di_container()))
    results.append(("サービス実装", test_services()))
    results.append(("アプリケーション初期化", test_application()))
    
    # 結果サマリー
    print("\n" + "="*50)
    print("テスト結果サマリー")
    print("="*50)
    
    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\n合計: {passed}/{len(results)} テスト通過")
    
    if passed == len(results):
        print("SUCCESS: 全テストが正常に完了しました！")
        print("依存性注入パターンの導入が成功しました。")
        return 0
    else:
        print("WARNING: 一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    sys.exit(main())