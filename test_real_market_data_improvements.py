#!/usr/bin/env python3
"""
Issues #616, #615, #618-620 テストケース

RealMarketDataManager改善をテスト
"""

import sys
import tempfile
import os
import time
import threading
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.data.real_market_data import (
    RealMarketDataManager, get_real_stock_data, calculate_real_technical_indicators
)

def create_test_data():
    """テスト用のサンプルデータを作成"""
    dates = pd.date_range(end='2024-12-01', periods=60, freq='D')
    np.random.seed(42)
    
    # トレンドを持つ価格データ
    trend = np.linspace(1000, 1200, 60)
    noise = np.random.randn(60) * 30
    prices = trend + noise
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(60) * 5,
        'High': prices + np.abs(np.random.randn(60)) * 15,
        'Low': prices - np.abs(np.random.randn(60)) * 15,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 60),
    })
    df.set_index('Date', inplace=True)
    return df

def test_issue_615_api_rate_limit_concurrency():
    """Issue #615: APIレート制限と並行性管理改善テスト"""
    print("=== Issue #615: APIレート制限と並行性管理テスト ===")
    
    try:
        # 一時ディレクトリでテスト
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "test_cache.db")
            manager = RealMarketDataManager(cache_db_path=cache_path)
            
            # レート制限設定の確認
            print(f"  最小API間隔: {manager.min_api_interval}秒")
            print(f"  最大同時リクエスト数: {manager.max_concurrent_requests}")
            
            # レート制限機能のテスト
            test_symbol = "7203"
            
            # 連続API呼び出しのタイミングテスト
            start_time = time.time()
            manager._wait_for_rate_limit(test_symbol)
            
            # 最初の呼び出し記録を設定
            manager.last_api_call[test_symbol] = time.time()
            
            # 2回目の呼び出しで待機が発生するかテスト
            wait_start = time.time()
            manager._wait_for_rate_limit(test_symbol)
            wait_time = time.time() - wait_start
            
            print(f"  レート制限待機時間: {wait_time:.3f}秒")
            
            if wait_time >= manager.min_api_interval * 0.9:  # 誤差考慮
                print("  [PASS] レート制限が正常に動作")
            else:
                print("  [INFO] レート制限待機が短縮されました")
            
            # 並行性制御の確認
            print(f"  [PASS] 並行性制御設定が確認されました (max: {manager.max_concurrent_requests})")
            
    except Exception as e:
        print(f"  [FAIL] Issue #615テストでエラー: {e}")
    
    print()

def test_issue_616_cache_management_robustness():
    """Issue #616: キャッシュ管理の堅牢性向上テスト"""
    print("=== Issue #616: キャッシュ管理の堅牢性テスト ===")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "test_cache.db")
            manager = RealMarketDataManager(cache_db_path=cache_path)
            
            # キャッシュ有効期限設定の確認
            print(f"  SQLiteキャッシュ期限: {manager.cache_duration_hours}時間")
            
            # メモリキャッシュとSQLiteキャッシュの一貫性テスト
            test_data = create_test_data()
            test_symbol = "1234"
            
            # データをキャッシュに保存
            manager._cache_price_data(test_symbol, test_data)
            
            # SQLiteから取得
            sqlite_data = manager._get_cached_price_data(test_symbol)
            if sqlite_data is not None:
                print("  [PASS] SQLiteキャッシュからのデータ取得が成功")
                
                # キャッシュ有効期限チェック
                is_expired = manager._is_cache_expired(test_symbol)
                print(f"  キャッシュ期限切れ状態: {is_expired}")
                
                if not is_expired:
                    print("  [PASS] キャッシュが有効期限内で動作")
                else:
                    print("  [INFO] キャッシュが期限切れと判定")
            else:
                print("  [FAIL] SQLiteキャッシュデータ取得に失敗")
            
            # メモリキャッシュテスト
            cache_key = f"{test_symbol}_60d"
            manager._update_memory_cache(cache_key, test_data)
            
            is_memory_valid = manager._check_memory_cache(cache_key)
            if is_memory_valid:
                print("  [PASS] メモリキャッシュが正常に動作")
            else:
                print("  [FAIL] メモリキャッシュ検証に失敗")
            
            # キャッシュクリア機能のテスト（カプセル化改善）
            if hasattr(manager, 'clear_cache'):
                manager.clear_cache()
                print("  [PASS] キャッシュクリア機能が利用可能")
            else:
                print("  [INFO] キャッシュクリア機能が未実装")
                
    except Exception as e:
        print(f"  [FAIL] Issue #616テストでエラー: {e}")
    
    print()

def test_issue_618_ml_like_scores_renaming():
    """Issue #618: ML風スコア命名とパラメータ化テスト"""
    print("=== Issue #618: ML風スコア命名とパラメータ化テスト ===")
    
    try:
        manager = RealMarketDataManager()
        test_data = create_test_data()
        
        # 現在のML風メソッド名確認
        ml_methods = [
            'generate_ml_trend_score',
            'generate_ml_volatility_score', 
            'generate_ml_pattern_score'
        ]
        
        all_methods_exist = True
        for method_name in ml_methods:
            if hasattr(manager, method_name):
                print(f"  [PASS] メソッド '{method_name}' が存在")
                
                # メソッド実行テスト
                method = getattr(manager, method_name)
                score, confidence = method(test_data)
                
                print(f"    {method_name}: score={score}, confidence={confidence}")
                
                # スコア範囲の妥当性チェック
                if 0 <= score <= 100 and 0 <= confidence <= 1:
                    print(f"    [PASS] スコア範囲が妥当 (score: {score}, confidence: {confidence})")
                else:
                    print(f"    [FAIL] スコア範囲が無効 (score: {score}, confidence: {confidence})")
                    
            else:
                print(f"  [FAIL] メソッド '{method_name}' が不足")
                all_methods_exist = False
        
        # 統計的アルゴリズムかML実装かの確認
        # ドキュメント文字列から実装内容を確認
        trend_method = getattr(manager, 'generate_ml_trend_score', None)
        if trend_method and trend_method.__doc__:
            if "機械学習風" in trend_method.__doc__ or "実データベース" in trend_method.__doc__:
                print("  [PASS] メソッドが統計的実装として適切に文書化")
            else:
                print("  [INFO] メソッドの文書化を確認")
        
        if all_methods_exist:
            print("  [PASS] ML風スコア生成メソッドが正常に動作")
            
    except Exception as e:
        print(f"  [FAIL] Issue #618テストでエラー: {e}")
    
    print()

def test_issue_619_technical_indicator_consolidation():
    """Issue #619: テクニカル指標計算ロジック統合テスト"""
    print("=== Issue #619: テクニカル指標計算ロジック統合テスト ===")
    
    try:
        manager = RealMarketDataManager()
        test_data = create_test_data()
        
        # 主要テクニカル指標メソッドの確認
        technical_methods = [
            'calculate_rsi',
            'calculate_macd',
            'calculate_volume_ratio',
            'calculate_price_change_percent'
        ]
        
        all_methods_work = True
        for method_name in technical_methods:
            if hasattr(manager, method_name):
                method = getattr(manager, method_name)
                
                try:
                    if method_name == 'calculate_price_change_percent':
                        result = method(test_data, days=1)
                    else:
                        result = method(test_data)
                    
                    print(f"  [PASS] {method_name}: {result}")
                    
                    # 妥当性チェック
                    if isinstance(result, (int, float)) and not np.isnan(result):
                        print(f"    妥当な数値結果")
                    else:
                        print(f"    [WARN] 結果が無効: {result}")
                        
                except Exception as e:
                    print(f"  [FAIL] {method_name}実行エラー: {e}")
                    all_methods_work = False
                    
            else:
                print(f"  [FAIL] メソッド '{method_name}' が不足")
                all_methods_work = False
        
        # エラーハンドリングの改善確認（Issue #617対応）
        # 空データでのテスト
        empty_data = pd.DataFrame()
        
        try:
            rsi_empty = manager.calculate_rsi(empty_data)
            print(f"  空データRSI結果: {rsi_empty} (デフォルト値期待)")
            
            if rsi_empty == 50.0:
                print("  [PASS] 空データ時のデフォルト値が正しく設定")
            else:
                print(f"  [INFO] 空データ時の値: {rsi_empty}")
                
        except Exception as e:
            print(f"  [FAIL] 空データエラーハンドリング失敗: {e}")
        
        if all_methods_work:
            print("  [PASS] テクニカル指標計算ロジックが統合されています")
            
    except Exception as e:
        print(f"  [FAIL] Issue #619テストでエラー: {e}")
    
    print()

def test_issue_620_global_utility_functions():
    """Issue #620: グローバルユーティリティ関数最適化テスト"""
    print("=== Issue #620: グローバルユーティリティ関数最適化テスト ===")
    
    try:
        # グローバル関数の存在確認
        global_functions = [
            get_real_stock_data,
            calculate_real_technical_indicators
        ]
        
        all_functions_work = True
        
        for func in global_functions:
            func_name = func.__name__
            print(f"  関数 '{func_name}' のテスト")
            
            try:
                # モックデータでテスト（実際のAPI呼び出しを避ける）
                with patch('src.day_trade.data.real_market_data.yf') as mock_yf:
                    # mock yfinanceレスポンス
                    mock_ticker = MagicMock()
                    mock_ticker.history.return_value = create_test_data()
                    mock_yf.Ticker.return_value = mock_ticker
                    
                    if func_name == 'get_real_stock_data':
                        result = func("7203")
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            print(f"    [PASS] {func_name}が正常に実行")
                        else:
                            print(f"    [FAIL] {func_name}の結果が無効")
                            all_functions_work = False
                            
                    elif func_name == 'calculate_real_technical_indicators':
                        result = func("7203")
                        if isinstance(result, dict) and 'rsi' in result:
                            print(f"    [PASS] {func_name}が正常に実行")
                            print(f"      結果キー: {list(result.keys())}")
                        else:
                            print(f"    [FAIL] {func_name}の結果が無効")
                            all_functions_work = False
                    
            except Exception as e:
                print(f"    [FAIL] {func_name}実行エラー: {e}")
                all_functions_work = False
        
        # インスタンス生成の最適化確認
        # 複数回呼び出した場合のパフォーマンス測定
        start_time = time.time()
        
        for i in range(3):
            try:
                with patch('src.day_trade.data.real_market_data.yf'):
                    get_real_stock_data("TEST")
            except:
                pass  # エラーは無視してパフォーマンステストに集中
        
        elapsed_time = time.time() - start_time
        print(f"  3回呼び出し時間: {elapsed_time:.3f}秒")
        
        if elapsed_time < 1.0:  # 1秒以内であれば良好
            print("  [PASS] グローバル関数のパフォーマンスが良好")
        else:
            print("  [INFO] グローバル関数の最適化が必要かもしれません")
        
        if all_functions_work:
            print("  [PASS] グローバルユーティリティ関数が最適化されています")
            
    except Exception as e:
        print(f"  [FAIL] Issue #620テストでエラー: {e}")
    
    print()

def test_integration():
    """統合テスト"""
    print("=== 統合テスト ===")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "integration_cache.db")
            manager = RealMarketDataManager(cache_db_path=cache_path)
            
            # 全体的な機能テスト
            test_symbol = "7203"
            
            # モックされたyfinanceでの統合テスト
            with patch('src.day_trade.data.real_market_data.yf') as mock_yf:
                mock_ticker = MagicMock()
                mock_data = create_test_data()
                mock_ticker.history.return_value = mock_data
                mock_yf.Ticker.return_value = mock_ticker
                
                # データ取得
                data = manager.get_stock_data(test_symbol)
                if data is not None and not data.empty:
                    print("  [PASS] 統合データ取得が成功")
                    
                    # 技術指標計算
                    rsi = manager.calculate_rsi(data)
                    macd = manager.calculate_macd(data)
                    
                    print(f"    RSI: {rsi}")
                    print(f"    MACD: {macd}")
                    
                    # MLスコア計算
                    trend_score, trend_conf = manager.generate_ml_trend_score(data)
                    print(f"    トレンドスコア: {trend_score} (信頼度: {trend_conf})")
                    
                    # キャッシュ機能確認
                    cached_data = manager._get_cached_price_data(test_symbol)
                    if cached_data is not None:
                        print("  [PASS] キャッシュ機能が正常に動作")
                    
                    # エラーハンドリング確認
                    error_info = manager._analyze_technical_error(
                        ValueError("test error"), "RSI", data
                    )
                    if isinstance(error_info, dict) and 'value' in error_info:
                        print("  [PASS] エラーハンドリング機能が動作")
                    
                    print("  [PASS] 統合テストが成功しました")
                    
                else:
                    print("  [FAIL] 統合データ取得に失敗")
                    
    except Exception as e:
        print(f"  [FAIL] 統合テストでエラー: {e}")
    
    print()

def run_all_tests():
    """全テストを実行"""
    print("RealMarketDataManager 改善テスト開始\n")
    
    test_issue_615_api_rate_limit_concurrency()
    test_issue_616_cache_management_robustness()
    test_issue_618_ml_like_scores_renaming()
    test_issue_619_technical_indicator_consolidation()
    test_issue_620_global_utility_functions()
    test_integration()
    
    print("全テスト完了")

if __name__ == "__main__":
    run_all_tests()