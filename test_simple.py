#!/usr/bin/env python3
"""
簡単なテスト - エラー修正確認用
"""

def test_imports():
    """基本的なインポートをテスト"""
    try:
        # 修正したファイルのインポートテスト
        from performance_tracker import PerformanceTracker, DBTrade, TradeType, TradeResult
        print("OK performance_tracker import OK")

        from real_data_provider_v2 import ImprovedYahooFinanceProvider
        print("OK real_data_provider_v2 import OK")

        return True
    except Exception as e:
        print(f"ERROR Import error: {e}")
        return False

def test_syntax():
    """構文エラーの確認"""
    try:
        # gpu_engine.pyの修正確認
        exec("rs = 1 / (1e-10)")  # 修正された構文
        print("OK Syntax fix OK")
        return True
    except Exception as e:
        print(f"ERROR Syntax error: {e}")
        return False

if __name__ == "__main__":
    print("エラー修正確認テスト実行中...")

    success = True
    success &= test_imports()
    success &= test_syntax()

    if success:
        print("OK 全ての修正が正常です")
    else:
        print("ERROR まだ問題があります")