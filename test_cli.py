"""
CLIの動作確認スクリプト
"""
import sys
from src.day_trade.cli.main import main

if __name__ == "__main__":
    # コマンドライン引数を設定
    if len(sys.argv) == 1:
        # 引数なしの場合はヘルプを表示
        sys.argv = ["test_cli.py", "--help"]
    
    main()