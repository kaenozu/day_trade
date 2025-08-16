#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Personal - リファクタリング版メインエントリーポイント

使用方法:
  python main.py           # デイトレード推奨（デフォルト）
  python main.py --quick   # 基本分析
  python main.py --web     # Webダッシュボード
  python main.py --help    # 詳細オプション
"""

import sys
from pathlib import Path

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Windows環境での文字化け対策
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


def main():
    """リファクタリング版メイン関数"""
    try:
        # ログシステムの初期化
        from daytrade_logging import setup_logging
        logger = setup_logging(debug='--debug' in sys.argv)

        # CLIの実行
        from daytrade_cli import DayTradeCLI
        cli = DayTradeCLI()
        return cli.execute()

    except KeyboardInterrupt:
        print("\\n🛑 処理を中断しました")
        return 1
    except ImportError as e:
        print(f"❌ 必要なモジュールが見つかりません: {e}")
        print("📥 pip install -r requirements.txt を実行してください")
        return 1
    except Exception as e:
        print(f"❌ システムエラーが発生しました: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # バナー表示
    print("🚀 Day Trade Personal - 93%精度AIシステム")
    print("📊 リファクタリング版 v2.0")
    print("=" * 50)

    # メイン処理実行
    sys.exit(main())