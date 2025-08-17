#!/usr/bin/env python3
"""
Day Trade Personal - コアエントリーポイント

リファクタリング後の軽量メインファイル
"""

import sys
from pathlib import Path

# システムパス追加
sys.path.append(str(Path(__file__).parent / "src"))

from src.day_trade.core.application import DayTradeApplication


def main():
    """メイン実行関数"""
    app = DayTradeApplication()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())