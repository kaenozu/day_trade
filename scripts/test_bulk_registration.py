#!/usr/bin/env python3
"""
銘柄一括登録テストスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.day_trade.data.stock_master import StockMasterManager  # noqa: E402
from src.day_trade.utils.logging_config import setup_logging  # noqa: E402

# ロギング設定
setup_logging()

def test_registration():
    """小規模テスト"""
    stock_master = StockMasterManager()

    # テスト用証券コード（有名な銘柄）
    test_codes = ['7203', '9984', '6758']  # トヨタ、ソフトバンク、ソニー

    print("=== 銘柄一括登録テスト ===")
    print(f"テスト対象: {test_codes}")

    # 一括取得・更新
    result = stock_master.bulk_fetch_and_update_companies(
        codes=test_codes,
        batch_size=3,
        delay=0.5
    )

    print("結果:")
    print(f"  成功: {result['success']}")
    print(f"  失敗: {result['failed']}")
    print(f"  スキップ: {result['skipped']}")
    print(f"  合計: {result['total']}")

    # 登録された銘柄を確認
    print("\n登録確認:")
    for code in test_codes:
        stock = stock_master.get_stock_by_code(code)
        if stock:
            print(f"  {code}: {stock.name} ({stock.market})")
        else:
            print(f"  {code}: 登録失敗")

if __name__ == "__main__":
    test_registration()
