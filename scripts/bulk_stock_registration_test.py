#!/usr/bin/env python3
"""
銘柄一括登録テストスクリプト（小規模版）

Issue #122: 銘柄を一括で追加する機能の実装
- 最初のN件のみを処理してテスト
- パフォーマンスと動作確認用
"""

<<<<<<< HEAD
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import logging
=======
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

<<<<<<< HEAD
from src.day_trade.data.stock_master import StockMasterManager
from src.day_trade.models.stock import Stock
from src.day_trade.models.database import db_manager
from src.day_trade.utils.logging_config import setup_logging
=======
# isortとruffの要件を満たすため、パス設定後のimportは別ブロックとして配置
if True:  # パス設定後のimportブロック
    from src.day_trade.data.stock_master import StockMasterManager
    from src.day_trade.models.database import db_manager
    from src.day_trade.models.stock import Stock
    from src.day_trade.utils.logging_config import setup_logging
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

# ロギング設定
setup_logging()
logger = logging.getLogger(__name__)


def load_test_stock_codes(limit: int = 50) -> List[str]:
    """テスト用の証券コードを読み込み"""
    data_dir = project_root / "data" / "stock_lists"
    csv_path = data_dir / "jpx_stock_codes.csv"

    logger.info(f"テスト用証券コード読み込み: {csv_path}")

    df = pd.read_csv(csv_path)
<<<<<<< HEAD
    stock_codes = df['stock_code'].astype(str).head(limit).tolist()
=======
    stock_codes = df["stock_code"].astype(str).head(limit).tolist()
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

    logger.info(f"テスト対象: {len(stock_codes)}件")
    logger.info(f"証券コード: {stock_codes}")

    return stock_codes


def get_existing_codes() -> set[str]:
    """既存の証券コードを取得"""
    existing = set()

    try:
        with db_manager.get_session() as session:
            stocks = session.query(Stock).all()
            for stock in stocks:
                existing.add(stock.code)
    except Exception as e:
        logger.error(f"既存コード取得エラー: {e}")

    logger.info(f"既存証券コード数: {len(existing)}")
    return existing


def register_test_stocks(limit: int = 20) -> Dict[str, Any]:
    """テスト用の銘柄登録"""
    logger.info("=== テスト用銘柄登録開始 ===")

    # テスト対象の証券コード読み込み
    all_codes = load_test_stock_codes(limit * 2)  # 余裕を持って読み込み
    existing_codes = get_existing_codes()

    # 新規のみ抽出
    new_codes = [code for code in all_codes if code not in existing_codes][:limit]

    if not new_codes:
        logger.info("新規追加対象がありません")
        return {"total": 0, "success": 0, "message": "新規対象なし"}

    logger.info(f"登録対象: {len(new_codes)}件")
    logger.info(f"対象コード: {new_codes}")

    # StockMasterManagerを使用して登録
    stock_master = StockMasterManager()

    start_time = time.time()
    results = stock_master.bulk_fetch_and_update_companies(
<<<<<<< HEAD
        codes=new_codes,
        batch_size=5,
        delay=0.3
=======
        codes=new_codes, batch_size=5, delay=0.3
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
    )

    total_time = time.time() - start_time

    logger.info("=== テスト登録完了 ===")
    logger.info(f"処理時間: {total_time:.2f}秒")
    logger.info(f"結果: {results}")

    return results


def verify_registration(codes: List[str]) -> Dict[str, Any]:
    """登録結果の検証"""
    logger.info("=== 登録結果検証 ===")

    verification = {
        "total_codes": len(codes),
        "found_in_db": 0,
        "missing_codes": [],
<<<<<<< HEAD
        "found_codes": []
=======
        "found_codes": [],
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
    }

    try:
        with db_manager.get_session() as session:
            for code in codes:
                stock = session.query(Stock).filter(Stock.code == code).first()
                if stock:
                    verification["found_in_db"] += 1
                    verification["found_codes"].append(code)
                    logger.info(f"✅ {code}: {stock.name}")
                else:
                    verification["missing_codes"].append(code)
                    logger.warning(f"❌ {code}: データベースに見つかりません")

    except Exception as e:
        logger.error(f"検証エラー: {e}")

<<<<<<< HEAD
    logger.info(f"検証結果: {verification['found_in_db']}/{verification['total_codes']}件がデータベースに存在")
=======
    logger.info(
        f"検証結果: {verification['found_in_db']}/{verification['total_codes']}件がデータベースに存在"
    )
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

    return verification


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="銘柄一括登録テスト")
<<<<<<< HEAD
    parser.add_argument("--limit", type=int, default=10, help="登録する銘柄数（デフォルト: 10）")
=======
    parser.add_argument(
        "--limit", type=int, default=10, help="登録する銘柄数（デフォルト: 10）"
    )
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
    parser.add_argument("--verify-only", action="store_true", help="検証のみ実行")

    args = parser.parse_args()

    try:
        if args.verify_only:
            # 検証のみ
            codes = load_test_stock_codes(50)
            existing = get_existing_codes()
            found_codes = [code for code in codes if code in existing]
<<<<<<< HEAD
            verify_result = verify_registration(found_codes[:20])  # 最初の20件のみ検証
=======
            _verify_result = verify_registration(found_codes[:20])  # 最初の20件のみ検証
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        else:
            # 実際の登録実行
            results = register_test_stocks(args.limit)

            if results.get("success", 0) > 0:
                print(f"✅ テスト登録完了: {results['success']}件成功")

                # 登録した銘柄の検証
                if "registered_codes" in results:
                    verify_registration(results["registered_codes"])
            else:
                print("⚠️ 新規登録はありませんでした")

    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
