#!/usr/bin/env python3
"""
銘柄一括登録スクリプト

Issue #122: 銘柄を一括で追加する機能の実装
- JPXから取得した証券コードリストを使用
- Issue #126のbulk機能を活用して効率的に処理
- 冪等性を確保し、エラー耐性を持つ
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.day_trade.data.stock_master import StockMasterManager
from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.utils.logging_config import setup_logging

# ロギング設定
setup_logging()
logger = logging.getLogger(__name__)


class BulkStockRegistration:
    """銘柄一括登録管理クラス"""

    def __init__(self):
        self.stock_master = StockMasterManager()
        self.stock_fetcher = StockFetcher()
        self.data_dir = project_root / "data" / "stock_lists"

    def load_stock_codes_from_csv(self, csv_path: Optional[Path] = None) -> List[str]:
        """
        CSVから証券コードを読み込み

        Args:
            csv_path: CSVファイルのパス（None の場合はデフォルト）

        Returns:
            証券コードのリスト
        """
        if csv_path is None:
            csv_path = self.data_dir / "jpx_stock_codes.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"証券コードCSVが見つかりません: {csv_path}")

        logger.info(f"証券コードCSV読み込み: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            stock_codes = df['stock_code'].astype(str).tolist()

            logger.info(f"読み込み証券コード数: {len(stock_codes)}")
            logger.info(f"サンプル: {stock_codes[:10]}")

            return stock_codes

        except Exception as e:
            logger.error(f"CSV読み込みエラー: {e}")
            raise

    def get_existing_stock_codes(self) -> set[str]:
        """
        データベースに既存の証券コードを取得

        Returns:
            既存証券コードのセット
        """
        logger.info("既存証券コード確認中...")

        try:
            from src.day_trade.models.stock import Stock
            from src.day_trade.models.database import db_manager

            existing_codes = set()

            # データベースから直接証券コードを取得
            with db_manager.get_session() as session:
                stocks = session.query(Stock).all()
                for stock in stocks:
                    existing_codes.add(stock.code)

            logger.info(f"既存証券コード数: {len(existing_codes)}")
            return existing_codes

        except Exception as e:
            logger.error(f"既存証券コード確認エラー: {e}")
            # エラーの場合は空のセットを返して全件処理
            return set()

    def filter_new_stock_codes(self, all_codes: List[str]) -> List[str]:
        """
        新規追加対象の証券コードをフィルタリング

        Args:
            all_codes: 全証券コードリスト

        Returns:
            新規追加対象の証券コードリスト
        """
        existing_codes = self.get_existing_stock_codes()
        new_codes = [code for code in all_codes if code not in existing_codes]

        logger.info(f"新規追加対象: {len(new_codes)}件")
        logger.info(f"既存スキップ: {len(all_codes) - len(new_codes)}件")

        return new_codes

    def bulk_register_stocks(
        self,
        stock_codes: List[str],
        batch_size: int = 50,
        delay: float = 0.1,
        max_failures: int = 100
    ) -> Dict[str, Any]:
        """
        銘柄を一括登録

        Args:
            stock_codes: 証券コードのリスト
            batch_size: バッチサイズ
            delay: バッチ間の遅延（秒）
            max_failures: 許容する最大失敗数

        Returns:
            処理結果の統計情報
        """
        logger.info("=== 銘柄一括登録開始 ===")
        logger.info(f"対象銘柄数: {len(stock_codes)}")
        logger.info(f"バッチサイズ: {batch_size}")
        logger.info(f"遅延: {delay}秒")

        start_time = time.time()
        results = {
            "total": len(stock_codes),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "failed_codes": [],
            "batches_processed": 0
        }

        try:
            # Issue #126のbulk機能を活用
            result = self.stock_master.bulk_fetch_and_update_companies(
                codes=stock_codes,
                batch_size=batch_size,
                delay=delay
            )

            # 結果をマージ
            results.update(result)

            total_time = time.time() - start_time
            avg_time = total_time / len(stock_codes) if stock_codes else 0

            logger.info("=== 一括登録完了 ===")
            logger.info(f"処理時間: {total_time:.2f}秒")
            logger.info(f"平均時間/銘柄: {avg_time:.3f}秒")
            logger.info(f"成功: {results['success']}件")
            logger.info(f"失敗: {results['failed']}件")
            logger.info(f"スキップ: {results['skipped']}件")
            logger.info(f"成功率: {(results['success'] / results['total'] * 100):.1f}%")

            return results

        except Exception as e:
            logger.error(f"一括登録エラー: {e}")
            raise

    def generate_registration_report(self, results: Dict[str, Any]) -> str:
        """
        登録結果のレポートを生成

        Args:
            results: 処理結果

        Returns:
            レポート文字列
        """
        report = []
        report.append("=" * 60)
        report.append("銘柄一括登録結果レポート")
        report.append("=" * 60)
        report.append(f"対象銘柄数: {results['total']:,}")
        report.append(f"成功登録: {results['success']:,}")
        report.append(f"失敗: {results['failed']:,}")
        report.append(f"スキップ: {results['skipped']:,}")
        report.append(f"成功率: {(results['success'] / results['total'] * 100):.1f}%")

        if results.get('failed_codes'):
            report.append("\n失敗した証券コード:")
            for code in results['failed_codes'][:20]:  # 最初の20件のみ表示
                report.append(f"  - {code}")
            if len(results['failed_codes']) > 20:
                report.append(f"  ... 他{len(results['failed_codes']) - 20}件")

        report.append("=" * 60)
        return "\n".join(report)

    def run_full_registration(
        self,
        include_existing: bool = False,
        batch_size: int = 50,
        delay: float = 0.1
    ) -> Dict[str, Any]:
        """
        完全な一括登録処理を実行

        Args:
            include_existing: 既存銘柄も再処理するか
            batch_size: バッチサイズ
            delay: バッチ間の遅延

        Returns:
            処理結果
        """
        logger.info("=== 銘柄一括登録プロセス開始 ===")

        try:
            # 1. CSVから証券コード読み込み
            all_codes = self.load_stock_codes_from_csv()

            # 2. 新規のみ処理するかフィルタリング
            if include_existing:
                target_codes = all_codes
                logger.info("既存銘柄も含めて処理します")
            else:
                target_codes = self.filter_new_stock_codes(all_codes)
                logger.info("新規銘柄のみ処理します")

            if not target_codes:
                logger.info("処理対象の銘柄がありません")
                return {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "skipped": 0,
                    "message": "処理対象なし"
                }

            # 3. 一括登録実行
            results = self.bulk_register_stocks(
                target_codes,
                batch_size=batch_size,
                delay=delay
            )

            # 4. レポート生成
            report = self.generate_registration_report(results)
            logger.info(f"\n{report}")

            return results

        except Exception as e:
            logger.error(f"一括登録プロセスエラー: {e}", exc_info=True)
            raise


def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="銘柄一括登録スクリプト")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="既存銘柄も再処理する"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="バッチサイズ（デフォルト: 20）"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="バッチ間遅延秒数（デフォルト: 0.2）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際の登録を行わずに実行計画のみ表示"
    )

    args = parser.parse_args()

    try:
        registration = BulkStockRegistration()

        if args.dry_run:
            # ドライラン: 実行計画のみ表示
            logger.info("=== ドライランモード ===")
            all_codes = registration.load_stock_codes_from_csv()
            if args.include_existing:
                target_codes = all_codes
            else:
                target_codes = registration.filter_new_stock_codes(all_codes)

            logger.info(f"実行計画:")
            logger.info(f"  対象銘柄数: {len(target_codes)}")
            logger.info(f"  バッチサイズ: {args.batch_size}")
            logger.info(f"  推定バッチ数: {len(target_codes) // args.batch_size + 1}")
            logger.info(f"  推定実行時間: {len(target_codes) * 0.3:.1f}秒")

            return

        # 実際の登録実行
        results = registration.run_full_registration(
            include_existing=args.include_existing,
            batch_size=args.batch_size,
            delay=args.delay
        )

        # 成功確認
        if results["success"] > 0:
            print(f"✅ 銘柄登録完了: {results['success']}件成功")
        else:
            print("⚠️ 新規登録された銘柄はありませんでした")

    except Exception as e:
        logger.error(f"メイン処理エラー: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
