#!/usr/bin/env python3
"""
銘柄一括取得のパフォーマンステスト

Issue #126の改善効果を測定するため、
従来の個別取得と新しい一括取得のパフォーマンスを比較する。
"""

<<<<<<< HEAD
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
=======
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

<<<<<<< HEAD
from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.data.stock_master import StockMasterManager
from src.day_trade.utils.logging_config import setup_logging
=======
# isortとruffの要件を満たすため、パス設定後のimportは別ブロックとして配置
if True:  # パス設定後のimportブロック
    from src.day_trade.data.stock_fetcher import StockFetcher
    from src.day_trade.data.stock_master import StockMasterManager
    from src.day_trade.utils.logging_config import setup_logging
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

# ロギング設定
setup_logging()
logger = logging.getLogger(__name__)


class BulkFetchPerformanceTest:
    """一括取得パフォーマンステストクラス"""

    def __init__(self):
        self.stock_fetcher = StockFetcher()
        self.stock_master = StockMasterManager()

        # テスト用銘柄コード（少数で実際のテスト）
        self.test_codes = [
            "7203",  # トヨタ自動車
            "9984",  # ソフトバンクグループ
            "6758",  # ソニーグループ
            "4063",  # 信越化学工業
            "8306",  # 三菱UFJフィナンシャル・グループ
            "6501",  # 日立製作所
            "8035",  # 東京エレクトロン
            "9984",  # ソフトバンクグループ
            "4502",  # 武田薬品工業
<<<<<<< HEAD
            "4568"   # 第一三共
=======
            "4568",  # 第一三共
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        ]

    def test_individual_fetch(self, codes: List[str]) -> Dict[str, Any]:
        """個別取得のパフォーマンステスト"""
        logger.info(f"個別取得テスト開始: {len(codes)}銘柄")

        start_time = time.time()
        results = {}
        success_count = 0

        for code in codes:
            try:
                company_info = self.stock_fetcher.get_company_info(code)
                if company_info:
                    results[code] = company_info
                    success_count += 1
                else:
                    results[code] = None
            except Exception as e:
                logger.error(f"個別取得エラー {code}: {e}")
                results[code] = None

        total_time = time.time() - start_time
        avg_time = total_time / len(codes) if codes else 0

        performance_data = {
            "method": "individual",
            "total_codes": len(codes),
            "successful_codes": success_count,
            "total_time": total_time,
            "avg_time_per_code": avg_time,
<<<<<<< HEAD
            "success_rate": success_count / len(codes) if codes else 0
=======
            "success_rate": success_count / len(codes) if codes else 0,
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        }

        logger.info(f"個別取得完了: {performance_data}")
        return performance_data

    def test_bulk_fetch(self, codes: List[str], batch_size: int = 50) -> Dict[str, Any]:
        """一括取得のパフォーマンステスト"""
<<<<<<< HEAD
        logger.info(f"一括取得テスト開始: {len(codes)}銘柄 (バッチサイズ: {batch_size})")
=======
        logger.info(
            f"一括取得テスト開始: {len(codes)}銘柄 (バッチサイズ: {batch_size})"
        )
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

        start_time = time.time()

        try:
            results = self.stock_fetcher.bulk_get_company_info(
                codes=codes,
                batch_size=batch_size,
<<<<<<< HEAD
                delay=0.05  # テスト用の短い遅延
=======
                delay=0.05,  # テスト用の短い遅延
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            )
            success_count = sum(1 for result in results.values() if result is not None)
        except Exception as e:
            logger.error(f"一括取得エラー: {e}")
            results = {}
            success_count = 0

        total_time = time.time() - start_time
        avg_time = total_time / len(codes) if codes else 0

        performance_data = {
            "method": "bulk",
            "total_codes": len(codes),
            "successful_codes": success_count,
            "total_time": total_time,
            "avg_time_per_code": avg_time,
            "success_rate": success_count / len(codes) if codes else 0,
<<<<<<< HEAD
            "batch_size": batch_size
=======
            "batch_size": batch_size,
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        }

        logger.info(f"一括取得完了: {performance_data}")
        return performance_data

    def test_master_individual_update(self, codes: List[str]) -> Dict[str, Any]:
        """マスタ個別更新のパフォーマンステスト"""
        logger.info(f"マスタ個別更新テスト開始: {len(codes)}銘柄")

        start_time = time.time()
        success_count = 0

        for code in codes:
            try:
                stock = self.stock_master.fetch_and_update_stock_info(code)
                if stock:
                    success_count += 1
            except Exception as e:
                logger.error(f"個別更新エラー {code}: {e}")

        total_time = time.time() - start_time
        avg_time = total_time / len(codes) if codes else 0

        performance_data = {
            "method": "master_individual",
            "total_codes": len(codes),
            "successful_codes": success_count,
            "total_time": total_time,
            "avg_time_per_code": avg_time,
<<<<<<< HEAD
            "success_rate": success_count / len(codes) if codes else 0
=======
            "success_rate": success_count / len(codes) if codes else 0,
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        }

        logger.info(f"マスタ個別更新完了: {performance_data}")
        return performance_data

<<<<<<< HEAD
    def test_master_bulk_update(self, codes: List[str], batch_size: int = 50) -> Dict[str, Any]:
        """マスタ一括更新のパフォーマンステスト"""
        logger.info(f"マスタ一括更新テスト開始: {len(codes)}銘柄 (バッチサイズ: {batch_size})")
=======
    def test_master_bulk_update(
        self, codes: List[str], batch_size: int = 50
    ) -> Dict[str, Any]:
        """マスタ一括更新のパフォーマンステスト"""
        logger.info(
            f"マスタ一括更新テスト開始: {len(codes)}銘柄 (バッチサイズ: {batch_size})"
        )
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

        start_time = time.time()

        try:
            result = self.stock_master.bulk_fetch_and_update_companies(
                codes=codes,
                batch_size=batch_size,
<<<<<<< HEAD
                delay=0.05  # テスト用の短い遅延
            )
        except Exception as e:
            logger.error(f"マスタ一括更新エラー: {e}")
            result = {"success": 0, "failed": len(codes), "skipped": 0, "total": len(codes)}
=======
                delay=0.05,  # テスト用の短い遅延
            )
        except Exception as e:
            logger.error(f"マスタ一括更新エラー: {e}")
            result = {
                "success": 0,
                "failed": len(codes),
                "skipped": 0,
                "total": len(codes),
            }
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

        total_time = time.time() - start_time
        avg_time = total_time / len(codes) if codes else 0

        performance_data = {
            "method": "master_bulk",
            "total_codes": result["total"],
            "successful_codes": result["success"],
            "failed_codes": result["failed"],
            "skipped_codes": result["skipped"],
            "total_time": total_time,
            "avg_time_per_code": avg_time,
<<<<<<< HEAD
            "success_rate": result["success"] / result["total"] if result["total"] else 0,
            "batch_size": batch_size
=======
            "success_rate": result["success"] / result["total"]
            if result["total"]
            else 0,
            "batch_size": batch_size,
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        }

        logger.info(f"マスタ一括更新完了: {performance_data}")
        return performance_data

    def run_comparison_test(self, test_codes: List[str] = None) -> Dict[str, Any]:
        """包括的な比較テストを実行"""
        if test_codes is None:
            test_codes = self.test_codes

        logger.info("=" * 60)
        logger.info("銘柄一括取得パフォーマンス比較テスト開始")
        logger.info(f"テスト対象: {len(test_codes)}銘柄")
        logger.info("=" * 60)

        results = {}

        try:
            # 1. StockFetcher個別取得テスト
            logger.info("1/4: StockFetcher個別取得テスト")
            results["fetcher_individual"] = self.test_individual_fetch(test_codes)
            time.sleep(2)  # API負荷軽減のため

            # 2. StockFetcher一括取得テスト
            logger.info("2/4: StockFetcher一括取得テスト")
            results["fetcher_bulk"] = self.test_bulk_fetch(test_codes, batch_size=5)
            time.sleep(2)

            # 3. StockMaster個別更新テスト（軽量版）
            logger.info("3/4: StockMaster個別更新テスト（スキップ - DB負荷考慮）")
            # DB負荷を考慮してスキップ
            results["master_individual"] = {
                "method": "master_individual",
                "status": "skipped",
<<<<<<< HEAD
                "reason": "database_load_consideration"
=======
                "reason": "database_load_consideration",
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            }

            # 4. StockMaster一括更新テスト（軽量版）
            logger.info("4/4: StockMaster一括更新テスト（スキップ - DB負荷考慮）")
            # DB負荷を考慮してスキップ
            results["master_bulk"] = {
                "method": "master_bulk",
                "status": "skipped",
<<<<<<< HEAD
                "reason": "database_load_consideration"
=======
                "reason": "database_load_consideration",
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            }

        except Exception as e:
            logger.error(f"比較テスト中にエラー: {e}")

        # 結果の分析
        self.analyze_results(results)

        return results

    def analyze_results(self, results: Dict[str, Any]):
        """テスト結果の分析と表示"""
        logger.info("=" * 60)
        logger.info("パフォーマンステスト結果分析")
        logger.info("=" * 60)

        # 有効な結果のみを分析
<<<<<<< HEAD
        valid_results = {k: v for k, v in results.items()
                        if isinstance(v, dict) and "total_time" in v}
=======
        valid_results = {
            k: v
            for k, v in results.items()
            if isinstance(v, dict) and "total_time" in v
        }
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

        if len(valid_results) < 2:
            logger.warning("比較に十分な結果が得られませんでした")
            return

        # パフォーマンス比較
        for method_name, data in valid_results.items():
<<<<<<< HEAD
            logger.info(f"{method_name.upper()}: "
                       f"{data['total_time']:.2f}秒 "
                       f"({data['avg_time_per_code']:.3f}秒/銘柄) "
                       f"成功率: {data['success_rate']:.1%}")
=======
            logger.info(
                f"{method_name.upper()}: "
                f"{data['total_time']:.2f}秒 "
                f"({data['avg_time_per_code']:.3f}秒/銘柄) "
                f"成功率: {data['success_rate']:.1%}"
            )
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

        # 改善効果の計算
        if "fetcher_individual" in valid_results and "fetcher_bulk" in valid_results:
            individual_time = valid_results["fetcher_individual"]["total_time"]
            bulk_time = valid_results["fetcher_bulk"]["total_time"]

            if individual_time > 0:
                improvement = (individual_time - bulk_time) / individual_time * 100
<<<<<<< HEAD
                speedup = individual_time / bulk_time if bulk_time > 0 else float('inf')
=======
                speedup = individual_time / bulk_time if bulk_time > 0 else float("inf")
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

                logger.info("=" * 40)
                logger.info("StockFetcher改善効果:")
                logger.info(f"  時間短縮: {improvement:.1f}%")
                logger.info(f"  速度向上: {speedup:.1f}倍")
                logger.info("=" * 40)


def main():
    """メイン実行関数"""
    logger.info("銘柄一括取得パフォーマンステスト開始")

    # テストインスタンス作成
    test = BulkFetchPerformanceTest()

    # 比較テスト実行
    results = test.run_comparison_test()

    logger.info("パフォーマンステスト完了")
    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("テストが中断されました")
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}", exc_info=True)
