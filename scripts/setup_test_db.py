#!/usr/bin/env python3
"""
テストデータベースセットアップスクリプト

一括銘柄登録のテスト用データベース環境を構築する。

機能:
- テスト用データベースの作成・初期化
- サンプル銘柄データの投入
- データベーススキーマの検証
- テストデータのクリーンアップ
- パフォーマンステスト用大量データ生成

Usage:
    python scripts/setup_test_db.py --init
    python scripts/setup_test_db.py --sample-data 1000
    python scripts/setup_test_db.py --validate
    python scripts/setup_test_db.py --cleanup
"""

import argparse
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.day_trade.models.database import DatabaseConfig, DatabaseManager, get_default_database_manager
from src.day_trade.models.stock import Stock, PriceData, Trade, WatchlistItem, Alert
from src.day_trade.models.enums import TradeType, AlertType
from src.day_trade.utils.logging_config import get_context_logger

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_context_logger(__name__)


class TestDataGenerator:
    """テストデータ生成"""

    # 実際の企業名・セクターの例
    SAMPLE_COMPANIES = [
        ("トヨタ自動車", "輸送用機器", "自動車"),
        ("ソニーグループ", "電気機器", "電気機器"),
        ("ソフトバンクグループ", "情報・通信", "通信業"),
        ("キーエンス", "電気機器", "電子部品"),
        ("ファーストリテイリング", "小売業", "小売業"),
        ("東京エレクトロン", "電気機器", "半導体"),
        ("信越化学工業", "化学", "化学"),
        ("KDDI", "情報・通信", "通信業"),
        ("NTTドコモ", "情報・通信", "通信業"),
        ("日本電信電話", "情報・通信", "通信業"),
        ("三菱UFJフィナンシャル・グループ", "銀行業", "銀行業"),
        ("三井住友フィナンシャルグループ", "銀行業", "銀行業"),
        ("みずほフィナンシャルグループ", "銀行業", "銀行業"),
        ("任天堂", "情報・通信", "ゲーム・エンタメ"),
        ("村田製作所", "電気機器", "電子部品"),
        ("アサヒグループホールディングス", "食料品", "飲料"),
        ("花王", "化学", "日用品"),
        ("武田薬品工業", "医薬品", "医薬品"),
        ("第一三共", "医薬品", "医薬品"),
        ("中外製薬", "医薬品", "医薬品")
    ]

    MARKETS = ["東証プライム", "東証スタンダード", "東証グロース", "札証", "名証"]

    def __init__(self):
        self.generated_codes: set = set()

    def generate_stock_data(self, count: int) -> List[Dict[str, str]]:
        """株式データを生成"""
        stocks = []

        for i in range(count):
            # ユニークなコードを生成
            code = self._generate_unique_code()

            # ランダムに実際の企業データまたは生成データを選択
            if i < len(self.SAMPLE_COMPANIES) and random.random() > 0.3:
                name, sector, industry = self.SAMPLE_COMPANIES[i]
            else:
                name = f"テスト会社{i:04d}"
                sector = random.choice([
                    "情報・通信", "電気機器", "輸送用機器", "化学", "医薬品",
                    "食料品", "小売業", "銀行業", "不動産業", "建設業"
                ])
                industry = f"{sector}関連"

            market = random.choice(self.MARKETS)

            stocks.append({
                "code": code,
                "name": name,
                "market": market,
                "sector": sector,
                "industry": industry
            })

        return stocks

    def _generate_unique_code(self) -> str:
        """ユニークな4桁コードを生成"""
        while True:
            code = f"{random.randint(1000, 9999)}"
            if code not in self.generated_codes:
                self.generated_codes.add(code)
                return code

    def generate_price_data(self, stock_codes: List[str], days: int = 100) -> List[Dict[str, Any]]:
        """価格データを生成"""
        price_data = []
        base_date = datetime.now() - timedelta(days=days)

        for stock_code in stock_codes:
            # 基準価格をランダムに設定
            base_price = random.uniform(500, 5000)
            current_price = base_price

            for day in range(days):
                date = base_date + timedelta(days=day)

                # 価格変動をシミュレート
                change_percent = random.uniform(-0.05, 0.05)  # ±5%
                current_price *= (1 + change_percent)
                current_price = max(current_price, base_price * 0.5)  # 最低価格制限

                # OHLC生成
                high = current_price * random.uniform(1.0, 1.02)
                low = current_price * random.uniform(0.98, 1.0)
                open_price = current_price * random.uniform(0.995, 1.005)

                volume = random.randint(10000, 1000000)

                price_data.append({
                    "stock_code": stock_code,
                    "datetime": date,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(current_price, 2),
                    "volume": volume
                })

        return price_data

    def generate_trade_data(self, stock_codes: List[str], count: int) -> List[Dict[str, Any]]:
        """取引データを生成"""
        trades = []
        base_date = datetime.now() - timedelta(days=30)

        for i in range(count):
            stock_code = random.choice(stock_codes)
            trade_type = random.choice([TradeType.BUY, TradeType.SELL])
            quantity = random.randint(100, 1000) * 100  # 100株単位
            price = round(random.uniform(500, 5000), 2)
            commission = round(price * quantity * 0.001, 2)  # 0.1%の手数料

            trade_date = base_date + timedelta(days=random.randint(0, 30))

            trades.append({
                "stock_code": stock_code,
                "trade_type": trade_type,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "trade_datetime": trade_date,
                "memo": f"テスト取引{i}"
            })

        return trades


class TestDatabaseManager:
    """テストデータベース管理"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: データベースファイルパス（Noneの場合はデフォルト）
        """
        if db_path:
            config = DatabaseConfig(database_url=f"sqlite:///{db_path}")
            self.db_manager = DatabaseManager(config)
        else:
            self.db_manager = get_default_database_manager()

        self.data_generator = TestDataGenerator()

    def initialize_database(self):
        """データベースを初期化"""
        logger.info("データベース初期化開始")

        try:
            # テーブル削除（既存データをクリア）
            self.db_manager.drop_tables()
            logger.info("既存テーブルを削除")

            # テーブル作成
            self.db_manager.create_tables()
            logger.info("テーブルを作成")

            logger.info("データベース初期化完了")

        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
            raise

    def insert_sample_stocks(self, count: int):
        """サンプル銘柄データを挿入"""
        logger.info(f"サンプル銘柄データ挿入開始: {count}件")

        try:
            stock_data = self.data_generator.generate_stock_data(count)

            with self.db_manager.session_scope() as session:
                for data in stock_data:
                    stock = Stock(**data)
                    session.add(stock)

                logger.info(f"銘柄データ挿入完了: {count}件")

            return [data["code"] for data in stock_data]

        except Exception as e:
            logger.error(f"銘柄データ挿入エラー: {e}")
            raise

    def insert_sample_price_data(self, stock_codes: List[str], days: int = 100):
        """サンプル価格データを挿入"""
        logger.info(f"価格データ挿入開始: {len(stock_codes)}銘柄 x {days}日")

        try:
            price_data = self.data_generator.generate_price_data(stock_codes, days)

            batch_size = 1000
            total_records = len(price_data)

            with self.db_manager.session_scope() as session:
                for i in range(0, total_records, batch_size):
                    batch = price_data[i:i + batch_size]

                    for data in batch:
                        price_record = PriceData(**data)
                        session.add(price_record)

                    # 進捗表示
                    processed = min(i + batch_size, total_records)
                    logger.info(f"価格データ処理中: {processed}/{total_records} ({processed/total_records*100:.1f}%)")

            logger.info(f"価格データ挿入完了: {total_records}件")

        except Exception as e:
            logger.error(f"価格データ挿入エラー: {e}")
            raise

    def insert_sample_trades(self, stock_codes: List[str], count: int):
        """サンプル取引データを挿入"""
        logger.info(f"取引データ挿入開始: {count}件")

        try:
            trade_data = self.data_generator.generate_trade_data(stock_codes, count)

            with self.db_manager.session_scope() as session:
                for data in trade_data:
                    trade = Trade(**data)
                    session.add(trade)

            logger.info(f"取引データ挿入完了: {count}件")

        except Exception as e:
            logger.error(f"取引データ挿入エラー: {e}")
            raise

    def validate_database_schema(self) -> Dict[str, Any]:
        """データベーススキーマを検証"""
        logger.info("データベーススキーマ検証開始")

        validation_results = {
            "tables": {},
            "constraints": {},
            "indexes": {},
            "data_counts": {}
        }

        try:
            with self.db_manager.session_scope() as session:
                # テーブル存在確認とデータ件数チェック
                tables_to_check = [
                    (Stock, "stocks"),
                    (PriceData, "price_data"),
                    (Trade, "trades"),
                    (WatchlistItem, "watchlist_items"),
                    (Alert, "alerts")
                ]

                for model, table_name in tables_to_check:
                    try:
                        count = session.query(model).count()
                        validation_results["tables"][table_name] = "存在"
                        validation_results["data_counts"][table_name] = count
                        logger.info(f"テーブル {table_name}: {count}件")

                    except Exception as e:
                        validation_results["tables"][table_name] = f"エラー: {e}"
                        validation_results["data_counts"][table_name] = 0

                # 基本的な整合性チェック
                stock_count = session.query(Stock).count()
                price_data_count = session.query(PriceData).count()

                if stock_count > 0 and price_data_count > 0:
                    avg_price_per_stock = price_data_count / stock_count
                    validation_results["data_quality"] = {
                        "avg_price_records_per_stock": avg_price_per_stock,
                        "data_integrity": "良好" if avg_price_per_stock > 10 else "要確認"
                    }

            logger.info("データベーススキーマ検証完了")
            return validation_results

        except Exception as e:
            logger.error(f"スキーマ検証エラー: {e}")
            validation_results["error"] = str(e)
            return validation_results

    def cleanup_test_data(self):
        """テストデータをクリーンアップ"""
        logger.info("テストデータクリーンアップ開始")

        try:
            with self.db_manager.session_scope() as session:
                # 外部キー制約を考慮した順序で削除

                # アラートを削除
                alert_count = session.query(Alert).count()
                session.query(Alert).delete()
                logger.info(f"アラート削除: {alert_count}件")

                # ウォッチリストを削除
                watchlist_count = session.query(WatchlistItem).count()
                session.query(WatchlistItem).delete()
                logger.info(f"ウォッチリスト削除: {watchlist_count}件")

                # 取引履歴を削除
                trade_count = session.query(Trade).count()
                session.query(Trade).delete()
                logger.info(f"取引履歴削除: {trade_count}件")

                # 価格データを削除
                price_count = session.query(PriceData).count()
                session.query(PriceData).delete()
                logger.info(f"価格データ削除: {price_count}件")

                # 銘柄データを削除
                stock_count = session.query(Stock).count()
                session.query(Stock).delete()
                logger.info(f"銘柄データ削除: {stock_count}件")

            logger.info("テストデータクリーンアップ完了")

        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")
            raise

    def generate_performance_test_data(self, stock_count: int = 5000):
        """パフォーマンステスト用の大量データを生成"""
        logger.info(f"パフォーマンステスト用データ生成開始: {stock_count}銘柄")

        start_time = time.time()

        try:
            # 大量の銘柄データを生成
            stock_codes = self.insert_sample_stocks(stock_count)

            # 価格データを生成（少数の銘柄のみ、メモリ制限のため）
            sample_codes = stock_codes[:min(100, len(stock_codes))]
            self.insert_sample_price_data(sample_codes, days=365)

            # 取引データを生成
            trade_count = stock_count * 2  # 銘柄数の2倍の取引
            self.insert_sample_trades(stock_codes, trade_count)

            elapsed_time = time.time() - start_time
            logger.info(f"パフォーマンステスト用データ生成完了: {elapsed_time:.2f}秒")

            return {
                "stock_count": stock_count,
                "price_data_stocks": len(sample_codes),
                "price_data_days": 365,
                "trade_count": trade_count,
                "generation_time": elapsed_time
            }

        except Exception as e:
            logger.error(f"パフォーマンステスト用データ生成エラー: {e}")
            raise


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="テストデータベースセットアップスクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python scripts/setup_test_db.py --init
    python scripts/setup_test_db.py --sample-data 1000
    python scripts/setup_test_db.py --validate
    python scripts/setup_test_db.py --performance-data 5000
    python scripts/setup_test_db.py --cleanup
        """
    )

    parser.add_argument(
        '--init',
        action='store_true',
        help='データベースを初期化（既存データは削除される）'
    )

    parser.add_argument(
        '--sample-data',
        type=int,
        metavar='COUNT',
        help='サンプルデータを生成（銘柄数を指定）'
    )

    parser.add_argument(
        '--performance-data',
        type=int,
        metavar='COUNT',
        help='パフォーマンステスト用の大量データを生成'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='データベーススキーマと整合性を検証'
    )

    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='テストデータをクリーンアップ'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        help='データベースファイルパス（指定しない場合はデフォルト）'
    )

    args = parser.parse_args()

    try:
        logger.info("=== テストデータベースセットアップ開始 ===")

        # データベースマネージャー初期化
        db_manager = TestDatabaseManager(args.db_path)

        # 初期化
        if args.init:
            db_manager.initialize_database()

        # サンプルデータ生成
        if args.sample_data:
            if args.sample_data <= 0:
                logger.error("サンプルデータ数は1以上である必要があります")
                return 1

            stock_codes = db_manager.insert_sample_stocks(args.sample_data)

            # 追加データ生成（小規模なサンプル）
            if args.sample_data <= 100:
                db_manager.insert_sample_price_data(stock_codes, days=30)
                db_manager.insert_sample_trades(stock_codes, args.sample_data)
                logger.info("サンプル用の価格・取引データも生成しました")

        # パフォーマンステスト用データ
        if args.performance_data:
            if args.performance_data <= 0:
                logger.error("パフォーマンステスト用データ数は1以上である必要があります")
                return 1

            stats = db_manager.generate_performance_test_data(args.performance_data)
            logger.info(f"パフォーマンステスト用データ統計: {stats}")

        # 検証
        if args.validate:
            validation_result = db_manager.validate_database_schema()

            logger.info("=== データベース検証結果 ===")
            for table_name, status in validation_result["tables"].items():
                count = validation_result["data_counts"].get(table_name, 0)
                logger.info(f"テーブル {table_name}: {status} ({count}件)")

            if "data_quality" in validation_result:
                quality = validation_result["data_quality"]
                logger.info(f"データ品質: {quality['data_integrity']}")
                logger.info(f"銘柄あたり平均価格データ: {quality['avg_price_records_per_stock']:.1f}件")

            if "error" in validation_result:
                logger.error(f"検証エラー: {validation_result['error']}")
                return 1

        # クリーンアップ
        if args.cleanup:
            confirmation = input("全てのテストデータを削除しますか？ (yes/no): ")
            if confirmation.lower() in ['yes', 'y']:
                db_manager.cleanup_test_data()
            else:
                logger.info("クリーンアップをキャンセルしました")

        logger.info("=== テストデータベースセットアップ完了 ===")

        return 0

    except KeyboardInterrupt:
        logger.info("処理が中断されました")
        return 1

    except Exception as e:
        logger.error(f"予期しないエラー: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
