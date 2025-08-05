#!/usr/bin/env python3
"""
テストデータベース自動初期化スクリプト

CI/CD環境でテスト実行前にデータベースを自動的に初期化し、
テスト用の基本データを投入する。
"""

<<<<<<< HEAD
import os
import sys
import sqlite3
from pathlib import Path
import logging
=======
import logging
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

<<<<<<< HEAD
from src.day_trade.models.database import db_manager, Base
from src.day_trade.models.stock import Stock, PriceData, WatchlistItem, Alert
from src.day_trade.models.enums import AlertType
from datetime import datetime, timedelta
from decimal import Decimal
=======
# isortとruffの要件を満たすため、パス設定後のimportは別ブロックとして配置
if True:  # パス設定後のimportブロック
    from src.day_trade.models.database import Base, db_manager
    from src.day_trade.models.enums import AlertType
    from src.day_trade.models.stock import Alert, PriceData, Stock, WatchlistItem
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_test_database():
    """テスト用データベースをセットアップ"""

    logger.info("🗄️ テストデータベース初期化開始")

    try:
        # テスト用データベースファイルのパス
        test_db_path = "test_day_trade.db"

        # 既存のテストDBがあれば削除
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
            logger.info(f"既存のテストDB削除: {test_db_path}")

        # 環境変数でテスト用DBを指定
        os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"

        # データベース初期化
        engine = db_manager.engine
        Base.metadata.create_all(engine)
        logger.info("✅ テーブル作成完了")

        # テスト用基本データを投入
        populate_test_data()

        logger.info("🎉 テストデータベース初期化完了")
        return True

    except Exception as e:
        logger.error(f"❌ テストデータベース初期化失敗: {e}")
        return False


def populate_test_data():
    """テスト用の基本データを投入"""

    logger.info("📊 テスト用データ投入開始")

    with db_manager.session_scope() as session:
<<<<<<< HEAD

=======
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        # 1. テスト用銘柄データ
        test_stocks = [
            Stock(
                code="7203",
                name="トヨタ自動車",
                market="東証プライム",
                sector="自動車・輸送機器",
<<<<<<< HEAD
                industry="自動車"
=======
                industry="自動車",
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            ),
            Stock(
                code="9984",
                name="ソフトバンクグループ",
                market="東証プライム",
                sector="情報・通信業",
<<<<<<< HEAD
                industry="通信"
=======
                industry="通信",
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            ),
            Stock(
                code="6758",
                name="ソニーグループ",
                market="東証プライム",
                sector="電気機器",
<<<<<<< HEAD
                industry="エレクトロニクス"
=======
                industry="エレクトロニクス",
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            ),
            Stock(
                code="4063",
                name="信越化学工業",
                market="東証プライム",
                sector="化学",
<<<<<<< HEAD
                industry="化学"
=======
                industry="化学",
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            ),
            Stock(
                code="8306",
                name="三菱UFJフィナンシャル・グループ",
                market="東証プライム",
                sector="銀行業",
<<<<<<< HEAD
                industry="銀行"
            )
=======
                industry="銀行",
            ),
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        ]

        # 既存の銘柄データを確認してUpsert
        added_count = 0
        for stock in test_stocks:
<<<<<<< HEAD
            existing_stock = session.query(Stock).filter(Stock.code == stock.code).first()
=======
            existing_stock = (
                session.query(Stock).filter(Stock.code == stock.code).first()
            )
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            if not existing_stock:
                session.add(stock)
                added_count += 1
            else:
                # 既存の銘柄情報を更新
                existing_stock.name = stock.name
                existing_stock.market = stock.market
                existing_stock.sector = stock.sector
                existing_stock.industry = stock.industry

        session.flush()  # IDを取得するため
<<<<<<< HEAD
        logger.info(f"テスト銘柄データ投入完了: {added_count}件追加, {len(test_stocks) - added_count}件更新")
=======
        logger.info(
            f"テスト銘柄データ投入完了: {added_count}件追加, {len(test_stocks) - added_count}件更新"
        )
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

        # 2. テスト用価格データ（過去30日分）
        base_date = datetime.now() - timedelta(days=30)
        price_data_entries = []

        for stock in test_stocks:
            base_price = {
                "7203": Decimal("2800.00"),
                "9984": Decimal("9500.00"),
                "6758": Decimal("15000.00"),
                "4063": Decimal("25000.00"),
<<<<<<< HEAD
                "8306": Decimal("1200.00")
=======
                "8306": Decimal("1200.00"),
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            }.get(stock.code, Decimal("1000.00"))

            for i in range(30):
                date = base_date + timedelta(days=i)
                # 簡単な価格変動シミュレーション
                variation = Decimal(str((i % 10 - 5) * 0.02))  # -10% to +10%
                price = base_price * (Decimal("1.0") + variation)

                price_entry = PriceData(
                    stock_code=stock.code,
                    datetime=date,
                    open=price * Decimal("0.99"),
                    high=price * Decimal("1.02"),
                    low=price * Decimal("0.98"),
                    close=price,
<<<<<<< HEAD
                    volume=10000 + (i * 1000)
=======
                    volume=10000 + (i * 1000),
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
                )
                price_data_entries.append(price_entry)

        for entry in price_data_entries:
            session.add(entry)

        logger.info(f"✅ テスト価格データ投入完了: {len(price_data_entries)}件")

        # 3. テスト用ウォッチリストデータ
        watchlist_items = [
            WatchlistItem(
<<<<<<< HEAD
                stock_code="7203",
                group_name="自動車株",
                memo="トヨタ監視用"
            ),
            WatchlistItem(
                stock_code="9984",
                group_name="通信株",
                memo="ソフトバンク監視用"
            ),
            WatchlistItem(
                stock_code="6758",
                group_name="エレクトロニクス",
                memo="ソニー監視用"
            )
=======
                stock_code="7203", group_name="自動車株", memo="トヨタ監視用"
            ),
            WatchlistItem(
                stock_code="9984", group_name="通信株", memo="ソフトバンク監視用"
            ),
            WatchlistItem(
                stock_code="6758", group_name="エレクトロニクス", memo="ソニー監視用"
            ),
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        ]

        for item in watchlist_items:
            session.add(item)

        logger.info(f"✅ テストウォッチリストデータ投入完了: {len(watchlist_items)}件")

        # 4. テスト用アラートデータ
        alert_items = [
            Alert(
                stock_code="7203",
                alert_type=AlertType.PRICE_ABOVE,
                threshold=3000.0,
                memo="高値警戒",
<<<<<<< HEAD
                is_active=True
=======
                is_active=True,
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            ),
            Alert(
                stock_code="9984",
                alert_type=AlertType.PRICE_BELOW,
                threshold=9000.0,
                memo="安値注意",
<<<<<<< HEAD
                is_active=True
=======
                is_active=True,
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            ),
            Alert(
                stock_code="6758",
                alert_type=AlertType.CHANGE_PERCENT_UP,
                threshold=5.0,
                memo="急騰監視",
<<<<<<< HEAD
                is_active=True
            )
=======
                is_active=True,
            ),
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
        ]

        for alert in alert_items:
            session.add(alert)

        logger.info(f"✅ テストアラートデータ投入完了: {len(alert_items)}件")


def verify_test_database():
    """テストデータベースの整合性確認"""

    logger.info("🔍 テストデータベース整合性確認")

    try:
        with db_manager.session_scope() as session:
<<<<<<< HEAD

=======
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            # 各テーブルのレコード数確認
            stock_count = session.query(Stock).count()
            price_count = session.query(PriceData).count()
            watchlist_count = session.query(WatchlistItem).count()
            alert_count = session.query(Alert).count()

<<<<<<< HEAD
            logger.info(f"📊 データ確認結果:")
=======
            logger.info("📊 データ確認結果:")
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            logger.info(f"  - 銘柄: {stock_count}件")
            logger.info(f"  - 価格データ: {price_count}件")
            logger.info(f"  - ウォッチリスト: {watchlist_count}件")
            logger.info(f"  - アラート: {alert_count}件")

            # 基本的な整合性チェック
            if stock_count == 0:
                raise ValueError("銘柄データが投入されていません")

            if price_count == 0:
                raise ValueError("価格データが投入されていません")

            # リレーション整合性チェック
            test_stock = session.query(Stock).filter(Stock.code == "7203").first()
            if not test_stock:
                raise ValueError("テスト銘柄(7203)が見つかりません")

<<<<<<< HEAD
            test_prices = session.query(PriceData).filter(PriceData.stock_code == "7203").count()
=======
            test_prices = (
                session.query(PriceData).filter(PriceData.stock_code == "7203").count()
            )
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)
            if test_prices == 0:
                raise ValueError("テスト銘柄(7203)の価格データが見つかりません")

            logger.info("✅ データベース整合性確認完了")
            return True

    except Exception as e:
        logger.error(f"❌ データベース整合性確認失敗: {e}")
        return False


def cleanup_test_database():
    """テストデータベースのクリーンアップ"""

    test_db_path = "test_day_trade.db"

    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        logger.info(f"🗑️ テストデータベース削除: {test_db_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="テストデータベース管理")
<<<<<<< HEAD
    parser.add_argument("action", choices=["setup", "verify", "cleanup"],
                       help="実行するアクション")
=======
    parser.add_argument(
        "action", choices=["setup", "verify", "cleanup"], help="実行するアクション"
    )
>>>>>>> 5f9b0b2 (fix: 最重要問題である循環importエラーを解決)

    args = parser.parse_args()

    if args.action == "setup":
        success = setup_test_database()
        if success and verify_test_database():
            logger.info("🎉 テストデータベースセットアップ完了")
            sys.exit(0)
        else:
            logger.error("❌ テストデータベースセットアップ失敗")
            sys.exit(1)

    elif args.action == "verify":
        if verify_test_database():
            logger.info("✅ テストデータベース正常")
            sys.exit(0)
        else:
            logger.error("❌ テストデータベース異常")
            sys.exit(1)

    elif args.action == "cleanup":
        cleanup_test_database()
        logger.info("🗑️ テストデータベースクリーンアップ完了")
        sys.exit(0)
