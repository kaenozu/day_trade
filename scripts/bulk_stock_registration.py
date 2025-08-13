#!/usr/bin/env python3
"""
銘柄一括登録スクリプト

JPX（日本取引所グループ）から提供される上場銘柄一覧CSVを利用して、
銘柄マスタを一括で更新する。

機能:
1. JPX上場銘柄一覧CSVのダウンロード
2. CSVファイルの解析とデータ抽出
3. 既存銘柄の更新・新規銘柄の追加
4. エラーハンドリングと進捗表示
5. 詳細なログ出力

Usage:
    python scripts/bulk_stock_registration.py
    python scripts/bulk_stock_registration.py --csv-path data.csv --batch-size 100
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.day_trade.data.stock_master import create_stock_master_manager  # noqa: E402
from src.day_trade.models.database import get_default_database_manager  # noqa: E402
from src.day_trade.models.stock import Stock  # noqa: E402
from src.day_trade.utils.logging_config import get_context_logger  # noqa: E402

# ログ設定
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)  # logsディレクトリがない場合は作成

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            log_dir
            / f"bulk_registration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = get_context_logger(__name__)


class JPXDataDownloader:
    """JPX上場銘柄一覧ダウンローダー"""

    # JPXの上場銘柄一覧CSV URL（2024年現在）
    JPX_LISTED_STOCKS_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

    # 代替URL（CSVフォーマット）
    ALTERNATIVE_URLS = [
        "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/listed_securities.csv",
        # 他の可能性があるURL
    ]

    def __init__(self, timeout: int = 30, retries: int = 3):
        """
        Args:
            timeout: リクエストタイムアウト（秒）
            retries: 再試行回数
        """
        self.timeout = timeout
        self.session = requests.Session()

        # リトライ設定
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # User-Agentを設定
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def download_csv(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        JPX上場銘柄一覧CSVをダウンロード

        Args:
            output_path: 出力ファイルパス（Noneの場合は一時ファイルを使用）

        Returns:
            ダウンロードしたファイルのパス（失敗時はNone）
        """
        if output_path is None:
            output_path = (
                PROJECT_ROOT
                / "data"
                / f"jpx_stocks_{datetime.now().strftime('%Y%m%d')}.csv"
            )

        # ディレクトリを作成
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # メインURLを試行
        success = self._download_from_url(self.JPX_LISTED_STOCKS_URL, output_path)

        if not success:
            logger.warning(
                f"メインURL {self.JPX_LISTED_STOCKS_URL} からのダウンロードに失敗、代替URLを試行"
            )

            # 代替URLを試行
            for alt_url in self.ALTERNATIVE_URLS:
                logger.info(f"代替URL試行: {alt_url}")
                if self._download_from_url(alt_url, output_path):
                    success = True
                    break

        if success:
            logger.info(f"銘柄一覧ダウンロード完了: {output_path}")
            return output_path
        else:
            logger.error("すべてのURLからのダウンロードに失敗しました")
            return None

    def _download_from_url(self, url: str, output_path: Path) -> bool:
        """
        指定URLからファイルをダウンロード

        Args:
            url: ダウンロードURL
            output_path: 出力先パス

        Returns:
            成功時True、失敗時False
        """
        try:
            logger.info(f"ダウンロード開始: {url}")

            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()

            # ファイルサイズチェック
            content_length = response.headers.get("content-length")
            if content_length:
                file_size = int(content_length)
                logger.info(f"ファイルサイズ: {file_size / 1024 / 1024:.2f} MB")

            # ファイル保存
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # ファイルサイズ検証
            actual_size = output_path.stat().st_size
            if actual_size == 0:
                logger.error("ダウンロードファイルのサイズが0バイトです")
                return False

            logger.info(f"ダウンロード成功: {actual_size} バイト")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"ダウンロードエラー: {e}")
            return False
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            return False


class StockDataParser:
    """株価データパーサー"""

    def __init__(self):
        self.parsed_stocks: List[Dict[str, str]] = []
        self.error_count = 0

    def parse_csv(self, csv_path: Path) -> Tuple[List[Dict[str, str]], int]:
        """
        CSVファイルを解析して銘柄データを抽出

        Args:
            csv_path: CSVファイルパス

        Returns:
            (銘柄データのリスト, エラー数)
        """
        logger.info(f"CSVファイル解析開始: {csv_path}")

        try:
            # エンコーディング検出を試行
            encodings = ["utf-8", "shift_jis", "cp932", "utf-8-sig"]

            for encoding in encodings:
                try:
                    with open(csv_path, encoding=encoding) as f:
                        f.read()  # エンコーディング確認のみ
                        logger.info(f"エンコーディング {encoding} で読み込み成功")
                        break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(
                    "サポートされているエンコーディングでファイルを読み込めませんでした"
                )

            # CSV解析
            with open(csv_path, encoding=encoding) as f:
                # CSV方言を検出
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter

                logger.info(f"CSV区切り文字: '{delimiter}'")

                reader = csv.DictReader(f, delimiter=delimiter)

                # ヘッダー検証
                headers = reader.fieldnames
                logger.info(f"CSVヘッダー: {headers}")

                # 必要なフィールドをマッピング
                field_mapping = self._detect_field_mapping(headers)

                if not field_mapping:
                    raise ValueError("必要なフィールドが見つかりません")

                # データ読み込み
                row_count = 0
                for row in reader:
                    try:
                        stock_data = self._extract_stock_data(row, field_mapping)
                        if stock_data:
                            self.parsed_stocks.append(stock_data)
                        else:
                            self.error_count += 1
                            logger.debug(
                                f"無効なデータをスキップ (行 {row_count + 1}): {row}"
                            )

                        row_count += 1

                        if row_count % 100 == 0:
                            logger.info(f"処理済み行数: {row_count}")

                    except Exception as e:
                        self.error_count += 1
                        logger.warning(f"行解析エラー (行 {row_count + 1}): {e}")

                        if self.error_count > 50:  # エラーが多すぎる場合は中断
                            logger.error("エラーが多すぎるため処理を中断します")
                            break

            logger.info(
                f"CSV解析完了: {len(self.parsed_stocks)}件の銘柄を抽出、{self.error_count}件のエラー"
            )
            return self.parsed_stocks, self.error_count

        except Exception as e:
            logger.error(f"CSV解析エラー: {e}")
            return [], self.error_count

    def _detect_field_mapping(self, headers: List[str]) -> Optional[Dict[str, str]]:
        """
        CSVヘッダーから必要フィールドのマッピングを検出

        Args:
            headers: CSVヘッダーのリスト

        Returns:
            フィールドマッピング辞書（見つからない場合はNone）
        """
        # 可能性のあるフィールド名パターン
        field_patterns = {
            "code": ["コード", "code", "証券コード", "銘柄コード", "銘柄CODE", "Code"],
            "name": ["銘柄名", "name", "名称", "会社名", "銘柄", "Name"],
            "market": ["市場", "market", "市場区分", "Market", "Section"],
            "sector": ["セクター", "sector", "業種", "33業種", "Sector"],
            "industry": ["業種名", "industry", "業種区分", "Industry"],
        }

        mapping = {}

        for field, patterns in field_patterns.items():
            found = False
            for pattern in patterns:
                for header in headers:
                    if header and pattern.lower() in header.lower():
                        mapping[field] = header
                        found = True
                        break
                if found:
                    break

        # 最低限必要なフィールド（コード、名称）があるかチェック
        if "code" in mapping and "name" in mapping:
            logger.info(f"フィールドマッピング検出: {mapping}")
            return mapping
        else:
            logger.error(
                f"必須フィールドが見つかりません。検出されたマッピング: {mapping}"
            )
            return None

    def _extract_stock_data(
        self, row: Dict[str, str], field_mapping: Dict[str, str]
    ) -> Optional[Dict[str, str]]:
        """
        CSVの1行から銘柄データを抽出

        Args:
            row: CSV行データ
            field_mapping: フィールドマッピング

        Returns:
            銘柄データ辞書（無効な場合はNone）
        """
        try:
            # 必須フィールドを抽出
            code = row.get(field_mapping["code"], "").strip()
            name = row.get(field_mapping["name"], "").strip()

            # コードの検証
            if not code or not code.isdigit() or len(code) != 4:
                return None

            # 名称の検証
            if not name:
                return None

            # オプションフィールドを抽出
            market = row.get(field_mapping.get("market", ""), "").strip()
            sector = row.get(field_mapping.get("sector", ""), "").strip()
            industry = row.get(field_mapping.get("industry", ""), "").strip()

            return {
                "code": code,
                "name": name,
                "market": market if market else "未分類",
                "sector": sector if sector else "未分類",
                "industry": industry if industry else "未分類",
            }

        except Exception as e:
            logger.warning(f"データ抽出エラー: {e}")
            return None


class BulkStockRegistrar:
    """銘柄一括登録処理"""

    def __init__(self, batch_size: int = 50):
        """
        Args:
            batch_size: バッチサイズ
        """
        self.batch_size = batch_size
        self.stock_master = create_stock_master_manager()
        self.db_manager = get_default_database_manager()

        self.stats = {
            "total_processed": 0,
            "newly_added": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": time.time(),
        }

    def register_stocks(self, stock_data_list: List[Dict[str, str]]) -> Dict[str, int]:
        """
        銘柄データを一括登録

        Args:
            stock_data_list: 銘柄データのリスト

        Returns:
            処理統計情報
        """
        logger.info(f"銘柄一括登録開始: {len(stock_data_list)}件")

        try:
            # 既存銘柄コードを取得
            existing_codes = self._get_existing_codes()
            logger.info(f"既存銘柄数: {len(existing_codes)}")

            # バッチ処理
            for i in range(0, len(stock_data_list), self.batch_size):
                batch = stock_data_list[i : i + self.batch_size]
                batch_number = i // self.batch_size + 1
                total_batches = (
                    len(stock_data_list) + self.batch_size - 1
                ) // self.batch_size

                logger.info(
                    f"バッチ {batch_number}/{total_batches} 処理中... ({len(batch)}件)"
                )

                self._process_batch(batch, existing_codes)

                # 進捗表示
                processed = min(i + self.batch_size, len(stock_data_list))
                progress = (processed / len(stock_data_list)) * 100
                logger.info(
                    f"進捗: {progress:.1f}% ({processed}/{len(stock_data_list)})"
                )

        except Exception as e:
            logger.error(f"一括登録処理エラー: {e}")
            self.stats["errors"] += 1

        # 統計情報を更新
        self.stats["total_processed"] = len(stock_data_list)
        elapsed_time = time.time() - self.stats["start_time"]

        logger.info("=== 銘柄一括登録完了 ===")
        logger.info(f"処理時間: {elapsed_time:.2f}秒")
        logger.info(f"総処理件数: {self.stats['total_processed']}")
        logger.info(f"新規追加: {self.stats['newly_added']}")
        logger.info(f"更新: {self.stats['updated']}")
        logger.info(f"スキップ: {self.stats['skipped']}")
        logger.info(f"エラー: {self.stats['errors']}")

        return self.stats

    def _get_existing_codes(self) -> Set[str]:
        """既存の銘柄コード一覧を取得"""
        try:
            with self.db_manager.session_scope() as session:
                result = session.query(Stock.code).all()
                return {code[0] for code in result}
        except Exception as e:
            logger.error(f"既存銘柄コード取得エラー: {e}")
            return set()

    def _process_batch(self, batch: List[Dict[str, str]], existing_codes: Set[str]):
        """バッチ処理"""
        try:
            with self.db_manager.session_scope() as session:
                for stock_data in batch:
                    try:
                        self._process_single_stock(session, stock_data, existing_codes)
                    except Exception as e:
                        logger.error(
                            f"銘柄処理エラー {stock_data.get('code', 'UNKNOWN')}: {e}"
                        )
                        self.stats["errors"] += 1
        except Exception as e:
            logger.error(f"バッチ処理エラー: {e}")
            self.stats["errors"] += len(batch)

    def _process_single_stock(
        self, session, stock_data: Dict[str, str], existing_codes: Set[str]
    ):
        """単一銘柄の処理"""
        code = stock_data["code"]

        if code in existing_codes:
            # 既存銘柄の更新
            stock = session.query(Stock).filter(Stock.code == code).first()
            if stock:
                updated = False

                # データが異なる場合のみ更新
                if stock.name != stock_data["name"]:
                    stock.name = stock_data["name"]
                    updated = True

                if stock.market != stock_data["market"]:
                    stock.market = stock_data["market"]
                    updated = True

                if stock.sector != stock_data["sector"]:
                    stock.sector = stock_data["sector"]
                    updated = True

                if stock.industry != stock_data["industry"]:
                    stock.industry = stock_data["industry"]
                    updated = True

                if updated:
                    self.stats["updated"] += 1
                    logger.debug(f"銘柄更新: {code} - {stock_data['name']}")
                else:
                    self.stats["skipped"] += 1
            else:
                self.stats["skipped"] += 1
        else:
            # 新規銘柄の追加
            new_stock = Stock(
                code=code,
                name=stock_data["name"],
                market=stock_data["market"],
                sector=stock_data["sector"],
                industry=stock_data["industry"],
            )
            session.add(new_stock)
            existing_codes.add(code)  # 重複チェック用に追加
            self.stats["newly_added"] += 1
            logger.debug(f"新規銘柄追加: {code} - {stock_data['name']}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="JPX上場銘柄一覧を利用した銘柄マスタ一括更新",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python scripts/bulk_stock_registration.py
    python scripts/bulk_stock_registration.py --csv-path data/stocks.csv
    python scripts/bulk_stock_registration.py --batch-size 100 --skip-download
        """,
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        help="使用するCSVファイルパス（指定しない場合は自動ダウンロード）",
    )

    parser.add_argument(
        "--batch-size", type=int, default=50, help="バッチサイズ（デフォルト: 50）"
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="ダウンロードをスキップして既存CSVファイルを使用",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="テスト実行（実際のデータベース更新は行わない）",
    )

    args = parser.parse_args()

    try:
        # ログディレクトリ作成
        (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

        logger.info("=== 銘柄一括登録処理開始 ===")

        csv_path = None

        # CSVファイルの取得
        if args.csv_path:
            csv_path = Path(args.csv_path)
            if not csv_path.exists():
                logger.error(f"指定されたCSVファイルが見つかりません: {csv_path}")
                return 1
            logger.info(f"指定CSVファイル使用: {csv_path}")

        elif not args.skip_download:
            logger.info("JPX上場銘柄一覧をダウンロード中...")
            downloader = JPXDataDownloader()
            csv_path = downloader.download_csv()

            if csv_path is None:
                logger.error("CSVファイルのダウンロードに失敗しました")
                return 1

        else:
            logger.error("--skip-download指定時は--csv-pathも指定してください")
            return 1

        # CSVファイルの解析
        parser_obj = StockDataParser()
        stock_data_list, error_count = parser_obj.parse_csv(csv_path)

        if not stock_data_list:
            logger.error("有効な銘柄データが見つかりませんでした")
            return 1

        if error_count > 0:
            logger.warning(f"{error_count}件の解析エラーがありました")

        # 一括登録処理
        if args.dry_run:
            logger.info("DRY RUN: 実際のデータベース更新は行いません")
            logger.info(f"登録予定の銘柄数: {len(stock_data_list)}")

            # サンプル表示
            for i, stock in enumerate(stock_data_list[:5]):
                logger.info(f"サンプル {i + 1}: {stock}")

            if len(stock_data_list) > 5:
                logger.info(f"... 他 {len(stock_data_list) - 5} 件")

        else:
            registrar = BulkStockRegistrar(batch_size=args.batch_size)
            stats = registrar.register_stocks(stock_data_list)

            # 成功率チェック
            if stats["errors"] > stats["total_processed"] * 0.1:  # エラー率10%以上
                logger.warning("エラー率が高いため、データ品質を確認してください")

        logger.info("=== 処理完了 ===")
        return 0

    except KeyboardInterrupt:
        logger.info("処理が中断されました")
        return 1

    except Exception as e:
        logger.error(f"予期しないエラー: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
