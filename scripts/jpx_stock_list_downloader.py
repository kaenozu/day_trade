#!/usr/bin/env python3
"""
JPX東証上場銘柄一覧ダウンロードスクリプト

Issue #122: 銘柄を一括で追加する機能の実装
- JPXから東証上場銘柄一覧（Excel形式）をダウンロード
- 証券コードを抽出してCSV形式で保存
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.day_trade.utils.logging_config import setup_logging  # noqa: E402

# ロギング設定
setup_logging()
logger = logging.getLogger(__name__)


class JPXStockListDownloader:
    """JPX東証上場銘柄一覧ダウンローダー"""

    def __init__(self):
        # JPXの東証上場銘柄一覧ExcelファイルのURL
        self.jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        self.output_dir = project_root / "data" / "stock_lists"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_jpx_stock_list(self, save_path: Optional[Path] = None) -> Path:
        """
        JPXから東証上場銘柄一覧をダウンロード

        Args:
            save_path: 保存先パス（None の場合は自動生成）

        Returns:
            ダウンロードしたファイルのパス
        """
        if save_path is None:
            save_path = self.output_dir / "jpx_stock_list.xls"

        logger.info(f"JPX上場銘柄一覧をダウンロード中: {self.jpx_url}")

        try:
            # HTTPヘッダーを設定（JPXサイトのアクセス制限対応）
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = requests.get(self.jpx_url, headers=headers, timeout=30)
            response.raise_for_status()

            # ファイルに保存
            with open(save_path, "wb") as f:
                f.write(response.content)

            logger.info(f"ダウンロード完了: {save_path}")
            logger.info(f"ファイルサイズ: {save_path.stat().st_size:,} bytes")

            return save_path

        except requests.RequestException as e:
            logger.error(f"ダウンロードエラー: {e}")
            raise
        except Exception as e:
            logger.error(f"ファイル保存エラー: {e}")
            raise

    def parse_stock_codes(self, excel_path: Path) -> List[str]:
        """
        Excelファイルから証券コードを抽出

        Args:
            excel_path: Excelファイルのパス

        Returns:
            証券コードのリスト
        """
        logger.info(f"証券コード抽出開始: {excel_path}")

        try:
            # Excelファイルを読み込み（複数シートの可能性を考慮）
            excel_file = pd.ExcelFile(excel_path)
            logger.info(f"シート名: {excel_file.sheet_names}")

            # 最初のシートを読み込み
            df = pd.read_excel(excel_file, sheet_name=0)
            logger.info(f"データ形状: {df.shape}")
            logger.info(f"列名: {list(df.columns)}")

            # 最初の数行を表示してデータ構造を確認
            logger.info("データサンプル:")
            logger.info(f"\n{df.head()}")

            # 証券コードらしき列を探す
            stock_codes = []

            # 一般的な証券コード列名のパターン
            code_column_patterns = [
                "コード",
                "code",
                "証券コード",
                "銘柄コード",
                "Code",
                "Stock Code",
                "証券コード",
                "ticker",
            ]

            code_column = None
            for col in df.columns:
                col_str = str(col)
                if any(pattern in col_str for pattern in code_column_patterns):
                    code_column = col
                    break

            if code_column is None:
                # 最初の列を証券コードと仮定
                code_column = df.columns[0]
                logger.warning(
                    f"証券コード列が特定できないため、最初の列を使用: {code_column}"
                )
            else:
                logger.info(f"証券コード列を特定: {code_column}")

            # 証券コードを抽出
            codes = df[code_column].dropna().astype(str)

            # 4桁の数字のみを抽出（日本の証券コード形式）
            for code in codes:
                code_clean = str(code).strip()
                if code_clean.isdigit() and len(code_clean) == 4:
                    stock_codes.append(code_clean)

            logger.info(f"抽出した証券コード数: {len(stock_codes)}")
            logger.info(f"サンプル証券コード: {stock_codes[:10]}")

            return stock_codes

        except Exception as e:
            logger.error(f"証券コード抽出エラー: {e}")
            raise

    def save_stock_codes_csv(
        self, stock_codes: List[str], csv_path: Optional[Path] = None
    ) -> Path:
        """
        証券コードをCSVファイルに保存

        Args:
            stock_codes: 証券コードのリスト
            csv_path: CSVファイルのパス（None の場合は自動生成）

        Returns:
            保存したCSVファイルのパス
        """
        if csv_path is None:
            csv_path = self.output_dir / "jpx_stock_codes.csv"

        logger.info(f"証券コードをCSV保存: {csv_path}")

        try:
            # DataFrameを作成
            df = pd.DataFrame(
                {
                    "stock_code": stock_codes,
                    "source": "JPX",
                    "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                }
            )

            # CSV保存
            df.to_csv(csv_path, index=False, encoding="utf-8")

            logger.info(f"CSV保存完了: {len(stock_codes)}件の証券コード")

            return csv_path

        except Exception as e:
            logger.error(f"CSV保存エラー: {e}")
            raise

    def process_jpx_stock_list(self) -> tuple[Path, List[str]]:
        """
        JPX上場銘柄一覧の完全な処理

        Returns:
            (CSVファイルパス, 証券コードリスト)
        """
        logger.info("=== JPX東証上場銘柄一覧処理開始 ===")

        try:
            # 1. Excelファイルダウンロード
            excel_path = self.download_jpx_stock_list()

            # 2. 証券コード抽出
            stock_codes = self.parse_stock_codes(excel_path)

            # 3. CSV保存
            csv_path = self.save_stock_codes_csv(stock_codes)

            logger.info("=== 処理完了 ===")
            logger.info(f"抽出銘柄数: {len(stock_codes)}")
            logger.info(f"CSV出力: {csv_path}")

            return csv_path, stock_codes

        except Exception as e:
            logger.error(f"処理エラー: {e}")
            raise


def main():
    """メイン実行関数"""
    try:
        downloader = JPXStockListDownloader()
        csv_path, stock_codes = downloader.process_jpx_stock_list()

        print("✅ 処理完了")
        print(f"📁 CSV出力: {csv_path}")
        print(f"📊 抽出銘柄数: {len(stock_codes)}")
        print(f"📋 サンプル証券コード: {stock_codes[:10]}")

    except Exception as e:
        logger.error(f"メイン処理エラー: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
