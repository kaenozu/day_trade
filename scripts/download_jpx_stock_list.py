
import io
import logging
import requests
import pandas as pd
from typing import List, Optional
from rich.progress import track

# 親ディレクトリをsys.pathに追加して、srcパッケージを見つけられるようにする
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.day_trade.data.stock_master import stock_master

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# JPXから提供されている東証上場銘柄一覧のURL
JPX_STOCK_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

def download_jpx_stock_list() -> Optional[bytes]:
    """
    JPXのサイトから東証上場銘柄一覧のExcelファイルをダウンロードする。

    Returns:
        Optional[bytes]: ダウンロードしたファイルのコンテンツ。失敗した場合はNone。
    """
    try:
        logging.info(f"Downloading stock list from {JPX_STOCK_LIST_URL}")
        response = requests.get(JPX_STOCK_LIST_URL, timeout=30)
        response.raise_for_status()  # HTTPエラーがあれば例外を発生させる
        logging.info("Download successful.")
        return response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download file: {e}")
        return None

def extract_stock_codes_from_excel(file_content: bytes) -> List[str]:
    """
    ダウンロードしたExcelファイル（バイナリ）から証券コードのリストを抽出する。

    Args:
        file_content (bytes): Excelファイルのコンテンツ。

    Returns:
        List[str]: 証券コードのリスト。
    """
    if not file_content:
        logging.warning("File content is empty, cannot extract codes.")
        return []

    try:
        logging.info("Extracting stock codes from the excel file.")
        # ダウンロードしたコンテンツをファイルのように扱えるようにする
        file_like_object = io.BytesIO(file_content)
        
        # Excelファイルを読み込む
        # ヘッダーが1行目にあると仮定し、'コード'列を抽出
        df = pd.read_excel(file_like_object, sheet_name=0, header=0)

        # 'コード'列が存在するか確認
        if 'コード' not in df.columns:
            logging.error("Column 'コード' not found in the Excel file.")
            return []

        # 'コード'列から値を取得し、NaNを除外してから文字列に変換
        codes = df['コード'].dropna().astype(str).tolist()
        
        logging.info(f"Successfully extracted {len(codes)} stock codes.")
        return codes
    except Exception as e:
        logging.error(f"Failed to process Excel file: {e}")
        return []

def update_stock_master(stock_codes: List[str]):
    """
    取得した証券コードを使って銘柄マスタを更新する。

    Args:
        stock_codes (List[str]): 証券コードのリスト。
    """
    logging.info("Updating stock master...")
    # trackを使用して進捗を可視化
    for code in track(stock_codes, description="Processing..."):
        try:
            stock_master.fetch_and_update_stock_info(code)
        except Exception as e:
            logging.error(f"Failed to update stock {code}: {e}")
    logging.info("Stock master update finished.")

def main():
    """
    メイン処理
    """
    file_content = download_jpx_stock_list()
    if file_content:
        stock_codes = extract_stock_codes_from_excel(file_content)
        if stock_codes:
            print(f"取得した証券コード数: {len(stock_codes)}")
            update_stock_master(stock_codes)
        else:
            print("証券コードの抽出に失敗しました。")
    else:
        print("ファイルのダウンロードに失敗しました。")

if __name__ == "__main__":
    main()
