import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    指定されたティッカーシンボルの株価データを取得する。

    :param ticker: 株価データを取得するティッカーシンボル (例: "AAPL")
    :param start_date: データ取得開始日 (YYYY-MM-DD形式)
    :param end_date: データ取得終了日 (YYYY-MM-DD形式)
    :return: 取得した株価データ (pandas DataFrame)
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}")
            return pd.DataFrame()
        # カラム名をCLIインターフェースの要件に合わせる
        data.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Adj Close': 'Adj Close'
        }, inplace=True)
        # Sentimentカラムを追加 (ダミー値)
        data['Sentiment'] = 0 # 初期値として0を設定
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # テスト実行
    ticker = "AAPL"
    start = "2023-01-01"
    end = "2023-01-31"
    df = fetch_stock_data(ticker, start, end)
    if not df.empty:
        print(f"Successfully fetched data for {ticker}:")
        print(df.head())
        print(df.tail())
    else:
        print(f"Failed to fetch data for {ticker}")
