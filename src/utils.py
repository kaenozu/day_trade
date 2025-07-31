import os
import yfinance as yf

def load_tickers(file_path="tickers.txt"):
    """
    指定されたファイルから銘柄ティッカーシンボルを読み込む。
    各行の先頭と末尾の空白を削除し、空行は無視する。
    """
    tickers = []
    if not os.path.exists(file_path):
        print(f"Error: Tickers file not found at {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ticker = line.strip()
            if ticker:  # 空行を無視
                tickers.append(ticker)
    return tickers

def get_stock_data(ticker, period="1d", interval="1m"):
    """
    指定された銘柄の株価データをyfinanceから取得する。
    period: 取得期間 (例: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
    interval: データの間隔 (例: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # テスト用のtickers.txtを作成 (もし存在しない場合)
    if not os.path.exists("tickers.txt"):
        with open("tickers.txt", "w", encoding="utf-8") as f:
            f.write("9984.T\n")
            f.write("7203.T\n")
            f.write("6758.T\n")

    # load_tickers 関数をテスト
    loaded_tickers = load_tickers()
    print("Loaded Tickers:", loaded_tickers)

    # 存在しないファイルをテスト
    non_existent_tickers = load_tickers("non_existent.txt")
    print("Non-existent Tickers:", non_existent_tickers)

    # get_stock_data 関数をテスト
    if loaded_tickers:
        sample_ticker = loaded_tickers[0]
        print(f"\nFetching data for {sample_ticker}...")
        stock_data = get_stock_data(sample_ticker, period="1d", interval="1m")
        if stock_data is not None:
            print(stock_data.head())
        else:
            print(f"Failed to fetch data for {sample_ticker}")
    else:
        print("\nNo tickers loaded to test get_stock_data.")
