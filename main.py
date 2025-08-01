import argparse
import pandas as pd
import json
import os
import logging
import yfinance as yf
import datetime
from src.engine import AnalysisEngine, moving_average_cross_strategy, rsi_strategy, volume_analysis_strategy, bollinger_bands_strategy, atr_strategy, sentiment_analysis_strategy, vwap_strategy
from src.data_fetcher import fetch_stock_data
from src.backtester import run_backtest
from src.config_manager import load_config

# USDからJPYへの為替レートを取得する関数
def get_usd_jpy_exchange_rate():
    ticker_symbol = "USDJPY=X"
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    try:
        data = yf.download(ticker_symbol, start=yesterday, end=today, interval="1d", progress=False)
        if not data.empty:
            latest_rate = data['Close'].iloc[-1].item()
            logging.debug(f"現在のUSD/JPY為替レート: {latest_rate:.2f}")
            return latest_rate
        else:
            logging.error(f"USD/JPY為替レートのデータ取得に失敗しました。データが空です。")
            return None
    except Exception as e:
        logging.error(f"USD/JPY為替レートの取得中にエラーが発生しました: {e}")
        return None

def main():
    # ロギング設定を引数で制御できるように変更
    parser = argparse.ArgumentParser(description="Day Trade Analysis CLI")
    parser.add_argument("--log_level", type=str, default="INFO", # デフォルトをINFOに変更
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="ログレベルを設定 (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # サブパーサーを定義する前に、一旦引数をパースしてログレベルを取得
    temp_args, remaining_args = parser.parse_known_args()
    
    numeric_level = getattr(logging, temp_args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {temp_args.log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_config()

    # サブパーサーの定義
    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")

    # run_analysis command
    run_analysis_parser = subparsers.add_parser("run_analysis", help="市場分析を実行")
    run_analysis_parser.add_argument("--strategies_file", type=str, default=None, help="実行する戦略のリストを含むJSONファイルのパス")
    run_analysis_parser.add_argument("--tickers", type=str, default=",".join(config.get('fetch_data', {}).get('default_tickers', ['AAPL'])), help="直接データを取得する株価シンボル (カンマ区切り、例: AAPL,MSFT)")
    run_analysis_parser.add_argument("--start_date", type=str, default=None, help="取得データの開始日 (YYYY-MM-DD)")
    run_analysis_parser.add_argument("--end_date", type=str, default=None, help="取得データの終了日 (YYYY-MM-DD)")

    # fetch_data command
    fetch_data_parser = subparsers.add_parser("fetch_data", help="Yahoo Financeから株価データを取得")
    fetch_data_parser.add_argument("--ticker", type=str, default=config.get('fetch_data', {}).get('default_ticker', 'AAPL'), help="株価シンボル (例: AAPL)")
    fetch_data_parser.add_argument("--start_date", type=str, default=config.get('fetch_data', {}).get('default_start_date', '2023-01-01'), help="開始日 (YYYY-MM-DD)")
    fetch_data_parser.add_argument("--end_date", type=str, default=config.get('fetch_data', {}).get('default_end_date', '2023-01-31'), help="終了日 (YYYY-MM-DD)")
    fetch_data_parser.add_argument("--output_path", type=str, default=config.get('fetch_data', {}).get('default_output_path', 'stock_data.csv'), help="出力CSVファイルのパス")

    # run_backtest command
    run_backtest_parser = subparsers.add_parser("run_backtest", help="戦略のバックテストを実行")
    run_backtest_parser.add_argument("--data_path", type=str, default=None, help="バックテスト用株価データCSVファイルのパス")
    run_backtest_parser.add_argument("--strategies_file", type=str, default=None, help="バックテストする戦略のリストを含むJSONファイルのパス")
    run_backtest_parser.add_argument("--initial_capital", type=float, default=config.get('backtest', {}).get('initial_capital', 10000.0), help="バックテストの初期資本")
    run_backtest_parser.add_argument("--tickers", type=str, default=",".join(config.get('fetch_data', {}).get('default_tickers', ['AAPL'])), help="直接データを取得する株価シンボル (カンマ区切り、例: AAPL,MSFT)")
    run_backtest_parser.add_argument("--start_date", type=str, default=None, help="取得データの開始日 (YYYY-MM-DD)")
    run_backtest_parser.add_argument("--end_date", type=str, default=None, help="取得データの終了日 (YYYY-MM-DD)")

    # get_current_price_jpy command
    get_current_price_jpy_parser = subparsers.add_parser("get_current_price_jpy", help="現在の株価を日本円で取得")
    get_current_price_jpy_parser.add_argument("--ticker", type=str, required=True, help="株価シンボル (例: AAPL)")

    parser.set_defaults(command='run_analysis') # デフォルトコマンドを設定
    args = parser.parse_args(remaining_args) # 残りの引数をパース

    # デフォルトコマンドがrun_analysisの場合、run_analysis_parserのデフォルト引数を設定
    if args.command == 'run_analysis' and not hasattr(args, 'tickers'):
        run_analysis_parser.parse_args([], namespace=args) # run_analysis_parserのデフォルト引数をargsに設定

    if args.command == "run_analysis":
        tickers = args.tickers.split(',')
        start_date = args.start_date or config.get('fetch_data', {}).get('default_start_date', '2023-01-01')
        end_date = args.end_date or config.get('fetch_data', {}).get('default_end_date', '2023-01-31')

        for ticker_symbol in tickers:
            logging.info(f"--- 銘柄: {ticker_symbol} の分析を開始します ---")
            logging.info(f"銘柄: {ticker_symbol} のデータを {start_date} から {end_date} まで取得中...")
            data = fetch_stock_data(ticker_symbol, start_date, end_date)

            if data.empty:
                logging.error(f"銘柄: {ticker_symbol} のデータ取得に失敗しました。分析を中止します。")
                continue # 次の銘柄へ

            engine = AnalysisEngine()

            engine.register_strategy("MA Cross", moving_average_cross_strategy)
            engine.register_strategy("RSI", rsi_strategy)
            engine.register_strategy("Volume Analysis", volume_analysis_strategy)
            engine.register_strategy("VWAP", vwap_strategy)
            engine.register_strategy("Bollinger Bands", bollinger_bands_strategy)
            engine.register_strategy("ATR", atr_strategy)
            engine.register_strategy("Sentiment Analysis", sentiment_analysis_strategy)

            if args.strategies_file:
                try:
                    with open(args.strategies_file, 'r') as f:
                        strategies_json = f.read()
                    parsed_strategies = json.loads(strategies_json)
                    engine.set_active_strategies(parsed_strategies)
                    logging.debug(f"戦略ファイル {args.strategies_file} から戦略を読み込みました。")
                except FileNotFoundError:
                    logging.error(f"戦略ファイル {args.strategies_file} が見つかりませんでした。")
                    continue # 次の銘柄へ
                except json.JSONDecodeError as e:
                    logging.error(f"戦略JSONファイルの解析エラー: {e}")
                    continue # 次の銘柄へ
            else:
                default_strategies = config.get('strategies', {}).get('default_strategies', [])
                if default_strategies:
                    engine.set_active_strategies(default_strategies)
                    logging.debug(f"config.yaml からデフォルト戦略 {default_strategies} を読み込みました。")

            analysis_results = engine.run_analysis(data)
            logging.info(f"分析結果: {analysis_results}")

            final_decision = engine.make_ensemble_decision(analysis_results)
            logging.info(f"最終決定: {final_decision}")

            # 現在の株価を日本円で表示
            logging.info(f"銘柄: {ticker_symbol} の現在の株価を日本円で取得中...")
            stock_info = yf.Ticker(ticker_symbol)
            current_price_usd = stock_info.history(period="1d")['Close'].iloc[-1].item() if not stock_info.history(period="1d").empty else None

            if current_price_usd is None:
                logging.error(f"銘柄: {ticker_symbol} の現在の株価（USD）を取得できませんでした。")
            else:
                exchange_rate_usd_jpy = get_usd_jpy_exchange_rate()
                if exchange_rate_usd_jpy is None:
                    logging.error("USD/JPY為替レートを取得できませんでした。")
                else:
                    current_price_jpy = current_price_usd * exchange_rate_usd_jpy
                    logging.info(f"銘柄: {ticker_symbol} の現在の株価: {current_price_usd:.2f} USD ({current_price_jpy:.2f} JPY)")
            logging.info(f"--- 銘柄: {ticker_symbol} の分析を終了します ---")

    elif args.command == "fetch_data":
        logging.info(f"銘柄: {args.ticker} のデータを {args.start_date} から {args.end_date} まで取得中...")
        df = fetch_stock_data(args.ticker, args.start_date, args.end_date)
        if not df.empty:
            df.to_csv(args.output_path, index=True)
            logging.debug(f"データは {args.output_path} に保存されました。")
        else:
            logging.error(f"銘柄: {args.ticker} のデータ取得に失敗しました。")

    elif args.command == "run_backtest":
        tickers = args.tickers.split(',')
        start_date = config.get('fetch_data', {}).get('default_start_date', '2023-01-01')
        end_date = config.get('fetch_data', {}).get('default_end_date', '2023-01-31')

        for ticker_symbol in tickers:
            logging.info(f"--- 銘柄: {ticker_symbol} のバックテストを開始します ---")
            logging.info(f"銘柄: {ticker_symbol} のデータを {start_date} から {end_date} まで取得中...")
            data = fetch_stock_data(ticker_symbol, start_date, end_date)

            if data.empty:
                logging.error(f"銘柄: {ticker_symbol} のデータ取得に失敗しました。バックテストを中止します。")
                continue # 次の銘柄へ

            # バックテストのためにインデックスを日付に設定
            data.index.name = 'Date'
            data.index = pd.to_datetime(data.index)

            all_strategies = {
                "MA Cross": moving_average_cross_strategy,
                "RSI": rsi_strategy,
                "Volume Analysis": volume_analysis_strategy,
                "VWAP": vwap_strategy,
                "Bollinger Bands": bollinger_bands_strategy,
                "ATR": atr_strategy,
                "Sentiment Analysis": sentiment_analysis_strategy
            }

            strategies_to_backtest = {}
            if args.strategies_file:
                try:
                    with open(args.strategies_file, 'r') as f:
                        strategies_json = f.read()
                    parsed_strategies = json.loads(strategies_json)
                    for s_name in parsed_strategies:
                        if s_name in all_strategies:
                            strategies_to_backtest[s_name] = all_strategies[s_name]
                        else:
                            logging.debug(f"戦略 '{s_name}' が見つかりませんでした。スキップします。")
                except FileNotFoundError:
                    logging.error(f"戦略ファイル {args.strategies_file} が見つかりませんでした。")
                    continue # 次の銘柄へ
                except json.JSONDecodeError as e:
                    logging.error(f"戦略JSONファイルの解析エラー: {e}")
                    continue # 次の銘柄へ
            else:
                default_strategies = config.get('strategies', {}).get('default_strategies', [])
                if default_strategies:
                    for s_name in default_strategies:
                        if s_name in all_strategies:
                            strategies_to_backtest[s_name] = all_strategies[s_name]
                        else:
                            logging.debug(f"config からの戦略 '{s_name}' が見つかりませんでした。スキップします。")

            if not strategies_to_backtest:
                logging.error("バックテストに有効な戦略が指定されていません。")
                continue # 次の銘柄へ

            backtest_results = run_backtest(data, strategies_to_backtest, args.initial_capital)
            logging.info(f"バックテスト結果: {backtest_results}")
            logging.info(f"--- 銘柄: {ticker_symbol} のバックテストを終了します ---")

    elif args.command == "get_current_price_jpy":
        ticker_symbol = args.ticker
        logging.info(f"銘柄: {ticker_symbol} の現在の株価を日本円で取得中...")
        
        stock_info = yf.Ticker(ticker_symbol)
        current_price_usd = stock_info.history(period="1d")['Close'].iloc[-1].item() if not stock_info.history(period="1d").empty else None

        if current_price_usd is None:
            logging.error(f"銘柄: {ticker_symbol} の現在の株価（USD）を取得できませんでした。")
            return

        exchange_rate_usd_jpy = get_usd_jpy_exchange_rate()

        if exchange_rate_usd_jpy is None:
            logging.error("USD/JPY為替レートを取得できませんでした。")
            return

        current_price_jpy = current_price_usd * exchange_rate_usd_jpy
        logging.info(f"銘柄: {ticker_symbol} の現在の株価: {current_price_usd:.2f} USD ({current_price_jpy:.2f} JPY)")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
