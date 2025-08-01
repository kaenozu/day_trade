import argparse
import pandas as pd
import json
import os
import logging
import yfinance as yf # yfinanceをインポート
import datetime # datetimeをインポート
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
            logging.info(f"現在のUSD/JPY為替レート: {latest_rate:.2f}")
            return latest_rate
        else:
            logging.error(f"USD/JPY為替レートのデータ取得に失敗しました。データが空です。")
            return None
    except Exception as e:
        logging.error(f"USD/JPY為替レートの取得中にエラーが発生しました: {e}")
        return None

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_config()

    parser = argparse.ArgumentParser(description="Day Trade Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")

    # run_analysis command
    run_analysis_parser = subparsers.add_parser("run_analysis", help="市場分析を実行")
    run_analysis_parser.add_argument("--strategies_file", type=str, default=None, help="実行する戦略のリストを含むJSONファイルのパス")
    run_analysis_parser.add_argument("--ticker", type=str, required=True, help="直接データを取得する株価シンボル (例: AAPL)")
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
    run_backtest_parser.add_argument("--ticker", type=str, default=None, help="直接データを取得する株価シンボル (例: AAPL)")
    run_backtest_parser.add_argument("--start_date", type=str, default=None, help="取得データの開始日 (YYYY-MM-DD)")
    run_backtest_parser.add_argument("--end_date", type=str, default=None, help="取得データの終了日 (YYYY-MM-DD)")

    # get_current_price_jpy command
    get_current_price_jpy_parser = subparsers.add_parser("get_current_price_jpy", help="現在の株価を日本円で取得")
    get_current_price_jpy_parser.add_argument("--ticker", type=str, required=True, help="株価シンボル (例: AAPL)")

    args = parser.parse_args()

    if args.command == "run_analysis":
        ticker_symbol = args.ticker
        start_date = args.start_date or config.get('fetch_data', {}).get('default_start_date', '2023-01-01')
        end_date = args.end_date or config.get('fetch_data', {}).get('default_end_date', '2023-01-31')

        logging.info(f"銘柄: {ticker_symbol} のデータを {start_date} から {end_date} まで取得中...")
        data = fetch_stock_data(ticker_symbol, start_date, end_date)

        if data.empty:
            logging.error(f"銘柄: {ticker_symbol} のデータ取得に失敗しました。分析を中止します。")
            return

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
                logging.info(f"戦略ファイル {args.strategies_file} から戦略を読み込みました。")
            except FileNotFoundError:
                logging.error(f"戦略ファイル {args.strategies_file} が見つかりませんでした。")
                return
            except json.JSONDecodeError as e:
                logging.error(f"戦略JSONファイルの解析エラー: {e}")
                return
        else:
            default_strategies = config.get('strategies', {}).get('default_strategies', [])
            if default_strategies:
                engine.set_active_strategies(default_strategies)
                logging.info(f"config.yaml からデフォルト戦略 {default_strategies} を読み込みました。")

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

    elif args.command == "fetch_data":
        logging.info(f"銘柄: {args.ticker} のデータを {args.start_date} から {args.end_date} まで取得中...")
        df = fetch_stock_data(args.ticker, args.start_date, args.end_date)
        if not df.empty:
            df.to_csv(args.output_path, index=True)
            logging.info(f"データは {args.output_path} に保存されました。")
        else:
            logging.error(f"銘柄: {args.ticker} のデータ取得に失敗しました。")

    elif args.command == "run_backtest":
        data = None
        ticker_symbol = args.ticker if args.ticker else config.get('fetch_data', {}).get('default_ticker', 'AAPL')
        if args.ticker:
            logging.info(f"銘柄: {ticker_symbol} のデータを {args.start_date or config.get('fetch_data', {}).get('default_start_date')} から {args.end_date or config.get('fetch_data', {}).get('default_end_date')} まで取得中...")
            data = fetch_stock_data(
                args.ticker,
                args.start_date or config.get('fetch_data', {}).get('default_start_date'),
                args.end_date or config.get('fetch_data', {}).get('default_end_date')
            )
            if data.empty:
                logging.error(f"銘柄: {ticker_symbol} のデータ取得に失敗しました。")
                return
            # バックテストのためにインデックスを日付に設定
            data.index.name = 'Date'
            data.index = pd.to_datetime(data.index)
        elif args.data_path:
            logging.info(f"データファイル {args.data_path} と初期資本 {args.initial_capital} でバックテストを実行中...")
            try:
                data = pd.read_csv(args.data_path, index_col='Date', parse_dates=True)
            except FileNotFoundError:
                logging.error(f"データファイル {args.data_path} が見つかりませんでした。")
                return
        else:
            logging.error("run_backtest コマンドには --data_path または --ticker のいずれかを指定する必要があります。")
            return

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
                        logging.warning(f"戦略 '{s_name}' が見つかりませんでした。スキップします。")
            except FileNotFoundError:
                logging.error(f"戦略ファイル {args.strategies_file} が見つかりませんでした。")
                return
            except json.JSONDecodeError as e:
                logging.error(f"戦略JSONファイルの解析エラー: {e}")
                return
        else:
            default_strategies = config.get('strategies', {}).get('default_strategies', [])
            if default_strategies:
                for s_name in default_strategies:
                    if s_name in all_strategies:
                        strategies_to_backtest[s_name] = all_strategies[s_name]
                    else:
                        logging.warning(f"config からの戦略 '{s_name}' が見つかりませんでした。スキップします。")

        if not strategies_to_backtest:
            logging.error("バックテストに有効な戦略が指定されていません。")
            return

        backtest_results = run_backtest(data, strategies_to_backtest, args.initial_capital)
        logging.info(f"バックテスト結果: {backtest_results}")

    elif args.command == "get_current_price_jpy": # 新しいコマンドの処理
        ticker_symbol = args.ticker
        logging.info(f"銘柄: {ticker_symbol} の現在の株価を日本円で取得中...")
        
        # 現在の株価（USD）を取得
        stock_info = yf.Ticker(ticker_symbol)
        current_price_usd = stock_info.history(period="1d")['Close'].iloc[-1] if not stock_info.history(period="1d").empty else None

        if current_price_usd is None:
            logging.error(f"銘柄: {ticker_symbol} の現在の株価（USD）を取得できませんでした。")
            return

        # USD/JPY為替レートを取得
        exchange_rate_usd_jpy = get_usd_jpy_exchange_rate()

        if exchange_rate_usd_jpy is None:
            logging.error("USD/JPY為替レートを取得できませんでした。")
            return

        # 日本円に換算
        current_price_jpy = current_price_usd * exchange_rate_usd_jpy
        logging.info(f"銘柄: {ticker_symbol} の現在の株価: {current_price_usd:.2f} USD ({current_price_jpy:.2f} JPY)")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()