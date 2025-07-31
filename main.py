import argparse
import pandas as pd
import json
import os
from src.engine import AnalysisEngine, moving_average_cross_strategy, rsi_strategy, volume_analysis_strategy, bollinger_bands_strategy, atr_strategy, sentiment_analysis_strategy, vwap_strategy
from src.data_fetcher import fetch_stock_data
from src.backtester import run_backtest
from src.config_manager import load_config

def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Day Trade Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run_analysis command
    run_analysis_parser = subparsers.add_parser("run_analysis", help="Run market analysis")
    run_analysis_parser.add_argument("--data_path", type=str, default=None, help="Path to the stock data CSV file")
    run_analysis_parser.add_argument("--strategies_file", type=str, default=None, help="Path to a JSON file containing a list of strategies to run")
    run_analysis_parser.add_argument("--ticker", type=str, default=None, help="Stock ticker symbol (e.g., AAPL) to fetch data directly")
    run_analysis_parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYY-MM-DD) for fetched data")
    run_analysis_parser.add_argument("--end_date", type=str, default=None, help="End date (YYYY-MM-DD) for fetched data")

    # fetch_data command
    fetch_data_parser = subparsers.add_parser("fetch_data", help="Fetch stock data from Yahoo Finance")
    fetch_data_parser.add_argument("--ticker", type=str, default=config.get('fetch_data', {}).get('default_ticker', 'AAPL'), help="Stock ticker symbol (e.g., AAPL)")
    fetch_data_parser.add_argument("--start_date", type=str, default=config.get('fetch_data', {}).get('default_start_date', '2023-01-01'), help="Start date (YYYY-MM-DD)")
    fetch_data_parser.add_argument("--end_date", type=str, default=config.get('fetch_data', {}).get('default_end_date', '2023-01-31'), help="End date (YYYY-MM-DD)")
    fetch_data_parser.add_argument("--output_path", type=str, default=config.get('fetch_data', {}).get('default_output_path', 'stock_data.csv'), help="Output CSV file path")

    # run_backtest command
    run_backtest_parser = subparsers.add_parser("run_backtest", help="Run backtest for strategies")
    run_backtest_parser.add_argument("--data_path", type=str, default=None, help="Path to the stock data CSV file for backtesting")
    run_backtest_parser.add_argument("--strategies_file", type=str, default=None, help="Path to a JSON file containing a list of strategies to backtest")
    run_backtest_parser.add_argument("--initial_capital", type=float, default=config.get('backtest', {}).get('initial_capital', 10000.0), help="Initial capital for backtesting")
    run_backtest_parser.add_argument("--ticker", type=str, default=None, help="Stock ticker symbol (e.g., AAPL) to fetch data directly")
    run_backtest_parser.add_argument("--start_date", type=str, default=None, help="Start date (YYYY-MM-DD) for fetched data")
    run_backtest_parser.add_argument("--end_date", type=str, default=None, help="End date (YYYY-MM-DD) for fetched data")

    args = parser.parse_args()

    if args.command == "run_analysis":
        data = None
        if args.ticker:
            print(f"Fetching data for {args.ticker} from {args.start_date or config.get('fetch_data', {}).get('default_start_date')} to {args.end_date or config.get('fetch_data', {}).get('default_end_date')}")
            data = fetch_stock_data(
                args.ticker,
                args.start_date or config.get('fetch_data', {}).get('default_start_date'),
                args.end_date or config.get('fetch_data', {}).get('default_end_date')
            )
            if data.empty:
                print(f"Error: Failed to fetch data for {args.ticker}")
                return
        elif args.data_path:
            print(f"Running analysis with data from {args.data_path}")
            try:
                data = pd.read_csv(args.data_path)
            except FileNotFoundError:
                print(f"Error: Data file not found at {args.data_path}")
                return
        else:
            print("Error: Either --data_path or --ticker must be specified for run_analysis.")
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
            except FileNotFoundError:
                print(f"Error: Strategies file not found at {args.strategies_file}")
                return
            except json.JSONDecodeError as e:
                print(f"Error parsing strategies JSON from file: {e}")
                return
        else:
            default_strategies = config.get('strategies', {}).get('default_strategies', [])
            if default_strategies:
                engine.set_active_strategies(default_strategies)

        analysis_results = engine.run_analysis(data)
        print("Analysis Results:", analysis_results)

        final_decision = engine.make_ensemble_decision(analysis_results)
        print("Final Decision:", final_decision)

    elif args.command == "fetch_data":
        print(f"Fetching data for {args.ticker} from {args.start_date} to {args.end_date}")
        df = fetch_stock_data(args.ticker, args.start_date, args.end_date)
        if not df.empty:
            df.to_csv(args.output_path, index=True)
            print(f"Data saved to {args.output_path}")
        else:
            print(f"Failed to fetch data for {args.ticker}")

    elif args.command == "run_backtest":
        data = None
        if args.ticker:
            print(f"Fetching data for {args.ticker} from {args.start_date or config.get('fetch_data', {}).get('default_start_date')} to {args.end_date or config.get('fetch_data', {}).get('default_end_date')}")
            data = fetch_stock_data(
                args.ticker,
                args.start_date or config.get('fetch_data', {}).get('default_start_date'),
                args.end_date or config.get('fetch_data', {}).get('default_end_date')
            )
            if data.empty:
                print(f"Error: Failed to fetch data for {args.ticker}")
                return
            # バックテストのためにインデックスを日付に設定
            data.index.name = 'Date'
            data.index = pd.to_datetime(data.index)
        elif args.data_path:
            print(f"Running backtest with data from {args.data_path} and initial capital {args.initial_capital}")
            try:
                data = pd.read_csv(args.data_path, index_col='Date', parse_dates=True)
            except FileNotFoundError:
                print(f"Error: Data file not found at {args.data_path}")
                return
        else:
            print("Error: Either --data_path or --ticker must be specified for run_backtest.")
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
                        print(f"Warning: Strategy '{s_name}' not found and will be skipped.")
            except FileNotFoundError:
                print(f"Error: Strategies file not found at {args.strategies_file}")
                return
            except json.JSONDecodeError as e:
                print(f"Error parsing strategies JSON from file: {e}")
                return
        else:
            default_strategies = config.get('strategies', {}).get('default_strategies', [])
            if default_strategies:
                for s_name in default_strategies:
                    if s_name in all_strategies:
                        strategies_to_backtest[s_name] = all_strategies[s_name]
                    else:
                        print(f"Warning: Strategy '{s_name}' from config not found and will be skipped.")

        if not strategies_to_backtest:
            print("Error: No valid strategies specified for backtesting.")
            return

        backtest_results = run_backtest(data, strategies_to_backtest, args.initial_capital)
        print("Backtest Results:", backtest_results)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()