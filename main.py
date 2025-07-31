# main.py

import argparse
import pandas as pd
import json
import os
from src.engine import AnalysisEngine, moving_average_cross_strategy, rsi_strategy, volume_analysis_strategy, bollinger_bands_strategy, atr_strategy, sentiment_analysis_strategy, vwap_strategy

def main():
    parser = argparse.ArgumentParser(description="Day Trade Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run_analysis command
    run_analysis_parser = subparsers.add_parser("run_analysis", help="Run market analysis")
    run_analysis_parser.add_argument("--data_path", type=str, required=True, help="Path to the stock data CSV file")
    run_analysis_parser.add_argument("--strategies_file", type=str, default=None, help="Path to a JSON file containing a list of strategies to run")

    args = parser.parse_args()

    if args.command == "run_analysis":
        print(f"Running analysis with data from {args.data_path}")
        # Dummy data for now, replace with actual data loading
        try:
            dummy_data = pd.read_csv(args.data_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {args.data_path}")
            return

        engine = AnalysisEngine()

        # Register all available strategies
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

        analysis_results = engine.run_analysis(dummy_data)
        print("Analysis Results:", analysis_results)

        final_decision = engine.make_ensemble_decision(analysis_results)
        print("Final Decision:", final_decision)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
