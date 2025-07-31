import pandas as pd
from src.engine import AnalysisEngine

def run_backtest(data: pd.DataFrame, strategies: dict, initial_capital: float = 10000.0):
    """
    指定された戦略のリストに対してバックテストを実行する。

    :param data: バックテストに使用する株価データ (pandas DataFrame)
    :param strategies: AnalysisEngineに登録されている戦略の辞書
    :param initial_capital: 初期資金
    :return: バックテスト結果の辞書
    """
    capital = initial_capital
    positions = 0  # 保有株数
    trade_log = []

    engine = AnalysisEngine()
    for name, func in strategies.items():
        engine.register_strategy(name, func)

    # 各行（各時点）で分析を実行
    for i in range(1, len(data)):
        current_data = data.iloc[:i+1] # 現在までのデータを渡す
        analysis_results = engine.run_analysis(current_data)
        decision = engine.make_ensemble_decision(analysis_results)
        current_price = data['Close'].iloc[i]

        if decision == "BUY" and capital > 0 and positions == 0:
            # 全ての資金で株を購入 (簡略化のため)
            buy_shares = capital / current_price
            positions += buy_shares
            capital = 0
            trade_log.append({
                'Date': data.index[i].strftime('%Y-%m-%d'),
                'Type': 'BUY',
                'Price': current_price,
                'Shares': buy_shares,
                'Capital': capital,
                'Positions': positions
            })
            # print(f"BUY: Date={data.index[i]}, Price={current_price}, Shares={buy_shares}")
        elif decision == "SELL" and positions > 0:
            # 全ての株を売却
            capital += positions * current_price
            positions = 0
            trade_log.append({
                'Date': data.index[i].strftime('%Y-%m-%d'),
                'Type': 'SELL',
                'Price': current_price,
                'Shares': 0,
                'Capital': capital,
                'Positions': positions
            })
            # print(f"SELL: Date={data.index[i]}, Price={current_price}, Capital={capital}")

    # 最終的な評価
    final_value = capital + (positions * data['Close'].iloc[-1])
    profit_loss = final_value - initial_capital
    
    # 簡単な評価指標
    num_trades = len(trade_log)
    winning_trades = len([t for t in trade_log if t['Type'] == 'SELL' and t['Capital'] > initial_capital]) # 簡略化
    win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "profit_loss": profit_loss,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "trade_log": trade_log
    }

if __name__ == "__main__":
    # テスト用のダミーデータ
    dummy_data = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
        'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
        'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900],
        'Sentiment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]
    }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=30)))

    # テスト用の戦略辞書
    from src.engine import moving_average_cross_strategy, rsi_strategy, sentiment_analysis_strategy
    test_strategies = {
        "MA Cross": moving_average_cross_strategy,
        "RSI": rsi_strategy,
        "Sentiment Analysis": sentiment_analysis_strategy
    }

    results = run_backtest(dummy_data, test_strategies)
    print("Backtest Results:", results)