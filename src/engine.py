import pandas as pd

class AnalysisEngine:
    def __init__(self):
        self.strategies = {}

    def register_strategy(self, name, strategy_func):
        """
        分析戦略を登録する。
        :param name: 戦略の名前
        :param strategy_func: 戦略を実行する関数 (データフレームを受け取り、結果を返す)
        """
        self.strategies[name] = strategy_func

    def run_analysis(self, data: pd.DataFrame):
        """
        登録された全ての分析戦略を実行し、結果を統合する。
        :param data: 分析対象の株価データ (pandas DataFrame)
        :return: 各戦略の結果をまとめた辞書
        """
        results = {}
        for name, strategy_func in self.strategies.items():
            try:
                strategy_result = strategy_func(data)
                results[name] = strategy_result
            except Exception as e:
                print(f"Error running strategy {name}: {e}")
                results[name] = {"error": str(e)}
        return results

def make_ensemble_decision(self, analysis_results):
        """
        複数の分析戦略の結果を統合し、最終的な推奨を決定する。
        :param analysis_results: 各戦略の結果をまとめた辞書
        :return: 最終的な推奨 (例: "BUY", "SELL", "HOLD")
        """
        buy_signals = 0
        sell_signals = 0

        # 例: 各戦略の結果を評価し、シグナルをカウント
        for strategy_name, result in analysis_results.items():
            if "average_close" in result and result["average_close"] > 103: # 例示的な条件
                buy_signals += 1
            if "total_volume" in result and result["total_volume"] < 6000: # 例示的な条件
                sell_signals += 1

        if buy_signals > sell_signals:
            return "BUY"
        elif sell_signals > buy_signals:
            return "SELL"
        else:
            return "HOLD"

if __name__ == "__main__":
    # テスト用のダミーデータ
    dummy_data = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 106, 107, 108, 109],
        'Low': [99, 100, 101, 102, 103],
        'Close': [102, 103, 104, 105, 106],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    })

    # ダミーの分析戦略
    def sample_strategy_1(data):
        # 例: 終値の平均を計算
        return {"average_close": data['Close'].mean()}

    def sample_strategy_2(data):
        # 例: ボリュームの合計を計算
        return {"total_volume": data['Volume'].sum()}

    # エンジンを初期化
    engine = AnalysisEngine()

    # 戦略を登録
    engine.register_strategy("Strategy A", sample_strategy_1)
    engine.register_strategy("Strategy B", sample_strategy_2)

    # 分析を実行
    analysis_results = engine.run_analysis(dummy_data)
    print("Analysis Results:", analysis_results)

    # アンサンブル判定を実行
    final_decision = engine.make_ensemble_decision(analysis_results)
    print("Final Decision:", final_decision)

    # エラーを発生させる戦略のテスト
    def error_strategy(data):
        raise ValueError("This strategy always fails!")

    engine.register_strategy("Error Strategy", error_strategy)
    error_results = engine.run_analysis(dummy_data)
    print("Error Test Results:", error_results)

    # エラーを含む結果でのアンサンブル判定
    error_decision = engine.make_ensemble_decision(error_results)
    print("Decision with Errors:", error_decision)
