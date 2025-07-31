import pandas as pd

class AnalysisEngine:
    def __init__(self):
        self.strategies = {}
        self.active_strategies = None

    def register_strategy(self, name, strategy_func):
        """
        分析戦略を登録する。
        :param name: 戦略の名前
        :param strategy_func: 戦略を実行する関数 (データフレームを受け取り、結果を返す)
        """
        self.strategies[name] = strategy_func

    def set_active_strategies(self, strategy_names: list):
        """
        実行する戦略をフィルタリングする。
        :param strategy_names: 実行を許可する戦略の名前のリスト
        """
        self.active_strategies = set(strategy_names)

    def run_analysis(self, data: pd.DataFrame):
        """
        登録された全ての分析戦略を実行し、結果を統合する。
        :param data: 分析対象の株価データ (pandas DataFrame)
        :return: 各戦略の結果をまとめた辞書
        """
        results = {}
        for name, strategy_func in self.strategies.items():
            if self.active_strategies and name not in self.active_strategies:
                continue
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
        各戦略のシグナルに重み付けを行い、総合的な判断を下す。
        :param analysis_results: 各戦略の結果をまとめた辞書
        :return: 最終的な推奨 (例: "BUY", "SELL", "HOLD")
        """
        buy_score = 0
        sell_score = 0

        # 各戦略の結果を評価し、スコアを計算
        for strategy_name, result in analysis_results.items():
            if "signal" in result:
                signal = result["signal"]

                if strategy_name == "MA Cross":
                    if signal == "BUY":
                        buy_score += 1.0
                    elif signal == "SELL":
                        sell_score += 1.0
                
                elif strategy_name == "RSI":
                    if signal == "OVERSOLD":
                        buy_score += 1.0
                    elif signal == "OVERBOUGHT":
                        sell_score += 1.0

                elif strategy_name == "Volume Analysis":
                    if signal == "HIGH_VOLUME":
                        buy_score += 0.5 # 出来高急増は買いにやや有利
                    elif signal == "LOW_VOLUME":
                        sell_score += 0.5 # 出来高減少は売りにやや有利

                elif strategy_name == "VWAP":
                    if signal == "BUY":
                        buy_score += 1.0
                    elif signal == "SELL":
                        sell_score += 1.0

                elif strategy_name == "Bollinger Bands":
                    if signal == "BUY":
                        buy_score += 1.0
                    elif signal == "SELL":
                        sell_score += 1.0

                elif strategy_name == "ATR":
                    # ATRはボラティリティを示すため、直接的な売買シグナルではない
                    # ただし、高ボラティリティはリスクと機会の両方を示す
                    pass

                elif strategy_name == "Sentiment Analysis":
                    if signal == "POSITIVE_SENTIMENT":
                        buy_score += 0.7 # センチメントは補助的な要素としてやや低めの重み
                    elif signal == "NEGATIVE_SENTIMENT":
                        sell_score += 0.7 # センチメントは補助的な要素としてやや低めの重み

        if buy_score > sell_score:
            return "BUY"
        elif sell_score > buy_score:
            return "SELL"
        else:
            return "HOLD"

def moving_average_cross_strategy(data: pd.DataFrame, short_window=5, long_window=25):
    """
    移動平均線クロス戦略を実装する。
    :param data: 株価データ (pandas DataFrame, 'Close'カラムが必要)
    :param short_window: 短期移動平均線の期間
    :param long_window: 長期移動平均線の期間
    :return: 戦略の結果 (辞書)
    """
    if 'Close' not in data.columns:
        return {"error": "'Close' column not found in data"}

    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()

    # ゴールデンクロスとデッドクロスの判定
    # ゴールデンクロス: 短期SMAが長期SMAを上抜ける
    # デッドクロス: 短期SMAが長期SMAを下抜ける
    signals = []
    for i in range(1, len(data)):
        if data['SMA_Short'].iloc[i] > data['SMA_Long'].iloc[i] and \
           data['SMA_Short'].iloc[i-1] <= data['SMA_Long'].iloc[i-1]:
            signals.append("BUY")
        elif data['SMA_Short'].iloc[i] < data['SMA_Long'].iloc[i] and \
             data['SMA_Short'].iloc[i-1] >= data['SMA_Long'].iloc[i-1]:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    
    # 最新のシグナルを返す
    latest_signal = signals[-1] if signals else "HOLD"

    return {
        "signal": latest_signal,
        "SMA_Short": data['SMA_Short'].iloc[-1],
        "SMA_Long": data['SMA_Long'].iloc[-1]
    }

def rsi_strategy(data: pd.DataFrame, window=14, overbought_threshold=70, oversold_threshold=30):
    """
    RSI (Relative Strength Index) 戦略を実装する。
    :param data: 株価データ (pandas DataFrame, 'Close'カラムが必要)
    :param window: RSIの計算期間
    :param overbought_threshold: 買われすぎの閾値
    :param oversold_threshold: 売られすぎの閾値
    :return: 戦略の結果 (辞書)
    """
    if 'Close' not in data.columns:
        return {"error": "'Close' column not found in data"}

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    latest_rsi = rsi.iloc[-1]

    signal = "NEUTRAL"
    if latest_rsi > overbought_threshold:
        signal = "OVERBOUGHT"
    elif latest_rsi < oversold_threshold:
        signal = "OVERSOLD"

    return {
        "signal": signal,
        "RSI": latest_rsi
    }

def volume_analysis_strategy(data: pd.DataFrame, window=20, volume_multiplier=1.5):
    """
    出来高分析戦略を実装する。
    :param data: 株価データ (pandas DataFrame, 'Volume'カラムが必要)
    :param window: 平均出来高の計算期間
    :param volume_multiplier: 平均出来高に対する倍率
    :return: 戦略の結果 (辞書)
    """
    if 'Volume' not in data.columns:
        return {"error": "'Volume' column not found in data"}

    data['Volume_MA'] = data['Volume'].rolling(window=window).mean()

    latest_volume = data['Volume'].iloc[-1]
    latest_volume_ma = data['Volume_MA'].iloc[-1]

    signal = "NORMAL_VOLUME"
    if latest_volume > latest_volume_ma * volume_multiplier:
        signal = "HIGH_VOLUME"
    elif latest_volume < latest_volume_ma / volume_multiplier:
        signal = "LOW_VOLUME"

    return {
        "signal": signal,
        "Latest_Volume": latest_volume,
        "Volume_MA": latest_volume_ma
    }

def vwap_strategy(data: pd.DataFrame):
    """
    VWAP (Volume Weighted Average Price) 戦略を実装する。
    :param data: 株価データ (pandas DataFrame, 'Close', 'High', 'Low', 'Volume'カラムが必要)
    :return: 戦略の結果 (辞書)
    """
    if not all(col in data.columns for col in ['Close', 'High', 'Low', 'Volume']):
        return {"error": "Required columns (Close, High, Low, Volume) not found in data"}

    # Typical Price (TP) = (High + Low + Close) / 3
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3

    # Cumulative TP * Volume
    data['TP_Volume'] = data['TP'] * data['Volume']

    # Cumulative Volume
    data['Cumulative_Volume'] = data['Volume'].cumsum()

    # VWAP = Cumulative TP * Volume / Cumulative Volume
    data['VWAP'] = data['TP_Volume'].cumsum() / data['Cumulative_Volume']

    latest_vwap = data['VWAP'].iloc[-1]
    latest_close = data['Close'].iloc[-1]

    signal = "NEUTRAL"
    if latest_close > latest_vwap:
        signal = "BUY"
    elif latest_close < latest_vwap:
        signal = "SELL"

    return {
        "signal": signal,
        "VWAP": latest_vwap,
        "Latest_Close": latest_close
    }

def bollinger_bands_strategy(data: pd.DataFrame, window=20, num_std_dev=2):
    """
    ボリンジャーバンド戦略を実装する。
    :param data: 株価データ (pandas DataFrame, 'Close'カラムが必要)
    :param window: 移動平均線の期間
    :param num_std_dev: 標準偏差の倍率
    :return: 戦略の結果 (辞書)
    """
    if 'Close' not in data.columns:
        return {"error": "'Close' column not found in data"}

    data['SMA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()

    data['Upper_Band'] = data['SMA'] + (data['STD'] * num_std_dev)
    data['Lower_Band'] = data['SMA'] - (data['STD'] * num_std_dev)

    latest_close = data['Close'].iloc[-1]
    latest_sma = data['SMA'].iloc[-1]
    latest_upper_band = data['Upper_Band'].iloc[-1]
    latest_lower_band = data['Lower_Band'].iloc[-1]

    signal = "NEUTRAL"
    if latest_close < latest_lower_band:
        signal = "BUY"
    elif latest_close > latest_upper_band:
        signal = "SELL"

    return {
        "signal": signal,
        "SMA": latest_sma,
        "Upper_Band": latest_upper_band,
        "Lower_Band": latest_lower_band
    }

def atr_strategy(data: pd.DataFrame, window=14):
    """
    ATR (Average True Range) 戦略を実装する。
    :param data: 株価データ (pandas DataFrame, 'High', 'Low', 'Close'カラムが必要)
    :param window: ATRの計算期間
    :return: 戦略の結果 (辞書)
    """
    if not all(col in data.columns for col in ['High', 'Low', 'Close']):
        return {"error": "Required columns (High, Low, Close) not found in data"}

    # True Range (TR)
    data['TR1'] = abs(data['High'] - data['Low'])
    data['TR2'] = abs(data['High'] - data['Close'].shift())
    data['TR3'] = abs(data['Low'] - data['Close'].shift())
    data['TR'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)

    # Average True Range (ATR)
    data['ATR'] = data['TR'].rolling(window=window).mean()

    latest_atr = data['ATR'].iloc[-1]

    # ATRに基づくシグナル (例: ATRが急増したらボラティリティ上昇)
    # ここでは単純に最新のATR値を返す
    signal = "NEUTRAL"
    if latest_atr > data['ATR'].iloc[-window:-1].mean() * 1.2: # 例: 過去の平均より20%高ければ高ボラティリティ
        signal = "HIGH_VOLATILITY"
    elif latest_atr < data['ATR'].iloc[-window:-1].mean() * 0.8: # 例: 過去の平均より20%低ければ低ボラティリティ
        signal = "LOW_VOLATILITY"

    return {
        "signal": signal,
        "ATR": latest_atr
    }

def sentiment_analysis_strategy(data: pd.DataFrame):
    """
    市場センチメント分析戦略を実装する (プレースホルダー)。
    データフレームに 'Sentiment' カラム (例: -1: 悲観, 0: 中立, 1: 楽観) があることを前提とする。
    :param data: 株価データ (pandas DataFrame, 'Sentiment'カラムが必要)
    :return: 戦略の結果 (辞書)
    """
    if 'Sentiment' not in data.columns:
        return {"error": "'Sentiment' column not found in data"}

    latest_sentiment = data['Sentiment'].iloc[-1]

    signal = "NEUTRAL_SENTIMENT"
    if latest_sentiment > 0:
        signal = "POSITIVE_SENTIMENT"
    elif latest_sentiment < 0:
        signal = "NEGATIVE_SENTIMENT"

    return {
        "signal": signal,
        "Sentiment": latest_sentiment
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
    engine.register_strategy("MA Cross", moving_average_cross_strategy)
    engine.register_strategy("RSI", rsi_strategy)
    engine.register_strategy("Volume Analysis", volume_analysis_strategy)
    engine.register_strategy("VWAP", vwap_strategy)
    engine.register_strategy("Bollinger Bands", bollinger_bands_strategy)
    engine.register_strategy("ATR", atr_strategy)
    engine.register_strategy("Sentiment Analysis", sentiment_analysis_strategy)

    # アクティブな戦略を設定 (例: MA CrossとRSI、Sentiment Analysisのみ実行)
    engine.set_active_strategies(["MA Cross", "RSI", "Sentiment Analysis"])

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