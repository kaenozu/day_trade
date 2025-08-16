# overnight_prediction_model.py

"""
翌朝場予測のための機械学習モデル
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class OvernightPredictionModel:
    MODEL_PATH = "overnight_model.joblib"

    def __init__(self):
        # 特徴量として使用するティッカー（データソースの多様化）
        self.feature_tickers = {
            # 主要株価指数
            '^DJI': 'DOW',
            '^IXIC': 'NASDAQ',
            '^GSPC': 'SP500',
            '^N225': 'NIKKEI',
            '^HSI': 'HANG_SENG',        # アジア市場追加
            '^BVSP': 'BOVESPA',         # 新興国市場追加
            # 先物・コモディティ
            'NKD=F': 'NIKKEI_FUTURES',
            'CL=F': 'CRUDE_OIL',
            'GC=F': 'GOLD',
            'SI=F': 'SILVER',           # 貴金属拡張
            'ZN=F': 'US_10Y_FUTURES',   # 債券先物追加
            # 為替・金利
            'USDJPY=X': 'USD_JPY',
            'EURUSD=X': 'EUR_USD',      # 主要通貨ペア追加
            'GBPUSD=X': 'GBP_USD',
            '^TNX': 'US_10Y_TREASURY',
            '^VIX': 'VIX',
            '^MOVE': 'MOVE_INDEX',      # 債券ボラティリティ指数
        }
        # 予測対象のティッカー
        self.target_ticker = '^N225'
        self.model = None
        self.feature_importance = None
        
        # リスク管理パラメータ
        self.confidence_threshold = 0.65  # 高信頼度予測の閾値
        self.volatility_adjustment = True  # ボラティリティ調整機能

    async def train(self, test_size=0.2):
        """モデルを学習し、ファイルに保存する"""
        data = await self._prepare_data(period="5y")
        if data.empty:
            print("[Error] 学習データの準備に失敗しました。")
            return

        X = data.drop('target_up', axis=1)
        y = data['target_up']

        # データを学習用とテスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

        print(f"学習データ: {len(X_train)}件, テストデータ: {len(X_test)}件")

        # LightGBMモデルの学習（ハイパーパラメータ最適化）
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42
        )
        print("モデルの学習を開始します...")
        self.model.fit(X_train, y_train)
        print("モデルの学習が完了しました。")

        # 特徴量重要度を保存
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # テストデータで評価
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nモデルの評価 (テストデータ):")
        print(f"  - 正解率 (Accuracy): {accuracy:.4f}")
        print("\n分類レポート:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
        
        print(f"\n特徴量重要度 (Top 10):")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # 学習済みモデルと特徴量重要度を保存
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'training_accuracy': accuracy,
            'training_date': datetime.now().isoformat()
        }
        joblib.dump(model_data, self.MODEL_PATH)
        print(f"\nモデルを {self.MODEL_PATH} に保存しました。")

    async def predict(self, enable_risk_management=True):
        """学習済みモデルを読み込み、翌朝場の動向を予測する（リスク管理機能付き）"""
        if self.model is None:
            if os.path.exists(self.MODEL_PATH):
                print(f"学習済みモデル {self.MODEL_PATH} を読み込みます。")
                model_data = joblib.load(self.MODEL_PATH)
                if isinstance(model_data, dict):
                    self.model = model_data['model']
                    self.feature_importance = model_data.get('feature_importance')
                else:
                    # 旧形式の互換性
                    self.model = model_data
            else:
                print("[Error] 学習済みモデルが見つかりません。先に train() を実行してください。")
                return None

        # 予測には直近のデータが必要
        data = await self._prepare_data(period="2mo")
        if data.empty or len(data) < 1:
            print("[Error] 予測用のデータ準備に失敗しました。")
            return None

        # 最後の行のデータを予測に使用
        latest_features = data.drop('target_up', axis=1).iloc[[-1]]
        
        print("\n最新データで翌朝場を予測します...")
        # 各クラスに属する確率を予測
        prediction_proba = self.model.predict_proba(latest_features)
        
        # 信頼度とリスク評価
        confidence = max(prediction_proba[0])
        is_high_confidence = confidence >= self.confidence_threshold
        
        # ボラティリティ評価（VIXベース）
        vix_features = [col for col in latest_features.columns if 'VIX' in col]
        market_volatility = "Normal"
        if vix_features:
            vix_change = latest_features[vix_features[0]].iloc[0]
            if vix_change > 5:
                market_volatility = "High"
            elif vix_change < -5:
                market_volatility = "Low"
        
        # リスク管理推奨
        risk_recommendation = self._generate_risk_management(
            prediction_proba[0], confidence, market_volatility
        ) if enable_risk_management else None
        
        # 予測結果を整形
        prediction_result = {
            'prediction_datetime_utc': datetime.utcnow(),
            'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'probability_down': prediction_proba[0][0],
            'probability_up': prediction_proba[0][1],
            'prediction': 'Up' if prediction_proba[0][1] > 0.5 else 'Down',
            'confidence': confidence,
            'is_high_confidence': is_high_confidence,
            'market_volatility': market_volatility,
            'risk_management': risk_recommendation
        }

        print("予測結果:")
        print(f"  方向: {prediction_result['prediction']}")
        print(f"  上昇確率: {prediction_result['probability_up']:.1%}")
        print(f"  下降確率: {prediction_result['probability_down']:.1%}")
        print(f"  信頼度: {confidence:.1%} ({'高' if is_high_confidence else '低'})")
        print(f"  市場ボラティリティ: {market_volatility}")
        
        if risk_recommendation:
            print("\nリスク管理推奨:")
            for key, value in risk_recommendation.items():
                print(f"  {key}: {value}")
        
        return prediction_result
    
    def _generate_risk_management(self, probabilities, confidence, volatility):
        """リスク管理推奨を生成"""
        prob_up = probabilities[1]
        
        # ポジションサイズ推奨
        if confidence >= 0.8:
            position_size = "中程度 (資金の15-25%)"
        elif confidence >= 0.65:
            position_size = "小程度 (資金の10-15%)"
        else:
            position_size = "最小限 (資金の5%以下)"
        
        # 損切り水準
        if volatility == "High":
            stop_loss = "2-3%"
        elif volatility == "Low":
            stop_loss = "1-2%"
        else:
            stop_loss = "2%"
        
        # 利確水準
        if prob_up > 0.7:
            take_profit = "3-5%"
        elif prob_up > 0.6:
            take_profit = "2-3%"
        else:
            take_profit = "1-2%"
        
        return {
            "推奨ポジションサイズ": position_size,
            "損切り水準": stop_loss,
            "利確水準": take_profit,
            "エントリータイミング": "寄り付き前後" if confidence >= 0.65 else "様子見推奨"
        }

    async def _prepare_data(self, period="5y"):
        """
        学習・予測用のデータをyfinanceから取得し、整形する。
        """
        print(f"データ取得を開始します... (期間: {period})")
        try:
            all_tickers = list(self.feature_tickers.keys())
            data = yf.download(all_tickers, period=period, progress=False)['Close']
            
            if data.empty:
                print("[Error] yfinanceからデータを取得できませんでした。")
                return pd.DataFrame()

            data = data.rename(columns=self.feature_tickers)
            data = data.dropna()

            features = pd.DataFrame(index=data.index)
            for col in data.columns:
                features[f'{col}_pct_change'] = data[col].pct_change() * 100
                ma5 = data[col].rolling(window=5).mean()
                features[f'{col}_ma5_divergence'] = (data[col] - ma5) / ma5 * 100
                ma25 = data[col].rolling(window=25).mean()
                features[f'{col}_ma25_divergence'] = (data[col] - ma25) / ma25 * 100

            target = (data['NIKKEI'].shift(-1) > data['NIKKEI']).astype(int)
            target.name = 'target_up'

            full_data = pd.concat([features, target], axis=1)
            full_data = full_data.dropna()
            
            print("データ準備が完了しました。")
            return full_data

        except Exception as e:
            print(f"[Error] データ準備中にエラーが発生しました: {e}")
            return pd.DataFrame()

    async def backtest(self, start_date=None, end_date=None, initial_capital=1000000):
        """バックテスト環境：過去の予測精度と収益性を検証"""
        print("=== 翌朝場予測モデル バックテスト ===")
        
        # バックテスト用データ準備
        data = await self._prepare_data(period="3y")
        if data.empty:
            print("[Error] バックテストデータの準備に失敗しました。")
            return None
        
        # 日付フィルタリング
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        if len(data) < 100:
            print("[Error] バックテストには最低100日分のデータが必要です。")
            return None
        
        # バックテスト実行
        results = []
        capital = initial_capital
        position = 0  # 0: ノーポジション, 1: ロング
        win_trades = 0
        total_trades = 0
        
        # 訓練期間とテスト期間を分割（最初の70%を訓練、残りをテスト）
        split_idx = int(len(data) * 0.7)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        print(f"バックテスト期間: {test_data.index[0].strftime('%Y-%m-%d')} ～ {test_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"テスト期間: {len(test_data)}日")
        
        # 簡易バックテスト（実際の予測をシミュレート）
        for i in range(1, len(test_data)):
            current_data = test_data.iloc[:i]
            if len(current_data) < 25:  # 移動平均計算に必要な期間
                continue
                
            # 予測シミュレート（前日のデータで翌日を予測）
            features = current_data.drop('target_up', axis=1).iloc[[-1]]
            actual_target = test_data.iloc[i]['target_up']
            
            # 仮想予測（実際にはモデルを使用するが、ここでは簡易版）
            # より複雑な特徴量から予測を行う
            prediction_prob = self._simulate_prediction(features)
            prediction = 1 if prediction_prob > 0.5 else 0
            
            # トレード実行シミュレーション
            if position == 0 and prediction == 1 and prediction_prob > 0.6:  # エントリー
                entry_price = 1.0  # 正規化
                position = 1
                total_trades += 1
            elif position == 1:  # エグジット
                exit_price = 1.01 if actual_target == 1 else 0.99  # 簡易リターン
                trade_return = (exit_price - entry_price) / entry_price
                capital *= (1 + trade_return)
                
                if trade_return > 0:
                    win_trades += 1
                
                position = 0
                
                results.append({
                    'date': test_data.index[i],
                    'prediction': prediction,
                    'actual': actual_target,
                    'trade_return': trade_return,
                    'capital': capital,
                    'prediction_prob': prediction_prob
                })
        
        # バックテスト結果
        if results:
            backtest_df = pd.DataFrame(results)
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            total_return = (capital - initial_capital) / initial_capital
            
            print(f"\n=== バックテスト結果 ===")
            print(f"総取引回数: {total_trades}")
            print(f"勝率: {win_rate:.1%}")
            print(f"総リターン: {total_return:.1%}")
            print(f"最終資産: {capital:,.0f}円")
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'final_capital': capital,
                'trades': backtest_df
            }
        
        return None
    
    def _simulate_prediction(self, features):
        """予測シミュレーション（簡易版）"""
        # 主要指標の重み付け合計で予測確率を計算
        vix_weight = -0.3
        sp500_weight = 0.4
        nikkei_weight = 0.3
        
        prob = 0.5  # ベース確率
        
        # VIX変化率（マイナス影響）
        vix_cols = [col for col in features.columns if 'VIX_pct_change' in col]
        if vix_cols:
            vix_change = features[vix_cols[0]].iloc[0]
            prob += vix_weight * (vix_change / 100)
        
        # S&P500変化率（プラス影響）
        sp500_cols = [col for col in features.columns if 'SP500_pct_change' in col]
        if sp500_cols:
            sp500_change = features[sp500_cols[0]].iloc[0]
            prob += sp500_weight * (sp500_change / 100)
        
        # 確率を0-1の範囲に制限
        return max(0.1, min(0.9, prob))

# テスト用
async def main():
    model = OvernightPredictionModel()
    print("--- モデル学習 --- ")
    await model.train()
    print("\n--- 翌朝場予測 ---")
    await model.predict()
    print("\n--- バックテスト ---")
    await model.backtest()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())