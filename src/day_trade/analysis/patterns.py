"""
チャートパターン認識エンジン
価格データからチャートパターンを認識する
"""
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class ChartPatternRecognizer:
    """チャートパターン認識クラス"""
    
    @staticmethod
    def golden_dead_cross(
        df: pd.DataFrame,
        fast_period: int = 5,
        slow_period: int = 20,
        column: str = 'Close'
    ) -> pd.DataFrame:
        """
        ゴールデンクロス・デッドクロスの検出
        
        Args:
            df: 価格データのDataFrame
            fast_period: 短期移動平均の期間
            slow_period: 長期移動平均の期間
            column: 計算対象の列名
            
        Returns:
            クロスポイントを含むDataFrame
        """
        try:
            # 移動平均を計算
            fast_ma = df[column].rolling(window=fast_period).mean()
            slow_ma = df[column].rolling(window=slow_period).mean()
            
            # クロスポイントを検出
            fast_above = fast_ma > slow_ma
            fast_above_shifted = fast_above.shift(1).fillna(False)
            golden_cross = (fast_above & ~fast_above_shifted)
            dead_cross = (~fast_above & fast_above_shifted)
            
            # 信頼度スコアを計算（クロス角度に基づく）
            ma_diff = fast_ma - slow_ma
            ma_diff_change = ma_diff.diff()
            
            golden_confidence = golden_cross * (ma_diff_change.abs() / df[column] * 100)
            dead_confidence = dead_cross * (ma_diff_change.abs() / df[column] * 100)
            
            return pd.DataFrame({
                f'Fast_MA_{fast_period}': fast_ma,
                f'Slow_MA_{slow_period}': slow_ma,
                'Golden_Cross': golden_cross,
                'Dead_Cross': dead_cross,
                'Golden_Confidence': golden_confidence.clip(0, 100),
                'Dead_Confidence': dead_confidence.clip(0, 100)
            })
            
        except Exception as e:
            logger.error(f"ゴールデン・デッドクロス検出エラー: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def support_resistance_levels(
        df: pd.DataFrame,
        window: int = 20,
        num_levels: int = 3,
        column: str = 'Close'
    ) -> Dict[str, List[float]]:
        """
        サポート・レジスタンスラインの検出
        
        Args:
            df: 価格データのDataFrame
            window: 極値検出のウィンドウサイズ
            num_levels: 検出するレベル数
            column: 計算対象の列名
            
        Returns:
            サポート・レジスタンスレベルの辞書
        """
        try:
            prices = df[column].values
            
            # 極大値と極小値を検出
            max_idx = argrelextrema(prices, np.greater, order=window)[0]
            min_idx = argrelextrema(prices, np.less, order=window)[0]
            
            # 極値の価格を取得
            resistance_candidates = prices[max_idx] if len(max_idx) > 0 else []
            support_candidates = prices[min_idx] if len(min_idx) > 0 else []
            
            # クラスタリングして主要なレベルを特定
            def cluster_levels(levels, num_clusters):
                if len(levels) == 0:
                    return []
                
                levels = np.array(levels)
                clustered = []
                
                # K-meansの簡易版
                sorted_levels = np.sort(levels)
                if len(sorted_levels) <= num_clusters:
                    return sorted_levels.tolist()
                
                # 等間隔でクラスタ中心を初期化
                indices = np.linspace(0, len(sorted_levels) - 1, num_clusters, dtype=int)
                centers = sorted_levels[indices]
                
                # 各レベルを最も近いクラスタに割り当て
                for _ in range(10):  # 最大10回反復
                    clusters = [[] for _ in range(num_clusters)]
                    
                    for level in sorted_levels:
                        distances = np.abs(centers - level)
                        nearest = np.argmin(distances)
                        clusters[nearest].append(level)
                    
                    # クラスタ中心を更新
                    new_centers = []
                    for cluster in clusters:
                        if cluster:
                            new_centers.append(np.mean(cluster))
                    
                    if len(new_centers) == num_clusters:
                        centers = np.array(new_centers)
                
                return sorted(centers.tolist(), reverse=True)
            
            resistance_levels = cluster_levels(resistance_candidates, num_levels)
            support_levels = cluster_levels(support_candidates, num_levels)
            
            return {
                'resistance': resistance_levels,
                'support': sorted(support_levels)
            }
            
        except Exception as e:
            logger.error(f"サポート・レジスタンス検出エラー: {e}")
            return {'resistance': [], 'support': []}
    
    @staticmethod
    def breakout_detection(
        df: pd.DataFrame,
        lookback: int = 20,
        threshold: float = 0.02,
        volume_factor: float = 1.5
    ) -> pd.DataFrame:
        """
        ブレイクアウトパターンの検出
        
        Args:
            df: 価格データのDataFrame（Close, Volume列が必要）
            lookback: 過去を見る期間
            threshold: ブレイクアウト閾値（%）
            volume_factor: ボリューム増加係数
            
        Returns:
            ブレイクアウトシグナルを含むDataFrame
        """
        try:
            # ローリング最高値・最安値
            rolling_high = df['High'].rolling(window=lookback).max()
            rolling_low = df['Low'].rolling(window=lookback).min()
            
            # ボリューム移動平均
            volume_ma = df['Volume'].rolling(window=lookback).mean()
            
            # 上方ブレイクアウト
            upward_breakout = (
                (df['Close'] > rolling_high.shift(1)) &
                (df['Volume'] > volume_ma * volume_factor)
            )
            
            # 下方ブレイクアウト
            downward_breakout = (
                (df['Close'] < rolling_low.shift(1)) &
                (df['Volume'] > volume_ma * volume_factor)
            )
            
            # ブレイクアウトの強度を計算
            upward_strength = np.where(
                upward_breakout,
                ((df['Close'] - rolling_high.shift(1)) / rolling_high.shift(1)) * 100,
                0
            )
            
            downward_strength = np.where(
                downward_breakout,
                ((rolling_low.shift(1) - df['Close']) / rolling_low.shift(1)) * 100,
                0
            )
            
            # 信頼度スコア（ブレイクアウトの強度とボリューム増加率に基づく）
            volume_increase = df['Volume'] / volume_ma - 1
            
            upward_confidence = np.where(
                upward_breakout,
                np.minimum(
                    (upward_strength * 10) * (1 + volume_increase.clip(0, 2)),
                    100
                ),
                0
            )
            
            downward_confidence = np.where(
                downward_breakout,
                np.minimum(
                    (downward_strength * 10) * (1 + volume_increase.clip(0, 2)),
                    100
                ),
                0
            )
            
            return pd.DataFrame({
                'Rolling_High': rolling_high,
                'Rolling_Low': rolling_low,
                'Upward_Breakout': upward_breakout,
                'Downward_Breakout': downward_breakout,
                'Upward_Strength': upward_strength,
                'Downward_Strength': downward_strength,
                'Upward_Confidence': upward_confidence,
                'Downward_Confidence': downward_confidence
            })
            
        except Exception as e:
            logger.error(f"ブレイクアウト検出エラー: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def trend_line_detection(
        df: pd.DataFrame,
        window: int = 20,
        min_touches: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """
        トレンドラインの検出
        
        Args:
            df: 価格データのDataFrame
            window: 極値検出のウィンドウサイズ
            min_touches: 最小接触回数
            
        Returns:
            トレンドライン情報の辞書
        """
        try:
            prices_high = df['High'].values
            prices_low = df['Low'].values
            
            # 極大値と極小値を検出
            max_idx = argrelextrema(prices_high, np.greater, order=window)[0]
            min_idx = argrelextrema(prices_low, np.less, order=window)[0]
            
            result = {}
            
            # 上昇トレンドライン（極小値を結ぶ）
            if len(min_idx) >= min_touches:
                X = min_idx.reshape(-1, 1)
                y = prices_low[min_idx]
                
                model = LinearRegression()
                model.fit(X, y)
                
                slope = model.coef_[0]
                intercept = model.intercept_
                r2 = model.score(X, y)
                
                # 最新のトレンドライン値
                current_value = slope * len(df) + intercept
                
                result['support_trend'] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r2': r2,
                    'current_value': current_value,
                    'touches': len(min_idx),
                    'angle': np.degrees(np.arctan(slope / np.mean(prices_low))) 
                }
            
            # 下降トレンドライン（極大値を結ぶ）
            if len(max_idx) >= min_touches:
                X = max_idx.reshape(-1, 1)
                y = prices_high[max_idx]
                
                model = LinearRegression()
                model.fit(X, y)
                
                slope = model.coef_[0]
                intercept = model.intercept_
                r2 = model.score(X, y)
                
                # 最新のトレンドライン値
                current_value = slope * len(df) + intercept
                
                result['resistance_trend'] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r2': r2,
                    'current_value': current_value,
                    'touches': len(max_idx),
                    'angle': np.degrees(np.arctan(slope / np.mean(prices_high)))
                }
            
            return result
            
        except Exception as e:
            logger.error(f"トレンドライン検出エラー: {e}")
            return {}
    
    @staticmethod
    def detect_all_patterns(
        df: pd.DataFrame,
        golden_cross_fast: int = 5,
        golden_cross_slow: int = 20,
        support_resistance_window: int = 20,
        breakout_lookback: int = 20,
        trend_window: int = 20
    ) -> Dict[str, any]:
        """
        全パターンを検出
        
        Args:
            df: 価格データのDataFrame
            各種パラメータ
            
        Returns:
            全パターン検出結果の辞書
        """
        try:
            results = {}
            
            # ゴールデン・デッドクロス
            cross_data = ChartPatternRecognizer.golden_dead_cross(
                df, golden_cross_fast, golden_cross_slow
            )
            results['crosses'] = cross_data
            
            # 最新のクロスシグナル
            if len(cross_data) > 0:
                latest_idx = -1
                if cross_data['Golden_Cross'].iloc[latest_idx]:
                    results['latest_signal'] = {
                        'type': 'Golden Cross',
                        'confidence': cross_data['Golden_Confidence'].iloc[latest_idx]
                    }
                elif cross_data['Dead_Cross'].iloc[latest_idx]:
                    results['latest_signal'] = {
                        'type': 'Dead Cross',
                        'confidence': cross_data['Dead_Confidence'].iloc[latest_idx]
                    }
            
            # サポート・レジスタンス
            levels = ChartPatternRecognizer.support_resistance_levels(
                df, support_resistance_window
            )
            results['levels'] = levels
            
            # ブレイクアウト
            breakout_data = ChartPatternRecognizer.breakout_detection(
                df, breakout_lookback
            )
            results['breakouts'] = breakout_data
            
            # トレンドライン
            trends = ChartPatternRecognizer.trend_line_detection(
                df, trend_window
            )
            results['trends'] = trends
            
            # 総合スコア（各パターンの信頼度を統合）
            confidence_scores = []
            
            # クロスシグナルの信頼度
            if 'latest_signal' in results:
                confidence_scores.append(results['latest_signal']['confidence'])
            
            # ブレイクアウトの信頼度
            if len(breakout_data) > 0:
                latest_up = breakout_data['Upward_Confidence'].iloc[-1]
                latest_down = breakout_data['Downward_Confidence'].iloc[-1]
                if latest_up > 0:
                    confidence_scores.append(latest_up)
                if latest_down > 0:
                    confidence_scores.append(latest_down)
            
            # トレンドの信頼度（R²値を使用）
            for trend_type, trend_info in trends.items():
                if 'r2' in trend_info:
                    confidence_scores.append(trend_info['r2'] * 100)
            
            if confidence_scores:
                results['overall_confidence'] = np.mean(confidence_scores)
            else:
                results['overall_confidence'] = 0
            
            return results
            
        except Exception as e:
            logger.error(f"全パターン検出エラー: {e}")
            return {}


# 使用例
if __name__ == "__main__":
    import numpy as np
    from datetime import datetime, timedelta
    
    # サンプルデータ作成
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)
    
    # トレンドのあるデータを生成
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    close_prices = trend + noise
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.randn(100) * 0.5,
        'High': close_prices + np.abs(np.random.randn(100)) * 2,
        'Low': close_prices - np.abs(np.random.randn(100)) * 2,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    df.set_index('Date', inplace=True)
    
    # パターン認識
    recognizer = ChartPatternRecognizer()
    
    print("=== ゴールデン・デッドクロス ===")
    crosses = recognizer.golden_dead_cross(df)
    golden_dates = df.index[crosses['Golden_Cross']]
    dead_dates = df.index[crosses['Dead_Cross']]
    
    print(f"ゴールデンクロス: {len(golden_dates)}回")
    for date in golden_dates:
        confidence = crosses.loc[date, 'Golden_Confidence']
        print(f"  {date.date()}: 信頼度 {confidence:.1f}%")
    
    print(f"\nデッドクロス: {len(dead_dates)}回")
    for date in dead_dates:
        confidence = crosses.loc[date, 'Dead_Confidence']
        print(f"  {date.date()}: 信頼度 {confidence:.1f}%")
    
    print("\n=== サポート・レジスタンス ===")
    levels = recognizer.support_resistance_levels(df)
    print(f"レジスタンス: {[f'{level:.2f}' for level in levels['resistance']]}")
    print(f"サポート: {[f'{level:.2f}' for level in levels['support']]}")
    
    print("\n=== 全パターン検出 ===")
    all_patterns = recognizer.detect_all_patterns(df)
    print(f"総合信頼度: {all_patterns['overall_confidence']:.1f}%")
    
    if 'latest_signal' in all_patterns:
        signal = all_patterns['latest_signal']
        print(f"最新シグナル: {signal['type']} (信頼度: {signal['confidence']:.1f}%)")