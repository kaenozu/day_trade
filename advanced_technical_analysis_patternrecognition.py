#!/usr/bin/env python3
"""
advanced_technical_analysis.py - PatternRecognition

リファクタリングにより分割されたモジュール
"""

class PatternRecognition:
    """パターン認識システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_candlestick_patterns(self, data: pd.DataFrame) -> List[PatternMatch]:
        """ローソク足パターン検出"""

        patterns = []

        try:
            open_price = data['Open']
            high = data['High']
            low = data['Low']
            close = data['Close']

            # 1. Doji
            doji_mask = abs(close - open_price) <= (high - low) * 0.1
            patterns.extend(self._create_pattern_matches("Doji", doji_mask, "reversal", 0.7))

            # 2. Hammer
            body = abs(close - open_price)
            upper_shadow = high - np.maximum(close, open_price)
            lower_shadow = np.minimum(close, open_price) - low
            hammer_mask = (lower_shadow > 2 * body) & (upper_shadow < body)
            patterns.extend(self._create_pattern_matches("Hammer", hammer_mask, "reversal", 0.8))

            # 3. Engulfing
            bullish_engulf = (close > open_price) & (close.shift(1) < open_price.shift(1)) & \
                           (close > open_price.shift(1)) & (open_price < close.shift(1))
            bearish_engulf = (close < open_price) & (close.shift(1) > open_price.shift(1)) & \
                           (close < open_price.shift(1)) & (open_price > close.shift(1))

            patterns.extend(self._create_pattern_matches("Bullish Engulfing", bullish_engulf, "reversal", 0.85))
            patterns.extend(self._create_pattern_matches("Bearish Engulfing", bearish_engulf, "reversal", 0.85))

            # 4. Morning Star / Evening Star
            morning_star = self._detect_morning_star(open_price, high, low, close)
            evening_star = self._detect_evening_star(open_price, high, low, close)

            patterns.extend(self._create_pattern_matches("Morning Star", morning_star, "reversal", 0.9))
            patterns.extend(self._create_pattern_matches("Evening Star", evening_star, "reversal", 0.9))

        except Exception as e:
            self.logger.error(f"ローソク足パターン検出エラー: {e}")

        return patterns

    def detect_price_patterns(self, data: pd.DataFrame) -> List[PatternMatch]:
        """価格パターン検出"""

        patterns = []

        try:
            close = data['Close']
            high = data['High']
            low = data['Low']

            # 1. Head and Shoulders
            patterns.extend(self._detect_head_shoulders(high, low, close))

            # 2. Double Top/Bottom
            patterns.extend(self._detect_double_top_bottom(high, low, close))

            # 3. Triangle Patterns
            patterns.extend(self._detect_triangles(high, low, close))

            # 4. Flag and Pennant
            patterns.extend(self._detect_flag_pennant(high, low, close))

            # 5. Cup and Handle
            patterns.extend(self._detect_cup_handle(close))

        except Exception as e:
            self.logger.error(f"価格パターン検出エラー: {e}")

        return patterns

    def _create_pattern_matches(self, pattern_name: str, mask: pd.Series,
                              pattern_type: str, reliability: float) -> List[PatternMatch]:
        """パターンマッチ作成"""

        matches = []
        indices = mask[mask].index

        for idx in indices:
            match = PatternMatch(
                pattern_name=pattern_name,
                match_score=reliability * 100,
                start_index=idx,
                end_index=idx,
                pattern_type=pattern_type,
                reliability=reliability
            )
            matches.append(match)

        return matches

    def _detect_morning_star(self, open_price: pd.Series, high: pd.Series,
                           low: pd.Series, close: pd.Series) -> pd.Series:
        """明けの明星パターン"""

        # 3日間のパターン
        day1_bear = close.shift(2) < open_price.shift(2)  # 1日目：陰線
        day2_small = abs(close.shift(1) - open_price.shift(1)) < abs(close.shift(2) - open_price.shift(2)) * 0.3  # 2日目：小さい実体
        day3_bull = close > open_price  # 3日目：陽線
        day3_recovery = close > (open_price.shift(2) + close.shift(2)) / 2  # 3日目：1日目の半分以上回復

        return day1_bear & day2_small & day3_bull & day3_recovery

    def _detect_evening_star(self, open_price: pd.Series, high: pd.Series,
                           low: pd.Series, close: pd.Series) -> pd.Series:
        """宵の明星パターン"""

        # 3日間のパターン
        day1_bull = close.shift(2) > open_price.shift(2)  # 1日目：陽線
        day2_small = abs(close.shift(1) - open_price.shift(1)) < abs(close.shift(2) - open_price.shift(2)) * 0.3  # 2日目：小さい実体
        day3_bear = close < open_price  # 3日目：陰線
        day3_decline = close < (open_price.shift(2) + close.shift(2)) / 2  # 3日目：1日目の半分以下に下落

        return day1_bull & day2_small & day3_bear & day3_decline

    def _detect_head_shoulders(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[PatternMatch]:
        """ヘッド・アンド・ショルダーズパターン"""
        # 簡単な実装（実際はより複雑な判定が必要）
        return []

    def _detect_double_top_bottom(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[PatternMatch]:
        """ダブルトップ・ボトムパターン"""
        # 簡単な実装
        return []

    def _detect_triangles(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[PatternMatch]:
        """三角形パターン"""
        # 簡単な実装
        return []

    def _detect_flag_pennant(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[PatternMatch]:
        """フラッグ・ペナントパターン"""
        # 簡単な実装
        return []

    def _detect_cup_handle(self, close: pd.Series) -> List[PatternMatch]:
        """カップ・アンド・ハンドルパターン"""
        # 簡単な実装
        return []
