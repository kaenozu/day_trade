#!/usr/bin/env python3
"""
Next-Gen AI Trading Environment
強化学習用マルチアセット取引環境シミュレーター

PPO/A3C/DQN アルゴリズム対応・完全セーフモード実装
"""

# gymライブラリの軽量代替実装
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    # 軽量な代替実装
    class spaces:
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

            def sample(self):
                return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
import copy

from ..utils.logging_config import get_context_logger
from ..data.advanced_ml_engine import AdvancedMLEngine

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class TradingAction:
    """取引アクション定義"""
    position_size: float  # -1.0 ~ 1.0 (ショート～ロング)
    asset_allocation: Dict[str, float]  # 資産配分 (合計=1.0)
    risk_level: float  # 0.0 ~ 1.0 (リスクレベル)
    hold_period: int = 1  # 保有期間 (日数)

@dataclass
class MarketState:
    """市場状態定義"""
    prices: Dict[str, float]  # 現在価格
    technical_indicators: Dict[str, Dict[str, float]]  # テクニカル指標
    market_sentiment: Dict[str, float]  # 市場センチメント
    volatility: Dict[str, float]  # ボラティリティ
    volume: Dict[str, float]  # 出来高
    macro_indicators: Dict[str, float]  # マクロ経済指標
    portfolio_state: Dict[str, Any]  # ポートフォリオ状態
    timestamp: datetime

@dataclass
class TradingReward:
    """報酬構造定義"""
    profit_loss: float = 0.0  # 損益 (40%)
    risk_adjusted_return: float = 0.0  # リスク調整済みリターン (35%)
    drawdown_penalty: float = 0.0  # ドローダウンペナルティ (15%)
    trading_costs: float = 0.0  # 取引コスト (10%)
    total_reward: float = 0.0

class MultiAssetTradingEnvironment:
    """
    マルチアセット取引環境

    【重要】完全セーフモード - 実際の取引は一切実行しません
    シミュレーション・学習・研究目的のみの実装
    """

    def __init__(self,
                 symbols: List[str],
                 initial_balance: float = 1000000.0,  # 初期資金: 100万円
                 max_position_size: float = 0.2,  # 最大ポジションサイズ: 20%
                 transaction_cost: float = 0.001,  # 取引手数料: 0.1%
                 lookback_window: int = 60,  # ルックバック期間: 60日
                 reward_scaling: float = 1000.0,  # 報酬スケーリング
                 risk_free_rate: float = 0.02):  # リスクフリーレート: 2%

        super().__init__()

        self.symbols = symbols
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.lookbook_window = lookback_window
        self.reward_scaling = reward_scaling
        self.risk_free_rate = risk_free_rate

        # 状態空間定義 (512次元)
        self.state_dim = 512
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # アクション空間定義 (連続空間)
        # [position_size (-1 to 1), asset_allocation (softmax), risk_level (0 to 1)]
        self.action_dim = 1 + len(symbols) + 1  # position + allocation + risk
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        # 環境状態
        self.current_step = 0
        self.max_steps = 1000
        self.market_data = {}
        self.portfolio = {}
        self.balance = initial_balance
        self.positions = {symbol: 0.0 for symbol in symbols}
        self.trade_history = []
        self.performance_history = []

        # リスク管理
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.volatility_window = 20

        # 状態履歴
        self.state_history = []
        self.reward_history = []

        logger.info("Multi-Asset Trading Environment initialized")
        logger.info(f"Target assets: {symbols}, Initial balance: {initial_balance:,.0f}")
        logger.info("Safe mode - No real trading functionality")

    def reset(self) -> np.ndarray:
        """環境リセット"""

        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.trade_history = []
        self.performance_history = []
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        self.state_history = []
        self.reward_history = []

        # 初期市場状態生成
        initial_state = self._generate_initial_market_state()
        observation = self._market_state_to_observation(initial_state)

        logger.debug("取引環境リセット完了")
        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """環境ステップ実行"""

        self.current_step += 1

        # アクション解析
        trading_action = self._parse_action(action)

        # 市場状態更新
        market_state = self._update_market_state()

        # 取引実行（シミュレーション）
        trade_result = self._execute_trade(trading_action, market_state)

        # 報酬計算
        reward = self._calculate_reward(trade_result, market_state)

        # ポートフォリオ更新
        self._update_portfolio(trade_result, market_state)

        # 次の観測状態
        observation = self._market_state_to_observation(market_state)

        # 終了条件チェック
        done = self._is_episode_done()

        # 情報辞書
        info = self._get_step_info(trade_result, reward, market_state)

        # 履歴更新
        self.state_history.append(market_state)
        self.reward_history.append(reward.total_reward)
        self.performance_history.append({
            'step': self.current_step,
            'balance': self.balance,
            'total_value': self._calculate_total_portfolio_value(market_state),
            'positions': copy.deepcopy(self.positions),
            'reward': reward.total_reward
        })

        return observation, reward.total_reward, done, info

    def _generate_initial_market_state(self) -> MarketState:
        """初期市場状態生成"""

        # シミュレーション用の初期価格（実際の取引データは使用しない）
        prices = {symbol: 100.0 + np.random.normal(0, 5) for symbol in self.symbols}

        # 初期テクニカル指標（ランダム生成）
        technical_indicators = {}
        for symbol in self.symbols:
            technical_indicators[symbol] = {
                'sma_20': prices[symbol] * (1 + np.random.normal(0, 0.05)),
                'rsi': 50 + np.random.normal(0, 10),
                'macd': np.random.normal(0, 2),
                'bollinger_upper': prices[symbol] * 1.1,
                'bollinger_lower': prices[symbol] * 0.9,
                'volume_ratio': 1.0 + np.random.normal(0, 0.2)
            }

        # 初期市場センチメント
        market_sentiment = {symbol: np.random.uniform(-1, 1) for symbol in self.symbols}

        # 初期ボラティリティ
        volatility = {symbol: 0.2 + np.random.exponential(0.1) for symbol in self.symbols}

        # 初期出来高
        volume = {symbol: 1000000 * (1 + np.random.exponential(0.5)) for symbol in self.symbols}

        # マクロ経済指標（シミュレーション）
        macro_indicators = {
            'interest_rate': 0.02 + np.random.normal(0, 0.005),
            'inflation_rate': 0.025 + np.random.normal(0, 0.003),
            'gdp_growth': 0.03 + np.random.normal(0, 0.01),
            'market_fear_index': 15 + np.random.exponential(5),
            'currency_strength': 1.0 + np.random.normal(0, 0.05)
        }

        # ポートフォリオ状態
        portfolio_state = {
            'total_value': self.balance,
            'cash_ratio': 1.0,
            'equity_ratio': 0.0,
            'leverage_ratio': 0.0,
            'diversification_score': 0.0
        }

        return MarketState(
            prices=prices,
            technical_indicators=technical_indicators,
            market_sentiment=market_sentiment,
            volatility=volatility,
            volume=volume,
            macro_indicators=macro_indicators,
            portfolio_state=portfolio_state,
            timestamp=datetime.now()
        )

    def _parse_action(self, action: np.ndarray) -> TradingAction:
        """アクション解析"""

        # アクション正規化
        action = np.clip(action, -1.0, 1.0)

        # ポジションサイズ
        position_size = action[0] * self.max_position_size

        # 資産配分（Softmax正規化）
        raw_allocation = action[1:1+len(self.symbols)]
        exp_allocation = np.exp(raw_allocation - np.max(raw_allocation))
        asset_allocation = exp_allocation / np.sum(exp_allocation)

        allocation_dict = {symbol: float(alloc) for symbol, alloc in zip(self.symbols, asset_allocation)}

        # リスクレベル
        risk_level = (action[-1] + 1.0) / 2.0  # -1~1 -> 0~1

        return TradingAction(
            position_size=position_size,
            asset_allocation=allocation_dict,
            risk_level=risk_level
        )

    def _update_market_state(self) -> MarketState:
        """市場状態更新（シミュレーション）"""

        # 前の状態から価格変動をシミュレート
        if self.state_history:
            prev_state = self.state_history[-1]
            prices = {}

            for symbol in self.symbols:
                prev_price = prev_state.prices[symbol]
                volatility = prev_state.volatility[symbol]

                # ランダムウォーク + ドリフト
                drift = np.random.normal(0.0001, 0.005)  # 小さな正のドリフト
                shock = np.random.normal(0, volatility * 0.1)
                price_change = drift + shock

                new_price = prev_price * (1 + price_change)
                prices[symbol] = max(new_price, 0.1)  # 価格が負にならないよう制限
        else:
            # 初期状態の場合
            return self._generate_initial_market_state()

        # テクニカル指標更新（簡単化）
        technical_indicators = {}
        for symbol in self.symbols:
            price = prices[symbol]
            technical_indicators[symbol] = {
                'sma_20': price * (1 + np.random.normal(0, 0.02)),
                'rsi': max(0, min(100, 50 + np.random.normal(0, 15))),
                'macd': np.random.normal(0, 3),
                'bollinger_upper': price * (1.05 + np.random.uniform(0, 0.1)),
                'bollinger_lower': price * (0.95 - np.random.uniform(0, 0.1)),
                'volume_ratio': max(0.1, 1.0 + np.random.normal(0, 0.3))
            }

        # センチメント更新（平均回帰）
        market_sentiment = {}
        for symbol in self.symbols:
            prev_sentiment = self.state_history[-1].market_sentiment[symbol] if self.state_history else 0
            sentiment_change = np.random.normal(-0.1 * prev_sentiment, 0.2)  # 平均回帰
            market_sentiment[symbol] = np.clip(prev_sentiment + sentiment_change, -1, 1)

        # ボラティリティ更新（GARCH風）
        volatility = {}
        for symbol in self.symbols:
            prev_vol = self.state_history[-1].volatility[symbol] if self.state_history else 0.2
            vol_shock = np.random.exponential(0.05)
            volatility[symbol] = max(0.05, min(1.0, 0.9 * prev_vol + 0.1 * vol_shock))

        # 出来高更新
        volume = {}
        for symbol in self.symbols:
            base_volume = 1000000
            vol_multiplier = 1 + volatility[symbol] + abs(market_sentiment[symbol])
            volume[symbol] = base_volume * vol_multiplier * (1 + np.random.exponential(0.3))

        # マクロ指標（ゆっくり変化）
        if self.state_history:
            prev_macro = self.state_history[-1].macro_indicators
            macro_indicators = {}
            for key, prev_value in prev_macro.items():
                change = np.random.normal(0, 0.001)  # 小さな変化
                macro_indicators[key] = prev_value + change
        else:
            macro_indicators = {
                'interest_rate': 0.02,
                'inflation_rate': 0.025,
                'gdp_growth': 0.03,
                'market_fear_index': 20,
                'currency_strength': 1.0
            }

        # ポートフォリオ状態計算
        total_value = self._calculate_total_portfolio_value_with_prices(prices)
        cash_value = self.balance

        portfolio_state = {
            'total_value': total_value,
            'cash_ratio': cash_value / total_value if total_value > 0 else 1.0,
            'equity_ratio': (total_value - cash_value) / total_value if total_value > 0 else 0.0,
            'leverage_ratio': self._calculate_leverage_ratio(),
            'diversification_score': self._calculate_diversification_score()
        }

        return MarketState(
            prices=prices,
            technical_indicators=technical_indicators,
            market_sentiment=market_sentiment,
            volatility=volatility,
            volume=volume,
            macro_indicators=macro_indicators,
            portfolio_state=portfolio_state,
            timestamp=datetime.now()
        )

    def _execute_trade(self, action: TradingAction, market_state: MarketState) -> Dict[str, Any]:
        """取引実行（シミュレーション）"""

        trades_executed = []
        total_cost = 0.0

        try:
            # 各資産での取引実行
            for symbol, allocation in action.asset_allocation.items():
                if allocation > 0.001:  # 最小取引単位
                    current_price = market_state.prices[symbol]
                    current_position = self.positions[symbol]

                    # 目標ポジション計算
                    target_position_value = self.balance * action.position_size * allocation
                    target_shares = target_position_value / current_price

                    # 取引量計算
                    trade_shares = target_shares - current_position

                    if abs(trade_shares) > 0.001:  # 最小取引量チェック
                        # 取引コスト計算
                        trade_value = abs(trade_shares * current_price)
                        cost = trade_value * self.transaction_cost

                        # 残高チェック
                        if trade_shares > 0:  # 買い注文
                            required_cash = trade_value + cost
                            if required_cash <= self.balance:
                                # 取引実行
                                self.positions[symbol] += trade_shares
                                self.balance -= required_cash
                                total_cost += cost

                                trades_executed.append({
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'shares': trade_shares,
                                    'price': current_price,
                                    'value': trade_value,
                                    'cost': cost
                                })
                        else:  # 売り注文
                            # 売却可能チェック
                            if abs(trade_shares) <= current_position:
                                # 取引実行
                                self.positions[symbol] += trade_shares  # trade_sharesは負値
                                self.balance += trade_value - cost
                                total_cost += cost

                                trades_executed.append({
                                    'symbol': symbol,
                                    'action': 'SELL',
                                    'shares': abs(trade_shares),
                                    'price': current_price,
                                    'value': trade_value,
                                    'cost': cost
                                })

            # 取引履歴に記録
            if trades_executed:
                self.trade_history.append({
                    'step': self.current_step,
                    'timestamp': market_state.timestamp,
                    'trades': trades_executed,
                    'total_cost': total_cost,
                    'portfolio_value_before': self._calculate_total_portfolio_value(market_state),
                    'action': action
                })

            return {
                'success': True,
                'trades_executed': trades_executed,
                'total_cost': total_cost,
                'message': f"{len(trades_executed)} 件の取引を実行"
            }

        except Exception as e:
            logger.error(f"取引実行エラー: {e}")
            return {
                'success': False,
                'trades_executed': [],
                'total_cost': 0.0,
                'message': f"取引エラー: {str(e)}"
            }

    def _calculate_reward(self, trade_result: Dict[str, Any], market_state: MarketState) -> TradingReward:
        """報酬計算"""

        # 現在のポートフォリオ価値
        current_value = self._calculate_total_portfolio_value(market_state)

        # 前ステップからのリターン
        if self.performance_history:
            prev_value = self.performance_history[-1]['total_value']
            raw_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            raw_return = (current_value - self.initial_balance) / self.initial_balance

        # 1. 利益・損失 (40%)
        profit_loss = raw_return * 40.0

        # 2. リスク調整済みリターン (35%)
        if len(self.performance_history) >= self.volatility_window:
            recent_returns = [p['total_value'] / self.performance_history[i-1]['total_value'] - 1
                            for i, p in enumerate(self.performance_history[-self.volatility_window:]) if i > 0]

            if recent_returns:
                volatility = np.std(recent_returns)
                if volatility > 0:
                    sharpe_ratio = (np.mean(recent_returns) - self.risk_free_rate / 252) / volatility
                    risk_adjusted_return = sharpe_ratio * 35.0
                else:
                    risk_adjusted_return = 0
            else:
                risk_adjusted_return = 0
        else:
            risk_adjusted_return = 0

        # 3. ドローダウンペナルティ (15%)
        self.peak_balance = max(self.peak_balance, current_value)
        current_drawdown = (self.peak_balance - current_value) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        drawdown_penalty = -current_drawdown * 15.0

        # 4. 取引コスト (10%)
        trading_costs = -trade_result['total_cost'] / current_value * 10.0 if current_value > 0 else 0

        # 総合報酬
        total_reward = profit_loss + risk_adjusted_return + drawdown_penalty + trading_costs

        # スケーリング
        total_reward *= self.reward_scaling

        return TradingReward(
            profit_loss=profit_loss,
            risk_adjusted_return=risk_adjusted_return,
            drawdown_penalty=drawdown_penalty,
            trading_costs=trading_costs,
            total_reward=total_reward
        )

    def _update_portfolio(self, trade_result: Dict[str, Any], market_state: MarketState):
        """ポートフォリオ更新"""

        # ポートフォリオ統計更新（既に_execute_tradeで実行済み）
        pass

    def _market_state_to_observation(self, market_state: MarketState) -> np.ndarray:
        """市場状態を観測ベクトルに変換"""

        observation = []

        # 価格情報 (資産数 × 1)
        prices = [market_state.prices[symbol] for symbol in self.symbols]
        observation.extend(prices)

        # テクニカル指標 (資産数 × 6指標)
        for symbol in self.symbols:
            indicators = market_state.technical_indicators[symbol]
            observation.extend([
                indicators['sma_20'],
                indicators['rsi'],
                indicators['macd'],
                indicators['bollinger_upper'],
                indicators['bollinger_lower'],
                indicators['volume_ratio']
            ])

        # 市場センチメント (資産数 × 1)
        sentiment = [market_state.market_sentiment[symbol] for symbol in self.symbols]
        observation.extend(sentiment)

        # ボラティリティ (資産数 × 1)
        volatility = [market_state.volatility[symbol] for symbol in self.symbols]
        observation.extend(volatility)

        # 出来高 (資産数 × 1) - 正規化
        volumes = [np.log(market_state.volume[symbol]) for symbol in self.symbols]
        observation.extend(volumes)

        # マクロ経済指標 (5項目)
        macro = list(market_state.macro_indicators.values())
        observation.extend(macro)

        # ポートフォリオ状態 (4項目)
        portfolio = list(market_state.portfolio_state.values())
        observation.extend(portfolio)

        # 現在のポジション (資産数 × 1)
        positions = [self.positions[symbol] for symbol in self.symbols]
        observation.extend(positions)

        # 過去のパフォーマンス特徴量 (10項目)
        if len(self.performance_history) >= 10:
            recent_returns = []
            for i in range(1, 11):
                prev_value = self.performance_history[-i-1]['total_value']
                curr_value = self.performance_history[-i]['total_value']
                ret = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
                recent_returns.append(ret)
            observation.extend(recent_returns)
        else:
            observation.extend([0.0] * 10)

        # 観測ベクトルをself.state_dimに合わせて調整
        current_dim = len(observation)
        if current_dim < self.state_dim:
            # 不足分をゼロパディング
            observation.extend([0.0] * (self.state_dim - current_dim))
        elif current_dim > self.state_dim:
            # 余分な部分を切り捨て
            observation = observation[:self.state_dim]

        return np.array(observation, dtype=np.float32)

    def _is_episode_done(self) -> bool:
        """エピソード終了判定"""

        # 最大ステップ数到達
        if self.current_step >= self.max_steps:
            return True

        # 破産チェック
        current_value = self._calculate_total_portfolio_value(self.state_history[-1] if self.state_history else None)
        if current_value < self.initial_balance * 0.1:  # 90%損失で終了
            logger.warning("破産により環境終了")
            return True

        # 最大ドローダウンチェック
        if self.max_drawdown > 0.5:  # 50%ドローダウンで終了
            logger.warning("最大ドローダウン到達により環境終了")
            return True

        return False

    def _get_step_info(self, trade_result: Dict[str, Any], reward: TradingReward, market_state: MarketState) -> Dict[str, Any]:
        """ステップ情報取得"""

        current_value = self._calculate_total_portfolio_value(market_state)

        return {
            'step': self.current_step,
            'balance': self.balance,
            'total_portfolio_value': current_value,
            'positions': copy.deepcopy(self.positions),
            'trades_executed': trade_result['trades_executed'],
            'total_trading_cost': trade_result['total_cost'],
            'reward_components': {
                'profit_loss': reward.profit_loss,
                'risk_adjusted_return': reward.risk_adjusted_return,
                'drawdown_penalty': reward.drawdown_penalty,
                'trading_costs': reward.trading_costs
            },
            'max_drawdown': self.max_drawdown,
            'portfolio_return': (current_value - self.initial_balance) / self.initial_balance
        }

    def _calculate_total_portfolio_value(self, market_state: Optional[MarketState] = None) -> float:
        """現在のポートフォリオ総価値計算"""

        if market_state is None:
            # 簡易計算（最新価格がない場合）
            return self.balance + sum(self.positions.values()) * 100  # 仮の価格

        return self._calculate_total_portfolio_value_with_prices(market_state.prices)

    def _calculate_total_portfolio_value_with_prices(self, prices: Dict[str, float]) -> float:
        """価格指定でのポートフォリオ総価値計算"""

        total_value = self.balance

        for symbol in self.symbols:
            if symbol in prices and symbol in self.positions:
                position_value = self.positions[symbol] * prices[symbol]
                total_value += position_value

        return total_value

    def _calculate_leverage_ratio(self) -> float:
        """レバレッジ比率計算"""

        total_position_value = 0
        for symbol in self.symbols:
            if self.state_history:
                price = self.state_history[-1].prices[symbol]
                total_position_value += abs(self.positions[symbol] * price)

        total_equity = self.balance + total_position_value
        return total_position_value / total_equity if total_equity > 0 else 0

    def _calculate_diversification_score(self) -> float:
        """分散化スコア計算"""

        if not self.state_history:
            return 0.0

        total_value = self._calculate_total_portfolio_value(self.state_history[-1])
        if total_value <= self.balance:
            return 0.0

        # ハーフィンダール指数ベースの分散化スコア
        weights = []
        for symbol in self.symbols:
            position_value = self.positions[symbol] * self.state_history[-1].prices[symbol]
            weight = position_value / (total_value - self.balance)
            weights.append(weight ** 2)

        herfindahl_index = sum(weights)
        diversification_score = 1 - herfindahl_index

        return diversification_score

    def get_environment_info(self) -> Dict[str, Any]:
        """環境情報取得"""
        return {
            "symbols": self.symbols,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_steps": self.max_steps,
            "initial_balance": self.initial_balance,
            "transaction_cost": self.transaction_cost,
            "observation_space": {
                "shape": self.observation_space.shape,
                "low": self.observation_space.low,
                "high": self.observation_space.high
            },
            "action_space": {
                "shape": self.action_space.shape,
                "low": self.action_space.low,
                "high": self.action_space.high
            }
        }

    def seed(self, seed: int = None) -> List[int]:
        """乱数シード設定"""
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """ポートフォリオサマリー取得"""

        if not self.state_history:
            return {"status": "環境未開始"}

        current_state = self.state_history[-1]
        current_value = self._calculate_total_portfolio_value(current_state)

        # パフォーマンス統計
        if len(self.performance_history) > 1:
            returns = []
            for i in range(1, len(self.performance_history)):
                prev_value = self.performance_history[i-1]['total_value']
                curr_value = self.performance_history[i]['total_value']
                ret = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
                returns.append(ret)

            total_return = (current_value - self.initial_balance) / self.initial_balance
            volatility = np.std(returns) if returns else 0
            sharpe_ratio = (np.mean(returns) - self.risk_free_rate / 252) / volatility if volatility > 0 else 0
        else:
            total_return = 0
            volatility = 0
            sharpe_ratio = 0

        return {
            'current_step': self.current_step,
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'total_portfolio_value': current_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trade_history),
            'positions': copy.deepcopy(self.positions),
            'diversification_score': self._calculate_diversification_score(),
            'leverage_ratio': self._calculate_leverage_ratio()
        }

    def render(self, mode='human'):
        """環境レンダリング（可視化）"""

        if mode == 'human':
            summary = self.get_portfolio_summary()
            print(f"Step: {summary['current_step']}")
            print(f"Portfolio Value: ¥{summary['total_portfolio_value']:,.0f}")
            print(f"Total Return: {summary['total_return_pct']:.2f}%")
            print(f"Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
            print(f"Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
            print(f"Total Trades: {summary['total_trades']}")
            print("---")

        return summary if mode == 'rgb_array' else None

    def close(self):
        """環境終了処理"""
        logger.info("Multi-Asset Trading Environment 終了")

# 環境ファクトリー関数
def create_trading_environment(symbols: List[str] = None, **kwargs) -> MultiAssetTradingEnvironment:
    """取引環境作成"""

    if symbols is None:
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]  # デフォルト銘柄（シミュレーション用）

    env = MultiAssetTradingEnvironment(symbols=symbols, **kwargs)
    logger.info(f"取引環境作成完了: {len(symbols)} 資産")

    return env

if __name__ == "__main__":
    # 環境テスト
    print("=== Multi-Asset Trading Environment テスト ===")

    # 環境作成
    env = create_trading_environment(
        symbols=["STOCK_A", "STOCK_B", "STOCK_C"],
        initial_balance=1000000,
        max_steps=100
    )

    # リセットと初期状態
    initial_obs = env.reset()
    print(f"初期観測次元: {initial_obs.shape}")
    print(f"アクション次元: {env.action_space.shape}")

    # ランダムエージェントでテスト
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print(f"Step {step + 1}: Reward={reward:.3f}, Portfolio=¥{info['total_portfolio_value']:,.0f}")

        if done:
            break

    # 最終サマリー
    final_summary = env.get_portfolio_summary()
    print(f"\n最終結果:")
    print(f"トータルリターン: {final_summary['total_return_pct']:.2f}%")
    print(f"シャープレシオ: {final_summary['sharpe_ratio']:.3f}")
    print(f"最大ドローダウン: {final_summary['max_drawdown_pct']:.2f}%")

    env.close()
    print("テスト完了")
