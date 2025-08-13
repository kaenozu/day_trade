#!/usr/bin/env python3
"""
統合高度バックテストエンジン

Issue #753対応: バックテスト機能強化
すべての高度機能を統合したメインエンジン
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
import json

from .advanced_metrics import (
    AdvancedRiskMetrics,
    AdvancedReturnMetrics,
    MarketRegimeMetrics,
    AdvancedBacktestAnalyzer,
    MultiTimeframeAnalyzer
)

from .ml_integration import (
    MLBacktestConfig,
    MLBacktestResult,
    MLEnsembleBacktester
)

from .reporting import BacktestReportGenerator

from .types import (
    BacktestConfig,
    BacktestResult,
    BacktestMode,
    OptimizationObjective
)

warnings.filterwarnings("ignore")


@dataclass
class EnhancedBacktestConfig:
    """統合高度バックテスト設定"""

    # 基本設定
    basic_config: BacktestConfig

    # 高度分析有効化フラグ
    enable_advanced_risk_metrics: bool = True
    enable_advanced_return_metrics: bool = True
    enable_market_regime_analysis: bool = True
    enable_multi_timeframe_analysis: bool = False
    enable_ml_integration: bool = False

    # マルチタイムフレーム設定
    timeframes: List[str] = None

    # ML統合設定
    ml_config: Optional[MLBacktestConfig] = None

    # レポート設定
    generate_comprehensive_report: bool = True
    report_output_dir: Optional[str] = None
    include_charts: bool = True
    include_pdf: bool = True
    include_dashboard: bool = True

    # パフォーマンス設定
    parallel_processing: bool = False
    cache_intermediate_results: bool = True

    def __post_init__(self):
        """初期化後処理"""
        if self.timeframes is None:
            self.timeframes = ['1d', '1w']

        if self.enable_ml_integration and self.ml_config is None:
            self.ml_config = MLBacktestConfig()


@dataclass
class EnhancedBacktestResult:
    """統合高度バックテスト結果"""

    # 基本結果
    basic_result: BacktestResult

    # 高度分析結果
    advanced_risk_metrics: Optional[AdvancedRiskMetrics] = None
    advanced_return_metrics: Optional[AdvancedReturnMetrics] = None
    market_regime_metrics: Optional[MarketRegimeMetrics] = None

    # マルチタイムフレーム結果
    multi_timeframe_results: Optional[Dict[str, Any]] = None

    # ML統合結果
    ml_result: Optional[MLBacktestResult] = None

    # レポート情報
    report_info: Optional[Dict[str, Any]] = None

    # メタデータ
    analysis_time: float = 0.0
    config_used: Optional[EnhancedBacktestConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'basic_result': self.basic_result.to_dict() if hasattr(self.basic_result, 'to_dict') else {},
            'advanced_risk_metrics': self.advanced_risk_metrics.__dict__ if self.advanced_risk_metrics else {},
            'advanced_return_metrics': self.advanced_return_metrics.__dict__ if self.advanced_return_metrics else {},
            'market_regime_metrics': self.market_regime_metrics.__dict__ if self.market_regime_metrics else {},
            'multi_timeframe_results': self.multi_timeframe_results or {},
            'ml_result': self.ml_result.__dict__ if self.ml_result else {},
            'analysis_time': self.analysis_time,
            'report_paths': self.report_info or {}
        }


class EnhancedBacktestEngine:
    """統合高度バックテストエンジン"""

    def __init__(self, config: EnhancedBacktestConfig):
        """
        初期化

        Args:
            config: 統合設定
        """
        self.config = config

        # アナライザー初期化
        self.advanced_analyzer = AdvancedBacktestAnalyzer(
            confidence_level=0.95
        )

        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()

        # ML統合バックテスター（必要に応じて）
        self.ml_backtester = None
        if config.enable_ml_integration and config.ml_config:
            self.ml_backtester = MLEnsembleBacktester(config.ml_config)

        # レポート生成器
        self.report_generator = None
        if config.generate_comprehensive_report:
            self.report_generator = BacktestReportGenerator(
                output_dir=config.report_output_dir
            )

        # 結果キャッシュ
        self.cache = {} if config.cache_intermediate_results else None

    def run_enhanced_backtest(self,
                            historical_data: pd.DataFrame,
                            symbols: List[str],
                            strategy_function: Optional[callable] = None,
                            benchmark_data: Optional[pd.DataFrame] = None) -> EnhancedBacktestResult:
        """
        統合高度バックテスト実行

        Args:
            historical_data: 歴史的市場データ
            symbols: 対象銘柄リスト
            strategy_function: 取引戦略関数
            benchmark_data: ベンチマークデータ

        Returns:
            EnhancedBacktestResult: 統合結果
        """
        start_time = datetime.now()

        try:
            # 1. 基本バックテスト実行
            basic_result = self._run_basic_backtest(
                historical_data, symbols, strategy_function
            )

            # 2. 高度分析実行
            enhanced_result = EnhancedBacktestResult(
                basic_result=basic_result,
                config_used=self.config
            )

            # データ準備
            returns, portfolio_values = self._prepare_analysis_data(basic_result, historical_data)

            # 3. 高度リスク分析
            if self.config.enable_advanced_risk_metrics:
                enhanced_result.advanced_risk_metrics = self._analyze_advanced_risk(
                    returns, portfolio_values
                )

            # 4. 高度リターン分析
            if self.config.enable_advanced_return_metrics:
                enhanced_result.advanced_return_metrics = self._analyze_advanced_return(
                    returns, basic_result.trades, benchmark_data
                )

            # 5. 市場レジーム分析
            if self.config.enable_market_regime_analysis:
                enhanced_result.market_regime_metrics = self._analyze_market_regimes(
                    returns, historical_data
                )

            # 6. マルチタイムフレーム分析
            if self.config.enable_multi_timeframe_analysis:
                enhanced_result.multi_timeframe_results = self._analyze_multiple_timeframes(
                    historical_data
                )

            # 7. ML統合分析
            if self.config.enable_ml_integration and self.ml_backtester:
                enhanced_result.ml_result = self._run_ml_integration_analysis(
                    historical_data, symbols, benchmark_data
                )

            # 8. 包括的レポート生成
            if self.config.generate_comprehensive_report and self.report_generator:
                enhanced_result.report_info = self._generate_comprehensive_report(
                    enhanced_result
                )

            # 分析時間記録
            end_time = datetime.now()
            enhanced_result.analysis_time = (end_time - start_time).total_seconds()

            return enhanced_result

        except Exception as e:
            # エラー時も基本的な結果を返す
            print(f"高度バックテスト実行エラー: {e}")

            # 最小限の結果を返す
            basic_config = self.config.basic_config
            fallback_result = BacktestResult(
                config=basic_config,
                start_date=basic_config.start_date,
                end_date=basic_config.end_date,
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                value_at_risk=0.0,
                conditional_var=0.0,
                trades=[],
                positions=[],
                daily_returns=[],
                portfolio_value_history=[],
                drawdown_history=[]
            )

            end_time = datetime.now()
            return EnhancedBacktestResult(
                basic_result=fallback_result,
                analysis_time=(end_time - start_time).total_seconds(),
                config_used=self.config
            )

    def _run_basic_backtest(self,
                          historical_data: pd.DataFrame,
                          symbols: List[str],
                          strategy_function: Optional[callable]) -> BacktestResult:
        """基本バックテスト実行"""

        # 基本的なバックテスト実装（簡易版）
        config = self.config.basic_config

        # データから基本的な指標を計算
        if 'Close' in historical_data.columns:
            prices = historical_data['Close']
            returns = prices.pct_change().dropna()

            # 基本的な指標計算
            total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / (volatility) if volatility > 0 else 0

            # ドローダウン計算
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # カルマーレシオ
            annualized_return = returns.mean() * 252
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

            # 簡易取引統計
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            profit_factor = (positive_returns.sum() / abs(negative_returns.sum())) if len(negative_returns) > 0 else float('inf')

            # VaR計算
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

            # ポートフォリオ価値履歴
            initial_value = float(config.initial_capital)
            portfolio_values = [initial_value * val for val in cumulative.tolist()]

        else:
            # データが不足している場合のデフォルト値
            total_return = annualized_return = volatility = 0.0
            sharpe_ratio = calmar_ratio = 0.0
            max_drawdown = 0.0
            win_rate = profit_factor = 0.0
            var_95 = cvar_95 = 0.0
            returns = pd.Series([], dtype=float)
            portfolio_values = [float(config.initial_capital)]
            drawdown = pd.Series([0.0])

        return BacktestResult(
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            calmar_ratio=float(calmar_ratio),
            total_trades=len(returns),  # 簡易実装
            winning_trades=len(returns[returns > 0]) if len(returns) > 0 else 0,
            losing_trades=len(returns[returns < 0]) if len(returns) > 0 else 0,
            win_rate=float(win_rate),
            profit_factor=float(profit_factor) if not np.isinf(profit_factor) else 999.0,
            value_at_risk=float(var_95),
            conditional_var=float(cvar_95),
            trades=[],  # 実装省略
            positions=[],  # 実装省略
            daily_returns=returns.tolist(),
            portfolio_value_history=[config.initial_capital] + [config.initial_capital * (1 + r) for r in returns.cumsum().tolist()],
            drawdown_history=drawdown.tolist() if len(drawdown) > 0 else [0.0]
        )

    def _prepare_analysis_data(self, basic_result: BacktestResult,
                             historical_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """分析用データ準備"""

        # リターンデータ
        if basic_result.daily_returns:
            returns = pd.Series(
                basic_result.daily_returns,
                index=pd.date_range(
                    start=basic_result.start_date,
                    periods=len(basic_result.daily_returns),
                    freq='D'
                )
            )
        else:
            returns = pd.Series([], dtype=float)

        # ポートフォリオ価値
        if basic_result.portfolio_value_history:
            portfolio_values = pd.Series(
                [float(v) for v in basic_result.portfolio_value_history],
                index=pd.date_range(
                    start=basic_result.start_date,
                    periods=len(basic_result.portfolio_value_history),
                    freq='D'
                )
            )
        else:
            portfolio_values = pd.Series([float(basic_result.config.initial_capital)])

        return returns, portfolio_values

    def _analyze_advanced_risk(self, returns: pd.Series,
                             portfolio_values: pd.Series) -> AdvancedRiskMetrics:
        """高度リスク分析"""

        cache_key = "advanced_risk"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        risk_metrics = self.advanced_analyzer.calculate_advanced_risk_metrics(
            returns, portfolio_values
        )

        if self.cache:
            self.cache[cache_key] = risk_metrics

        return risk_metrics

    def _analyze_advanced_return(self, returns: pd.Series, trades: List[Any],
                               benchmark_data: Optional[pd.DataFrame]) -> AdvancedReturnMetrics:
        """高度リターン分析"""

        cache_key = "advanced_return"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # ベンチマークリターン準備
        benchmark_returns = None
        if benchmark_data is not None and 'Close' in benchmark_data.columns and len(benchmark_data) > 1:
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()

            # リターンデータの長さを合わせる
            if len(benchmark_returns) > len(returns):
                benchmark_returns = benchmark_returns.iloc[:len(returns)]
            elif len(benchmark_returns) < len(returns):
                returns = returns.iloc[:len(benchmark_returns)]

        return_metrics = self.advanced_analyzer.calculate_advanced_return_metrics(
            returns, trades, benchmark_returns
        )

        if self.cache:
            self.cache[cache_key] = return_metrics

        return return_metrics

    def _analyze_market_regimes(self, returns: pd.Series,
                              historical_data: pd.DataFrame) -> MarketRegimeMetrics:
        """市場レジーム分析"""

        cache_key = "market_regime"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # 市場データの準備
        if 'Close' in historical_data.columns and 'Volume' in historical_data.columns:
            market_data = historical_data[['Close', 'Volume']].copy()
        else:
            # 最小限のデータで実行
            market_data = pd.DataFrame({
                'Close': np.random.uniform(950, 1050, len(returns)),
                'Volume': np.random.uniform(1e6, 3e6, len(returns))
            }, index=returns.index if len(returns) > 0 else pd.date_range('2023-01-01', periods=10))

        regime_metrics = self.advanced_analyzer.analyze_market_regimes(
            returns, market_data
        )

        if self.cache:
            self.cache[cache_key] = regime_metrics

        return regime_metrics

    def _analyze_multiple_timeframes(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """マルチタイムフレーム分析"""

        cache_key = "multi_timeframe"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        # OHLCV形式のデータが必要
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        if all(col in historical_data.columns for col in required_columns):
            # インデックスがDatetimeIndexでない場合は設定
            if not isinstance(historical_data.index, pd.DatetimeIndex):
                if 'timestamp' in historical_data.columns:
                    historical_data = historical_data.set_index('timestamp')
                else:
                    # デフォルトの日付インデックスを設定
                    historical_data.index = pd.date_range(
                        start='2023-01-01',
                        periods=len(historical_data),
                        freq='D'
                    )

            results = self.multi_timeframe_analyzer.analyze_multiple_timeframes(
                historical_data, self.config.timeframes
            )
        else:
            # データが不足している場合は空の結果
            results = {}

        if self.cache:
            self.cache[cache_key] = results

        return results

    def _run_ml_integration_analysis(self, historical_data: pd.DataFrame,
                                   symbols: List[str],
                                   benchmark_data: Optional[pd.DataFrame]) -> MLBacktestResult:
        """ML統合分析"""

        if not self.ml_backtester:
            raise ValueError("ML統合バックテスターが初期化されていません")

        cache_key = "ml_integration"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]

        try:
            ml_result = self.ml_backtester.run_ml_backtest(
                historical_data, symbols, benchmark_data
            )
        except Exception as e:
            print(f"ML統合分析エラー: {e}")
            # デフォルトのML結果を返す
            ml_result = self._create_default_ml_result()

        if self.cache:
            self.cache[cache_key] = ml_result

        return ml_result

    def _create_default_ml_result(self) -> MLBacktestResult:
        """デフォルトML結果作成"""
        return MLBacktestResult(
            total_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            alpha=0.0,
            beta=1.0,
            prediction_accuracy=0.5,
            direction_accuracy=0.5,
            signal_precision=0.0,
            signal_recall=0.0,
            signal_f1_score=0.0,
            model_contributions={},
            dynamic_weight_evolution=pd.DataFrame(),
            feature_importance={},
            information_ratio=0.0,
            tracking_error=0.0,
            maximum_drawdown=0.0,
            predictions=[],
            portfolio_evolution=pd.DataFrame(),
            model_performance_history=pd.DataFrame()
        )

    def _generate_comprehensive_report(self, result: EnhancedBacktestResult) -> Dict[str, Any]:
        """包括的レポート生成"""

        if not self.report_generator:
            return {}

        try:
            report_info = self.report_generator.generate_comprehensive_report(
                backtest_result=result.basic_result,
                advanced_risk_metrics=result.advanced_risk_metrics,
                advanced_return_metrics=result.advanced_return_metrics,
                market_regime_metrics=result.market_regime_metrics,
                ml_result=result.ml_result,
                report_name="enhanced_backtest"
            )

            return report_info

        except Exception as e:
            print(f"レポート生成エラー: {e}")
            return {'error': str(e)}

    def export_results(self, result: EnhancedBacktestResult,
                      output_path: str, format: str = 'json') -> str:
        """結果エクスポート"""

        output_file = Path(output_path)

        if format.lower() == 'json':
            with open(output_file.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            return str(output_file.with_suffix('.json'))

        elif format.lower() == 'csv':
            # 基本指標をCSVで出力
            basic_metrics = {
                'total_return': result.basic_result.total_return,
                'annualized_return': result.basic_result.annualized_return,
                'volatility': result.basic_result.volatility,
                'sharpe_ratio': result.basic_result.sharpe_ratio,
                'max_drawdown': result.basic_result.max_drawdown,
                'win_rate': result.basic_result.win_rate,
                'analysis_time': result.analysis_time
            }

            df = pd.DataFrame([basic_metrics])
            csv_path = output_file.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            return str(csv_path)

        else:
            raise ValueError(f"サポートされていない形式: {format}")

    def get_performance_summary(self, result: EnhancedBacktestResult) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""

        summary = {
            'basic_performance': {
                'total_return': result.basic_result.total_return,
                'sharpe_ratio': result.basic_result.sharpe_ratio,
                'max_drawdown': result.basic_result.max_drawdown,
                'win_rate': result.basic_result.win_rate
            },
            'analysis_metadata': {
                'analysis_time': result.analysis_time,
                'features_enabled': {
                    'advanced_risk': result.advanced_risk_metrics is not None,
                    'advanced_return': result.advanced_return_metrics is not None,
                    'market_regime': result.market_regime_metrics is not None,
                    'multi_timeframe': result.multi_timeframe_results is not None,
                    'ml_integration': result.ml_result is not None
                }
            }
        }

        # 高度指標があれば追加
        if result.advanced_risk_metrics:
            summary['advanced_risk'] = {
                'var_1d': result.advanced_risk_metrics.var_1,
                'cvar_1d': result.advanced_risk_metrics.cvar_1,
                'sortino_ratio': result.advanced_risk_metrics.sortino_ratio,
                'ulcer_index': result.advanced_risk_metrics.ulcer_index
            }

        if result.advanced_return_metrics:
            summary['advanced_return'] = {
                'calmar_ratio': result.advanced_return_metrics.calmar_ratio,
                'information_ratio': result.advanced_return_metrics.information_ratio,
                'profit_factor': result.advanced_return_metrics.profit_factor,
                'expectancy': result.advanced_return_metrics.expectancy
            }

        if result.ml_result:
            summary['ml_performance'] = {
                'prediction_accuracy': result.ml_result.prediction_accuracy,
                'excess_return': result.ml_result.excess_return,
                'information_ratio': result.ml_result.information_ratio
            }

        return summary


def create_quick_backtest_config(start_date: datetime,
                               end_date: datetime,
                               initial_capital: float = 1000000,
                               enable_all_features: bool = True) -> EnhancedBacktestConfig:
    """クイック設定作成ヘルパー"""

    from decimal import Decimal

    basic_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal(str(initial_capital))
    )

    return EnhancedBacktestConfig(
        basic_config=basic_config,
        enable_advanced_risk_metrics=enable_all_features,
        enable_advanced_return_metrics=enable_all_features,
        enable_market_regime_analysis=enable_all_features,
        enable_multi_timeframe_analysis=False,  # 重いため無効
        enable_ml_integration=False,  # 重いため無効
        generate_comprehensive_report=True,
        include_charts=True,
        include_pdf=False,  # 軽量化のため無効
        include_dashboard=False  # 軽量化のため無効
    )