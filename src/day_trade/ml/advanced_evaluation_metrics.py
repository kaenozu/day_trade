#!/usr/bin/env python3
"""
Advanced Evaluation Metrics for Ensemble Learning

Issue #462: 高度評価指標システム実装
金融特化の評価指標でアンサンブル性能を多角的に評価
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class AdvancedMetrics:
    """高度評価指標結果"""
    # 基本メトリクス
    rmse: float
    mae: float
    mape: float
    r2_score: float
    
    # 方向性評価
    hit_rate: float
    precision_up: float
    recall_up: float
    precision_down: float
    recall_down: float
    
    # リスク調整済みメトリクス
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    
    # ボラティリティメトリクス
    prediction_volatility: float
    volatility_ratio: float
    tracking_error: float
    
    # 統計的メトリクス
    information_ratio: float
    jensen_alpha: float
    beta: float
    correlation: float
    
    # 信頼性メトリクス
    confidence_calibration: float
    prediction_intervals_coverage: float
    stability_score: float


class AdvancedEvaluationMetrics:
    """
    高度評価指標システム
    
    金融予測モデル専用の包括的評価指標:
    1. 基本精度指標
    2. 方向性評価指標  
    3. リスク調整済み指標
    4. ボラティリティ評価
    5. 統計的指標
    6. 信頼性評価
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        初期化
        
        Args:
            risk_free_rate: リスクフリーレート（年率）
        """
        self.risk_free_rate = risk_free_rate
        
    def evaluate_comprehensive(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             confidence: Optional[np.ndarray] = None,
                             timestamps: Optional[np.ndarray] = None,
                             benchmark: Optional[np.ndarray] = None) -> AdvancedMetrics:
        """
        包括的評価実行
        
        Args:
            y_true: 実際の値
            y_pred: 予測値
            confidence: 予測信頼度（オプション）
            timestamps: タイムスタンプ（オプション）
            benchmark: ベンチマーク（オプション）
            
        Returns:
            AdvancedMetrics: 評価結果
        """
        try:
            # 基本メトリクス
            basic_metrics = self._calculate_basic_metrics(y_true, y_pred)
            
            # 方向性評価
            direction_metrics = self._calculate_direction_metrics(y_true, y_pred)
            
            # リスク調整済みメトリクス
            risk_metrics = self._calculate_risk_adjusted_metrics(y_true, y_pred)
            
            # ボラティリティメトリクス
            volatility_metrics = self._calculate_volatility_metrics(y_true, y_pred)
            
            # 統計的メトリクス
            statistical_metrics = self._calculate_statistical_metrics(y_true, y_pred, benchmark)
            
            # 信頼性メトリクス
            reliability_metrics = self._calculate_reliability_metrics(y_true, y_pred, confidence)
            
            # 結果統合
            return AdvancedMetrics(
                # 基本メトリクス
                rmse=basic_metrics['rmse'],
                mae=basic_metrics['mae'],
                mape=basic_metrics['mape'],
                r2_score=basic_metrics['r2_score'],
                
                # 方向性評価
                hit_rate=direction_metrics['hit_rate'],
                precision_up=direction_metrics['precision_up'],
                recall_up=direction_metrics['recall_up'],
                precision_down=direction_metrics['precision_down'],
                recall_down=direction_metrics['recall_down'],
                
                # リスク調整済みメトリクス
                sharpe_ratio=risk_metrics['sharpe_ratio'],
                calmar_ratio=risk_metrics['calmar_ratio'],
                sortino_ratio=risk_metrics['sortino_ratio'],
                maximum_drawdown=risk_metrics['maximum_drawdown'],
                
                # ボラティリティメトリクス
                prediction_volatility=volatility_metrics['prediction_volatility'],
                volatility_ratio=volatility_metrics['volatility_ratio'],
                tracking_error=volatility_metrics['tracking_error'],
                
                # 統計的メトリクス
                information_ratio=statistical_metrics['information_ratio'],
                jensen_alpha=statistical_metrics['jensen_alpha'],
                beta=statistical_metrics['beta'],
                correlation=statistical_metrics['correlation'],
                
                # 信頼性メトリクス
                confidence_calibration=reliability_metrics['confidence_calibration'],
                prediction_intervals_coverage=reliability_metrics['prediction_intervals_coverage'],
                stability_score=reliability_metrics['stability_score']
            )
            
        except Exception as e:
            logger.error(f"包括的評価エラー: {e}")
            # エラー時はデフォルト値を返す
            return self._create_default_metrics()
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """基本メトリクス計算"""
        try:
            # MSE, RMSE, MAE
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # R²スコア
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2_score': r2_score
            }
        
        except Exception as e:
            logger.warning(f"基本メトリクス計算エラー: {e}")
            return {'rmse': 0, 'mae': 0, 'mape': 0, 'r2_score': 0}
    
    def _calculate_direction_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """方向性評価メトリクス計算"""
        try:
            if len(y_true) < 2:
                return {'hit_rate': 0.5, 'precision_up': 0, 'recall_up': 0, 
                       'precision_down': 0, 'recall_down': 0}
            
            # 変化方向計算
            true_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)
            
            true_direction = np.sign(true_diff)
            pred_direction = np.sign(pred_diff)
            
            # Hit Rate（方向的中率）
            hit_rate = np.mean(true_direction == pred_direction)
            
            # 上昇予測の精度・再現率
            up_pred_mask = pred_direction > 0
            up_true_mask = true_direction > 0
            
            if np.sum(up_pred_mask) > 0:
                precision_up = np.sum(up_pred_mask & up_true_mask) / np.sum(up_pred_mask)
            else:
                precision_up = 0
            
            if np.sum(up_true_mask) > 0:
                recall_up = np.sum(up_pred_mask & up_true_mask) / np.sum(up_true_mask)
            else:
                recall_up = 0
            
            # 下降予測の精度・再現率
            down_pred_mask = pred_direction < 0
            down_true_mask = true_direction < 0
            
            if np.sum(down_pred_mask) > 0:
                precision_down = np.sum(down_pred_mask & down_true_mask) / np.sum(down_pred_mask)
            else:
                precision_down = 0
            
            if np.sum(down_true_mask) > 0:
                recall_down = np.sum(down_pred_mask & down_true_mask) / np.sum(down_true_mask)
            else:
                recall_down = 0
            
            return {
                'hit_rate': hit_rate,
                'precision_up': precision_up,
                'recall_up': recall_up,
                'precision_down': precision_down,
                'recall_down': recall_down
            }
        
        except Exception as e:
            logger.warning(f"方向性メトリクス計算エラー: {e}")
            return {'hit_rate': 0, 'precision_up': 0, 'recall_up': 0, 
                   'precision_down': 0, 'recall_down': 0}
    
    def _calculate_risk_adjusted_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """リスク調整済みメトリクス計算"""
        try:
            # リターン計算（前日比）
            if len(y_true) < 2:
                return {'sharpe_ratio': 0, 'calmar_ratio': 0, 'sortino_ratio': 0, 'maximum_drawdown': 0}
            
            true_returns = np.diff(y_true) / y_true[:-1]
            pred_returns = np.diff(y_pred) / y_pred[:-1]
            
            # 予測リターンの統計量
            mean_return = np.mean(pred_returns)
            return_std = np.std(pred_returns)
            
            # Sharpe Ratio
            if return_std > 0:
                sharpe_ratio = (mean_return - self.risk_free_rate / 252) / return_std * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + pred_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            maximum_drawdown = np.min(drawdowns)
            
            # Calmar Ratio
            if maximum_drawdown < 0:
                calmar_ratio = (mean_return * 252) / abs(maximum_drawdown)
            else:
                calmar_ratio = 0
            
            # Sortino Ratio（下方偏差使用）
            downside_returns = pred_returns[pred_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns)
                if downside_deviation > 0:
                    sortino_ratio = (mean_return - self.risk_free_rate / 252) / downside_deviation * np.sqrt(252)
                else:
                    sortino_ratio = 0
            else:
                sortino_ratio = sharpe_ratio
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'maximum_drawdown': maximum_drawdown
            }
        
        except Exception as e:
            logger.warning(f"リスク調整メトリクス計算エラー: {e}")
            return {'sharpe_ratio': 0, 'calmar_ratio': 0, 'sortino_ratio': 0, 'maximum_drawdown': 0}
    
    def _calculate_volatility_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """ボラティリティメトリクス計算"""
        try:
            if len(y_true) < 2:
                return {'prediction_volatility': 0, 'volatility_ratio': 1, 'tracking_error': 0}
            
            # リターン計算
            true_returns = np.diff(y_true) / y_true[:-1]
            pred_returns = np.diff(y_pred) / y_pred[:-1]
            
            # 予測ボラティリティ
            prediction_volatility = np.std(pred_returns) * np.sqrt(252)
            
            # 実際のボラティリティ
            actual_volatility = np.std(true_returns) * np.sqrt(252)
            
            # ボラティリティ比
            if actual_volatility > 0:
                volatility_ratio = prediction_volatility / actual_volatility
            else:
                volatility_ratio = 1
            
            # トラッキングエラー
            tracking_error = np.std(pred_returns - true_returns) * np.sqrt(252)
            
            return {
                'prediction_volatility': prediction_volatility,
                'volatility_ratio': volatility_ratio,
                'tracking_error': tracking_error
            }
        
        except Exception as e:
            logger.warning(f"ボラティリティメトリクス計算エラー: {e}")
            return {'prediction_volatility': 0, 'volatility_ratio': 1, 'tracking_error': 0}
    
    def _calculate_statistical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     benchmark: Optional[np.ndarray] = None) -> Dict[str, float]:
        """統計的メトリクス計算"""
        try:
            if len(y_true) < 2:
                return {'information_ratio': 0, 'jensen_alpha': 0, 'beta': 1, 'correlation': 0}
            
            # リターン計算
            true_returns = np.diff(y_true) / y_true[:-1]
            pred_returns = np.diff(y_pred) / y_pred[:-1]
            
            # 相関係数
            correlation = np.corrcoef(true_returns, pred_returns)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            # ベンチマーク比較（ある場合）
            if benchmark is not None and len(benchmark) >= len(y_true):
                benchmark_returns = np.diff(benchmark[:len(y_true)]) / benchmark[:-1]
                
                # Beta計算
                benchmark_var = np.var(benchmark_returns)
                if benchmark_var > 0:
                    beta = np.cov(pred_returns, benchmark_returns)[0, 1] / benchmark_var
                else:
                    beta = 1
                
                # Jensen's Alpha
                risk_free_daily = self.risk_free_rate / 252
                jensen_alpha = (np.mean(pred_returns) - risk_free_daily) - beta * (np.mean(benchmark_returns) - risk_free_daily)
                
                # Information Ratio
                excess_returns = pred_returns - benchmark_returns
                if np.std(excess_returns) > 0:
                    information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                else:
                    information_ratio = 0
            else:
                beta = 1
                jensen_alpha = 0
                information_ratio = 0
            
            return {
                'information_ratio': information_ratio,
                'jensen_alpha': jensen_alpha * 252,  # 年率化
                'beta': beta,
                'correlation': correlation
            }
        
        except Exception as e:
            logger.warning(f"統計的メトリクス計算エラー: {e}")
            return {'information_ratio': 0, 'jensen_alpha': 0, 'beta': 1, 'correlation': 0}
    
    def _calculate_reliability_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     confidence: Optional[np.ndarray] = None) -> Dict[str, float]:
        """信頼性メトリクス計算"""
        try:
            # 信頼度校正（キャリブレーション）
            if confidence is not None:
                confidence_calibration = self._calculate_confidence_calibration(y_true, y_pred, confidence)
                prediction_intervals_coverage = self._calculate_prediction_intervals_coverage(y_true, y_pred, confidence)
            else:
                confidence_calibration = 0.5
                prediction_intervals_coverage = 0.5
            
            # 安定性スコア（予測の一貫性）
            stability_score = self._calculate_stability_score(y_pred)
            
            return {
                'confidence_calibration': confidence_calibration,
                'prediction_intervals_coverage': prediction_intervals_coverage,
                'stability_score': stability_score
            }
        
        except Exception as e:
            logger.warning(f"信頼性メトリクス計算エラー: {e}")
            return {'confidence_calibration': 0.5, 'prediction_intervals_coverage': 0.5, 'stability_score': 0.5}
    
    def _calculate_confidence_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        confidence: np.ndarray) -> float:
        """信頼度キャリブレーション計算"""
        try:
            # エラーの絶対値
            errors = np.abs(y_true - y_pred)
            
            # 信頼度と精度の相関（負の相関が望ましい）
            if len(confidence) == len(errors) and np.std(confidence) > 0 and np.std(errors) > 0:
                calibration_corr = np.corrcoef(confidence, errors)[0, 1]
                # 負の相関を正のスコアに変換（-1～1を0～1にマッピング）
                calibration_score = (1 - calibration_corr) / 2
            else:
                calibration_score = 0.5
            
            return max(0, min(1, calibration_score))
        
        except Exception:
            return 0.5
    
    def _calculate_prediction_intervals_coverage(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                               confidence: np.ndarray) -> float:
        """予測区間カバレッジ計算"""
        try:
            # 信頼度に基づく予測区間を構築
            # 信頼度が高いほど狭い区間、低いほど広い区間
            prediction_std = np.std(y_pred) if len(y_pred) > 1 else 1.0
            
            # 信頼度を区間幅に変換（信頼度が低いほど広い区間）
            interval_widths = (1 - confidence) * prediction_std * 2
            
            # 予測区間
            lower_bounds = y_pred - interval_widths / 2
            upper_bounds = y_pred + interval_widths / 2
            
            # カバレッジ計算
            coverage = np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds))
            
            return coverage
        
        except Exception:
            return 0.5
    
    def _calculate_stability_score(self, y_pred: np.ndarray) -> float:
        """安定性スコア計算"""
        try:
            if len(y_pred) < 3:
                return 0.5
            
            # 予測値の二次微分（変化率の変化）を計算
            first_diff = np.diff(y_pred)
            second_diff = np.diff(first_diff)
            
            # 安定性は二次微分の小ささで評価
            stability_variance = np.var(second_diff)
            max_variance = np.var(y_pred)  # 正規化用
            
            if max_variance > 0:
                stability_score = np.exp(-stability_variance / max_variance)
            else:
                stability_score = 1.0
            
            return min(1.0, max(0.0, stability_score))
        
        except Exception:
            return 0.5
    
    def _create_default_metrics(self) -> AdvancedMetrics:
        """デフォルトメトリクス作成"""
        return AdvancedMetrics(
            rmse=0, mae=0, mape=0, r2_score=0,
            hit_rate=0.5, precision_up=0, recall_up=0, precision_down=0, recall_down=0,
            sharpe_ratio=0, calmar_ratio=0, sortino_ratio=0, maximum_drawdown=0,
            prediction_volatility=0, volatility_ratio=1, tracking_error=0,
            information_ratio=0, jensen_alpha=0, beta=1, correlation=0,
            confidence_calibration=0.5, prediction_intervals_coverage=0.5, stability_score=0.5
        )
    
    def generate_evaluation_report(self, metrics: AdvancedMetrics, 
                                  save_path: Optional[str] = None) -> str:
        """評価レポート生成"""
        report = f"""
=== アンサンブル学習システム 高度評価レポート ===

【基本精度指標】
RMSE: {metrics.rmse:.4f}
MAE: {metrics.mae:.4f}
MAPE: {metrics.mape:.2f}%
R²スコア: {metrics.r2_score:.4f}

【方向性評価指標】
方向的中率: {metrics.hit_rate:.3f}
上昇予測精度: {metrics.precision_up:.3f}
上昇予測再現率: {metrics.recall_up:.3f}
下降予測精度: {metrics.precision_down:.3f}
下降予測再現率: {metrics.recall_down:.3f}

【リスク調整済み指標】
シャープレシオ: {metrics.sharpe_ratio:.3f}
カルマーレシオ: {metrics.calmar_ratio:.3f}
ソルティノレシオ: {metrics.sortino_ratio:.3f}
最大ドローダウン: {metrics.maximum_drawdown:.3%}

【ボラティリティ評価】
予測ボラティリティ: {metrics.prediction_volatility:.3%}
ボラティリティ比: {metrics.volatility_ratio:.3f}
トラッキングエラー: {metrics.tracking_error:.3%}

【統計的指標】
情報レシオ: {metrics.information_ratio:.3f}
ジェンセンアルファ: {metrics.jensen_alpha:.3%}
ベータ: {metrics.beta:.3f}
相関係数: {metrics.correlation:.3f}

【信頼性評価】
信頼度キャリブレーション: {metrics.confidence_calibration:.3f}
予測区間カバレッジ: {metrics.prediction_intervals_coverage:.3f}
安定性スコア: {metrics.stability_score:.3f}

【総合評価】
予測精度: {'優秀' if metrics.hit_rate > 0.6 else '良好' if metrics.hit_rate > 0.55 else '改善要'}
リスク管理: {'優秀' if metrics.sharpe_ratio > 1.0 else '良好' if metrics.sharpe_ratio > 0.5 else '改善要'}
安定性: {'優秀' if metrics.stability_score > 0.8 else '良好' if metrics.stability_score > 0.6 else '改善要'}
"""
        
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"評価レポート保存: {save_path}")
            except Exception as e:
                logger.error(f"レポート保存エラー: {e}")
        
        return report
    
    def plot_performance_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                confidence: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None):
        """性能分析の可視化"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            import matplotlib.dates as mdates
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('アンサンブル学習システム 性能分析', fontsize=16)
            
            # 1. 予測 vs 実測
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
            axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            axes[0, 0].set_xlabel('実測値')
            axes[0, 0].set_ylabel('予測値')
            axes[0, 0].set_title('予測 vs 実測')
            
            # 2. 時系列プロット
            x = range(len(y_true))
            axes[0, 1].plot(x, y_true, label='実測', alpha=0.7)
            axes[0, 1].plot(x, y_pred, label='予測', alpha=0.7)
            if confidence is not None:
                # 信頼区間
                std_pred = np.std(y_pred) if len(y_pred) > 1 else 1
                lower = y_pred - confidence * std_pred
                upper = y_pred + confidence * std_pred
                axes[0, 1].fill_between(x, lower, upper, alpha=0.2, label='信頼区間')
            axes[0, 1].set_xlabel('時間')
            axes[0, 1].set_ylabel('値')
            axes[0, 1].set_title('時系列比較')
            axes[0, 1].legend()
            
            # 3. 残差プロット
            residuals = y_true - y_pred
            axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('予測値')
            axes[1, 0].set_ylabel('残差')
            axes[1, 0].set_title('残差分析')
            
            # 4. エラー分布
            axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('予測誤差')
            axes[1, 1].set_ylabel('頻度')
            axes[1, 1].set_title('誤差分布')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"性能分析グラフ保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib未インストール、可視化はスキップされます")
        except Exception as e:
            logger.error(f"性能分析可視化エラー: {e}")


if __name__ == "__main__":
    print("=== Advanced Evaluation Metrics テスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    n_samples = 500
    
    # 現実的な株価風データ
    t = np.linspace(0, 4*np.pi, n_samples)
    trend = 100 + 20 * t / (4*np.pi)
    noise = 5 * np.random.randn(n_samples)
    y_true = trend + 10 * np.sin(t) + noise
    
    # 予測データ（ある程度の精度を持つ）
    y_pred = y_true + 2 * np.random.randn(n_samples)
    
    # 信頼度（ランダム）
    confidence = np.random.uniform(0.3, 0.9, n_samples)
    
    # 評価実行
    evaluator = AdvancedEvaluationMetrics()
    metrics = evaluator.evaluate_comprehensive(y_true, y_pred, confidence)
    
    # 結果表示
    report = evaluator.generate_evaluation_report(metrics)
    print(report)
    
    # 性能分析可視化（テスト用）
    # evaluator.plot_performance_analysis(y_true, y_pred, confidence)
    
    print("\n=== 主要メトリクス ===")
    print(f"RMSE: {metrics.rmse:.4f}")
    print(f"方向的中率: {metrics.hit_rate:.3f}")
    print(f"シャープレシオ: {metrics.sharpe_ratio:.3f}")
    print(f"最大ドローダウン: {metrics.maximum_drawdown:.3%}")
    print(f"安定性スコア: {metrics.stability_score:.3f}")