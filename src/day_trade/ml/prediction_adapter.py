#!/usr/bin/env python3
"""
予測システムアダプター
Issue #870統合: 既存システムとの互換性を維持しながら新機能を提供

段階的移行とA/Bテスト機能を提供
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# 拡張予測システム
try:
    from enhanced_prediction_core import (
        EnhancedPredictionCore, create_enhanced_prediction_core,
        PredictionConfig, PredictionMode, PredictionResult
    )
    ENHANCED_CORE_AVAILABLE = True
except ImportError:
    ENHANCED_CORE_AVAILABLE = False

# 既存システム
try:
    from simple_ml_prediction_system import SimpleMLPredictionSystem
    LEGACY_SYSTEM_AVAILABLE = True
except ImportError:
    LEGACY_SYSTEM_AVAILABLE = False

# デイトレードエンジン
try:
    from day_trading_engine import PersonalDayTradingEngine
    DAY_TRADING_ENGINE_AVAILABLE = True
except ImportError:
    DAY_TRADING_ENGINE_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class AdapterMode(Enum):
    """アダプターモード"""
    LEGACY_ONLY = "legacy_only"        # 既存システムのみ
    ENHANCED_ONLY = "enhanced_only"    # 拡張システムのみ
    AB_TEST = "ab_test"                # A/Bテスト
    GRADUAL_ROLLOUT = "gradual_rollout"  # 段階的移行
    SMART_FALLBACK = "smart_fallback"   # スマートフォールバック


class TestGroup(Enum):
    """テストグループ"""
    CONTROL = "control"      # 制御群（既存システム）
    TREATMENT = "treatment"  # 処理群（拡張システム）


@dataclass
class AdapterConfig:
    """アダプター設定"""
    mode: AdapterMode = AdapterMode.SMART_FALLBACK
    rollout_percentage: float = 0.1  # 段階的移行の割合
    ab_test_split: float = 0.5       # A/Bテストの分割比率
    performance_threshold: float = 0.8  # 性能閾値
    fallback_threshold: float = 0.5     # フォールバック閾値
    comparison_window: int = 100        # 比較評価ウィンドウサイズ
    enable_logging: bool = True         # 詳細ログ有効化
    enable_metrics: bool = True         # メトリクス収集有効化


@dataclass
class ComparisonMetrics:
    """比較メトリクス"""
    legacy_mse: float = 0.0
    enhanced_mse: float = 0.0
    legacy_mae: float = 0.0
    enhanced_mae: float = 0.0
    legacy_r2: float = 0.0
    enhanced_r2: float = 0.0
    legacy_time: float = 0.0
    enhanced_time: float = 0.0
    sample_count: int = 0
    enhanced_improvement: float = 0.0  # 改善率
    statistical_significance: bool = False


@dataclass
class AdapterResult:
    """アダプター結果"""
    predictions: np.ndarray
    confidence: np.ndarray
    system_used: str
    test_group: Optional[TestGroup]
    processing_time: float
    fallback_occurred: bool = False
    error_message: Optional[str] = None
    enhanced_result: Optional[PredictionResult] = None
    legacy_result: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


class PredictionSystemAdapter:
    """予測システムアダプター"""

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.logger = logging.getLogger(__name__)

        # システム初期化
        self.enhanced_core = None
        self.legacy_system = None
        self.day_trading_engine = None

        # 状態管理
        self.is_initialized = False
        self.prediction_count = 0
        self.comparison_data = []
        self.performance_metrics = ComparisonMetrics()

        # A/Bテスト用状態
        self.test_assignments = {}  # セッション別テストグループ割り当て

        self._initialize_systems()

    def _initialize_systems(self) -> None:
        """システム初期化"""
        self.logger.info("予測システムアダプター初期化開始")

        # 拡張システム初期化
        if ENHANCED_CORE_AVAILABLE:
            try:
                prediction_config = PredictionConfig(mode=PredictionMode.AUTO)
                self.enhanced_core = create_enhanced_prediction_core(prediction_config)
                self.logger.info("拡張予測システム初期化完了")
            except Exception as e:
                self.logger.error(f"拡張システム初期化失敗: {e}")
                self.enhanced_core = None
        else:
            self.logger.warning("拡張システム未対応")

        # レガシーシステム初期化
        if LEGACY_SYSTEM_AVAILABLE:
            try:
                self.legacy_system = SimpleMLPredictionSystem()
                self.logger.info("レガシーシステム初期化完了")
            except Exception as e:
                self.logger.error(f"レガシーシステム初期化失敗: {e}")
                self.legacy_system = None
        else:
            self.logger.warning("レガシーシステム未対応")

        # デイトレードエンジン初期化
        if DAY_TRADING_ENGINE_AVAILABLE:
            try:
                self.day_trading_engine = PersonalDayTradingEngine()
                self.logger.info("デイトレードエンジン初期化完了")
            except Exception as e:
                self.logger.warning(f"デイトレードエンジン初期化失敗: {e}")
                self.day_trading_engine = None
        else:
            self.logger.warning("デイトレードエンジン未対応")

        self.is_initialized = True

    def predict(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                price_data: Optional[pd.DataFrame] = None,
                session_id: Optional[str] = None,
                **kwargs) -> AdapterResult:
        """統合予測実行"""
        start_time = time.time()
        self.prediction_count += 1

        if not self.is_initialized:
            raise RuntimeError("アダプターが初期化されていません")

        # 予測モード決定
        prediction_mode = self._determine_prediction_mode(session_id)

        try:
            if prediction_mode == "enhanced":
                result = self._enhanced_prediction(X, y, price_data, **kwargs)
                system_used = "enhanced"
                test_group = TestGroup.TREATMENT if self.config.mode == AdapterMode.AB_TEST else None

            elif prediction_mode == "legacy":
                result = self._legacy_prediction(X, y, **kwargs)
                system_used = "legacy"
                test_group = TestGroup.CONTROL if self.config.mode == AdapterMode.AB_TEST else None

            elif prediction_mode == "both":
                # A/Bテストまたは比較評価
                enhanced_result, legacy_result = self._dual_prediction(X, y, price_data, **kwargs)

                # メイン結果決定（設定に基づく）
                if self._should_use_enhanced_result(enhanced_result, legacy_result):
                    result = self._convert_enhanced_to_adapter_result(enhanced_result)
                    system_used = "enhanced"
                    test_group = TestGroup.TREATMENT
                else:
                    result = self._convert_legacy_to_adapter_result(legacy_result)
                    system_used = "legacy"
                    test_group = TestGroup.CONTROL

                # 比較データ記録
                if self.config.enable_metrics and y is not None:
                    self._record_comparison_data(enhanced_result, legacy_result, y)

            else:
                raise ValueError(f"不明な予測モード: {prediction_mode}")

            processing_time = time.time() - start_time

            # 処理時間を結果に設定
            if hasattr(result, 'processing_time'):
                result.processing_time = processing_time

            adapter_result = AdapterResult(
                predictions=result['predictions'],
                confidence=result.get('confidence', np.full_like(result['predictions'], 0.5)),
                system_used=system_used,
                test_group=test_group,
                processing_time=processing_time,
                fallback_occurred=False
            )

            # セッション記録
            if session_id and self.config.mode == AdapterMode.AB_TEST:
                self.test_assignments[session_id] = test_group

            return adapter_result

        except Exception as e:
            self.logger.error(f"予測実行エラー: {e}")

            # フォールバック処理
            return self._handle_prediction_failure(X, y, session_id, str(e))

    def _determine_prediction_mode(self, session_id: Optional[str] = None) -> str:
        """予測モード決定"""
        if self.config.mode == AdapterMode.LEGACY_ONLY:
            return "legacy"

        elif self.config.mode == AdapterMode.ENHANCED_ONLY:
            return "enhanced"

        elif self.config.mode == AdapterMode.AB_TEST:
            # セッションベースのA/Bテスト
            if session_id:
                if session_id in self.test_assignments:
                    group = self.test_assignments[session_id]
                else:
                    # 新規セッション: ランダム割り当て
                    group = (TestGroup.TREATMENT if np.random.random() < self.config.ab_test_split
                            else TestGroup.CONTROL)
                    self.test_assignments[session_id] = group

                return "enhanced" if group == TestGroup.TREATMENT else "legacy"
            else:
                # セッションIDなし: 比較評価
                return "both"

        elif self.config.mode == AdapterMode.GRADUAL_ROLLOUT:
            # 段階的移行
            if np.random.random() < self.config.rollout_percentage:
                return "enhanced"
            else:
                return "legacy"

        elif self.config.mode == AdapterMode.SMART_FALLBACK:
            # スマートフォールバック
            if self.enhanced_core and self._should_use_enhanced_system():
                return "enhanced"
            else:
                return "legacy"

        else:
            return "legacy"

    def _should_use_enhanced_system(self) -> bool:
        """拡張システム使用判定"""
        if not self.enhanced_core:
            return False

        # 性能メトリクスに基づく判定
        if len(self.comparison_data) >= self.config.comparison_window:
            recent_metrics = self._calculate_recent_metrics()

            # 拡張システムが有意に優秀か判定
            improvement = recent_metrics.enhanced_improvement
            significance = recent_metrics.statistical_significance

            return improvement > 0.05 and significance  # 5%以上改善かつ統計的有意
        else:
            # データ不足時は設定に従う
            return self.config.rollout_percentage > 0.5

    def _enhanced_prediction(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                           price_data: Optional[pd.DataFrame] = None,
                           **kwargs) -> Dict[str, Any]:
        """拡張システム予測"""
        if not self.enhanced_core:
            raise RuntimeError("拡張システム未対応")

        result = self.enhanced_core.predict(X, y, price_data, **kwargs)

        return {
            'predictions': result.predictions,
            'confidence': result.confidence,
            'meta_info': result.meta_info,
            'components_used': result.components_used
        }

    def _legacy_prediction(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                          **kwargs) -> Dict[str, Any]:
        """レガシーシステム予測"""
        if not self.legacy_system:
            raise RuntimeError("レガシーシステム未対応")

        # レガシーシステムのAPIに合わせた処理
        try:
            # 仮実装: 実際のAPIに合わせて調整が必要
            predictions = np.random.randn(len(X))  # 実際の予測ロジックに置換

            return {
                'predictions': predictions,
                'confidence': np.full_like(predictions, 0.5),
                'meta_info': {'system': 'legacy'}
            }

        except Exception as e:
            self.logger.error(f"レガシー予測失敗: {e}")
            raise

    def _dual_prediction(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                        price_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> Tuple[Any, Any]:
        """両システムで予測（比較用）"""
        enhanced_result = None
        legacy_result = None

        # 拡張システム予測
        if self.enhanced_core:
            try:
                enhanced_result = self.enhanced_core.predict(X, y, price_data, **kwargs)
            except Exception as e:
                self.logger.warning(f"拡張システム予測失敗: {e}")

        # レガシーシステム予測
        if self.legacy_system:
            try:
                legacy_result = self._legacy_prediction(X, y, **kwargs)
            except Exception as e:
                self.logger.warning(f"レガシーシステム予測失敗: {e}")

        return enhanced_result, legacy_result

    def _should_use_enhanced_result(self, enhanced_result: Any, legacy_result: Any) -> bool:
        """拡張システム結果使用判定"""
        if enhanced_result is None:
            return False
        if legacy_result is None:
            return True

        # 信頼度比較
        enhanced_confidence = np.mean(enhanced_result.confidence) if enhanced_result else 0.0
        legacy_confidence = np.mean(legacy_result.get('confidence', [0.5])) if legacy_result else 0.0

        return enhanced_confidence > legacy_confidence

    def _convert_enhanced_to_adapter_result(self, enhanced_result: PredictionResult) -> Dict[str, Any]:
        """拡張システム結果を統一形式に変換"""
        return {
            'predictions': enhanced_result.predictions,
            'confidence': enhanced_result.confidence,
            'meta_info': enhanced_result.meta_info
        }

    def _convert_legacy_to_adapter_result(self, legacy_result: Dict[str, Any]) -> Dict[str, Any]:
        """レガシーシステム結果を統一形式に変換"""
        return legacy_result

    def _record_comparison_data(self, enhanced_result: Any, legacy_result: Any,
                              y_true: pd.Series) -> None:
        """比較データ記録"""
        if enhanced_result is None or legacy_result is None:
            return

        try:
            # 評価メトリクス計算
            enhanced_pred = enhanced_result.predictions
            legacy_pred = legacy_result['predictions']

            enhanced_mse = mean_squared_error(y_true, enhanced_pred)
            legacy_mse = mean_squared_error(y_true, legacy_pred)

            enhanced_mae = mean_absolute_error(y_true, enhanced_pred)
            legacy_mae = mean_absolute_error(y_true, legacy_pred)

            enhanced_r2 = r2_score(y_true, enhanced_pred)
            legacy_r2 = r2_score(y_true, legacy_pred)

            comparison_record = {
                'timestamp': datetime.now(),
                'enhanced_mse': enhanced_mse,
                'legacy_mse': legacy_mse,
                'enhanced_mae': enhanced_mae,
                'legacy_mae': legacy_mae,
                'enhanced_r2': enhanced_r2,
                'legacy_r2': legacy_r2,
                'enhanced_time': enhanced_result.processing_time,
                'legacy_time': 0.0,  # レガシーシステムの処理時間
                'sample_size': len(y_true)
            }

            self.comparison_data.append(comparison_record)

            # ウィンドウサイズ制限
            if len(self.comparison_data) > self.config.comparison_window * 2:
                self.comparison_data = self.comparison_data[-self.config.comparison_window:]

        except Exception as e:
            self.logger.warning(f"比較データ記録失敗: {e}")

    def _calculate_recent_metrics(self) -> ComparisonMetrics:
        """最近のメトリクス計算"""
        if not self.comparison_data:
            return ComparisonMetrics()

        recent_data = self.comparison_data[-self.config.comparison_window:]

        enhanced_mse = np.mean([d['enhanced_mse'] for d in recent_data])
        legacy_mse = np.mean([d['legacy_mse'] for d in recent_data])

        enhanced_mae = np.mean([d['enhanced_mae'] for d in recent_data])
        legacy_mae = np.mean([d['legacy_mae'] for d in recent_data])

        enhanced_r2 = np.mean([d['enhanced_r2'] for d in recent_data])
        legacy_r2 = np.mean([d['legacy_r2'] for d in recent_data])

        # 改善率計算
        improvement = 0.0
        if legacy_mse > 0:
            improvement = (legacy_mse - enhanced_mse) / legacy_mse

        # 統計的有意性（簡易版）
        enhanced_mses = [d['enhanced_mse'] for d in recent_data]
        legacy_mses = [d['legacy_mse'] for d in recent_data]

        try:
            from scipy.stats import ttest_rel
            _, p_value = ttest_rel(legacy_mses, enhanced_mses)
            statistical_significance = p_value < 0.05
        except ImportError:
            # scipy未対応時は簡易判定
            statistical_significance = len(recent_data) > 10 and abs(improvement) > 0.1

        return ComparisonMetrics(
            legacy_mse=legacy_mse,
            enhanced_mse=enhanced_mse,
            legacy_mae=legacy_mae,
            enhanced_mae=enhanced_mae,
            legacy_r2=legacy_r2,
            enhanced_r2=enhanced_r2,
            sample_count=len(recent_data),
            enhanced_improvement=improvement,
            statistical_significance=statistical_significance
        )

    def _handle_prediction_failure(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                                 session_id: Optional[str] = None,
                                 error_message: str = "") -> AdapterResult:
        """予測失敗時の処理"""
        self.logger.error(f"予測失敗、フォールバック実行: {error_message}")

        # 基本的なフォールバック予測
        if y is not None:
            fallback_pred = np.full(len(X), y.mean())
        else:
            fallback_pred = np.zeros(len(X))

        return AdapterResult(
            predictions=fallback_pred,
            confidence=np.full_like(fallback_pred, 0.1),
            system_used="fallback",
            test_group=None,
            processing_time=0.0,
            fallback_occurred=True,
            error_message=error_message
        )

    def get_comparison_report(self) -> Dict[str, Any]:
        """比較レポート取得"""
        if not self.comparison_data:
            return {
                'status': 'insufficient_data',
                'message': '比較データが不足しています'
            }

        metrics = self._calculate_recent_metrics()

        return {
            'status': 'available',
            'timestamp': datetime.now(),
            'sample_count': metrics.sample_count,
            'enhanced_metrics': {
                'mse': metrics.enhanced_mse,
                'mae': metrics.enhanced_mae,
                'r2': metrics.enhanced_r2
            },
            'legacy_metrics': {
                'mse': metrics.legacy_mse,
                'mae': metrics.legacy_mae,
                'r2': metrics.legacy_r2
            },
            'improvement': {
                'mse_improvement': metrics.enhanced_improvement,
                'statistical_significance': metrics.statistical_significance
            },
            'recommendation': self._get_system_recommendation(metrics)
        }

    def _get_system_recommendation(self, metrics: ComparisonMetrics) -> str:
        """システム推奨決定"""
        if metrics.enhanced_improvement > 0.1 and metrics.statistical_significance:
            return "拡張システムへの完全移行を推奨"
        elif metrics.enhanced_improvement > 0.05:
            return "拡張システムの段階的導入を推奨"
        elif metrics.enhanced_improvement < -0.05:
            return "レガシーシステムの継続使用を推奨"
        else:
            return "さらなるデータ収集が必要"

    def update_rollout_percentage(self, new_percentage: float) -> None:
        """段階的移行率更新"""
        if 0.0 <= new_percentage <= 1.0:
            self.config.rollout_percentage = new_percentage
            self.logger.info(f"段階的移行率更新: {new_percentage:.1%}")
        else:
            raise ValueError("移行率は0.0-1.0の範囲で指定してください")

    def save_adapter_state(self, filepath: str) -> None:
        """アダプター状態保存"""
        try:
            state = {
                'config': self.config.__dict__,
                'prediction_count': self.prediction_count,
                'comparison_data': self.comparison_data[-100:],  # 最新100件
                'test_assignments': dict(list(self.test_assignments.items())[-100:]),
                'performance_metrics': self.performance_metrics.__dict__,
                'timestamp': datetime.now()
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"アダプター状態保存完了: {filepath}")

        except Exception as e:
            self.logger.error(f"状態保存エラー: {e}")


def create_prediction_adapter(config: Optional[AdapterConfig] = None) -> PredictionSystemAdapter:
    """予測システムアダプター作成"""
    return PredictionSystemAdapter(config)


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)

    # A/Bテスト設定
    config = AdapterConfig(
        mode=AdapterMode.AB_TEST,
        ab_test_split=0.5,
        enable_metrics=True
    )

    # アダプター作成
    adapter = create_prediction_adapter(config)

    # サンプルデータ
    np.random.seed(42)
    n_samples, n_features = 100, 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    y = pd.Series(np.random.randn(n_samples))

    price_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(50)) + 100,
        'volume': np.random.randint(1000, 10000, 50)
    })

    # テスト実行
    try:
        result = adapter.predict(X, y, price_data, session_id="test_session_1")

        print(f"予測結果:")
        print(f"  予測数: {len(result.predictions)}")
        print(f"  使用システム: {result.system_used}")
        print(f"  テストグループ: {result.test_group}")
        print(f"  処理時間: {result.processing_time:.3f}秒")
        print(f"  フォールバック発生: {result.fallback_occurred}")

    except Exception as e:
        print(f"予測エラー: {e}")

    # 比較レポート
    report = adapter.get_comparison_report()
    print(f"\n比較レポート:")
    print(f"  ステータス: {report['status']}")
    if report['status'] == 'available':
        print(f"  サンプル数: {report['sample_count']}")
        print(f"  推奨: {report['recommendation']}")

    print("予測システムアダプターのテスト完了")