#!/usr/bin/env python3
"""
Data validation utilities for Dynamic Weighting System

このモジュールは入力データの検証とデータ統計の生成を行います。
パフォーマンス管理の一部として、データの品質と整合性を保証します。
"""

from typing import Dict, List, Any, Union, Optional
import numpy as np

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class DataValidator:
    """
    データ検証・統計生成クラス

    入力データの妥当性チェック、統計情報の生成、
    データ品質の監視を行います。
    """

    def __init__(self):
        """初期化"""
        self.validation_history = []

    def normalize_to_array(
        self,
        data: Union[float, int, np.ndarray, List[float]],
        name: str
    ) -> np.ndarray:
        """
        Issue #475対応: データの一貫した配列変換

        Args:
            data: 変換対象データ
            name: データ名（エラーメッセージ用）

        Returns:
            正規化されたnp.ndarray

        Raises:
            TypeError: サポートされていない型の場合
            ValueError: 無効なデータの場合
        """
        try:
            # None チェック
            if data is None:
                raise ValueError(f"{name}がNoneです")

            # 型別処理
            if isinstance(data, (int, float)):
                # 単一値の場合
                if np.isnan(data) or np.isinf(data):
                    raise ValueError(f"{name}に無効な値が含まれています: {data}")
                return np.array([float(data)])

            elif isinstance(data, (list, tuple)):
                # リスト/タプルの場合
                if len(data) == 0:
                    raise ValueError(f"{name}が空です")
                array_data = np.array(data, dtype=float)

            elif isinstance(data, np.ndarray):
                # NumPy配列の場合
                if data.size == 0:
                    raise ValueError(f"{name}が空の配列です")
                array_data = data.astype(float)

            else:
                # その他の型（pd.Series等も含む）
                try:
                    array_data = np.array(data, dtype=float)
                except Exception as e:
                    raise TypeError(
                        f"{name}の型{type(data)}はサポートされていません: {e}"
                    )

            # 1次元に変換
            array_data = np.atleast_1d(array_data.flatten())

            # 有効値チェック
            if np.any(np.isnan(array_data)) or np.any(np.isinf(array_data)):
                raise ValueError(f"{name}に無効な値(NaN/Inf)が含まれています")

            return array_data

        except Exception as e:
            logger.error(f"データ正規化エラー ({name}): {e}")
            raise

    def validate_input_data(
        self,
        predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
        actuals: Union[float, int, np.ndarray, List[float]]
    ) -> Dict[str, Any]:
        """
        Issue #475対応: 入力データの検証

        Args:
            predictions: 予測値
            actuals: 実際値

        Returns:
            検証結果レポート
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'model_stats': {},
            'data_shape': {},
            'validation_id': len(self.validation_history)
        }

        try:
            # 実際値の検証
            normalized_actuals = self.normalize_to_array(actuals, "actuals")
            report['data_shape']['actuals'] = len(normalized_actuals)

            # 予測値の検証
            for model_name, pred in predictions.items():
                try:
                    normalized_pred = self.normalize_to_array(
                        pred, f"predictions[{model_name}]"
                    )
                    report['data_shape'][model_name] = len(normalized_pred)
                    report['model_stats'][model_name] = {
                        'mean': float(np.mean(normalized_pred)),
                        'std': float(np.std(normalized_pred)),
                        'min': float(np.min(normalized_pred)),
                        'max': float(np.max(normalized_pred))
                    }

                    # 次元チェック
                    if (len(normalized_pred) != len(normalized_actuals) and
                        len(normalized_pred) > 1 and len(normalized_actuals) > 1):
                        report['warnings'].append(
                            f"{model_name}: 次元不一致 "
                            f"{len(normalized_pred)} vs {len(normalized_actuals)}"
                        )

                except Exception as e:
                    report['errors'].append(f"{model_name}: {str(e)}")
                    report['valid'] = False

        except Exception as e:
            report['errors'].append(f"実際値検証エラー: {str(e)}")
            report['valid'] = False

        # 検証履歴に追加
        self.validation_history.append({
            'timestamp': np.datetime64('now'),
            'valid': report['valid'],
            'error_count': len(report['errors']),
            'warning_count': len(report['warnings'])
        })

        # 履歴サイズ管理
        if len(self.validation_history) > 100:
            self.validation_history.pop(0)

        return report

    def check_data_consistency(
        self,
        predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
        actuals: Union[float, int, np.ndarray, List[float]]
    ) -> Dict[str, Any]:
        """
        データの一貫性チェック

        Args:
            predictions: 予測値
            actuals: 実際値

        Returns:
            一貫性チェック結果
        """
        consistency_report = {
            'consistent': True,
            'issues': [],
            'statistics': {}
        }

        try:
            normalized_actuals = self.normalize_to_array(actuals, "actuals")
            
            # 範囲チェック
            actual_range = np.max(normalized_actuals) - np.min(normalized_actuals)
            consistency_report['statistics']['actual_range'] = float(actual_range)

            # 各モデルとの一貫性チェック
            for model_name, pred in predictions.items():
                try:
                    normalized_pred = self.normalize_to_array(pred, model_name)
                    pred_range = np.max(normalized_pred) - np.min(normalized_pred)
                    
                    # 範囲比較
                    if pred_range > actual_range * 5:  # 予測範囲が実際値の5倍以上
                        consistency_report['issues'].append(
                            f"{model_name}: 予測範囲が異常に大きい "
                            f"(予測: {pred_range:.3f}, 実際: {actual_range:.3f})"
                        )
                        consistency_report['consistent'] = False
                    
                    # スケール比較
                    actual_scale = np.mean(np.abs(normalized_actuals))
                    pred_scale = np.mean(np.abs(normalized_pred))
                    
                    if pred_scale > actual_scale * 10 or pred_scale < actual_scale / 10:
                        consistency_report['issues'].append(
                            f"{model_name}: スケールの不一致 "
                            f"(予測: {pred_scale:.3f}, 実際: {actual_scale:.3f})"
                        )
                        consistency_report['consistent'] = False

                except Exception as e:
                    consistency_report['issues'].append(
                        f"{model_name}: 一貫性チェックエラー - {str(e)}"
                    )

        except Exception as e:
            consistency_report['consistent'] = False
            consistency_report['issues'].append(f"チェック処理エラー: {str(e)}")

        return consistency_report

    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        検証統計の取得

        Returns:
            検証統計情報
        """
        if not self.validation_history:
            return {'no_data': True}

        stats = {
            'total_validations': len(self.validation_history),
            'success_rate': 0.0,
            'average_errors': 0.0,
            'average_warnings': 0.0,
            'recent_trend': 'stable'
        }

        try:
            valid_count = sum(1 for v in self.validation_history if v['valid'])
            stats['success_rate'] = valid_count / len(self.validation_history)
            
            error_counts = [v['error_count'] for v in self.validation_history]
            warning_counts = [v['warning_count'] for v in self.validation_history]
            
            stats['average_errors'] = np.mean(error_counts)
            stats['average_warnings'] = np.mean(warning_counts)

            # トレンド分析（直近10件と全体の比較）
            if len(self.validation_history) >= 10:
                recent_success = sum(
                    1 for v in self.validation_history[-10:] if v['valid']
                ) / 10
                overall_success = stats['success_rate']
                
                if recent_success > overall_success + 0.1:
                    stats['recent_trend'] = 'improving'
                elif recent_success < overall_success - 0.1:
                    stats['recent_trend'] = 'degrading'

        except Exception as e:
            logger.error(f"検証統計計算エラー: {e}")
            stats['calculation_error'] = str(e)

        return stats

    def reset_validation_history(self):
        """検証履歴をリセット"""
        self.validation_history.clear()
        logger.info("データ検証履歴をリセットしました")

    def get_data_health_score(
        self,
        predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
        actuals: Union[float, int, np.ndarray, List[float]]
    ) -> float:
        """
        データ健全性スコアを計算

        Args:
            predictions: 予測値
            actuals: 実際値

        Returns:
            健全性スコア（0.0-1.0、高いほど良い）
        """
        try:
            # 基本検証
            validation_result = self.validate_input_data(predictions, actuals)
            base_score = 1.0 if validation_result['valid'] else 0.5
            
            # 警告数によるペナルティ
            warning_penalty = len(validation_result['warnings']) * 0.1
            base_score = max(0.0, base_score - warning_penalty)
            
            # 一貫性チェック
            consistency_result = self.check_data_consistency(predictions, actuals)
            if not consistency_result['consistent']:
                base_score *= 0.7  # 30%減点
            
            # 統計的品質評価
            try:
                normalized_actuals = self.normalize_to_array(actuals, "actuals")
                actual_std = np.std(normalized_actuals)
                
                # 変動が極端に小さい、または大きい場合はペナルティ
                if actual_std < 1e-6:  # ほぼ定数
                    base_score *= 0.8
                elif actual_std > 1000:  # 極端に大きな変動
                    base_score *= 0.9
            except Exception:
                base_score *= 0.9  # エラー時は軽微なペナルティ

            return max(0.0, min(1.0, base_score))

        except Exception as e:
            logger.error(f"データ健全性スコア計算エラー: {e}")
            return 0.0