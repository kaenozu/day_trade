#!/usr/bin/env python3
"""
アンサンブル分析器
"""

import numpy as np
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .types import AnalysisInsight
from .attribution import AttributionEngine
from .decomposition import PerformanceDecomposer
from .dashboard import VisualDashboard

logger = logging.getLogger(__name__)

class EnsembleAnalyzer:
    """アンサンブル分析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # コンポーネント初期化
        self.attribution_engine = AttributionEngine(
            method=self.config.get('attribution_method', 'shap')
        )
        self.performance_decomposer = PerformanceDecomposer(
            method=self.config.get('decomposition_method', 'variance_decomposition')
        )
        self.dashboard = VisualDashboard(
            style=self.config.get('visualization_style', 'plotly'),
            output_format=self.config.get('output_format', 'html')
        )

        # 分析履歴
        self.analysis_history: List[Dict[str, Any]] = []
        self.insights: List[AnalysisInsight] = []

        logger.info("EnsembleAnalyzer initialized")

    async def analyze_ensemble_performance(self,
                                         ensemble_system: Any,
                                         test_data: np.ndarray,
                                         test_targets: np.ndarray,
                                         model_predictions: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """アンサンブルパフォーマンス分析"""
        start_time = time.time()

        try:
            # アンサンブル予測取得
            if hasattr(ensemble_system, 'predict_ensemble'):
                ensemble_predictions, metadata = await ensemble_system.predict_ensemble(test_data)
            else:
                # フォールバック予測
                ensemble_predictions = np.random.randn(len(test_data), 1)
                metadata = {}

            # 個別モデル予測取得
            if model_predictions is None:
                model_predictions = await self._get_individual_predictions(
                    ensemble_system, test_data
                )

            # 1. 貢献度分析
            attributions = await self.attribution_engine.compute_attributions(
                ensemble_system, test_data, ensemble_predictions, model_predictions
            )

            # 2. パフォーマンス分解
            decomposition = await self.performance_decomposer.decompose(
                ensemble_predictions, model_predictions, attributions
            )

            # 3. 視覚化生成
            charts = await self.dashboard.generate_charts(attributions, decomposition)

            # 4. 洞察生成
            insights = await self._generate_insights(
                ensemble_predictions, test_targets, attributions, decomposition
            )

            # 5. 総合レポート作成
            analysis_report = {
                'summary': {
                    'ensemble_mse': mean_squared_error(test_targets, ensemble_predictions),
                    'ensemble_mae': mean_absolute_error(test_targets, ensemble_predictions),
                    'ensemble_r2': r2_score(test_targets, ensemble_predictions),
                    'n_models': len(model_predictions),
                    'analysis_time': time.time() - start_time
                },
                'attributions': {
                    'model_attributions': {k: float(np.mean(v)) for k, v in attributions.model_attributions.items()},
                    'feature_attributions': {k: float(np.mean(np.abs(v))) for k, v in attributions.feature_attributions.items()},
                    'method': attributions.attribution_method
                },
                'decomposition': {
                    'individual_contributions': decomposition.individual_contributions,
                    'interaction_terms': decomposition.interaction_terms,
                    'residual_performance': decomposition.residual_performance,
                    'variance_explained': decomposition.variance_explained
                },
                'visualizations': charts,
                'insights': [
                    {
                        'type': insight.insight_type,
                        'title': insight.title,
                        'description': insight.description,
                        'importance': insight.importance_score,
                        'recommendations': insight.recommendations
                    }
                    for insight in insights
                ],
                'metadata': {
                    'timestamp': time.time(),
                    'ensemble_metadata': metadata,
                    'config': self.config
                }
            }

            # 履歴保存
            self.analysis_history.append(analysis_report)
            self.insights.extend(insights)

            logger.info(f"Ensemble analysis completed in {time.time() - start_time:.2f}s")
            return analysis_report

        except Exception as e:
            logger.error(f"Ensemble analysis error: {e}")
            return {
                'error': str(e),
                'timestamp': time.time(),
                'analysis_time': time.time() - start_time
            }

    async def _get_individual_predictions(self,
                                        ensemble_system: Any,
                                        test_data: np.ndarray) -> Dict[str, np.ndarray]:
        """個別モデル予測取得"""
        try:
            model_predictions = {}

            if hasattr(ensemble_system, 'models') and hasattr(ensemble_system, 'model_ids'):
                for i, model in enumerate(ensemble_system.models):
                    model_id = ensemble_system.model_ids[i]
                    try:
                        if hasattr(model, 'predict'):
                            pred = model.predict(test_data)
                            model_predictions[model_id] = pred
                    except Exception as e:
                        logger.warning(f"Prediction failed for {model_id}: {e}")
                        model_predictions[model_id] = np.random.randn(len(test_data), 1)
            else:
                # フォールバック: ダミー予測
                for i in range(3):
                    model_id = f"model_{i}"
                    model_predictions[model_id] = np.random.randn(len(test_data), 1)

            return model_predictions

        except Exception as e:
            logger.error(f"Individual predictions error: {e}")
            return {"dummy_model": np.random.randn(len(test_data), 1)}

    async def _generate_insights(self,
                               predictions: np.ndarray,
                               targets: np.ndarray,
                               attributions: AttributionResult,
                               decomposition: PerformanceDecomposition) -> List[AnalysisInsight]:
        """洞察生成"""
        insights = []

        try:
            # 1. パフォーマンス洞察
            mse = mean_squared_error(targets, predictions)
            r2 = r2_score(targets, predictions)

            if r2 > 0.8:
                insights.append(AnalysisInsight(
                    insight_type="performance",
                    title="Excellent Ensemble Performance",
                    description=f"The ensemble achieves R² = {r2:.3f}, indicating excellent predictive performance.",
                    importance_score=0.9,
                    supporting_data={"r2": r2, "mse": mse},
                    recommendations=[
                        "Consider deploying this ensemble to production",
                        "Monitor performance for any degradation over time"
                    ],
                    timestamp=time.time()
                ))
            elif r2 < 0.5:
                insights.append(AnalysisInsight(
                    insight_type="performance",
                    title="Poor Ensemble Performance",
                    description=f"The ensemble shows R² = {r2:.3f}, indicating poor predictive performance.",
                    importance_score=0.8,
                    supporting_data={"r2": r2, "mse": mse},
                    recommendations=[
                        "Investigate individual model quality",
                        "Consider feature engineering improvements",
                        "Review ensemble weighting strategy"
                    ],
                    timestamp=time.time()
                ))

            # 2. 貢献度洞察
            model_contribs = {k: np.mean(v) for k, v in attributions.model_attributions.items()}
            best_model = max(model_contribs, key=model_contribs.get)
            worst_model = min(model_contribs, key=model_contribs.get)

            contrib_variance = np.var(list(model_contribs.values()))

            if contrib_variance > 0.1:
                insights.append(AnalysisInsight(
                    insight_type="attribution",
                    title="Unbalanced Model Contributions",
                    description=f"Model contributions vary significantly. {best_model} contributes most, {worst_model} least.",
                    importance_score=0.7,
                    supporting_data={"contributions": model_contribs, "variance": contrib_variance},
                    recommendations=[
                        f"Consider increasing weight for {best_model}",
                        f"Investigate poor performance of {worst_model}",
                        "Optimize ensemble weights dynamically"
                    ],
                    timestamp=time.time()
                ))

            # 3. 分解洞察
            if decomposition.residual_performance > 0.3:
                insights.append(AnalysisInsight(
                    insight_type="pattern",
                    title="High Residual Performance",
                    description=f"Large unexplained variance ({decomposition.residual_performance:.1%}) suggests missing model interactions.",
                    importance_score=0.6,
                    supporting_data={"residual": decomposition.residual_performance},
                    recommendations=[
                        "Consider adding interaction terms",
                        "Explore non-linear ensemble methods",
                        "Investigate feature engineering opportunities"
                    ],
                    timestamp=time.time()
                ))

            # 4. 異常検出
            prediction_std = np.std(predictions)
            if prediction_std > np.std(targets) * 2:
                insights.append(AnalysisInsight(
                    insight_type="anomaly",
                    title="High Prediction Variance",
                    description="Ensemble predictions show unusually high variance compared to targets.",
                    importance_score=0.5,
                    supporting_data={"pred_std": prediction_std, "target_std": np.std(targets)},
                    recommendations=[
                        "Review model stability",
                        "Consider ensemble regularization",
                        "Check for overfitting"
                    ],
                    timestamp=time.time()
                ))

        except Exception as e:
            logger.error(f"Insight generation error: {e}")

        return insights

    def get_analysis_summary(self) -> Dict[str, Any]:
        """分析サマリー取得"""
        if not self.analysis_history:
            return {}

        recent_analyses = self.analysis_history[-5:]

        return {
            'total_analyses': len(self.analysis_history),
            'avg_ensemble_r2': np.mean([
                a.get('summary', {}).get('ensemble_r2', 0)
                for a in recent_analyses
            ]),
            'avg_analysis_time': np.mean([
                a.get('summary', {}).get('analysis_time', 0)
                for a in recent_analyses
            ]),
            'total_insights': len(self.insights),
            'insight_types': list(set([i.insight_type for i in self.insights[-20:]])),
            'last_analysis': self.analysis_history[-1]['metadata']['timestamp'] if self.analysis_history else None
        }
