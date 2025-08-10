#!/usr/bin/env python3
"""
リスク分析統合コーディネーター
Risk Analysis Coordinator

生成AI・深層学習・ルールベース分析の統合管理
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from .generative_ai_engine import GenerativeAIRiskEngine, RiskAnalysisRequest, RiskAnalysisResult
from .fraud_detection_engine import FraudDetectionEngine, FraudDetectionRequest
from ..realtime.alert_system import AlertManager, AlertLevel

logger = get_context_logger(__name__)

@dataclass
class RiskAssessmentSummary:
    """リスク評価サマリー"""
    request_id: str
    overall_risk_score: float  # 0-1
    risk_category: str  # "low", "medium", "high", "critical"
    confidence_score: float   # 0-1
    analysis_methods: List[str]
    key_risk_factors: List[str]
    recommendations: List[str]
    estimated_loss_potential: float
    processing_time_total: float
    component_results: Dict[str, Any]
    timestamp: datetime

class RiskAnalysisCoordinator:
    """リスク分析統合コーディネーター"""

    def __init__(self):
        # コンポーネント初期化
        self.generative_ai_engine = GenerativeAIRiskEngine()
        self.fraud_detection_engine = FraudDetectionEngine()
        self.alert_manager = AlertManager()

        # 分析履歴
        self.analysis_history: List[RiskAssessmentSummary] = []
        self.daily_statistics = {}

        # 設定
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }

        self.alert_thresholds = {
            'medium_risk': 0.5,
            'high_risk': 0.7,
            'critical_risk': 0.85
        }

        # パフォーマンス統計
        self.performance_stats = {
            'total_assessments': 0,
            'successful_assessments': 0,
            'avg_processing_time': 0.0,
            'component_usage': {
                'generative_ai': 0,
                'fraud_detection': 0,
                'rule_based': 0
            },
            'risk_distribution': {
                'low': 0, 'medium': 0, 'high': 0, 'critical': 0
            }
        }

        logger.info("リスク分析コーディネーター初期化完了")

    async def comprehensive_risk_assessment(
        self,
        transaction_data: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        enable_ai_analysis: bool = True,
        enable_fraud_detection: bool = True
    ) -> RiskAssessmentSummary:
        """包括的リスク評価"""

        start_time = time.time()
        request_id = f"RISK_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"

        logger.info(f"包括的リスク評価開始: {request_id}")

        try:
            # 並列分析実行
            analysis_tasks = []
            analysis_methods = []

            # 生成AI分析
            if enable_ai_analysis:
                ai_request = self._create_ai_analysis_request(
                    request_id, transaction_data, market_context, user_profile
                )
                analysis_tasks.append(self._run_ai_analysis(ai_request))
                analysis_methods.append("GenerativeAI")

            # 不正検知分析
            if enable_fraud_detection:
                fraud_request = self._create_fraud_detection_request(
                    request_id, transaction_data
                )
                analysis_tasks.append(self._run_fraud_analysis(fraud_request))
                analysis_methods.append("FraudDetection")

            # 基本ルールベース分析
            analysis_tasks.append(self._run_basic_risk_analysis(transaction_data))
            analysis_methods.append("RuleBasedAnalysis")

            # 並列実行
            analysis_results = await asyncio.gather(
                *analysis_tasks, return_exceptions=True
            )

            # 結果統合
            summary = await self._integrate_analysis_results(
                request_id, analysis_results, analysis_methods,
                transaction_data, start_time
            )

            # アラート判定
            await self._evaluate_and_send_alerts(summary)

            # 履歴保存
            self.analysis_history.append(summary)
            if len(self.analysis_history) > 10000:
                self.analysis_history = self.analysis_history[-5000:]

            # 統計更新
            self._update_performance_stats(summary, True)

            logger.info(f"リスク評価完了: {request_id} - {summary.risk_category}")
            return summary

        except Exception as e:
            logger.error(f"リスク評価エラー: {e}")

            # エラー時のフォールバック評価
            error_summary = self._create_error_summary(
                request_id, str(e), transaction_data, start_time
            )
            self._update_performance_stats(error_summary, False)
            return error_summary

    def _create_ai_analysis_request(
        self,
        request_id: str,
        transaction_data: Dict[str, Any],
        market_context: Optional[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]]
    ) -> RiskAnalysisRequest:
        """AI分析リクエスト作成"""

        return RiskAnalysisRequest(
            transaction_id=request_id,
            symbol=transaction_data.get('symbol', 'UNKNOWN'),
            transaction_type=transaction_data.get('type', 'unknown'),
            amount=float(transaction_data.get('amount', 0)),
            timestamp=datetime.fromisoformat(transaction_data.get('timestamp', datetime.now().isoformat())),
            market_data=market_context or {},
            user_profile=user_profile or {},
            additional_context=transaction_data
        )

    def _create_fraud_detection_request(
        self,
        request_id: str,
        transaction_data: Dict[str, Any]
    ) -> FraudDetectionRequest:
        """不正検知リクエスト作成"""

        return FraudDetectionRequest(
            transaction_id=request_id,
            user_id=transaction_data.get('user_id', 'unknown'),
            amount=float(transaction_data.get('amount', 0)),
            timestamp=datetime.fromisoformat(transaction_data.get('timestamp', datetime.now().isoformat())),
            transaction_type=transaction_data.get('type', 'unknown'),
            account_balance=float(transaction_data.get('account_balance', 0)),
            location=transaction_data.get('location', 'unknown'),
            device_info=transaction_data.get('device_info', {}),
            transaction_history=transaction_data.get('history', []),
            market_conditions=transaction_data.get('market_conditions', {})
        )

    async def _run_ai_analysis(self, request: RiskAnalysisRequest) -> RiskAnalysisResult:
        """AI分析実行"""

        try:
            result = await self.generative_ai_engine.analyze_risk_comprehensive(
                request,
                use_gpt4=True,
                use_claude=True,
                use_ensemble=True
            )
            self.performance_stats['component_usage']['generative_ai'] += 1
            return result
        except Exception as e:
            logger.error(f"AI分析エラー: {e}")
            raise

    async def _run_fraud_analysis(self, request: FraudDetectionRequest):
        """不正検知分析実行"""

        try:
            result = await self.fraud_detection_engine.detect_fraud(request)
            self.performance_stats['component_usage']['fraud_detection'] += 1
            return result
        except Exception as e:
            logger.error(f"不正検知エラー: {e}")
            raise

    async def _run_basic_risk_analysis(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """基本リスク分析"""

        start_time = time.time()

        try:
            risk_score = 0.0
            risk_factors = []

            # 金額ベースリスク
            amount = float(transaction_data.get('amount', 0))
            if amount > 50000000:  # 5000万円以上
                risk_score += 0.4
                risk_factors.append("超高額取引")
            elif amount > 10000000:  # 1000万円以上
                risk_score += 0.2
                risk_factors.append("高額取引")

            # 時間帯リスク
            timestamp_str = transaction_data.get('timestamp', datetime.now().isoformat())
            timestamp = datetime.fromisoformat(timestamp_str)
            hour = timestamp.hour

            if hour < 6 or hour > 22:
                risk_score += 0.15
                risk_factors.append("時間外取引")

            # 取引タイプリスク
            transaction_type = transaction_data.get('type', 'unknown')
            high_risk_types = ['margin', 'options', 'futures', 'fx']
            if transaction_type.lower() in high_risk_types:
                risk_score += 0.1
                risk_factors.append("高リスク商品")

            # 地理的リスク
            location = transaction_data.get('location', 'domestic')
            if location in ['foreign', 'high_risk_country']:
                risk_score += 0.2
                risk_factors.append("地理的リスク")

            # 市場状況リスク
            market_conditions = transaction_data.get('market_conditions', {})
            volatility = market_conditions.get('volatility', 0.2)
            if volatility > 0.4:
                risk_score += 0.15
                risk_factors.append("高ボラティリティ")

            # 正規化
            risk_score = min(1.0, max(0.0, risk_score))

            self.performance_stats['component_usage']['rule_based'] += 1

            return {
                'source': 'rule_based_analysis',
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'processing_time': time.time() - start_time,
                'confidence': 0.7
            }

        except Exception as e:
            logger.error(f"基本リスク分析エラー: {e}")
            return {
                'source': 'rule_based_analysis',
                'risk_score': 0.5,
                'risk_factors': ['分析エラー'],
                'processing_time': time.time() - start_time,
                'confidence': 0.3,
                'error': str(e)
            }

    async def _integrate_analysis_results(
        self,
        request_id: str,
        analysis_results: List[Any],
        analysis_methods: List[str],
        transaction_data: Dict[str, Any],
        start_time: float
    ) -> RiskAssessmentSummary:
        """分析結果統合"""

        valid_results = []
        component_results = {}

        for i, result in enumerate(analysis_results):
            method = analysis_methods[i] if i < len(analysis_methods) else f"Method_{i}"

            if isinstance(result, Exception):
                logger.warning(f"分析メソッド {method} でエラー: {result}")
                component_results[method] = {'error': str(result)}
                continue

            valid_results.append((method, result))
            component_results[method] = self._serialize_result(result)

        if not valid_results:
            raise ValueError("すべての分析メソッドが失敗しました")

        # 重み付き統合
        risk_scores = []
        confidences = []
        all_recommendations = []
        all_risk_factors = []
        method_weights = {
            'GenerativeAI': 0.5,
            'FraudDetection': 0.3,
            'RuleBasedAnalysis': 0.2
        }

        for method, result in valid_results:
            weight = method_weights.get(method, 0.1)

            if hasattr(result, 'risk_score'):
                # RiskAnalysisResult
                risk_scores.append((result.risk_score, weight))
                confidences.append(result.confidence)
                all_recommendations.extend(result.recommendations)
                all_risk_factors.extend(result.risk_factors.keys() if hasattr(result, 'risk_factors') else [])

            elif hasattr(result, 'fraud_probability'):
                # FraudDetectionResult
                risk_scores.append((result.fraud_probability, weight))
                confidences.append(result.confidence)
                all_recommendations.append(result.recommended_action)
                all_risk_factors.extend(result.risk_factors.keys())

            elif isinstance(result, dict):
                # 基本分析結果
                risk_scores.append((result.get('risk_score', 0.5), weight))
                confidences.append(result.get('confidence', 0.5))
                all_risk_factors.extend(result.get('risk_factors', []))

        # 加重平均リスクスコア計算
        if risk_scores:
            weighted_sum = sum(score * weight for score, weight in risk_scores)
            total_weight = sum(weight for _, weight in risk_scores)
            overall_risk_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        else:
            overall_risk_score = 0.5

        # リスクカテゴリ決定
        if overall_risk_score >= self.risk_thresholds['critical']:
            risk_category = 'critical'
        elif overall_risk_score >= self.risk_thresholds['high']:
            risk_category = 'high'
        elif overall_risk_score >= self.risk_thresholds['medium']:
            risk_category = 'medium'
        else:
            risk_category = 'low'

        # 損失ポテンシャル推定
        amount = float(transaction_data.get('amount', 0))
        estimated_loss_potential = amount * overall_risk_score * 0.1  # 簡略化

        return RiskAssessmentSummary(
            request_id=request_id,
            overall_risk_score=overall_risk_score,
            risk_category=risk_category,
            confidence_score=np.mean(confidences) if confidences else 0.5,
            analysis_methods=[method for method, _ in valid_results],
            key_risk_factors=list(set(all_risk_factors))[:10],  # 重複削除、上位10個
            recommendations=list(set(all_recommendations))[:5],   # 重複削除、上位5個
            estimated_loss_potential=estimated_loss_potential,
            processing_time_total=time.time() - start_time,
            component_results=component_results,
            timestamp=datetime.now()
        )

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """結果シリアライズ"""

        if hasattr(result, '__dict__'):
            return asdict(result) if hasattr(result, '_fields') else result.__dict__
        elif isinstance(result, dict):
            return result
        else:
            return {'value': str(result), 'type': type(result).__name__}

    async def _evaluate_and_send_alerts(self, summary: RiskAssessmentSummary):
        """アラート評価・送信"""

        risk_score = summary.overall_risk_score

        # アラートレベル決定
        alert_level = None
        if risk_score >= self.alert_thresholds['critical_risk']:
            alert_level = AlertLevel.CRITICAL
        elif risk_score >= self.alert_thresholds['high_risk']:
            alert_level = AlertLevel.HIGH
        elif risk_score >= self.alert_thresholds['medium_risk']:
            alert_level = AlertLevel.MEDIUM

        if alert_level:
            await self.alert_manager.create_alert(
                title=f"リスク評価アラート: {summary.risk_category.upper()}",
                message=f"取引ID: {summary.request_id}\n"
                       f"リスクスコア: {risk_score:.3f}\n"
                       f"推定損失: ¥{summary.estimated_loss_potential:,.0f}\n"
                       f"主要要因: {', '.join(summary.key_risk_factors[:3])}",
                level=alert_level,
                source="RiskCoordinator",
                metadata={
                    'request_id': summary.request_id,
                    'risk_score': risk_score,
                    'risk_category': summary.risk_category,
                    'analysis_methods': summary.analysis_methods
                }
            )

            logger.info(f"リスクアラート送信: {alert_level} - {summary.request_id}")

    def _create_error_summary(
        self,
        request_id: str,
        error_message: str,
        transaction_data: Dict[str, Any],
        start_time: float
    ) -> RiskAssessmentSummary:
        """エラー時サマリー作成"""

        return RiskAssessmentSummary(
            request_id=request_id,
            overall_risk_score=0.5,  # 中程度リスクとして扱う
            risk_category='medium',
            confidence_score=0.3,
            analysis_methods=['error_fallback'],
            key_risk_factors=['分析システムエラー'],
            recommendations=['システム管理者に連絡', '手動確認実施'],
            estimated_loss_potential=float(transaction_data.get('amount', 0)) * 0.05,
            processing_time_total=time.time() - start_time,
            component_results={'error': error_message},
            timestamp=datetime.now()
        )

    def _update_performance_stats(self, summary: RiskAssessmentSummary, success: bool):
        """パフォーマンス統計更新"""

        self.performance_stats['total_assessments'] += 1

        if success:
            self.performance_stats['successful_assessments'] += 1
            self.performance_stats['risk_distribution'][summary.risk_category] += 1

        # 平均処理時間更新
        total = self.performance_stats['total_assessments']
        old_avg = self.performance_stats['avg_processing_time']
        new_time = summary.processing_time_total

        self.performance_stats['avg_processing_time'] = (
            (old_avg * (total - 1) + new_time) / total
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""

        total = self.performance_stats['total_assessments']
        success_rate = (self.performance_stats['successful_assessments'] /
                       max(1, total))

        return {
            'total_assessments': total,
            'success_rate': success_rate,
            'avg_processing_time': self.performance_stats['avg_processing_time'],
            'component_usage': self.performance_stats['component_usage'],
            'risk_distribution': self.performance_stats['risk_distribution'],
            'recent_analyses': len(self.analysis_history),
            'generative_ai_stats': self.generative_ai_engine.get_performance_stats(),
            'fraud_detection_stats': self.fraud_engine.get_stats()
        }

    def get_recent_assessments(self, limit: int = 20) -> List[RiskAssessmentSummary]:
        """最近の評価結果取得"""
        return self.analysis_history[-limit:] if self.analysis_history else []

    async def batch_risk_assessment(
        self,
        transactions: List[Dict[str, Any]],
        concurrent_limit: int = 10
    ) -> List[RiskAssessmentSummary]:
        """バッチリスク評価"""

        logger.info(f"バッチリスク評価開始: {len(transactions)}件")

        semaphore = asyncio.Semaphore(concurrent_limit)

        async def assess_single(transaction):
            async with semaphore:
                return await self.comprehensive_risk_assessment(transaction)

        tasks = [assess_single(tx) for tx in transactions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [r for r in results if not isinstance(r, Exception)]
        error_count = len(results) - len(valid_results)

        logger.info(f"バッチ評価完了: 成功{len(valid_results)}件, エラー{error_count}件")

        return valid_results

# テスト・デモ用関数
async def test_risk_coordinator():
    """リスクコーディネーターテスト"""

    coordinator = RiskAnalysisCoordinator()

    # テスト用取引データ
    test_transaction = {
        'symbol': '7203',
        'type': 'buy',
        'amount': 5000000,  # 500万円
        'timestamp': datetime.now().isoformat(),
        'user_id': 'user_123',
        'account_balance': 10000000,
        'location': 'domestic',
        'device_info': {'type': 'mobile', 'os': 'ios'},
        'history': [],
        'market_conditions': {'volatility': 0.25}
    }

    print("🔍 包括的リスク評価テスト開始...")

    # リスク評価実行
    assessment = await coordinator.comprehensive_risk_assessment(test_transaction)

    print(f"✅ 評価完了: {assessment.request_id}")
    print(f"📊 リスクスコア: {assessment.overall_risk_score:.3f}")
    print(f"⚠️ リスクレベル: {assessment.risk_category}")
    print(f"🎯 信頼度: {assessment.confidence_score:.3f}")
    print(f"⏱️ 処理時間: {assessment.processing_time_total:.3f}秒")
    print(f"🔧 使用手法: {', '.join(assessment.analysis_methods)}")
    print(f"💰 推定損失ポテンシャル: ¥{assessment.estimated_loss_potential:,.0f}")

    # パフォーマンス統計
    stats = coordinator.get_performance_summary()
    print(f"\n📈 システム統計:")
    print(f"  総評価数: {stats['total_assessments']}")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  平均処理時間: {stats['avg_processing_time']:.3f}秒")

if __name__ == "__main__":
    asyncio.run(test_risk_coordinator())
