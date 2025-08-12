#!/usr/bin/env python3
"""
Generative AI Risk Analysis Engine
生成AI統合リスク分析エンジン

GPT-4/Claude統合による次世代金融リスク分析システム
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

# 生成AI統合
try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class RiskAnalysisRequest:
    """リスク分析リクエスト"""

    transaction_id: str
    symbol: str
    transaction_type: str
    amount: float
    timestamp: datetime
    market_data: Dict[str, Any]
    user_profile: Dict[str, Any]
    additional_context: Dict[str, Any] = None


@dataclass
class RiskAnalysisResult:
    """リスク分析結果"""

    request_id: str
    risk_score: float  # 0-1 (1が最高リスク)
    risk_level: str  # "low", "medium", "high", "critical"
    confidence: float  # 0-1 (1が最高信頼度)
    explanation: str  # 生成AI による自然言語説明
    recommendations: List[str]
    risk_factors: Dict[str, float]
    processing_time: float
    ai_models_used: List[str]
    timestamp: datetime


@dataclass
class GenerativeAIConfig:
    """生成AI設定"""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_model_gpt: str = "gpt-4"
    default_model_claude: str = "claude-3-opus-20240229"
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout_seconds: int = 10
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


class GenerativeAIRiskEngine:
    """生成AI統合リスク分析エンジン"""

    def __init__(self, config: Optional[GenerativeAIConfig] = None):
        self.config = config or GenerativeAIConfig()

        # OpenAI クライアント初期化
        if OPENAI_AVAILABLE and self.config.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.config.openai_api_key)
            logger.info("OpenAI GPT-4 クライアント初期化完了")
        else:
            self.openai_client = None
            logger.warning("OpenAI API利用不可")

        # Anthropic クライアント初期化
        if ANTHROPIC_AVAILABLE and self.config.anthropic_api_key:
            self.anthropic_client = AsyncAnthropic(api_key=self.config.anthropic_api_key)
            logger.info("Anthropic Claude クライアント初期化完了")
        else:
            self.anthropic_client = None
            logger.warning("Anthropic API利用不可")

        # キャッシュシステム
        self.analysis_cache: Dict[str, RiskAnalysisResult] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

        # パフォーマンス統計
        self.performance_stats = {
            "total_analyses": 0,
            "avg_processing_time": 0.0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "cache_hits": 0,
            "gpt4_calls": 0,
            "claude_calls": 0,
        }

        logger.info("生成AI統合リスク分析エンジン初期化完了")

    async def analyze_risk_comprehensive(
        self,
        request: RiskAnalysisRequest,
        use_gpt4: bool = True,
        use_claude: bool = True,
        use_ensemble: bool = True,
    ) -> RiskAnalysisResult:
        """包括的リスク分析（生成AI統合）"""

        start_time = time.time()
        request_hash = self._generate_request_hash(request)

        # キャッシュチェック
        if self.config.enable_caching:
            cached_result = self._get_cached_result(request_hash)
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                return cached_result

        try:
            # 並列AI分析実行
            analysis_tasks = []
            ai_models_used = []

            if use_gpt4 and self.openai_client:
                analysis_tasks.append(self._analyze_with_gpt4(request))
                ai_models_used.append("gpt-4")

            if use_claude and self.anthropic_client:
                analysis_tasks.append(self._analyze_with_claude(request))
                ai_models_used.append("claude-3-opus")

            # 基本リスク分析も並列実行
            analysis_tasks.append(self._basic_risk_analysis(request))
            ai_models_used.append("heuristic-analyzer")

            # 並列実行
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # 結果統合
            if use_ensemble and len(analysis_results) > 1:
                final_result = self._ensemble_analysis(
                    request, analysis_results, ai_models_used, start_time
                )
            else:
                # 単一モデル結果使用
                valid_results = [r for r in analysis_results if not isinstance(r, Exception)]
                if valid_results:
                    final_result = valid_results[0]
                else:
                    raise ValueError("すべてのAIモデル分析が失敗しました")

            # キャッシュ保存
            if self.config.enable_caching:
                self._cache_result(request_hash, final_result)

            # 統計更新
            self.performance_stats["total_analyses"] += 1
            self.performance_stats["successful_analyses"] += 1
            self.performance_stats["avg_processing_time"] = (
                self.performance_stats["avg_processing_time"]
                * (self.performance_stats["total_analyses"] - 1)
                + (time.time() - start_time)
            ) / self.performance_stats["total_analyses"]

            return final_result

        except Exception as e:
            self.performance_stats["total_analyses"] += 1
            self.performance_stats["failed_analyses"] += 1
            logger.error(f"リスク分析エラー: {e}")

            # フォールバック分析
            return await self._fallback_analysis(request, start_time)

    async def _analyze_with_gpt4(self, request: RiskAnalysisRequest) -> Dict[str, Any]:
        """GPT-4による高度リスク分析"""

        self.performance_stats["gpt4_calls"] += 1

        # プロンプト構築
        system_prompt = """あなたは世界最高レベルの金融リスク分析専門家です。
以下の取引データを分析し、リスクレベルと詳細な説明を提供してください。

出力フォーマット:
{
  "risk_score": 0.0-1.0の数値,
  "risk_level": "low/medium/high/critical",
  "explanation": "詳細な日本語説明",
  "key_risk_factors": ["要因1", "要因2", "要因3"],
  "recommendations": ["推奨事項1", "推奨事項2"]
}"""

        user_prompt = f"""
取引情報:
- 銘柄: {request.symbol}
- 取引タイプ: {request.transaction_type}
- 金額: {request.amount:,.0f}円
- タイムスタンプ: {request.timestamp}

市場データ:
{json.dumps(request.market_data, ensure_ascii=False, indent=2)}

ユーザープロファイル:
{json.dumps(request.user_profile, ensure_ascii=False, indent=2)}

この取引のリスクを包括的に分析してください。"""

        try:
            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model=self.config.default_model_gpt,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ),
                timeout=self.config.timeout_seconds,
            )

            # JSON解析
            content = response.choices[0].message.content
            analysis = json.loads(content)

            return {
                "source": "gpt-4",
                "risk_score": float(analysis.get("risk_score", 0.5)),
                "risk_level": analysis.get("risk_level", "medium"),
                "explanation": analysis.get("explanation", ""),
                "recommendations": analysis.get("recommendations", []),
                "risk_factors": analysis.get("key_risk_factors", []),
                "raw_response": content,
            }

        except asyncio.TimeoutError:
            logger.error("GPT-4 API タイムアウト")
            return self._create_fallback_response("gpt-4", "API timeout")
        except json.JSONDecodeError as e:
            logger.error(f"GPT-4 レスポンス解析エラー: {e}")
            return self._create_fallback_response("gpt-4", f"JSON parse error: {e}")
        except Exception as e:
            logger.error(f"GPT-4 分析エラー: {e}")
            return self._create_fallback_response("gpt-4", str(e))

    async def _analyze_with_claude(self, request: RiskAnalysisRequest) -> Dict[str, Any]:
        """Claudeによる高度リスク分析"""

        self.performance_stats["claude_calls"] += 1

        prompt = f"""あなたは金融リスク管理の専門家です。以下の取引を分析してください。

取引データ:
- ID: {request.transaction_id}
- 銘柄: {request.symbol}
- 種類: {request.transaction_type}
- 金額: {request.amount:,.0f}円
- 時刻: {request.timestamp}

市場状況: {json.dumps(request.market_data, ensure_ascii=False)}
ユーザー情報: {json.dumps(request.user_profile, ensure_ascii=False)}

以下のJSON形式で回答してください:
{{
  "risk_score": 0.0-1.0の数値,
  "risk_level": "low/medium/high/critical",
  "explanation": "詳細な説明",
  "recommendations": ["推奨事項のリスト"],
  "confidence": 0.0-1.0の信頼度
}}"""

        try:
            response = await asyncio.wait_for(
                self.anthropic_client.messages.create(
                    model=self.config.default_model_claude,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                ),
                timeout=self.config.timeout_seconds,
            )

            content = response.content[0].text
            analysis = json.loads(content)

            return {
                "source": "claude-3-opus",
                "risk_score": float(analysis.get("risk_score", 0.5)),
                "risk_level": analysis.get("risk_level", "medium"),
                "explanation": analysis.get("explanation", ""),
                "recommendations": analysis.get("recommendations", []),
                "confidence": float(analysis.get("confidence", 0.7)),
                "raw_response": content,
            }

        except Exception as e:
            logger.error(f"Claude分析エラー: {e}")
            return self._create_fallback_response("claude-3-opus", str(e))

    async def _basic_risk_analysis(self, request: RiskAnalysisRequest) -> Dict[str, Any]:
        """基本的なヒューリスティックリスク分析"""

        risk_score = 0.0
        risk_factors = []

        # 金額ベースリスク
        if request.amount > 10000000:  # 1000万円以上
            risk_score += 0.3
            risk_factors.append("高額取引")
        elif request.amount > 1000000:  # 100万円以上
            risk_score += 0.1
            risk_factors.append("中額取引")

        # 時間帯リスク
        hour = request.timestamp.hour
        if hour < 9 or hour > 15:  # 取引時間外
            risk_score += 0.2
            risk_factors.append("時間外取引")

        # 市場データリスク
        if "volatility" in request.market_data:
            volatility = request.market_data["volatility"]
            if volatility > 0.3:
                risk_score += 0.2
                risk_factors.append("高ボラティリティ")

        # ユーザープロファイルリスク
        if "risk_tolerance" in request.user_profile:
            risk_tolerance = request.user_profile["risk_tolerance"]
            if risk_tolerance == "conservative" and request.amount > 500000:
                risk_score += 0.15
                risk_factors.append("リスク許容度不整合")

        # 正規化
        risk_score = min(1.0, max(0.0, risk_score))

        # リスクレベル決定
        if risk_score >= 0.8:
            risk_level = "critical"
        elif risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "source": "heuristic-analyzer",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "explanation": f"基本分析によるリスクスコア: {risk_score:.2f}",
            "recommendations": self._generate_basic_recommendations(risk_score),
            "risk_factors": risk_factors,
            "confidence": 0.8,
        }

    def _ensemble_analysis(
        self,
        request: RiskAnalysisRequest,
        results: List[Dict[str, Any]],
        models_used: List[str],
        start_time: float,
    ) -> RiskAnalysisResult:
        """アンサンブル分析結果統合"""

        valid_results = [r for r in results if isinstance(r, dict) and "risk_score" in r]

        if not valid_results:
            raise ValueError("有効な分析結果がありません")

        # 重み付き平均（GPT-4とClaudeを高重視）
        weights = {"gpt-4": 0.4, "claude-3-opus": 0.4, "heuristic-analyzer": 0.2}

        weighted_risk_score = 0.0
        total_weight = 0.0
        all_recommendations = []
        all_risk_factors = []
        explanations = []
        confidences = []

        for result in valid_results:
            source = result.get("source", "unknown")
            weight = weights.get(source, 0.1)

            weighted_risk_score += result.get("risk_score", 0.5) * weight
            total_weight += weight

            all_recommendations.extend(result.get("recommendations", []))
            all_risk_factors.extend(result.get("risk_factors", []))
            explanations.append(f"[{source}] {result.get('explanation', '')}")
            confidences.append(result.get("confidence", 0.5))

        # 正規化
        if total_weight > 0:
            final_risk_score = weighted_risk_score / total_weight
        else:
            final_risk_score = 0.5

        # リスクレベル決定
        if final_risk_score >= 0.8:
            risk_level = "critical"
        elif final_risk_score >= 0.6:
            risk_level = "high"
        elif final_risk_score >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        # 統合説明文生成
        combined_explanation = "\n\n".join(explanations)

        return RiskAnalysisResult(
            request_id=request.transaction_id,
            risk_score=final_risk_score,
            risk_level=risk_level,
            confidence=np.mean(confidences),
            explanation=combined_explanation,
            recommendations=list(set(all_recommendations)),  # 重複除去
            risk_factors={f"factor_{i}": 1.0 for i, f in enumerate(set(all_risk_factors))},
            processing_time=time.time() - start_time,
            ai_models_used=models_used,
            timestamp=datetime.now(),
        )

    async def _fallback_analysis(
        self, request: RiskAnalysisRequest, start_time: float
    ) -> RiskAnalysisResult:
        """フォールバック分析"""

        logger.warning("フォールバック分析を実行")

        # 基本分析のみ実行
        basic_result = await self._basic_risk_analysis(request)

        return RiskAnalysisResult(
            request_id=request.transaction_id,
            risk_score=basic_result["risk_score"],
            risk_level=basic_result["risk_level"],
            confidence=0.6,  # フォールバック時は信頼度低下
            explanation=f"フォールバック分析: {basic_result['explanation']}",
            recommendations=basic_result.get("recommendations", ["詳細分析を再実行してください"]),
            risk_factors={"fallback_analysis": 1.0},
            processing_time=time.time() - start_time,
            ai_models_used=["fallback-analyzer"],
            timestamp=datetime.now(),
        )

    def _create_fallback_response(self, source: str, error: str) -> Dict[str, Any]:
        """フォールバックレスポンス作成"""
        return {
            "source": source,
            "risk_score": 0.5,
            "risk_level": "medium",
            "explanation": f"分析エラー ({source}): {error}",
            "recommendations": ["システム管理者に連絡してください"],
            "confidence": 0.3,
            "error": error,
        }

    def _generate_basic_recommendations(self, risk_score: float) -> List[str]:
        """基本推奨事項生成"""
        if risk_score >= 0.8:
            return [
                "取引を停止してください",
                "リスク管理責任者に相談してください",
                "詳細なデューデリジェンスを実施してください",
            ]
        elif risk_score >= 0.6:
            return [
                "取引前に追加確認を行ってください",
                "取引金額の見直しを検討してください",
                "市場状況を再確認してください",
            ]
        elif risk_score >= 0.3:
            return [
                "定期的な監視を継続してください",
                "ポジションサイズに注意してください",
            ]
        else:
            return [
                "通常どおり取引を継続できます",
                "引き続き市場動向を監視してください",
            ]

    def _generate_request_hash(self, request: RiskAnalysisRequest) -> str:
        """リクエストハッシュ生成（キャッシュ用）"""
        import hashlib

        hash_data = (
            f"{request.transaction_id}-{request.symbol}-{request.amount}-{request.timestamp}"
        )
        return hashlib.md5(hash_data.encode()).hexdigest()

    def _get_cached_result(self, request_hash: str) -> Optional[RiskAnalysisResult]:
        """キャッシュ結果取得"""
        if request_hash not in self.analysis_cache:
            return None

        # TTL チェック
        cache_time = self.cache_timestamps.get(request_hash)
        if not cache_time:
            return None

        if datetime.now() - cache_time > timedelta(seconds=self.config.cache_ttl_seconds):
            # 期限切れキャッシュ削除
            del self.analysis_cache[request_hash]
            del self.cache_timestamps[request_hash]
            return None

        return self.analysis_cache[request_hash]

    def _cache_result(self, request_hash: str, result: RiskAnalysisResult):
        """結果キャッシュ保存"""
        self.analysis_cache[request_hash] = result
        self.cache_timestamps[request_hash] = datetime.now()

        # キャッシュサイズ制限（1000件）
        if len(self.analysis_cache) > 1000:
            # 最も古いキャッシュを削除
            oldest_key = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
            del self.analysis_cache[oldest_key]
            del self.cache_timestamps[oldest_key]

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        return {
            **self.performance_stats,
            "cache_size": len(self.analysis_cache),
            "success_rate": (
                self.performance_stats["successful_analyses"]
                / max(1, self.performance_stats["total_analyses"])
            ),
            "models_available": {
                "gpt4": self.openai_client is not None,
                "claude": self.anthropic_client is not None,
            },
        }

    def clear_cache(self):
        """キャッシュクリア"""
        self.analysis_cache.clear()
        self.cache_timestamps.clear()
        logger.info("分析キャッシュをクリアしました")


# 使用例とテスト
async def test_generative_ai_risk_engine():
    """生成AI リスク分析エンジンテスト"""

    # 設定（実際のAPIキーは環境変数から取得）
    config = GenerativeAIConfig(
        openai_api_key="dummy_key",  # 実際は os.getenv("OPENAI_API_KEY")
        anthropic_api_key="dummy_key",  # 実際は os.getenv("ANTHROPIC_API_KEY")
        temperature=0.3,
        max_tokens=800,
    )

    engine = GenerativeAIRiskEngine(config)

    # テスト用リクエスト
    test_request = RiskAnalysisRequest(
        transaction_id="TEST_001",
        symbol="7203",  # トヨタ
        transaction_type="buy",
        amount=5000000,  # 500万円
        timestamp=datetime.now(),
        market_data={
            "current_price": 2500,
            "volatility": 0.25,
            "volume": 1000000,
            "trend": "upward",
        },
        user_profile={
            "risk_tolerance": "moderate",
            "experience_level": "intermediate",
            "portfolio_value": 10000000,
        },
    )

    # 分析実行
    print("🤖 生成AI統合リスク分析開始...")

    result = await engine.analyze_risk_comprehensive(
        test_request,
        use_gpt4=False,  # テスト時はダミーキーなのでFalse
        use_claude=False,  # テスト時はダミーキーなのでFalse
        use_ensemble=True,
    )

    print("✅ 分析完了!")
    print(f"🎯 リスクスコア: {result.risk_score:.2f}")
    print(f"⚠️ リスクレベル: {result.risk_level}")
    print(f"🕐 処理時間: {result.processing_time:.2f}秒")
    print(f"🧠 使用モデル: {result.ai_models_used}")
    print(f"📊 説明: {result.explanation[:100]}...")

    # パフォーマンス統計
    stats = engine.get_performance_stats()
    print(f"📈 統計: {stats}")


if __name__ == "__main__":
    asyncio.run(test_generative_ai_risk_engine())
