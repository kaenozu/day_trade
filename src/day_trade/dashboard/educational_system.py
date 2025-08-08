#!/usr/bin/env python3
"""
教育・学習支援システム
Issue #319: 分析ダッシュボード強化

投資戦略解説・用語集・学習機能の実装
"""

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class EducationalSystem:
    """教育・学習支援システム"""

    def __init__(self, content_dir: str = "educational_content"):
        """
        初期化

        Args:
            content_dir: 教育コンテンツディレクトリ
        """
        self.content_dir = Path(content_dir)
        self.content_dir.mkdir(exist_ok=True)

        # コンテンツ初期化
        self.glossary = self._initialize_glossary()
        self.strategies = self._initialize_strategies()
        self.case_studies = self._initialize_case_studies()
        self.learning_modules = self._initialize_learning_modules()

        # ユーザー学習進捗管理
        self.user_progress = {}

        logger.info("教育・学習支援システム初期化完了")

    def _initialize_glossary(self) -> Dict[str, Dict[str, Any]]:
        """用語集初期化"""
        return {
            # 基本用語
            "sharpe_ratio": {
                "term": "シャープレシオ",
                "category": "リスク指標",
                "definition": "リスクに対するリターンの効率性を測る指標。(ポートフォリオリターン - 無リスク金利) / 標準偏差で計算される。",
                "formula": "(Rp - Rf) / σp",
                "example": "シャープレシオが1.0なら、1単位のリスクに対して1単位の超過リターンを得ていることを意味する。",
                "related_terms": ["volatility", "risk_free_rate", "excess_return"],
                "difficulty": "初級",
                "importance": "高",
            },
            "volatility": {
                "term": "ボラティリティ",
                "category": "リスク指標",
                "definition": "投資対象の価格変動の激しさを表す指標。通常、リターンの標準偏差で測定される。",
                "formula": "σ = √(Σ(Ri - R̄)² / n)",
                "example": "年率ボラティリティが20%の株式は、68%の確率で年間リターンが平均±20%の範囲に収まる。",
                "related_terms": ["standard_deviation", "risk", "variance"],
                "difficulty": "初級",
                "importance": "高",
            },
            "max_drawdown": {
                "term": "最大ドローダウン",
                "category": "リスク指標",
                "definition": "投資期間中の最高値から最安値までの最大下落率。リスク管理の重要な指標。",
                "formula": "MDD = (Trough Value - Peak Value) / Peak Value",
                "example": "100万円が80万円まで下落した場合、最大ドローダウンは-20%。",
                "related_terms": ["peak_to_trough", "risk_management", "downside_risk"],
                "difficulty": "初級",
                "importance": "高",
            },
            "var": {
                "term": "VaR（Value at Risk）",
                "category": "リスク指標",
                "definition": "一定期間・信頼水準において予想される最大損失額。リスク管理の標準的指標。",
                "formula": "VaR = μ - z × σ （正規分布仮定時）",
                "example": "1日間、95%信頼水準でのVaRが200万円なら、95%の確率で1日の損失は200万円以下。",
                "related_terms": ["cvar", "confidence_level", "tail_risk"],
                "difficulty": "中級",
                "importance": "高",
            },
            "beta": {
                "term": "ベータ",
                "category": "市場指標",
                "definition": "市場全体に対する個別証券の感応度を示す指標。市場のβ=1.0を基準とする。",
                "formula": "β = Cov(Ri, Rm) / Var(Rm)",
                "example": "β=1.2の銘柄は、市場が10%上昇時に12%上昇、10%下落時に12%下落する傾向。",
                "related_terms": ["alpha", "systematic_risk", "capm"],
                "difficulty": "中級",
                "importance": "中",
            },
            "alpha": {
                "term": "アルファ",
                "category": "パフォーマンス指標",
                "definition": "市場リターンを上回る超過リターン。運用スキルを示す指標。",
                "formula": "α = Rp - [Rf + β(Rm - Rf)]",
                "example": "年率アルファが3%なら、市場リスクを調整した後で年3%の超過リターンを獲得。",
                "related_terms": ["beta", "capm", "active_return"],
                "difficulty": "中級",
                "importance": "中",
            },
            "sortino_ratio": {
                "term": "ソルティノレシオ",
                "category": "リスク指標",
                "definition": "下方偏差のみを考慮したリスク調整後リターン指標。シャープレシオの改良版。",
                "formula": "(Rp - Rf) / Downside Deviation",
                "example": "上昇時のボラティリティは無視し、下落リスクのみでリターンを評価。",
                "related_terms": ["sharpe_ratio", "downside_risk", "semi_variance"],
                "difficulty": "中級",
                "importance": "中",
            },
            "modern_portfolio_theory": {
                "term": "現代ポートフォリオ理論",
                "category": "理論",
                "definition": "ハリー・マーコウィッツが提唱した、分散投資によるリスク軽減を数学的に示した理論。",
                "formula": "効率的フロンティア上での最適化",
                "example": "異なる相関を持つ資産を組み合わせることで、同リターンでより低リスクを実現。",
                "related_terms": [
                    "efficient_frontier",
                    "diversification",
                    "correlation",
                ],
                "difficulty": "上級",
                "importance": "高",
            },
            "black_litterman": {
                "term": "Black-Littermanモデル",
                "category": "理論",
                "definition": "投資家の主観的見解を市場均衡リターンに反映させるポートフォリオ最適化手法。",
                "formula": "μBL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹μ + P'Ω⁻¹Q]",
                "example": "「日本株が米国株をアウトパフォーム」という見解を数量化して最適化に反映。",
                "related_terms": [
                    "portfolio_optimization",
                    "bayesian",
                    "equilibrium_return",
                ],
                "difficulty": "上級",
                "importance": "中",
            },
        }

    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """投資戦略解説初期化"""
        return {
            "buy_and_hold": {
                "name": "バイ・アンド・ホールド戦略",
                "category": "長期投資",
                "description": "優良な投資対象を長期間保有し続ける最もシンプルな投資戦略",
                "advantages": [
                    "取引コストが最小限",
                    "税務効率が良い",
                    "時間分散効果",
                    "心理的ストレスが少ない",
                ],
                "disadvantages": [
                    "短期的な下落に対処できない",
                    "市場変化への適応性に欠ける",
                    "機会損失の可能性",
                ],
                "suitable_for": ["初心者", "長期投資家", "忙しい投資家"],
                "implementation": {
                    "steps": [
                        "投資対象の選定（インデックスファンド推奨）",
                        "定期的な積立投資設定",
                        "年1-2回のリバランシング",
                        "長期保有の継続",
                    ],
                    "tools": ["インデックスファンド", "ETF", "個別優良株"],
                },
                "historical_performance": "S&P500で過去50年平均年率約10%",
                "risk_level": "中リスク",
                "time_horizon": "10年以上",
            },
            "momentum": {
                "name": "モメンタム戦略",
                "category": "テクニカル分析",
                "description": "価格の上昇傾向（モメンタム）が継続することを前提とした投資戦略",
                "advantages": [
                    "トレンドフォロー効果",
                    "強い銘柄への集中投資",
                    "市場の非効率性を活用",
                ],
                "disadvantages": [
                    "急激な反転リスク",
                    "高い取引コスト",
                    "ボラティリティが大きい",
                ],
                "suitable_for": [
                    "中級者以上",
                    "アクティブ投資家",
                    "リスク許容度の高い投資家",
                ],
                "implementation": {
                    "steps": [
                        "モメンタム指標の設定（3-12ヶ月リターン）",
                        "スクリーニング基準の決定",
                        "定期的なポートフォリオ入替",
                        "リスク管理ルールの設定",
                    ],
                    "indicators": ["相対強度", "移動平均", "価格モメンタム"],
                },
                "historical_performance": "年率12-15%程度（高ボラティリティ）",
                "risk_level": "高リスク",
                "time_horizon": "1-5年",
            },
            "mean_reversion": {
                "name": "平均回帰戦略",
                "category": "統計的アービトラージ",
                "description": "価格が平均値から大きく乖離した後、平均に回帰する性質を利用する戦略",
                "advantages": [
                    "統計的根拠のある手法",
                    "市場の過度な反応を収益機会に",
                    "相対的に安定したリターン",
                ],
                "disadvantages": [
                    "トレンド相場で機能しない",
                    "タイミングが重要",
                    "長期間の含み損の可能性",
                ],
                "suitable_for": ["上級者", "統計知識のある投資家", "短期-中期投資家"],
                "implementation": {
                    "steps": [
                        "平均回帰指標の設定（ボリンジャーバンド等）",
                        "エントリー・エグジット条件の決定",
                        "リスク管理の徹底",
                        "バックテスト検証",
                    ],
                    "indicators": ["ボリンジャーバンド", "RSI", "統計的乖離度"],
                },
                "historical_performance": "年率8-12%程度",
                "risk_level": "中-高リスク",
                "time_horizon": "数ヶ月-2年",
            },
            "sector_rotation": {
                "name": "セクターローテーション戦略",
                "category": "マクロ戦略",
                "description": "経済サイクルに応じて有望なセクターに資金を配分する戦略",
                "advantages": [
                    "経済サイクルを活用",
                    "分散投資効果",
                    "マクロ経済の理解向上",
                ],
                "disadvantages": [
                    "タイミングの難しさ",
                    "経済予測の困難性",
                    "頻繁な銘柄入替コスト",
                ],
                "suitable_for": ["中級者以上", "マクロ経済に興味のある投資家"],
                "implementation": {
                    "steps": [
                        "経済サイクルの判断",
                        "各フェーズでの有望セクター選定",
                        "セクターETF/個別株での投資実行",
                        "経済指標モニタリング",
                    ],
                    "economic_phases": ["回復期", "拡大期", "減速期", "後退期"],
                },
                "historical_performance": "市場平均+2-5%程度",
                "risk_level": "中リスク",
                "time_horizon": "1-3年",
            },
        }

    def _initialize_case_studies(self) -> Dict[str, Dict[str, Any]]:
        """過去事例分析初期化"""
        return {
            "dot_com_bubble": {
                "title": "ITバブル崩壊 (2000-2001)",
                "period": "1995-2003",
                "description": "インターネット関連企業の過度な期待により発生したバブルとその崩壊",
                "key_events": [
                    "1995-1999: インターネット株の急騰",
                    "2000年3月: NASDAQ最高値更新",
                    "2000年4月-2001年: 急激な下落開始",
                    "2002年10月: NASDAQ底値（最高値から78%下落）",
                ],
                "lessons": [
                    "過度な期待は危険な投機を生む",
                    "基本的価値（ファンダメンタルズ）の重要性",
                    "分散投資の効果",
                    "長期視点の重要性",
                ],
                "strategies_that_worked": [
                    "バリュー株投資",
                    "分散投資",
                    "定期的なリバランシング",
                ],
                "strategies_that_failed": [
                    "成長株への集中投資",
                    "モメンタム戦略",
                    "レバレッジ投資",
                ],
                "portfolio_impact": {
                    "technology_heavy": "-70% to -80%",
                    "diversified": "-30% to -50%",
                    "value_focused": "-10% to -30%",
                },
            },
            "financial_crisis_2008": {
                "title": "リーマンショック (2008)",
                "period": "2007-2009",
                "description": "サブプライムローン問題に端を発した世界的金融危機",
                "key_events": [
                    "2007年: サブプライム問題表面化",
                    "2008年9月: リーマンブラザーズ破綻",
                    "2008年10月: 株価大暴落",
                    "2009年3月: 市場底値形成",
                ],
                "lessons": [
                    "システミックリスクの脅威",
                    "流動性危機の深刻さ",
                    "相関の上昇（分散投資の限界）",
                    "政府・中央銀行の役割",
                ],
                "strategies_that_worked": [
                    "現金・国債保有",
                    "質への逃避",
                    "底値でのナンピン買い",
                ],
                "strategies_that_failed": [
                    "レバレッジ戦略",
                    "金融セクター集中投資",
                    "短期売買",
                ],
                "portfolio_impact": {
                    "stocks": "-50% to -60%",
                    "bonds": "+5% to +20%",
                    "real_estate": "-40% to -60%",
                    "commodities": "-30% to -50%",
                },
            },
            "covid19_crash_recovery": {
                "title": "コロナショック (2020)",
                "period": "2020年2月-2021年",
                "description": "COVID-19パンデミックによる経済活動停止と史上最速の回復",
                "key_events": [
                    "2020年2月: 感染拡大開始",
                    "2020年3月: 株価急落（-35%）",
                    "2020年4月: 底値形成",
                    "2020年4月-2021年: 急速な回復",
                ],
                "lessons": [
                    "政策対応の重要性",
                    "テクノロジー株の強さ",
                    "業界格差の拡大",
                    "V字回復の可能性",
                ],
                "strategies_that_worked": [
                    "テクノロジー株投資",
                    "成長株戦略",
                    "押し目買い",
                ],
                "strategies_that_failed": [
                    "バリュー株投資",
                    "航空・観光株",
                    "エネルギー株",
                ],
                "portfolio_impact": {
                    "technology": "+80% to +120%",
                    "healthcare": "+20% to +40%",
                    "travel_leisure": "-60% to -80%",
                    "energy": "-50% to -70%",
                },
            },
        }

    def _initialize_learning_modules(self) -> Dict[str, Dict[str, Any]]:
        """学習モジュール初期化"""
        return {
            "basic_concepts": {
                "title": "投資の基本概念",
                "level": "初級",
                "duration": "2-3時間",
                "modules": [
                    {
                        "name": "リスクとリターン",
                        "topics": [
                            "リスクの種類",
                            "リスクとリターンの関係",
                            "分散投資",
                        ],
                        "exercises": ["ポートフォリオのリスク計算", "相関係数の理解"],
                    },
                    {
                        "name": "投資商品の理解",
                        "topics": ["株式", "債券", "投資信託", "ETF"],
                        "exercises": ["商品比較表作成", "手数料計算"],
                    },
                    {
                        "name": "パフォーマンス評価",
                        "topics": [
                            "リターン計算",
                            "ベンチマーク比較",
                            "リスク調整後リターン",
                        ],
                        "exercises": ["シャープレシオ計算", "ベンチマーク分析"],
                    },
                ],
            },
            "portfolio_theory": {
                "title": "ポートフォリオ理論",
                "level": "中級",
                "duration": "4-5時間",
                "modules": [
                    {
                        "name": "現代ポートフォリオ理論",
                        "topics": [
                            "マーコウィッツ理論",
                            "効率的フロンティア",
                            "最適化",
                        ],
                        "exercises": [
                            "効率的フロンティア作成",
                            "最適ポートフォリオ計算",
                        ],
                    },
                    {
                        "name": "CAPM理論",
                        "topics": ["βの概念", "αの計算", "証券市場線"],
                        "exercises": ["β計算", "α分析"],
                    },
                    {
                        "name": "実践的ポートフォリオ構築",
                        "topics": ["リバランシング", "資産配分", "実装上の注意点"],
                        "exercises": ["リバランシング計算", "取引コスト考慮"],
                    },
                ],
            },
            "risk_management": {
                "title": "リスク管理",
                "level": "上級",
                "duration": "5-6時間",
                "modules": [
                    {
                        "name": "VaRとリスク指標",
                        "topics": ["VaR計算", "ストレステスト", "相関リスク"],
                        "exercises": ["VaR計算", "ストレステスト実施"],
                    },
                    {
                        "name": "ヘッジ戦略",
                        "topics": ["オプション活用", "先物活用", "通貨ヘッジ"],
                        "exercises": ["プロテクティブプット", "カバードコール"],
                    },
                ],
            },
        }

    def get_term_explanation(self, term: str) -> Optional[Dict[str, Any]]:
        """用語解説取得"""
        term_key = term.lower().replace(" ", "_")

        if term_key in self.glossary:
            explanation = self.glossary[term_key].copy()
            explanation["search_timestamp"] = datetime.now().isoformat()
            return explanation
        else:
            # 部分一致検索
            matches = []
            for key, value in self.glossary.items():
                if term.lower() in value["term"].lower() or term.lower() in key:
                    matches.append(value)

            if matches:
                return {
                    "partial_matches": matches,
                    "search_term": term,
                    "search_timestamp": datetime.now().isoformat(),
                }

        return None

    def search_glossary(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """用語集検索"""
        results = []
        query_lower = query.lower()

        for key, term_data in self.glossary.items():
            # カテゴリフィルター
            if category and term_data.get("category") != category:
                continue

            # 検索マッチング
            if (
                query_lower in term_data["term"].lower()
                or query_lower in term_data["definition"].lower()
                or query_lower in key
            ):
                result = term_data.copy()
                result["relevance_score"] = self._calculate_relevance(
                    query_lower, term_data
                )
                results.append(result)

        # 関連度順でソート
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    def get_strategy_guide(self, strategy: str) -> Optional[Dict[str, Any]]:
        """投資戦略ガイド取得"""
        if strategy in self.strategies:
            guide = self.strategies[strategy].copy()
            guide["access_timestamp"] = datetime.now().isoformat()
            return guide
        return None

    def get_case_study(self, case: str) -> Optional[Dict[str, Any]]:
        """過去事例分析取得"""
        if case in self.case_studies:
            study = self.case_studies[case].copy()
            study["access_timestamp"] = datetime.now().isoformat()
            return study
        return None

    def get_learning_module(self, module: str) -> Optional[Dict[str, Any]]:
        """学習モジュール取得"""
        if module in self.learning_modules:
            learning_module = self.learning_modules[module].copy()
            learning_module["access_timestamp"] = datetime.now().isoformat()
            return learning_module
        return None

    def get_quiz_questions(
        self, topic: str = None, difficulty: str = None, count: int = 5
    ) -> List[Dict[str, Any]]:
        """クイズ問題生成"""
        questions = []

        # 用語集ベースの問題生成
        eligible_terms = list(self.glossary.values())

        if difficulty:
            eligible_terms = [
                t for t in eligible_terms if t.get("difficulty") == difficulty
            ]

        # ランダムに選択
        selected_terms = random.sample(eligible_terms, min(count, len(eligible_terms)))

        for i, term in enumerate(selected_terms):
            question = self._generate_question_from_term(term, i + 1)
            questions.append(question)

        return questions

    def _generate_question_from_term(
        self, term: Dict[str, Any], question_num: int
    ) -> Dict[str, Any]:
        """用語から問題を生成"""
        question_types = ["definition", "formula", "application"]
        question_type = random.choice(question_types)

        if question_type == "definition":
            return {
                "id": question_num,
                "type": "multiple_choice",
                "question": f"「{term['term']}」の正しい定義はどれですか？",
                "options": self._generate_definition_options(term),
                "correct_answer": 0,  # 最初の選択肢が正解
                "explanation": term["definition"],
                "difficulty": term.get("difficulty", "中級"),
                "category": term.get("category", "一般"),
            }
        elif question_type == "formula" and "formula" in term:
            return {
                "id": question_num,
                "type": "multiple_choice",
                "question": f"「{term['term']}」の計算式として正しいものはどれですか？",
                "options": [
                    term["formula"],
                    "R² = α + β",
                    "σ = √(μ)",
                    "P/E = Price × Earnings",
                ],
                "correct_answer": 0,
                "explanation": f"正しい計算式は {term['formula']} です。",
                "difficulty": term.get("difficulty", "中級"),
                "category": term.get("category", "一般"),
            }
        else:
            return {
                "id": question_num,
                "type": "true_false",
                "question": f"「{term['term']}」について、次の説明は正しいですか？\n{term.get('example', term['definition'])}",
                "correct_answer": True,
                "explanation": term["definition"],
                "difficulty": term.get("difficulty", "中級"),
                "category": term.get("category", "一般"),
            }

    def _generate_definition_options(self, correct_term: Dict[str, Any]) -> List[str]:
        """定義選択肢生成"""
        options = [correct_term["definition"]]

        # 他の用語の定義をダミー選択肢として使用
        other_terms = [t for t in self.glossary.values() if t != correct_term]
        dummy_options = random.sample(other_terms, min(3, len(other_terms)))

        for dummy in dummy_options:
            options.append(dummy["definition"])

        return options

    def _calculate_relevance(self, query: str, term_data: Dict[str, Any]) -> float:
        """検索関連度計算"""
        score = 0.0

        # 完全一致
        if query == term_data["term"].lower():
            score += 10.0

        # タイトル部分一致
        if query in term_data["term"].lower():
            score += 5.0

        # 定義内一致
        if query in term_data["definition"].lower():
            score += 2.0

        # 重要度ボーナス
        if term_data.get("importance") == "高":
            score += 1.0

        return score

    def get_categories(self) -> List[str]:
        """利用可能カテゴリ一覧"""
        categories = set()
        for term in self.glossary.values():
            if "category" in term:
                categories.add(term["category"])
        return sorted(list(categories))

    def get_learning_progress(self, user_id: str) -> Dict[str, Any]:
        """学習進捗取得"""
        return self.user_progress.get(
            user_id,
            {
                "completed_modules": [],
                "quiz_scores": {},
                "study_time": 0,
                "last_activity": None,
            },
        )

    def update_learning_progress(
        self,
        user_id: str,
        module: str = None,
        quiz_score: Dict[str, Any] = None,
        study_time: int = None,
    ) -> Dict[str, Any]:
        """学習進捗更新"""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {
                "completed_modules": [],
                "quiz_scores": {},
                "study_time": 0,
                "last_activity": None,
            }

        progress = self.user_progress[user_id]

        if module and module not in progress["completed_modules"]:
            progress["completed_modules"].append(module)

        if quiz_score:
            progress["quiz_scores"][quiz_score["topic"]] = quiz_score

        if study_time:
            progress["study_time"] += study_time

        progress["last_activity"] = datetime.now().isoformat()

        return progress

    def get_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """学習推奨取得"""
        progress = self.get_learning_progress(user_id)
        recommendations = []

        # 未完了モジュール推奨
        completed = set(progress["completed_modules"])
        all_modules = set(self.learning_modules.keys())
        remaining = all_modules - completed

        for module_key in remaining:
            module = self.learning_modules[module_key]
            recommendations.append(
                {
                    "type": "learning_module",
                    "title": module["title"],
                    "level": module["level"],
                    "estimated_time": module["duration"],
                    "reason": "まだ完了していない学習モジュールです",
                }
            )

        # 復習推奨
        if progress["quiz_scores"]:
            low_score_topics = [
                topic
                for topic, score in progress["quiz_scores"].items()
                if score.get("percentage", 100) < 70
            ]

            for topic in low_score_topics:
                recommendations.append(
                    {
                        "type": "review",
                        "title": f"{topic}の復習",
                        "level": "復習",
                        "estimated_time": "30分",
                        "reason": "クイズスコアが70%を下回っています",
                    }
                )

        return recommendations[:5]  # 上位5個を返す


if __name__ == "__main__":
    # テスト実行
    print("教育・学習支援システム テスト")
    print("=" * 60)

    try:
        edu_system = EducationalSystem()

        # 用語検索テスト
        print("\n1. 用語検索テスト")
        term_result = edu_system.get_term_explanation("sharpe_ratio")
        if term_result:
            print(f"✅ 用語取得成功: {term_result['term']}")
            print(f"   定義: {term_result['definition'][:100]}...")

        # カテゴリ検索テスト
        print("\n2. カテゴリ検索テスト")
        risk_terms = edu_system.search_glossary("", category="リスク指標")
        print(f"✅ リスク指標カテゴリ: {len(risk_terms)}個の用語")

        # 戦略ガイドテスト
        print("\n3. 投資戦略ガイドテスト")
        strategy = edu_system.get_strategy_guide("buy_and_hold")
        if strategy:
            print(f"✅ 戦略取得成功: {strategy['name']}")
            print(f"   カテゴリ: {strategy['category']}")
            print(f"   利点数: {len(strategy['advantages'])}")

        # 過去事例テスト
        print("\n4. 過去事例分析テスト")
        case = edu_system.get_case_study("financial_crisis_2008")
        if case:
            print(f"✅ 事例取得成功: {case['title']}")
            print(f"   期間: {case['period']}")
            print(f"   教訓数: {len(case['lessons'])}")

        # クイズ生成テスト
        print("\n5. クイズ生成テスト")
        quiz_questions = edu_system.get_quiz_questions(count=3, difficulty="初級")
        print(f"✅ クイズ生成成功: {len(quiz_questions)}問")
        for i, q in enumerate(quiz_questions, 1):
            print(f"   問題{i}: {q['question'][:50]}...")

        # 学習進捗テスト
        print("\n6. 学習進捗テスト")
        user_id = "test_user_001"
        progress = edu_system.update_learning_progress(
            user_id=user_id,
            module="basic_concepts",
            study_time=120,  # 2時間
        )
        print(f"✅ 学習進捗更新: ユーザー{user_id}")
        print(f"   完了モジュール: {len(progress['completed_modules'])}")
        print(f"   学習時間: {progress['study_time']}分")

        # 推奨システムテスト
        print("\n7. 学習推奨テスト")
        recommendations = edu_system.get_recommendations(user_id)
        print(f"✅ 推奨取得成功: {len(recommendations)}件")
        for rec in recommendations[:2]:
            print(f"   - {rec['title']} ({rec['level']})")

        print("\n✅ 教育・学習支援システム テスト完了！")
        print(f"用語集: {len(edu_system.glossary)}語")
        print(f"投資戦略: {len(edu_system.strategies)}種")
        print(f"過去事例: {len(edu_system.case_studies)}件")
        print(f"学習モジュール: {len(edu_system.learning_modules)}個")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
