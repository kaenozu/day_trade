"""
レース分析システム

レースの特徴、競争力、展開予想を分析
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from collections import defaultdict
from statistics import mean, median, stdev
from dataclasses import dataclass

from ..core.data_models import Race as RaceModel, RaceEntry as RaceEntryModel, STADIUMS
from ..core.stadium_manager import StadiumManager, StadiumCharacteristics
from ..data.database import Database, get_database, Race, RaceEntry, Racer
from .racer_analyzer import RacerAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RaceAnalysis:
    """レース分析結果"""
    race_id: str
    race_info: Dict[str, Any]
    
    # 競争力分析
    competitiveness: str              # "激戦", "混戦", "実力差", "大荒れ"
    upset_probability: float          # 大荒れ確率 (0.0-1.0)
    
    # 注目要素
    key_factors: List[str]            # 注目すべき要因
    race_pattern: str                 # レースパターン
    
    # 選手分析
    favorites: List[int]              # 本命候補（艇番）
    dark_horses: List[int]            # 穴候補（艇番）
    
    # 展開予想
    expected_pattern: str             # 予想展開パターン
    start_prediction: Dict[int, str]  # {艇番: 予想スタート}
    
    # 戦略提案
    betting_strategy: str             # 推奨投票戦略
    confidence: float                 # 分析信頼度


class RaceAnalyzer:
    """レース分析クラス"""
    
    def __init__(self, 
                 database: Optional[Database] = None,
                 racer_analyzer: Optional[RacerAnalyzer] = None):
        """
        初期化
        
        Args:
            database: データベース
            racer_analyzer: 選手分析器
        """
        self.database = database or get_database()
        self.racer_analyzer = racer_analyzer or RacerAnalyzer(database)
        self.stadium_manager = StadiumManager()
    
    def analyze_race(self, race_id: str) -> Optional[RaceAnalysis]:
        """
        レースを総合分析
        
        Args:
            race_id: レースID
            
        Returns:
            レース分析結果
        """
        with self.database.get_session() as session:
            # レース基本情報取得
            race = session.query(Race).filter(Race.id == race_id).first()
            if not race:
                logger.warning(f"レース {race_id} が見つかりません")
                return None
            
            # 出走情報取得
            entries = session.query(RaceEntry).join(Racer).filter(
                RaceEntry.race_id == race_id
            ).all()
            
            if not entries:
                logger.warning(f"レース {race_id} の出走情報がありません")
                return None
            
            # 各種分析実行
            competitiveness_analysis = self._analyze_competitiveness(race, entries)
            factor_analysis = self._analyze_key_factors(race, entries)
            racer_evaluation = self._evaluate_racers(entries)
            pattern_analysis = self._analyze_race_pattern(race, entries)
            strategy_analysis = self._suggest_betting_strategy(race, entries, competitiveness_analysis)
            
            # レース情報整理
            race_info = {
                'date': race.date.isoformat(),
                'stadium_number': race.stadium_number,
                'stadium_name': STADIUMS.get(race.stadium_number, {}).name if STADIUMS.get(race.stadium_number) else f"競技場{race.stadium_number}",
                'race_number': race.race_number,
                'title': race.title,
                'grade': race.grade_number,
                'distance': race.distance,
                'entry_count': len(entries)
            }
            
            return RaceAnalysis(
                race_id=race_id,
                race_info=race_info,
                competitiveness=competitiveness_analysis['level'],
                upset_probability=competitiveness_analysis['upset_prob'],
                key_factors=factor_analysis['factors'],
                race_pattern=pattern_analysis['pattern'],
                favorites=racer_evaluation['favorites'],
                dark_horses=racer_evaluation['dark_horses'],
                expected_pattern=pattern_analysis['expected_pattern'],
                start_prediction=pattern_analysis['start_prediction'],
                betting_strategy=strategy_analysis['strategy'],
                confidence=self._calculate_analysis_confidence(race, entries)
            )
    
    def compare_races(self, race_ids: List[str]) -> Dict[str, Any]:
        """
        複数レースを比較
        
        Args:
            race_ids: レースIDリスト
            
        Returns:
            比較結果
        """
        analyses = {}
        for race_id in race_ids:
            analysis = self.analyze_race(race_id)
            if analysis:
                analyses[race_id] = analysis
        
        if not analyses:
            return {'error': '有効なレースデータがありません'}
        
        # 比較結果生成
        return {
            'race_count': len(analyses),
            'competitiveness_distribution': self._get_distribution([a.competitiveness for a in analyses.values()]),
            'average_upset_probability': mean([a.upset_probability for a in analyses.values()]),
            'most_competitive': max(analyses.items(), key=lambda x: x[1].upset_probability),
            'recommended_focus': self._select_focus_races(analyses),
            'strategy_summary': self._summarize_strategies(analyses)
        }
    
    def find_upset_opportunities(self, 
                               date_filter: Optional[date] = None,
                               stadium_filter: Optional[int] = None,
                               min_upset_prob: float = 0.3) -> List[Dict[str, Any]]:
        """
        大穴のチャンスがあるレースを発見
        
        Args:
            date_filter: 日付フィルター
            stadium_filter: 競技場フィルター
            min_upset_prob: 最小大荒れ確率
            
        Returns:
            大穴チャンスレース一覧
        """
        with self.database.get_session() as session:
            # レース検索
            query = session.query(Race)
            
            if date_filter:
                query = query.filter(Race.date == date_filter)
            
            if stadium_filter:
                query = query.filter(Race.stadium_number == stadium_filter)
            
            races = query.all()
            
            upset_opportunities = []
            
            for race in races:
                analysis = self.analyze_race(race.id)
                if not analysis:
                    continue
                
                if analysis.upset_probability >= min_upset_prob:
                    upset_opportunities.append({
                        'race_id': race.id,
                        'race_info': analysis.race_info,
                        'upset_probability': analysis.upset_probability,
                        'competitiveness': analysis.competitiveness,
                        'dark_horses': analysis.dark_horses,
                        'key_factors': analysis.key_factors,
                        'betting_strategy': analysis.betting_strategy,
                        'confidence': analysis.confidence
                    })
            
            # 大荒れ確率順でソート
            upset_opportunities.sort(
                key=lambda x: x['upset_probability'],
                reverse=True
            )
            
            return upset_opportunities[:10]  # 上位10レース
    
    def _analyze_competitiveness(self, race: Race, entries: List[RaceEntry]) -> Dict[str, Any]:
        """競争激しさを分析"""
        if not entries:
            return {'level': '不明', 'upset_prob': 0.0}
        
        # 選手成績データ収集
        win_rates = []
        class_levels = []
        
        for entry in entries:
            if entry.racer:
                win_rates.append(float(entry.racer.national_winning_rate))
                class_levels.append(entry.racer.class_number)
        
        if not win_rates:
            return {'level': '不明', 'upset_prob': 0.0}
        
        # 勝率の分散分析
        avg_win_rate = mean(win_rates)
        win_rate_std = stdev(win_rates) if len(win_rates) > 1 else 0
        
        # 競争レベル判定
        if win_rate_std < 2.0:
            competitiveness = "激戦"
            upset_prob = 0.7
        elif win_rate_std < 5.0:
            competitiveness = "混戦"
            upset_prob = 0.5
        elif win_rate_std < 10.0:
            competitiveness = "実力差"
            upset_prob = 0.3
        else:
            competitiveness = "大荒れ"
            upset_prob = 0.8
        
        # クラス分布による調整
        a1_count = class_levels.count(1)
        if a1_count >= 4:
            upset_prob *= 0.8  # A1級多数は安定
        elif a1_count == 0:
            upset_prob *= 1.2  # A1級なしは荒れやすい
        
        return {
            'level': competitiveness,
            'upset_prob': min(upset_prob, 1.0),
            'avg_win_rate': avg_win_rate,
            'win_rate_std': win_rate_std,
            'class_distribution': dict(zip(*zip(*[(c, class_levels.count(c)) for c in set(class_levels)])))
        }
    
    def _analyze_key_factors(self, race: Race, entries: List[RaceEntry]) -> Dict[str, List[str]]:
        """レースの注目要因を分析"""
        factors = []
        
        # 競技場特性
        stadium_char = self.stadium_manager.get_characteristics(race.stadium_number)
        if stadium_char:
            if stadium_char.is_tidal:
                factors.append("潮汐の影響")
            if stadium_char.is_windy:
                factors.append("風の影響")
            if stadium_char.inside_advantage > 0.3:
                factors.append("1コース有利")
            elif stadium_char.dash_advantage > 0.3:
                factors.append("ダッシュ勢有利")
        
        # レースグレード
        if race.grade_number >= 4:
            factors.append("G1・SG級レース")
        elif race.grade_number >= 2:
            factors.append("G2・G3級レース")
        
        # 出走選手の特徴
        a1_count = sum(1 for e in entries if e.racer and e.racer.class_number == 1)
        if a1_count >= 4:
            factors.append("A1級多数出走")
        
        young_count = sum(1 for e in entries if e.racer and e.racer.age and e.racer.age < 25)
        if young_count >= 2:
            factors.append("若手選手多数")
        
        veteran_count = sum(1 for e in entries if e.racer and e.racer.age and e.racer.age > 50)
        if veteran_count >= 2:
            factors.append("ベテラン選手多数")
        
        return {'factors': factors}
    
    def _evaluate_racers(self, entries: List[RaceEntry]) -> Dict[str, List[int]]:
        """選手を評価して本命・穴を判定"""
        if not entries:
            return {'favorites': [], 'dark_horses': []}
        
        # 選手評価スコア計算
        racer_scores = []
        
        for entry in entries:
            if not entry.racer:
                continue
            
            score = 0.0
            
            # 基本成績評価
            score += float(entry.racer.national_winning_rate) * 2.0
            score += float(entry.racer.national_quinella_rate) * 1.0
            
            # クラス評価
            class_bonus = {1: 20, 2: 10, 3: 5, 4: 0}
            score += class_bonus.get(entry.racer.class_number, 0)
            
            # 艇番評価（1-3号艇は有利）
            if entry.boat_number <= 3:
                score += 5
            
            racer_scores.append((entry.boat_number, score))
        
        # スコア順でソート
        racer_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 本命・穴の判定
        favorites = [boat for boat, score in racer_scores[:2]]  # 上位2艇
        
        # 穴は下位から意外性のある艇を選択
        dark_horses = []
        for boat, score in racer_scores[3:]:  # 4位以下
            if score > 30:  # ある程度の実力があるが順位が低い
                dark_horses.append(boat)
        
        return {
            'favorites': favorites,
            'dark_horses': dark_horses[:2]  # 最大2艇
        }
    
    def _analyze_race_pattern(self, race: Race, entries: List[RaceEntry]) -> Dict[str, Any]:
        """レースパターンを分析"""
        # 競技場特性取得
        stadium_char = self.stadium_manager.get_characteristics(race.stadium_number)
        
        # パターン分類
        if stadium_char and stadium_char.inside_advantage > 0.3:
            pattern = "1コース中心"
            expected_pattern = "1コースが逃げ切るパターン"
        elif stadium_char and stadium_char.dash_advantage > 0.3:
            pattern = "ダッシュ決着"
            expected_pattern = "4-6コースのダッシュ決着"
        else:
            pattern = "実力勝負"
            expected_pattern = "実力通りの決着"
        
        # スタート予想（簡易版）
        start_prediction = {}
        for entry in entries:
            if entry.boat_number <= 2:
                start_prediction[entry.boat_number] = "好スタート"
            elif entry.boat_number >= 5:
                start_prediction[entry.boat_number] = "ダッシュ"
            else:
                start_prediction[entry.boat_number] = "普通"
        
        return {
            'pattern': pattern,
            'expected_pattern': expected_pattern,
            'start_prediction': start_prediction
        }
    
    def _suggest_betting_strategy(self, 
                                race: Race, 
                                entries: List[RaceEntry], 
                                competitiveness: Dict[str, Any]) -> Dict[str, str]:
        """投票戦略を提案"""
        upset_prob = competitiveness.get('upset_prob', 0.0)
        comp_level = competitiveness.get('level', '不明')
        
        if comp_level == "激戦":
            strategy = "薄く広く多点買い"
        elif comp_level == "混戦":
            strategy = "本命サイドと穴の両建て"
        elif comp_level == "実力差":
            strategy = "本命中心の手堅い買い"
        elif comp_level == "大荒れ":
            strategy = "穴狙いの高配当狙い"
        else:
            strategy = "様子見推奨"
        
        return {'strategy': strategy}
    
    def _calculate_analysis_confidence(self, race: Race, entries: List[RaceEntry]) -> float:
        """分析信頼度を計算"""
        confidence = 1.0
        
        # データ完全性
        if not all(e.racer for e in entries):
            confidence *= 0.8
        
        # 出走数
        if len(entries) < 6:
            confidence *= 0.9
        
        # 日付の新しさ
        if race.date:
            days_diff = (date.today() - race.date).days
            if days_diff > 7:
                confidence *= max(0.5, 1.0 - (days_diff / 30.0))
        
        return round(confidence, 2)
    
    def _get_distribution(self, values: List[str]) -> Dict[str, int]:
        """分布を計算"""
        distribution = defaultdict(int)
        for value in values:
            distribution[value] += 1
        return dict(distribution)
    
    def _select_focus_races(self, analyses: Dict[str, RaceAnalysis]) -> List[str]:
        """注目レースを選択"""
        # 信頼度と大荒れ確率でソート
        sorted_races = sorted(
            analyses.items(),
            key=lambda x: (x[1].confidence, x[1].upset_probability),
            reverse=True
        )
        
        return [race_id for race_id, _ in sorted_races[:3]]
    
    def _summarize_strategies(self, analyses: Dict[str, RaceAnalysis]) -> Dict[str, int]:
        """戦略の集計"""
        strategy_counts = defaultdict(int)
        for analysis in analyses.values():
            strategy_counts[analysis.betting_strategy] += 1
        return dict(strategy_counts)