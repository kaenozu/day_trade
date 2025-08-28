"""
選手分析システム

選手の成績、特徴、調子を分析
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from collections import defaultdict
from statistics import mean, median
from dataclasses import dataclass

from ..core.data_models import Racer, RacerClass, STADIUMS
from ..data.database import Database, get_database
from ..data.database import Racer as RacerDB, Race, RaceEntry

logger = logging.getLogger(__name__)


@dataclass
class RacerAnalysis:
    """選手分析結果"""
    racer_number: int
    name: str
    class_number: int
    
    # 基本成績
    national_win_rate: float
    national_quinella_rate: float
    
    # 競技場適性
    stadium_affinity: Dict[int, float]  # {競技場番号: 適性度}
    best_stadiums: List[int]           # 得意競技場
    worst_stadiums: List[int]          # 苦手競技場
    
    # 艇番適性
    boat_number_performance: Dict[int, float]  # {艇番: 成績}
    best_boat_numbers: List[int]
    
    # 調子・トレンド
    recent_form: str                   # "好調", "普通", "不調"
    form_trend: str                   # "上昇", "安定", "下降"
    
    # 特徴・強み
    strengths: List[str]
    weaknesses: List[str]
    
    # 信頼度
    analysis_confidence: float         # 0.0-1.0


class RacerAnalyzer:
    """選手分析クラス"""
    
    def __init__(self, database: Optional[Database] = None):
        """
        初期化
        
        Args:
            database: データベース
        """
        self.database = database or get_database()
    
    def analyze_racer(self, 
                     racer_number: int,
                     analysis_days: int = 180) -> Optional[RacerAnalysis]:
        """
        選手を総合分析
        
        Args:
            racer_number: 選手番号
            analysis_days: 分析対象日数
            
        Returns:
            選手分析結果
        """
        with self.database.get_session() as session:
            # 選手基本情報取得
            racer = session.query(RacerDB).filter(
                RacerDB.number == racer_number
            ).first()
            
            if not racer:
                logger.warning(f"選手 {racer_number} が見つかりません")
                return None
            
            # 期間内出走データ取得
            cutoff_date = date.today() - timedelta(days=analysis_days)
            entries = session.query(RaceEntry).join(Race).filter(
                RaceEntry.racer_number == racer_number,
                Race.date >= cutoff_date
            ).all()
            
            if not entries:
                logger.warning(f"選手 {racer_number} の期間内データがありません")
                return None
            
            # 各種分析実行
            stadium_affinity = self._analyze_stadium_affinity(entries)
            boat_performance = self._analyze_boat_number_performance(entries)
            form_analysis = self._analyze_recent_form(entries)
            characteristics = self._analyze_racer_characteristics(racer, entries)
            
            # 分析結果統合
            return RacerAnalysis(
                racer_number=racer.number,
                name=racer.name,
                class_number=racer.class_number,
                national_win_rate=float(racer.national_winning_rate),
                national_quinella_rate=float(racer.national_quinella_rate),
                stadium_affinity=stadium_affinity['affinity'],
                best_stadiums=stadium_affinity['best'][:3],
                worst_stadiums=stadium_affinity['worst'][:3],
                boat_number_performance=boat_performance['performance'],
                best_boat_numbers=boat_performance['best'][:2],
                recent_form=form_analysis['form'],
                form_trend=form_analysis['trend'],
                strengths=characteristics['strengths'],
                weaknesses=characteristics['weaknesses'],
                analysis_confidence=self._calculate_confidence(entries)
            )
    
    def compare_racers(self, 
                      racer_numbers: List[int],
                      comparison_aspects: List[str] = None) -> Dict[str, Any]:
        """
        複数選手を比較
        
        Args:
            racer_numbers: 選手番号リスト
            comparison_aspects: 比較観点
            
        Returns:
            比較結果
        """
        if comparison_aspects is None:
            comparison_aspects = ['winning_rate', 'class', 'recent_form']
        
        analyses = {}
        for racer_num in racer_numbers:
            analysis = self.analyze_racer(racer_num)
            if analysis:
                analyses[racer_num] = analysis
        
        if not analyses:
            return {'error': '有効な選手データがありません'}
        
        # 比較結果生成
        comparison = {
            'racers': len(analyses),
            'aspects': {}
        }
        
        if 'winning_rate' in comparison_aspects:
            comparison['aspects']['winning_rate'] = {
                'ranking': sorted(
                    analyses.items(),
                    key=lambda x: x[1].national_win_rate,
                    reverse=True
                ),
                'average': mean([a.national_win_rate for a in analyses.values()]),
                'best': max(analyses.values(), key=lambda x: x.national_win_rate).racer_number
            }
        
        if 'class' in comparison_aspects:
            class_distribution = defaultdict(int)
            for analysis in analyses.values():
                class_distribution[analysis.class_number] += 1
            
            comparison['aspects']['class'] = {
                'distribution': dict(class_distribution),
                'best_class': min(analyses.values(), key=lambda x: x.class_number).class_number
            }
        
        if 'recent_form' in comparison_aspects:
            form_counts = defaultdict(int)
            for analysis in analyses.values():
                form_counts[analysis.recent_form] += 1
            
            comparison['aspects']['recent_form'] = {
                'distribution': dict(form_counts),
                'good_form_count': form_counts.get('好調', 0)
            }
        
        return comparison
    
    def find_emerging_racers(self, 
                           class_filter: Optional[int] = None,
                           min_races: int = 20) -> List[Dict[str, Any]]:
        """
        調子の良い新進選手を発見
        
        Args:
            class_filter: クラスフィルター
            min_races: 最小出走回数
            
        Returns:
            新進選手リスト
        """
        with self.database.get_session() as session:
            # 基本クエリ
            query = session.query(RacerDB)
            
            if class_filter:
                query = query.filter(RacerDB.class_number == class_filter)
            
            racers = query.all()
            
            emerging_candidates = []
            
            for racer in racers:
                # 最近の出走データ取得
                cutoff_date = date.today() - timedelta(days=90)
                recent_entries = session.query(RaceEntry).join(Race).filter(
                    RaceEntry.racer_number == racer.number,
                    Race.date >= cutoff_date
                ).count()
                
                if recent_entries < min_races:
                    continue
                
                # 簡易分析
                analysis = self.analyze_racer(racer.number, 90)
                if not analysis:
                    continue
                
                # 新進性の判定
                if (analysis.recent_form == "好調" and 
                    analysis.form_trend in ["上昇", "安定"] and
                    analysis.national_win_rate > 0.0):
                    
                    emerging_candidates.append({
                        'racer_number': racer.number,
                        'name': racer.name,
                        'class': racer.class_number,
                        'win_rate': analysis.national_win_rate,
                        'form': analysis.recent_form,
                        'trend': analysis.form_trend,
                        'recent_races': recent_entries,
                        'confidence': analysis.analysis_confidence
                    })
            
            # 勝率と調子でソート
            emerging_candidates.sort(
                key=lambda x: (x['confidence'], x['win_rate']),
                reverse=True
            )
            
            return emerging_candidates[:20]  # 上位20名
    
    def _analyze_stadium_affinity(self, entries: List[RaceEntry]) -> Dict[str, Any]:
        """競技場適性を分析"""
        stadium_counts = defaultdict(int)
        stadium_performance = defaultdict(list)
        
        for entry in entries:
            if entry.race:
                stadium_num = entry.race.stadium_number
                stadium_counts[stadium_num] += 1
                
                # 実際の結果データがあれば成績を記録
                # 現在は出走回数のみ
                stadium_performance[stadium_num].append(1)
        
        # 適性度計算（出走回数ベース）
        total_entries = sum(stadium_counts.values())
        affinity = {}
        
        for stadium_num, count in stadium_counts.items():
            if count >= 3:  # 最低3回以上の出走
                affinity[stadium_num] = round(count / total_entries * 100, 1)
        
        # ソート
        sorted_by_affinity = sorted(affinity.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'affinity': affinity,
            'best': [s[0] for s in sorted_by_affinity[:5]],
            'worst': [s[0] for s in sorted_by_affinity[-5:]],
            'total_stadiums': len(stadium_counts)
        }
    
    def _analyze_boat_number_performance(self, entries: List[RaceEntry]) -> Dict[str, Any]:
        """艇番成績を分析"""
        boat_counts = defaultdict(int)
        boat_performance = defaultdict(list)
        
        for entry in entries:
            boat_num = entry.boat_number
            boat_counts[boat_num] += 1
            
            # 実際の結果があれば成績記録
            boat_performance[boat_num].append(1)
        
        # 成績計算（現在は出走回数ベース）
        performance = {}
        for boat_num, count in boat_counts.items():
            if count > 0:
                performance[boat_num] = round(count / sum(boat_counts.values()) * 100, 1)
        
        sorted_performance = sorted(performance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'performance': performance,
            'best': [b[0] for b in sorted_performance[:3]],
            'worst': [b[0] for b in sorted_performance[-3:]],
            'distribution': dict(boat_counts)
        }
    
    def _analyze_recent_form(self, entries: List[RaceEntry]) -> Dict[str, str]:
        """最近の調子を分析"""
        if not entries:
            return {'form': '不明', 'trend': '不明'}
        
        # 日付でソート
        sorted_entries = sorted(entries, key=lambda e: e.race.date if e.race else date.min)
        
        # 最近30日の成績を重視
        recent_cutoff = date.today() - timedelta(days=30)
        recent_entries = [e for e in sorted_entries if e.race and e.race.date >= recent_cutoff]
        
        if len(recent_entries) < 3:
            return {'form': '普通', 'trend': '安定'}
        
        # 簡易的な調子判定（実際の結果データがあればより詳細に）
        recent_races = len(recent_entries)
        
        if recent_races >= 10:
            form = "好調"
        elif recent_races >= 5:
            form = "普通"
        else:
            form = "不調"
        
        # トレンド分析（期間比較）
        mid_point = len(sorted_entries) // 2
        if mid_point > 0:
            first_half = sorted_entries[:mid_point]
            second_half = sorted_entries[mid_point:]
            
            if len(second_half) > len(first_half) * 1.2:
                trend = "上昇"
            elif len(second_half) < len(first_half) * 0.8:
                trend = "下降"
            else:
                trend = "安定"
        else:
            trend = "安定"
        
        return {'form': form, 'trend': trend}
    
    def _analyze_racer_characteristics(self, racer: RacerDB, entries: List[RaceEntry]) -> Dict[str, List[str]]:
        """選手の特徴を分析"""
        strengths = []
        weaknesses = []
        
        # クラス別特徴
        if racer.class_number == 1:
            strengths.append("A1級の実力")
        elif racer.class_number == 2:
            strengths.append("A2級の安定性")
        
        # 勝率による特徴
        if racer.national_winning_rate > 6.5:
            strengths.append("高勝率")
        elif racer.national_winning_rate < 4.0:
            weaknesses.append("低勝率")
        
        # 複勝率による特徴
        if racer.national_quinella_rate > 50.0:
            strengths.append("安定した複勝率")
        elif racer.national_quinella_rate < 30.0:
            weaknesses.append("複勝率の低さ")
        
        # 年齢による特徴
        if racer.age and racer.age < 25:
            strengths.append("若さ")
        elif racer.age and racer.age > 50:
            strengths.append("経験豊富")
        
        # 出走数による特徴
        if len(entries) > 50:
            strengths.append("豊富な出走経験")
        elif len(entries) < 10:
            weaknesses.append("出走データ不足")
        
        return {'strengths': strengths, 'weaknesses': weaknesses}
    
    def _calculate_confidence(self, entries: List[RaceEntry]) -> float:
        """分析信頼度を計算"""
        if not entries:
            return 0.0
        
        # データ量による信頼度
        data_confidence = min(len(entries) / 50.0, 1.0)
        
        # 期間の新しさによる信頼度
        if entries:
            latest_date = max(e.race.date for e in entries if e.race)
            days_ago = (date.today() - latest_date).days
            freshness_confidence = max(1.0 - (days_ago / 180.0), 0.1)
        else:
            freshness_confidence = 0.1
        
        # 総合信頼度
        overall_confidence = (data_confidence + freshness_confidence) / 2.0
        
        return round(overall_confidence, 2)