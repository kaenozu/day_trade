"""
Boatraceデータ処理システム

収集したデータの前処理、分析、統計計算
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from collections import defaultdict
from statistics import mean, median, stdev

from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session

from .database import Database, get_database, Race, RaceEntry, Racer, RaceResult
from ..core.data_models import RacerClass, RaceGrade

logger = logging.getLogger(__name__)


class DataProcessor:
    """データ処理クラス"""
    
    def __init__(self, database: Optional[Database] = None):
        """
        初期化
        
        Args:
            database: データベース
        """
        self.database = database or get_database()
    
    def calculate_racer_statistics(self, 
                                 racer_number: int,
                                 days_back: int = 90) -> Dict[str, Any]:
        """
        選手統計を計算
        
        Args:
            racer_number: 選手番号
            days_back: 遡る日数
            
        Returns:
            選手統計情報
        """
        with self.database.get_session() as session:
            # 基本情報取得
            racer = session.query(Racer).filter(
                Racer.number == racer_number
            ).first()
            
            if not racer:
                return {'error': f'選手 {racer_number} が見つかりません'}
            
            # 期間内の出走データ取得
            cutoff_date = date.today() - timedelta(days=days_back)
            
            entries = session.query(RaceEntry).join(Race).filter(
                and_(
                    RaceEntry.racer_number == racer_number,
                    Race.date >= cutoff_date
                )
            ).all()
            
            if not entries:
                return {
                    'racer': {
                        'number': racer.number,
                        'name': racer.name,
                        'class': racer.class_number
                    },
                    'period_stats': {'races': 0}
                }
            
            # 統計計算
            stats = self._calculate_entry_statistics(entries)
            
            return {
                'racer': {
                    'number': racer.number,
                    'name': racer.name,
                    'age': racer.age,
                    'weight': racer.weight,
                    'class': racer.class_number,
                    'national_winning_rate': float(racer.national_winning_rate),
                    'national_quinella_rate': float(racer.national_quinella_rate)
                },
                'period_stats': stats,
                'analysis_period': f'{days_back}日間',
                'last_updated': datetime.now().isoformat()
            }
    
    def calculate_stadium_statistics(self, 
                                   stadium_number: int,
                                   days_back: int = 180) -> Dict[str, Any]:
        """
        競技場統計を計算
        
        Args:
            stadium_number: 競技場番号
            days_back: 遡る日数
            
        Returns:
            競技場統計情報
        """
        with self.database.get_session() as session:
            # 期間内のレース取得
            cutoff_date = date.today() - timedelta(days=days_back)
            
            races = session.query(Race).filter(
                and_(
                    Race.stadium_number == stadium_number,
                    Race.date >= cutoff_date
                )
            ).all()
            
            if not races:
                return {'error': f'競技場 {stadium_number} のデータが見つかりません'}
            
            # 各レースの出走情報を取得
            all_entries = []
            for race in races:
                entries = session.query(RaceEntry).filter(
                    RaceEntry.race_id == race.id
                ).all()
                all_entries.extend(entries)
            
            # 統計計算
            stadium_stats = self._calculate_stadium_entry_statistics(all_entries)
            
            return {
                'stadium_number': stadium_number,
                'period_stats': stadium_stats,
                'total_races': len(races),
                'analysis_period': f'{days_back}日間',
                'last_updated': datetime.now().isoformat()
            }
    
    def analyze_race_competitiveness(self, race_id: str) -> Dict[str, Any]:
        """
        レース競争力を分析
        
        Args:
            race_id: レースID
            
        Returns:
            競争力分析結果
        """
        with self.database.get_session() as session:
            # レース情報取得
            race = session.query(Race).filter(Race.id == race_id).first()
            if not race:
                return {'error': f'レース {race_id} が見つかりません'}
            
            # 出走情報取得
            entries = session.query(RaceEntry).join(Racer).filter(
                RaceEntry.race_id == race_id
            ).all()
            
            if not entries:
                return {'error': '出走情報が見つかりません'}
            
            # 選手成績分析
            win_rates = []
            class_distribution = defaultdict(int)
            ages = []
            
            for entry in entries:
                if entry.racer:
                    win_rates.append(float(entry.racer.national_winning_rate))
                    class_distribution[entry.racer.class_number] += 1
                    ages.append(entry.racer.age)
            
            # 統計計算
            avg_win_rate = mean(win_rates) if win_rates else 0
            win_rate_std = stdev(win_rates) if len(win_rates) > 1 else 0
            avg_age = mean(ages) if ages else 0
            
            # 競争激しさの判定
            if win_rate_std < 5.0:
                competitiveness = "非常に激しい"
            elif win_rate_std < 10.0:
                competitiveness = "激しい"
            elif win_rate_std < 15.0:
                competitiveness = "標準的"
            else:
                competitiveness = "実力差大"
            
            return {
                'race_id': race_id,
                'race_info': {
                    'date': race.date.isoformat(),
                    'stadium_number': race.stadium_number,
                    'race_number': race.race_number,
                    'title': race.title,
                    'grade': race.grade_number
                },
                'competitiveness': {
                    'level': competitiveness,
                    'average_win_rate': round(avg_win_rate, 2),
                    'win_rate_std': round(win_rate_std, 2),
                    'average_age': round(avg_age, 1)
                },
                'class_distribution': dict(class_distribution),
                'entry_count': len(entries)
            }
    
    def get_daily_summary(self, target_date: date) -> Dict[str, Any]:
        """
        日別サマリーを取得
        
        Args:
            target_date: 対象日
            
        Returns:
            日別サマリー
        """
        with self.database.get_session() as session:
            # 日別レース取得
            races = session.query(Race).filter(Race.date == target_date).all()
            
            if not races:
                return {'error': f'{target_date} のレースが見つかりません'}
            
            # 統計計算
            total_races = len(races)
            stadiums = set(race.stadium_number for race in races)
            grades = [race.grade_number for race in races]
            
            # グレード別集計
            grade_counts = defaultdict(int)
            for grade in grades:
                grade_counts[grade] += 1
            
            # 競技場別レース数
            stadium_counts = defaultdict(int)
            for race in races:
                stadium_counts[race.stadium_number] += 1
            
            # 出走選手総数
            total_entries = session.query(RaceEntry).join(Race).filter(
                Race.date == target_date
            ).count()
            
            return {
                'date': target_date.isoformat(),
                'summary': {
                    'total_races': total_races,
                    'total_stadiums': len(stadiums),
                    'total_entries': total_entries,
                    'average_entries_per_race': round(total_entries / total_races, 1) if total_races > 0 else 0
                },
                'grade_distribution': dict(grade_counts),
                'stadium_distribution': dict(stadium_counts),
                'stadiums_active': sorted(list(stadiums))
            }
    
    def find_similar_races(self, 
                          race_id: str,
                          similarity_factors: List[str] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        類似レースを検索
        
        Args:
            race_id: 基準レースID
            similarity_factors: 類似性要因
            limit: 取得件数上限
            
        Returns:
            類似レース一覧
        """
        if similarity_factors is None:
            similarity_factors = ['stadium', 'grade', 'distance']
        
        with self.database.get_session() as session:
            # 基準レース取得
            base_race = session.query(Race).filter(Race.id == race_id).first()
            if not base_race:
                return []
            
            # 類似性クエリ構築
            query = session.query(Race).filter(Race.id != race_id)
            
            if 'stadium' in similarity_factors:
                query = query.filter(Race.stadium_number == base_race.stadium_number)
            
            if 'grade' in similarity_factors:
                query = query.filter(Race.grade_number == base_race.grade_number)
            
            if 'distance' in similarity_factors:
                query = query.filter(Race.distance == base_race.distance)
            
            # 日付順（新しい順）
            similar_races = query.order_by(Race.date.desc()).limit(limit).all()
            
            return [
                {
                    'race_id': race.id,
                    'date': race.date.isoformat(),
                    'stadium_number': race.stadium_number,
                    'race_number': race.race_number,
                    'title': race.title,
                    'grade': race.grade_number,
                    'distance': race.distance
                }
                for race in similar_races
            ]
    
    def _calculate_entry_statistics(self, entries: List[RaceEntry]) -> Dict[str, Any]:
        """出走情報から統計を計算"""
        if not entries:
            return {'races': 0}
        
        # 競技場別集計
        stadium_counts = defaultdict(int)
        boat_number_counts = defaultdict(int)
        
        for entry in entries:
            race = entry.race
            if race:
                stadium_counts[race.stadium_number] += 1
            boat_number_counts[entry.boat_number] += 1
        
        # 艇番別成績（実際の結果データがあれば集計）
        # 現在は出走情報のみなので簡易統計
        
        return {
            'races': len(entries),
            'stadium_distribution': dict(stadium_counts),
            'boat_number_distribution': dict(boat_number_counts),
            'most_common_stadium': max(stadium_counts, key=stadium_counts.get) if stadium_counts else None,
            'most_common_boat_number': max(boat_number_counts, key=boat_number_counts.get)
        }
    
    def _calculate_stadium_entry_statistics(self, entries: List[RaceEntry]) -> Dict[str, Any]:
        """競技場の出走統計を計算"""
        if not entries:
            return {}
        
        boat_number_performance = defaultdict(list)
        class_distribution = defaultdict(int)
        
        for entry in entries:
            boat_number_performance[entry.boat_number].append(entry)
            if entry.racer:
                class_distribution[entry.racer.class_number] += 1
        
        # 艇番別出走回数
        boat_number_counts = {
            boat_num: len(entries) 
            for boat_num, entries in boat_number_performance.items()
        }
        
        return {
            'total_entries': len(entries),
            'boat_number_distribution': boat_number_counts,
            'class_distribution': dict(class_distribution),
            'average_entries_per_boat': round(len(entries) / 6, 1) if entries else 0
        }