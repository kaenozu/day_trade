"""
レース管理システム

レース情報の管理、分析、フィルタリング機能
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass

from .data_models import Race, RaceEntry, Racer, RaceGrade, RacerClass, STADIUMS
from .stadium_manager import StadiumManager
from .api_client import BoatraceAPIClient

logger = logging.getLogger(__name__)


@dataclass
class RaceFilter:
    """レースフィルター条件"""
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    stadium_numbers: Optional[Set[int]] = None
    grade_numbers: Optional[Set[int]] = None
    race_numbers: Optional[Set[int]] = None  # 1R, 2R, ...
    min_distance: Optional[int] = None
    max_distance: Optional[int] = None


@dataclass
class RacerFilter:
    """選手フィルター条件"""
    class_numbers: Optional[Set[int]] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    min_winning_rate: Optional[float] = None
    max_winning_rate: Optional[float] = None
    racer_numbers: Optional[Set[int]] = None


class RaceManager:
    """レース管理クラス"""
    
    def __init__(self, api_client: Optional[BoatraceAPIClient] = None):
        """
        初期化
        
        Args:
            api_client: APIクライアント
        """
        self.api_client = api_client or BoatraceAPIClient()
        self.stadium_manager = StadiumManager()
        
        # レースキャッシュ
        self.races_cache: Dict[str, Race] = {}
        self.cache_updated: Dict[date, datetime] = {}
    
    def get_today_races(self, 
                       use_cache: bool = True,
                       stadium_numbers: Optional[List[int]] = None) -> List[Race]:
        """
        今日のレース一覧を取得
        
        Args:
            use_cache: キャッシュを使用するか
            stadium_numbers: 特定の競技場のみ（Noneで全競技場）
            
        Returns:
            今日のレース一覧
        """
        try:
            programs_data = self.api_client.get_today_programs()
            races = self._parse_programs_data(programs_data)
            
            # 競技場フィルタ
            if stadium_numbers:
                races = [r for r in races if r.stadium_number in stadium_numbers]
            
            # キャッシュ更新
            if use_cache:
                self._update_cache(races)
            
            logger.info(f"今日のレース取得完了: {len(races)} races")
            return races
            
        except Exception as e:
            logger.error(f"今日のレース取得失敗: {e}")
            return []
    
    def get_races_by_date(self, 
                         target_date: date,
                         use_cache: bool = True) -> List[Race]:
        """
        指定日のレース一覧を取得
        
        Args:
            target_date: 対象日
            use_cache: キャッシュを使用するか
            
        Returns:
            指定日のレース一覧
        """
        try:
            programs_data = self.api_client.get_programs(target_date, use_cache)
            races = self._parse_programs_data(programs_data)
            
            # キャッシュ更新
            if use_cache:
                self._update_cache(races)
            
            logger.info(f"{target_date} のレース取得完了: {len(races)} races")
            return races
            
        except Exception as e:
            logger.error(f"{target_date} のレース取得失敗: {e}")
            return []
    
    def get_races_by_stadium(self, 
                            stadium_number: int,
                            target_date: Optional[date] = None) -> List[Race]:
        """
        特定競技場のレース一覧を取得
        
        Args:
            stadium_number: 競技場番号
            target_date: 対象日（Noneで今日）
            
        Returns:
            競技場のレース一覧
        """
        if target_date:
            races = self.get_races_by_date(target_date)
        else:
            races = self.get_today_races()
        
        return [r for r in races if r.stadium_number == stadium_number]
    
    def get_race_by_id(self, race_id: str) -> Optional[Race]:
        """
        レースIDでレースを取得
        
        Args:
            race_id: レースID (YYYYMMDD_競技場_レース番号)
            
        Returns:
            レース情報
        """
        # キャッシュから検索
        if race_id in self.races_cache:
            return self.races_cache[race_id]
        
        # レースIDから日付を抽出
        try:
            date_str = race_id.split('_')[0]
            target_date = datetime.strptime(date_str, '%Y%m%d').date()
            
            # 指定日のレースを取得
            races = self.get_races_by_date(target_date)
            
            # 該当レースを検索
            for race in races:
                if race.id == race_id:
                    return race
                    
        except (ValueError, IndexError) as e:
            logger.error(f"無効なレースID: {race_id} - {e}")
        
        return None
    
    def filter_races(self, 
                    races: List[Race],
                    race_filter: Optional[RaceFilter] = None) -> List[Race]:
        """
        レースをフィルタリング
        
        Args:
            races: レース一覧
            race_filter: フィルター条件
            
        Returns:
            フィルタされたレース一覧
        """
        if not race_filter:
            return races
        
        filtered_races = []
        
        for race in races:
            # 日付フィルター
            if race_filter.date_from and race.date < race_filter.date_from:
                continue
            if race_filter.date_to and race.date > race_filter.date_to:
                continue
            
            # 競技場フィルター
            if race_filter.stadium_numbers and race.stadium_number not in race_filter.stadium_numbers:
                continue
            
            # グレードフィルター
            if race_filter.grade_numbers and race.grade_number not in race_filter.grade_numbers:
                continue
            
            # レース番号フィルター
            if race_filter.race_numbers and race.race_number not in race_filter.race_numbers:
                continue
            
            # 距離フィルター
            if race_filter.min_distance and race.distance < race_filter.min_distance:
                continue
            if race_filter.max_distance and race.distance > race_filter.max_distance:
                continue
            
            filtered_races.append(race)
        
        return filtered_races
    
    def find_races_by_racer(self, 
                           racer_number: int,
                           target_date: Optional[date] = None,
                           days_range: int = 7) -> List[Race]:
        """
        選手が出走するレースを検索
        
        Args:
            racer_number: 選手番号
            target_date: 基準日（Noneで今日）
            days_range: 検索範囲（日数）
            
        Returns:
            選手が出走するレース一覧
        """
        base_date = target_date or date.today()
        found_races = []
        
        # 指定範囲の日付でレースを検索
        for i in range(-days_range, days_range + 1):
            search_date = base_date + timedelta(days=i)
            races = self.get_races_by_date(search_date)
            
            for race in races:
                for entry in race.entries:
                    if entry.racer.number == racer_number:
                        found_races.append(race)
                        break
        
        return sorted(found_races, key=lambda r: (r.date, r.stadium_number, r.race_number))
    
    def get_race_statistics(self, races: List[Race]) -> Dict[str, Any]:
        """
        レース統計情報を取得
        
        Args:
            races: レース一覧
            
        Returns:
            統計情報
        """
        if not races:
            return {}
        
        # 基本統計
        total_races = len(races)
        stadiums = set(r.stadium_number for r in races)
        grades = set(r.grade_number for r in races)
        dates = set(r.date for r in races)
        
        # 競技場別集計
        stadium_counts = defaultdict(int)
        for race in races:
            stadium_counts[race.stadium_number] += 1
        
        # グレード別集計
        grade_counts = defaultdict(int)
        for race in races:
            grade_counts[race.grade_number] += 1
        
        # 選手クラス別集計
        class_counts = defaultdict(int)
        for race in races:
            for entry in race.entries:
                class_counts[entry.racer.class_number] += 1
        
        # 距離別集計
        distance_counts = defaultdict(int)
        for race in races:
            distance_counts[race.distance] += 1
        
        return {
            'total_races': total_races,
            'stadiums_count': len(stadiums),
            'grades_count': len(grades), 
            'date_range': {
                'start': min(dates) if dates else None,
                'end': max(dates) if dates else None,
                'days': len(dates)
            },
            'stadium_distribution': dict(stadium_counts),
            'grade_distribution': dict(grade_counts),
            'class_distribution': dict(class_counts),
            'distance_distribution': dict(distance_counts)
        }
    
    def get_upcoming_featured_races(self, 
                                  days_ahead: int = 3) -> List[Race]:
        """
        今後の注目レースを取得
        
        Args:
            days_ahead: 先の日数
            
        Returns:
            注目レース一覧（SG、G1、G2）
        """
        featured_races = []
        base_date = date.today()
        
        for i in range(days_ahead + 1):
            search_date = base_date + timedelta(days=i)
            races = self.get_races_by_date(search_date)
            
            # 注目グレードのレースを抽出
            for race in races:
                if race.grade_number >= 3:  # G2以上
                    featured_races.append(race)
        
        return sorted(featured_races, 
                     key=lambda r: (r.date, -r.grade_number, r.stadium_number))
    
    def analyze_race_competitiveness(self, race: Race) -> Dict[str, Any]:
        """
        レースの競争力を分析
        
        Args:
            race: レース情報
            
        Returns:
            競争力分析結果
        """
        if not race.entries:
            return {}
        
        # 選手成績分析
        win_rates = [entry.racer.national_winning_rate for entry in race.entries]
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
        
        # クラス分布
        class_distribution = defaultdict(int)
        for entry in race.entries:
            class_distribution[entry.racer.class_number] += 1
        
        # 年齢分析
        ages = [entry.racer.age for entry in race.entries]
        avg_age = sum(ages) / len(ages) if ages else 0
        
        # 競技場特性取得
        stadium_char = self.stadium_manager.get_characteristics(race.stadium_number)
        
        # 競争激しさの指標
        win_rate_std = (sum((rate - avg_win_rate) ** 2 for rate in win_rates) / len(win_rates)) ** 0.5 if win_rates else 0
        competitiveness = "高" if win_rate_std < 5.0 else "中" if win_rate_std < 10.0 else "低"
        
        return {
            'average_win_rate': float(avg_win_rate),
            'win_rate_std': win_rate_std,
            'competitiveness': competitiveness,
            'average_age': avg_age,
            'class_distribution': dict(class_distribution),
            'stadium_characteristics': stadium_char,
            'recommended_strategy': self._get_race_strategy(race, stadium_char)
        }
    
    def _parse_programs_data(self, programs_data: Dict[str, Any]) -> List[Race]:
        """プログラムデータをレースオブジェクトに変換"""
        races = []
        
        for program_data in programs_data.get('programs', []):
            try:
                race = Race.from_api_data(program_data)
                races.append(race)
            except Exception as e:
                logger.warning(f"レースデータ解析エラー: {e}")
                continue
        
        return sorted(races, key=lambda r: (r.date, r.stadium_number, r.race_number))
    
    def _update_cache(self, races: List[Race]) -> None:
        """キャッシュを更新"""
        for race in races:
            self.races_cache[race.id] = race
        
        # 日付別キャッシュ更新時刻記録
        for race_date in set(race.date for race in races):
            self.cache_updated[race_date] = datetime.now()
    
    def _get_race_strategy(self, race: Race, stadium_char) -> str:
        """レース戦略を取得"""
        if not stadium_char:
            return "バランス型予想"
        
        if stadium_char.inside_advantage > 0.3:
            return "1コース軸予想"
        elif stadium_char.dash_advantage > 0.3:
            return "ダッシュ軸予想"
        elif stadium_char.is_tidal:
            return "潮汐考慮予想"
        else:
            return "実力重視予想"
    
    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self.races_cache.clear()
        self.cache_updated.clear()
        logger.info("レースキャッシュクリア完了")