"""
Boatraceデータ収集システム

BoatraceOpenAPIからデータを収集してデータベースに保存
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.exc import IntegrityError

from ..core.api_client import BoatraceAPIClient
from ..core.data_models import Race as RaceModel
from .database import Database, get_database, Racer, Race, RaceEntry, RaceResult

logger = logging.getLogger(__name__)


class DataCollector:
    """データ収集クラス"""
    
    def __init__(self, 
                 api_client: Optional[BoatraceAPIClient] = None,
                 database: Optional[Database] = None):
        """
        初期化
        
        Args:
            api_client: APIクライアント
            database: データベース
        """
        self.api_client = api_client or BoatraceAPIClient()
        self.database = database or get_database()
        
        # 統計情報
        self.stats = {
            'races_collected': 0,
            'racers_collected': 0,
            'entries_collected': 0,
            'results_collected': 0,
            'errors': 0
        }
    
    def collect_today_data(self) -> Dict[str, int]:
        """
        今日のデータを収集
        
        Returns:
            収集統計
        """
        logger.info("今日のデータ収集開始")
        
        try:
            # APIから今日のプログラムデータを取得
            programs_data = self.api_client.get_today_programs()
            
            # データベースに保存
            result = self._save_programs_data(programs_data)
            
            logger.info(f"今日のデータ収集完了: {result}")
            return result
            
        except Exception as e:
            logger.error(f"今日のデータ収集失敗: {e}")
            self.stats['errors'] += 1
            return {'error': str(e)}
    
    def collect_date_data(self, target_date: date) -> Dict[str, int]:
        """
        指定日のデータを収集
        
        Args:
            target_date: 対象日
            
        Returns:
            収集統計
        """
        logger.info(f"{target_date} のデータ収集開始")
        
        try:
            # APIからプログラムデータを取得
            programs_data = self.api_client.get_programs(target_date)
            
            # データベースに保存
            result = self._save_programs_data(programs_data)
            
            logger.info(f"{target_date} のデータ収集完了: {result}")
            return result
            
        except Exception as e:
            logger.error(f"{target_date} のデータ収集失敗: {e}")
            self.stats['errors'] += 1
            return {'error': str(e)}
    
    def collect_date_range_data(self, 
                               start_date: date, 
                               end_date: date,
                               skip_existing: bool = True) -> Dict[str, Any]:
        """
        日付範囲のデータを収集
        
        Args:
            start_date: 開始日
            end_date: 終了日
            skip_existing: 既存データをスキップするか
            
        Returns:
            収集統計
        """
        logger.info(f"日付範囲データ収集開始: {start_date} - {end_date}")
        
        total_stats = {
            'dates_processed': 0,
            'races_collected': 0,
            'racers_collected': 0,
            'entries_collected': 0,
            'errors': 0
        }
        
        current_date = start_date
        while current_date <= end_date:
            try:
                # 既存データチェック
                if skip_existing and self._date_data_exists(current_date):
                    logger.info(f"{current_date} のデータは既に存在、スキップ")
                    current_date += timedelta(days=1)
                    continue
                
                # データ収集
                day_result = self.collect_date_data(current_date)
                
                if 'error' not in day_result:
                    total_stats['dates_processed'] += 1
                    total_stats['races_collected'] += day_result.get('races', 0)
                    total_stats['racers_collected'] += day_result.get('racers', 0)
                    total_stats['entries_collected'] += day_result.get('entries', 0)
                else:
                    total_stats['errors'] += 1
                    
            except Exception as e:
                logger.error(f"{current_date} の処理エラー: {e}")
                total_stats['errors'] += 1
                
            current_date += timedelta(days=1)
        
        logger.info(f"日付範囲データ収集完了: {total_stats}")
        return total_stats
    
    def collect_results_data(self, target_date: date) -> Dict[str, int]:
        """
        結果データを収集
        
        Args:
            target_date: 対象日
            
        Returns:
            収集統計
        """
        logger.info(f"{target_date} の結果データ収集開始")
        
        try:
            # APIから結果データを取得
            results_data = self.api_client.get_results(target_date)
            
            # データベースに保存
            result = self._save_results_data(results_data)
            
            logger.info(f"{target_date} の結果データ収集完了: {result}")
            return result
            
        except Exception as e:
            logger.error(f"{target_date} の結果データ収集失敗: {e}")
            self.stats['errors'] += 1
            return {'error': str(e)}
    
    def collect_previews_data(self, target_date: date) -> Dict[str, int]:
        """
        直前情報データを収集
        
        Args:
            target_date: 対象日
            
        Returns:
            収集統計
        """
        logger.info(f"{target_date} の直前情報データ収集開始")
        
        try:
            # APIから直前情報データを取得
            previews_data = self.api_client.get_previews(target_date)
            
            # データベースに保存（出走情報を更新）
            result = self._update_entries_with_previews(previews_data)
            
            logger.info(f"{target_date} の直前情報データ収集完了: {result}")
            return result
            
        except Exception as e:
            logger.error(f"{target_date} の直前情報データ収集失敗: {e}")
            self.stats['errors'] += 1
            return {'error': str(e)}
    
    def _save_programs_data(self, programs_data: Dict[str, Any]) -> Dict[str, int]:
        """プログラムデータをデータベースに保存"""
        stats = {'races': 0, 'racers': 0, 'entries': 0}
        
        with self.database.get_session() as session:
            for program_data in programs_data.get('programs', []):
                try:
                    # レースモデルに変換
                    race_model = RaceModel.from_api_data(program_data)
                    
                    # レースを保存
                    race_saved = self._save_race(session, race_model)
                    if race_saved:
                        stats['races'] += 1
                    
                    # 選手・出走情報を保存
                    for entry in race_model.entries:
                        # 選手保存
                        racer_saved = self._save_racer(session, entry.racer)
                        if racer_saved:
                            stats['racers'] += 1
                        
                        # 出走情報保存
                        entry_saved = self._save_race_entry(session, entry)
                        if entry_saved:
                            stats['entries'] += 1
                
                except Exception as e:
                    logger.error(f"プログラムデータ保存エラー: {e}")
                    continue
        
        return stats
    
    def _save_race(self, session, race_model: RaceModel) -> bool:
        """レースをデータベースに保存"""
        try:
            race = Race(
                id=race_model.id,
                date=race_model.date,
                stadium_number=race_model.stadium_number,
                race_number=race_model.race_number,
                title=race_model.title,
                subtitle=race_model.subtitle,
                grade_number=race_model.grade_number,
                distance=race_model.distance,
                closed_at=race_model.closed_at,
                weather=race_model.weather.value if race_model.weather else None,
                wind_velocity=race_model.wind_velocity,
                wave_height=race_model.wave_height,
                water_temperature=race_model.water_temperature,
                air_temperature=race_model.air_temperature
            )
            
            # UPSERT実行
            session.merge(race)
            return True
            
        except Exception as e:
            logger.error(f"レース保存エラー: {race_model.id} - {e}")
            return False
    
    def _save_racer(self, session, racer_model) -> bool:
        """選手をデータベースに保存"""
        try:
            # 既存チェック
            existing_racer = session.query(Racer).filter(
                Racer.number == racer_model.number
            ).first()
            
            if existing_racer:
                # 更新
                existing_racer.name = racer_model.name
                existing_racer.age = racer_model.age
                existing_racer.weight = racer_model.weight
                existing_racer.class_number = racer_model.class_number
                existing_racer.national_winning_rate = racer_model.national_winning_rate
                existing_racer.national_quinella_rate = racer_model.national_quinella_rate
                existing_racer.national_trifecta_rate = racer_model.national_trifecta_rate
                existing_racer.updated_at = datetime.now()
                return False  # 新規作成ではない
            else:
                # 新規作成
                racer = Racer(
                    number=racer_model.number,
                    name=racer_model.name,
                    age=racer_model.age,
                    weight=racer_model.weight,
                    class_number=racer_model.class_number,
                    national_winning_rate=racer_model.national_winning_rate,
                    national_quinella_rate=racer_model.national_quinella_rate,
                    national_trifecta_rate=racer_model.national_trifecta_rate
                )
                session.add(racer)
                return True
            
        except Exception as e:
            logger.error(f"選手保存エラー: {racer_model.number} - {e}")
            return False
    
    def _save_race_entry(self, session, entry_model) -> bool:
        """出走情報をデータベースに保存"""
        try:
            entry = RaceEntry(
                race_id=entry_model.race_id,
                boat_number=entry_model.boat_number,
                racer_number=entry_model.racer.number,
                local_winning_rate=entry_model.racer.local_winning_rate,
                local_quinella_rate=entry_model.racer.local_quinella_rate,
                local_trifecta_rate=entry_model.racer.local_trifecta_rate,
                motor_number=entry_model.racer.motor_number,
                motor_winning_rate=entry_model.racer.motor_winning_rate,
                motor_quinella_rate=entry_model.racer.motor_quinella_rate,
                boat_hull_number=entry_model.racer.boat_number,
                boat_winning_rate=entry_model.racer.boat_winning_rate,
                boat_quinella_rate=entry_model.racer.boat_quinella_rate,
                exhibition_time=entry_model.exhibition_time,
                exhibition_rank=entry_model.exhibition_rank,
                weight_adjustment=entry_model.weight_adjustment,
                start_course=entry_model.start_course
            )
            
            # UPSERT実行
            session.merge(entry)
            return True
            
        except Exception as e:
            logger.error(f"出走情報保存エラー: {entry_model.race_id}_{entry_model.boat_number} - {e}")
            return False
    
    def _save_results_data(self, results_data: Dict[str, Any]) -> Dict[str, int]:
        """結果データをデータベースに保存"""
        stats = {'results': 0}
        
        # 実装省略（結果データの構造が明確でないため）
        # 実際の結果データ構造に合わせて実装
        
        return stats
    
    def _update_entries_with_previews(self, previews_data: Dict[str, Any]) -> Dict[str, int]:
        """直前情報で出走情報を更新"""
        stats = {'entries_updated': 0}
        
        # 実装省略（直前情報データの構造が明確でないため）
        # 実際の直前情報データ構造に合わせて実装
        
        return stats
    
    def _date_data_exists(self, target_date: date) -> bool:
        """指定日のデータが既に存在するかチェック"""
        with self.database.get_session() as session:
            count = session.query(Race).filter(Race.date == target_date).count()
            return count > 0
    
    def get_collection_stats(self) -> Dict[str, int]:
        """収集統計を取得"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """統計をリセット"""
        for key in self.stats:
            self.stats[key] = 0