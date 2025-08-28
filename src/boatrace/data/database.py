"""
Boatraceデータベース管理

SQLAlchemyを使用したデータベース操作
"""

import logging
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime, Float, 
    Text, Boolean, ForeignKey, Index, UniqueConstraint, text,
    DECIMAL, Time
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

logger = logging.getLogger(__name__)

Base = declarative_base()


class Stadium(Base):
    """競技場マスタ"""
    __tablename__ = 'stadiums'
    
    number = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    location = Column(String(100), nullable=False) 
    prefecture = Column(String(20), nullable=False)
    
    # 特性情報
    water_type = Column(String(20))  # river, lake, sea, bay
    is_tidal = Column(Boolean, default=False)
    is_windy = Column(Boolean, default=False) 
    has_waves = Column(Boolean, default=False)
    inside_advantage = Column(Float, default=0.0)
    dash_advantage = Column(Float, default=0.0)
    winter_difficulty = Column(Float, default=0.0)
    summer_difficulty = Column(Float, default=0.0)
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Racer(Base):
    """選手マスタ"""
    __tablename__ = 'racers'
    
    number = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer)
    weight = Column(Float)
    class_number = Column(Integer, default=4)  # 1:A1, 2:A2, 3:B1, 4:B2
    
    # 全国成績
    national_winning_rate = Column(DECIMAL(5, 2), default=0)
    national_quinella_rate = Column(DECIMAL(5, 2), default=0)
    national_trifecta_rate = Column(DECIMAL(5, 2), default=0)
    
    # 登録日・更新日
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # インデックス
    __table_args__ = (
        Index('idx_racer_name', 'name'),
        Index('idx_racer_class', 'class_number'),
    )


class Race(Base):
    """レースマスタ"""
    __tablename__ = 'races'
    
    id = Column(String(20), primary_key=True)  # YYYYMMDD_競技場_レース番号
    date = Column(Date, nullable=False)
    stadium_number = Column(Integer, ForeignKey('stadiums.number'), nullable=False)
    race_number = Column(Integer, nullable=False)
    
    title = Column(String(200))
    subtitle = Column(String(200))
    grade_number = Column(Integer, default=1)
    distance = Column(Integer, default=1800)
    closed_at = Column(Time)  # 投票締切時刻
    
    # 気象条件
    weather = Column(String(20))
    wind_velocity = Column(Float)
    wave_height = Column(Float)
    water_temperature = Column(Float)
    air_temperature = Column(Float)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # リレーション
    stadium = relationship("Stadium")
    entries = relationship("RaceEntry", back_populates="race")
    
    # インデックス
    __table_args__ = (
        Index('idx_race_date_stadium', 'date', 'stadium_number'),
        Index('idx_race_grade', 'grade_number'),
        UniqueConstraint('date', 'stadium_number', 'race_number', name='uq_race_key')
    )


class RaceEntry(Base):
    """出走情報"""
    __tablename__ = 'race_entries'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(String(20), ForeignKey('races.id'), nullable=False)
    boat_number = Column(Integer, nullable=False)
    racer_number = Column(Integer, ForeignKey('racers.number'), nullable=False)
    
    # 当日成績
    local_winning_rate = Column(DECIMAL(5, 2))
    local_quinella_rate = Column(DECIMAL(5, 2))
    local_trifecta_rate = Column(DECIMAL(5, 2))
    
    # モーター・ボート情報
    motor_number = Column(Integer)
    motor_winning_rate = Column(DECIMAL(5, 2))
    motor_quinella_rate = Column(DECIMAL(5, 2))
    boat_hull_number = Column(Integer)
    boat_winning_rate = Column(DECIMAL(5, 2))
    boat_quinella_rate = Column(DECIMAL(5, 2))
    
    # 展示・直前情報
    exhibition_time = Column(Float)
    exhibition_rank = Column(Integer)
    weight_adjustment = Column(Float)
    start_course = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # リレーション
    race = relationship("Race", back_populates="entries")
    racer = relationship("Racer")
    
    # インデックス
    __table_args__ = (
        Index('idx_entry_race_boat', 'race_id', 'boat_number'),
        Index('idx_entry_racer', 'racer_number'),
        UniqueConstraint('race_id', 'boat_number', name='uq_entry_key')
    )


class RaceResult(Base):
    """レース結果"""
    __tablename__ = 'race_results'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(String(20), ForeignKey('races.id'), nullable=False)
    
    # 着順情報
    first_place = Column(Integer, nullable=False)
    second_place = Column(Integer, nullable=False)
    third_place = Column(Integer, nullable=False)
    fourth_place = Column(Integer)
    fifth_place = Column(Integer)
    sixth_place = Column(Integer)
    
    # タイム情報
    first_time = Column(Float)
    second_time = Column(Float)
    third_time = Column(Float)
    
    # 配当情報
    win_payout = Column(DECIMAL(8, 0))          # 単勝
    place_show_payout = Column(DECIMAL(8, 0))   # 複勝
    exacta_payout = Column(DECIMAL(8, 0))       # 2連単
    quinella_payout = Column(DECIMAL(8, 0))     # 2連複
    trifecta_payout = Column(DECIMAL(8, 0))     # 3連単
    trio_payout = Column(DECIMAL(8, 0))         # 3連複
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # リレーション
    race = relationship("Race")
    
    # インデックス
    __table_args__ = (
        Index('idx_result_race', 'race_id'),
        UniqueConstraint('race_id', name='uq_result_race')
    )


class Prediction(Base):
    """予想データ"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(String(20), ForeignKey('races.id'), nullable=False)
    predicted_at = Column(DateTime, default=datetime.now)
    
    # 予想エンジン情報
    engine_name = Column(String(50), nullable=False)
    engine_version = Column(String(20))
    confidence = Column(DECIMAL(3, 2))  # 0.00 - 1.00
    
    # 1着予想確率 (各艇の1着確率)
    win_prob_1 = Column(DECIMAL(5, 4), default=0)
    win_prob_2 = Column(DECIMAL(5, 4), default=0)  
    win_prob_3 = Column(DECIMAL(5, 4), default=0)
    win_prob_4 = Column(DECIMAL(5, 4), default=0)
    win_prob_5 = Column(DECIMAL(5, 4), default=0)
    win_prob_6 = Column(DECIMAL(5, 4), default=0)
    
    # 3着以内確率
    place_prob_1 = Column(DECIMAL(5, 4), default=0)
    place_prob_2 = Column(DECIMAL(5, 4), default=0)
    place_prob_3 = Column(DECIMAL(5, 4), default=0)
    place_prob_4 = Column(DECIMAL(5, 4), default=0)
    place_prob_5 = Column(DECIMAL(5, 4), default=0)
    place_prob_6 = Column(DECIMAL(5, 4), default=0)
    
    # 推奨買い目
    recommended_bets = Column(Text)  # JSON形式
    
    created_at = Column(DateTime, default=datetime.now)
    
    # リレーション
    race = relationship("Race")
    
    # インデックス
    __table_args__ = (
        Index('idx_prediction_race_engine', 'race_id', 'engine_name'),
        Index('idx_prediction_date', 'predicted_at'),
    )


class BettingTicket(Base):
    """購入舟券"""
    __tablename__ = 'betting_tickets'
    
    id = Column(Integer, primary_key=True)
    race_id = Column(String(20), ForeignKey('races.id'), nullable=False)
    
    # 舟券情報
    bet_type = Column(String(20), nullable=False)  # win, exacta, trifecta, etc
    numbers = Column(String(50), nullable=False)   # "1-2-3" など
    amount = Column(DECIMAL(8, 0), nullable=False)
    odds = Column(DECIMAL(6, 1))
    
    # 購入・結果情報
    purchased_at = Column(DateTime, default=datetime.now)
    is_hit = Column(Boolean)
    payout = Column(DECIMAL(8, 0), default=0)
    profit = Column(DECIMAL(8, 0), default=0)  # 利益 = 配当 - 購入金額
    
    # 戦略情報
    strategy_name = Column(String(50))
    confidence = Column(DECIMAL(3, 2))
    
    created_at = Column(DateTime, default=datetime.now)
    
    # リレーション
    race = relationship("Race")
    
    # インデックス
    __table_args__ = (
        Index('idx_ticket_race', 'race_id'),
        Index('idx_ticket_date', 'purchased_at'),
        Index('idx_ticket_strategy', 'strategy_name'),
    )


class Database:
    """データベース管理クラス"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        初期化
        
        Args:
            database_url: データベースURL（デフォルト：SQLite）
        """
        if database_url is None:
            db_path = Path("data/databases/boatrace.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{db_path.absolute()}"
        elif database_url == ":memory:":
            # インメモリデータベース用の正しい形式
            database_url = "sqlite:///:memory:"
        
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,  # SQLログ出力
            pool_pre_ping=True,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        
        logger.info(f"データベース初期化: {database_url}")
    
    def create_tables(self) -> None:
        """テーブルを作成"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("データベーステーブル作成完了")
            
            # 競技場マスタデータの初期投入
            self._initialize_stadium_data()
            
        except Exception as e:
            logger.error(f"テーブル作成エラー: {e}")
            raise
    
    def drop_tables(self) -> None:
        """テーブルを削除"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("データベーステーブル削除完了")
        except Exception as e:
            logger.error(f"テーブル削除エラー: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """データベースセッションを取得"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"データベースセッションエラー: {e}")
            raise
        finally:
            session.close()
    
    def _initialize_stadium_data(self) -> None:
        """競技場マスタデータを初期化"""
        from ..core.data_models import STADIUMS
        from ..core.stadium_manager import StadiumManager
        
        with self.get_session() as session:
            # 既存データチェック
            existing_count = session.query(Stadium).count()
            if existing_count > 0:
                logger.info("競技場データは既に初期化済み")
                return
            
            # 競技場データ投入
            stadium_manager = StadiumManager()
            
            for num, stadium_info in STADIUMS.items():
                char = stadium_manager.get_characteristics(num)
                
                stadium = Stadium(
                    number=stadium_info.number,
                    name=stadium_info.name,
                    location=stadium_info.location,
                    prefecture=stadium_info.prefecture,
                    water_type=char.water_type.value if char else None,
                    is_tidal=char.is_tidal if char else False,
                    is_windy=char.is_windy if char else False,
                    has_waves=char.has_waves if char else False,
                    inside_advantage=char.inside_advantage if char else 0.0,
                    dash_advantage=char.dash_advantage if char else 0.0,
                    winter_difficulty=char.winter_difficulty if char else 0.0,
                    summer_difficulty=char.summer_difficulty if char else 0.0,
                    notes=", ".join(char.notes) if char and char.notes else None
                )
                
                session.add(stadium)
            
            session.commit()
            logger.info(f"競技場マスタデータ初期化完了: {len(STADIUMS)} stadiums")


# グローバルデータベースインスタンス
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """データベースインスタンスを取得"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def get_db_session():
    """データベースセッションを取得（依存性注入用）"""
    db = get_database()
    return db.get_session()


def init_database(database_url: Optional[str] = None) -> Database:
    """データベースを初期化"""
    global _db_instance
    _db_instance = Database(database_url)
    _db_instance.create_tables()
    return _db_instance