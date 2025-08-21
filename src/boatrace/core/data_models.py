"""
競艇データモデル定義

レース、選手、競技場などの基本的なデータ構造を定義
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
import json


class RaceGrade(Enum):
    """レースグレード"""
    GENERAL = 1      # 一般戦
    G3 = 2          # G3
    G2 = 3          # G2
    G1 = 4          # G1
    SG = 5          # SG


class RacerClass(Enum):
    """選手クラス"""
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4


class Weather(Enum):
    """天候"""
    FINE = 1        # 晴
    CLOUDY = 2      # 曇
    RAINY = 3       # 雨
    SNOWY = 4       # 雪
    TYPHOON = 5     # 台風


@dataclass
class Stadium:
    """競技場情報"""
    number: int
    name: str
    location: str
    prefecture: str
    
    @property
    def display_name(self) -> str:
        """表示用名前"""
        return f"{self.number:02d}. {self.name}"


# 全国24競技場のマスターデータ
STADIUMS = {
    1: Stadium(1, "桐生", "群馬県みどり市", "群馬県"),
    2: Stadium(2, "戸田", "埼玉県戸田市", "埼玉県"),
    3: Stadium(3, "江戸川", "東京都江戸川区", "東京都"),
    4: Stadium(4, "平和島", "東京都大田区", "東京都"),
    5: Stadium(5, "多摩川", "東京都府中市", "東京都"),
    6: Stadium(6, "浜名湖", "静岡県湖西市", "静岡県"),
    7: Stadium(7, "蒲郡", "愛知県蒲郡市", "愛知県"),
    8: Stadium(8, "常滑", "愛知県常滑市", "愛知県"),
    9: Stadium(9, "津", "三重県津市", "三重県"),
    10: Stadium(10, "三国", "福井県坂井市", "福井県"),
    11: Stadium(11, "びわこ", "滋賀県大津市", "滋賀県"),
    12: Stadium(12, "住之江", "大阪府大阪市", "大阪府"),
    13: Stadium(13, "尼崎", "兵庫県尼崎市", "兵庫県"),
    14: Stadium(14, "鳴門", "徳島県鳴門市", "徳島県"),
    15: Stadium(15, "丸亀", "香川県丸亀市", "香川県"),
    16: Stadium(16, "児島", "岡山県倉敷市", "岡山県"),
    17: Stadium(17, "宮島", "広島県廿日市市", "広島県"),
    18: Stadium(18, "徳山", "山口県周南市", "山口県"),
    19: Stadium(19, "下関", "山口県下関市", "山口県"),
    20: Stadium(20, "若松", "福岡県北九州市", "福岡県"),
    21: Stadium(21, "芦屋", "福岡県遠賀郡", "福岡県"),
    22: Stadium(22, "福岡", "福岡県福岡市", "福岡県"),
    23: Stadium(23, "唐津", "佐賀県唐津市", "佐賀県"),
    24: Stadium(24, "大村", "長崎県大村市", "長崎県"),
}


@dataclass
class Racer:
    """選手情報"""
    number: int                                    # 選手番号
    name: str                                      # 選手名
    age: int                                       # 年齢
    weight: float                                  # 体重
    class_number: int                              # クラス番号
    
    # 全国成績
    national_winning_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    national_quinella_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    national_trifecta_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # 当地成績
    local_winning_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    local_quinella_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    local_trifecta_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # モーター成績
    motor_number: Optional[int] = None
    motor_winning_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    motor_quinella_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    
    # ボート成績
    boat_number: Optional[int] = None
    boat_winning_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    boat_quinella_rate: Decimal = field(default_factory=lambda: Decimal('0'))
    
    @property
    def racer_class(self) -> RacerClass:
        """選手クラス"""
        return RacerClass(self.class_number)
    
    @property
    def display_name(self) -> str:
        """表示用名前"""
        return f"{self.number} {self.name}"
    
    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> 'Racer':
        """APIデータから選手オブジェクトを生成"""
        return cls(
            number=data.get('racer_number', 0),
            name=data.get('racer_name', ''),
            age=data.get('racer_age', 0),
            weight=data.get('racer_weight', 0.0),
            class_number=data.get('racer_class_number', 4),
            
            national_winning_rate=Decimal(str(data.get('racer_national_winning_rate', 0))),
            national_quinella_rate=Decimal(str(data.get('racer_national_quinella_rate', 0))),
            national_trifecta_rate=Decimal(str(data.get('racer_national_trifecta_rate', 0))),
            
            local_winning_rate=Decimal(str(data.get('racer_local_winning_rate', 0))),
            local_quinella_rate=Decimal(str(data.get('racer_local_quinella_rate', 0))),
            local_trifecta_rate=Decimal(str(data.get('racer_local_trifecta_rate', 0))),
            
            motor_number=data.get('motor_number'),
            motor_winning_rate=Decimal(str(data.get('motor_winning_rate', 0))),
            motor_quinella_rate=Decimal(str(data.get('motor_quinella_rate', 0))),
            
            boat_number=data.get('boat_number'),
            boat_winning_rate=Decimal(str(data.get('boat_winning_rate', 0))),
            boat_quinella_rate=Decimal(str(data.get('boat_quinella_rate', 0))),
        )


@dataclass
class RaceEntry:
    """出走情報"""
    race_id: str                # レースID
    boat_number: int           # 艇番
    racer: Racer              # 選手情報
    
    # 展示タイム関連
    exhibition_time: Optional[float] = None
    exhibition_rank: Optional[int] = None
    
    # 直前情報
    weight_adjustment: Optional[float] = None  # 減量
    start_course: Optional[int] = None         # 進入コース
    
    @property
    def display_info(self) -> str:
        """表示用情報"""
        return f"{self.boat_number}. {self.racer.display_name}"


@dataclass 
class Race:
    """レース情報"""
    id: str                           # レースID (日付_競技場_レース番号)
    date: date                        # レース日
    stadium_number: int               # 競技場番号
    race_number: int                  # レース番号
    
    title: str                        # レースタイトル
    subtitle: str = ""                # レースサブタイトル
    grade_number: int = 1             # グレード番号
    distance: int = 1800              # 距離
    closed_at: Optional[time] = None  # 投票締切時刻
    
    # 気象条件
    weather: Optional[Weather] = None
    wind_velocity: Optional[float] = None  # 風速
    wave_height: Optional[float] = None    # 波高
    water_temperature: Optional[float] = None  # 水温
    air_temperature: Optional[float] = None    # 気温
    
    # 出走選手
    entries: List[RaceEntry] = field(default_factory=list)
    
    @property 
    def stadium(self) -> Stadium:
        """競技場情報"""
        return STADIUMS.get(self.stadium_number)
    
    @property
    def grade(self) -> RaceGrade:
        """レースグレード"""
        return RaceGrade(self.grade_number)
    
    @property
    def display_name(self) -> str:
        """表示用名前"""
        stadium_name = self.stadium.name if self.stadium else f"競技場{self.stadium_number}"
        return f"{self.date.strftime('%m/%d')} {stadium_name} {self.race_number}R"
    
    @property
    def full_title(self) -> str:
        """完全なタイトル"""
        if self.subtitle:
            return f"{self.title} {self.subtitle}"
        return self.title
    
    def get_entry_by_boat_number(self, boat_number: int) -> Optional[RaceEntry]:
        """艇番で出走情報を取得"""
        for entry in self.entries:
            if entry.boat_number == boat_number:
                return entry
        return None
    
    def get_entries_by_class(self, racer_class: RacerClass) -> List[RaceEntry]:
        """クラス別出走選手を取得"""
        return [entry for entry in self.entries 
                if entry.racer.racer_class == racer_class]
    
    @classmethod
    def from_api_data(cls, data: Dict[str, Any]) -> 'Race':
        """APIデータからレースオブジェクトを生成"""
        race_date = datetime.strptime(data['race_date'], '%Y-%m-%d').date()
        stadium_number = data['race_stadium_number']
        race_number = data['race_number']
        
        race_id = f"{race_date.strftime('%Y%m%d')}_{stadium_number:02d}_{race_number:02d}"
        
        # 投票締切時刻の解析
        closed_at = None
        if data.get('race_closed_at'):
            try:
                closed_at = datetime.strptime(data['race_closed_at'], '%H:%M:%S').time()
            except ValueError:
                pass
        
        race = cls(
            id=race_id,
            date=race_date,
            stadium_number=stadium_number,
            race_number=race_number,
            title=data.get('race_title', ''),
            subtitle=data.get('race_subtitle', ''),
            grade_number=data.get('race_grade_number', 1),
            distance=data.get('race_distance', 1800),
            closed_at=closed_at,
        )
        
        # 出走情報を追加
        boats = data.get('boats', [])
        for boat_data in boats:
            boat_number = boat_data.get('racer_boat_number', 0)
            racer = Racer.from_api_data(boat_data)
            
            entry = RaceEntry(
                race_id=race_id,
                boat_number=boat_number,
                racer=racer
            )
            race.entries.append(entry)
        
        return race


@dataclass
class RaceResult:
    """レース結果"""
    race_id: str
    race: Race
    finish_order: List[int]        # 着順 [1位艇番, 2位艇番, ...]
    finish_times: List[float]      # 着順タイム
    
    # 配当情報
    win_payout: Optional[Decimal] = None          # 単勝
    place_show_payout: Optional[Decimal] = None   # 複勝
    exacta_payout: Optional[Decimal] = None       # 2連単
    quinella_payout: Optional[Decimal] = None     # 2連複  
    trifecta_payout: Optional[Decimal] = None     # 3連単
    trio_payout: Optional[Decimal] = None         # 3連複
    
    @property
    def winner_boat_number(self) -> int:
        """勝艇番"""
        return self.finish_order[0] if self.finish_order else 0
    
    @property
    def winner_racer(self) -> Optional[Racer]:
        """優勝選手"""
        if self.finish_order:
            entry = self.race.get_entry_by_boat_number(self.finish_order[0])
            return entry.racer if entry else None
        return None


@dataclass
class PredictionResult:
    """予想結果"""
    race_id: str
    predicted_at: datetime
    
    # 予想着順（確率付き）
    win_probabilities: Dict[int, Decimal]      # {艇番: 1着確率}
    place_probabilities: Dict[int, Decimal]    # {艇番: 3着以内確率}
    
    # 推奨買い目
    recommended_bets: List[Dict[str, Any]] = field(default_factory=list)
    
    # 信頼度
    confidence: Decimal = field(default_factory=lambda: Decimal('0'))
    
    @property
    def top_win_prediction(self) -> int:
        """1着予想（最も確率が高い艇番）"""
        if not self.win_probabilities:
            return 0
        return max(self.win_probabilities, key=self.win_probabilities.get)


def parse_stadiums_config() -> Dict[int, Stadium]:
    """競技場設定を読み込み"""
    return STADIUMS