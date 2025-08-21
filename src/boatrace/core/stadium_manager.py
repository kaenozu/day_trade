"""
競技場管理システム

全国24競技場の情報管理と特性分析
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .data_models import Stadium, STADIUMS

logger = logging.getLogger(__name__)


class StadiumType(Enum):
    """競技場タイプ"""
    RIVER = "river"          # 河川
    LAKE = "lake"            # 湖
    SEA = "sea"              # 海
    BAY = "bay"              # 湾


@dataclass
class StadiumCharacteristics:
    """競技場特性"""
    stadium_number: int
    water_type: StadiumType
    
    # 基本特性
    is_tidal: bool = False              # 潮汐の影響
    is_windy: bool = False              # 風の影響強
    has_waves: bool = False             # 波立ちやすい
    
    # コース特性
    inside_advantage: float = 0.0       # インコース有利度 (-1.0 ~ 1.0)
    dash_advantage: float = 0.0         # ダッシュコース有利度 (-1.0 ~ 1.0)
    
    # 季節要因
    winter_difficulty: float = 0.0      # 冬場の走りにくさ (0.0 ~ 1.0)
    summer_difficulty: float = 0.0      # 夏場の走りにくさ (0.0 ~ 1.0)
    
    # 備考
    notes: List[str] = field(default_factory=list)


class StadiumManager:
    """競技場管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.stadiums = STADIUMS.copy()
        self.characteristics = self._initialize_characteristics()
    
    def _initialize_characteristics(self) -> Dict[int, StadiumCharacteristics]:
        """競技場特性データを初期化"""
        characteristics = {}
        
        # 各競技場の特性を設定
        characteristics[1] = StadiumCharacteristics(  # 桐生
            stadium_number=1,
            water_type=StadiumType.LAKE,
            inside_advantage=0.3,
            winter_difficulty=0.7,
            notes=["標高70mの淡水湖", "冬場は気温が低く選手に厳しい", "1コース有利"]
        )
        
        characteristics[2] = StadiumCharacteristics(  # 戸田
            stadium_number=2,
            water_type=StadiumType.RIVER,
            inside_advantage=0.4,
            notes=["首都圏唯一の河川", "1コースの勝率が高い", "比較的静水面"]
        )
        
        characteristics[3] = StadiumCharacteristics(  # 江戸川
            stadium_number=3,
            water_type=StadiumType.RIVER,
            is_tidal=True,
            inside_advantage=-0.2,
            dash_advantage=0.3,
            notes=["潮汐の影響大", "満潮時は1コース不利", "4-6コースにチャンス"]
        )
        
        characteristics[4] = StadiumCharacteristics(  # 平和島
            stadium_number=4,  
            water_type=StadiumType.SEA,
            is_tidal=True,
            has_waves=True,
            inside_advantage=-0.1,
            notes=["海水で潮汐影響", "大時計が特徴", "波立ちやすい"]
        )
        
        characteristics[5] = StadiumCharacteristics(  # 多摩川
            stadium_number=5,
            water_type=StadiumType.RIVER,
            inside_advantage=0.2,
            notes=["淡水の静水面", "比較的1コース有利", "企画レースが多い"]
        )
        
        characteristics[6] = StadiumCharacteristics(  # 浜名湖
            stadium_number=6,
            water_type=StadiumType.LAKE,
            is_windy=True,
            inside_advantage=0.1,
            summer_difficulty=0.3,
            notes=["汽水湖", "風の影響受けやすい", "夏場は藻が発生"]
        )
        
        characteristics[7] = StadiumCharacteristics(  # 蒲郡
            stadium_number=7,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.3,
            dash_advantage=0.4,
            notes=["三河湾の海水", "潮汐で水面変化", "ダッシュ勢有利"]
        )
        
        characteristics[8] = StadiumCharacteristics(  # 常滑
            stadium_number=8,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.2,
            notes=["伊勢湾の海水", "潮汐の影響", "4-6コースにチャンス"]
        )
        
        characteristics[9] = StadiumCharacteristics(  # 津
            stadium_number=9,
            water_type=StadiumType.SEA,
            is_tidal=True,
            has_waves=True,
            inside_advantage=-0.1,
            notes=["伊勢湾最奥部", "風波の影響", "潮汐変化大"]
        )
        
        characteristics[10] = StadiumCharacteristics(  # 三国
            stadium_number=10,
            water_type=StadiumType.RIVER,
            is_windy=True,
            inside_advantage=0.2,
            winter_difficulty=0.8,
            notes=["九頭竜川河口", "冬場は雪と寒さ", "風の影響強"]
        )
        
        characteristics[11] = StadiumCharacteristics(  # びわこ
            stadium_number=11,
            water_type=StadiumType.LAKE,
            is_windy=True,
            inside_advantage=0.0,
            notes=["日本最大の湖", "風向きで水面変化", "標高85m"]
        )
        
        characteristics[12] = StadiumCharacteristics(  # 住之江
            stadium_number=12,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.2,
            dash_advantage=0.3,
            notes=["大阪湾の海水", "潮汐影響大", "ダッシュ決まりやすい"]
        )
        
        characteristics[13] = StadiumCharacteristics(  # 尼崎
            stadium_number=13,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.1,
            notes=["大阪湾奥部", "潮汐変化", "センタープールでの開催も"]
        )
        
        characteristics[14] = StadiumCharacteristics(  # 鳴門
            stadium_number=14,
            water_type=StadiumType.SEA,
            is_tidal=True,
            has_waves=True,
            inside_advantage=-0.3,
            dash_advantage=0.5,
            notes=["鳴門海峡近く", "潮流激しい", "大幅にダッシュ有利"]
        )
        
        characteristics[15] = StadiumCharacteristics(  # 丸亀
            stadium_number=15,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.1,
            notes=["瀬戸内海", "穏やかな海面", "潮汐影響あり"]
        )
        
        characteristics[16] = StadiumCharacteristics(  # 児島
            stadium_number=16,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.2,
            dash_advantage=0.2,
            notes=["瀬戸内海", "潮汐変化", "ダッシュ勢にもチャンス"]
        )
        
        characteristics[17] = StadiumCharacteristics(  # 宮島
            stadium_number=17,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.4,
            dash_advantage=0.6,
            notes=["広島湾", "潮汐差最大", "全国一ダッシュ有利"]
        )
        
        characteristics[18] = StadiumCharacteristics(  # 徳山
            stadium_number=18,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.2,
            notes=["周防灘", "潮汐影響", "企画レース多数"]
        )
        
        characteristics[19] = StadiumCharacteristics(  # 下関
            stadium_number=19,
            water_type=StadiumType.SEA,
            is_tidal=True,
            has_waves=True,
            inside_advantage=-0.2,
            dash_advantage=0.3,
            notes=["関門海峡近く", "潮流強い", "荒れやすい水面"]
        )
        
        characteristics[20] = StadiumCharacteristics(  # 若松
            stadium_number=20,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.2,
            notes=["洞海湾", "潮汐影響", "工業地帯の海"]
        )
        
        characteristics[21] = StadiumCharacteristics(  # 芦屋
            stadium_number=21,
            water_type=StadiumType.RIVER,
            inside_advantage=0.3,
            notes=["遠賀川河口", "1コース有利", "比較的穏やか"]
        )
        
        characteristics[22] = StadiumCharacteristics(  # 福岡
            stadium_number=22,
            water_type=StadiumType.SEA,
            is_tidal=True,
            inside_advantage=-0.1,
            notes=["博多湾", "潮汐変化", "都市部の競技場"]
        )
        
        characteristics[23] = StadiumCharacteristics(  # 唐津
            stadium_number=23,
            water_type=StadiumType.SEA,
            is_windy=True,
            has_waves=True,
            inside_advantage=-0.2,
            notes=["玄界灘", "風波の影響大", "荒れた水面"]
        )
        
        characteristics[24] = StadiumCharacteristics(  # 大村
            stadium_number=24,
            water_type=StadiumType.LAKE,
            inside_advantage=0.4,
            notes=["大村湾", "穏やかな湖", "1コース最有利"]
        )
        
        return characteristics
    
    def get_stadium(self, stadium_number: int) -> Optional[Stadium]:
        """競技場情報を取得"""
        return self.stadiums.get(stadium_number)
    
    def get_characteristics(self, stadium_number: int) -> Optional[StadiumCharacteristics]:
        """競技場特性を取得"""
        return self.characteristics.get(stadium_number)
    
    def get_all_stadiums(self) -> Dict[int, Stadium]:
        """全競技場を取得"""
        return self.stadiums.copy()
    
    def get_stadiums_by_type(self, water_type: StadiumType) -> List[Stadium]:
        """水面タイプ別競技場を取得"""
        result = []
        for num, stadium in self.stadiums.items():
            char = self.characteristics.get(num)
            if char and char.water_type == water_type:
                result.append(stadium)
        return result
    
    def get_tidal_stadiums(self) -> List[Stadium]:
        """潮汐影響のある競技場を取得"""
        result = []
        for num, stadium in self.stadiums.items():
            char = self.characteristics.get(num)
            if char and char.is_tidal:
                result.append(stadium)
        return result
    
    def get_inside_advantage_stadiums(self, threshold: float = 0.2) -> List[Stadium]:
        """インコース有利な競技場を取得"""
        result = []
        for num, stadium in self.stadiums.items():
            char = self.characteristics.get(num)
            if char and char.inside_advantage >= threshold:
                result.append(stadium)
        return result
    
    def get_dash_advantage_stadiums(self, threshold: float = 0.2) -> List[Stadium]:
        """ダッシュ有利な競技場を取得"""
        result = []
        for num, stadium in self.stadiums.items():
            char = self.characteristics.get(num)
            if char and char.dash_advantage >= threshold:
                result.append(stadium)
        return result
    
    def search_stadiums(self, 
                       prefecture: Optional[str] = None,
                       water_type: Optional[StadiumType] = None,
                       name_contains: Optional[str] = None) -> List[Stadium]:
        """競技場検索"""
        result = []
        for stadium in self.stadiums.values():
            # 都道府県フィルター
            if prefecture and stadium.prefecture != prefecture:
                continue
            
            # 水面タイプフィルター
            if water_type:
                char = self.characteristics.get(stadium.number)
                if not char or char.water_type != water_type:
                    continue
            
            # 名前フィルター
            if name_contains and name_contains not in stadium.name:
                continue
            
            result.append(stadium)
        
        return sorted(result, key=lambda s: s.number)
    
    def get_stadium_analysis(self, stadium_number: int) -> Optional[Dict[str, any]]:
        """競技場分析情報を取得"""
        stadium = self.get_stadium(stadium_number)
        char = self.get_characteristics(stadium_number)
        
        if not stadium or not char:
            return None
        
        return {
            'stadium': stadium,
            'characteristics': char,
            'water_type_name': char.water_type.value,
            'advantages': self._analyze_advantages(char),
            'difficulties': self._analyze_difficulties(char),
            'recommendations': self._get_recommendations(char)
        }
    
    def _analyze_advantages(self, char: StadiumCharacteristics) -> List[str]:
        """有利な要素を分析"""
        advantages = []
        
        if char.inside_advantage > 0.2:
            advantages.append("1コース有利")
        elif char.inside_advantage < -0.2:
            advantages.append("アウトコースにチャンス")
        
        if char.dash_advantage > 0.2:
            advantages.append("ダッシュ勢有利")
        
        return advantages
    
    def _analyze_difficulties(self, char: StadiumCharacteristics) -> List[str]:
        """困難な要素を分析"""
        difficulties = []
        
        if char.is_tidal:
            difficulties.append("潮汐の影響")
        
        if char.is_windy:
            difficulties.append("風の影響")
        
        if char.has_waves:
            difficulties.append("波立ちやすい")
        
        if char.winter_difficulty > 0.5:
            difficulties.append("冬場は厳しい条件")
        
        if char.summer_difficulty > 0.5:
            difficulties.append("夏場は厳しい条件")
        
        return difficulties
    
    def _get_recommendations(self, char: StadiumCharacteristics) -> List[str]:
        """おすすめ戦略"""
        recommendations = []
        
        if char.inside_advantage > 0.3:
            recommendations.append("1コース軸で予想")
        elif char.dash_advantage > 0.3:
            recommendations.append("4-6コースの巻き返しに注目")
        
        if char.is_tidal:
            recommendations.append("満潮・干潮時間をチェック")
        
        if char.is_windy:
            recommendations.append("風向き・風速を確認")
        
        return recommendations