#!/usr/bin/env python3
"""
Day Trade Personal - 東証銘柄データ

Issue #912対応: 東証全銘柄の拡充
"""

from typing import List, Dict, Optional
import logging


class TokyoStockExchange:
    """東証銘柄管理クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_all_symbols(self) -> List[str]:
        """東証全銘柄コードを取得"""
        return list(self._get_symbol_database().keys())
        
    def get_major_symbols(self) -> List[str]:
        """主要銘柄を取得"""
        major_codes = [
            # 日経225主要構成銘柄
            '7203', '8306', '9984', '6758', '7974', '4689', '9434',
            '6861', '6954', '8035', '4063', '9432', '2914', '4502',
            '4568', '8316', '8411', '3382', '9983', '4503', '6367',
            '7267', '9766', '4755', '6098', '6501', '6503', '7751',
            '6326', '6594', '3436', '4523', '6702', '6971', '8001',
            '8002', '8053', '9062', '9064', '9021', '9022', '9202'
        ]
        return major_codes
        
    def get_by_sector(self, sector: str) -> List[str]:
        """業種別銘柄を取得"""
        sector_map = {
            'tech': ['6758', '7974', '4689', '9984', '9434', '6861', '6954'],
            'automotive': ['7203', '7267', '7269', '7270'],
            'finance': ['8306', '8316', '8411', '8331', '8354'],
            'retail': ['8035', '3382', '9983', '7974'],
            'pharma': ['4502', '4503', '4568', '4523'],
            'manufacturing': ['6501', '6503', '6326', '6594', '6702'],
            'telecom': ['9432', '9984', '9434'],
            'energy': ['5020', '1605', '1662'],
            'materials': ['4063', '5401', '5411'],
            'utilities': ['9501', '9502', '9503', '9531'],
            'real_estate': ['8802', '8804', '8815'],
            'transportation': ['9021', '9022', '9062', '9064']
        }
        return sector_map.get(sector.lower(), [])
        
    def get_company_info(self, symbol: str) -> Optional[Dict[str, str]]:
        """企業情報を取得"""
        database = self._get_symbol_database()
        return database.get(symbol)
        
    def search_by_name(self, keyword: str) -> List[str]:
        """企業名による検索"""
        database = self._get_symbol_database()
        keyword_lower = keyword.lower()
        
        matches = []
        for symbol, info in database.items():
            if keyword_lower in info['name'].lower() or keyword_lower in info.get('name_en', '').lower():
                matches.append(symbol)
                
        return matches
        
    def get_market_cap_ranking(self, top_n: int = 100) -> List[str]:
        """時価総額上位銘柄を取得"""
        # 実際の実装では市場データAPIから取得
        # ここでは主要銘柄を時価総額順に並べたもの
        top_symbols = [
            '7203',  # トヨタ自動車
            '6758',  # ソニー
            '9984',  # ソフトバンクグループ
            '8306',  # 三菱UFJ銀行
            '6861',  # キーエンス
            '7974',  # 任天堂
            '4689',  # ヤフー
            '9434',  # ソフトバンク
            '6954',  # ファナック
            '8035',  # 東京エレクトロン
        ]
        return top_symbols[:top_n]
        
    def _get_symbol_database(self) -> Dict[str, Dict[str, str]]:
        """銘柄データベース（拡張可能）"""
        return {
            # テクノロジー・通信
            '6758': {'name': 'ソニー', 'name_en': 'Sony', 'sector': 'tech', 'market': 'prime'},
            '7974': {'name': '任天堂', 'name_en': 'Nintendo', 'sector': 'tech', 'market': 'prime'},
            '4689': {'name': 'ヤフー', 'name_en': 'Yahoo', 'sector': 'tech', 'market': 'prime'},
            '9984': {'name': 'ソフトバンクグループ', 'name_en': 'SoftBank Group', 'sector': 'telecom', 'market': 'prime'},
            '9434': {'name': 'ソフトバンク', 'name_en': 'SoftBank', 'sector': 'telecom', 'market': 'prime'},
            '6861': {'name': 'キーエンス', 'name_en': 'Keyence', 'sector': 'tech', 'market': 'prime'},
            '6954': {'name': 'ファナック', 'name_en': 'Fanuc', 'sector': 'tech', 'market': 'prime'},
            '8035': {'name': '東京エレクトロン', 'name_en': 'Tokyo Electron', 'sector': 'tech', 'market': 'prime'},
            '9432': {'name': 'NTT', 'name_en': 'NTT', 'sector': 'telecom', 'market': 'prime'},
            
            # 自動車
            '7203': {'name': 'トヨタ自動車', 'name_en': 'Toyota Motor', 'sector': 'automotive', 'market': 'prime'},
            '7267': {'name': 'ホンダ', 'name_en': 'Honda', 'sector': 'automotive', 'market': 'prime'},
            '7269': {'name': 'スズキ', 'name_en': 'Suzuki', 'sector': 'automotive', 'market': 'prime'},
            '7270': {'name': 'SUBARU', 'name_en': 'Subaru', 'sector': 'automotive', 'market': 'prime'},
            
            # 金融
            '8306': {'name': '三菱UFJ銀行', 'name_en': 'MUFG Bank', 'sector': 'finance', 'market': 'prime'},
            '8316': {'name': '三井住友銀行', 'name_en': 'Sumitomo Mitsui Banking', 'sector': 'finance', 'market': 'prime'},
            '8411': {'name': 'みずほ銀行', 'name_en': 'Mizuho Bank', 'sector': 'finance', 'market': 'prime'},
            '8331': {'name': '千葉銀行', 'name_en': 'Chiba Bank', 'sector': 'finance', 'market': 'prime'},
            '8354': {'name': 'ふくおか銀行', 'name_en': 'Fukuoka Bank', 'sector': 'finance', 'market': 'prime'},
            
            # 小売・消費
            '3382': {'name': 'セブン&アイ', 'name_en': 'Seven & i Holdings', 'sector': 'retail', 'market': 'prime'},
            '9983': {'name': 'ファーストリテイリング', 'name_en': 'Fast Retailing', 'sector': 'retail', 'market': 'prime'},
            '2914': {'name': '日本たばこ産業', 'name_en': 'Japan Tobacco', 'sector': 'retail', 'market': 'prime'},
            
            # 医薬品
            '4502': {'name': '武田薬品工業', 'name_en': 'Takeda Pharmaceutical', 'sector': 'pharma', 'market': 'prime'},
            '4503': {'name': 'アステラス製薬', 'name_en': 'Astellas Pharma', 'sector': 'pharma', 'market': 'prime'},
            '4568': {'name': '第一三共', 'name_en': 'Daiichi Sankyo', 'sector': 'pharma', 'market': 'prime'},
            '4523': {'name': 'エーザイ', 'name_en': 'Eisai', 'sector': 'pharma', 'market': 'prime'},
            
            # 製造業
            '6501': {'name': '日立製作所', 'name_en': 'Hitachi', 'sector': 'manufacturing', 'market': 'prime'},
            '6503': {'name': '三菱電機', 'name_en': 'Mitsubishi Electric', 'sector': 'manufacturing', 'market': 'prime'},
            '6326': {'name': 'クボタ', 'name_en': 'Kubota', 'sector': 'manufacturing', 'market': 'prime'},
            '6594': {'name': '日本電産', 'name_en': 'Nidec', 'sector': 'manufacturing', 'market': 'prime'},
            '6702': {'name': '富士通', 'name_en': 'Fujitsu', 'sector': 'tech', 'market': 'prime'},
            '6971': {'name': '京セラ', 'name_en': 'Kyocera', 'sector': 'manufacturing', 'market': 'prime'},
            
            # 商社
            '8001': {'name': '伊藤忠商事', 'name_en': 'Itochu', 'sector': 'trading', 'market': 'prime'},
            '8002': {'name': '丸紅', 'name_en': 'Marubeni', 'sector': 'trading', 'market': 'prime'},
            '8053': {'name': '住友商事', 'name_en': 'Sumitomo Corp', 'sector': 'trading', 'market': 'prime'},
            
            # 運輸
            '9021': {'name': 'JR東海', 'name_en': 'JR Central', 'sector': 'transportation', 'market': 'prime'},
            '9022': {'name': 'JR東日本', 'name_en': 'JR East', 'sector': 'transportation', 'market': 'prime'},
            '9062': {'name': '日本通運', 'name_en': 'Nippon Express', 'sector': 'transportation', 'market': 'prime'},
            '9064': {'name': 'ヤマト運輸', 'name_en': 'Yamato Transport', 'sector': 'transportation', 'market': 'prime'},
            '9202': {'name': 'ANA', 'name_en': 'ANA Holdings', 'sector': 'transportation', 'market': 'prime'},
            
            # エネルギー・素材
            '5020': {'name': 'ENEOS', 'name_en': 'ENEOS Holdings', 'sector': 'energy', 'market': 'prime'},
            '1605': {'name': '国際石油開発帝石', 'name_en': 'INPEX', 'sector': 'energy', 'market': 'prime'},
            '4063': {'name': '信越化学工業', 'name_en': 'Shin-Etsu Chemical', 'sector': 'materials', 'market': 'prime'},
            '5401': {'name': '新日鐵住金', 'name_en': 'Nippon Steel', 'sector': 'materials', 'market': 'prime'},
            
            # 公益事業
            '9501': {'name': '東京電力', 'name_en': 'TEPCO', 'sector': 'utilities', 'market': 'prime'},
            '9502': {'name': '中部電力', 'name_en': 'Chubu Electric', 'sector': 'utilities', 'market': 'prime'},
            '9503': {'name': '関西電力', 'name_en': 'Kansai Electric', 'sector': 'utilities', 'market': 'prime'},
            '9531': {'name': '東京ガス', 'name_en': 'Tokyo Gas', 'sector': 'utilities', 'market': 'prime'},
            
            # 不動産
            '8802': {'name': '三菱地所', 'name_en': 'Mitsubishi Estate', 'sector': 'real_estate', 'market': 'prime'},
            '8804': {'name': '東京建物', 'name_en': 'Tokyo Tatemono', 'sector': 'real_estate', 'market': 'prime'},
            
            # その他注目銘柄
            '4755': {'name': '楽天', 'name_en': 'Rakuten', 'sector': 'tech', 'market': 'prime'},
            '6098': {'name': 'リクルート', 'name_en': 'Recruit Holdings', 'sector': 'services', 'market': 'prime'},
            '7751': {'name': 'キヤノン', 'name_en': 'Canon', 'sector': 'tech', 'market': 'prime'},
            '9766': {'name': 'コナミ', 'name_en': 'Konami', 'sector': 'entertainment', 'market': 'prime'},
            '3436': {'name': 'SUMCO', 'name_en': 'SUMCO', 'sector': 'materials', 'market': 'prime'},
            '6367': {'name': 'ダイキン工業', 'name_en': 'Daikin Industries', 'sector': 'manufacturing', 'market': 'prime'},
        }
        
    def get_extended_symbol_set(self, include_small_cap: bool = False) -> List[str]:
        """拡張銘柄セット取得"""
        base_symbols = list(self._get_symbol_database().keys())
        
        if include_small_cap:
            # 小型株も含める場合の追加銘柄
            small_cap_symbols = [
                '2432', '3092', '3668', '3900', '4324', '4385', '4751',
                '6028', '6178', '6182', '6200', '6301', '6758', '7003'
            ]
            base_symbols.extend(small_cap_symbols)
            
        return sorted(list(set(base_symbols)))  # 重複除去・ソート


# シングルトンインスタンス
tse = TokyoStockExchange()


def get_symbol_recommendations(risk_level: str = 'medium') -> List[str]:
    """リスクレベル別推奨銘柄"""
    risk_map = {
        'low': tse.get_major_symbols()[:20],      # 低リスク: 大型株20銘柄
        'medium': tse.get_major_symbols(),        # 中リスク: 主要銘柄全て
        'high': tse.get_extended_symbol_set(True) # 高リスク: 小型株含む拡張セット
    }
    return risk_map.get(risk_level, tse.get_major_symbols())


def get_diversified_portfolio(count: int = 10) -> List[str]:
    """分散ポートフォリオ用銘柄選定"""
    sectors = ['tech', 'automotive', 'finance', 'pharma', 'manufacturing']
    diversified = []
    
    for sector in sectors:
        sector_symbols = tse.get_by_sector(sector)
        if sector_symbols:
            diversified.extend(sector_symbols[:count//len(sectors)])
            
    # 不足分は時価総額上位から補完
    if len(diversified) < count:
        top_symbols = tse.get_market_cap_ranking(count)
        for symbol in top_symbols:
            if symbol not in diversified:
                diversified.append(symbol)
                if len(diversified) >= count:
                    break
                    
    return diversified[:count]