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
        
    def get_tier1_symbols(self) -> List[str]:
        """Tier 1: 主要銘柄（現在の74銘柄）"""
        return list(self._get_symbol_database().keys())
        
    def get_tier2_symbols(self) -> List[str]:
        """Tier 2: 中型株・成長株（+200銘柄）"""
        tier2_symbols = [
            # 中型株（時価総額1000億円以上）
            '1332', '1605', '1662', '1721', '1801', '1802', '1803', '1928', '1963',
            '2002', '2269', '2282', '2432', '2433', '2503', '2531', '2801', '2802',
            '2871', '2914', '3086', '3092', '3099', '3101', '3103', '3105', '3141',
            '3148', '3156', '3167', '3175', '3197', '3231', '3284', '3288', '3349',
            '3360', '3382', '3401', '3402', '3405', '3407', '3433', '3436', '3452',
            '3543', '3551', '3563', '3591', '3632', '3635', '3648', '3668', '3681',
            '3683', '3694', '3762', '3765', '3769', '3774', '3823', '3832', '3844',
            '3863', '3864', '3932', '3938', '3941', '3962', '3984', '3993', '4004',
            '4021', '4041', '4042', '4043', '4061', '4088', '4091', '4183', '4188',
            '4203', '4204', '4208', '4272', '4307', '4324', '4343', '4362', '4369',
            '4371', '4372', '4385', '4452', '4463', '4519', '4544', '4549', '4612',
            '4613', '4631', '4661', '4676', '4681', '4684', '4686', '4704', '4707',
            '4708', '4709', '4716', '4722', '4726', '4728', '4731', '4732', '4739',
            '4743', '4751', '4755', '4768', '4777', '4812', '4813', '4816', '4819',
            '4901', '4902', '4911', '4912', '4921', '4922', '4927', '4967', '4968',
            '5019', '5101', '5108', '5110', '5201', '5202', '5214', '5301', '5332',
            '5333', '5334', '5401', '5411', '5541', '5631', '5703', '5706', '5707',
            '5711', '5741', '5801', '5802', '5803', '5901', '5943', '5947', '5952',
            '6028', '6062', '6098', '6113', '6134', '6136', '6141', '6146', '6178',
            '6182', '6200', '6301', '6302', '6305', '6324', '6330', '6361', '6370',
            '6417', '6432', '6448', '6457', '6460', '6462', '6471', '6473', '6479',
            '6502', '6504', '6506', '6508', '6645', '6674', '6701', '6703', '6724',
            '6740', '6752', '6753', '6762', '6770', '6773', '6806', '6807', '6841',
            '6857', '6902', '6920', '6923', '6925', '6926', '6952', '6963', '6965',
            '6976', '6988', '7004', '7011', '7012', '7013', '7148', '7164', '7167',
            '7201', '7202', '7205', '7211', '7269', '7270', '7272', '7276'
        ]
        return tier2_symbols
        
    def get_tier3_symbols(self) -> List[str]:
        """Tier 3: 小型株・特殊銘柄（+500銘柄）"""
        tier3_symbols = [
            # 小型株・新興市場（成長性重視）
            '1435', '1439', '1447', '1515', '1518', '1726', '1766', '1780', '1789',
            '1878', '1882', '1888', '1890', '1893', '1899', '1951', '1952', '1954',
            '1973', '1975', '2121', '2122', '2124', '2127', '2130', '2131', '2146',
            '2148', '2150', '2151', '2154', '2158', '2160', '2164', '2168', '2170',
            '2173', '2175', '2176', '2181', '2183', '2186', '2193', '2195', '2201',
            '2203', '2206', '2208', '2209', '2211', '2212', '2216', '2218', '2220',
            '2264', '2266', '2267', '2281', '2287', '2291', '2293', '2296', '2304',
            '2306', '2309', '2317', '2325', '2327', '2329', '2331', '2334', '2336',
            '2337', '2340', '2345', '2351', '2352', '2353', '2354', '2359', '2362',
            '2366', '2367', '2370', '2371', '2372', '2373', '2374', '2375', '2376',
            '2378', '2379', '2381', '2383', '2384', '2385', '2388', '2389', '2391',
            '2395', '2397', '2399', '2402', '2403', '2404', '2405', '2407', '2408',
            '2410', '2412', '2413', '2415', '2418', '2427', '2428', '2429', '2435',
            '2437', '2438', '2440', '2441', '2442', '2443', '2445', '2464', '2467',
            '2471', '2472', '2473', '2477', '2481', '2483', '2485', '2487', '2489',
            '2491', '2492', '2493', '2497', '2498', '2499', '2501', '2533', '2540',
            '2542', '2543', '2544', '2545', '2551', '2573', '2590', '2593', '2594',
            '2597', '2598', '2599', '2607', '2613', '2651', '2670', '2681', '2685',
            '2692', '2694', '2695', '2702', '2715', '2726', '2729', '2730', '2731',
            '2733', '2734', '2735', '2737', '2739', '2742', '2743', '2751', '2753',
            '2754', '2760', '2763', '2764', '2768', '2784', '2809', '2810', '2811',
            '2812', '2815', '2820', '2831', '2834', '2836', '2837', '2841', '2842',
            '2844', '2845', '2853', '2855', '2856', '2860', '2875', '2882', '2883',
            '2884', '2886', '2891', '2897', '2899', '2904', '2905', '2908', '2915',
            '2917', '2918', '2922', '2925', '2927', '2929', '2930', '2933', '2934',
            '2935', '2936', '2937', '2939', '2940', '2945', '2948', '2950', '2951',
            '2952', '2954', '2955', '2957', '2970', '2975', '2982', '2983', '2986',
            '2987', '2991', '2993', '2995', '3001', '3002', '3003', '3004', '3021',
            '3023', '3024', '3025', '3028', '3036', '3048', '3050', '3064', '3065',
            '3067', '3068', '3075', '3076', '3079', '3088', '3089', '3098', '3107',
            '3110', '3111', '3112', '3113', '3115', '3116', '3121', '3132', '3133',
            '3134', '3135', '3139', '3140', '3150', '3151', '3153', '3154', '3160',
            '3161', '3162', '3166', '3168', '3169', '3171', '3176', '3178', '3179',
            '3180', '3181', '3182', '3183', '3185', '3186', '3187', '3191', '3192',
            '3194', '3195', '3196', '3198', '3199', '3201', '3202', '3204', '3221',
            '3222', '3232', '3241', '3248', '3254', '3255', '3258', '3264', '3266',
            '3267', '3268', '3269', '3276', '3289', '3291', '3292', '3293', '3297',
            '3299', '3328', '3329', '3333', '3341', '3347', '3350', '3352', '3355',
            '3358', '3374', '3375', '3377', '3385', '3387', '3388', '3392', '3393',
            '3395', '3398', '3415', '3421', '3422', '3435', '3437', '3440', '3441',
            '3445', '3446', '3447', '3449', '3453', '3454', '3455', '3458', '3459',
            '3461', '3462', '3465', '3466', '3467', '3468', '3469', '3471', '3472',
            '3479', '3482', '3484', '3486', '3491', '3492', '3496', '3497', '3498',
            '3524', '3526', '3529', '3532', '3533', '3537', '3538', '3539', '3540',
            '3542', '3548', '3549', '3553', '3558', '3561', '3564', '3565', '3566',
            '3567', '3569', '3571', '3577', '3580', '3581', '3584', '3588', '3589',
            '3593', '3596', '3604', '3606', '3608', '3612', '3613', '3630', '3633',
            '3639', '3640', '3645', '3646', '3649', '3650', '3653', '3658', '3661',
            '3662', '3663', '3664', '3665', '3666', '3667', '3669', '3670', '3671',
            '3672', '3673', '3675', '3676', '3677', '3678', '3679', '3680', '3682',
            '3684', '3685', '3687', '3688', '3689', '3690', '3691', '3695', '3696',
            '3697', '3698', '3699', '3700', '3701', '3703', '3708', '3709', '3710',
            '3715', '3716', '3719', '3723', '3724', '3726', '3727', '3734', '3735',
            '3738', '3739', '3741', '3747', '3748', '3753', '3758', '3763', '3766',
            '3768', '3770', '3771', '3773', '3775', '3777', '3778', '3779', '3788',
            '3793', '3796', '3799', '3804', '3807', '3810', '3814', '3815', '3817',
            '3822', '3824', '3825', '3826', '3834', '3835', '3836', '3837', '3839',
            '3840', '3841', '3842', '3843', '3845', '3848', '3849', '3851', '3852',
            '3853', '3854', '3856', '3857', '3858', '3859', '3861', '3865', '3866',
            '3867', '3868', '3877', '3878', '3880', '3881', '3883', '3884', '3885',
            '3886', '3892', '3895', '3896', '3901', '3902', '3903', '3905', '3907',
            '3908', '3909', '3911', '3914', '3915', '3916', '3917', '3918', '3919',
            '3921', '3922', '3923', '3924', '3925', '3926', '3928', '3929', '3931',
            '3933', '3934', '3935', '3936', '3937', '3939', '3940', '3942', '3943',
            '3944', '3945', '3946', '3947', '3948', '3949', '3950', '3951', '3953',
            '3954', '3955', '3956', '3957', '3958', '3959', '3960', '3961', '3963',
            '3964', '3965', '3966', '3967', '3968', '3969', '3970', '3971', '3972',
            '3973', '3975', '3976', '3977', '3978', '3979', '3981', '3982', '3985',
            '3986', '3987', '3988', '3989', '3990', '3991', '3992', '3994', '3995',
            '3996', '3997', '3998', '3999', '4004', '4005', '4007', '4008', '4009',
            '4013', '4014', '4031', '4033', '4045', '4046', '4047', '4053', '4055',
            '4056', '4057', '4058', '4059', '4060', '4062', '4095', '4097', '4099'
        ]
        return tier3_symbols
        
    def get_extended_symbol_set(self, include_small_cap: bool = False) -> List[str]:
        """拡張銘柄セット取得（後方互換性のため残存）"""
        if include_small_cap:
            return self.get_comprehensive_symbols()
        else:
            return self.get_extended_symbols()
            
    def get_extended_symbols(self) -> List[str]:
        """Tier 1 + Tier 2: 主要銘柄 + 中型株（~274銘柄）"""
        symbols = self.get_tier1_symbols() + self.get_tier2_symbols()
        return sorted(list(set(symbols)))  # 重複除去・ソート
        
    def get_comprehensive_symbols(self) -> List[str]:
        """Tier 1 + Tier 2 + Tier 3: 包括的銘柄（~774銘柄）"""
        symbols = self.get_tier1_symbols() + self.get_tier2_symbols() + self.get_tier3_symbols()
        return sorted(list(set(symbols)))  # 重複除去・ソート
        
    def get_all_tse_symbols(self) -> List[str]:
        """全東証銘柄（研究・バックテスト用）"""
        # 実際の実装では外部APIまたはCSVファイルから4000銘柄を取得
        # ここでは現実的な実装として包括的銘柄を返す
        return self.get_comprehensive_symbols()


# シングルトンインスタンス
tse = TokyoStockExchange()


def get_symbol_recommendations(risk_level: str = 'medium') -> List[str]:
    """リスクレベル別推奨銘柄"""
    risk_map = {
        'low': tse.get_tier1_symbols()[:20],      # 低リスク: 主要株20銘柄
        'medium': tse.get_extended_symbols(),     # 中リスク: 主要株+中型株
        'high': tse.get_comprehensive_symbols()   # 高リスク: 小型株含む包括セット
    }
    return risk_map.get(risk_level, tse.get_tier1_symbols())


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