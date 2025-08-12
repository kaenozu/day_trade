#!/usr/bin/env python3
"""
TOPIX500銘柄マスター管理システム
Issue #314: TOPIX500全銘柄対応

TOPIX500構成銘柄の管理とセクター分類システム
"""

import sqlite3
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ..utils.logging_config import get_context_logger
except ImportError:

    def get_context_logger(name):
        import logging

        return logging.getLogger(name)


logger = get_context_logger(__name__)


class TOPIX500MasterManager:
    """
    TOPIX500銘柄マスター管理システム

    銘柄情報・セクター分類・業種データの統合管理
    """

    def __init__(self, db_path: str = "data/topix500_master.db"):
        """
        初期化

        Args:
            db_path: データベースファイルパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # データベース初期化
        self._initialize_database()

        # セクター分類マッピング
        self.sector_mapping = self._load_sector_mapping()

        logger.info("TOPIX500マスター管理システム初期化完了")

    def _initialize_database(self):
        """データベースの初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # TOPIX500銘柄マスターテーブル
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS topix500_master (
                        code TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        sector_code TEXT NOT NULL,
                        sector_name TEXT NOT NULL,
                        subsector_name TEXT,
                        market_cap BIGINT,
                        listing_date DATE,
                        topix_weight REAL,
                        is_active BOOLEAN DEFAULT TRUE,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data_source TEXT DEFAULT 'manual'
                    )
                """
                )

                # セクター分類マスターテーブル
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sector_master (
                        sector_code TEXT PRIMARY KEY,
                        sector_name TEXT NOT NULL,
                        sector_name_en TEXT,
                        parent_sector TEXT,
                        sector_description TEXT,
                        avg_volatility REAL,
                        typical_pe_ratio REAL,
                        growth_characteristics TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # パフォーマンス監視テーブル
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS processing_performance (
                        date DATE,
                        total_symbols INTEGER,
                        successful_symbols INTEGER,
                        failed_symbols INTEGER,
                        processing_time_seconds REAL,
                        memory_usage_mb REAL,
                        error_details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.commit()
                logger.info("データベーステーブル初期化完了")

        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
            raise

    def _load_sector_mapping(self) -> Dict[str, Dict]:
        """
        セクター分類マッピングの読み込み

        Returns:
            セクター情報辞書
        """
        # 東証33業種分類に基づくセクターマッピング
        sector_data = {
            "0050": {
                "name": "水産・農林業",
                "name_en": "Fishery, Agriculture & Forestry",
                "parent": "食品",
                "volatility": 0.25,
                "pe_ratio": 15.0,
            },
            "1050": {
                "name": "鉱業",
                "name_en": "Mining",
                "parent": "資源",
                "volatility": 0.35,
                "pe_ratio": 12.0,
            },
            "2050": {
                "name": "建設業",
                "name_en": "Construction",
                "parent": "建設・不動産",
                "volatility": 0.30,
                "pe_ratio": 14.0,
            },
            "3050": {
                "name": "食料品",
                "name_en": "Foods",
                "parent": "食品",
                "volatility": 0.18,
                "pe_ratio": 18.0,
            },
            "3100": {
                "name": "繊維製品",
                "name_en": "Textiles & Apparels",
                "parent": "消費財",
                "volatility": 0.28,
                "pe_ratio": 16.0,
            },
            "3150": {
                "name": "パルプ・紙",
                "name_en": "Pulp & Paper",
                "parent": "素材",
                "volatility": 0.24,
                "pe_ratio": 13.0,
            },
            "3200": {
                "name": "化学",
                "name_en": "Chemical",
                "parent": "素材",
                "volatility": 0.26,
                "pe_ratio": 16.0,
            },
            "3250": {
                "name": "医薬品",
                "name_en": "Pharmaceutical",
                "parent": "ヘルスケア",
                "volatility": 0.22,
                "pe_ratio": 24.0,
            },
            "3300": {
                "name": "石油・石炭製品",
                "name_en": "Oil & Coal Products",
                "parent": "エネルギー",
                "volatility": 0.32,
                "pe_ratio": 10.0,
            },
            "3350": {
                "name": "ゴム製品",
                "name_en": "Rubber Products",
                "parent": "素材",
                "volatility": 0.27,
                "pe_ratio": 15.0,
            },
            "3400": {
                "name": "ガラス・土石製品",
                "name_en": "Glass & Ceramics Products",
                "parent": "素材",
                "volatility": 0.29,
                "pe_ratio": 14.0,
            },
            "3450": {
                "name": "鉄鋼",
                "name_en": "Iron & Steel",
                "parent": "素材",
                "volatility": 0.34,
                "pe_ratio": 11.0,
            },
            "3500": {
                "name": "非鉄金属",
                "name_en": "Nonferrous Metals",
                "parent": "素材",
                "volatility": 0.36,
                "pe_ratio": 13.0,
            },
            "3550": {
                "name": "金属製品",
                "name_en": "Metal Products",
                "parent": "素材",
                "volatility": 0.28,
                "pe_ratio": 15.0,
            },
            "3600": {
                "name": "機械",
                "name_en": "Machinery",
                "parent": "機械",
                "volatility": 0.30,
                "pe_ratio": 17.0,
            },
            "3650": {
                "name": "電気機器",
                "name_en": "Electric Appliances",
                "parent": "テクノロジー",
                "volatility": 0.32,
                "pe_ratio": 20.0,
            },
            "3700": {
                "name": "輸送用機器",
                "name_en": "Transportation Equipment",
                "parent": "自動車",
                "volatility": 0.28,
                "pe_ratio": 16.0,
            },
            "3750": {
                "name": "精密機器",
                "name_en": "Precision Instruments",
                "parent": "テクノロジー",
                "volatility": 0.31,
                "pe_ratio": 22.0,
            },
            "3800": {
                "name": "その他製品",
                "name_en": "Other Products",
                "parent": "その他",
                "volatility": 0.26,
                "pe_ratio": 17.0,
            },
            "4050": {
                "name": "電気・ガス業",
                "name_en": "Electric Power & Gas",
                "parent": "インフラ",
                "volatility": 0.20,
                "pe_ratio": 14.0,
            },
            "5050": {
                "name": "陸運業",
                "name_en": "Land Transportation",
                "parent": "運輸",
                "volatility": 0.25,
                "pe_ratio": 15.0,
            },
            "5100": {
                "name": "海運業",
                "name_en": "Marine Transportation",
                "parent": "運輸",
                "volatility": 0.45,
                "pe_ratio": 12.0,
            },
            "5150": {
                "name": "空運業",
                "name_en": "Air Transportation",
                "parent": "運輸",
                "volatility": 0.40,
                "pe_ratio": 13.0,
            },
            "5200": {
                "name": "倉庫・運輸関連業",
                "name_en": "Warehousing & Harbor Transportation Services",
                "parent": "運輸",
                "volatility": 0.27,
                "pe_ratio": 16.0,
            },
            "5250": {
                "name": "情報・通信業",
                "name_en": "Information & Communication",
                "parent": "テクノロジー",
                "volatility": 0.35,
                "pe_ratio": 25.0,
            },
            "6050": {
                "name": "卸売業",
                "name_en": "Wholesale Trade",
                "parent": "商社・卸売",
                "volatility": 0.24,
                "pe_ratio": 14.0,
            },
            "6100": {
                "name": "小売業",
                "name_en": "Retail Trade",
                "parent": "小売",
                "volatility": 0.28,
                "pe_ratio": 18.0,
            },
            "7050": {
                "name": "銀行業",
                "name_en": "Banks",
                "parent": "金融",
                "volatility": 0.32,
                "pe_ratio": 8.0,
            },
            "7100": {
                "name": "証券・商品先物取引業",
                "name_en": "Securities & Commodity Futures",
                "parent": "金融",
                "volatility": 0.48,
                "pe_ratio": 12.0,
            },
            "7150": {
                "name": "保険業",
                "name_en": "Insurance",
                "parent": "金融",
                "volatility": 0.30,
                "pe_ratio": 10.0,
            },
            "7200": {
                "name": "その他金融業",
                "name_en": "Other Financing Business",
                "parent": "金融",
                "volatility": 0.35,
                "pe_ratio": 15.0,
            },
            "8050": {
                "name": "不動産業",
                "name_en": "Real Estate",
                "parent": "建設・不動産",
                "volatility": 0.33,
                "pe_ratio": 16.0,
            },
            "9050": {
                "name": "サービス業",
                "name_en": "Services",
                "parent": "サービス",
                "volatility": 0.30,
                "pe_ratio": 20.0,
            },
        }

        return sector_data

    def initialize_sector_master(self):
        """セクターマスターデータの初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for sector_code, sector_info in self.sector_mapping.items():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO sector_master (
                            sector_code, sector_name, sector_name_en, parent_sector,
                            avg_volatility, typical_pe_ratio
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            sector_code,
                            sector_info["name"],
                            sector_info["name_en"],
                            sector_info["parent"],
                            sector_info["volatility"],
                            sector_info["pe_ratio"],
                        ),
                    )

                conn.commit()
                logger.info(f"セクターマスターデータ初期化完了: {len(self.sector_mapping)}セクター")

        except Exception as e:
            logger.error(f"セクターマスター初期化エラー: {e}")
            raise

    def load_topix500_sample_data(self):
        """
        TOPIX500サンプルデータの読み込み

        実際の運用では、公式データソースから取得
        """
        try:
            # 主要TOPIX500銘柄のサンプルデータ
            sample_stocks = [
                # テクノロジー・電気機器
                ("6758", "ソニーグループ", "3650", "電気機器"),
                ("9984", "ソフトバンクグループ", "5250", "情報・通信業"),
                ("4689", "ヤフー", "5250", "情報・通信業"),
                ("4307", "野村総合研究所", "5250", "情報・通信業"),
                ("9434", "ソフトバンク", "5250", "情報・通信業"),
                # 自動車・輸送機器
                ("7203", "トヨタ自動車", "3700", "輸送用機器"),
                ("7267", "ホンダ", "3700", "輸送用機器"),
                ("7201", "日産自動車", "3700", "輸送用機器"),
                ("7269", "スズキ", "3700", "輸送用機器"),
                ("7261", "マツダ", "3700", "輸送用機器"),
                # 金融
                ("8306", "三菱UFJフィナンシャル・グループ", "7050", "銀行業"),
                ("8316", "三井住友フィナンシャルグループ", "7050", "銀行業"),
                ("8411", "みずほフィナンシャルグループ", "7050", "銀行業"),
                ("8601", "大和証券グループ本社", "7100", "証券・商品先物取引業"),
                ("8725", "MS&ADインシュアランスグループ", "7150", "保険業"),
                # 商社
                ("8058", "三菱商事", "6050", "卸売業"),
                ("8031", "三井物産", "6050", "卸売業"),
                ("8053", "住友商事", "6050", "卸売業"),
                ("2768", "双日", "6050", "卸売業"),
                ("8002", "丸紅", "6050", "卸売業"),
                # 素材・化学
                ("4063", "信越化学工業", "3200", "化学"),
                ("4005", "住友化学", "3200", "化学"),
                ("4183", "三井化学", "3200", "化学"),
                ("5401", "日本製鉄", "3450", "鉄鋼"),
                ("5406", "神戸製鋼所", "3450", "鉄鋼"),
                # 医薬品・ヘルスケア
                ("4568", "第一三共", "3250", "医薬品"),
                ("4523", "エーザイ", "3250", "医薬品"),
                ("4507", "塩野義製薬", "3250", "医薬品"),
                ("4502", "武田薬品工業", "3250", "医薬品"),
                ("4503", "アステラス製薬", "3250", "医薬品"),
                # 消費財・小売
                ("9983", "ファーストリテイリング", "6100", "小売業"),
                ("3382", "セブン&アイ・ホールディングス", "6100", "小売業"),
                ("8267", "イオン", "6100", "小売業"),
                ("2914", "日本たばこ産業", "3050", "食料品"),
                ("2502", "アサヒグループホールディングス", "3050", "食料品"),
                # インフラ・電力
                ("9501", "東京電力ホールディングス", "4050", "電気・ガス業"),
                ("9503", "関西電力", "4050", "電気・ガス業"),
                ("9531", "東京ガス", "4050", "電気・ガス業"),
                ("9064", "ヤマトホールディングス", "5050", "陸運業"),
                ("9020", "東日本旅客鉄道", "5050", "陸運業"),
                # 不動産・建設
                ("8801", "三井不動産", "8050", "不動産業"),
                ("8802", "三菱地所", "8050", "不動産業"),
                ("1812", "鹿島建設", "2050", "建設業"),
                ("1801", "大成建設", "2050", "建設業"),
                ("1802", "大林組", "2050", "建設業"),
                # 機械・精密機器
                ("6503", "三菱電機", "3650", "電気機器"),
                ("6501", "日立製作所", "3650", "電気機器"),
                ("7751", "キヤノン", "3750", "精密機器"),
                ("6954", "ファナック", "3600", "機械"),
                ("6367", "ダイキン工業", "3600", "機械"),
            ]

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for code, name, sector_code, sector_name in sample_stocks:
                    # 時価総額は簡易計算（実際のデータでは外部APIから取得）
                    mock_market_cap = hash(code) % 10000000 + 1000000  # 100万-1000万のモック値

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO topix500_master (
                            code, name, sector_code, sector_name, market_cap,
                            topix_weight, is_active, data_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            code,
                            name,
                            sector_code,
                            sector_name,
                            mock_market_cap,
                            0.5,
                            True,
                            "sample_data",  # デフォルトweight
                        ),
                    )

                conn.commit()
                logger.info(f"TOPIX500サンプルデータ読み込み完了: {len(sample_stocks)}銘柄")

        except Exception as e:
            logger.error(f"サンプルデータ読み込みエラー: {e}")
            raise

    def get_all_active_symbols(self) -> List[str]:
        """
        アクティブな全銘柄コード取得

        Returns:
            アクティブ銘柄コードリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT code FROM topix500_master
                    WHERE is_active = TRUE
                    ORDER BY market_cap DESC
                """
                )

                symbols = [row[0] for row in cursor.fetchall()]
                logger.info(f"アクティブ銘柄取得: {len(symbols)}銘柄")
                return symbols

        except Exception as e:
            logger.error(f"銘柄取得エラー: {e}")
            return []

    def get_symbols_by_sector(self, sector_code: str) -> List[Dict]:
        """
        セクター別銘柄取得

        Args:
            sector_code: セクターコード

        Returns:
            セクター銘柄情報リスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT code, name, market_cap, topix_weight
                    FROM topix500_master
                    WHERE sector_code = ? AND is_active = TRUE
                    ORDER BY market_cap DESC
                """,
                    (sector_code,),
                )

                results = []
                for row in cursor.fetchall():
                    results.append(
                        {
                            "code": row[0],
                            "name": row[1],
                            "market_cap": row[2],
                            "topix_weight": row[3],
                        }
                    )

                logger.info(f"セクター別銘柄取得: {sector_code} - {len(results)}銘柄")
                return results

        except Exception as e:
            logger.error(f"セクター別銘柄取得エラー: {e}")
            return []

    def get_sector_summary(self) -> Dict[str, Dict]:
        """
        セクターサマリー情報取得

        Returns:
            セクター別統計情報
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        tm.sector_code,
                        tm.sector_name,
                        COUNT(*) as stock_count,
                        AVG(tm.market_cap) as avg_market_cap,
                        SUM(tm.topix_weight) as total_weight,
                        sm.avg_volatility,
                        sm.typical_pe_ratio
                    FROM topix500_master tm
                    LEFT JOIN sector_master sm ON tm.sector_code = sm.sector_code
                    WHERE tm.is_active = TRUE
                    GROUP BY tm.sector_code, tm.sector_name
                    ORDER BY total_weight DESC
                """
                )

                sector_summary = {}
                for row in cursor.fetchall():
                    sector_summary[row[0]] = {
                        "sector_name": row[1],
                        "stock_count": row[2],
                        "avg_market_cap": row[3],
                        "total_weight": row[4],
                        "avg_volatility": row[5],
                        "typical_pe_ratio": row[6],
                    }

                logger.info(f"セクターサマリー取得: {len(sector_summary)}セクター")
                return sector_summary

        except Exception as e:
            logger.error(f"セクターサマリー取得エラー: {e}")
            return {}

    def create_balanced_batches(self, batch_size: int = 50) -> List[List[str]]:
        """
        バランス考慮バッチ作成

        セクター分散を考慮した効率的なバッチ分割

        Args:
            batch_size: バッチサイズ

        Returns:
            バランス考慮銘柄バッチリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT code, sector_code, market_cap
                    FROM topix500_master
                    WHERE is_active = TRUE
                    ORDER BY sector_code, market_cap DESC
                """
                )

                stocks_by_sector = {}
                for row in cursor.fetchall():
                    sector = row[1]
                    if sector not in stocks_by_sector:
                        stocks_by_sector[sector] = []
                    stocks_by_sector[sector].append(row[0])

                # セクター分散バッチ作成
                batches = []
                current_batch = []
                sector_rotation = list(stocks_by_sector.keys())
                sector_idx = 0

                while any(stocks_by_sector.values()):
                    if len(current_batch) >= batch_size:
                        batches.append(current_batch)
                        current_batch = []

                    # 現在のセクターから銘柄を取得
                    current_sector = sector_rotation[sector_idx % len(sector_rotation)]
                    if stocks_by_sector[current_sector]:
                        symbol = stocks_by_sector[current_sector].pop(0)
                        current_batch.append(symbol)

                    sector_idx += 1

                    # 空のセクターを除去
                    if not stocks_by_sector[current_sector]:
                        sector_rotation.remove(current_sector)
                        if not sector_rotation:
                            break

                if current_batch:
                    batches.append(current_batch)

                logger.info(f"バランス考慮バッチ作成完了: {len(batches)}バッチ")
                return batches

        except Exception as e:
            logger.error(f"バッチ作成エラー: {e}")
            return []

    def record_processing_performance(
        self,
        total_symbols: int,
        successful_symbols: int,
        processing_time: float,
        memory_usage: float,
        error_details: Optional[str] = None,
    ):
        """
        処理パフォーマンス記録

        Args:
            total_symbols: 処理対象銘柄数
            successful_symbols: 成功銘柄数
            processing_time: 処理時間（秒）
            memory_usage: メモリ使用量（MB）
            error_details: エラー詳細
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO processing_performance (
                        date, total_symbols, successful_symbols, failed_symbols,
                        processing_time_seconds, memory_usage_mb, error_details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        date.today(),
                        total_symbols,
                        successful_symbols,
                        total_symbols - successful_symbols,
                        processing_time,
                        memory_usage,
                        error_details,
                    ),
                )

                conn.commit()
                logger.info("処理パフォーマンス記録完了")

        except Exception as e:
            logger.error(f"パフォーマンス記録エラー: {e}")

    def get_performance_statistics(self, days: int = 30) -> Dict:
        """
        パフォーマンス統計取得

        Args:
            days: 過去日数

        Returns:
            パフォーマンス統計情報
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT
                        AVG(processing_time_seconds) as avg_time,
                        MIN(processing_time_seconds) as min_time,
                        MAX(processing_time_seconds) as max_time,
                        AVG(memory_usage_mb) as avg_memory,
                        AVG(CAST(successful_symbols AS FLOAT) / total_symbols * 100) as success_rate,
                        COUNT(*) as total_runs
                    FROM processing_performance
                    WHERE date >= date('now', '-{days} days')
                """
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "avg_processing_time": row[0] or 0,
                        "min_processing_time": row[1] or 0,
                        "max_processing_time": row[2] or 0,
                        "avg_memory_usage": row[3] or 0,
                        "success_rate": row[4] or 0,
                        "total_runs": row[5] or 0,
                    }
                else:
                    return {}

        except Exception as e:
            logger.error(f"パフォーマンス統計取得エラー: {e}")
            return {}


if __name__ == "__main__":
    # テスト実行
    print("=== TOPIX500マスター管理システム テスト ===")

    try:
        # システム初期化
        master_manager = TOPIX500MasterManager()

        # セクターマスター初期化
        print("1. セクターマスター初期化...")
        master_manager.initialize_sector_master()

        # サンプルデータ読み込み
        print("2. TOPIX500サンプルデータ読み込み...")
        master_manager.load_topix500_sample_data()

        # アクティブ銘柄取得
        print("3. アクティブ銘柄取得テスト...")
        active_symbols = master_manager.get_all_active_symbols()
        print(f"   アクティブ銘柄数: {len(active_symbols)}")
        print(f"   上位5銘柄: {active_symbols[:5]}")

        # セクターサマリー
        print("4. セクターサマリー取得テスト...")
        sector_summary = master_manager.get_sector_summary()
        print(f"   セクター数: {len(sector_summary)}")

        for sector_code, info in list(sector_summary.items())[:5]:
            print(f"   {sector_code}: {info['sector_name']} ({info['stock_count']}銘柄)")

        # バッチ作成テスト
        print("5. バランス考慮バッチ作成テスト...")
        batches = master_manager.create_balanced_batches(batch_size=10)
        print(f"   作成バッチ数: {len(batches)}")
        print(f"   バッチサイズ: {[len(batch) for batch in batches]}")

        # パフォーマンス記録テスト
        print("6. パフォーマンス記録テスト...")
        master_manager.record_processing_performance(
            total_symbols=len(active_symbols),
            successful_symbols=len(active_symbols) - 2,
            processing_time=15.5,
            memory_usage=750.0,
        )

        performance = master_manager.get_performance_statistics()
        if performance:
            print(f"   平均処理時間: {performance['avg_processing_time']:.1f}秒")
            print(f"   成功率: {performance['success_rate']:.1f}%")

        print("\n✅ TOPIX500マスター管理システム テスト完了！")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
