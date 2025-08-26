#!/usr/bin/env python3
"""
TOPIX500 Analysis System - Data Loader

TOPIX500マスターデータの読み込みと管理機能
"""

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_classes import TOPIX500Symbol

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class DataLoader:
    """TOPIX500データロード機能"""

    def __init__(self):
        """データローダー初期化"""
        self.topix500_symbols = {}
        self.sector_mapping = {}
        logger.info("TOPIX500データローダー初期化完了")

    async def load_topix500_master_data(
        self, master_data_path: Optional[str] = None
    ) -> bool:
        """
        TOPIX500マスターデータ読み込み

        Args:
            master_data_path: マスターデータファイルパス（Noneの場合は模擬データ生成）

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            logger.info("TOPIX500マスターデータ読み込み開始")

            if master_data_path and Path(master_data_path).exists():
                # 実際のファイルから読み込み
                df = pd.read_csv(master_data_path)
                symbols_data = df.to_dict("records")
            else:
                # 模擬データ生成
                symbols_data = await self._generate_mock_topix500_data()

            # TOPIX500銘柄情報構築
            for data in symbols_data:
                symbol = TOPIX500Symbol(
                    symbol=data["symbol"],
                    name=data.get("name", f"Company_{data['symbol']}"),
                    sector=data.get("sector", "Technology"),
                    industry=data.get("industry", "Software"),
                    market_cap=data.get("market_cap", 100000000000),
                    weight_in_index=data.get("weight", 0.2),
                    listing_date=pd.to_datetime(data.get("listing_date", "2020-01-01")),
                    is_active=data.get("is_active", True),
                )

                self.topix500_symbols[symbol.symbol] = symbol

                # セクターマッピング構築
                if symbol.sector not in self.sector_mapping:
                    self.sector_mapping[symbol.sector] = []
                self.sector_mapping[symbol.sector].append(symbol.symbol)

            logger.info(
                f"TOPIX500マスターデータ読み込み完了: {len(self.topix500_symbols)}銘柄"
            )
            logger.info(f"セクター数: {len(self.sector_mapping)}")

            # セクター別銘柄数表示
            for sector, symbols in self.sector_mapping.items():
                logger.info(f"  {sector}: {len(symbols)}銘柄")

            return True

        except Exception as e:
            logger.error(f"TOPIX500マスターデータ読み込みエラー: {e}")
            traceback.print_exc()
            return False

    async def _generate_mock_topix500_data(self) -> List[Dict[str, Any]]:
        """TOPIX500模擬データ生成"""
        sectors = [
            "Technology",
            "Financials",
            "Consumer Discretionary",
            "Industrials",
            "Health Care",
            "Consumer Staples",
            "Materials",
            "Energy",
            "Utilities",
            "Real Estate",
            "Communication Services",
        ]

        industries = {
            "Technology": ["Software", "Semiconductors", "Hardware", "IT Services"],
            "Financials": ["Banks", "Insurance", "Securities", "REITs"],
            "Consumer Discretionary": ["Retail", "Automotive", "Media", "Restaurants"],
            "Industrials": ["Machinery", "Transportation", "Construction", "Aerospace"],
            "Health Care": [
                "Pharmaceuticals",
                "Medical Devices",
                "Biotechnology",
                "Healthcare Services",
            ],
        }

        symbols_data = []

        for i in range(500):
            sector = np.random.choice(sectors)
            industry_list = industries.get(sector, ["Other"])
            industry = np.random.choice(industry_list)

            # 銘柄コード生成（4桁数字）
            symbol = f"{1000 + i:04d}"

            # 市場キャップ（対数正規分布）
            market_cap = np.random.lognormal(22, 1.5)  # 約100億円中心

            # インデックス重み（市場キャップベース）
            weight = min(market_cap / 1000000000000 * 10, 5.0)  # 最大5%

            symbol_data = {
                "symbol": symbol,
                "name": f"Company_{symbol}",
                "sector": sector,
                "industry": industry,
                "market_cap": market_cap,
                "weight": weight,
                "listing_date": "2018-01-01",
                "is_active": True,
            }

            symbols_data.append(symbol_data)

        logger.info(f"TOPIX500模擬データ生成完了: {len(symbols_data)}銘柄")
        return symbols_data

    def get_symbols(self) -> Dict[str, TOPIX500Symbol]:
        """銘柄情報取得"""
        return self.topix500_symbols

    def get_sector_mapping(self) -> Dict[str, List[str]]:
        """セクターマッピング取得"""
        return self.sector_mapping
