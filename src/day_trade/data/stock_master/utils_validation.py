"""
銘柄マスタのバリデーション機能モジュール

このモジュールは銘柄データの検証、データ品質分析機能を提供します。
"""

from typing import Dict, List

from sqlalchemy import func

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockValidationUtils:
    """銘柄データバリデーション・品質分析クラス"""

    def __init__(self, db_manager, config=None):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
            config: 設定オブジェクト
        """
        self.db_manager = db_manager
        self.config = config or {}

    def validate_stock_data(self, code: str, name: str = None) -> bool:
        """
        銘柄データのバリデーション

        Args:
            code: 証券コード
            name: 銘柄名

        Returns:
            バリデーション結果
        """
        if self.config.get("require_code", True) and not code:
            return False
        if self.config.get("require_name", True) and not name:
            return False
        if self.config.get("validate_code_format", True) and not code.isdigit():
            return False
        if name and len(name) > self.config.get("max_name_length", 100):
            return False
        return True

    def analyze_data_quality(self) -> Dict[str, float]:
        """
        データ品質の分析

        Returns:
            品質指標の辞書
        """
        try:
            with self.db_manager.session_scope() as session:
                # 総銘柄数
                total_stocks = session.query(func.count(Stock.id)).scalar()

                if total_stocks == 0:
                    return {"completeness": 0.0, "validity": 0.0, "consistency": 0.0}

                # 完全性（欠損データの割合）
                complete_stocks = (
                    session.query(func.count(Stock.id))
                    .filter(Stock.code.isnot(None))
                    .filter(Stock.name.isnot(None))
                    .filter(Stock.sector.isnot(None))
                    .filter(Stock.industry.isnot(None))
                    .scalar()
                )
                completeness = complete_stocks / total_stocks

                # 有効性（正しいフォーマットのコードの割合）
                # 簡易的に4桁の数値をチェック
                valid_code_pattern = r'^[0-9]{4}$'
                valid_codes = (
                    session.query(func.count(Stock.id))
                    .filter(Stock.code.op('REGEXP')(valid_code_pattern))
                    .scalar()
                )
                validity = valid_codes / total_stocks

                # 一貫性（重複コードのチェック）
                unique_codes = session.query(Stock.code).distinct().count()
                consistency = unique_codes / total_stocks

                return {
                    "completeness": round(completeness, 4),
                    "validity": round(validity, 4),
                    "consistency": round(consistency, 4),
                    "overall_quality": round((completeness + validity + consistency) / 3, 4),
                }

        except Exception as e:
            logger.error(f"データ品質分析エラー: {e}")
            return {"completeness": 0.0, "validity": 0.0, "consistency": 0.0}

    def get_missing_data_report(self) -> Dict[str, List[str]]:
        """
        欠損データのレポートを生成

        Returns:
            欠損項目別の銘柄コードリスト
        """
        try:
            missing_report = {}

            with self.db_manager.session_scope() as session:
                # 名前が欠損している銘柄
                missing_name = (
                    session.query(Stock.code)
                    .filter((Stock.name.is_(None)) | (Stock.name == ""))
                    .limit(50)  # レポートサイズ制限
                    .all()
                )
                missing_report["missing_name"] = [stock.code for stock in missing_name]

                # セクターが欠損している銘柄
                missing_sector = (
                    session.query(Stock.code)
                    .filter((Stock.sector.is_(None)) | (Stock.sector == ""))
                    .limit(50)
                    .all()
                )
                missing_report["missing_sector"] = [stock.code for stock in missing_sector]

                # 業種が欠損している銘柄
                missing_industry = (
                    session.query(Stock.code)
                    .filter((Stock.industry.is_(None)) | (Stock.industry == ""))
                    .limit(50)
                    .all()
                )
                missing_report["missing_industry"] = [stock.code for stock in missing_industry]

                # 市場区分が欠損している銘柄
                missing_market = (
                    session.query(Stock.code)
                    .filter((Stock.market.is_(None)) | (Stock.market == ""))
                    .limit(50)
                    .all()
                )
                missing_report["missing_market"] = [stock.code for stock in missing_market]

            return missing_report

        except Exception as e:
            logger.error(f"欠損データレポート生成エラー: {e}")
            return {}

    def get_duplicate_codes(self) -> List[str]:
        """
        重複する証券コードを取得

        Returns:
            重複する証券コードのリスト
        """
        try:
            with self.db_manager.session_scope() as session:
                # 重複するコードを検索
                duplicate_codes = (
                    session.query(Stock.code)
                    .group_by(Stock.code)
                    .having(func.count(Stock.code) > 1)
                    .all()
                )
                
                return [code.code for code in duplicate_codes]

        except Exception as e:
            logger.error(f"重複コード取得エラー: {e}")
            return []

    def validate_code_format(self, code: str) -> bool:
        """
        証券コードのフォーマット検証

        Args:
            code: 証券コード

        Returns:
            フォーマットが正しいかどうか
        """
        if not code:
            return False
        
        # 基本的には4桁の数字
        if len(code) == 4 and code.isdigit():
            return True
            
        # 一部のREITや投資信託は異なるフォーマット
        if len(code) == 5 and code.isdigit():
            return True
            
        return False

    def get_format_violation_report(self) -> List[Dict[str, str]]:
        """
        フォーマット違反の銘柄レポートを取得

        Returns:
            フォーマット違反銘柄の情報リスト
        """
        try:
            violations = []
            
            with self.db_manager.session_scope() as session:
                stocks = session.query(Stock).all()
                
                for stock in stocks:
                    issues = []
                    
                    # コードフォーマットチェック
                    if not self.validate_code_format(stock.code):
                        issues.append("無効なコードフォーマット")
                    
                    # 名前の長さチェック
                    max_name_length = self.config.get("max_name_length", 100)
                    if stock.name and len(stock.name) > max_name_length:
                        issues.append(f"名前が長すぎる(>{max_name_length}文字)")
                    
                    if issues:
                        violations.append({
                            "code": stock.code,
                            "name": stock.name or "",
                            "issues": ", ".join(issues)
                        })
                
                return violations[:50]  # 上位50件に制限
                
        except Exception as e:
            logger.error(f"フォーマット違反レポート生成エラー: {e}")
            return []