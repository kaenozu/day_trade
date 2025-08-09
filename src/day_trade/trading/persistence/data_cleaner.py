"""
データクリーニング

重複除去・データ検証・整合性チェック機能
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

from ...utils.logging_config import get_context_logger, log_business_event
from ...utils.enhanced_error_handler import get_default_error_handler
from ..core.types import Trade, TradeStatus, TradeType
from .db_manager import TradeDatabaseManager

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


class DataCleaner:
    """
    データクリーニングクラス

    取引データの品質管理・重複除去・整合性検証機能を提供
    """

    def __init__(self, db_manager: TradeDatabaseManager):
        """
        初期化

        Args:
            db_manager: データベース管理インスタンス
        """
        self.db_manager = db_manager
        logger.info("データクリーナー初期化完了")

    def clean_trades_data(self, trades: List[Trade]) -> Dict[str, any]:
        """
        取引データ包括クリーニング

        Args:
            trades: 取引データリスト

        Returns:
            クリーニング結果辞書
        """
        start_time = datetime.now()
        original_count = len(trades)
        
        logger.info(f"データクリーニング開始: {original_count}件")
        
        try:
            # 各種クリーニング実行
            duplicate_result = self.remove_duplicates(trades)
            cleaned_trades = duplicate_result['cleaned_trades']
            
            validation_result = self.validate_trade_data(cleaned_trades)
            valid_trades = validation_result['valid_trades']
            
            consistency_result = self.check_data_consistency(valid_trades)
            consistent_trades = consistency_result['consistent_trades']
            
            # 結果統計
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                'original_count': original_count,
                'final_count': len(consistent_trades),
                'removed_duplicates': duplicate_result['duplicates_removed'],
                'invalid_trades': validation_result['invalid_count'],
                'inconsistent_trades': consistency_result['inconsistent_count'],
                'cleaned_trades': consistent_trades,
                'cleaning_time_seconds': processing_time,
                'data_quality_score': self._calculate_quality_score(original_count, len(consistent_trades)),
            }
            
            log_business_event(
                f"データクリーニング完了: {original_count}件→{len(consistent_trades)}件",
                result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"データクリーニングエラー: {e}")
            return {
                'original_count': original_count,
                'final_count': 0,
                'error': str(e),
                'cleaned_trades': [],
            }

    def remove_duplicates(self, trades: List[Trade]) -> Dict[str, any]:
        """
        重複取引データ除去

        Args:
            trades: 取引データリスト

        Returns:
            重複除去結果
        """
        original_count = len(trades)
        seen_trades: Set[str] = set()
        cleaned_trades: List[Trade] = []
        
        # 重複判定キー生成関数
        def generate_duplicate_key(trade: Trade) -> str:
            return f"{trade.symbol}_{trade.trade_type.value}_{trade.quantity}_{trade.price}_{trade.timestamp.isoformat()}"
        
        try:
            for trade in trades:
                duplicate_key = generate_duplicate_key(trade)
                
                if duplicate_key not in seen_trades:
                    seen_trades.add(duplicate_key)
                    cleaned_trades.append(trade)
                else:
                    logger.debug(f"重複取引除去: {trade.id}")
            
            duplicates_removed = original_count - len(cleaned_trades)
            
            result = {
                'original_count': original_count,
                'cleaned_count': len(cleaned_trades),
                'duplicates_removed': duplicates_removed,
                'cleaned_trades': cleaned_trades,
            }
            
            logger.info(f"重複除去完了: {duplicates_removed}件除去")
            return result
            
        except Exception as e:
            logger.error(f"重複除去エラー: {e}")
            return {
                'original_count': original_count,
                'cleaned_count': 0,
                'duplicates_removed': 0,
                'cleaned_trades': [],
                'error': str(e)
            }

    def validate_trade_data(self, trades: List[Trade]) -> Dict[str, any]:
        """
        取引データ検証

        Args:
            trades: 取引データリスト

        Returns:
            検証結果辞書
        """
        valid_trades: List[Trade] = []
        invalid_trades: List[Tuple[Trade, List[str]]] = []
        
        for trade in trades:
            validation_errors = self._validate_single_trade(trade)
            
            if not validation_errors:
                valid_trades.append(trade)
            else:
                invalid_trades.append((trade, validation_errors))
                logger.warning(f"無効な取引データ: {trade.id} - {validation_errors}")
        
        result = {
            'total_trades': len(trades),
            'valid_count': len(valid_trades),
            'invalid_count': len(invalid_trades),
            'valid_trades': valid_trades,
            'invalid_trades': invalid_trades,
        }
        
        logger.info(f"データ検証完了: {len(valid_trades)}件有効, {len(invalid_trades)}件無効")
        return result

    def _validate_single_trade(self, trade: Trade) -> List[str]:
        """
        単一取引データ検証

        Args:
            trade: 取引データ

        Returns:
            検証エラーリスト
        """
        errors = []
        
        try:
            # 基本データ検証
            if not trade.id or len(trade.id) < 5:
                errors.append("取引ID不正")
            
            if not trade.symbol or len(trade.symbol) < 1:
                errors.append("銘柄コード不正")
            
            if trade.quantity <= 0:
                errors.append("数量不正")
            
            if trade.price <= Decimal("0"):
                errors.append("価格不正")
            
            if trade.commission < Decimal("0"):
                errors.append("手数料不正")
            
            # 日時検証
            current_time = datetime.now()
            min_date = datetime(2000, 1, 1)  # 最小許可日時
            max_future_date = current_time + timedelta(days=1)  # 未来日制限
            
            if trade.timestamp < min_date:
                errors.append("取引日時が古すぎる")
            
            if trade.timestamp > max_future_date:
                errors.append("取引日時が未来すぎる")
            
            # 取引タイプ検証
            if trade.trade_type not in [TradeType.BUY, TradeType.SELL]:
                errors.append("取引タイプ不正")
            
            # ステータス検証
            valid_statuses = [TradeStatus.EXECUTED, TradeStatus.PENDING, TradeStatus.CANCELLED]
            if hasattr(trade, 'status') and trade.status not in valid_statuses:
                errors.append("取引ステータス不正")
            
            # 価格レンジ検証（明らかに異常な価格）
            if trade.price > Decimal("100000"):  # 10万円超
                errors.append("価格が高すぎる")
            
            if trade.price < Decimal("1"):  # 1円未満
                errors.append("価格が低すぎる")
            
            # 数量検証（単元株制）
            if trade.quantity % 100 != 0:
                errors.append("単元株数に準拠していない")
            
        except Exception as e:
            errors.append(f"検証処理エラー: {e}")
        
        return errors

    def check_data_consistency(self, trades: List[Trade]) -> Dict[str, any]:
        """
        データ整合性チェック

        Args:
            trades: 取引データリスト

        Returns:
            整合性チェック結果
        """
        consistent_trades: List[Trade] = []
        inconsistent_trades: List[Tuple[Trade, List[str]]] = []
        
        # 銘柄別取引履歴構築
        symbol_trades: Dict[str, List[Trade]] = {}
        for trade in trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)
        
        # 銘柄別に整合性チェック
        for symbol, symbol_trade_list in symbol_trades.items():
            # 時刻順でソート
            symbol_trade_list.sort(key=lambda t: t.timestamp)
            
            position_quantity = 0  # 仮想ポジション追跡
            
            for trade in symbol_trade_list:
                consistency_errors = []
                
                # 売買整合性チェック
                if trade.trade_type == TradeType.SELL:
                    if position_quantity < trade.quantity:
                        consistency_errors.append(
                            f"売却数量({trade.quantity})が保有数量({position_quantity})を超過"
                        )
                
                # ポジション更新
                if trade.trade_type == TradeType.BUY:
                    position_quantity += trade.quantity
                elif trade.trade_type == TradeType.SELL:
                    position_quantity -= trade.quantity
                
                # 手数料整合性チェック
                expected_commission = self._calculate_expected_commission(trade.quantity, trade.price)
                commission_diff = abs(trade.commission - expected_commission)
                if commission_diff > Decimal("10"):  # 10円以上の差異
                    consistency_errors.append(f"手数料異常: 実際{trade.commission}円 vs 期待{expected_commission}円")
                
                # 結果分類
                if not consistency_errors:
                    consistent_trades.append(trade)
                else:
                    inconsistent_trades.append((trade, consistency_errors))
                    logger.warning(f"整合性エラー: {trade.id} - {consistency_errors}")
        
        result = {
            'total_trades': len(trades),
            'consistent_count': len(consistent_trades),
            'inconsistent_count': len(inconsistent_trades),
            'consistent_trades': consistent_trades,
            'inconsistent_trades': inconsistent_trades,
        }
        
        logger.info(f"整合性チェック完了: {len(consistent_trades)}件整合, {len(inconsistent_trades)}件不整合")
        return result

    def _calculate_expected_commission(self, quantity: int, price: Decimal) -> Decimal:
        """
        期待手数料計算

        Args:
            quantity: 数量
            price: 価格

        Returns:
            期待手数料
        """
        trade_value = Decimal(quantity) * price
        commission_rate = Decimal("0.001")  # 0.1%
        commission = trade_value * commission_rate
        min_commission = Decimal("100")
        
        return max(commission, min_commission)

    def fix_data_issues(self, trades: List[Trade]) -> Dict[str, any]:
        """
        データ問題自動修正

        Args:
            trades: 取引データリスト

        Returns:
            修正結果辞書
        """
        fixed_trades: List[Trade] = []
        fix_count = 0
        
        for trade in trades:
            original_trade = trade
            fixed_trade = self._fix_single_trade(trade)
            
            if fixed_trade != original_trade:
                fix_count += 1
                logger.debug(f"取引データ修正: {trade.id}")
            
            fixed_trades.append(fixed_trade)
        
        result = {
            'total_trades': len(trades),
            'fixed_count': fix_count,
            'fixed_trades': fixed_trades,
        }
        
        logger.info(f"データ修正完了: {fix_count}件修正")
        return result

    def _fix_single_trade(self, trade: Trade) -> Trade:
        """
        単一取引データ修正

        Args:
            trade: 取引データ

        Returns:
            修正済み取引データ
        """
        # 修正可能な問題の自動修正
        fixed_trade = Trade(
            id=trade.id,
            symbol=trade.symbol.upper().strip(),  # 大文字化・空白除去
            trade_type=trade.trade_type,
            quantity=trade.quantity,
            price=trade.price,
            timestamp=trade.timestamp,
            commission=max(trade.commission, Decimal("0")),  # 負の手数料を0に修正
            status=getattr(trade, 'status', TradeStatus.EXECUTED),
            notes=getattr(trade, 'notes', '').strip(),
        )
        
        return fixed_trade

    def generate_data_quality_report(self, trades: List[Trade]) -> Dict[str, any]:
        """
        データ品質レポート生成

        Args:
            trades: 取引データリスト

        Returns:
            品質レポート辞書
        """
        try:
            # 各種品質チェック実行
            duplicate_result = self.remove_duplicates(trades.copy())
            validation_result = self.validate_trade_data(trades)
            consistency_result = self.check_data_consistency(trades)
            
            # 統計計算
            total_trades = len(trades)
            quality_issues = (
                duplicate_result['duplicates_removed'] +
                validation_result['invalid_count'] +
                consistency_result['inconsistent_count']
            )
            
            quality_score = self._calculate_quality_score(total_trades, total_trades - quality_issues)
            
            # 銘柄別統計
            symbol_stats = self._calculate_symbol_statistics(trades)
            
            # 時系列統計
            temporal_stats = self._calculate_temporal_statistics(trades)
            
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'total_trades': total_trades,
                'quality_score': quality_score,
                'quality_issues': {
                    'duplicates': duplicate_result['duplicates_removed'],
                    'invalid_data': validation_result['invalid_count'],
                    'inconsistent_data': consistency_result['inconsistent_count'],
                },
                'symbol_statistics': symbol_stats,
                'temporal_statistics': temporal_stats,
                'recommendations': self._generate_data_quality_recommendations(
                    duplicate_result, validation_result, consistency_result
                ),
            }
            
            logger.info(f"データ品質レポート生成完了: スコア{quality_score:.1f}%")
            return report
            
        except Exception as e:
            logger.error(f"品質レポート生成エラー: {e}")
            return {
                'error': str(e),
                'report_timestamp': datetime.now().isoformat(),
            }

    def _calculate_quality_score(self, total: int, clean: int) -> float:
        """データ品質スコア計算"""
        return (clean / total * 100) if total > 0 else 0.0

    def _calculate_symbol_statistics(self, trades: List[Trade]) -> Dict[str, any]:
        """銘柄別統計計算"""
        symbol_counts = {}
        for trade in trades:
            symbol_counts[trade.symbol] = symbol_counts.get(trade.symbol, 0) + 1
        
        return {
            'total_symbols': len(symbol_counts),
            'trades_per_symbol': symbol_counts,
            'most_traded_symbol': max(symbol_counts.items(), key=lambda x: x[1])[0] if symbol_counts else None,
        }

    def _calculate_temporal_statistics(self, trades: List[Trade]) -> Dict[str, any]:
        """時系列統計計算"""
        if not trades:
            return {}
        
        timestamps = [trade.timestamp for trade in trades]
        
        return {
            'earliest_trade': min(timestamps).isoformat(),
            'latest_trade': max(timestamps).isoformat(),
            'time_span_days': (max(timestamps) - min(timestamps)).days,
        }

    def _generate_data_quality_recommendations(
        self, 
        duplicate_result: Dict,
        validation_result: Dict,
        consistency_result: Dict
    ) -> List[str]:
        """データ品質改善推奨事項生成"""
        recommendations = []
        
        if duplicate_result['duplicates_removed'] > 0:
            recommendations.append("重複データ除去処理の定期実行を推奨")
        
        if validation_result['invalid_count'] > 0:
            recommendations.append("データ入力時の検証強化を推奨")
        
        if consistency_result['inconsistent_count'] > 0:
            recommendations.append("取引記録の整合性チェック強化を推奨")
        
        return recommendations

    def archive_old_trades(self, cutoff_days: int = 365) -> Dict[str, int]:
        """
        古い取引データのアーカイブ

        Args:
            cutoff_days: アーカイブ対象日数

        Returns:
            アーカイブ結果
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=cutoff_days)
            
            # 古い取引データ取得
            all_trades = self.db_manager.load_trades_from_db()
            archive_trades = [t for t in all_trades if t.timestamp < cutoff_date]
            
            if not archive_trades:
                logger.info("アーカイブ対象データなし")
                return {'archived_count': 0}
            
            # アーカイブファイル作成
            archive_filename = f"trades_archive_{cutoff_date.strftime('%Y%m%d')}.json"
            archive_path = f"./archives/{archive_filename}"
            
            # バックアップ実行
            backup_success = self.db_manager.backup_database(archive_path)
            
            archived_count = len(archive_trades) if backup_success else 0
            
            result = {
                'archived_count': archived_count,
                'archive_path': archive_path if backup_success else None,
                'cutoff_date': cutoff_date.isoformat(),
            }
            
            logger.info(f"アーカイブ完了: {archived_count}件")
            return result
            
        except Exception as e:
            logger.error(f"アーカイブエラー: {e}")
            return {'archived_count': 0, 'error': str(e)}