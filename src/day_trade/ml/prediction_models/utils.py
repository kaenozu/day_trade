#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ユーティリティモジュール

共通的なヘルパー関数と定数定義
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """ロギング設定のセットアップ"""
    logger = logging.getLogger("ml_prediction_models")
    
    if logger.handlers:
        return logger
    
    # ログレベル設定
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラー（指定された場合）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_symbol(symbol: str) -> bool:
    """銘柄コード検証"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # 基本的な検証ルール
    symbol = symbol.strip().upper()
    
    # 長さチェック
    if len(symbol) < 1 or len(symbol) > 10:
        return False
    
    # 英数字チェック
    if not symbol.replace('.', '').isalnum():
        return False
    
    return True


def sanitize_model_id(model_id: str) -> str:
    """モデルIDのサニタイズ"""
    if not model_id:
        return "unknown_model"
    
    # 安全な文字のみ許可
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    sanitized = ''.join(c if c in safe_chars else '_' for c in str(model_id))
    
    # 長さ制限
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    
    return sanitized


def format_performance_metrics(metrics: Dict[str, float], precision: int = 3) -> Dict[str, str]:
    """性能メトリクスのフォーマット"""
    formatted = {}
    
    for key, value in metrics.items():
        if value is None:
            formatted[key] = "N/A"
        elif isinstance(value, float):
            if key.endswith('_score') or key.endswith('_rate'):
                # パーセンテージ表示
                formatted[key] = f"{value * 100:.{precision-2}f}%"
            else:
                formatted[key] = f"{value:.{precision}f}"
        else:
            formatted[key] = str(value)
    
    return formatted


def create_model_summary_report(summary: Dict[str, Any]) -> str:
    """モデルサマリーレポートの作成"""
    lines = []
    lines.append("="*60)
    lines.append("ML Prediction Models Summary Report")
    lines.append("="*60)
    
    # 基本情報
    lines.append(f"Total Models: {summary.get('total_models', 0)}")
    lines.append(f"Symbols Covered: {len(summary.get('symbols_covered', []))}")
    lines.append(f"Model Types: {', '.join([str(t) for t in summary.get('model_types', [])])}")
    
    # 銘柄一覧
    symbols = summary.get('symbols_covered', [])
    if symbols:
        lines.append(f"\nCovered Symbols:")
        for symbol in sorted(symbols)[:10]:  # 最大10個まで表示
            lines.append(f"  - {symbol}")
        if len(symbols) > 10:
            lines.append(f"  ... and {len(symbols) - 10} more")
    
    # 最近の活動
    recent_models = summary.get('recent_models', [])
    if recent_models:
        lines.append(f"\nRecent Training Activity:")
        for model in recent_models[:5]:  # 最大5個まで表示
            lines.append(f"  - {model.get('model_id', 'Unknown')} ({model.get('created_at', 'Unknown date')})")
    
    # エラー情報
    if 'error' in summary:
        lines.append(f"\nError: {summary['error']}")
    
    lines.append("="*60)
    
    return "\n".join(lines)


def merge_config_dicts(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """設定辞書のマージ（再帰的）"""
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def calculate_memory_usage(obj) -> int:
    """オブジェクトのメモリ使用量推定（バイト）"""
    import sys
    
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全な除算（ゼロ除算回避）"""
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """文字列の切り詰め"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def convert_bytes_to_readable(bytes_size: int) -> str:
    """バイト数を読みやすい形式に変換"""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(bytes_size)
    
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    
    return f"{size:.1f} PB"


def get_file_age_days(file_path: Path) -> int:
    """ファイルの作成からの経過日数"""
    try:
        from datetime import datetime
        created_time = datetime.fromtimestamp(file_path.stat().st_ctime)
        age = datetime.now() - created_time
        return age.days
    except Exception:
        return 0


def cleanup_old_files(directory: Path, max_age_days: int = 30, 
                     pattern: str = "*", dry_run: bool = True) -> List[Path]:
    """古いファイルのクリーンアップ"""
    cleaned_files = []
    
    try:
        for file_path in directory.glob(pattern):
            if file_path.is_file() and get_file_age_days(file_path) > max_age_days:
                if not dry_run:
                    file_path.unlink()
                cleaned_files.append(file_path)
    except Exception as e:
        logging.getLogger(__name__).error(f"ファイルクリーンアップエラー: {e}")
    
    return cleaned_files


def create_backup_filename(original_path: Path) -> Path:
    """バックアップファイル名の生成"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = original_path.stem
    suffix = original_path.suffix
    parent = original_path.parent
    
    backup_name = f"{stem}_backup_{timestamp}{suffix}"
    return parent / backup_name


# 定数定義
DEFAULT_MODEL_CONFIGS = {
    'performance_threshold': 0.6,
    'min_data_quality': 'fair',
    'max_model_age_days': 90,
    'max_models_per_symbol': 10,
    'cleanup_interval_days': 7
}

SUPPORTED_FILE_EXTENSIONS = ['.joblib', '.pkl', '.pickle', '.json', '.yaml', '.yml']

ERROR_MESSAGES = {
    'invalid_symbol': "無効な銘柄コードです",
    'insufficient_data': "データが不足しています",
    'model_not_found': "モデルが見つかりません",
    'training_failed': "モデル訓練に失敗しました",
    'prediction_failed': "予測処理に失敗しました",
    'config_error': "設定エラーが発生しました"
}


class ModelConstants:
    """モデル関連の定数"""
    
    # データ品質閾値
    MIN_DATA_SAMPLES = 30
    MIN_FEATURE_COUNT = 5
    MAX_MISSING_RATE = 0.1
    
    # モデル性能閾値
    MIN_ACCURACY = 0.5
    MIN_F1_SCORE = 0.5
    MIN_R2_SCORE = 0.3
    
    # アンサンブル設定
    MIN_MODELS_FOR_ENSEMBLE = 2
    MAX_MODELS_FOR_ENSEMBLE = 10
    DEFAULT_ENSEMBLE_WEIGHTS = 0.33
    
    # ファイルサイズ制限
    MAX_MODEL_SIZE_MB = 100
    MAX_METADATA_SIZE_KB = 10
    
    # タイムアウト設定
    TRAINING_TIMEOUT_MINUTES = 30
    PREDICTION_TIMEOUT_SECONDS = 10