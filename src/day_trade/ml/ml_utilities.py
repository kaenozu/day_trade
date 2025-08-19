"""
ML Utilities and Helper Classes

ml_prediction_models_improved.py からのリファクタリング抽出
メタデータ管理、データ準備パイプライン、ユーティリティクラス
"""

import sqlite3
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict

# 設定とエラーのインポート
from .ml_config import ModelType, PredictionTask, DataQuality, TrainingConfig
from .ml_exceptions import DataPreparationError, ModelMetadataError

# 機械学習ライブラリ（フォールバック対応）
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 外部システムインポート
try:
    from enhanced_feature_engineering import enhanced_feature_engineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    from real_data_provider import RealDataProvider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False


@dataclass
class ModelMetadata:
    """モデルメタデータ（強化版）"""
    model_id: str
    model_type: ModelType
    task: PredictionTask
    symbol: str
    version: str
    created_at: datetime
    updated_at: datetime

    # 訓練情報
    feature_columns: List[str]
    target_info: Dict[str, Any]
    training_samples: int
    training_period: str
    data_quality: DataQuality

    # モデル設定
    hyperparameters: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    feature_selection_config: Dict[str, Any]

    # 性能メトリクス
    performance_metrics: Dict[str, float]
    cross_validation_scores: List[float]
    feature_importance: Dict[str, float]

    # システム情報
    is_classifier: bool
    model_size_mb: float
    training_time_seconds: float
    python_version: str
    sklearn_version: str
    framework_versions: Dict[str, str]

    # 品質管理
    validation_status: str = "pending"
    deployment_status: str = "development"
    performance_threshold_met: bool = False
    data_drift_detected: bool = False

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            **asdict(self),
            'model_type': self.model_type.value,
            'task': self.task.value,
            'data_quality': self.data_quality.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class ModelPerformance:
    """モデル性能（強化版）"""
    model_id: str
    symbol: str
    task: PredictionTask
    model_type: ModelType

    # 基本性能指標
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # クロスバリデーション
    cross_val_mean: float
    cross_val_std: float
    cross_val_scores: List[float]

    # 回帰指標（該当する場合）
    r2_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None

    # 詳細分析
    feature_importance: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None

    # 時間指標
    training_time: float = 0.0
    prediction_time: float = 0.0

    # 品質指標
    prediction_stability: float = 0.0
    confidence_calibration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['task'] = self.task.value
        result['model_type'] = self.model_type.value
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.tolist()
        return result


@dataclass
class PredictionResult:
    """予測結果（強化版）"""
    symbol: str
    timestamp: datetime
    model_type: ModelType
    task: PredictionTask
    model_version: str

    # 予測結果
    prediction: Union[str, float]
    confidence: float
    prediction_interval: Optional[Tuple[float, float]] = None

    # 詳細情報
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    feature_values: Dict[str, float] = field(default_factory=dict)
    feature_importance_contribution: Dict[str, float] = field(default_factory=dict)

    # メタ情報
    model_performance_history: Dict[str, float] = field(default_factory=dict)
    data_quality_assessment: Optional[DataQuality] = None
    explanation: str = ""

    # 品質指標
    prediction_stability_score: float = 0.0
    confidence_calibration_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        result['task'] = self.task.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.data_quality_assessment:
            result['data_quality_assessment'] = self.data_quality_assessment.value
        return result


@dataclass
class EnsemblePrediction:
    """アンサンブル予測結果（強化版）"""
    symbol: str
    timestamp: datetime

    # 最終予測
    final_prediction: Union[str, float]
    confidence: float
    prediction_interval: Optional[Tuple[float, float]] = None

    # 個別モデル情報
    model_predictions: Dict[str, Any] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    model_quality_scores: Dict[str, float] = field(default_factory=dict)

    # アンサンブル品質
    consensus_strength: float = 0.0
    disagreement_score: float = 0.0
    prediction_stability: float = 0.0
    diversity_score: float = 0.0

    # メタ情報
    total_models_used: int = 0
    excluded_models: List[str] = field(default_factory=list)
    ensemble_method: str = "weighted_average"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class ModelMetadataManager:
    """モデルメタデータ管理システム（強化版）"""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._init_metadata_database()

    def _init_metadata_database(self):
        """メタデータ用データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # モデルメタデータテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    task TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    feature_columns TEXT NOT NULL,
                    target_info TEXT NOT NULL,
                    training_samples INTEGER,
                    training_period TEXT,
                    data_quality TEXT,
                    hyperparameters TEXT,
                    preprocessing_config TEXT,
                    feature_selection_config TEXT,
                    performance_metrics TEXT,
                    cross_validation_scores TEXT,
                    feature_importance TEXT,
                    is_classifier BOOLEAN,
                    model_size_mb REAL,
                    training_time_seconds REAL,
                    python_version TEXT,
                    sklearn_version TEXT,
                    framework_versions TEXT,
                    validation_status TEXT DEFAULT 'pending',
                    deployment_status TEXT DEFAULT 'development',
                    performance_threshold_met BOOLEAN DEFAULT FALSE,
                    data_drift_detected BOOLEAN DEFAULT FALSE
                )
            """)

            # モデル性能履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    evaluation_date TEXT NOT NULL,
                    dataset_type TEXT NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    r2_score REAL,
                    mse REAL,
                    rmse REAL,
                    mae REAL,
                    cross_val_mean REAL,
                    cross_val_std REAL,
                    prediction_stability REAL,
                    confidence_calibration REAL,
                    feature_drift_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES model_metadata (model_id)
                )
            """)

            # アンサンブル予測履歴テーブル（強化版）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    task TEXT NOT NULL,
                    final_prediction TEXT,
                    confidence REAL,
                    consensus_strength REAL,
                    disagreement_score REAL,
                    prediction_stability REAL,
                    diversity_score REAL,
                    model_count INTEGER,
                    avg_model_quality REAL,
                    confidence_variance REAL,
                    model_predictions TEXT,
                    model_weights TEXT,
                    model_quality_scores TEXT,
                    excluded_models TEXT,
                    ensemble_method TEXT,
                    quality_metrics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 予測精度追跡テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_accuracy_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    predicted_value TEXT,
                    actual_value TEXT,
                    prediction_date TEXT,
                    evaluation_date TEXT,
                    confidence_at_prediction REAL,
                    accuracy_score REAL,
                    error_magnitude REAL,
                    model_used TEXT,
                    task TEXT,
                    was_correct BOOLEAN,
                    error_category TEXT,
                    market_conditions TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # モデル重み履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    task TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    static_weight REAL,
                    dynamic_weight REAL,
                    performance_contribution REAL,
                    confidence_contribution REAL,
                    quality_contribution REAL,
                    weight_change_reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_metadata(self, metadata: ModelMetadata) -> bool:
        """メタデータ保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_metadata VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id,
                    metadata.model_type.value,
                    metadata.task.value,
                    metadata.symbol,
                    metadata.version,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    json.dumps(metadata.feature_columns),
                    json.dumps(metadata.target_info),
                    metadata.training_samples,
                    metadata.training_period,
                    metadata.data_quality.value,
                    json.dumps(metadata.hyperparameters),
                    json.dumps(metadata.preprocessing_config),
                    json.dumps(metadata.feature_selection_config),
                    json.dumps(metadata.performance_metrics),
                    json.dumps(metadata.cross_validation_scores),
                    json.dumps(metadata.feature_importance),
                    metadata.is_classifier,
                    metadata.model_size_mb,
                    metadata.training_time_seconds,
                    metadata.python_version,
                    metadata.sklearn_version,
                    json.dumps(metadata.framework_versions),
                    metadata.validation_status,
                    metadata.deployment_status,
                    metadata.performance_threshold_met,
                    metadata.data_drift_detected
                ))
            return True
        except Exception as e:
            self.logger.error(f"メタデータ保存エラー: {e}")
            return False

    def load_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """メタデータ読み込み"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM model_metadata WHERE model_id = ?
                """, (model_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_metadata(row)
                return None

        except Exception as e:
            self.logger.error(f"メタデータ読み込みエラー: {e}")
            return None

    def _row_to_metadata(self, row) -> ModelMetadata:
        """データベース行をModelMetadataに変換"""
        return ModelMetadata(
            model_id=row[0],
            model_type=ModelType(row[1]),
            task=PredictionTask(row[2]),
            symbol=row[3],
            version=row[4],
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            feature_columns=json.loads(row[7]),
            target_info=json.loads(row[8]),
            training_samples=row[9],
            training_period=row[10],
            data_quality=DataQuality(row[11]),
            hyperparameters=json.loads(row[12]),
            preprocessing_config=json.loads(row[13]),
            feature_selection_config=json.loads(row[14]),
            performance_metrics=json.loads(row[15]),
            cross_validation_scores=json.loads(row[16]),
            feature_importance=json.loads(row[17]),
            is_classifier=bool(row[18]),
            model_size_mb=row[19],
            training_time_seconds=row[20],
            python_version=row[21],
            sklearn_version=row[22],
            framework_versions=json.loads(row[23]),
            validation_status=row[24],
            deployment_status=row[25],
            performance_threshold_met=bool(row[26]),
            data_drift_detected=bool(row[27])
        )

    def get_model_versions(self, symbol: str, model_type: ModelType, task: PredictionTask) -> List[str]:
        """モデルバージョン一覧取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version FROM model_metadata
                    WHERE symbol = ? AND model_type = ? AND task = ?
                    ORDER BY created_at DESC
                """, (symbol, model_type.value, task.value))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"バージョン一覧取得エラー: {e}")
            return []


class DataPreparationPipeline:
    """データ準備パイプライン（強化版）"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler() if self.config.enable_scaling and SKLEARN_AVAILABLE else None
        self.feature_selector = None

    async def prepare_training_data(self, symbol: str, period: str = "1y",
                                  data_provider=None) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series], DataQuality]:
        """訓練データ準備（強化版）"""
        try:
            self.logger.info(f"データ準備開始: {symbol}")

            # データ取得
            data = await self._fetch_data(symbol, period, data_provider)

            # データ品質評価
            is_valid, quality, quality_message = self._assess_data_quality(data)
            if not is_valid or quality < self.config.min_data_quality:
                raise DataPreparationError(f"データ品質不足: {quality_message}")

            self.logger.info(f"データ品質評価: {quality.value} - {quality_message}")

            # 特徴量エンジニアリング
            features = await self._engineer_features(symbol, data)

            # 特徴量後処理
            features = self._postprocess_features(features)

            # ターゲット変数作成
            targets = self._create_target_variables(data)

            # データ整合性チェック
            features, targets = self._align_data(features, targets)

            self.logger.info(f"データ準備完了: features={features.shape}, quality={quality.value}")

            return features, targets, quality

        except Exception as e:
            self.logger.error(f"データ準備エラー: {e}")
            raise DataPreparationError(f"データ準備失敗: {e}") from e

    async def _fetch_data(self, symbol: str, period: str, data_provider) -> pd.DataFrame:
        """データ取得（強化版）"""
        if data_provider and REAL_DATA_PROVIDER_AVAILABLE:
            # 実データプロバイダー使用
            try:
                data = await data_provider.get_stock_data(symbol, period)
                if not data.empty:
                    return data
                else:
                    self.logger.warning(f"実データプロバイダーが空データを返しました: {symbol}")
            except Exception as e:
                self.logger.warning(f"実データプロバイダーエラー: {e}")

        # フォールバック: 基本的なダミーデータ
        return self._create_fallback_data(symbol, period)

    def _create_fallback_data(self, symbol: str, period: str) -> pd.DataFrame:
        """フォールバック用ダミーデータ作成"""
        self.logger.info(f"フォールバックデータ作成: {symbol}")
        
        # 期間に応じたサンプル数決定
        period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 365)
        
        dates = pd.date_range(end=datetime.now(), periods=period_days, freq='D')
        np.random.seed(hash(symbol) % 2**32)
        
        base_price = 1000 + (hash(symbol) % 1000)
        prices = base_price + np.cumsum(np.random.randn(period_days) * 10)
        volumes = 1000000 + np.random.randint(-500000, 500000, period_days)
        
        return pd.DataFrame({
            'Open': prices + np.random.randn(period_days) * 5,
            'High': prices + np.abs(np.random.randn(period_days)) * 5,
            'Low': prices - np.abs(np.random.randn(period_days)) * 5,
            'Close': prices,
            'Volume': volumes,
            'Adj Close': prices
        }, index=dates)

    def _assess_data_quality(self, data: pd.DataFrame) -> Tuple[bool, DataQuality, str]:
        """データ品質評価（強化版）"""
        try:
            issues = []
            quality_score = 100.0

            # 基本チェック
            if data.empty:
                return False, DataQuality.INSUFFICIENT, "データが空です"

            # データサイズチェック
            if len(data) < 30:
                return False, DataQuality.INSUFFICIENT, f"データ量不足: {len(data)}行"

            # 欠損値チェック
            missing_count = data.isnull().sum().sum()
            missing_rate = missing_count / (len(data) * len(data.columns))

            if missing_rate > 0.2:
                quality_score -= 30
                issues.append(f"欠損率高: {missing_rate:.1%}")
            elif missing_rate > 0.1:
                quality_score -= 15
                issues.append(f"欠損率中: {missing_rate:.1%}")

            # 必須カラムチェック
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                quality_score -= 25
                issues.append(f"必須カラム不足: {missing_columns}")

            # データ整合性チェック
            if 'High' in data.columns and 'Low' in data.columns:
                invalid_hl = (data['High'] < data['Low']).sum()
                if invalid_hl > 0:
                    quality_score -= 20
                    issues.append(f"High < Low: {invalid_hl}件")

            # 品質レベル決定
            if quality_score >= 90:
                quality = DataQuality.EXCELLENT
            elif quality_score >= 75:
                quality = DataQuality.GOOD
            elif quality_score >= 60:
                quality = DataQuality.FAIR
            elif quality_score >= 40:
                quality = DataQuality.POOR
            else:
                quality = DataQuality.INSUFFICIENT

            message = f"品質スコア: {quality_score:.1f}" + (f", 問題: {'; '.join(issues)}" if issues else "")
            success = quality != DataQuality.INSUFFICIENT

            return success, quality, message

        except Exception as e:
            return False, DataQuality.INSUFFICIENT, f"評価エラー: {e}"

    async def _engineer_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """特徴量エンジニアリング（強化版）"""
        try:
            # 外部特徴量エンジニアリング使用
            if FEATURE_ENGINEERING_AVAILABLE:
                try:
                    features = await enhanced_feature_engineer(symbol, data)
                    if not features.empty:
                        return features
                except Exception as e:
                    self.logger.warning(f"拡張特徴量エンジニアリング失敗: {e}")

            # フォールバック: 基本的な特徴量作成
            return self._create_basic_features(data)

        except Exception as e:
            self.logger.error(f"特徴量エンジニアリングエラー: {e}")
            raise DataPreparationError(f"特徴量作成失敗: {e}") from e

    def _create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本的な特徴量作成"""
        features = pd.DataFrame(index=data.index)

        try:
            # 価格ベース特徴量
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                features['price_change'] = data['Close'].pct_change()
                features['price_range'] = (data['High'] - data['Low']) / data['Close']
                features['body_ratio'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-8)

            # 移動平均
            for window in [5, 10, 20]:
                if 'Close' in data.columns:
                    features[f'sma_{window}'] = data['Close'].rolling(window).mean()
                    features[f'price_to_sma_{window}'] = data['Close'] / features[f'sma_{window}']

            # ボリューム特徴量
            if 'Volume' in data.columns:
                features['volume_change'] = data['Volume'].pct_change()
                features['volume_sma_5'] = data['Volume'].rolling(5).mean()

            # 欠損値除去
            features = features.dropna()

            self.logger.info(f"基本特徴量作成完了: {features.shape}")
            return features

        except Exception as e:
            self.logger.error(f"基本特徴量作成エラー: {e}")
            # 最小限の特徴量を返す
            return pd.DataFrame({
                'close_price': data.get('Close', pd.Series(index=data.index, dtype=float)),
                'volume': data.get('Volume', pd.Series(index=data.index, dtype=float))
            }).dropna()

    def _postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """特徴量後処理"""
        try:
            # スケーリング
            if self.scaler and self.config.enable_scaling:
                numeric_columns = features.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    features[numeric_columns] = self.scaler.fit_transform(features[numeric_columns])

            # 異常値処理
            if self.config.outlier_detection:
                features = self._remove_outliers(features)

            return features

        except Exception as e:
            self.logger.warning(f"特徴量後処理エラー: {e}")
            return features

    def _remove_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """異常値除去"""
        try:
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = features[col].quantile(0.25)
                Q3 = features[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                features[col] = features[col].clip(lower=lower_bound, upper=upper_bound)
            return features
        except Exception as e:
            self.logger.warning(f"異常値除去エラー: {e}")
            return features

    def _create_target_variables(self, data: pd.DataFrame) -> Dict[PredictionTask, pd.Series]:
        """ターゲット変数作成"""
        targets = {}

        try:
            if 'Close' in data.columns:
                # 価格方向予測
                price_change = data['Close'].pct_change()
                targets[PredictionTask.PRICE_DIRECTION] = pd.cut(
                    price_change, bins=[-np.inf, -0.01, 0.01, np.inf],
                    labels=['down', 'flat', 'up']
                ).astype(str)

                # 価格回帰予測
                targets[PredictionTask.PRICE_REGRESSION] = data['Close'].shift(-1)

            return targets

        except Exception as e:
            self.logger.error(f"ターゲット変数作成エラー: {e}")
            return {}

    def _align_data(self, features: pd.DataFrame, targets: Dict[PredictionTask, pd.Series]) -> Tuple[pd.DataFrame, Dict[PredictionTask, pd.Series]]:
        """データ整合性チェック"""
        try:
            # 共通インデックス取得
            common_index = features.index
            for target in targets.values():
                common_index = common_index.intersection(target.index)

            # データ整列
            aligned_features = features.loc[common_index]
            aligned_targets = {task: target.loc[common_index] for task, target in targets.items()}

            # NaN除去
            for task, target in aligned_targets.items():
                valid_mask = target.notna() & aligned_features.notna().all(axis=1)
                aligned_targets[task] = target[valid_mask]
                if task == list(aligned_targets.keys())[0]:  # 最初のタスクでマスク適用
                    aligned_features = aligned_features[valid_mask]

            return aligned_features, aligned_targets

        except Exception as e:
            self.logger.error(f"データ整列エラー: {e}")
            return features, targets