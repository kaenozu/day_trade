import numpy as np
import pandas as pd


def load_training_data_for_symbol(
    symbol: str, data_path: str = "./data/training_data"
) -> pd.DataFrame:
    """指定された銘柄の訓練データをロード (DVCで管理されることを想定)"""
    # 実際にはここでDVCからデータセットをロードするロジックが入る
    # 例: dvc.api.read(f"{data_path}/{symbol}.csv", rev="HEAD")

    # 今はモックデータとして、既存のデータフレームを返すか、エラーとする
    # ここでエラーを発生させることで、実際のデータロードの実装が必須となることを強制する
    # raise NotImplementedError(f"訓練データ ({symbol}) のロードは実装されていません。DVCなどのデータバージョン管理システムとの連携が必要です。")

    # 仮のデータフレームを返す (デバッグ/テスト用)
    # 実際はデータパスから読み込む
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    mock_data = pd.DataFrame(
        {
            "Open": np.random.uniform(100, 200, 100),
            "High": np.random.uniform(100, 200, 100),
            "Low": np.random.uniform(100, 200, 100),
            "Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.randint(100000, 1000000, 100),
        },
        index=dates,
    )
    return mock_data
