import yaml
import os

def load_config(config_path: str = 'config.yaml'):
    """
    指定されたパスから設定ファイルを読み込む。

    :param config_path: 設定ファイルへのパス
    :return: 設定内容を格納した辞書
    """
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using default settings.")
        return {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing config file {config_path}: {e}")
        return {}

if __name__ == "__main__":
    # テスト用のダミー設定ファイルを作成
    dummy_config_content = """
    data:
      output_dir: ./data
    strategies:
      default_strategies: ["MA Cross", "RSI"]
      ma_cross:
        short_window: 5
        long_window: 25
    """
    with open('test_config.yaml', 'w') as f:
        f.write(dummy_config_content)

    config = load_config('test_config.yaml')
    print("Loaded Config:", config)

    # 存在しないファイルを読み込むテスト
    non_existent_config = load_config('non_existent.yaml')
    print("Non-existent Config:", non_existent_config)

    # テスト用のダミー設定ファイルを削除
    os.remove('test_config.yaml')
