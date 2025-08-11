    ARCH_AVAILABLE = True
    logger.info(
        "arch利用可能"
    )
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch未インストール - pip install archでインストールしてください")

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
    logger.info(
        "scikit-learn利用可能"
    )
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn未インストール")