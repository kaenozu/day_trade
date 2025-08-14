# Project Overview

This project, "Day Trade Personal," is a personal-use stock analysis system designed for individual investors, as indicated by `README.md`. It aims to provide simplified stock analysis recommendations with a stated "93% AI accuracy" and "ultra-fast analysis." The system is built primarily with Python and utilizes common data science and machine learning libraries.

Despite claims of AI accuracy and advanced features, a detailed analysis of `daytrade.py` and `day_trading_engine.py` reveals that the core analysis logic for generating trading signals (`DayTradingRecommendation`) currently relies on random number generation (`np.random.uniform`) for key indicators like intraday volatility, volume ratio, and price momentum. This suggests that while the system's architecture and configuration files (e.g., `ml.json`, `analysis.json`) lay the groundwork for sophisticated machine learning, the actual AI models are not yet integrated or actively providing real-world predictions.

The project structure includes:
*   **`daytrade.py`**: The main entry point, handling command-line arguments and orchestrating different analysis modes (quick, multi-symbol, day trading, history, alerts).
*   **`day_trading_engine.py`**: Contains the logic for generating day trading signals and recommendations, including market session awareness and pre-market forecasts. However, the underlying data generation for these recommendations is currently simulated.
*   **`config/`**: A directory containing various configuration files for machine learning models (`ml.json`), analysis parameters (`analysis.json`, `indicators_config.json`), and stock master data management (`stock_master_config.json`). These files define the intended use of advanced features like LSTM, Random Forest, SVM, ensemble models, and various technical indicators (MACD, Bollinger Bands, RSI).
*   **`requirements.txt`**: Lists core Python dependencies for data processing (pandas, numpy), machine learning (scikit-learn), financial data fetching (yfinance), and visualization (matplotlib, seaborn).

## Building and Running

The project can be set up and run with the following steps, as described in `README.md`:

1.  **Clone the repository:**
    ```bash
    git clone [this repository]
    cd day_trade
    ```
2.  **Install required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run analysis:**
    *   **Default (Day Trade Recommendation):**
        ```bash
        python daytrade.py
        ```
    *   **Quick mode (TOP3 recommendation):**
        ```bash
        python daytrade.py --quick
        ```
    *   **Multi-symbol analysis (e.g., 10 symbols):**
        ```bash
        python daytrade.py --multi 10
        ```
    *   **Portfolio recommendation (e.g., for 1,000,000 JPY):**
        ```bash
        python daytrade.py --portfolio 1000000
        ```
    *   **With chart display:**
        ```bash
        python daytrade.py --chart
        ```
    *   **Specific symbols (e.g., 7203, 8306):**
        ```bash
        python daytrade.py --symbols 7203,8306
        ```
    *   **Safe mode (low risk only):**
        ```bash
        python daytrade.py --safe
        ```
    *   **View analysis history:**
        ```bash
        python daytrade.py --history
        ```
    *   **Check alerts:**
        ```bash
        python daytrade.py --alerts
        ```

## Development Conventions

Based on the file analysis:

*   **Language**: Python 3.8+ (recommended 3.11+)
*   **Dependency Management**: `requirements.txt` is used for managing project dependencies.
*   **Configuration**: JSON files in the `config/` directory are used for various settings, including ML model parameters, analysis logic, and data fetching.
*   **Code Structure**: The project seems to follow a modular structure with clear separation of concerns (e.g., `daytrade.py` for main logic, `day_trading_engine.py` for core trading logic, `src/` for source code components).
*   **Personal Use Focus**: The project is explicitly designed for personal use, excluding complex commercial features like Redis, Kubernetes, FastAPI, etc. This implies a focus on simplicity and local execution.
*   **Testing**: The presence of `tests/` directory and `pytest.ini` suggests the use of Pytest for unit and integration testing.
