from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]


class Config:
    # Directories — relative to project root, works on any machine
    DATA_DIR = _PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    SENSOR_DIR = PROCESSED_DIR / "sensor_data"
    TSLAB_DIR = PROCESSED_DIR / "tslab_anomalies"
    SYNTHETIC_DIR = PROCESSED_DIR / "synthetic_data"
    ML_DATA_DIR = PROCESSED_DIR / "ml_data"
    IMPUTED_DIR = PROCESSED_DIR / "Imputed_anomalies"
    IMPUTED_PLOTS_DIR = PROCESSED_DIR / "Imputed_vs_original_anomalies_plot"
    MODEL_DIR = _PROJECT_ROOT / "models"
    REPORT_DIR = _PROJECT_ROOT / "reports"
    METRICS_DIR = REPORT_DIR / "metrics"

    RAW_FILE = RAW_DIR / "1596_UTA-I_2023-05-06_-_2023-06-10_labels.csv"

    # Synthetic anomaly parameters
    NUM_ANOMALIES = 500
    X_MIN, X_MAX = 5, 10
    Y_MIN, Y_MAX = -350, 350
    D_MIN, D_MAX = 1, 4

    # Feature engineering
    WINDOW_SIZES = [5, 10, 15]
    STRIDE = 3
    STEP_THRESHOLD = 1000
    STD_THRESHOLD_MULTIPLIER = 1
    IQR_LOWER_QUARTILE = 0.1
    IQR_UPPER_QUARTILE = 0.9

    # Classifier hyperparameters — single source of truth for steps 4 and 7
    SVM_C = 10
    SVM_KERNEL = "rbf"
    SVM_GAMMA = 0.1
    SVM_RANDOM_STATE = 101

    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 20
    RF_RANDOM_STATE = 101

    # Train/test splits
    TEST_SIZE = 0.2
    EVAL_TEST_SIZE = 0.3
    RANDOM_STATE = 101
    CV_FOLDS = 5
