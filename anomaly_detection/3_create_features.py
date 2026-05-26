import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from anomaly_detection.config import Config


class StepDetector:
    """Detects step-change anomalies via windowed first-difference thresholding."""

    def __init__(self, synthetic_data_dir, statistical_statistical_features_dir,
                 window_sizes=None, stride=None, step_threshold=None):
        self.synthetic_data_dir = str(synthetic_data_dir)
        self.statistical_statistical_features_dir = str(statistical_statistical_features_dir)
        self.window_sizes = window_sizes if window_sizes is not None else Config.WINDOW_SIZES
        self.stride = stride if stride is not None else Config.STRIDE
        self.step_threshold = step_threshold if step_threshold is not None else Config.STEP_THRESHOLD

    def load_csv_files(self):
        csv_files = [f for f in os.listdir(self.synthetic_data_dir) if f.endswith(".csv")]
        return {f: pd.read_csv(os.path.join(self.synthetic_data_dir, f)) for f in csv_files}

    def step_signal_feature(self):
        data_dict = self.load_csv_files()
        os.makedirs(self.statistical_statistical_features_dir, exist_ok=True)
        for file, data in data_dict.items():
            column_name = data.columns[2]
            for window_size in self.window_sizes:
                anomalies_indices = []
                for i in range(0, len(data) - window_size + 1, self.stride):
                    window_data = data[column_name][i:i + window_size]
                    steps = window_data.diff().fillna(0).abs()
                    anomalies_indices.extend(window_data[steps > self.step_threshold].index)
                data[f"step_variable_ws{window_size}"] = 0
                data.loc[anomalies_indices, f"step_variable_ws{window_size}"] = 1
            output_file_path = os.path.join(self.statistical_statistical_features_dir, file)
            data.to_csv(output_file_path, index=False)


class MeanAndStandardDeviation:
    def __init__(self, statistical_statistical_features_dir, window_sizes=None,
                 stride=None, threshold_multiplier=None):
        self.statistical_statistical_features_dir = str(statistical_statistical_features_dir)
        self.window_sizes = window_sizes if window_sizes is not None else Config.WINDOW_SIZES
        self.stride = stride if stride is not None else Config.STRIDE
        self.threshold_multiplier = (
            threshold_multiplier if threshold_multiplier is not None
            else Config.STD_THRESHOLD_MULTIPLIER
        )

    def load_csv_files(self):
        csv_files = [f for f in os.listdir(self.statistical_statistical_features_dir) if f.endswith(".csv")]
        return {f: pd.read_csv(os.path.join(self.statistical_statistical_features_dir, f)) for f in csv_files}

    def detect_and_evaluate(self):
        data_dict = self.load_csv_files()
        for file, data in data_dict.items():
            column_name = data.columns[2]
            for window_size in self.window_sizes:
                anomalies_indices = []
                for i in range(0, len(data) - window_size + 1, self.stride):
                    window_data = data[column_name][i:i + window_size]
                    mean, std = window_data.mean(), window_data.std()
                    lower = mean - std * self.threshold_multiplier
                    upper = mean + std * self.threshold_multiplier
                    anomalies_indices.extend(
                        window_data[(window_data < lower) | (window_data > upper)].index
                    )
                data[f"std_anomaly_ws{window_size}"] = 0
                data.loc[anomalies_indices, f"std_anomaly_ws{window_size}"] = 1
            output_file_path = os.path.join(self.statistical_statistical_features_dir, file)
            data.to_csv(output_file_path, index=False)


class InterQuartileRange:
    def __init__(self, statistical_statistical_features_dir, window_sizes=None, stride=None,
                 threshold_multiplier=None, lower_quartile=None, upper_quartile=None):
        self.statistical_statistical_features_dir = str(statistical_statistical_features_dir)
        self.window_sizes = window_sizes if window_sizes is not None else Config.WINDOW_SIZES
        self.stride = stride if stride is not None else Config.STRIDE
        self.threshold_multiplier = (
            threshold_multiplier if threshold_multiplier is not None
            else Config.STD_THRESHOLD_MULTIPLIER
        )
        self.lower_quartile = (
            lower_quartile if lower_quartile is not None else Config.IQR_LOWER_QUARTILE
        )
        self.upper_quartile = (
            upper_quartile if upper_quartile is not None else Config.IQR_UPPER_QUARTILE
        )

    def load_csv_files(self):
        csv_files = [f for f in os.listdir(self.statistical_statistical_features_dir) if f.endswith(".csv")]
        return {f: pd.read_csv(os.path.join(self.statistical_statistical_features_dir, f)) for f in csv_files}

    def detect_and_evaluate_iqr(self):
        data_dict = self.load_csv_files()
        for file, data in data_dict.items():
            column_name = data.columns[2]
            for window_size in self.window_sizes:
                anomalies_indices = []
                for i in range(0, len(data) - window_size + 1, self.stride):
                    window_data = data[column_name][i:i + window_size]
                    lower = window_data.quantile(self.lower_quartile)
                    upper = window_data.quantile(self.upper_quartile)
                    anomalies_indices.extend(
                        window_data[(window_data < lower) | (window_data > upper)].index
                    )
                data[f"iqr_anomaly_ws{window_size}"] = 0
                data.loc[anomalies_indices, f"iqr_anomaly_ws{window_size}"] = 1
                data["Anomaly"] = data.pop("Anomaly")
            output_file_path = os.path.join(self.statistical_statistical_features_dir, file)
            data.to_csv(output_file_path, index=False)


def unsupervised_ml(synthetic_data_dir, statistical_features_dir):
    step_detector = StepDetector(synthetic_data_dir, statistical_features_dir)
    step_detector.step_signal_feature()

    mean_std = MeanAndStandardDeviation(statistical_features_dir)
    mean_std.detect_and_evaluate()

    iqr = InterQuartileRange(statistical_features_dir)
    iqr.detect_and_evaluate_iqr()

    print("All files have been updated with Unsupervised Machine Learning Labels")


def run():
    Config.ML_DATA_DIR.mkdir(parents=True, exist_ok=True)
    unsupervised_ml(Config.SYNTHETIC_DIR, Config.ML_DATA_DIR)


def main():
    run()


if __name__ == "__main__":
    main()
