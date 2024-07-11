import os
import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings('ignore')


class StepDetector:
    """Class for detecting the step signal feature in the data"""
    def __init__(self, synthetic_data_dir, statistical_statistical_features_dir, window_sizes=[5, 10, 15], stride=3):
        self.synthetic_data_dir = synthetic_data_dir
        self.statistical_statistical_features_dir = statistical_statistical_features_dir
        self.window_sizes = window_sizes
        self.stride = stride

    def load_csv_files(self):
        csv_files = [file for file in os.listdir(self.synthetic_data_dir) if file.endswith(".csv")]
        return {file: pd.read_csv(os.path.join(self.synthetic_data_dir, file)) for file in csv_files}

    def step_signal_feature(self):
        data_dict = self.load_csv_files()
        for file, data in data_dict.items():
            column_name = data.columns[2]
            os.makedirs(self.statistical_statistical_features_dir, exist_ok=True)
            for window_size in self.window_sizes:
                anomalies_indices = []
                for i in range(0, len(data) - window_size + 1, self.stride):
                    window_data = data[column_name][i:i + window_size].copy()
                    rate_of_change = np.abs(window_data.iloc[-1] - window_data.iloc[0])
                    if rate_of_change > 1000:
                        anomalies_indices.append(i)
                data[f'step_variable_ws{window_size}'] = 0
                data.loc[anomalies_indices, f'step_variable_ws{window_size}'] = 1
            output_file_path = os.path.join(self.statistical_statistical_features_dir, file)
            data.to_csv(output_file_path, index=False)


class MeanAndStandardDeviation:
    def __init__(self, statistical_statistical_features_dir, window_sizes=[5, 10, 15], stride=3, threshold_multiplier=1):
        self.statistical_statistical_features_dir = statistical_statistical_features_dir
        self.window_sizes = window_sizes
        self.stride = stride
        self.threshold_multiplier = threshold_multiplier

    def load_csv_files(self):
        csv_files = [file for file in os.listdir(self.statistical_statistical_features_dir) if file.endswith(".csv")]
        return {file: pd.read_csv(os.path.join(self.statistical_statistical_features_dir, file)) for file in csv_files}

    def detect_and_evaluate(self):
        data_dict = self.load_csv_files()
        for file, data in data_dict.items():
            column_name = data.columns[2]
            for window_size in self.window_sizes:
                anomalies_indices = []
                for i in range(0, len(data) - window_size + 1, self.stride):
                    window_data = data[column_name][i:i + window_size]
                    mean, std = window_data.mean(), window_data.std()
                    lower_bound, upper_bound = mean - std * self.threshold_multiplier, mean + std * self.threshold_multiplier
                    anomalies_indices.extend(window_data[(window_data < lower_bound) | (window_data > upper_bound)].index)
                data[f'std_anomaly_ws{window_size}'] = 0
                data.loc[anomalies_indices, [f'std_anomaly_ws{window_size}']] = 1
            output_file_path = os.path.join(self.statistical_statistical_features_dir, file)
            data.to_csv(output_file_path, index=False)


class InterQuartileRange:
    def __init__(self, statistical_statistical_features_dir, window_sizes=[5, 10, 15], stride=3, threshold_multiplier=1, lower_quartile=0.1, upper_quartile=0.9):
        self.statistical_statistical_features_dir = statistical_statistical_features_dir
        self.window_sizes = window_sizes
        self.stride = stride
        self.threshold_multiplier = threshold_multiplier
        self.lower_quartile = lower_quartile
        self.upper_quartile = upper_quartile

    def load_csv_files(self):
        csv_files = [file for file in os.listdir(self.statistical_statistical_features_dir) if file.endswith(".csv")]
        return {file: pd.read_csv(os.path.join(self.statistical_statistical_features_dir, file)) for file in csv_files}

    def detect_and_evaluate_iqr(self):
        data_dict = self.load_csv_files()
        for file, data in data_dict.items():
            column_name = data.columns[2]
            for window_size in self.window_sizes:
                anomalies_indices = []
                for i in range(0, len(data) - window_size + 1, self.stride):
                    window_data = data[column_name][i:i + window_size]
                    lower_bound = window_data.quantile(self.lower_quartile)
                    upper_bound = window_data.quantile(self.upper_quartile)
                    anomalies_indices.extend(window_data[(window_data < lower_bound) | (window_data > upper_bound)].index)
                data[f'iqr_anomaly_ws{window_size}'] = 0
                data.loc[anomalies_indices, f'iqr_anomaly_ws{window_size}'] = 1

                data['Anomaly'] = data.pop('Anomaly')
            output_file_path = os.path.join(self.statistical_statistical_features_dir, file)
            data.to_csv(output_file_path, index=False)


def unsupervised_ml(synthetic_data_dir, statistical_statistical_features_dir):
    step_detector = StepDetector(synthetic_data_dir, statistical_statistical_features_dir)
    step_detector.step_signal_feature()
    mean_std = MeanAndStandardDeviation(statistical_statistical_features_dir)
    mean_std.detect_and_evaluate()
    iqr = InterQuartileRange(statistical_statistical_features_dir)
    iqr.detect_and_evaluate_iqr()
    print("All files have been updated with Unsupervised Machine Learning Labels")


def main():
    synthetic_data_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/data/processed/synthetic_data"
    statistical_features_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/data/processed/ml_data"
    
    if not os.path.exists(statistical_features_dir):
        os.makedirs(statistical_features_dir)

    unsupervised_ml(synthetic_data_dir, statistical_features_dir)

if __name__ == "__main__":
    main()
    