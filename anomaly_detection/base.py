import os
import pandas as pd
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings("ignore")


class MachineLearning:
    """Shared base: data loading and preprocessing for all supervised ML modules."""

    def __init__(self, statistical_features_dir, model_save_dir=None):
        self.statistical_features_dir = str(statistical_features_dir)
        self.model_save_dir = str(model_save_dir) if model_save_dir else None
        if self.model_save_dir:
            os.makedirs(self.model_save_dir, exist_ok=True)

    def load_data(self):
        csv_files = [f for f in os.listdir(self.statistical_features_dir) if f.endswith(".csv")]
        return {f: pd.read_csv(os.path.join(self.statistical_features_dir, f)) for f in csv_files}

    def preprocess_data(self):
        """Return {filename: (feature_df, original_df)} with scaled signals and temporal features."""
        data_dict = self.load_data()
        processed_data = {}
        scaler = RobustScaler()

        for file, data in data_dict.items():
            original_data = data.copy()
            column_name = data.columns[2]
            data["Timestamp"] = pd.to_datetime(data["Timestamp"], dayfirst=True)
            data["Day"] = data["Timestamp"].dt.day
            data["Month"] = data["Timestamp"].dt.month
            data["Year"] = data["Timestamp"].dt.year
            data["Hour"] = data["Timestamp"].dt.hour
            data["Minute"] = data["Timestamp"].dt.minute

            data["original_signal"] = scaler.fit_transform(
                data["original_signal"].values.reshape(-1, 1)
            )
            data[column_name] = scaler.fit_transform(
                data[column_name].values.reshape(-1, 1)
            )

            columns_order = [
                "Day", "Month", "Year", "Hour", "Minute",
                "original_signal", column_name,
                "step_variable_ws5", "step_variable_ws10", "step_variable_ws15",
                "std_anomaly_ws5", "std_anomaly_ws10", "std_anomaly_ws15",
                "iqr_anomaly_ws5", "iqr_anomaly_ws10", "iqr_anomaly_ws15",
                "Anomaly",
            ]
            data = data[columns_order]
            processed_data[file] = (data, original_data)
        return processed_data
