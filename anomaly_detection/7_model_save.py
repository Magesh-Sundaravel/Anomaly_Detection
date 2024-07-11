import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


class MachineLearning:
    def __init__(self, statistical_features_dir, model_save_dir):
        self.statistical_features_dir = statistical_features_dir
        self.model_save_dir = model_save_dir

        # Create model save directory if it doesn't exist
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def load_data(self):
        csv_files = [file for file in os.listdir(self.statistical_features_dir) if file.endswith(".csv")]
        return {file: pd.read_csv(os.path.join(self.statistical_features_dir, file)) for file in csv_files}

    def preprocess_data(self):
        data_dict = self.load_data()
        processed_data = {}
        scaler = RobustScaler()

        for file, data in data_dict.items():
            original_data = data.copy()
            column_name = data.columns[2]
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True)
            data['Day'] = data['Timestamp'].dt.day
            data['Month'] = data['Timestamp'].dt.month
            data['Year'] = data['Timestamp'].dt.year
            data['Hour'] = data['Timestamp'].dt.hour
            data['Minute'] = data['Timestamp'].dt.minute

            data['original_signal'] = scaler.fit_transform(data['original_signal'].values.reshape(-1, 1))
            data[column_name] = scaler.fit_transform(data[column_name].values.reshape(-1, 1))

            columns_order = ['Day', 'Month', 'Year', 'Hour', 'Minute', 'original_signal', column_name,
                             'step_variable_ws5', 'step_variable_ws10', 'step_variable_ws15',
                             'std_anomaly_ws5', 'std_anomaly_ws10', 'std_anomaly_ws15',
                             'iqr_anomaly_ws5', 'iqr_anomaly_ws10', 'iqr_anomaly_ws15', 'Anomaly']
            data = data[columns_order]
            X = data.drop("Anomaly", axis=1)
            y = data["Anomaly"]
            processed_data[file] = (X, y, original_data)
        return processed_data


class SupportVectorMachine(MachineLearning):
    def __init__(self, statistical_features_dir, model_save_dir):
        super().__init__(statistical_features_dir, model_save_dir)

    def train_test_split(self):
        processed_data = self.preprocess_data()
        for file, (X, y, original_data) in processed_data.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
            pipeline = Pipeline([('svc', SVC(C=10, kernel='rbf', gamma=0.1, random_state=101))])
            svc = pipeline.fit(X_train, y_train)

            # Save the model in the specified directory
            model_filename = os.path.join(self.model_save_dir, 'svm_model.pkl')
            joblib.dump(svc, model_filename)

class RandomForest(MachineLearning):
    def __init__(self, statistical_features_dir, model_save_dir):
        super().__init__(statistical_features_dir, model_save_dir)

    def train_test_split(self):
        processed_data = self.preprocess_data()

        for file, (X, y, original_data) in processed_data.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
            clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=101)
            rf = clf.fit(X_train, y_train)


            # Save the model in the specified directory
            model_filename = os.path.join(self.model_save_dir, 'rf_model.pkl')
            joblib.dump(rf, model_filename)


def supervised_ml(statistical_features_dir, model_save_dir):
    svm_classifier = SupportVectorMachine(statistical_features_dir, model_save_dir)
    svm_classifier.train_test_split()
    
    rf_classifier = RandomForest(statistical_features_dir, model_save_dir)
    rf_classifier.train_test_split()

    print('The models have been saved in the directory')

def main():
    statistical_features_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/data/processed/ml_data"
    model_save_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/models"

    supervised_ml(statistical_features_dir, model_save_dir)

if __name__ == "__main__":
    main()
