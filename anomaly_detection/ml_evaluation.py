import os
import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report


import warnings

warnings.filterwarnings('ignore')


class MachineLearning:
    def __init__(self, statistical_features_dir):
        self.statistical_features_dir = statistical_features_dir

    def load_data(self):
        csv_files = [file for file in os.listdir(self.statistical_features_dir) if file.endswith(".csv")]
        return {file: pd.read_csv(os.path.join(self.statistical_features_dir, file)) for file in csv_files}

    def preprocess_data(self):
        data_dict = self.load_data()
        processed_data = {}
        scaler = RobustScaler()

        for file, data in data_dict.items():
            column_name  = data.columns[2]
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True)
            data['Day'] = data['Timestamp'].dt.day
            data['Month'] = data['Timestamp'].dt.month
            data['Year'] = data['Timestamp'].dt.year
            data['Hour'] = data['Timestamp'].dt.hour
            data['Minute'] = data['Timestamp'].dt.minute

            data['original_signal'] = scaler.fit_transform(data['original_signal'].values.reshape(-1, 1))
            data[column_name] = scaler.fit_transform(data[column_name].values.reshape(-1,1))

            columns_order = ['Day', 'Month', 'Year', 'Hour', 'Minute', 'original_signal', column_name,
                             'step_variable_ws5', 'step_variable_ws10', 'step_variable_ws15',
                             'std_anomaly_ws5', 'std_anomaly_ws10', 'std_anomaly_ws15',
                             'iqr_anomaly_ws5', 'iqr_anomaly_ws10', 'iqr_anomaly_ws15', 'Anomaly']
            data = data[columns_order]
            X = data.drop("Anomaly", axis=1)
            y = data["Anomaly"]
            processed_data[file] = (X, y)
        return processed_data


class SupportVectorMachine(MachineLearning):
    def __init__(self, statistical_features_dir):
        super().__init__(statistical_features_dir)

    def train_test_split(self):
        processed_data = self.preprocess_data()
        for file, (X, y) in processed_data.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
            #  Define the parameter grid for grid search
            param_grid = {
                'svc__C': [0.1, 1, 10],
                'svc__gamma': [0.1, 1, 10, 'scale', 'auto'],
                'svc__kernel': ['linear', 'rbf']
            }

            # Create pipeline with SVC
            pipeline = Pipeline([('svc', SVC(random_state=101))])

            # Initialize GridSearchCV
            randomized_search = RandomizedSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

            # Perform Grid Search
            randomized_search.fit(X_train, y_train)

            # Get the best parameters
            best_params = randomized_search.best_params_
            print("Best parameters:", best_params)

            # Use the best parameters to instantiate the SVC model
            best_svc = randomized_search.best_estimator_
            best_svc.fit(X_train, y_train)

            # Make predictions
            y_pred_svm = best_svc.predict(X_test)

            # # Print classification report
            # print("Classification Report:")
            # print(classification_report(y_test, y_pred_svm))

class RandomForest(MachineLearning):
    def __init__(self, statistical_features_dir):
        super().__init__(statistical_features_dir)

    def train_test_split(self):
        processed_data = self.preprocess_data()
        for file, (X, y) in processed_data.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

             # Define the parameter grid for grid search
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20]           
            }

            # Initialize Random Forest classifier
            clf = RandomForestClassifier(random_state=101)

            # Initialize GridSearchCV
            randomized_search = RandomizedSearchCV(clf, param_grid, cv=5, n_jobs=-1)

            # Perform Grid Search
            randomized_search.fit(X_train, y_train)

            # Get the best parameters
            best_params = randomized_search.best_params_
            print("Best parameters:", best_params)

            # Use the best parameters to instantiate the Random Forest model
            best_rf = randomized_search.best_estimator_

            # Fit the model
            best_rf.fit(X_train, y_train)

            # Make predictions
            y_pred_rf = best_rf.predict(X_test)

            # # Print classification report
            # print("Classification Report:")
            # print(classification_report(y_test, y_pred_rf))



def supervised_ml(statistical_features_dir):
    svm_classifier = SupportVectorMachine(statistical_features_dir)
    svm_classifier.train_test_split()
    rf_classifier = RandomForest(statistical_features_dir)
    rf_classifier.train_test_split()

def main():
    statistical_features_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/data/processed/statistical_features"
    supervised_ml(statistical_features_dir)

if __name__ == "__main__":
    main()
    