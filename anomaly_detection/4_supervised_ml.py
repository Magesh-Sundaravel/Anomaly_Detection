import os

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from anomaly_detection.base import MachineLearning
from anomaly_detection.config import Config

import warnings
warnings.filterwarnings("ignore")


class SupportVectorMachine(MachineLearning):
    def __init__(self, statistical_features_dir):
        super().__init__(statistical_features_dir)

    def train_test_split(self):
        processed_data = self.preprocess_data()
        for file, (data, original_data) in processed_data.items():
            X = data.drop("Anomaly", axis=1)
            y = data["Anomaly"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
            )
            pipeline = Pipeline([("svc", SVC(
                C=Config.SVM_C, kernel=Config.SVM_KERNEL,
                gamma=Config.SVM_GAMMA, random_state=Config.SVM_RANDOM_STATE,
            ))])
            svc = pipeline.fit(X_train, y_train)
            y_pred_svm = svc.predict(X_test)
            test_indices = X_test.index.to_list()
            original_data.loc[test_indices, "SVM_anomaly"] = y_pred_svm
            original_data.to_csv(os.path.join(self.statistical_features_dir, file), index=False)


class RandomForest(MachineLearning):
    def __init__(self, statistical_features_dir):
        super().__init__(statistical_features_dir)

    def train_test_split(self):
        processed_data = self.preprocess_data()
        for file, (data, original_data) in processed_data.items():
            X = data.drop("Anomaly", axis=1)
            y = data["Anomaly"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
            )
            clf = RandomForestClassifier(
                n_estimators=Config.RF_N_ESTIMATORS,
                max_depth=Config.RF_MAX_DEPTH,
                random_state=Config.RF_RANDOM_STATE,
            )
            rf = clf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            test_indices = X_test.index.to_list()
            original_data.loc[test_indices, "RF_anomaly"] = y_pred_rf
            original_data["Anomaly"] = original_data.pop("Anomaly")
            original_data = original_data.dropna()
            original_data.sort_values("Timestamp", inplace=True)
            original_data.to_csv(os.path.join(self.statistical_features_dir, file), index=False)


def supervised_ml(statistical_features_dir):
    SupportVectorMachine(statistical_features_dir).train_test_split()
    RandomForest(statistical_features_dir).train_test_split()
    print("All files have been updated with Supervised Machine Learning Labels")


def run():
    supervised_ml(Config.ML_DATA_DIR)


def main():
    run()


if __name__ == "__main__":
    main()
