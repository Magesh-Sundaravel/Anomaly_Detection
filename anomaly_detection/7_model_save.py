import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from anomaly_detection.base import MachineLearning
from anomaly_detection.config import Config

import warnings
warnings.filterwarnings("ignore")


class SupportVectorMachine(MachineLearning):
    def __init__(self, statistical_features_dir, model_save_dir):
        super().__init__(statistical_features_dir, model_save_dir)

    def train_and_save(self):
        processed_data = self.preprocess_data()
        for file, (data, _) in processed_data.items():
            X = data.drop("Anomaly", axis=1)
            y = data["Anomaly"]
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
            )
            pipeline = Pipeline([("svc", SVC(
                C=Config.SVM_C, kernel=Config.SVM_KERNEL,
                gamma=Config.SVM_GAMMA, random_state=Config.SVM_RANDOM_STATE,
            ))])
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, os.path.join(self.model_save_dir, "SVC_model.pkl"))


class RandomForestModel(MachineLearning):
    def __init__(self, statistical_features_dir, model_save_dir):
        super().__init__(statistical_features_dir, model_save_dir)

    def train_and_save(self):
        processed_data = self.preprocess_data()
        for file, (data, _) in processed_data.items():
            X = data.drop("Anomaly", axis=1)
            y = data["Anomaly"]
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
            )
            clf = RandomForestClassifier(
                n_estimators=Config.RF_N_ESTIMATORS,
                max_depth=Config.RF_MAX_DEPTH,
                random_state=Config.RF_RANDOM_STATE,
            )
            clf.fit(X_train, y_train)
            joblib.dump(clf, os.path.join(self.model_save_dir, "RF_model.pkl"))


def supervised_ml(statistical_features_dir, model_save_dir):
    SupportVectorMachine(statistical_features_dir, model_save_dir).train_and_save()
    RandomForestModel(statistical_features_dir, model_save_dir).train_and_save()
    print(f"Models saved to {model_save_dir}")


def run():
    supervised_ml(Config.ML_DATA_DIR, Config.MODEL_DIR)


def main():
    run()


if __name__ == "__main__":
    main()
