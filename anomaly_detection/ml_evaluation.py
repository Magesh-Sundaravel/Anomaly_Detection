import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from anomaly_detection.base import MachineLearning
from anomaly_detection.config import Config

import warnings
warnings.filterwarnings("ignore")


def _save_confusion_matrix(cm, model_name, output_dir, file_stem):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix\n{file_stem}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_{model_name}_{file_stem}.png"), dpi=100)
    plt.close()


class SupportVectorMachine(MachineLearning):
    def __init__(self, statistical_features_dir):
        super().__init__(statistical_features_dir)

    def evaluate(self, tune_hyperparams=False):
        processed_data = self.preprocess_data()
        cm_dir = str(Config.METRICS_DIR / "confusion_matrices")

        for file, (data, _) in processed_data.items():
            file_stem = file.replace(".csv", "")
            X = data.drop("Anomaly", axis=1)
            y = data["Anomaly"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.EVAL_TEST_SIZE,
                stratify=y, random_state=Config.RANDOM_STATE,
            )

            if tune_hyperparams:
                param_dist = {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto", 0.01, 0.1],
                }
                search = RandomizedSearchCV(
                    SVC(), param_distributions=param_dist,
                    n_iter=10, cv=3, scoring="f1", random_state=Config.RANDOM_STATE,
                )
                search.fit(X_train, y_train)
                clf = search.best_estimator_
                print(f"SVM best params ({file_stem}): {search.best_params_}")
            else:
                clf = SVC(
                    C=Config.SVM_C, kernel=Config.SVM_KERNEL,
                    gamma=Config.SVM_GAMMA, random_state=Config.SVM_RANDOM_STATE,
                )
                clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            print(f"\n=== SVM — {file_stem} ===")
            print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

            cv = StratifiedKFold(
                n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE
            )
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1", n_jobs=1)
            print(
                f"Cross-val F1 ({Config.CV_FOLDS}-fold): "
                f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            )

            _save_confusion_matrix(confusion_matrix(y_test, y_pred), "SVM", cm_dir, file_stem)


class RandomForest(MachineLearning):
    def __init__(self, statistical_features_dir):
        super().__init__(statistical_features_dir)

    def evaluate(self, tune_hyperparams=False):
        processed_data = self.preprocess_data()
        cm_dir = str(Config.METRICS_DIR / "confusion_matrices")

        for file, (data, _) in processed_data.items():
            file_stem = file.replace(".csv", "")
            X = data.drop("Anomaly", axis=1)
            y = data["Anomaly"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.EVAL_TEST_SIZE,
                stratify=y, random_state=Config.RANDOM_STATE,
            )

            if tune_hyperparams:
                param_dist = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                }
                search = RandomizedSearchCV(
                    RandomForestClassifier(random_state=Config.RF_RANDOM_STATE),
                    param_distributions=param_dist,
                    n_iter=10, cv=3, scoring="f1", random_state=Config.RANDOM_STATE,
                )
                search.fit(X_train, y_train)
                clf = search.best_estimator_
                print(f"RF best params ({file_stem}): {search.best_params_}")
            else:
                clf = RandomForestClassifier(
                    n_estimators=Config.RF_N_ESTIMATORS,
                    max_depth=Config.RF_MAX_DEPTH,
                    random_state=Config.RF_RANDOM_STATE,
                )
                clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            print(f"\n=== Random Forest — {file_stem} ===")
            print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

            cv = StratifiedKFold(
                n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE
            )
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1", n_jobs=1)
            print(
                f"Cross-val F1 ({Config.CV_FOLDS}-fold): "
                f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            )

            _save_confusion_matrix(confusion_matrix(y_test, y_pred), "RF", cm_dir, file_stem)


def evaluate(statistical_features_dir, tune_hyperparams=False):
    SupportVectorMachine(statistical_features_dir).evaluate(tune_hyperparams)
    RandomForest(statistical_features_dir).evaluate(tune_hyperparams)


def run():
    evaluate(Config.ML_DATA_DIR)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection models")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run RandomizedSearchCV hyperparameter tuning before evaluating",
    )
    args = parser.parse_args()
    evaluate(Config.ML_DATA_DIR, tune_hyperparams=args.tune)


if __name__ == "__main__":
    main()
