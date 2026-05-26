import argparse
import importlib
import sys

from anomaly_detection.config import Config


_STEP_MODULES = {
    1: "anomaly_detection.1_separate_and_reconstruct_anomalies",
    2: "anomaly_detection.2_genearte_synthetic_anomalies",
    3: "anomaly_detection.3_create_features",
    4: "anomaly_detection.4_supervised_ml",
    5: "anomaly_detection.5_plot_anomalies",
    6: "anomaly_detection.6_anomaly_imputer",
    7: "anomaly_detection.7_model_save",
}

_STEP_LABELS = {
    1: "Separate & reconstruct anomaly labels",
    2: "Generate synthetic anomalies",
    3: "Create unsupervised features",
    4: "Train supervised models (SVM + RF)",
    5: "Plot anomaly detection results",
    6: "Impute anomalies",
    7: "Save trained models to disk",
}


def run_step(step_num):
    print(f"\n[Step {step_num}] {_STEP_LABELS[step_num]}")
    mod = importlib.import_module(_STEP_MODULES[step_num])
    mod.run()


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly detection pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k}: {v}" for k, v in _STEP_LABELS.items()),
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers to run (e.g. --steps 3,4,5). "
             "Defaults to all steps.",
    )
    args = parser.parse_args()

    if args.steps:
        try:
            steps = [int(s.strip()) for s in args.steps.split(",")]
        except ValueError:
            print("Error: --steps must be comma-separated integers, e.g. --steps 3,4,5")
            sys.exit(1)
        invalid = [s for s in steps if s not in _STEP_MODULES]
        if invalid:
            print(f"Error: unknown step(s): {invalid}. Valid steps: {list(_STEP_MODULES)}")
            sys.exit(1)
    else:
        steps = sorted(_STEP_MODULES)

    print(f"Data root: {Config.DATA_DIR}")
    print(f"Running steps: {steps}")

    for step in steps:
        run_step(step)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
