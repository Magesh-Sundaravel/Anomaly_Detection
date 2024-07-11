import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')



class SyntheticAnomalyGenerator:
    def __init__(self, tslab_dir, synthetic_data_dir, num_anomalies=500, x_min=5, x_max=10,
                 y_min=-350, y_max=350, d_min=1, d_max=4):
        self.tslab_dir = tslab_dir
        self.synthetic_data_dir = synthetic_data_dir
        self.num_anomalies = num_anomalies
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.d_min, self.d_max = d_min, d_max
        self.data = {}

    def read_data(self):
        csv_files = os.listdir(self.tslab_dir)
        for file in csv_files:
            try:
                data = pd.read_csv(os.path.join(self.tslab_dir, file))
                self.data[file] = data
            except Exception as e:
                print(f"Error reading {file}: {e}")

    def generate_synthetic_anomaly(self):
        for file, data in self.data.items():
            try:
                data['original_signal'] = data.iloc[:, 1]
                column_name = data.columns[1]
                data = data[['Timestamp', 'original_signal', column_name, 'Anomaly']]
                original_signal = data[column_name]

                current_point = 0
                anomaly_indices = []

                for i in range(self.num_anomalies):
                    time_period = int(np.random.uniform(self.x_min, self.x_max))
                    anomaly_start_point = current_point + time_period
                    anomaly_magnitude = np.random.uniform(self.y_min, self.y_max)
                    anomaly_duration = int(np.random.uniform(self.d_min, self.d_max))

                    for j in range(anomaly_start_point, min(anomaly_start_point + anomaly_duration, len(original_signal))):
                        original_signal.iloc[j] += anomaly_magnitude
                        anomaly_indices.append(j)
                    current_point = anomaly_start_point

                data[column_name] = original_signal.round(2).astype(float)
                data.loc[anomaly_indices, 'Anomaly'] = 1

                output_path = os.path.join(self.synthetic_data_dir, file)
                data.to_csv(output_path, index=False)

            except Exception as e:
                print(f"Error processing {file}: {e}")

        print(f"Synthetic anomalies generated and saved for all the files.")


if __name__ == '__main__':
    tslab_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/data/processed/tslab_anomalies"
    synthetic_data_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/data/processed/synthetic_data"

    if not os.path.exists(synthetic_data_dir):
        os.makedirs(synthetic_data_dir)

    synthetic_anomaly = SyntheticAnomalyGenerator(tslab_dir, synthetic_data_dir)
    synthetic_anomaly.read_data()
    synthetic_anomaly.generate_synthetic_anomaly()
