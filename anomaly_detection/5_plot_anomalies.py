import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")


class PlotAnomalies:
    
    def __init__(self, ml_data_dir, metrics_dir):
        self.ml_data_dir = ml_data_dir
        self.metrics_dir = metrics_dir
        self.dataframes = []

    def load_csv(self, file_names):
        for file_name in file_names:
            file_path = os.path.join(self.ml_data_dir, file_name)
            data = pd.read_csv(file_path)
            self.dataframes.append((file_name, data))
            self.plot_anomalies(data, file_name)

    def plot_all(self):
        for data in self.dataframes:
            self.plot_anomalies(data[1], data[0])

    def plot_anomalies(self, data, file_name):
        num_of_methods = 2       
        fig, axs = plt.subplots(num_of_methods, 1, figsize=(14, 7.5 * (num_of_methods - 1)))

        column_name = data.columns[2]
        timestamp = data['Timestamp']
        sensor_name = data[column_name]
        sns.set_palette('mako')
        file_name = file_name.split('.')[0]

        plots_folder = os.path.join(self.metrics_dir, 'Anomalies_plots')
        os.makedirs(plots_folder, exist_ok=True)

        for i in range(num_of_methods):
            method_name = data.columns[i + 12]
            
            axs[i].plot(timestamp, sensor_name, linewidth=0.5, linestyle='-', marker='.', markersize=4)
            
            true_anomalies = data[data['Anomaly'] == 1]
            axs[i].scatter(true_anomalies['Timestamp'], true_anomalies[column_name], label="True Anomalies",
                           color='green', marker='o', s=50, zorder=3)

            detected_anomalies = data[data[method_name] == 1]
            axs[i].scatter(detected_anomalies['Timestamp'], detected_anomalies[column_name], label=f"{method_name} anomalies",
                           color='red', marker='.', s=25, zorder=5)
            
            axs[i].set_xlabel('Timestamp',fontsize=14)
            axs[i].tick_params(axis='y', which='major', labelsize=14)  
            
            axs[i].set_ylabel("Values",fontsize=14)
            axs[i].set_title(f"Anomalies detected by {method_name}",fontsize=20)

            axs[i].legend(loc='best',fontsize=14,framealpha=0.5)

        plt.tight_layout()
        
        plot_file_path = os.path.join(plots_folder, f'Anomalies_detected_on_{file_name}_by_ML.png')
        plt.savefig(plot_file_path)


def main():
    ml_data_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/data/processed/ml_data"
    metrics_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/reports/metrics"

    file_names = [file_name for file_name in os.listdir(ml_data_dir) if file_name.endswith('.csv')]
    
    plot_anomalies = PlotAnomalies(ml_data_dir,metrics_dir)
    plot_anomalies.load_csv(file_names)
    plot_anomalies.plot_all()


if __name__ == "__main__":
    main()
    
    
    
    
 