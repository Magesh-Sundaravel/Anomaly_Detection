# Description: This script is used to impute synthetic anomalies in the dataset.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AnomalyImputer:
    """Class for imputing synthetic anomalies."""

    def __init__(self, ml_evaluation_dir, imputed_anomalies_dir, imputed_vs_original_dir):
        self.ml_evaluation_dir = ml_evaluation_dir
        self.imputed_anomalies_dir = imputed_anomalies_dir
        self.imputed_vs_original_dir = imputed_vs_original_dir
        self.dataset = self.load_csv_files()

    def load_csv_files(self):
        all_files = os.listdir(self.ml_evaluation_dir)
        csv_files = [file for file in all_files if file.endswith(".csv")]
        df_dict = {}
        
        for file in csv_files:
            file_path = os.path.join(self.ml_evaluation_dir, file)
            df = pd.read_csv(file_path)
            df_dict[file] = df
        return df_dict

    def impute_and_detect_anomalies(self):
        os.makedirs(self.imputed_anomalies_dir, exist_ok=True) 
        
        for file_name, df in self.dataset.items():  
            impute_data = df.iloc[:, 2].copy()
            anomalies_indices = df.index[df['Anomaly'] == 1]
            non_anomalies_indices = df.index[df['Anomaly'] == 0]
            df['impute_data'] = impute_data
            anomaly_mean  = 0
    
            for idx in anomalies_indices:
                if idx > 0 and idx < len(impute_data) - 1:
                    before_indices = [i for i in non_anomalies_indices if i < idx]
                    after_indices = [i for i in non_anomalies_indices if i > idx]

                    if before_indices and after_indices:
                        nearest_before_idx = before_indices[-1]
                        nearest_after_idx = after_indices[0]
                        anomaly_mean = (impute_data[nearest_before_idx] + impute_data[nearest_after_idx]) / 2

                    elif before_indices:
                        nearest_before_idx = before_indices[-1]
                        anomaly_mean = impute_data[nearest_before_idx]

                    elif after_indices:
                        nearest_after_idx = after_indices[0]
                        anomaly_mean = impute_data[nearest_after_idx]
                        
                    else:
                        anomaly_mean = impute_data[idx]  
    
                df.loc[idx,'impute_data'] = anomaly_mean.round(2)
    
            df = df[['Timestamp', 'original_signal', df.columns[2], 'impute_data']]
        
            imputed_file_path = os.path.join(self.imputed_anomalies_dir, file_name)
            df.to_csv(imputed_file_path, index=False)    
        
    def plot_anomalies(self):
        os.makedirs(self.imputed_vs_original_dir, exist_ok=True) 
    
    
        for file_name, df in self.dataset.items():  
            file_path = os.path.join(self.imputed_anomalies_dir, file_name)
            df = pd.read_csv(file_path)
            sensor_name = df.columns[2]
    
            fig, axs = plt.subplots(figsize=(18,10))
            
            axs.plot(df['Timestamp'], df[sensor_name], linewidth=0.5, label='Original Anomalies',color='green')
            axs.plot(df['Timestamp'], df['impute_data'], linewidth=0.4, label='Imputed Anomalies',color='red')
            axs.set_xlabel('Timestamp',fontsize=26)
            axs.set_ylabel('Values',fontsize=26)
            axs.tick_params(axis='y', which='major', labelsize=26)  

            axs.set_title(f"Comparison of original vs imputed_anomalies of {file_name}",fontsize=30)
            axs.legend(loc='best',fontsize=24,framealpha=0.5)

    
            plt.tight_layout()
            plot_file_path = os.path.join(self.imputed_vs_original_dir, f'imputed_vs_original_anomalies_plot{file_name}.png')
            plt.savefig(plot_file_path, dpi=100)

if __name__ == "__main__":
    ml_evaluation_dir = "/media/magesh/HardDisk/Thesis/anomaly_detection/data/processed/ml_data"
    imputed_anomalies_path = "/media/magesh/HardDisk/Thesis/anomaly_detection/data/processed/"

    anomalies_dir_name = "Imputed_anomalies"
    imputed_anomalies_dir = os.path.join(imputed_anomalies_path, anomalies_dir_name)

    imputed_vs_original  = "Imputed_vs_original_anomalies_plot"
    imputed_vs_original_dir = os.path.join(imputed_anomalies_path,imputed_vs_original)

    impute_anomalies = AnomalyImputer(ml_evaluation_dir,  imputed_anomalies_dir,imputed_vs_original_dir)
    impute_anomalies.impute_and_detect_anomalies()

    # impute_anomalies.plot_anomalies()

    print(f'All the files have been updated with imputed data and saved with plots')
