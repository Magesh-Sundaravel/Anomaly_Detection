import os
import pandas as pd

from anomaly_detection.config import Config


def separate_sensors(file_dir, sensor_dir):
    """
    Separates each sensor's data into individual CSV files.
    
    Parameters:
        file_dir (str): The path to the input CSV file.
        sensor_dir (str): The directory where the output CSV files will be saved.
    """
   
    data = pd.read_csv(file_dir)
    first_column = data.iloc[:, 0]

    if not os.path.exists(sensor_dir):
        os.makedirs(sensor_dir)

    for i in range(1, len(data.columns)):
        second_column = data.iloc[:, i]
        sensor_data = pd.concat([first_column, second_column], axis=1)

        # output_path_dir = os.path.join(sensor_dir, f'{data.columns[i]}.csv')
        # sensor_data.to_csv(output_path_dir, index=False)

    print("CSV files have been successfully separated and saved.")


# This function should only be run if the CSV files have anomalies marked by TSLab.

def reconstruct_columns(tslab_dir):
    # Get all CSV files in the tslab_dir directory
    csv_files = [file for file in os.listdir(tslab_dir) if file.endswith('.csv')]

    for file in csv_files:
        data = pd.read_csv(os.path.join(tslab_dir, file))

        labelled = data[data.columns[2:]]
        no_of_anomalies = len(labelled.columns.to_list())


        if no_of_anomalies > 0:
            data['Anomaly'] = data.filter(like='Label').bfill(axis=1).iloc[:, 0]
            data['Anomaly'] = data['Anomaly'].fillna(0)

            data = data.drop(columns=data.filter(like='Label').columns)

            anomaly_indices = data[data['Anomaly'] == 1].index.to_list()
            
            i = 0 
            while i < len(anomaly_indices): 
                start = anomaly_indices[i] 
                end = start  

                # Find the end of the continuous anomaly
                while end < len(data) - 1 and data.loc[end + 1, 'Anomaly'] == 1:  
                    end += 1 

                data.loc[start, 'Anomaly'] = 0
                data.loc[end , 'Anomaly'] = 0

                # Move to the next segment of anomalies
                i += (end - start + 1)        #  2157-2148 

        else:
            data['Anomaly'] = 0

        # Save the updated DataFrame to a new file
        output_path = os.path.join(tslab_dir, file)
        data.to_csv(output_path, index=False)
    
    print("All the columns have been Reconstructed")
        

def run():
    reconstruct_columns(Config.TSLAB_DIR)


def main():
    run()


if __name__ == "__main__":
    main()
