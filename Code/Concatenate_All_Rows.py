
########################### **SCRIPT TO CONCATENATE ALL ROWS IN ONE DATAFRAME ** ###################################


#################### Installing and importing necessary libraries ################################################

!pip install dask[dataframe]
import dask.dataframe as dd
import os


################## Utilizing partitions in Dask Dataframe structure to load csv(s) and concatenate rows from all batches ##########

# Base directory containing all the batch folders
base_dir = 'Results'

# Initialize an empty list for DataFrames
dfs = []

# Loop through each batch folder in the base directory
for batch_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, batch_folder)
    
    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # Process CSV files ending with 1.csv, 2.csv, and 3.csv in each folder
        for i in range(1, 4):
            file_path = os.path.join(folder_path, f'embeddings_df_{i}.csv')
            
            # Read each CSV with Dask if it exists
            if os.path.exists(file_path):
                df = dd.read_csv(file_path)
                dfs.append(df)
            else:
                print(f"File {file_path} does not exist.")
                
                
################################################ Concatenate all rows #############################################

# Concatenate all DataFrames in the list
concatenated_data = dd.concat(dfs, axis=0)


################################################### Save as csv ##################################################

# Save the concatenated data as a single CSV file
concatenated_data.to_csv("Results/Concatenated_output_all.csv", single_file=True, index=False)