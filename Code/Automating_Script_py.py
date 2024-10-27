
###############  ** AUTOMATING THE OPENAI MODEL ON THE 50000 RECORDS (3 BATCHES) ** ############################



############################################# Creating .jsonl file ##############################################
#### We are creating 3 .jsonl files per batch comprising of 50000 rows each (last batch has 46538 each)


from tqdm import tqdm
    
import pandas as pd
import json

no_of_batches = 3

# Load the CSV file into a DataFrame
csv_file = 'Dataset/unique_descriptors.csv'  # Replace with your actual CSV file path
df = pd.read_csv(csv_file, encoding='ISO-8859-1')

#Select >150000 and <=300000 records
df = df[1050000:1200000]

# Impute missing values as "NA"
df.fillna("NA", inplace=True)

# Function to construct each record in the required JSON format
def construct_json_record(row):
    return {
        "custom_id": str(row['d_text_id']),  # text_id should be a string
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "input": row['d_text'],  # d_text is the row containing text
            "model": "text-embedding-3-small",  # Model
            "encoding_format": "float"  # Optional
        }
    }

# Batch size based on total rows and number of batches
batch_size = len(df) // no_of_batches

# Loop to create batches and corresponding DataFrames dynamically
for i in range(no_of_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_df = df[start_idx:end_idx]

    # Create dynamically named DataFrames using globals()
    globals()[f'batch_number_{i+1}'] = batch_df

    # Output JSONL file path for each batch
    output_file = f'Batches/Batch_8/batch_{i+1}_50000_rows.jsonl'    ##49615 with entire dataset
    
    # Write each row of the sampled CSV as a JSON object in the .jsonl file
    with open(output_file, 'w') as f:
        for _, row in tqdm(batch_df.iterrows(), total=batch_df.shape[0], desc=f"Writing to JSONL batch {i+1}"):
            json_record = construct_json_record(row)
            f.write(json.dumps(json_record) + '\n')
    
    print(f"JSONL file created: {output_file}")
    
    
############################################ Extracting API Key ########################################################


# Extracting credentials
with open("openapi-key-ammad-2024.txt",'rb') as keys:
    api_key = keys.read().decode('utf-8').strip()
    
###################################### Uploading your Batch Input and Creating the Batch ##############################
#### For this you need to ensure that directory is empty and only contains the .jsonl input files we created above

from openai import OpenAI
import os

client = OpenAI(api_key=api_key)

# Directory containing the batch files
batch_directory = 'Batches/Batch_8'
i = 0

# Loop through each batch file, upload it, and create a batch
for batch_file in os.listdir(batch_directory):
    if batch_file.endswith('.jsonl'):
        i += 1
        var_name = f"batch_input_file_{i}"
        file_path = os.path.join(batch_directory, batch_file)

        # Upload the file and store the result in a dynamically named variable
        globals()[var_name] = client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )

        # Access the file ID
        batch_input_file_id = globals()[var_name].id

        vars = f"batch_df_{i}"

        # Create a batch using the uploaded file's ID
        globals()[vars] = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={
                "description": f"sample_testing_{batch_file}"
            }
        )

        print(f"Batch created for {batch_file} with file ID: {batch_input_file_id}")
    


############################################## Checking the status of the Batch ########################################


retrieval_list = []

for i in range(1, no_of_batches + 1):
    retrieval_list.append(globals()[f"batch_df_{i}"].id)

i = 0
# Automate the retrieval of batch results for each created batch
for batch_id in retrieval_list:
    # Retrieve batch details
    i += 1
    globals()[f"batch_details_{i}"] = client.batches.retrieve(batch_id)
    # You can now process or print the batch details
    print(f"Batch details for {batch_id}: {globals()[f'batch_details_{i}']}")
    


########################################### If you want to cancel the batch #############################################


# i = 0
# # Automate the cancellation of each created batch
# for batch_id in retrieval_list:
#     # Retrieve batch details to check status
#     i += 1
#     batch_details = client.batches.retrieve(batch_id)
#     batch_status = batch_details.status

#     if batch_status != 'failed':
#         # Cancel the batch if it is not failed
#         globals()[f"batch_cancel_{i}"] = client.batches.cancel(batch_id)
#         print(f"Batch {batch_id} cancellation status: {globals()[f'batch_cancel_{i}']}")
#     else:
#         # Handle the failed batch case
#         print(f"Batch {batch_id} cannot be canceled because its status is 'failed'.")



########################################## Obtaining results ############################################################


outputs = []
for i in range(1,no_of_batches+1):
    outputs.append(globals()[f"batch_details_{i}"].output_file_id)

i=0
for file_id in outputs:
    i+=1
    globals()[f"file_response_{i}"] = client.files.content(file_id)
    
#################################### Optional: If we want to save the output .JSONl file ################################

## Save all file_response in .txt file
with open("E:/McGill Material/RA/outputs.txt", "w") as f:
    for i in range(1, no_of_batches + 1):
        f.write(globals()[f"file_response_{i}"].text)


################################## Parsing the JSONL file to extract the embeddings #####################################


import json

for i in range(1,no_of_batches+1):
    # Split the text by new lines or some other delimiter
    globals()[f"json_strings_{i}"] = globals()[f"file_response_{i}"].text.splitlines()

for i in range(1,no_of_batches+1):
    # Parse each JSON object separately
    globals()[f"data_{i}"] = [json.loads(json_string) for json_string in globals()[f"json_strings_{i}"] if json_string.strip()]
    
    
############################################# Extracting the embeddings ##################################################


# Initialize lists for storing rows dynamically
for i in range(1, no_of_batches+1):
    globals()[f"rows_{i}"] = []

# Loop through each data batch
for i in range(1, no_of_batches+1):
    # Extract relevant fields for each batch
    for item in globals()[f"data_{i}"]:
        custom_id = item['custom_id']
        embedding = item['response']['body']['data'][0]['embedding']

        # Append each 'custom_id' and 'embedding' to the corresponding rows list
        globals()[f"rows_{i}"].append({'custom_id': custom_id, 'embedding': embedding})

    # Convert the list of rows to a DataFrame
    globals()[f"new_df_{i}"] = pd.DataFrame(globals()[f"rows_{i}"])

    # Creating a copy of the batch DataFrame
    globals()[f"embeddings_df_{i}"] = globals()[f"batch_number_{i}"].copy()

    # Reset the index of the DataFrame
    globals()[f"embeddings_df_{i}"] = globals()[f"embeddings_df_{i}"].reset_index(drop=True)

    # Concatenate the embedding column to the original DataFrame
    globals()[f"embeddings_df_{i}"] = pd.concat([globals()[f"embeddings_df_{i}"], globals()[f"new_df_{i}"]], axis=1)

    # Optional: Print the result to verify
    print(f"embeddings_df_{i} created with shape: {globals()[f'embeddings_df_{i}'].shape}")
    
    
####################################### Saving embeddings as csv ##########################################################


# Loop to save the embeddings_df as csv
for i in range(1, no_of_batches+1):
    globals()[f"embeddings_df_{i}"].drop(["custom_id"],axis=1,inplace=True)
    globals()[f"embeddings_df_{i}"].to_csv(f'Results/Batch_8/embeddings_df_{i}.csv', index=False)

