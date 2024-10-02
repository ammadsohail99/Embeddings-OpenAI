import pandas as pd
import json
import tqdm
from tqdm import tqdm

# Load the CSV file into a DataFrame
csv_file = 'Dataset/unique_descriptors.csv'  # Replace with your actual CSV file path
df = pd.read_csv(csv_file,encoding='ISO-8859-1')

# Randomly sample 1000 records with a fixed seed for reproducibility
random_state = 42 # To make results reproducible
sample_size = 1000

# Ensure the sample size is not greater than the available records
if sample_size > len(df):
    sample_size = len(df)

#Sampling
df_sample = df.sample(n=sample_size, random_state=random_state)

# Output JSONL file path
output_file = 'batch_1.jsonl'

# Function to construct each record in the required JSON format
def construct_json_record(row):
    return {
        "custom_id": str(row['d_text_id']), # text_id should be a string
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "input": row['d_text'],  # d_text is the row containing text
            "model": "text-embedding-3-small",  # Model
            "encoding_format": "float"  # Optional
        }
    }

# Write each row of the sampled CSV as a JSON object in the .jsonl file
with open(output_file, 'w') as f:
    for _, row in tqdm(df_sample.iterrows(), total=df_sample.shape[0], desc="Writing to JSONL"):
        json_record = construct_json_record(row)
        f.write(json.dumps(json_record) + '\n')

print(f"JSONL file created: {output_file}")


######################## Extracting API Key #############################

# Extracting credentials
with open("openapi-key-ammad-2024.txt",'rb') as keys:
    api_key = keys.read().decode('utf-8').strip()
    

####################### Uploading Your Batch Input File #########################

from openai import OpenAI
client = OpenAI(api_key=api_key)

batch_input_file = client.files.create(
  file=open("batch_1.jsonl", "rb"),
  purpose="batch"
)

####################### Creating the Batch ###################################

batch_input_file_id = batch_input_file.id

client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/embeddings",
    completion_window="24h",
    metadata={
      "description": "sample_testing_iteration_1"
    }
)

###################### Checking the Status of the Batch ########################

client.batches.retrieve('batch_66fcd3b3781c81909a825ca9b14ea7ba')

###################### Retrieving the Results ##################################

file_response = client.files.content("file-rrE1pyXAWhPPggN6OD91QxFK")
# print(file_response.text)

##################### Parsing the Response File ###################################

# Split the text by new lines or some other delimiter
json_strings = file_response.text.splitlines()

# Parse each JSON object separately
data = [json.loads(json_string) for json_string in json_strings if json_string.strip()]


#################### Extracting Embeddings column and Concatenating it to original dataframe ########################

rows = []

# Extract relevant fields
for item in data:
    custom_id = item['custom_id']
    embedding = item['response']['body']['data'][0]['embedding']
    
    # Create a row with 'custom_id' and 'embedding'
    rows.append({'custom_id': custom_id, 'embedding': embedding})

# Convert the list of rows to a DataFrame
new_df = pd.DataFrame(rows)

# Creating a copy
embeddings_df = df_sample.copy()

# Reset the index of embeddings_df
embeddings_df = embeddings_df.reset_index(drop=True)

# Concatenate the embedding column to embeddings_df
embeddings_df = pd.concat([embeddings_df, new_df['embedding']], axis=1)

# Final dataframe
# embeddings_df

################################### Exporting as .xlsx file #####################################

embeddings_df.to_excel("Results/Sample_1000_iteration_1.xlsx",index=False, engine='openpyxl')


