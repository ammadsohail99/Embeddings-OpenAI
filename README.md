# Embedding Large Text Data with OpenAI's Text-Embedding-ADA-002 Model

This repository showcases a project that involves processing large-scale text data using OpenAI's text-embedding-ada-002 model. The project implements a scalable and efficient pipeline to generate embeddings for over a million rows of data by leveraging batch processing and automation techniques. 

## 🚀 Project Overview

In this project, following tasks were done:

- Processed over 1 million rows of text data to generate embeddings.
- Used batch processing with batches of 50,000 rows to optimize API usage and manage memory constraints.
- Automated the entire workflow from data preparation to the final output.
- Leveraged Dask DataFrame to handle large datasets efficiently for processing and integration.

The resulting embeddings can be used for downstream tasks such as:

- Text similarity analysis
- Clustering and classification
- Recommendation systems
- Dimensionality reduction for visualization


## 🛠️ Setup Instructions

1. Clone the repository

`git clone https://github.com/your-username/embedding-large-text.git` \
`cd embedding-large-text`

2. Install dependencies

- Use the provided requirements.txt to set up your environment:

`pip install -r requirements.txt`

3. API Key Configuration

- Set up your OpenAI API key as an environment variable:

`export OPENAI_API_KEY="your_openai_api_key"`


## 📝 Key Features

- Batch API Calls: Handles large-scale data by breaking it into smaller, manageable batches.
- Scalability: Designed to work with datasets containing millions of rows.
- Flexibility: Easily adjustable batch sizes and customizable API parameters.
- Efficiency: Uses Dask for memory-efficient data manipulation.


## 🔧 Technical Stack

- Python: Core language for scripts
- OpenAI API: Text embedding generation
- Dask: Scalable data processing
- Pandas: Data manipulation
- JSONL: Input format for API compatibility


## 🤝 Contributions

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.


## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.


📬 Contact

For questions or collaboration, feel free to reach out at [syed.sohail@mail.mcgill.ca].
