import pandas as pd
import kagglehub 
from kagglehub import KaggleDatasetAdapter

# kagglehub.login()

# Download a dataset from kaggle
# path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
# print("Dataset downloaded to:", path)

# Set the path to the file you'd like to load
file_path = "spam.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "uciml/sms-spam-collection-dataset",
  file_path,
    pandas_kwargs={"encoding": "latin-1"} 
)

print("Dataset shape:", df.shape)
# print("First 5 records:", df.head())