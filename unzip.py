import zipfile
import os

# Define paths
zip_file_path = 'solar_data.zip'
extract_dir = 'data/'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Unzipping completed.")
