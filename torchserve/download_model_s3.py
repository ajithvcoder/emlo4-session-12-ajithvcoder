import boto3
import os

# Define the bucket name, S3 file key, and local file path
bucket_name = 'mybucket-emlo-mumbai'
s3_file_key = 'sd3_model/sd3.mar'
local_file_path = './model_store/sd3.mar'

# Check if the file already exists
if os.path.exists(local_file_path):
    print(f"File already exists at {local_file_path}")
else:
    # File does not exist, establish connection to S3 and download
    try:
        print("File not found. Establishing connection to S3...")
        s3 = boto3.client('s3')
        print("Connection established.")

        print(f"Starting to download file to {local_file_path}...")
        # Download the file
        s3.download_file(bucket_name, s3_file_key, local_file_path)
        print(f"File downloaded successfully to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
