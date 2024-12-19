

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# AWS S3 Configuration
s3 = boto3.client('s3')
# https://mybucket-emlo-mumbai.s3.ap-south-1.amazonaws.com/sd3_model/
# Define local file path and the S3 bucket details
local_file_path = './model_store/sd3.mar'
bucket_name = 'mybucket-emlo-mumbai'
s3_file_key = 'sd3_model/sd3.mar'  # This is the path where the file will be saved in the S3 bucket

try:
    # Upload the file to the specified S3 bucket
    s3.upload_file(local_file_path, bucket_name, s3_file_key)
    print(f'File uploaded successfully to s3://{bucket_name}/{s3_file_key}')
except FileNotFoundError:
    print(f"The file {local_file_path} was not found.")
except NoCredentialsError:
    print("Credentials not available.")
except PartialCredentialsError:
    print("Incomplete credentials provided.")
except Exception as e:
    print(f"Error occurred: {e}")
