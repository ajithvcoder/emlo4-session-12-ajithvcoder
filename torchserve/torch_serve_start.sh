#!/bin/bash

# Run the Python script to download the model from S3
echo "Running download_s3.py to fetch the model..."
python3 download_model_s3.py

# Check if the model exists
if [ ! -f ./model_store/sd3.mar ]; then
    echo "Error: The model file (sd3.mar) was not downloaded."
    exit 1
fi

echo "Model downloaded successfully. Starting TorchServe..."

# Start TorchServe
exec torchserve --start --ts-config=config.properties --model-store model_store --models sd3=sd3.mar --disable-token-auth --ncs --enable-model-api --foreground