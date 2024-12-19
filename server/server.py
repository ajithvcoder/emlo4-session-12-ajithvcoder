from typing import Annotated
import io
import os
import numpy as np
import requests
import json
import uuid
import boto3
from botocore.client import Config

from PIL import Image
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

TORCHSERVE_MANAGEMENT_API = os.getenv("TORCHSERVE_MANAGEMENT_API",'http://localhost:8081')
TORCHSERVE_INFERENCE_API = os.getenv("TORCHSERVE_INFERENCE_API","http://localhost:8080")

app = FastAPI(
    title="Stable Diffusion 3",
    description="Text to Image Service using SD3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    config=Config(
        region_name="ap-south-1",
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    ),
)
bucket_name = "mybucket-emlo-mumbai"
objects_prefix = "sd3-outputs"

# Store results in memory
results_map = {}

# Fill results_map with data in S3
try:
    s3_results = s3_client.list_objects(Bucket=bucket_name, Prefix=objects_prefix)
    if "Contents" in s3_results:
        for res in s3_results["Contents"]:
            job_id = res["Key"].split("/")[1]
            results_map[job_id] = {"status": "SUCCESS", "result": res["Key"]}
except Exception as e:
    print(f"Error loading existing S3 results: {e}")

def submit_inference(uid: str, text: str):
    """Submit inference request to torchserve and save to S3"""
    results_map[uid] = {"status": "PENDING"}
    try:
        # Call torchserve endpoint
        response = requests.post(f"{TORCHSERVE_INFERENCE_API}/predictions/sd3", data=text)
        
        if response.status_code != 200:
            raise Exception(f"Torchserve error: {response.text}")
            
        # Construct image from response
        image = Image.fromarray(np.array(json.loads(response.text), dtype="uint8"))
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        
        # Upload to S3
        filename = f"{objects_prefix}/{uid}/result.jpeg"
        s3_client.upload_fileobj(img_bytes, bucket_name, filename)
        
        # Store the S3 reference
        results_map[uid] = {
            "status": "SUCCESS",
            "result": filename
        }
        
    except Exception as e:
        print(f"ERROR :: {e}")
        results_map[uid] = {"status": "ERROR", "message": str(e)}

@app.post("/text-to-image")
async def text_to_image(text: str, background_tasks: BackgroundTasks):
    """Endpoint to submit text-to-image generation request"""
    uid = str(uuid.uuid4())
    background_tasks.add_task(submit_inference, uid, text)
    return {
        "job-id": uid,
        "message": "job submitted successfully"
    }

@app.get("/results/{uid}")
async def get_results(uid: str):
    """Get results for a specific job"""
    if uid not in results_map:
        return {"message": f"job-id={uid} is invalid", "status": "ERROR"}

    result = results_map[uid]
    
    if result["status"] == "SUCCESS":
        # Generate presigned URL for S3 access
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": result["result"]},
            ExpiresIn=3600 * 36,  # 36 hours
        )
        return {
            "status": "SUCCESS",
            "url": presigned_url
        }
    elif result["status"] == "ERROR":
        return {
            "status": "ERROR",
            "message": result.get("message", "Unknown error occurred")
        }
    else:
        return {
            "status": result["status"],
            "message": "Job is still processing"
        }

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Test S3 connection
        s3_client.list_objects(Bucket=bucket_name, Prefix=objects_prefix, MaxKeys=1)
        return {"status": "healthy", "s3_connection": "ok"}
    except Exception as e:
        return {"status": "unhealthy", "s3_connection": str(e)}

@app.get("/")
async def root():
    return {
        "message": "Welcome to SD3 API",
        "endpoints": {
            "POST /text-to-image": "Submit a text-to-image generation request",
            "GET /results/{uid}": "Get results for a specific job",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9080)