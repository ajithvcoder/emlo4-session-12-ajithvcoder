import logging
import zipfile
import time
from abc import ABC

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Loading sd3_handler...")


class SD3Handler(BaseHandler, ABC):
    def __init__(self):
        self.initialized = False
        logger.info("Initializing SD3Handler...")

    def initialize(self, ctx):
        """In this initialize function, the Stable Diffusion 3 model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        start_time = time.time()
        logger.info("Starting initialization...")
        
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        logger.info(f"Model directory: {model_dir}")
        
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA current device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
        logger.info(f'GPU ID: {properties.get("gpu_id")}')

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Log memory status before model loading
        if torch.cuda.is_available():
            logger.info(f"GPU Memory before loading: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        zip_path = model_dir + "/sd3-model.zip"
        extract_path = model_dir + "/model"
        logger.info(f"Starting to extract model from {zip_path} to {extract_path}")
        
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                logger.info(f"Zip file contents: {zip_ref.namelist()}")
                zip_ref.extractall(extract_path)
            logger.info("Model extraction completed successfully")
        except Exception as e:
            logger.error(f"Error during model extraction: {str(e)}")
            raise

        logger.info("Starting to load SD3 pipeline...")
        try:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                extract_path,
                torch_dtype=torch.bfloat16
            )
            logger.info("Pipeline loaded successfully")
            
            logger.info("Moving pipeline to device...")
            self.pipe = self.pipe.to(self.device)
            logger.info("Pipeline successfully moved to device")
            
        except Exception as e:
            logger.error(f"Error during pipeline loading: {str(e)}")
            raise

        # Log memory status after model loading
        if torch.cuda.is_available():
            logger.info(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        self.initialized = True
        end_time = time.time()
        logger.info(f"Initialization completed in {end_time - start_time:.2f} seconds")

    def preprocess(self, requests):
        """Basic text preprocessing, of the user's prompt."""
        logger.info("Starting preprocessing...")
        inputs = []
        for idx, data in enumerate(requests):
            logger.info(f"Processing request {idx + 1}/{len(requests)}")
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
                logger.info("Using body as input text was not found in data")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
                logger.info("Decoded bytes to string")
            logger.info(f"Processed input text: '{input_text}'")
            inputs.append(input_text)
        logger.info(f"Preprocessing completed. Total inputs: {len(inputs)}")
        return inputs

    def inference(self, inputs):
        """Generates the image relevant to the received text."""
        logger.info(f"Starting inference with {len(inputs)} inputs")
        start_time = time.time()
        
        try:
            logger.info("Running pipeline inference...")
            inferences = self.pipe(
                inputs,
                num_inference_steps=28,
                guidance_scale=7.0,
                width=1024,
                height=1024
            ).images
            logger.info(f"Generated {len(inferences)} images")
            
            # Log memory usage after inference
            if torch.cuda.is_available():
                logger.info(f"GPU Memory after inference: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise

        end_time = time.time()
        logger.info(f"Inference completed in {end_time - start_time:.2f} seconds")
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the generated image into Torchserve readable format."""
        logger.info("Starting postprocessing...")
        start_time = time.time()
        
        images = []
        try:
            for idx, image in enumerate(inference_output):
                logger.info(f"Processing output image {idx + 1}/{len(inference_output)}")
                images.append(np.array(image).tolist())
            
            logger.info(f"Postprocessing completed for {len(images)} images")
        except Exception as e:
            logger.error(f"Error during postprocessing: {str(e)}")
            raise

        end_time = time.time()
        logger.info(f"Postprocessing completed in {end_time - start_time:.2f} seconds")
        return images