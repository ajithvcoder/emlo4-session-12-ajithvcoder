import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import login

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long
from einops import rearrange

def setup_huggingface_auth(token: str):
    print("Logging into Hugging Face...")
    login(token=token, write_permission=False)

def get_models(name: str, device: torch.device):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    model = load_flow_model_quintized(name, device="cpu")
    model.eval()
    ae = load_ae(name, device="cpu")
    return model, ae, t5, clip

@torch.inference_mode()
def generate_image(model, ae, t5, clip, pulid_model, prompt, id_image_path, start_step, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fixed parameters
    width, height = 896, 1152
    num_steps = 20
    guidance = 4.0
    seed = -1
    
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if opts.seed is None:
        opts.seed = torch.Generator(device="cpu").seed()
    print(f"Generating '{opts.prompt}' with seed {opts.seed}")

    # Move t5 and clip to GPU for preparation
    t5, clip = t5.to(device), clip.to(device)
    x = get_noise(
        1,
        opts.height,
        opts.width,
        device=device,
        dtype=torch.bfloat16,
        seed=opts.seed,
    )
    timesteps = get_schedule(
        opts.num_steps,
        x.shape[-1] * x.shape[-2] // 4,
        shift=True,
    )

    inp = prepare(t5=t5, clip=clip, img=x, prompt=opts.prompt)
    
    # Offload t5 and clip back to CPU
    t5, clip = t5.cpu(), clip.cpu()
    torch.cuda.empty_cache()

    # Move PuLID components to GPU for ID embedding
    pulid_model.components_to_device(device)
    
    # Process ID image
    id_image = Image.open(id_image_path)
    id_image = np.array(id_image)
    id_image = resize_numpy_image_long(id_image, 1024)
    id_embeddings, _ = pulid_model.get_id_embedding(id_image, cal_uncond=False)
    
    # Offload PuLID components back to CPU
    pulid_model.components_to_device(torch.device("cpu"))
    torch.cuda.empty_cache()

    # Move model to GPU for generation
    model = model.to(device)
    
    # Generate image
    x = denoise(
        model, 
        **inp, 
        timesteps=timesteps, 
        guidance=opts.guidance, 
        id=id_embeddings, 
        id_weight=1.0,
        start_step=start_step,
    )

    # Offload model to CPU and move decoder to GPU
    model = model.cpu()
    torch.cuda.empty_cache()
    ae.decoder = ae.decoder.to(device)

    # Decode latents
    x = unpack(x.float(), opts.height, opts.width)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    # Offload decoder back to CPU
    ae.decoder = ae.decoder.cpu()
    torch.cuda.empty_cache()

    # Convert to PIL image
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    
    # Save image
    img.save(output_path)
    print(f"Image saved to {output_path}")
    return img

# python infer.py --prompt "portrait, pixar" --id_image /home/ubuntu/11-serverless/PuLID/1592308438962.jpeg --start_step 1 --output result.png

def main():
    parser = argparse.ArgumentParser(description="PuLID FLUX CLI")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--id_image", type=str, required=True, help="Path to ID image")
    parser.add_argument("--start_step", type=int, default=0, help="Timestep to start inserting ID")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--hf_token", type=str, default="", 
                      help="Hugging Face authentication token")
    parser.add_argument("--onnx_provider", type=str, default="gpu", choices=["gpu", "cpu"],
                      help="set onnx_provider to cpu (default gpu) can help reduce RAM usage")
    args = parser.parse_args()

    setup_huggingface_auth(args.hf_token)
    
    device = torch.device(args.device)
    
    # Initialize models
    print("Loading models...")
    model, ae, t5, clip = get_models("flux-dev", device)
    pulid_model = PuLIDPipeline(model, device="cpu", weight_dtype=torch.bfloat16,
                               onnx_provider=args.onnx_provider)
    # Set face helper device to CUDA for better performance
    pulid_model.face_helper.face_det.mean_tensor = pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
    pulid_model.face_helper.face_det.device = torch.device("cuda")
    pulid_model.face_helper.device = torch.device("cuda")
    pulid_model.device = torch.device("cuda")
    pulid_model.load_pretrain(version="v0.9.1")

    # Generate image
    print("Generating image...")
    generate_image(
        model=model,
        ae=ae,
        t5=t5,
        clip=clip,
        pulid_model=pulid_model,
        prompt=args.prompt,
        id_image_path=args.id_image,
        start_step=args.start_step,
        output_path=args.output
    )

if __name__ == "__main__":
    main() 