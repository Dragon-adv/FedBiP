import logging
import os
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPProcessor, CLIPModel

from model import model_types
from config import parse_args
from utils_model import save_model, load_model

from PIL import Image
import clip

args = parse_args()    

def unfreeze_layers_unet(unet, condition):
    print("Num trainable params unet: ", sum(p.numel() for p in unet.parameters() if p.requires_grad))
    return unet

def cvtImg(img):
    img = img.permute([0, 2, 3, 1])
    img = img - img.min()
    img = (img / img.max())
    return img.numpy().astype(np.float32)

def show_examples(x):
    plt.figure(figsize=(10, 10))
    imgs = cvtImg(x)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

def show_examples(x):
    plt.figure(figsize=(10, 5),dpi=200)
    imgs = cvtImg(x)
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

def show_images(images):
    images = [np.array(image) for image in images]
    images = np.concatenate(images, axis=1)
    return Image.fromarray(images)

def show_image(image):
    return Image.fromarray(image)

def prompt_with_template(profession, template):
    profession = profession.lower()
    custom_prompt = template.replace("{{placeholder}}", profession)
    return custom_prompt

def get_prompt_embeddings(prompt_domain, prompt_class, labels, tokenizer, text_encoder, device, padding_type="do_not_pad"):
    prompt_init = []
    for cid in labels:
        cid = int(cid)
        if args.dataset in ['bloodmnist', 'dermamnist', 'ucm']:
            c = args.categories[cid].lower().replace("_", " ")           
            padding = True
            max_length=tokenizer.model_max_length
            if args.dataset=='dermamnist':
                prompt_init.append(f'A dermatoscopic image of a {c}, a type of pigmented skin lesions')
            elif args.dataset=='bloodmnist':
                prompt_init.append(f'A microscopic image of a {c}, a type of blood cell')
            else:
                prompt_init.append(f'A centered satellite photo of a {c.lower().replace("_", " ")}')
        else:
            prompt_init.append(f'a X style of a X')            
            padding=True
            max_length=None

    inputs = tokenizer(prompt_init, 
        # max_length=tokenizer.model_max_length, 
        padding=padding,
        max_length=max_length, 
        truncation=True,
        return_tensors="pt"
    )
    input_ids = torch.LongTensor(inputs.input_ids)
    text_f = text_encoder(input_ids.to(device))[0]
    if args.dataset in ['bloodmnist', 'dermamnist', 'ucm']:
        st_idx_map = {
            'bloodmnist': 7,
            'dermamnist': 8,
            'ucm': 7
        }
        start_idx = st_idx_map[args.dataset]
        for idx, cid in enumerate(labels):
            num_prompt_class = len(prompt_class[cid])
            text_f[idx][start_idx:start_idx+num_prompt_class] = prompt_class[cid]
            
        start_idx_domain = 2
        num_prompt_domain_map = {
            'dermamnist': 4,
            'ucm': 3,
        }
        num_prompt_domain = num_prompt_domain_map[args.dataset]
        text_f[:, start_idx_domain:start_idx_domain+num_prompt_domain] = prompt_domain.unsqueeze(0).repeat(labels.shape[0], 1, 1)
        
    else:
        num_prompt_domain = 1
        text_f[:, 2:2+num_prompt_domain] = prompt_domain.unsqueeze(0).repeat(labels.shape[0], 1, 1)
        num_prompt_class = 1
        text_f[:, -1-num_prompt_class:-1] = prompt_class[labels]

    return text_f


def _select_weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _load_with_fallback(output_dir: str, filename_with_idx: str, filename_no_idx: str, map_location="cpu"):
    """
    Try loading artifact from output_dir with idx suffix first; if missing, fall back to non-suffix filename.
    Returns: (obj, used_path)
    """
    p1 = os.path.join(output_dir, filename_with_idx)
    if os.path.exists(p1):
        return torch.load(p1, map_location=map_location), p1
    p2 = os.path.join(output_dir, filename_no_idx)
    return torch.load(p2, map_location=map_location), p2

def main():
    args = parse_args()    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'test_config.yaml'), 'w'))

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    if args.scheduler == 'ddim':
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, 
            beta_schedule="scaled_linear", 
            clip_sample=False, 
            set_alpha_to_one=False,
            num_train_timesteps=1000,
            steps_offset=1,
        )
    elif args.scheduler == 'pndm':
        scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
    elif args.scheduler == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
        )
    else:
        raise NotImplementedError(args.scheduler)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    num_concepts=7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = _select_weight_dtype(getattr(args, "mixed_precision", "no"))
    # Use the same dtype for latents as model weights for numerical consistency.
    # (We patched diffusers' debug histc path to avoid bf16 incompatibilities.)
    latents_dtype = weight_dtype

    model=StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    model = model.to(device)
    # Precision control for inference (prefer args.mixed_precision over legacy args.fp16)
    model.unet.to(device=device, dtype=weight_dtype)
    model.vae.to(device=device, dtype=weight_dtype)
    model.text_encoder.to(device=device, dtype=weight_dtype)

    dataloader = None
    # following https://arxiv.org/pdf/2306.16064
    categories = args.categories

    def generate_data_per_domain_prompt(model, categories, unet, device, args, idx=None):
        domain = args.domain
        domains = args.domains
        sample_per_class_per_domain = 160
        
        did = domains.index(domain)

        # Load latent statistics from args.output_dir (Windows-friendly).
        if idx is not None:
            latents_mean, _ = _load_with_fallback(
                args.output_dir, f"mean_{idx}.pt", "mean.pt", map_location="cpu"
            )
            latents_std, _ = _load_with_fallback(
                args.output_dir, f"std_{idx}.pt", "std.pt", map_location="cpu"
            )
        else:
            latents_mean = torch.load(os.path.join(args.output_dir, "mean.pt"), map_location="cpu")
            latents_std = torch.load(os.path.join(args.output_dir, "std.pt"), map_location="cpu")

        latents_mean = latents_mean.to(device=device, dtype=latents_dtype)
        latents_std = latents_std.to(device=device, dtype=latents_dtype)
        concept = None

        # Load soft prompts from args.output_dir (Windows-friendly).
        if idx is not None:
            prompt_class, _ = _load_with_fallback(
                args.output_dir, f"prompt_class_{idx}.pth", "prompt_class.pth", map_location="cpu"
            )
            prompt_domain, _ = _load_with_fallback(
                args.output_dir, f"prompt_domain_{idx}.pth", "prompt_domain.pth", map_location="cpu"
            )
        else:
            prompt_class = torch.load(os.path.join(args.output_dir, "prompt_class.pth"), map_location="cpu")
            prompt_domain = torch.load(os.path.join(args.output_dir, "prompt_domain.pth"), map_location="cpu")

        if torch.is_tensor(prompt_class):
            prompt_class = prompt_class.to(device=device, dtype=weight_dtype)
        if torch.is_tensor(prompt_domain):
            prompt_domain = prompt_domain.to(device=device, dtype=weight_dtype)

        if "wnoise" in args.test_type and prompt_domain is not None:
            # add random noise to spec_concept     
            intensity = float(args.test_type.split("wnoise_")[1][:3])
            prompt_domain = prompt_domain + torch.randn_like(prompt_domain) * intensity

        # Save generated images under args.output_dir/generated_images/...
        images_root = os.path.join(args.output_dir, "generated_images")
        client_tag = f"client_{idx}" if idx is not None else "client"

        with torch.inference_mode():
            for cid, c in enumerate(categories):
                save_image_dir = os.path.join(images_root, client_tag, domain, args.test_type, c)
                os.makedirs(save_image_dir, exist_ok=True)

                for i in range(sample_per_class_per_domain):
                    if not (i >= args.start_idx and i <= args.end_idx):
                        continue
                    seed = did * 1000000 + cid * 1000 + i
                    labels = torch.tensor([cid], device=device)
                    prompt_embeds = get_prompt_embeddings(
                        prompt_domain,
                        prompt_class,
                        labels,
                        tokenizer,
                        text_encoder,
                        device=device,
                        padding_type="max_length",
                    )

                    sample = torch.randn(latents_mean.shape, device=device, dtype=latents_dtype)
                    latent = latents_mean + latents_std * sample
                    image = predict_cond(
                        model=model,
                        prompt=None,
                        prompt_embeds=prompt_embeds,
                        seed=seed,
                        condition=concept,
                        img_size=args.resolution,
                        num_inference_steps=args.num_inference_steps,
                        negative_prompt=args.negative_prompt,
                        latent=latent,
                    )
                    image.save(os.path.join(save_image_dir, f"{i}.jpg"))
    
    # Single-run generation: external scripts should schedule multiple clients/domains sequentially.
    requested_idx = getattr(args, "client_num", None)
    idx_to_use = None
    if requested_idx is not None and requested_idx >= 0:
        # Only use idx if the corresponding artifacts exist; otherwise fall back to non-suffix files.
        if os.path.exists(os.path.join(args.output_dir, f"mean_{requested_idx}.pt")) or os.path.exists(
            os.path.join(args.output_dir, f"prompt_domain_{requested_idx}.pth")
        ):
            idx_to_use = requested_idx
    # If user didn't request a valid idx, but multi-client artifacts exist, default to client 0.
    if idx_to_use is None and (
        os.path.exists(os.path.join(args.output_dir, "mean_0.pt"))
        or os.path.exists(os.path.join(args.output_dir, "prompt_domain_0.pth"))
        or os.path.exists(os.path.join(args.output_dir, "prompt_class_0.pth"))
    ):
        idx_to_use = 0

    generate_data_per_domain_prompt(
        model=model,
        categories=categories,
        unet=unet,
        device=device,
        args=args,
        idx=idx_to_use,
    )

def predict_cond(model, 
                prompt, 
                seed, 
                condition, 
                img_size,
                num_inference_steps=50,
                interpolator=None, 
                negative_prompt=None,
                latent=None,
                prompt_embeds=None,
                ):
    
    gen_device = None
    if latent is not None:
        gen_device = latent.device.type
    elif prompt_embeds is not None:
        gen_device = prompt_embeds.device.type
    else:
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed) if seed is not None else None
    output = model(prompt=prompt, prompt_embeds=prompt_embeds,
                height=img_size, width=img_size, 
                num_inference_steps=num_inference_steps, 
                generator=generator, 
                controlnet_cond=condition,
                controlnet_interpolator=interpolator,
                negative_prompt=negative_prompt,
                latents=latent.unsqueeze(0) if latent is not None else None,
                )
    image = output[0][0]
    return image

if __name__ == "__main__":
    main()