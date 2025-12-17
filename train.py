import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional
from tqdm.auto import tqdm
import json

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from ruamel.yaml import YAML

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from model import model_types
from config import parse_args
from utils_model import save_model, load_model

import wandb 
from PIL import Image

args = parse_args()
logger = get_logger(__name__)
domains = args.domains

# ==================== [File Logging Utility] ====================
def _write_file_log(message, log_file_path, epoch=None, is_main_process=True):
    if log_file_path and is_main_process:
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                prefix = f"[{epoch:02d}] " if epoch is not None else ""
                f.write(f"{prefix}{message}\n")
        except Exception as e:
            # Fallback to print if file logging fails, but suppress flush=True for console
            print(f"File Logging Failed: {e} - Message: {message}") 
# ================================================================

def get_prompt_embeddings(prompt_domain, prompt_class, labels, tokenizer, 
                          text_encoder, padding_type="do_not_pad", 
                          num_prompt_class=None, num_prompt_domain=None):
    prompt_init = []
    for cid in labels:

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
    text_f = text_encoder(input_ids.to('cuda'))[0]
    if args.dataset in ['bloodmnist', 'dermamnist', 'ucm']:
        st_idx_map_class = {
            'bloodmnist': 7,
            'dermamnist': 8,
            'ucm': 7
        }
        start_idx = st_idx_map_class[args.dataset]
        for idx, cid in enumerate(labels):
            text_f[idx][start_idx:start_idx+num_prompt_class[cid]] = prompt_class[cid]

        start_idx_domain = 2
        num_prompt_domain_map = {
            'dermamnist': 4,
            'ucm': 3,
        }
        num_prompt_domain = num_prompt_domain_map[args.dataset]
        for idx, cid in enumerate(labels):
            text_f[idx][start_idx_domain:start_idx_domain+num_prompt_domain] = prompt_domain

    else:
        num_prompt_domain = 1
        text_f[:, 2:2+num_prompt_domain] = prompt_domain.unsqueeze(0).repeat(labels.shape[0], 1, 1)
        num_prompt_class = 1
        text_f[:, -1-num_prompt_class:-1] = prompt_class[labels]

    return text_f

def log_validation(latents_test, prompt_domain, prompt_class, vae, text_encoder, tokenizer, unet, args, accelerator, scheduler, epoch, num_prompt_class=None, log_file_path=None):
    categories = args.categories

    # 定义文件写入函数
    def file_write(message):
        _write_file_log(message, log_file_path, epoch, accelerator.is_main_process)
        
    # --- 关键诊断行：如果这一行都没有写入，说明函数调用本身失败 ---
    file_write(f"--- DEBUG: log_validation called for Epoch {epoch} ---") 
    # -------------------------------------------------------------
        
    device=torch.device('cuda')

    # ========== [FIX: 包装模型初始化和移动到设备的代码块] ==========
    model = None
    try:
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
        model=model.to(device)
        model.set_progress_bar_config(disable=True)
    except Exception as e:
        file_write(f"FATAL ERROR during Pipeline Setup (OOM?): {type(e).__name__}: {e}")
        return # 如果出错，直接退出函数
    # =============================================================

    def predict_cond(model, latent, prompt, prompt_embds, seed, condition, img_size):
        generator = torch.Generator("cuda").manual_seed(seed)
        output = model(
            prompt=prompt, prompt_embeds=prompt_embds,
            height=img_size, width=img_size, latents=latent.unsqueeze(0),
            num_inference_steps=20, generator=generator, controlnet_cond=condition)        
        image = output[0][0]
        return image

    images=[]
    seed=1023123789
    for i in range(2):
        latents = latents_test[i][0].sample()
        labels = latents_test[i][1]
        if 'concept' in args.train_type:
            concepts = unet.one_hot_concept[labels] 
            prompt = "an image"
            bsz = latents.shape[0]
            prompt_embds = None
        elif 'prompt' in args.train_type:
            concepts = None
            prompt_embds = get_prompt_embeddings(prompt_domain, prompt_class, labels, tokenizer, text_encoder, padding_type="max_length", num_prompt_class=num_prompt_class)
            if prompt_embds is not None:
                prompt_embds = prompt_embds.to(dtype=unet.dtype) # 强制转换类型
            bsz = latents.shape[0]

        for i in range(bsz):
            images.append(predict_cond(
                model=model, latent=latents[i],
                prompt=prompt if prompt_embds is None else None,
                prompt_embds = prompt_embds[i].unsqueeze(0) if prompt_embds is not None else None,
                seed=seed, 
                condition=concepts[i].unsqueeze(0) if concepts is not None else None,
                img_size=args.resolution))
    
    # 仅在主进程且启用了日志记录时执行图像记录
    if accelerator.is_main_process:
        file_write(f"--- VALIDATION LOGGING START (Epoch {epoch}) ---")
        
        # 1. 尝试将图片转换为 Numpy 格式
        try:
            np_images = np.stack([np.asarray(img) for img in images])
        except Exception as e:
            file_write(f"ERROR: Failed to convert images to numpy: {e}")
            del model
            torch.cuda.empty_cache()
            return
            
        if hasattr(accelerator, "trackers"):
            logged_to_tb = False
            logged_to_wandb = False
            
            # 遍历所有找到的 tracker，并打印名称
            for idx, tracker in enumerate(accelerator.trackers):
                file_write(f"DEBUG: Found tracker at index {idx} with name: {tracker.name}")

                if tracker.name.lower() == "tensorboard":
                    # TensorBoard logging
                    tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    file_write("SUCCESS: Images logged to TensorBoard writer.")
                    logged_to_tb = True
                elif tracker.name.lower() == 'wandb':
                    # WandB logging
                    images_log = [np.array(image) for image in images]
                    images_log = np.concatenate(images_log, axis=1)
                    pil_image = Image.fromarray(images_log)
                    downsample_factor = 2
                    downsample_image = pil_image.resize((pil_image.size[0]//downsample_factor, pil_image.size[1]//downsample_factor))
                    
                    # 保持原有的 wandb 记录逻辑
                    tracker.log({"validation_images": wandb.Image(downsample_image)})
                    file_write("SUCCESS: Images logged to WandB (if enabled).")
                    logged_to_wandb = True
            
            if not (logged_to_tb or logged_to_wandb):
                tracker_names = [t.name for t in accelerator.trackers]
                file_write(f"WARNING: Image logging failed. Available tracker names were: {tracker_names}. Expected 'tensorboard' or 'wandb'.")
        else:
            file_write("WARNING: Accelerator has no 'trackers' attribute for image logging. This is unexpected.")

        file_write(f"--- VALIDATION LOGGING END (Epoch {epoch}) ---")

    del model
    torch.cuda.empty_cache()


def main():
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    # ==================== [MODIFICATION 1.1: Handle "none" string] ====================
    if args.report_to == "none":
        args.report_to = None
    # ==================================================================================

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )
    
    # ==================== [FIX: Define LOG_FILE_PATH & Initialize File after accelerator] ====================
    LOG_FILE_PATH = os.path.join(args.output_dir, 'validation_debug.log')
    if args.report_to != "none":
        if accelerator.is_main_process:
            with open(LOG_FILE_PATH, 'w') as f:
                f.write(f"--- STARTING TRAINING RUN ({args.domain}) ---\n")
                f.write(f"Debug Log File Path: {LOG_FILE_PATH}\n")
                f.write(f"Logging Epochs Every: {args.log_every_epochs}\n")
                f.write("----------------------------------------------\n")
    # =======================================================================================================


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # ==================== [FIX 2: Config Sanitization and Tracker Initialization] ====================
        # 创建一个清洗后的配置字典，只保留 TensorBoard 支持的类型，其他的转为字符串
        sanitized_config = {}
        for k, v in vars(args).items():
            # 如果是合法的类型，直接保留
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                sanitized_config[k] = v
            # 如果是 None，转为字符串 "None"
            elif v is None:
                sanitized_config[k] = "None"
            # 其他类型（如 list, dict 等），直接转为字符串表示
            else:
                sanitized_config[k] = str(v)

        # 使用清洗后的配置启动 trackers，这将解决ValueError和图片日志缺失的问题
        accelerator.init_trackers("fedbip_experiment", config=sanitized_config)
        # =================================================================================================

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

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Captions should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example[1] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        domain_ids = torch.tensor([example[2] for example in examples])
        class_ids = torch.tensor([example[3] for example in examples])
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
            "domain_ids": domain_ids,
            "class_ids": class_ids,
        }
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print('weight_dtype',weight_dtype)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # ==================== [MODIFICATION 2: Fine-grained Dtype for stability] ====================
    # Text Encoder (FP32 for numerical stability)
    text_encoder.to(accelerator.device, dtype=torch.float32)
    # VAE (FP32 for numerical stability)
    vae.to(accelerator.device, dtype=torch.float32)
    # UNet (FP16/weight_dtype for low VRAM)
    unet.to(accelerator.device, dtype=weight_dtype)
    # ============================================================================================

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset=='domainnet':
        from domainnet_data import get_dataloader, get_dataloader_domain
    elif args.dataset=='pacs':
        from pacs_data import get_dataloader, get_dataloader_domain
    elif args.dataset=='officehome':
        from officehome_data import get_dataloader, get_dataloader_domain
    elif args.dataset=='ucm':
        from ucm_data import get_dataloader, get_dataloader_domain
    elif args.dataset=='dermamnist':
        from dermamnist_data import get_dataloader, get_dataloader_domain
    elif args.dataset=='bloodmnist':
        from bloodmnist_data import get_dataloader, get_dataloader_domain

    split = 'train'
    args.client_num = 5
    trainloaders = []
    for i in range(args.client_num):
        trainloader = get_dataloader_domain(
                args, args.train_batch_size, None,
                'train', args.domain, tokenize_captions,  
                collate_fn, num_shot=args.num_shot,
                client_id=i)            
        trainloaders.append(trainloader)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(trainloaders[-1]) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(trainloaders[-1].dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    device=torch.device("cuda")
    print("Start training")

    # 创建 2 个随机噪声用于固定视角的验证
    val_latents = torch.randn((2, 4, 64, 64), device=accelerator.device, dtype=weight_dtype)
    val_labels = torch.tensor([0, 1], device=accelerator.device) # 假设测试类别 0 和 1
    latents_test = []
    # 简单封装一下以适配 log_validation 的接口预期
    class MockLatent:
        def sample(self): return val_latents
    latents_test = [[MockLatent(), val_labels], [MockLatent(), val_labels]]
    
    for idx, train_dataloader in enumerate(trainloaders):
        prompt_init = []
        global_step = 0
        loss_history=[]
        train_loss = 0.0
        curious_time=0
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for c in args.categories:
            if args.dataset=='ucm':
                prompt_init.append(f'A centered satellite photo of a {c.lower().replace("_", " ")}')
            elif args.dataset=='bloodmnist':
                prompt_init.append(f'A microscopic image of a {c.lower().replace("_", " ")}, a type of blood cell')
            elif args.dataset=='dermamnist':
                prompt_init.append(f'A dermatoscopic image of a {c}, a type of pigmented skin lesions')
            else:
                prompt_init.append(f'a {args.domain.lower()} style of a {c.lower()}')
        inputs = tokenizer(prompt_init, 
            max_length=tokenizer.model_max_length, 
            padding=True, #"padding", 
            truncation=True,
            return_tensors="pt"
        )

        text_f = text_encoder(inputs.input_ids.to(accelerator.device))[0]
        num_prompt_class = None
        num_prompt_domain = 1
        prompt_domain = text_f[0][2:2+num_prompt_domain]  
        num_prompt_class = 1
        prompt_class = text_f[:, -1-num_prompt_class:-1]  
        prompt_domain.requires_grad_(True)
        prompt_class.requires_grad_(True)
        trainable_params = [prompt_domain, prompt_class]    

        optimizer = torch.optim.Adam(
                trainable_params, # only the weight of MLP will be opitmized
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        for epoch in range(args.num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                # ==================== [MODIFICATION 3.1: VAE Encode Data Dtype to FP32] ====================
                # VAE编码输入像素值为FP32，然后将输出的latents转为FP16 (weight_dtype)
                latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float32)).latent_dist.sample()
                latents = latents * 0.18215
                latents = latents.to(dtype=weight_dtype) # 转换为UNet所需的FP16
                # ============================================================================================

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                labels = batch["class_ids"].to(device)
                if 'concept' in args.train_type:
                    # Q: Why removing the domain concept? 
                    # A: Because we think the domain concept is not able to be learned since each client has only one domain.
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                    class_concepts = unet.one_hot_concept[labels] 
                    # 确保 input_conditions 转为 FP16
                    batch["input_conditions"] = class_concepts.to(device, dtype=weight_dtype)
                elif 'prompt' in args.train_type:
                    # encoder_hidden_states is output as FP32 from get_prompt_embeddings
                    encoder_hidden_states = get_prompt_embeddings(prompt_domain, prompt_class, labels, tokenizer, text_encoder, num_prompt_class=num_prompt_class)
                    batch["input_conditions"] = None

                # ==================== [MODIFICATION 3.2: UNet Input Dtype - 确保 hidden states 为 FP16] ====================
                # 将FP32的 encoder_hidden_states 转换为 UNet 所需的 FP16/weight_dtype
                model_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states.to(dtype=weight_dtype), 
                    controlnet_cond=batch["input_conditions"]).sample
                # ==============================================================================================================
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                train_loss += loss.item()
                curious_time += timesteps.sum().item()

                loss.backward()
                
                # ==================== [MODIFICATION 4: Gradient Clipping for stability] ====================
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                # =========================================================================================

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1
                if global_step%1==0:
                    train_loss = train_loss/1
                    # ==================== [FIX 3: Robust Conditional logging check in main loop] ====================
                    # Scalar logging uses the accelerator built-in log mechanism
                    if accelerator.is_main_process and hasattr(accelerator, "trackers"):
                        accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    # ================================================================================================
                    loss_history.append(train_loss)
                    train_loss = 0.0
                    curious_time = 0

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break
                
                if not args.skip_evaluation and (global_step)%args.log_every_steps==0:
                    if 'concept' in args.train_type:
                        save_model(unet, args.output_dir+'/unet.pth')
                    elif 'prompt' in args.train_type:
                        torch.save(prompt_domain, args.output_dir+f'/prompt_domain_{idx}.pth')
                    elif 'prompt' in args.train_type:
                        torch.save(prompt_class, args.output_dir+f'/prompt_class_{idx}.pth')

                plt.figure()
                plt.plot(loss_history)
                plt.savefig(args.output_dir+'/loss_history.png')
                plt.close()

        # 强制在控制台输出提示信息 (使用 progress_bar.write 绕过 tqdm 干扰)
        if accelerator.is_main_process and (epoch % args.log_every_epochs == 0 or epoch == args.num_train_epochs - 1):
            progress_bar.write(f"--- EPOCH {epoch} END: Triggering Validation... ---", end='\n')

        if epoch%args.log_every_epochs==0 or epoch==args.num_train_epochs-1:        
            # 在调用 log_validation 时传入 log_file_path
            log_validation(latents_test, prompt_domain, prompt_class, vae, text_encoder, tokenizer, unet, args, accelerator, noise_scheduler, epoch, num_prompt_class, log_file_path=LOG_FILE_PATH)
            if 'concept' in args.train_type:
                save_model(unet, args.output_dir+'/unet.pth')
            elif 'prompt' in args.train_type:
                torch.save(prompt_domain, args.output_dir+f'/prompt_domain_{idx}.pth')
            elif 'prompt' in args.train_type:
                torch.save(prompt_class, args.output_dir+f'/prompt_class_{idx}.pth')

if __name__ == "__main__":
    main()