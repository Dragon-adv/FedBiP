import torch
from config import parse_args
import os

import torchvision.models as models
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Optional

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda'
args = parse_args()

tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)

if args.dataset == 'domainnet':
    from domainnet_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'pacs':
    from pacs_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'officehome':
    from officehome_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'bloodmnist':
    # Optional dependency: keep dynamic import to avoid failures when the module is missing.
    import importlib
    _m = importlib.import_module("bloodmnist_data")
    get_dataloader = getattr(_m, "get_dataloader")
    get_dataloader_domain = getattr(_m, "get_dataloader_domain")
elif args.dataset == 'dermamnist':
    from dermamnist_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'ucm':
    from ucm_data import get_dataloader, get_dataloader_domain

def tokenize_captions(examples, is_train=False):
    captions = []
    for caption in examples:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                "Caption values should contain either strings or lists of strings."
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


def _infer_test_type_from_train_setting(train_setting: str) -> str:
    # Examples:
    # - train_syn_wnoise_0.1_interpolated_80 -> syn_wnoise_0.1_interpolated
    # - syn_wnoise_0.1_test_interp -> syn_wnoise_0.1_test_interp
    # - syn_xxx_multiclient_5_80 -> syn_xxx
    s = train_setting
    if s.startswith("train_"):
        s = s[len("train_") :]
    # Drop trailing "_{num_shot}" if present
    parts = s.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        s = "_".join(parts[:-1])
    # Drop multiclient suffix if present: "..._multiclient_{k}" or "..._multiclient_{i}"
    if "_multiclient_" in s:
        s = s.split("_multiclient_")[0]
    return s


def _assert_non_empty_dir(expected_data_path: str):
    if not os.path.isdir(expected_data_path):
        raise FileNotFoundError(
            f"未找到合成数据目录：{expected_data_path}\n"
            f"请先运行 generate.py 生成数据（输出到 args.output_dir/generated_images）。"
        )
    for root, _, files in os.walk(expected_data_path):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                return
    raise FileNotFoundError(
        f"合成数据目录为空：{expected_data_path}\n"
        f"请先运行 generate.py 生成图片后再训练分类器。"
    )


class GeneratedImagesDataset(Dataset):
    """
    Reads generated images saved by generate.py under:
      {output_dir}/generated_images/{client_tag}/{domain}/{test_type}/{class_name}/*.jpg
    """

    def __init__(self, root_dir: str, categories, domains, test_type: Optional[str] = None):
        self.root_dir = root_dir
        self.categories = list(categories)
        self.domains = list(domains)
        self.test_type = test_type
        self._cat2id = {c: i for i, c in enumerate(self.categories)}
        self._dom2id = {d: i for i, d in enumerate(self.domains)}

        self.samples = []
        for r, _, files in os.walk(self.root_dir):
            for fn in files:
                if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                p = os.path.join(r, fn)
                rel = os.path.relpath(p, self.root_dir)
                parts = rel.split(os.sep)
                # Expected: [client_tag, domain, test_type, class, filename]
                if len(parts) < 5:
                    continue
                domain = parts[1]
                tt = parts[2]
                class_name = parts[3]
                if self.test_type is not None and tt != self.test_type:
                    continue
                if domain not in self._dom2id:
                    continue
                if class_name not in self._cat2id:
                    continue
                self.samples.append((p, self._dom2id[domain], self._cat2id[class_name]))

        if len(self.samples) == 0:
            hint = f"(test_type={self.test_type})" if self.test_type is not None else ""
            raise FileNotFoundError(
                f"在合成数据目录中未找到可用图片：{self.root_dir} {hint}\n"
                f"请确认 generate.py 的 output_dir/test_type/domain 与当前参数一致。"
            )

        base_transforms = [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        ]
        if args.random_flip:
            base_transforms.append(transforms.RandomHorizontalFlip())
        base_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, domain_id, class_id = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, domain_id, class_id


def collate_fn_images_only(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    domain_ids = torch.tensor([example[1] for example in examples])
    class_ids = torch.tensor([example[2] for example in examples])
    return {
        "pixel_values": pixel_values,
        "domain_ids": domain_ids,
        "class_ids": class_ids,
    }


def train(seed, train_setting):
    model = models.resnet18(pretrained=args.pretrained)
    print(f"Training with seed {seed} and setting {train_setting}")
    setup_seed(seed)
    categories = args.categories

    num_shots = int(train_setting.split("_")[-1])
    if 'multiclient' in train_setting:
        expected_data_path = os.path.join(args.output_dir, "generated_images")
        print(f"Loading training data from: {expected_data_path}")
        _assert_non_empty_dir(expected_data_path)
        # Use args.test_type if provided; otherwise infer from train_setting.
        expected_test_type = args.test_type if args.test_type else _infer_test_type_from_train_setting(train_setting)
        train_dataset = GeneratedImagesDataset(
            root_dir=expected_data_path,
            categories=args.categories,
            domains=args.domains,
            test_type=expected_test_type,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=collate_fn_images_only,
            shuffle=True,
            num_workers=args.num_workers,
        )
    elif 'fgl' in train_setting:
        num_shots = -1 
        train_dataloader = get_dataloader_domain(
            args, args.train_batch_size, None,
            train_setting, 'fgl', tokenize_captions,  
            collate_fn, num_shot=args.num_shot, num_workers=args.num_workers)
    else:
        # If this training setting points to synthetic data, load from generated_images.
        if "syn" in train_setting:
            expected_data_path = os.path.join(args.output_dir, "generated_images")
            print(f"Loading training data from: {expected_data_path}")
            _assert_non_empty_dir(expected_data_path)
            expected_test_type = args.test_type if args.test_type else _infer_test_type_from_train_setting(train_setting)
            train_dataset = GeneratedImagesDataset(
                root_dir=expected_data_path,
                categories=args.categories,
                domains=args.domains,
                test_type=expected_test_type,
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                collate_fn=collate_fn_images_only,
                shuffle=True,
                num_workers=args.num_workers,
            )
        else:
            train_dataloader = get_dataloader(
                args, args.train_batch_size, None,
                train_setting, tokenize_captions,
                collate_fn, num_shot=num_shots, num_workers=args.num_workers)

    test_dataloader = None
    if not args.skip_evaluation:
        if args.dataset=='pacs':
            num_shot_test = 32
        elif args.dataset=='ucm':
            num_shot_test = 8
        else:
            num_shot_test = -1
        test_dataloader = get_dataloader(
                args, args.train_batch_size, None,
                'test', tokenize_captions,  
                collate_fn, num_shot=num_shot_test, num_workers=args.num_workers)  
    
    num_epochs = int(getattr(args, "num_epochs", 50))
    num_classes = len(categories)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    
    num_steps_per_epoch = len(train_dataloader)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     num_steps_per_epoch * num_epochs,
    #     eta_min=0.001)

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = model(batch['pixel_values'].to(device))
            labels = batch['class_ids'].to(device)
            # Compute loss and perform backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()

        if not args.skip_evaluation and test_dataloader is not None and ((epoch+1) % 5 == 0 or epoch == num_epochs - 1):
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                d_count = defaultdict(list)

                for batch in test_dataloader:
                    inputs = batch["pixel_values"].to(device)
                    outputs = model(inputs)
                    labels = batch['class_ids'].to(device)
                    preds = torch.argmax(outputs, dim=1)
                    # Compute loss and accuracy
                    total_correct += preds.eq(labels).sum().item()
                    total_samples += inputs.size(0)
                    for i, did in enumerate(batch['domain_ids']):
                        d_count[did.item()].append(labels[i].item() == preds[i].item())

            tot_acc = 0.
            for k in d_count.keys():
                acc = sum(d_count[k]) / len(d_count[k]) * 100 
                tot_acc += acc
                print(f"{args.domains[k]}: {round(acc, 3)}", end=", ")

            if args.dataset in ['ucm', 'dermamnist']:
                print(f"Epoch {epoch+1}: Accuracy = {round(total_correct/total_samples, 3)}")
            else:
                print(f"{train_setting}/Epoch {epoch+1}: Accuracy = {round(tot_acc/len(d_count.keys()), 3)}")

if __name__=="__main__":    
    for seed in [0, 1, 2]:
        if not isinstance(args.train_type, list):
            args.train_type = [args.train_type]
        for train_setting in args.train_type:
            print(train_setting)
            train(seed, train_setting)