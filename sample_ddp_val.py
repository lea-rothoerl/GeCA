# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
from pathlib import Path

from medmnist import RetinaMNIST
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.distributed as dist

from extract_features import center_crop_arr
from models import DiT_models
#from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import pandas as pd

from torchvision import transforms
import torch.nn.functional as F

# Mammo
from Mammo_dataset import MammoDataset

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")

    return npz_path

class CustomDataset(Dataset):
    def __init__(self, labels_dir, annot, label_column, mode='train', fold=0, multi_class=False):

        self.datapath = Path(labels_dir)

        self.mammo_dataset = MammoDataset(root=labels_dir, 
                                          annotation_path=annot, 
                                          mode=mode, 
                                          transform=None,
                                          label_column=label_column)
        
        self.label = self.mammo_dataset.get_mapped_labels()
        self.all_labels = self.mammo_dataset.all_labels

    def __getitem__(self, idx):
        label_file = self.label[idx] 
        label_array = np.array(label_file, dtype=np.float32)

        return 0, torch.from_numpy(label_array)

    def __len__(self):
        return len(self.label)

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [128, 256, 512]
        assert args.num_classes == 1000

    # Load model:
    if args.image_space:
        # Create model:
        latent_size = args.image_size
    else:
        # Create model:
        assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    in_channels = args.in_channels

    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    checkpt = torch.load(ckpt_path, map_location="cuda")
    state_dict = checkpt["model"]

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    if not args.image_space:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"

    folder_name = f"{model_string_name}-GS-fold-{args.fold}-nstep-{args.num_sampling_steps}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(f'{sample_folder_dir}/gen_samples_val', exist_ok=True)

        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()

    temp_dataset = MammoDataset(root=args.image_root, 
                                annotation_path=args.annotation_path, 
                                mode='training', 
                                transform=None,
                                label_column=args.label_column#'finding_categories'
                                )
    
    dataset = CustomDataset(args.image_root, 
                            args.annotation_path, 
                            mode='test', 
                            fold=args.fold,
                            label_column=args.label_column#'finding_categories'
                            )
    
    full_labels = temp_dataset.all_labels
    dataset.all_labels = temp_dataset.all_labels
    dataset.mammo_dataset.all_labels = temp_dataset.all_labels
    dataset.mammo_dataset.label_to_index = {label: idx for idx, label in enumerate(dataset.all_labels)}
    dataset.label = dataset.mammo_dataset.get_mapped_labels()

    loader = DataLoader(
        dataset,
        batch_size=int(args.per_proc_batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    total = 0
    annotations = None
    dict_pd = []

    def update_label(diction, labl):
        for name, label in zip(full_labels,
                               labl):
            diction[name] = int(label)

        return diction

    def get_new_labels(y):
        """ Convert each multilabel vector to a unique string """
        yy = [''.join(str(l)) for l in y]
        y_new = LabelEncoder().fit_transform(yy)
        return y_new

    for _ in range(args.expand_ratio):
        for _, labl in loader:
            labl = labl.to(device).float()
            labl = labl.squeeze(dim=1)  

            z = torch.randn(labl.size(0), model.in_channels, latent_size, latent_size, device=device)

            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.zeros_like(labl, device=device)
                y = torch.cat([labl, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=labl)
                sample_fn = model.forward

            if 'GeCA' in args.model:
                model_kwargs['extras'] = model.seed(z, [latent_size, latent_size])

            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            if not args.image_space:
                samples = vae.decode(samples / 0.18215).sample
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            else:
                samples = (samples + 1.0) * 0.5
                samples = (samples * 255).to(device, dtype=torch.uint8)

            label = labl.cpu().numpy()
            annotations = label if annotations is None else np.concatenate([annotations, label], axis=0)

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                dict_client_img = {}
                index = i * dist.get_world_size() + rank + total
                dict_client_img['client_id'] = 'syn_' + str(index)
                dict_client_img['filename'] = os.path.join('gen_samples_val', f"{index:06d}.png")
                dict_client_img = update_label(dict_client_img, label[i])  
                dict_client_img['patient_name'] = index
                dict_pd.append(dict_client_img)

                Image.fromarray(sample).save(f"{sample_folder_dir}/gen_samples_val/{index:06d}.png")
            total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        df = pd.DataFrame(dict_pd)
        multi_label_list = df[full_labels].values
        df['multi_class_label'] = get_new_labels(multi_label_list)
        df.to_csv(os.path.join(sample_folder_dir, f'val_syn_{args.fold}.csv'))

        np.save(os.path.join(f"{sample_folder_dir}/gen_samples_val_annotations.npz"), annotations)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--in_channels", type=int, default=4)

    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action='store_true', default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--annotation-path", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--label-column", type=str, default="finding_categories")
    parser.add_argument("--expand_ratio", type=int, default=1)
    parser.add_argument("--image_space", action='store_true', default=False)


    args = parser.parse_args()
    main(args)
