import argparse
import torch
import numpy as np
import wandb
import torchvision.transforms as tf
from torchmetrics.image import KernelInceptionDistance, \
    LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FLD import FLD
from medmnist import RetinaMNIST

from TNBC_dataset import TNBCDataset
# Mammo
from Mammo_dataset import MammoDataset
from extract_features import center_crop_arr


def eval_kid(real_imgs, gen_imgs, device: str = 'cuda', shape=128):
    kid = KernelInceptionDistance(subset_size=len(gen_imgs) // 4, normalize=False).to(device)
    for real_img_path in tqdm(real_imgs):
        # Forward real sample through the same pre-processing
        real = Image.open(real_img_path).convert('RGB')
        real = np.array(center_crop_arr(real, shape))
        real = torch.from_numpy(real).permute(2, 0, 1).unsqueeze(0).to(device)
        kid.update(real, real=True)

    for gen_img_path in tqdm(gen_imgs):
        gen = torch.from_numpy(np.array(Image.open(gen_img_path))).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        kid.update(gen, real=False)

    kid_mean, kid_std = kid.compute()

    return kid_mean, kid_std


def eval_lpips_diversity(img_paths, device: str = 'cuda'):
    """
        Computes the mean and std LPIPS feature distance of generated images.
        The higher the score, the more diverse is the data.

        To reduce computational complexity, only a subset of images pairs is considered.
        Therefore, the list of images is randomly shuffled and pairs of images are drawn randomly from it.

    """

    lpips_score = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    to_tensor = tf.ToTensor()

    # img_paths = glob.glob(gen_path + "*.png")
    np.random.shuffle(img_paths)

    feature_dists = []

    for i in tqdm(range(0, len(img_paths) - 1)):
        img1 = to_tensor(Image.open(img_paths[i])).to(device).unsqueeze(0)
        img2 = to_tensor(Image.open(img_paths[i + 1])).to(device).unsqueeze(0)
        feature_dists.append(lpips_score(img1, img2).item())

    return np.mean(feature_dists), np.std(feature_dists)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--annotation-path", type=str, required=True)
    parser.add_argument('--gen', type=str, help='Path to generated images')
    parser.add_argument('--device_list', nargs='+', default='cuda:0', help='List of device(s) to use.')
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--label-column", type=str, default="finding_categories")
    parser.add_argument('--gg_extractor', type=str, default='dino', help='feature extractor to use')

    parser.add_argument('--ft', type=str, help='Path to feature extractor')

    args = parser.parse_args()

    exp_string_name = os.path.basename(os.path.normpath(args.gen))
    args.exp_string_name = exp_string_name
    wandb.init(
        # Set the project where this run will be logged
        project='GeCA-Eval',
        # Track hyperparameters and run metadata
        config=vars(args)
    )

    # TODO
    #real_images_dir = os.path.join(args.image_root, 'test')
    #real_imgs = [os.path.join(real_images_dir, filename) for filename in 
    #             os.listdir(real_images_dir) if filename.endswith('.png')]
    real_df = pd.read_csv(args.annotation_path)
    real_df = real_df[real_df['split'].str.lower() == 'test']
    real_imgs = [os.path.join(args.image_root, img_id) for img_id in real_df['image_id'].values]
    real_imgs = [p for p in real_imgs if os.path.exists(p)]


    synthpath = Path(args.gen) / f'val_syn_{args.fold}.csv'
    dataset_synth = pd.read_csv(synthpath, index_col=0)
    gen_imgs = [os.path.join(str(args.gen), filename) for filename in
                dataset_synth['filename'].values]

    if args.gg_extractor == 'dino':
        feature_extractor = DINOv2FeatureExtractor()
    elif args.gg_extractor == 'inception':
        feature_extractor = InceptionFeatureExtractor()
    elif args.gg_extractor == 'clip':
        feature_extractor = CLIPFeatureExtractor()

    dataset_real = MammoDataset(
        root=args.image_root, 
        transform=tf.Compose([tf.ToPILImage()]), 
        mode='training', 
        annotation_path=args.annotation_path,
        label_column=args.label_column)
    
    dataset_test = MammoDataset(
        root=args.image_root, 
        transform=tf.Compose([tf.ToPILImage()]), 
        mode='test', 
        annotation_path=args.annotation_path,
        label_column=args.label_column)

    train_feat = feature_extractor.get_features(dataset_real)
    test_feat = feature_extractor.get_features(dataset_test)
    gen_feat = feature_extractor.get_dir_features(
        args.gen,
        extension="png")

    generalization_gap = FLD(eval_feat="gap").compute_metric(train_feat, test_feat, gen_feat)
    print(f"GS Generalization Gap FLD: {generalization_gap:.3f}")

    wandb.log({'GG': generalization_gap},
              step=0)

    kid_mean, kid_std = eval_kid(real_imgs, gen_imgs, args.device_list[0], shape=args.image_size)


    wandb.log({'KID mean': kid_mean},
              step=0)
    wandb.log({'KID std': kid_std},
              step=0)

    print(f"KID mean: {kid_mean} std: {kid_std}")

    lpips_mean, lpips_std = eval_lpips_diversity(gen_imgs, args.device_list[0])

    wandb.log({'LPIPS mean': lpips_mean},
              step=0)
    wandb.log({'LPIPS std': lpips_std},
              step=0)

    print(f"LPIPS avg. dist {lpips_mean} std {lpips_std}")

    wandb.finish()
