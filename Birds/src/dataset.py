"""
dataset.py
----------
Loads the CUB-200-2011 (Caltech-UCSD Birds) dataset.

CUB-200-2011 facts:
  - 11,788 images, 200 bird species
  - ~5,994 train / ~5,794 test images
  - Images are variable size, subjects are well-cropped and centred
  - Must be downloaded manually (license restriction):

    1. Go to: https://www.vision.caltech.edu/datasets/cub_200_2011/
    2. Download: CUB_200_2011.tgz  (~1.1 GB)
    3. Extract into your data_dir so the layout looks like:
         data/
           CUB_200_2011/
             images/
               001.Black_footed_Albatross/
               002.Laysan_Albatross/
               ...
             train_test_split.txt
             images.txt
             classes.txt

We use the official train/test split from train_test_split.txt.
All 200 species are birds, so no filtering is needed.

With ~6k training images and strong augmentation this is a great dataset
for a from-scratch diffusion model — comparable to StanfordCars in size
but with better-cropped, same-category images.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import torchvision.transforms as transforms


# --------------------------------------------------------------------------- #
# CUB Dataset class                                                            #
# --------------------------------------------------------------------------- #

class CUBDataset(Dataset):
    """
    PyTorch Dataset for CUB-200-2011.

    Reads the official train/test split file so results are reproducible
    and comparable with published papers.

    Parameters
    ----------
    root      : Path that contains the 'CUB_200_2011/' folder.
    train     : If True, load training images; else load test images.
    transform : torchvision transform pipeline.
    """

    def __init__(self, root: str, train: bool = True, transform=None):
        self.root      = Path(root) / "CUB_200_2011"
        self.transform = transform
        self.train     = train

        self._check_exists()

        # Parse image paths
        images_file = self.root / "images.txt"
        split_file  = self.root / "train_test_split.txt"
        labels_file = self.root / "image_class_labels.txt"

        with open(images_file)  as f: image_lines  = f.readlines()
        with open(split_file)   as f: split_lines  = f.readlines()
        with open(labels_file)  as f: labels_lines = f.readlines()

        # is_training_image: 1 = train, 0 = test
        split_map = {
            int(l.split()[0]): int(l.split()[1])
            for l in split_lines
        }
        label_map = {
            int(l.split()[0]): int(l.split()[1]) - 1   # 0-indexed
            for l in labels_lines
        }

        self.samples: list[tuple[Path, int]] = []
        for line in image_lines:
            img_id, rel_path = line.strip().split(maxsplit=1)
            img_id = int(img_id)
            is_train = split_map[img_id] == 1
            if is_train == self.train:
                full_path = self.root / "images" / rel_path
                self.samples.append((full_path, label_map[img_id]))

    def _check_exists(self):
        images_dir = self.root / "images"
        if not images_dir.exists():
            raise FileNotFoundError(
                f"\n\nCUB-200-2011 dataset not found at: {self.root}\n\n"
                "Please download it manually:\n"
                "  1. Visit https://www.vision.caltech.edu/datasets/cub_200_2011/\n"
                "  2. Download CUB_200_2011.tgz (~1.1 GB)\n"
                "  3. Extract so the layout is:\n"
                f"       {self.root.parent}/\n"
                "         CUB_200_2011/\n"
                "           images/\n"
                "           train_test_split.txt\n"
                "           images.txt\n"
                "           classes.txt\n"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# --------------------------------------------------------------------------- #
# DataLoader factory                                                           #
# --------------------------------------------------------------------------- #

def get_dataloader(
    data_dir:    str  = "./data",
    img_size:    int  = 64,
    batch_size:  int  = 32,
    num_workers: int  = 4,
    use_both_splits: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for CUB-200-2011 bird images.

    Parameters
    ----------
    data_dir         : Directory that contains the CUB_200_2011/ folder.
    img_size         : Square size to resize images to.
    batch_size       : Mini-batch size.
    num_workers      : DataLoader worker threads.
    use_both_splits  : If True (default), merge train+test for ~11.7k images.
                       If False, use only the train split (~5.9k images).

    Returns
    -------
    DataLoader  (images in [-1, 1], shape B×3×img_size×img_size)
    """

    # ------------------------------------------------------------------ #
    # Transforms — aggressive augmentation to compensate for dataset size #
    # ------------------------------------------------------------------ #
    train_transform = transforms.Compose([
        # CUB images vary in size; centre-crop first to preserve subject
        transforms.Resize(int(img_size * 1.15)),          # slight oversize
        transforms.CenterCrop(img_size),                   # tight crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],        # → [-1, 1]
                             std= [0.5, 0.5, 0.5]),
    ])

    # ------------------------------------------------------------------ #
    # Build dataset(s)                                                     #
    # ------------------------------------------------------------------ #
    train_ds = CUBDataset(data_dir, train=True,  transform=train_transform)

    if use_both_splits:
        test_ds  = CUBDataset(data_dir, train=False, transform=train_transform)
        dataset  = ConcatDataset([train_ds, test_ds])
        n_train, n_test = len(train_ds), len(test_ds)
        print(f"[dataset] CUB-200-2011: {len(dataset)} images "
              f"(train={n_train}, test={n_test})")
    else:
        dataset = train_ds
        print(f"[dataset] CUB-200-2011 train split: {len(dataset)} images")

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        persistent_workers = True,
        drop_last   = True,
    )


# --------------------------------------------------------------------------- #
# Quick visual check:  python -m src.dataset                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    loader = get_dataloader(batch_size=16, num_workers=0)
    imgs, labels = next(iter(loader))
    imgs = (imgs + 1) / 2                          # [-1,1] → [0,1]

    grid = vutils.make_grid(imgs, nrow=4, padding=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(f"CUB-200-2011 sample batch  (n={len(imgs)})")
    plt.tight_layout()
    plt.savefig("sample_cub_birds.png", dpi=100)
    plt.show()
    print("Saved sample_cub_birds.png")
