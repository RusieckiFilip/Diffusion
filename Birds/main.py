"""
main.py
-------
Entry point for the CUB-200-2011 bird diffusion model.

Usage:
    python main.py                                         # train from scratch
    python main.py --resume checkpoints/ckpt_epoch_0050.pt
    python main.py --sample-only checkpoints/ckpt_epoch_0500.pt
    python main.py --config configs/config.yaml            # default

The script reads configs/config.yaml, builds the model and dataloader,
then hands everything to the Trainer.

NOTE: CUB-200-2011 must be downloaded manually first.
      See src/dataset.py or the README for instructions.
"""

import argparse
import yaml
import torch

from src.dataset import get_dataloader
from src.model   import SimpleUnet
from src.train   import (Trainer, sample_plot_image, precompute_schedule,
                          linear_beta_schedule, cosine_beta_schedule)


def parse_args():
    p = argparse.ArgumentParser(
        description="Bird Diffusion Model — CUB-200-2011"
    )
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--resume",      default=None,
                   help="Checkpoint path to resume training from")
    p.add_argument("--sample-only", default=None, dest="sample_only",
                   help="Load checkpoint and only generate a sample image")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    config = load_config(args.config)

    # ---- Build model --------------------------------------------------------
    mc = config["model"]
    model = SimpleUnet(
        image_channels = mc["image_channels"],
        down_channels  = tuple(mc["down_channels"]),
        up_channels    = tuple(mc["up_channels"]),
        time_emb_dim   = mc["time_emb_dim"],
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"SimpleUnet  |  parameters: {n:,}")

    # ---- Sample-only mode ---------------------------------------------------
    if args.sample_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt   = torch.load(args.sample_only, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()

        dc = config["diffusion"]
        T  = dc["T"]
        betas = (cosine_beta_schedule(T) if dc["schedule"] == "cosine"
                 else linear_beta_schedule(T, dc["beta_start"], dc["beta_end"]))
        schedule = precompute_schedule(betas)

        print(f"Loaded checkpoint — epoch {ckpt.get('epoch', '?')}")
        sample_plot_image(
            model      = model,
            schedule   = schedule,
            T          = T,
            img_size   = config["data"]["img_size"],
            device     = device,
            num_images = config["sampling"]["num_images"],
            save_path  = "sample_output.png",
        )
        return

    # ---- DataLoader ---------------------------------------------------------
    dc = config["data"]
    dataloader = get_dataloader(
        data_dir        = dc["data_dir"],
        img_size        = dc["img_size"],
        batch_size      = dc["batch_size"],
        num_workers     = dc["num_workers"],
        use_both_splits = dc.get("use_both_splits", True),
    )

    # ---- Train --------------------------------------------------------------
    trainer = Trainer(model, dataloader, config)
    if args.resume:
        trainer.resume(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
