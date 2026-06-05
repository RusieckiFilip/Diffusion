"""
train.py
--------
Contains:
  • Noise scheduler (forward process)
  • Loss function
  • sample_timestep / sample_plot_image  (backward / sampling)
  • Trainer class that wraps everything

Follows the tutorial structure closely so you can compare line-by-line.
"""

import os
import math
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# =========================================================================== #
# Noise Scheduler  —  Forward Process                                         #
# =========================================================================== #

def linear_beta_schedule(timesteps: int,
                          start: float = 0.0001,
                          end:   float = 0.02) -> torch.Tensor:
    """Linear variance schedule from Ho et al. 2020."""
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine variance schedule from Nichol & Dhariwal 2021.
    Often converges faster on small datasets.
    """
    steps  = timesteps + 1
    x      = torch.linspace(0, timesteps, steps)
    alphas = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas = alphas / alphas[0]
    betas  = 1 - (alphas[1:] / alphas[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def precompute_schedule(betas: torch.Tensor) -> dict:
    """
    Pre-compute all derived quantities needed for the closed-form
    forward process and the denoising step.
    """
    alphas              = 1.0 - betas
    alphas_cumprod      = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    return dict(
        betas                        = betas,
        alphas                       = alphas,
        alphas_cumprod               = alphas_cumprod,
        alphas_cumprod_prev          = alphas_cumprod_prev,
        sqrt_recip_alphas            = torch.sqrt(1.0 / alphas),
        sqrt_alphas_cumprod          = torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod= torch.sqrt(1.0 - alphas_cumprod),
        posterior_variance           = betas * (1.0 - alphas_cumprod_prev)
                                             / (1.0 - alphas_cumprod),
    )


def get_index_from_list(vals: torch.Tensor,
                        t:    torch.Tensor,
                        x_shape: tuple) -> torch.Tensor:
    """
    Gather a per-sample value from *vals* (shape [T]) using the batch
    of timestep indices *t* (shape [B]), and reshape to broadcast
    over the spatial dimensions of *x_shape*.
    vals is moved to t's device so everything stays on GPU.
    """
    batch_size = t.shape[0]
    out = vals.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def forward_diffusion_sample(
    x_0:      torch.Tensor,
    t:        torch.Tensor,
    schedule: dict,
    device:   str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form forward process:
        x_t = sqrt(ᾱ_t) · x_0  +  sqrt(1 - ᾱ_t) · ε,   ε ~ N(0, I)

    Returns (x_t, ε).
    """
    noise = torch.randn_like(x_0)
    sqrt_ac_t = get_index_from_list(
        schedule["sqrt_alphas_cumprod"], t, x_0.shape
    )
    sqrt_omc_t = get_index_from_list(
        schedule["sqrt_one_minus_alphas_cumprod"], t, x_0.shape
    )
    x_t = sqrt_ac_t.to(device) * x_0.to(device) \
        + sqrt_omc_t.to(device) * noise.to(device)
    return x_t, noise.to(device)


# =========================================================================== #
# Loss                                                                         #
# =========================================================================== #

def get_loss(model, x_0, t, schedule, device):
    """Simple L1 loss between true noise and predicted noise."""
    x_noisy, noise = forward_diffusion_sample(x_0, t, schedule, device)
    noise_pred     = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


# =========================================================================== #
# Sampling  —  Backward Process                                                #
# =========================================================================== #

def show_tensor_image(image: torch.Tensor, ax=None):
    """
    Convert a (1|3, H, W) tensor in [-1, 1] to a numpy image and show it.
    If *ax* is given, render onto that matplotlib Axes.
    """
    reverse_transforms = [
        lambda t: (t + 1) / 2,          # [-1,1] → [0,1]
        lambda t: t.clamp(0, 1),
        lambda t: t.permute(1, 2, 0),   # CHW → HWC
        lambda t: t.numpy(),
    ]
    img = image.squeeze(0).cpu()
    for fn in reverse_transforms:
        img = fn(img)
    if ax is None:
        plt.imshow(img)
    else:
        ax.imshow(img)


@torch.no_grad()
def sample_timestep(
    model,
    x:        torch.Tensor,
    t:        torch.Tensor,
    schedule: dict,
) -> torch.Tensor:
    """
    One step of the reverse (denoising) process.

    Given x_t and t, predict x_{t-1} using:
        μ_θ = (1/√α_t) · (x_t  -  β_t / √(1-ᾱ_t) · ε_θ(x_t, t))
    then add posterior noise when t > 0.
    """
    betas_t = get_index_from_list(
        schedule["betas"], t, x.shape
    )
    sqrt_omc_t = get_index_from_list(
        schedule["sqrt_one_minus_alphas_cumprod"], t, x.shape
    )
    sqrt_ra_t = get_index_from_list(
        schedule["sqrt_recip_alphas"], t, x.shape
    )

    # Predicted mean
    model_mean = sqrt_ra_t * (
        x - betas_t * model(x, t) / sqrt_omc_t
    )

    posterior_variance_t = get_index_from_list(
        schedule["posterior_variance"], t, x.shape
    )

    if t[0].item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(
    model,
    schedule:   dict,
    T:          int,
    img_size:   int,
    device:     str,
    num_images: int = 10,
    save_path:  str | None = None,
) -> None:
    """
    Start from pure Gaussian noise and iteratively denoise, displaying
    *num_images* intermediate frames as a strip (left=noisy, right=clean).

    Key: do NOT clamp between steps — only clamp for display.
    Clamping during sampling destroys the signal in early training.
    """
    img = torch.randn((1, 3, img_size, img_size), device=device)
    model.eval()

    stepsize = max(T // num_images, 1)
    # Collect frames: we want evenly spaced t values from high→low
    capture_at = set(range(0, T, stepsize))
    capture_at.add(0)
    frames: list[tuple[int, torch.Tensor]] = []   # (t_value, image_tensor)

    for i in reversed(range(T)):
        t   = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t, schedule)
        # NO clamp here — let values drift naturally

        if i in capture_at:
            frames.append((i, img.clone()))

    # frames are collected high-t → low-t; reverse so left=noisy, right=clean
    frames = frames[:num_images]   # show at most num_images

    fig, axes = plt.subplots(1, len(frames), figsize=(2 * len(frames), 2.5))
    if len(frames) == 1:
        axes = [axes]

    for ax, (t_val, frame) in zip(axes, frames):
        # Only clamp here for display purposes
        display = frame.clamp(-1.0, 1.0)
        show_tensor_image(display, ax=ax)
        ax.axis("off")
        ax.set_title(f"t={t_val}", fontsize=7)

    plt.suptitle("Denoising trajectory (left = noisy, right = clean)", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        print(f"  [sample] Saved → {save_path}")
    plt.close()
    model.train()


# =========================================================================== #
# Trainer                                                                      #
# =========================================================================== #

class Trainer:
    """
    Encapsulates training state and the main loop.

    Usage:
        trainer = Trainer(model, dataloader, config)
        trainer.train()
    """

    def __init__(self, model, dataloader, config: dict):
        self.model      = model
        self.dataloader = dataloader
        self.config     = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[trainer] Using device: {self.device}")
        self.model.to(self.device)

        # Build noise schedule
        T         = config["diffusion"]["T"]
        schedule  = config["diffusion"]["schedule"]
        b_start   = config["diffusion"]["beta_start"]
        b_end     = config["diffusion"]["beta_end"]

        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T, b_start, b_end)

        self.T        = T
        self.schedule = precompute_schedule(betas)
        self.img_size = config["data"]["img_size"]

        # Optimiser + LR scheduler
        lr = config["training"]["lr"]
        self.optimizer = Adam(model.parameters(), lr=lr)
        epochs = config["training"]["epochs"]
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr / 10
        )

        # Output dirs
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        self.results_dir    = config["training"]["results_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir,    exist_ok=True)

        self.save_every   = config["training"]["save_every"]
        self.sample_every = config["training"]["sample_every"]
        self.batch_size   = config["data"]["batch_size"]

    # ----------------------------------------------------------------------- #
    def train(self):
        epochs = self.config["training"]["epochs"]
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(self.dataloader,
                        desc=f"Epoch {epoch+1:03d}/{epochs}",
                        leave=False)

            for step, batch in enumerate(pbar):
                self.optimizer.zero_grad()

                # batch[0] → images  (B, 3, H, W)
                x_0 = batch[0].to(self.device)
                B   = x_0.size(0)
                t   = torch.randint(0, self.T, (B,),
                                    device=self.device).long()

                loss = get_loss(self.model, x_0, t,
                                self.schedule, self.device)
                loss.backward()

                # Gradient clipping — helps on small datasets
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            self.lr_scheduler.step()

            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"avg_loss={avg_loss:.4f} | "
                  f"lr={self.lr_scheduler.get_last_lr()[0]:.6f}")

            # ---- Generate sample strip ------------------------------------ #
            if (epoch + 1) % self.sample_every == 0:
                save_path = os.path.join(
                    self.results_dir, f"sample_epoch_{epoch+1:04d}.png"
                )
                sample_plot_image(
                    model     = self.model,
                    schedule  = self.schedule,
                    T         = self.T,
                    img_size  = self.img_size,
                    device    = self.device,
                    num_images= self.config["sampling"]["num_images"],
                    save_path = save_path,
                )

            # ---- Save checkpoint ------------------------------------------ #
            if (epoch + 1) % self.save_every == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"ckpt_epoch_{epoch+1:04d}.pt"
                )
                torch.save({
                    "epoch":       epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "avg_loss":    avg_loss,
                }, ckpt_path)
                print(f"  [checkpoint] Saved → {ckpt_path}")

        print("Training complete.")

    # ----------------------------------------------------------------------- #
    def resume(self, checkpoint_path: str):
        """Load a checkpoint and resume training."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"]
        print(f"[trainer] Resumed from epoch {start_epoch} "
              f"(loss={ckpt['avg_loss']:.4f})")
        return start_epoch
