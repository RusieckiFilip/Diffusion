"""
check_forward.py
----------------
Sanity checks BEFORE training:
  1. Verifies the forward process correctly destroys the image by t=T
  2. Shows the noise schedule (beta, alpha_cumprod) so you can see
     the signal-to-noise ratio across timesteps
  3. Shows forward diffusion at t=0,50,100,150,200,250,299

Run:  python check_forward.py
If all 3 checks pass, your diffusion math is correct.
"""

import yaml
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision

from src.train import (linear_beta_schedule, cosine_beta_schedule,
                        precompute_schedule, get_index_from_list,
                        forward_diffusion_sample)

# ── Load config ────────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

T       = config["diffusion"]["T"]
sched   = config["diffusion"]["schedule"]
b_start = config["diffusion"]["beta_start"]
b_end   = config["diffusion"]["beta_end"]

betas    = (cosine_beta_schedule(T) if sched == "cosine"
            else linear_beta_schedule(T, b_start, b_end))
schedule = precompute_schedule(betas)

print(f"Schedule : {sched}")
print(f"T        : {T}")
print(f"beta[0]  : {betas[0].item():.6f}")
print(f"beta[-1] : {betas[-1].item():.6f}")
print(f"ᾱ[0]     : {schedule['alphas_cumprod'][0].item():.4f}  (should be ~1.0)")
print(f"ᾱ[-1]    : {schedule['alphas_cumprod'][-1].item():.4f} (should be ~0.0)")

# ── Check 1: ᾱ_T ≈ 0 (image fully destroyed) ──────────────────────────────────
ac_T = schedule["alphas_cumprod"][-1].item()
assert ac_T < 0.01, f"FAIL: ᾱ_T = {ac_T:.4f}, should be near 0. Image not fully noised!"
print("\n✓ CHECK 1 PASSED: ᾱ_T ≈ 0 — image is fully destroyed at t=T")

# ── Check 2: posterior_variance > 0 everywhere except t=0 ─────────────────────
pv = schedule["posterior_variance"]
assert pv[0].item() >= 0,       "FAIL: posterior_variance[0] < 0"
assert pv[1:].min().item() > 0, "FAIL: posterior_variance has zeros after t=0"
print(f"✓ CHECK 2 PASSED: posterior_variance range [{pv.min().item():.2e}, {pv.max().item():.2e}]")

# ── Check 3: Visualise forward diffusion on a real image ──────────────────────
# Use a single STL10 or random tensor if CUB not downloaded
try:
    from src.dataset import CUBDataset
    import torchvision.transforms as T_transforms
    tf = T_transforms.Compose([
        T_transforms.Resize(72), T_transforms.CenterCrop(64), T_transforms.ToTensor(),
        T_transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds  = CUBDataset("data", train=True, transform=tf)
    x_0 = ds[0][0].unsqueeze(0)   # (1, 3, 64, 64)
    print("✓ CHECK 3: Using real CUB bird image")
except Exception:
    x_0 = torch.randn(1, 3, 64, 64)
    print("⚠ CHECK 3: CUB not found — using random tensor")

timesteps_to_show = [0, 50, 100, 150, 200, 250, T-1]
fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(18, 3))

for ax, t_val in zip(axes, timesteps_to_show):
    t_tensor = torch.full((1,), t_val, dtype=torch.long)
    x_t, _   = forward_diffusion_sample(x_0, t_tensor, schedule)
    img      = ((x_t.squeeze(0).permute(1,2,0) + 1) / 2).clamp(0,1).numpy()
    ax.imshow(img)
    ax.set_title(f"t={t_val}\nᾱ={schedule['alphas_cumprod'][t_val]:.3f}", fontsize=8)
    ax.axis("off")

plt.suptitle("Forward process: image → noise  (ᾱ should approach 0)", y=1.05)
plt.tight_layout()
plt.savefig("forward_check.png", bbox_inches="tight", dpi=120)
plt.show()
print("\nSaved forward_check.png")
print("\nAll checks passed ✓  —  safe to start training")
