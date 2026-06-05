# 🐦 Bird Diffusion Model — CUB-200-2011

A from-scratch DDPM (Denoising Diffusion Probabilistic Model) that generates
bird images, trained on the **CUB-200-2011 (Caltech-UCSD Birds)** dataset.

Inspired on the YouTube tutorial series:
[A Diffusion Model from Scratch in PyTorch](https://www.youtube.com/watch?v=a4Yfz2FxXiY&list=PLV8yxwGOxvvoQRvx3wSoIrA8l_UwnMD3q)

---

## ⚠️ Dataset Setup (Required Before Training)

CUB-200-2011 cannot be downloaded automatically due to licensing.

1. **Download** `CUB_200_2011.tgz` (~1.1 GB) from:
   https://www.vision.caltech.edu/datasets/cub_200_2011/

2. **Extract** into the `data/` folder so the layout looks like:
   ```
   data/
     CUB_200_2011/
       images/
         001.Black_footed_Albatross/
         002.Laysan_Albatross/
         ...
       train_test_split.txt
       images.txt
       classes.txt
   ```

3. Run `python main.py` — training starts immediately.

---

## Project Structure

```
bird-diffusion/
├── data/               ← Put CUB_200_2011/ here
├── configs/
│   └── config.yaml     ← All hyperparameters
├── src/
│   ├── __init__.py
│   ├── dataset.py      ← CUB-200-2011 Dataset class + DataLoader
│   ├── model.py        ← U-Net with sinusoidal time embeddings
│   └── train.py        ← Noise scheduler, loss, sampling, Trainer
├── checkpoints/        ← Auto-created; checkpoints saved here
├── results/            ← Auto-created; sample images saved here
├── main.py             ← CLI entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download CUB-200-2011 manually (see above), then:
python main.py

# 3. Resume interrupted training
python main.py --resume checkpoints/ckpt_epoch_0050.pt

# 4. Generate samples from a saved checkpoint
python main.py --sample-only checkpoints/ckpt_epoch_0500.pt

# 5. Verify the dataset is loaded correctly (shows a grid of birds)
python -m src.dataset
```

---

## Why CUB-200-2011?

| Property | CUB-200-2011 | STL-10 birds | StanfordCars |
|---|---|---|---|
| Total images | **11,788** | ~1,300 | ~8,000 |
| All birds? | ✅ all 200 species | ✅ | ❌ |
| Resolution | 200–500 px | 96×96 | variable |
| Subjects centred | ✅ tight crop | ✅ | mostly |
| Download works | manual (1.1 GB) | ✅ auto | ❌ broken |

CUB is the dataset most generative model papers on birds actually use.

---

## Architecture

### Forward Process (Noise Scheduler)
Cosine beta schedule (default) with closed-form sampling:
```
x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε,   ε ~ N(0,I)
```

### Backward Process (U-Net)
5-level encoder–decoder with:
- **Sinusoidal time embeddings** (dim=256) injected at every block
- **GroupNorm** — stable at batch=1 during inference
- **SiLU** activations
- Skip connections via channel concatenation

### Loss
Simple L1 between true noise and predicted noise:
```python
loss = F.l1_loss(noise, model(x_noisy, t))
```

### Sampling
Standard DDPM ancestral sampling over T=300 steps.

---

## Configuration (`configs/config.yaml`)

| Key | Default | Notes |
|-----|---------|-------|
| `data.img_size` | 64 | Try 128 with ≥8 GB VRAM |
| `data.batch_size` | 32 | Reduce to 16 if OOM |
| `data.use_both_splits` | true | Merge train+test → 11,788 images |
| `diffusion.T` | 300 | Try 1000 for sharper results |
| `diffusion.schedule` | cosine | Faster convergence than linear |
| `training.epochs` | 500 | ~300 for rough birds, ~500 for clear |
| `training.lr` | 0.0002 | Cosine-annealed to lr/10 |

---

## Expected Training Behaviour

| Epoch | Typical L1 loss | What you see |
|-------|----------------|--------------|
| 10–50 | ~0.35 | Noise with colour blobs |
| 100 | ~0.22 | Rough silhouettes |
| 200 | ~0.16 | Bird shapes, some feather texture |
| 500 | ~0.11 | Recognisable birds with detail |

With 11k+ images the model converges meaningfully by epoch 200.
