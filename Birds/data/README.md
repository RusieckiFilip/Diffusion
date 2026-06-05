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


