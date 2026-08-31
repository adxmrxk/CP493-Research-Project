# CP493: Long Water Streak Deraining Benchmark (Replication Package)

Code to reproduce the CP493 project: a 500-image benchmark testing whether pretrained restoration
models remove **long surface water streaks** (long vertical water trails on a camera lens) with **no
fine-tuning**. Five models (RCDNet, AT-GAN, Restormer, IDT, DiT) + two cascade orders + a baseline.

This work builds on the **Raindrop Clarity** paper (Jin et al., ECCV 2024). The dataset and most of the
model weights come from that project's GitHub.

**Author:** Adam · rakx0280@mylaurier.ca / adamrak239@gmail.com · CP493, Wilfrid Laurier University

**Supervisor:** Asif Khan, Wilfrid Laurier University

## Demo video

Video Link: https://youtu.be/_qhx8V6BSnc 

---

## Dataset & model weights

Everything you need to download comes from the **Raindrop Clarity** repo, except the RCDNet weight:

- **Dataset**: `DayRainDrop_Train.zip` (daytime split) from https://github.com/jinyeying/RaindropClarity
  ("Day Training Data"). The exact 500 benchmark images are listed in
  `Stage 1/selected_500_images_FINAL.txt`.
- **DiT, AT-GAN, Restormer, IDT weights**: pretrained checkpoints from the same Raindrop Clarity
  release (DiT and AT-GAN are their models; Restormer and IDT are their baseline checkpoints).
- **RCDNet weight**: from the RCDNet repo instead: https://github.com/hongwang01/RCDNet
  (`spa_model_best.pt`, trained on SPA-Data).
- **YOLOv8** (downstream detection): auto-downloads from `ultralytics` (COCO-pretrained).

**Environment:** Python 3 with `opencv-python numpy torch lpips ultralytics`. Model inference runs on a
GPU (project used a Kaggle NVIDIA T4); the analysis scripts run on any machine.

---

## Running the models (main step)

This is the core of the experiment. Each notebook in `Stage 2/notebooks/` runs **one configuration** on
the 500 images, and **scores it right in the notebook** with `Stage 2/scripts/score_pairs.py`, so the
per-image PSNR / SSIM / LPIPS and the mean scores come straight out of the cells, along with the
resource cost (`model_stats.py`).

### What to attach to every run

Attach these to any Kaggle run (GPU T4, Internet on):

- **Dataset:** the 500 Drop + Clear image pairs
- **RaindropClarity repo code** plus the scoring scripts (`score_pairs.py`, `model_stats.py`)
- **The model's weights:** DiT, AT-GAN, Restormer, IDT come from Raindrop Clarity; RCDNet from its own
  package; cascades need both

**Note:** RCDNet runs on its own code, but still attach the RaindropClarity repo, since the scorer
imports `utils.metrics` from it. So the repo is needed for every run.

Run these notebooks on a GPU:

| Notebook                                      | Configuration           |
| --------------------------------------------- | ----------------------- |
| `rcdnet_alone_500.ipynb`                      | RCDNet (CNN)            |
| `atgan_500.ipynb`                             | AT-GAN (GAN)            |
| `restormer_500.ipynb` / `restormer_128.ipynb` | Restormer (transformer) |
| `IDT_500.ipynb` / `IDT_128.ipynb`             | IDT (transformer)       |
| `dit_alone_500.ipynb`                         | DiT (diffusion)         |
| `rcdnet_then_dit_500.ipynb`                   | RCDNet → DiT cascade    |
| `dit_then_rcdnet_500.ipynb`                   | DiT → RCDNet cascade    |

Each run outputs the restored images plus a per-image score CSV and a summary. Those CSVs are the input
to the analysis stages below.

---

## Building the dataset

`Stage 1/`: the semi-automated detector that selects the long-water-streak images, then manual
verification. Reconstructs the exact 500-image benchmark.

```
python "Stage 1/find_long_water_streaks_features.py" --csv_path features_full.csv
python "Stage 1/build_500_dataset.py"
```

`selected_500_images_FINAL.txt` lists the final 500 image IDs.

## Statistical analysis

`Stage 3/run_stage3_stats.py`: takes the per-image score CSVs from the model runs and computes the
paired **significance tests**: a Friedman omnibus, Holm-corrected Wilcoxon post-hoc comparisons, effect
sizes, and bootstrap 95% confidence intervals. (The scores themselves come from the model notebooks;
this step does the cross-model significance testing on top of them.)

```
python "Stage 3/run_stage3_stats.py"
```

## Geometry-aware selector

`Stage 4/run_stage4_selector.py`: a decision-tree / random-forest selector that reuses the detector's
streak-geometry features to pick the best configuration per image, compared against a fixed-best model
and an oracle.

```
python "Stage 4/run_stage4_selector.py"
```

## Downstream object detection

`Stage 5/Stage5_Detection.ipynb`: runs a COCO-pretrained YOLOv8 detector on six image versions, framed
as a false-positive sensitivity test linking restoration quality to perception errors.

---

## Troubleshooting & tips

- **DiT is slow, so batch it.** The feed-forward models (RCDNet, AT-GAN, Restormer, IDT) restore an
  image in under 4 seconds and run the full 500 in minutes, but **DiT is ~64 seconds per image** (~9 h
  for 500), which hits Kaggle's time limit and made long single runs glitch and reprocess images. Split
  the 500 into **thirds (~167 images, ≈3 h each)** or **fourths (125 images, ≈2.2 h each)**, run them
  separately, and merge the per-image score CSVs locally. Only needed for the three configs that include
  DiT: `dit_alone`, `rcdnet_then_dit`, and `dit_then_rcdnet`; the others run whole.

**Reading the Kaggle console log.** After running the notebook, check the output for these lines:

Good run, you should see:

- `PLUMBING OK`: the pre-run smoke test passed
- `sid=''`: processing all images, not stuck on one folder
- `input_res 256` (or `input_res 128` for the native IDT/Restormer runs): correct protocol
- `images: N`, matches the count you expect for that batch
- image IDs advancing in order with no repeats, e.g. `00002 → 00003 → 00005 …`
- exactly **one** `DIT_EVAL_PID` line (for DiT runs)
- a final score CSV with one row per image (the expected row count)

Bad run, stop and restart the batch if you see:

- image IDs **repeating** or already-processed images being redone
- a **second** `Namespace(...)` start-up printed partway through the run
- **two** `DIT_EVAL_PID` lines (the multi-GPU duplication; the run is doubling)
- a score CSV with roughly **twice** the expected rows
- `sid=00199` instead of `sid=''` (stuck on a single folder)

---

Only the code needed to re-run the experiments is included. The dataset images and generated outputs are
not shipped; they are produced by downloading the dataset/weights above and running the stages.
