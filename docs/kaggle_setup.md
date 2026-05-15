# Kaggle Notebook Setup — ColPali LoRA Training

We use Kaggle Notebooks (free P100, 30 h/week) for the heavy LoRA training step.
Everything else (data prep, QA synth, indexing, eval, Streamlit) runs on free Colab T4
or your laptop.

## One-time setup

### 1. Kaggle account + API token

1. Go to https://www.kaggle.com → sign up / log in.
2. Profile → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json`.
3. Keep it handy — you'll paste its values into `.env` and Kaggle Secrets.

### 2. Upload the preprocessed corpus as a Kaggle Dataset

The training notebook needs `data/processed/reports.parquet` and the CXR images. Easiest path:

```bash
# Locally, after running scripts/download_data.py on Colab
# Zip the artefacts you'll need on Kaggle:
cd c:/Users/youhe/OneDrive/Desktop/413assigment2
mkdir kaggle_payload
cp -r data/processed kaggle_payload/
# Add the images that are referenced by reports.parquet (only those that are needed)
cp -r data/raw kaggle_payload/        # OR a curated subset; raw can be big
```

Then on https://www.kaggle.com/datasets → **New Dataset**:
- Title: `mimic-cxr-processed`
- Upload the `kaggle_payload/` folder.
- Make it **Private**.
- Click **Create**.

### 3. Add Hugging Face token to Kaggle Secrets (optional)

If you ever load gated models on Kaggle (we don't strictly need it for ColPali, but
useful if you want to also run MedGemma there):

1. Open any Kaggle notebook → **Add-ons** → **Secrets**.
2. Add a secret named `HF_TOKEN` with the value from your `.env`.

## Running the training

1. https://www.kaggle.com/code → **New Notebook**.
2. Settings panel (right side):
   - **Accelerator** = `GPU P100` (free tier).
   - **Persistence** = `Files only` (so `/kaggle/working` survives between sessions).
   - **Internet** = `On` (needed to download ColPali base checkpoint).
3. **Add data** → search for *your* `mimic-cxr-processed` dataset → Add.
4. Paste the contents of `notebooks/03_colpali_lora_train_kaggle.ipynb` into the
   notebook, or **Upload** the .ipynb file directly.
5. In the data-linking cell, set `DATA_ROOT` to `/kaggle/input/<your-dataset-slug>`.
6. **Save Version** → **Save & Run All (Commit)**. ~3-4 hours.
7. When done, **Output** → **Download** `colpali-cxr-lora/`.
8. Unzip into your local repo at `models/colpali-cxr-lora/final/`, then commit and push.

## Tips

- Kaggle gives 30 GPU-hours/week, reset Sundays. The full training run uses ~4 hours.
- If a session hits the 9-hour wall, **Save Version** at any time mid-run; the
  training script writes checkpoints every 250 steps to `models/colpali-cxr-lora/`.
- The smoke-test cell (`--limit-train 50`) runs in ~5 min — always run it first.
- After training, **do not skip** running `build_indices.py --backend colpali_lora`
  back on Colab — the index must match the adapter you trained.
