# Data Directory

Mostly gitignored. Populated by the pipeline (run `notebooks/kaggle_main.ipynb`):

| Path | Populated by | Contents |
|---|---|---|
| `raw/` | symlink to `/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset` | Reports CSV + 261K JPGs |
| `processed/reports.parquet` | `scripts/download_data.py` | 2,835 cleaned reports + CheXpert labels + image paths |
| `processed/splits.json` | `scripts/download_data.py` | patient-level train/val/test (2268 / 283 / 284) |
| `qa/qa_v1.jsonl` | `scripts/build_qa_dataset.py` | 6,216 validated train QA pairs |
| `qa/qa_test.jsonl` | `scripts/build_qa_dataset.py` | 811 held-out test QA pairs |
| `indices/biomedclip/` | `scripts/build_indices.py --backend biomedclip` | FAISS-IP image index |
| `indices/colpali_zs/` | `scripts/build_indices.py --backend colpali_zs` | ColPali multi-vector index (using patched v1.3 adapter) |
| `samples/` | `scripts/download_data.py` | 5 sample images for the demo |

## Kaggle subset

Source: https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset.
Schema is asserted on load by `src/cxr_intel/data/kaggle_loader.py`.

## License

MIMIC-CXR is governed by the [PhysioNet credentialed-use license](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). The Kaggle subset is a derivative; downstream users must comply with the original license and not redistribute raw images.
