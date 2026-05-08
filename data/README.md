# Data Directory

This directory is mostly gitignored. Layout populated by the pipeline:

| Path | Populated by | Contents |
|---|---|---|
| `raw/` | `scripts/download_data.py` | unzipped Kaggle MIMIC-CXR subset (CSV + images) |
| `processed/reports.parquet` | `scripts/download_data.py` | cleaned reports + CheXpert labels + image paths |
| `processed/splits.json` | `scripts/download_data.py` | patient-level train/val/test study IDs |
| `qa/qa_v1.jsonl` | `scripts/build_qa_dataset.py` | validated training QA pairs |
| `qa/qa_test.jsonl` | `scripts/build_qa_dataset.py` | held-out test QA pairs |
| `qa/cache/` | LLM router | hashed prompt → response cache |
| `indices/biomedclip/` | `scripts/build_indices.py --backend biomedclip` | FAISS-IP image index |
| `indices/colpali_zs/` | `scripts/build_indices.py --backend colpali_zs` | multi-vector ColPali index |
| `indices/colpali_lora/` | `scripts/build_indices.py --backend colpali_lora` | LoRA-tuned ColPali index |
| `samples/` | `scripts/download_data.py` | 5 sample images committed to the repo for the Streamlit demo |

## Kaggle Subset

Source: https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset.
The schema is asserted on load by `kaggle_loader.py` — we expect at minimum a `text`
column with the radiology report.

## License

MIMIC-CXR is governed by the [PhysioNet credentialed-use license](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
The Kaggle subset is a derivative; downstream users must comply with the original
license and not redistribute raw images.
