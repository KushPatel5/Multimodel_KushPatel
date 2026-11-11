# Multimodal Movie Genre Classification using MM-IMDb

**University of Adelaide | Master of Data Science | Assignment 2.2**  
**Student:** Kush Patel

## Project overview

This project investigates whether combining movie poster images with plot summaries improves multi-label movie genre classification compared with using either modality alone. It uses the MM-IMDb dataset, a frozen EfficientNet-B0 image encoder and a frozen BERT text encoder. Six classification approaches are compared:

- Image-only baseline
- Text-only baseline
- Early fusion
- Late fusion
- Simplified gated fusion
- Multi-head attention fusion

## Data and preprocessing

The MM-IMDb archive contains poster images and movie metadata. The notebook checks metadata completeness, image readability and dimensions, genre frequencies, and plot lengths. Posters are resized and normalised for EfficientNet-B0. Plot summaries are tokenised with BERT and limited to 160 tokens; the notebook reports the proportion affected by truncation.

Image and text features are extracted using pretrained, frozen encoders. Feature arrays may be cached to avoid repeated extraction. Feature standardisation is fitted using training data only.

**Dataset split:** The notebook uses a verified `mmimdb_official_splits.csv` file if one is supplied. Otherwise, it uses a seeded random 60/10/30 train/validation/test split. The random split is **not** the official MM-IMDb split; the choice must be reported with the experimental results. Validation data is used for model selection and thresholds; test data is reserved for final evaluation.

## Running the notebook

1. Open `MMIMDb_Assignment2_2_KushPatel_Style.ipynb` in Google Colab (GPU runtime recommended).
2. Install dependencies if needed: `pip install -r requirements.txt` (or install the listed packages in Colab).
3. Download the MM-IMDb archive from https://archive.org/download/mmimdb/mmimdb.tar.gz. Place it at `/content/mmimdb.tar.gz`, **or** restore the existing backup at `/content/drive/MyDrive/MMIMDb_Backup/mmimdb.tar.gz` using the notebook's Drive setup cell.
4. Optionally restore `/content/drive/MyDrive/MMIMDb_Backup/feature_cache` to `/content/feature_cache`.
5. If using an independently verified official split, provide `/content/mmimdb_official_splits.csv` with columns `id,split` and split values `train`, `val`/`validation`, and `test`.
6. Run the notebook from top to bottom. Feature extraction and training can take substantial time. GPU type, package versions, and cached features may affect execution time and small numerical differences.

The dataset and large model checkpoints are not included in this repository.

## Evaluation

The notebook reports micro-, macro- and weighted-F1, precision, recall, average precision, genre-level F1, training/validation behaviour and efficiency measures. It also inspects individual prediction errors and exploratory subgroup performance where metadata is available.

In the executed experiment used for the accompanying report, **late fusion had the highest test micro-F1 (0.6499)**, compared with **0.6265 for text-only** and **0.4230 for image-only**. These are results from that run, not guaranteed values for a fresh run or a different dataset split. Refer to `results/test_metrics.csv` for the actual results exported by your execution.

## Output files

The notebook writes outputs to `/content/mmimdb_final_outputs/`, including (depending on the completed cells):

- `test_metrics.csv`, `per_genre_f1.csv`, `overfitting_gaps.csv`
- `experiment_config.json`
- Supplementary CSV tables for image quality, truncation, failure cases, subgroup analysis and efficiency
- `figures/` containing PNG charts
- Trained classifier checkpoints, which may be too large for ordinary GitHub uploads

Copy the results folder to Google Drive if you need to retain it after the Colab session ends. Upload small CSV, JSON and PNG files to GitHub under `results/`. Do not commit the 8.69 GB dataset archive, extracted images, feature caches, secrets, or oversized checkpoints.

## Reproducibility notes

The notebook sets a random seed, documents experiment configuration and exports evaluation results. Exact reproducibility can still depend on GPU operations, dependency versions and whether the same dataset split is used. The attention and gated-fusion models are experimental comparisons; the gate is a simplified implementation, not a claim to reproduce a published GMU architecture.

## Dataset and model acknowledgements

- MM-IMDb: Arevalo et al., *Gated Multimodal Units for Information Fusion*, 2017.
- BERT: Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2019.
- EfficientNet: Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, 2019.

See the accompanying assignment report for the complete reference list and discussion of limitations and ethics.
