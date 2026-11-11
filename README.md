# Multimodal Movie Genre Classification Using MM-IMDb

**University of Adelaide | Master of Data Science | Assignment 2.2**  
**Student:** Kush Patel  
**Student ID:** A3015439

## Project overview

This individual project investigates whether combining movie poster images and plot summaries improves **multi-label movie genre classification** compared with using either source alone. Experiments use the MM-IMDb dataset and compare six classification strategies under a common train, validation and test partition.

The research question is: **How do image-only, text-only and multimodal fusion methods compare for predicting movie genres from posters and plot summaries?**

## Repository contents

```text
Multimodel_KushPatel/
├── MMIMDb_Assignment2.2_a3015439.ipynb
├── README.md
├── requirements.txt
└── MMIMDb_Final_Results/
    ├── figures/
    │   ├── bert_token_lengths.png
    │   ├── genre_distribution.png
    │   ├── model_comparison.png
    │   ├── per_genre_f1.png
    │   ├── poster_dimensions.png
    │   ├── subgroup_micro_f1.png
    │   └── training_curves.png
    ├── attention_model.pt
    ├── early_model.pt
    ├── gated_model.pt
    ├── image_model.pt
    ├── text_model.pt
    ├── bert_truncation.csv
    ├── efficiency.csv
    ├── experiment_config.json
    ├── failure_cases.csv
    ├── overfitting_gaps.csv
    ├── per_genre_f1.csv
    ├── poster_quality.csv
    ├── split_assignments.csv
    ├── subgroup_metrics.csv
    └── test_metrics.csv
```

The five `.pt` files are trained **classifier** checkpoints. The much larger pretrained feature encoders, original MM-IMDb archive, extracted posters and feature caches are **not** included.

## Dataset and preprocessing

The experiment uses **25,945 retained movies** with **23 possible genre labels**. Each movie can have multiple genres. The notebook examines duplicate plot summaries, genre imbalance, poster dimensions and readability, and BERT token lengths.

- **Posters:** resized to 224 × 224 and normalized for the pretrained EfficientNet-B0 encoder.
- **Plot summaries:** tokenized with `bert-base-uncased`, using a maximum sequence length of 160 tokens. In the executed run, approximately **46.41%** of plots exceeded this token limit, so some information was truncated.
- **Image features:** 1,280-dimensional vectors from frozen EfficientNet-B0.
- **Text features:** 768-dimensional masked mean-pooled vectors from frozen BERT.
- **Feature scaling:** fitted on training data, then applied to validation and test data.

**Important split limitation:** The uploaded experiment used a **seeded random 60/10/30 split** (seed 42): 15,567 training, 2,594 validation and 7,784 test movies. This is **not the official MM-IMDb split**. The notebook supports an externally supplied, verified official split CSV, but one was not used for the reported results. Do not compare these scores directly with published official-split benchmarks.

## Methods

Six approaches are evaluated:

1. **Image-only:** classifier trained on EfficientNet-B0 features.
2. **Text-only:** classifier trained on BERT features.
3. **Early fusion:** concatenates image and text representations before classification.
4. **Gated fusion:** uses a simplified learned gate to combine modalities; it is **not** claimed to reproduce the published GMU architecture.
5. **Multi-head attention fusion:** uses attention to combine projected modality representations.
6. **Late fusion:** combines image-only and text-only predicted probabilities, with the image weight selected on the validation set.

The neural classifiers use PyTorch, binary cross-entropy with logits, AdamW and early stopping based on validation micro-F1. Model selection and threshold tuning use validation data; the test set is used for final evaluation.

## Experimental results

The following values come from `MMIMDb_Final_Results/test_metrics.csv` in the completed repository run:

| Model | Test micro-F1 | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|
| Late fusion | **0.6499** | **0.5906** | **0.6538** |
| Gated fusion | 0.6397 | 0.5901 | 0.6485 |
| Early fusion | 0.6388 | 0.5853 | 0.6452 |
| Text-only | 0.6265 | 0.5615 | 0.6359 |
| Attention fusion | 0.6214 | 0.5622 | 0.6300 |
| Image-only | 0.4230 | 0.3097 | 0.4406 |

Late fusion achieved the highest **test micro-F1** in this run. Its improvement over text-only was approximately **0.0234** absolute micro-F1. More complex attention fusion did not outperform the simpler fusion strategies. These findings apply to this experiment and its particular data split; they do not establish that late fusion is universally superior.

The results folder also includes per-genre F1, precision/recall, average precision, training-versus-validation gaps, individual prediction errors, subgroup comparisons and classifier-only inference timing. The timing measurements **exclude** poster decoding and pretrained feature extraction at inference time, so they should not be treated as end-to-end deployment latency.

## Reproducing the experiment

**Environment:** Google Colab with a GPU is recommended. The notebook uses Python, PyTorch, torchvision, Hugging Face Transformers, scikit-learn, pandas, NumPy, Pillow, Matplotlib and tqdm. See `requirements.txt` for compatible package ranges. Exact numerical reproducibility can depend on the runtime and installed package versions.

1. Open [`MMIMDb_Assignment2.2_a3015439.ipynb`](MMIMDb_Assignment2.2_a3015439.ipynb) in Google Colab.
2. Install missing packages using `requirements.txt` or the notebook's environment setup.
3. Obtain the MM-IMDb dataset archive from [Internet Archive](https://archive.org/download/mmimdb/mmimdb.tar.gz). Place it at `/content/mmimdb.tar.gz`, or restore the user's existing backup from `/content/drive/MyDrive/MMIMDb_Backup/mmimdb.tar.gz` if available. **The backup path is specific to the original Colab setup and is not accessible to other users.**
4. Run notebook cells from top to bottom. The notebook extracts the archive when needed and generates/caches features. GPU feature extraction may take time.
5. For the same seeded random-split experiment, leave `/content/mmimdb_official_splits.csv` absent. To run an independently verified official split instead, provide a CSV at that location with `id` and `split` columns (`train`, `val`/`validation`, `test`); this constitutes a **different experiment** and its scores must be reported separately.
6. Inspect the generated outputs at `/content/mmimdb_final_outputs/`. The notebook also creates `/content/MMIMDb_Final_Results.zip` after verifying required outputs. Copy outputs to persistent storage before the Colab runtime disconnects.

**Reproducibility caveat:** The notebook requires the external MM-IMDb archive and may need to download pretrained EfficientNet-B0 and BERT weights. A fresh end-to-end run on a separate environment has not been independently certified. The repository's saved results document the completed Colab experiment.

## Output interpretation and limitations

- **Class imbalance:** frequent genres may dominate micro-F1; macro-F1 and per-genre F1 provide complementary information.
- **Plot truncation:** a 160-token limit removes part of longer summaries.
- **Frozen encoders:** neither EfficientNet-B0 nor BERT was fine-tuned for genre classification.
- **Overfitting:** training and validation F1 gaps indicate that several classifiers generalize less well than their training performance suggests.
- **Subgroup analysis:** country, language and release-period results are exploratory. Unequal sample sizes and genre distributions prevent causal fairness claims.
- **Split deviation:** results use the documented random split rather than the official benchmark split.

## Sources and acknowledgements

- Arevalo et al., *Gated Multimodal Units for Information Fusion* (2017), for the MM-IMDb dataset and multimodal research context.
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2019).
- Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* (2019).
- Vaswani et al., *Attention Is All You Need* (2017).
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization* (2019), for AdamW.

This repository contains an individual university assignment experiment, not a production-ready genre classification system.
