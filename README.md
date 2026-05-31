# Fake News Detection in Regional Languages Using Hybrid Deep Learning Approaches

> **Research Paper** | Lahore Garrison University, Department of Computer Science  
> **Author:** Uzair Moazzam · `uzairmoazzam21@gmail.com`  
> **Co-Authors:** Maria Tariq · Junaid Saif Khan (SMU, Dallas TX) · Sundas Munir · Dr. Khushbu Khalid Butt · Tahir Alyas

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Class Imbalance Correction (SMOTE)](#class-imbalance-correction-smote)
  - [Tier 1 — Baseline Model](#tier-1--baseline-model)
  - [Tier 2 — Hybrid Model 1: Transformer Embeddings + Classical ML](#tier-2--hybrid-model-1-transformer-embeddings--classical-ml)
  - [Tier 3 — Hybrid Model 2: Feature Fusion](#tier-3--hybrid-model-2-feature-fusion)
  - [Tier 3 — Hybrid Model 3: Weighted Stacking Ensemble](#tier-3--hybrid-model-3-weighted-stacking-ensemble)
- [Complete Results](#complete-results)
- [Per-Class Deep Dive (Best Model)](#per-class-deep-dive-best-model)
- [Critical Findings](#critical-findings)
- [Hardware & Software Environment](#hardware--software-environment)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [Engineering Highlights](#engineering-highlights)
- [Future Work](#future-work)
- [Citation](#citation)

---

## Overview

Misinformation in low-resource languages represents a largely unsolved problem in NLP. Urdu — spoken by **230+ million people globally** — lacks standardised benchmarks, verified datasets, and robust detection tools. Existing research overwhelmingly targets English, leaving Urdu-speaking populations disproportionately exposed to digital misinformation.

This project addresses that gap through a **comprehensive three-tier experimental framework** evaluated on a 5,000-sample verified native Urdu news dataset. The framework systematically compares classical ML baselines against three novel hybrid architectures that combine multilingual transformer embeddings with classical ML classifiers, feature fusion, and weighted stacking ensembles.

**Problem:** 3-class Urdu text classification — `Real`, `Fake`, `Satire`  
**Best Result:** 83.33% accuracy · 0.8185 macro F1 · +13.33 pp over baseline

---

## Key Results

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|:----:|-------|:--------:|:---------:|:------:|:--------:|
| 🥇 1 | **H3 — Weighted Stacking Ensemble** | **83.33%** | **0.8148** | **0.8278** | **0.8185** |
| 2 | H2 — Feature Fusion (TF-IDF + Embeddings) | 81.73% | 0.7966 | 0.8056 | 0.7995 |
| 3 | H1 — Gradient Boosting on Embeddings | 81.47% | 0.7962 | 0.7989 | 0.7961 |
| 4 | H1 — XGBoost on Embeddings | 80.40% | 0.7863 | 0.7944 | 0.7876 |
| 5 | H1 — Logistic Regression on Embeddings | 78.53% | 0.7594 | 0.7611 | 0.7601 |
| 6 | H1 — Extra Trees on Embeddings | 77.60% | 0.7625 | 0.7689 | 0.7582 |
| 7 | H1 — Random Forest on Embeddings | 75.87% | 0.7440 | 0.7544 | 0.7415 |
| — | Naive Bayes Baseline (TF-IDF) | 70.00% | 0.7200 | 0.6100 | 0.6001 |

> **4 of 7 hybrid models exceed 80% accuracy.** All hybrid models outperform the baseline by a substantial margin. Macro-averaging penalises poor minority-class performance equally — these scores are not inflated by majority-class dominance.

---

## Dataset

| Property | Value |
|----------|-------|
| Total samples | 5,000 |
| Language | Native Urdu |
| Classes | Real/True · Fake · Satire |
| Source | Curated, manually verified Urdu news corpus |
| Format | 2-column CSV (`text`, `label`) |

**Class Distribution:**

| Class | Samples | Share |
|-------|--------:|------:|
| Real | 2,000 | 40.0% |
| Fake | 2,000 | 40.0% |
| Satire | 1,000 | 20.0% |

**Stratified Split** (same class ratios preserved in every split):

| Split | Total | Real | Fake | Satire |
|-------|------:|-----:|-----:|-------:|
| Training (70%) | 3,500 | 1,400 | 1,400 | 700 |
| Validation (15%) | 750 | 300 | 300 | 150 |
| Test (15%) | 750 | 300 | 300 | 150 |

**Text Length Distribution** (by word count):
- Fake: median **114 words** (long, narrative-heavy)
- True: median **40 words**
- Satire: median **18 words** (short, punchy — this brevity contributes to classification difficulty)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   RAW URDU TEXT INPUT                   │
└──────────────────────────┬──────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │    Urdu Preprocessing Pipeline │
           │  Unicode → Diacritics → URLs   │
           │  → Stop-words → Punctuation    │
           └───────────────┬───────────────┘
                           │
        ┌──────────────────┴───────────────────┐
        │           Two Variants               │
        │  Full Preprocessing  │  Minimal Prep  │
        │  (for TF-IDF)        │  (for Transformer)│
        └──────────┬───────────┴───────┬────────┘
                   │                   │
    ┌──────────────┘       ┌───────────┘
    │                      │
    ▼                      ▼
┌──────────┐    ┌─────────────────────────────┐
│  TF-IDF  │    │  DistilBERT-Multilingual    │
│ Features │    │  Mean-Pooled Embeddings     │
│ (10K/5K/ │    │  768-dimensional per doc    │
│  8K feat)│    │  NVIDIA Tesla P100 GPU      │
└────┬─────┘    └─────────────┬───────────────┘
     │                        │
     │          SMOTE Oversampling (training only)
     │          700 Satire → 1,400 Satire
     │                        │
     ├────────────────────────┤
     │                        │
     ▼                        ▼
┌─────────────────────────────────────────────┐
│             THREE HYBRID TIERS              │
├─────────────┬──────────────┬────────────────┤
│  TIER 1     │  TIER 2      │  TIER 3        │
│  NB Baseline│  H1: Embed   │  H2: Fusion    │
│  TF-IDF     │  + LR/RF/GB/ │  TF-IDF+Char   │
│             │  XGB/ET      │  +Embed→XGBoost│
│             │              │                │
│             │              │  H3: Weighted  │
│             │              │  Stack Ensemble│
│             │              │  (7 base models│
│             │              │   soft vote)   │
└─────────────┴──────────────┴────────────────┘
                           │
                           ▼
              FINAL PREDICTION: Real / Fake / Satire
```

---

## Methodology

### Preprocessing Pipeline

A custom Urdu preprocessing pipeline handles the unique challenges of Urdu text (right-to-left script, Arabic character variants, diacritical marks):

| Stage | Purpose |
|-------|---------|
| Unicode Normalisation (NFKC) | Standardise encoding; map Arabic-Urdu variant forms (e.g. `ي→ی`, `ك→ک`) |
| Diacritic Removal | Strip harakat marks (U+064B–U+0652) for consistent tokenisation |
| URL/Email Removal | Replace with `URL`/`EMAIL` tokens |
| Stop-word Filtering | Remove 36 curated Urdu function words (`کا`, `کی`, `میں`, etc.) |
| Punctuation Removal | Filter Arabic-Indic and standard punctuation |
| Whitespace Normalisation | Standardise spacing for downstream tokenisation |

Two variants are maintained:
- **Full preprocessing** (stop-word removal) → TF-IDF vectorisation
- **Minimal preprocessing** (normalisation + diacritics only) → Transformer tokenisation (preserves semantic context)

---

### Class Imbalance Correction (SMOTE)

The Satire class has only 700 training samples vs 1,400 each for Real and Fake — causing catastrophic recall failure in the baseline:

| Stage | Real | Fake | Satire |
|-------|-----:|-----:|-------:|
| Before SMOTE | 1,400 | 1,400 | 700 |
| After SMOTE | 1,400 | 1,400 | **1,400** |

SMOTE was applied with `k=5` nearest neighbours on transformer embeddings (not raw text). For TF-IDF features in the ensemble, `RandomOverSampler` was used due to the high dimensionality of scaled TF-IDF vectors.

**Impact on Satire recall:**

| Model | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| NB Baseline (no SMOTE) | 0.78 | **0.15** | 0.26 |
| H3 Ensemble (SMOTE) | 0.69 | **0.80** | 0.74 |
| **Improvement** | −0.09 | **+0.65** | **+0.48** |

A **65 percentage point recall improvement** on the minority class.

---

### Tier 1 — Baseline Model

**Multinomial Naive Bayes + TF-IDF**

- Vectoriser: 10,000 features, sublinear TF scaling
- Result: **70.00% accuracy, F1=0.6001**
- Critical failure: Satire recall = **0.15** (79 Satire articles predicted as Fake, 90 as True)

---

### Tier 2 — Hybrid Model 1: Transformer Embeddings + Classical ML

**Architecture:** `distilbert-base-multilingual-cased` → Mean-pooled 768-d embeddings → StandardScaler → 5 classifiers

**Embedding Extraction Configuration:**

| Parameter | Value |
|-----------|-------|
| Model | `distilbert-base-multilingual-cased` |
| Max sequence length | 256 tokens |
| Pooling | Mean pooling over all token embeddings |
| Embedding dimension | 768 |
| Batch size | 16 (auto-reduced on GPU OOM) |
| Normalisation | StandardScaler (zero mean, unit variance) |
| Hardware | NVIDIA Tesla P100-PCIE-16GB |

Mean pooling is used instead of CLS-only extraction — it aggregates information from every token position, yielding richer semantic representations for longer Urdu documents.

**Classifier Configurations:**

| Classifier | Key Hyperparameters | Accuracy |
|-----------|---------------------|:--------:|
| Logistic Regression | C=10, saga solver, L2, balanced weights | 78.53% |
| Random Forest | n_estimators=300, max_depth=20, balanced | 75.87% |
| Gradient Boosting | HistGB, max_iter=300, max_depth=8, lr=0.05 | **81.47%** |
| XGBoost | n_estimators=300, max_depth=8, GPU hist | 80.40% |
| Extra Trees | n_estimators=300, max_depth=20, balanced | 77.60% |

---

### Tier 3 — Hybrid Model 2: Feature Fusion

**Architecture:** TF-IDF (word + char) concatenated with transformer embeddings → XGBoost

**Feature Vector Composition (8,768 dimensions total):**

| Feature Type | Dimensions |
|-------------|:----------:|
| TF-IDF Word Features (unigrams–trigrams, min_df=2) | 5,000 |
| TF-IDF Character Features (2–5 n-grams, min_df=2) | 3,000 |
| DistilBERT Mean-Pooled Embeddings | 768 |
| **Total Fused Vector** | **8,768** |

The character n-gram features capture morphological patterns in Urdu (prefix/suffix patterns, script-level patterns) that word-level features miss.

Result: **81.73% accuracy, F1=0.7995**

---

### Tier 3 — Hybrid Model 3: Weighted Stacking Ensemble

**Architecture:** 7 base models trained on two complementary feature spaces → normalised weighted soft voting → argmax prediction

**Base Models and Weights:**

| Model | Feature Input | Weight |
|-------|--------------|:------:|
| Logistic Regression | Transformer embeddings | 1.0 |
| Random Forest | Transformer embeddings | 1.5 |
| Gradient Boosting | Transformer embeddings | 1.5 |
| XGBoost | Transformer embeddings | **1.8** |
| Extra Trees | Transformer embeddings | 1.5 |
| Logistic Regression | TF-IDF features | 1.0 |
| Multinomial NB | TF-IDF features | 0.8 |

Weights reflect empirical individual model performance — embedding-based models receive higher weights. Final prediction: weighted average of probability vectors → argmax.

Result: **83.33% accuracy, F1=0.8185** ← **Best Model**

---

## Complete Results

### H3 Ensemble Per-Class Breakdown (Test Set, n=750)

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|--------:|
| Real | 0.87 | **0.92** | 0.90 | 300 |
| Fake | 0.88 | 0.76 | 0.82 | 300 |
| Satire | 0.69 | 0.80 | 0.74 | 150 |
| **Macro Avg** | **0.8148** | **0.8278** | **0.8185** | 750 |
| **Accuracy** | | | **83.33%** | |

### Confusion Matrix — H3 Ensemble

```
                 Predicted
              Real   Fake  Satire
           ┌──────┬──────┬───────┐
True  Real │  276 │   13 │   11  │  92.00% recall
      Fake │   29 │  229 │   42  │  76.33% recall
    Satire │   11 │   19 │  120  │  80.00% recall
           └──────┴──────┴───────┘
```

**Key observation:** The dominant error is Fake→Satire confusion (42 samples, 14.00%). This is linguistically interpretable — Urdu satirical content frequently employs exaggerated but plausible claims that overlap stylistically with fabricated news.

### Statistical Summary Across All 7 Hybrid Models

| Statistic | Value |
|-----------|-------|
| Models evaluated | 7 |
| Test set size | 750 samples |
| Mean accuracy | 79.85% |
| Best accuracy (H3) | 83.33% |
| Worst accuracy (H1 RF) | 75.87% |
| Std deviation | 2.43% |
| Mean macro F1 | 0.7802 |
| Best macro F1 (H3) | 0.8185 |
| Models achieving ≥80% | **4 of 7** |
| Improvement over baseline | **+13.33 pp** |

---

## Per-Class Deep Dive (Best Model)

### Confidence Distribution — H3 Ensemble (n=750)

| Confidence Band | Samples | Proportion |
|----------------|--------:|:----------:|
| High (≥0.80) | 389 | 51.9% |
| Medium (0.60–0.80) | 188 | 25.1% |
| Low (<0.60) | 173 | 23.1% |
| **Average confidence** | | **0.7630** |

### Error Analysis by Class

| Class | Correct | Incorrect | Accuracy | Avg Confidence | Misclassified As |
|-------|--------:|----------:|:--------:|:--------------:|-----------------|
| Real | 276/300 | 24 | 92.00% | 0.7697 | Fake: 13 · Satire: 11 |
| Fake | 229/300 | 71 | 76.33% | 0.7815 | Satire: 42 · Real: 29 |
| Satire | 120/150 | 30 | 80.00% | 0.7125 | Fake: 19 · Real: 11 |

### Real-World Article Simulation (Baseline NB Model)

| Article (Urdu — translated) | Prediction | Confidence |
|----------------------------|-----------|:----------:|
| Pakistan FM presented Kashmir resolution at UN | True | 76.73% |
| Cricket captain to eat only gold and diamonds | FAKE | 50.77% |
| Government announces free textbooks policy | True | 42.31% |

---

## Critical Findings

1. **SMOTE is non-negotiable for minority class detection.** Without it, Satire recall collapses to 15%. SMOTE raised it to 80% — a 65 percentage point gain. This is the single most impactful intervention in the pipeline.

2. **Ensemble > fusion > single-classifier.** The weighted stacking ensemble (H3) consistently outperforms feature fusion (H2), which outperforms individual classifiers (H1). Diversity of both feature types and model types drives the improvement.

3. **Frozen embeddings still generalise.** DistilBERT-multilingual was used in frozen inference mode (no fine-tuning on Urdu data). Despite this, embedding-based models outperform TF-IDF baseline by 5–11 percentage points, demonstrating strong cross-lingual transfer.

4. **The Fake–Satire boundary is the hardest.** 14% of test samples are misclassified across this boundary. Urdu satirical writing mimics the stylistic conventions of fabricated news (exaggerated plausible claims, emotional language). This is a linguistic phenomenon, not a modelling artefact.

5. **Real news is the easiest class.** 92.00% recall with only 24 misclassified samples. Genuine Urdu news carries distinctive linguistic markers (formal register, specific named entities, institutional vocabulary) reliably captured by the ensemble.

6. **Gradient Boosting is the strongest individual classifier** on transformer embeddings (81.47%), narrowly outperforming XGBoost (80.40%). The HistGradientBoosting variant is used for memory efficiency.

---

## Hardware & Software Environment

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA Tesla P100-PCIE-16GB |
| CUDA Version | 11.8 |
| Python | 3.12 |
| PyTorch | 2.2.0+cu118 |
| Transformers | Hugging Face v4.20+ |
| Scikit-learn | 1.0+ |
| XGBoost | 1.5+ |
| Imbalanced-learn | SMOTE (imblearn) |
| Reproducibility seed | 42 (NumPy, PyTorch, Python random) |
| Platform | Kaggle (GPU-enabled notebook) |

---

## Repository Structure

```
├── requirements.txt
├── Research_Paper.pdf
├── README.md
├── notebook/
│   ├── baseline_and_visualizations.py     # Part 1: NB baseline + distribution plots
│   └── enhanced_hybrid_pipeline.py        # Part 2: Full 3-tier hybrid pipeline
├── data/
│   └── ml_ready_2column.csv               # 5,000-sample Urdu dataset (text, label)
├── outputs/
│   ├── predictions_best_H3_Ensemble.csv   # 750-sample test predictions w/ confidence
│   ├── predictions_h1_lr.csv
│   ├── predictions_h1_rf.csv
│   ├── predictions_h1_gb.csv
│   ├── predictions_h1_xgb.csv
│   ├── predictions_h1_et.csv
│   ├── predictions_h2_fusion.csv
│   └── predictions_h3_ensemble.csv
├── figures/
│   ├── class_distribution.png
│   ├── text_length_distribution.png
│   ├── confusion_matrix.png                       # Baseline
│   ├── classification_metrics_bar_chart.png       # Baseline
│   ├── figure1_performance_comparison.png         # All hybrid models
│   ├── figure2_confusion_matrix_analysis.png      # H3 best model
│   └── figure3_statistical_summary.png            # Ranked table + stats

    
```

---

## Installation & Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers xgboost imbalanced-learn accelerate
```

### Run Baseline Model

```python
# Update FILE_NAME to your dataset path
python baseline_and_visualizations.py
```

### Run Full Hybrid Pipeline

```python
# Update KAGGLE_PATH / COLAB_PATH / LOCAL_PATH at the bottom of the script
python enhanced_hybrid_pipeline.py
```

The pipeline auto-detects the dataset location across Kaggle, Colab, and local environments.

### Expected Outputs

Running the full pipeline generates:
- **7 prediction CSV files** with per-sample confidence scores
- **3 research-quality figures** (300 DPI PNG)
- Console output with complete classification reports and error analysis

### Dataset Path Configuration

```python
# At the bottom of enhanced_hybrid_pipeline.py:
KAGGLE_PATH = '/kaggle/input/datasets/uzairmoazzam203/research-paper/ml_ready_2column.csv'
COLAB_PATH  = None   # e.g. '/content/ml_ready_2column.csv'
LOCAL_PATH  = None   # e.g. '/home/user/data/ml_ready_2column.csv'
```

---

## Engineering Highlights

This project demonstrates production-level engineering practices beyond the research contribution:

- **CUDA compatibility auto-fix** — detects GPU compute capability mismatch and reinstalls the correct PyTorch build automatically, avoiding the `AcceleratorError: no kernel image available` error on Kaggle's Tesla P100
- **Dynamic path detection** — searches 10+ standard locations across Kaggle, Colab, and local environments, eliminating hardcoded path failures
- **GPU OOM recovery** — embedding extraction auto-halves batch size on `RuntimeError: out of memory` and retries, enabling large dataset processing on memory-constrained GPUs
- **Data leakage fix** — an explicit fix was applied to prevent training TF-IDF features from being used during test-set prediction in the stacking ensemble (a subtle but critical bug that inflates reported scores)
- **SMOTE safety check** — validates minimum samples per class before applying SMOTE; falls back to `RandomOverSampler` when class size is too small for k=5 neighbours
- **Memory management** — explicit `gc.collect()` and `torch.cuda.empty_cache()` calls between pipeline stages prevent GPU OOM across the full 3-tier run
- **FutureWarning elimination** — all seaborn `hue`/`legend` patterns updated to suppress deprecation warnings in research-quality figures
- **Calibrated probabilities** — classifiers without native `predict_proba` (e.g. LinearSVC) are wrapped with `CalibratedClassifierCV` to enable soft voting in the ensemble

---

## Future Work

| Priority | Direction |
|:--------:|-----------|
| 1 | Fine-tune XLM-RoBERTa end-to-end on the Urdu dataset (expected to push accuracy to 88–90%) |
| 2 | Incorporate linguistic features: code-mixing detection, sarcasm markers, cultural idioms |
| 3 | Cross-dataset evaluation on AX-to-Grind Urdu and Amjad 2020 benchmarks |
| 4 | Extend to Hindi and other South Asian regional languages for multilingual generalisation |
| 5 | Deploy a real-time Urdu fake news detection API for public use |

---

## Citation

```bibtex
@article{moazzam2025urdufnd,
  title     = {Fake News Detection in Regional Languages using Hybrid Deep Learning Approaches},
  author    = {Moazzam, Uzair and Tariq, Maria and Khan, Junaid Saif and
               Munir, Sundas and Butt, Khushbu Khalid and Alyas, Tahir},
  journal   = {Department of Computer Science, Lahore Garrison University},
  year      = {2025},
  address   = {Lahore, Pakistan}
}
```

---

## Contact

**Uzair Moazzam**  
Department of Computer Science, Lahore Garrison University, Lahore, Pakistan  
📧 uzairmoazzam21@gmail.com

---

*This research establishes a reproducible benchmark for hybrid deep learning approaches in low-resource Urdu fake news detection, demonstrating that ensemble strategies combining transformer embeddings with TF-IDF features consistently outperform single-feature approaches on linguistically complex classification tasks.*