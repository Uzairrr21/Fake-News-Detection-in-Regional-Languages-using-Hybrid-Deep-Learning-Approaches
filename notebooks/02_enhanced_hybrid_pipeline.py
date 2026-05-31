
# ============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================================================

import subprocess
import sys
import os

# ============================================================================
# CUDA FIX — Must run BEFORE any torch import
# Reinstalls PyTorch with correct CUDA version for Kaggle GPU (sm_89/sm_90)
# ============================================================================

def fix_cuda_compatibility():
    """
    Fixes: AcceleratorError: CUDA error: no kernel image available
    Root cause: Pre-installed PyTorch compiled for older CUDA compute capability.
    Solution: Reinstall PyTorch nightly or cu121 build which supports sm_89/sm_90.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("No GPU found — skipping CUDA fix")
            return

        # Test if current PyTorch works with this GPU
        try:
            t = torch.zeros(1).cuda()
            _ = t + 1
            del t
            torch.cuda.empty_cache()
            print(f"✅ CUDA already compatible — {torch.cuda.get_device_name(0)}")
            return  # Already working — no fix needed
        except Exception:
            pass  # Needs fix

        print("🔧 Fixing CUDA compatibility — reinstalling PyTorch...")
        print("   This takes ~2 minutes on first run...")

       # 1. Force reinstall NumPy to a compatible version first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "numpy<2.0.0", "--force-reinstall"
        ])

        # 2. Reinstall PyTorch
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "--upgrade",
            "torch==2.2.2",
            "torchvision==0.17.2",
            "torchaudio==2.2.2",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        print("✅ PyTorch cu121 installed — GPU should now work")
        print("⚠️  Please RESTART the kernel and run again")
        import sys as _sys
        _sys.exit(0)  # Force restart to load new PyTorch

    except Exception as e:
        print(f"⚠️  CUDA fix failed: {e} — will attempt CPU fallback")

fix_cuda_compatibility()

# ============================================================================
# Now safe to import torch
# ============================================================================

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

try:
    import transformers
except ImportError:
    install_package("transformers")

try:
    import xgboost
except ImportError:
    install_package("xgboost")

try:
    import imblearn
except ImportError:
    install_package("imbalanced-learn")

try:
    import accelerate
except ImportError:
    install_package("accelerate")

import warnings
warnings.filterwarnings('ignore')

import os
import gc
import re
import time
import unicodedata
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Will use GradientBoosting instead.")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Device selection — GPU preferred
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.empty_cache()
    print(f"✅ Device: cuda — {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("Device: cpu")

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================================
# SECTION 2: FILE PATH DETECTION (FIX #1)
# ============================================================================

def find_dataset(filename='ml_ready_2column.csv'):
    """
    FIX #1: Dynamic file path detection.
    Supports full paths, Kaggle paths, Colab paths, and local paths.
    If a full/absolute path is passed and exists, returns it directly.
    """
    # If caller passed a full path that already exists — use it directly
    if os.path.exists(filename):
        print(f"✓ Dataset found at: {filename}")
        return filename

    # Extract just the base filename for searching common locations
    base = os.path.basename(filename)

    search_paths = [
        # Kaggle standard input paths
        f'/kaggle/input/{base}',
        f'/kaggle/input/datasets/uzairmoazzam203/research-paper/{base}',
        f'/kaggle/input/research-paper/{base}',
        f'/kaggle/working/{base}',
        # Colab paths
        os.path.join('/content', base),
        os.path.join('/content/drive/MyDrive', base),
        os.path.join('/content/drive/MyDrive/datasets', base),
        # Local paths
        os.path.join(os.getcwd(), base),
        os.path.join(os.path.expanduser('~'), base),
        base,
    ]

    for path in search_paths:
        if os.path.exists(path):
            print(f"✓ Dataset found at: {path}")
            return path

    print(f"✗ Dataset '{base}' not found in any standard location.")
    print("  Searched locations:")
    for p in search_paths:
        print(f"    - {p}")
    print("\n  Please set FILE_PATH manually at the bottom of this script.")
    return None

# ============================================================================
# SECTION 3: ENHANCED URDU PREPROCESSING
# ============================================================================

class EnhancedUrduPreprocessor:
    """
    Complete Urdu text preprocessing pipeline:
    - Unicode normalization (NFKC)
    - Diacritic removal
    - URL/email removal
    - Stop-word filtering
    - Punctuation/numeric removal
    - Whitespace normalization
    """

    def __init__(self):
        self.stop_words = set([
            'کا', 'کی', 'کے', 'میں', 'نے', 'ہے', 'ہیں', 'تھا', 'تھی',
            'تھے', 'گا', 'گی', 'گے', 'ہو', 'تھوں', 'اور', 'یا',
            'مگر', 'لیکن', 'کہ', 'جو', 'سے', 'پر', 'یہ', 'وہ',
            'اس', 'ان', 'کو', 'کر', 'دیا', 'کیا', 'ہوا', 'ہوئی',
            'بھی', 'تو', 'ہی', 'نہیں', 'اب', 'جب', 'پھر', 'کب'
        ])

    def normalize_unicode(self, text):
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize('NFKC', text)
        replacements = {
            'ي': 'ی', 'ى': 'ی', 'ك': 'ک', 'ة': 'ہ',
            '\u200c': '', '\u200d': '', '\u200e': '', '\u200f': ''
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def remove_diacritics(self, text):
        diacritics = r'[\u064B-\u0652\u0670\u06D6-\u06ED]'
        return re.sub(diacritics, '', text)

    def preprocess(self, text, remove_stops=True):
        """Full preprocessing with stop-word removal"""
        if not isinstance(text, str):
            return ""
        text = self.normalize_unicode(text)
        text = self.remove_diacritics(text)
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        text = re.sub(
            r'[۔،؍؎؏؛؞;:!?.(){}[\]"\'<>«»`~@#$%^&*+=|\\/_-]',
            ' ', text
        )
        text = re.sub(r'\s+', ' ', text).strip()

        if remove_stops:
            words = text.split()
            words = [w for w in words if w not in self.stop_words and len(w) > 1]
            text = ' '.join(words)

        return text

    def preprocess_minimal(self, text):
        """Light preprocessing for transformer input (preserves more context)"""
        if not isinstance(text, str):
            return ""
        text = self.normalize_unicode(text)
        text = self.remove_diacritics(text)
        return re.sub(r'\s+', ' ', text).strip()

# ============================================================================
# SECTION 4: DATASET LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_dataset(file_path, text_column='text', label_column='label'):
    """Load, clean, and prepare the dataset with label mapping."""

    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"✓ Loaded {len(df)} samples")

    # Rename columns for consistency
    df = df.rename(columns={text_column: 'text', label_column: 'label'})
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip().str.len() > 10]
    print(f"✓ After cleaning: {len(df)} samples")

    # Label mapping — handles string and numeric labels
    label_mapping = {
        'real': 0, 'Real': 0, 'REAL': 0, 'true': 0, 'True': 0,
        'TRUE': 0, 0: 0, '0': 0,
        'fake': 1, 'Fake': 1, 'FAKE': 1, 'false': 1, 'False': 1,
        'FALSE': 1, 1: 1, '1': 1,
        'satire': 2, 'Satire': 2, 'SATIRE': 2, 2: 2, '2': 2
    }

    df['label'] = df['label'].map(label_mapping)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    num_classes = df['label'].nunique()
    print(f"\n✓ {num_classes}-class classification")
    print(f"✓ Label distribution:")
    label_names = {0: 'Real/True', 1: 'Fake', 2: 'Satire'}
    for label in sorted(df['label'].unique()):
        count = df['label'].value_counts()[label]
        name = label_names.get(label, str(label))
        print(f"   Class {label} ({name}): {count} ({count/len(df)*100:.1f}%)")

    return df, num_classes


def create_splits(df, test_size=0.15, val_size=0.15):
    """Stratified train/val/test split ensuring balanced class representation."""
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=SEED
    )
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio,
        stratify=train_val_df['label'], random_state=SEED
    )
    print(f"\n✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )

# ============================================================================
# SECTION 5: CLASS IMBALANCE HANDLING (FIX #3)
# ============================================================================

def handle_class_imbalance(X, y, method='smote'):
    """
    FIX #3: Handle class imbalance (especially Satire class).
    This was the root cause of 15% satire recall.

    Methods:
    - 'smote': Synthetic Minority Oversampling (recommended)
    - 'random': Random oversampling
    - 'smotetomek': SMOTE + Tomek links cleaning
    """
    print(f"\n🔄 Handling class imbalance using {method.upper()}...")
    print(f"   Before: {Counter(y)}")

    try:
        if method == 'smote':
            # Check minimum samples per class for SMOTE
            min_samples = min(Counter(y).values())
            k_neighbors = min(5, min_samples - 1)
            if k_neighbors < 1:
                print("   Too few samples for SMOTE, using RandomOverSampler instead")
                sampler = RandomOverSampler(random_state=SEED)
            else:
                sampler = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
        elif method == 'smotetomek':
            sampler = SMOTETomek(random_state=SEED)
        else:
            sampler = RandomOverSampler(random_state=SEED)

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"   After:  {Counter(y_resampled)}")
        return X_resampled, y_resampled

    except Exception as e:
        print(f"   Warning: Resampling failed ({e}). Using original data.")
        return X, y

# ============================================================================
# SECTION 6: TRANSFORMER EMBEDDING EXTRACTOR
# ============================================================================

class TransformerExtractor:
    """
    Extracts contextual embeddings from multilingual transformer models.
    Uses mean pooling over all token embeddings for richer representation.
    Supports: distilbert-base-multilingual-cased, xlm-roberta-base
    """

    def __init__(self, model_name='distilbert-base-multilingual-cased', max_length=256):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"\n📦 Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model.eval()
        self.model = self.model.to(DEVICE)
        print(f"✓ Model loaded on {DEVICE}")

    def extract_embeddings(self, texts, batch_size=16):
        """
        Extract mean-pooled embeddings from transformer hidden states.
        Uses mean pooling over all tokens for richer representation than CLS.
        Automatically reduces batch size if GPU OOM occurs.
        """
        print(f"🔄 Extracting embeddings for {len(texts)} samples...")
        embeddings = []
        current_batch = batch_size

        with torch.no_grad():
            i = 0
            while i < len(texts):
                batch = texts[i:i + current_batch]
                try:
                    encoded = self.tokenizer(
                        batch.tolist() if hasattr(batch, 'tolist') else list(batch),
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

                    outputs = self.model.base_model(**encoded)

                    # Mean pooling over all token embeddings
                    attention_mask = encoded['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(
                        token_embeddings.size()
                    ).float()
                    
                    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    mean_pooled = (sum_embeddings / sum_mask).detach().cpu().tolist()

                    embeddings.append(mean_pooled)
                    i += current_batch

                    processed = min(i, len(texts))
                    if processed % 320 == 0 or processed == len(texts):
                        print(f"  ✓ {processed}/{len(texts)}")

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and current_batch > 2:
                        current_batch = max(2, current_batch // 2)
                        print(f"  ⚠️  GPU OOM — reducing batch size to {current_batch}")
                        torch.cuda.empty_cache()
                        # Retry same batch with smaller size
                    else:
                        raise e

        result = np.vstack(embeddings)
        print(f"✓ Embeddings shape: {result.shape}")
        return result

    def cleanup(self):
        """Free GPU memory after extraction."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        clear_memory()
        print("✓ Model memory freed")

# ============================================================================
# SECTION 7: HYBRID MODEL 1 — TRANSFORMER EMBEDDINGS + CLASSIFIERS
# ============================================================================

class HybridModel1:
    """
    Architecture: Transformer (distilbert-multilingual) → Mean-pooled embeddings
                  → StandardScaler → Multiple classical ML classifiers

    Classifiers: Logistic Regression, Random Forest, Gradient Boosting,
                 XGBoost (if available), Extra Trees
    """

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        self.classifiers = {}
        self.train_emb = None
        self.val_emb = None
        self.test_emb = None

    def extract_features(self, train_texts, val_texts, test_texts,
                         model_name='distilbert-base-multilingual-cased'):
        """Extract and scale transformer embeddings for all splits."""
        extractor = TransformerExtractor(model_name=model_name, max_length=256)
        extractor.load_model()

        self.train_emb = extractor.extract_embeddings(train_texts)
        self.val_emb = extractor.extract_embeddings(val_texts)
        self.test_emb = extractor.extract_embeddings(test_texts)

        # Scale embeddings
        self.train_emb = self.scaler.fit_transform(self.train_emb)
        self.val_emb = self.scaler.transform(self.val_emb)
        self.test_emb = self.scaler.transform(self.test_emb)

        extractor.cleanup()
        print("✓ Features extracted and scaled")

    def train(self, y_train, apply_smote=True):
        """Train all classifiers, optionally with SMOTE oversampling."""
        print("\n" + "=" * 60)
        print("🎯 HYBRID MODEL 1: Training Classifiers")
        print("=" * 60)

        # FIX #3: Apply SMOTE to fix class imbalance
        X_train = self.train_emb
        if apply_smote and self.num_classes > 2:
            X_train, y_train = handle_class_imbalance(
                self.train_emb, y_train, method='smote'
            )

        # 1. Logistic Regression
        print("\n1. Logistic Regression")
        self.classifiers['LR'] = LogisticRegression(
            max_iter=3000, C=10.0, solver='saga',
            class_weight='balanced', random_state=SEED,
            penalty='l2', multi_class='multinomial'
        )
        self.classifiers['LR'].fit(X_train, y_train)

        # 2. Random Forest
        print("2. Random Forest")
        self.classifiers['RF'] = RandomForestClassifier(
            n_estimators=300, max_depth=20,
            min_samples_split=3, min_samples_leaf=1,
            class_weight='balanced', random_state=SEED, n_jobs=-1
        )
        self.classifiers['RF'].fit(X_train, y_train)

       # 3. Gradient Boosting (Fast Histogram Mode)
        print("3. Gradient Boosting (Fast Histogram)")
        self.classifiers['GB'] = HistGradientBoostingClassifier(
            max_iter=300, max_depth=8,
            learning_rate=0.05,
            random_state=SEED
        )
        self.classifiers['GB'].fit(X_train, y_train)

        # 4. XGBoost (GPU Accelerated)
        if XGBOOST_AVAILABLE:
            print("4. XGBoost (GPU Accelerated)")
            scale_pos = len(y_train) / (self.num_classes * np.bincount(y_train))
            self.classifiers['XGB'] = XGBClassifier(
                n_estimators=300, max_depth=8,
                learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.9, gamma=0.1,
                random_state=SEED, eval_metric='mlogloss',
                use_label_encoder=False, n_jobs=-1,
                tree_method='hist', device='cuda'
            )
            self.classifiers['XGB'].fit(X_train, y_train)

        # 5. Extra Trees
        print("5. Extra Trees")
        self.classifiers['ET'] = ExtraTreesClassifier(
            n_estimators=300, max_depth=20,
            class_weight='balanced', random_state=SEED, n_jobs=-1
        )
        self.classifiers['ET'].fit(X_train, y_train)

        print("\n✓ All classifiers trained")

    def evaluate(self, y_test):
        """Evaluate all classifiers on test set."""
        results = {}
        predictions = {}
        probabilities = {}

        for name, clf in self.classifiers.items():
            y_pred = clf.predict(self.test_emb)

            # Get probabilities
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(self.test_emb)
            else:
                # Calibrate for classifiers without predict_proba
                cal_clf = CalibratedClassifierCV(clf, cv='prefit')
                cal_clf.fit(self.val_emb, y_test[:len(self.val_emb)])
                y_proba = cal_clf.predict_proba(self.test_emb)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            results[f'H1_{name}'] = {
                'accuracy': acc,
                'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1': f1
            }
            predictions[f'H1_{name}'] = y_pred
            probabilities[f'H1_{name}'] = y_proba

            print(f"\n  {name}: Acc={acc*100:.2f}% | F1={f1:.4f}")

        return results, predictions, probabilities

# ============================================================================
# SECTION 8: HYBRID MODEL 2 — FEATURE FUSION (TF-IDF + EMBEDDINGS)
# ============================================================================

class HybridModel2:
    """
    Architecture: TF-IDF word features (5000) + char features (3000)
                  + transformer embeddings (768)
                  → Concatenated → StandardScaler → XGBoost/GradientBoosting

    FIX: Reduced feature dimensions to prevent memory crash.
    """

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        # FIX #4: Reduced max_features from 10000 to 5000 to prevent memory crash
        self.tfidf_word = TfidfVectorizer(
            max_features=5000,       # Reduced from 10000
            ngram_range=(1, 3),      # Reduced from (1,4)
            min_df=2, max_df=0.85,
            sublinear_tf=True
        )
        # FIX #4: Reduced char features from 5000 to 3000
        self.tfidf_char = TfidfVectorizer(
            max_features=3000,       # Reduced from 5000
            analyzer='char',
            ngram_range=(2, 5),      # Reduced from (2,6)
            min_df=2, max_df=0.85
        )
        self.scaler = StandardScaler()
        self.classifier = None
        self.X_train = None
        self.X_val = None
        self.X_test = None

    def prepare_features(self, train_texts, val_texts, test_texts,
                         train_emb, val_emb, test_emb):
        """Fuse TF-IDF features with transformer embeddings."""
        print("\n" + "=" * 60)
        print("🔧 HYBRID MODEL 2: Feature Fusion")
        print("=" * 60)

        print("Creating TF-IDF word features...")
        # FIX: Use toarray() carefully — manageable with reduced features
        tfidf_word_train = self.tfidf_word.fit_transform(train_texts).toarray()
        tfidf_word_val = self.tfidf_word.transform(val_texts).toarray()
        tfidf_word_test = self.tfidf_word.transform(test_texts).toarray()

        print("Creating TF-IDF char features...")
        tfidf_char_train = self.tfidf_char.fit_transform(train_texts).toarray()
        tfidf_char_val = self.tfidf_char.transform(val_texts).toarray()
        tfidf_char_test = self.tfidf_char.transform(test_texts).toarray()

        print("Fusing all features...")
        self.X_train = np.hstack([tfidf_word_train, tfidf_char_train, train_emb])
        self.X_val = np.hstack([tfidf_word_val, tfidf_char_val, val_emb])
        self.X_test = np.hstack([tfidf_word_test, tfidf_char_test, test_emb])

        # Scale fused features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"✓ Fused feature dimensions: {self.X_train.shape}")
        feat_desc = (
            f"  Word TF-IDF: {tfidf_word_train.shape[1]} | "
            f"Char TF-IDF: {tfidf_char_train.shape[1]} | "
            f"Embeddings: {train_emb.shape[1]}"
        )
        print(feat_desc)

    def train(self, y_train, apply_smote=True):
        """Train fusion classifier with optional SMOTE."""
        print("\n🎯 Training on fused features...")

        X_train = self.X_train
        # FIX #3: Apply SMOTE for class imbalance
        if apply_smote and self.num_classes > 2:
            X_train, y_train = handle_class_imbalance(
                X_train, y_train, method='smote'
            )

        if XGBOOST_AVAILABLE:
            self.classifier = XGBClassifier(
                n_estimators=500, max_depth=10,
                learning_rate=0.01, subsample=0.9,
                colsample_bytree=0.9, gamma=0.05,
                min_child_weight=1, reg_alpha=0.1,
                reg_lambda=1, random_state=SEED,
                eval_metric='mlogloss', n_jobs=-1,
                use_label_encoder=False,
                tree_method='hist', device='cuda'
            )
        else:
            self.classifier = HistGradientBoostingClassifier(
                max_iter=500, max_depth=10,
                learning_rate=0.01,
                random_state=SEED
            )

        self.classifier.fit(X_train, y_train)
        print("✓ Fusion model training completed")

    def evaluate(self, y_test):
        """Evaluate fusion model on test set."""
        y_pred = self.classifier.predict(self.X_test)

        if hasattr(self.classifier, 'predict_proba'):
            y_proba = self.classifier.predict_proba(self.X_test)
        else:
            y_proba = np.zeros((len(y_test), self.num_classes))

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results = {
            'H2_Fusion': {
                'accuracy': acc,
                'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1': f1
            }
        }

        print(f"\n  Fusion: Acc={acc*100:.2f}% | F1={f1:.4f}")
        return results, y_pred, y_proba

# ============================================================================
# SECTION 9: HYBRID MODEL 3 — WEIGHTED STACKING ENSEMBLE (FIX #2)
# ============================================================================

class HybridModel3:
    """
    Architecture: Multiple base models trained on embeddings AND TF-IDF
                  → Weighted soft voting on probabilities
                  → Final prediction

    FIX #2: Fixed critical data leakage bug.
    Was using self.train_tfidf for lr_tfidf prediction.
    Now correctly uses self.test_tfidf for all test predictions.
    """

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.models = {}
        self.tfidf_min = 0
        self.train_emb = None
        self.test_emb = None
        self.train_tfidf = None
        self.test_tfidf = None  # FIX: Explicitly track test_tfidf separately

    def prepare_data(self, train_emb, test_emb, train_tfidf, test_tfidf):
        """Store all train/test feature matrices."""
        self.train_emb = train_emb
        self.test_emb = test_emb
        self.train_tfidf = train_tfidf
        self.test_tfidf = test_tfidf  # FIX #2: Store test_tfidf properly

    def train(self, y_train, apply_smote=True):
        """Train all base models with optional SMOTE."""
        print("\n" + "=" * 60)
        print("🏗️ HYBRID MODEL 3: Weighted Stacking Ensemble")
        print("=" * 60)

        train_emb = self.train_emb
        train_tfidf = self.train_tfidf
        y_emb = y_train
        y_tfidf = y_train

        # FIX #3: Apply SMOTE separately for embedding and TF-IDF features
        if apply_smote and self.num_classes > 2:
            print("\n🔄 Applying SMOTE for embedding models...")
            train_emb, y_emb = handle_class_imbalance(
                train_emb, y_train, method='smote'
            )
            print("\n🔄 Applying SMOTE for TF-IDF models...")
            # Shift TF-IDF to non-negative for SMOTE
            tfidf_shifted = train_tfidf - train_tfidf.min() + 1e-10
            tfidf_resampled, y_tfidf = handle_class_imbalance(
                tfidf_shifted, y_train, method='random'
            )
            train_tfidf = tfidf_resampled

        # --- Embedding-based models ---
        print("\n🔄 Training base models on embeddings...")

        self.models['lr_emb'] = LogisticRegression(
            max_iter=3000, C=10, class_weight='balanced',
            random_state=SEED, multi_class='multinomial'
        )
        self.models['lr_emb'].fit(train_emb, y_emb)

        self.models['rf_emb'] = RandomForestClassifier(
            n_estimators=300, max_depth=20,
            class_weight='balanced', random_state=SEED, n_jobs=-1
        )
        self.models['rf_emb'].fit(train_emb, y_emb)

        self.models['gb_emb'] = HistGradientBoostingClassifier(
            max_iter=300, max_depth=8,
            learning_rate=0.05, random_state=SEED
        )
        self.models['gb_emb'].fit(train_emb, y_emb)

        if XGBOOST_AVAILABLE:
            self.models['xgb_emb'] = XGBClassifier(
                n_estimators=300, max_depth=8,
                learning_rate=0.05, random_state=SEED,
                eval_metric='mlogloss', use_label_encoder=False,
                n_jobs=-1, tree_method='hist', device='cuda'
            )
            self.models['xgb_emb'].fit(train_emb, y_emb)

        self.models['et_emb'] = ExtraTreesClassifier(
            n_estimators=300, max_depth=20,
            class_weight='balanced', random_state=SEED, n_jobs=-1
        )
        self.models['et_emb'].fit(train_emb, y_emb)

        # --- TF-IDF based models ---
        print("\n🔄 Training base models on TF-IDF...")

        self.models['lr_tfidf'] = LogisticRegression(
            max_iter=3000, C=10, class_weight='balanced',
            random_state=SEED, multi_class='multinomial'
        )
        self.models['lr_tfidf'].fit(train_tfidf, y_tfidf)

        # Naive Bayes needs non-negative features
        self.tfidf_min = train_tfidf.min()
        tfidf_shifted_nb = train_tfidf - self.tfidf_min + 1e-10
        self.models['nb_tfidf'] = MultinomialNB(alpha=0.01)
        self.models['nb_tfidf'].fit(tfidf_shifted_nb, y_tfidf)

        print("\n✓ All ensemble base models trained")

    def predict(self, return_proba=False):
        """
        FIX #2: All predictions now correctly use self.test_emb
        and self.test_tfidf — no more train data leakage.
        """
        print("\n🎲 Generating weighted ensemble predictions...")

        probas = []
        weights = []

        # Model weights — higher for stronger models
        model_weights = {
            'lr_emb': 1.0, 'rf_emb': 1.5, 'gb_emb': 1.5,
            'xgb_emb': 1.8, 'et_emb': 1.5,
            'lr_tfidf': 1.0, 'nb_tfidf': 0.8
        }

        # Embedding models — predict on TEST embeddings
        emb_models = ['lr_emb', 'rf_emb', 'gb_emb', 'xgb_emb', 'et_emb']
        for name in emb_models:
            if name in self.models:
                try:
                    p = self.models[name].predict_proba(self.test_emb)
                    probas.append(p)
                    weights.append(model_weights[name])
                except Exception as e:
                    print(f"  Warning: {name} prediction failed: {e}")

        # TF-IDF models — FIX #2: predict on TEST tfidf (was train_tfidf before!)
        if 'lr_tfidf' in self.models:
            try:
                p = self.models['lr_tfidf'].predict_proba(self.test_tfidf)
                probas.append(p)
                weights.append(model_weights['lr_tfidf'])
            except Exception as e:
                print(f"  Warning: lr_tfidf prediction failed: {e}")

        if 'nb_tfidf' in self.models:
            try:
                # FIX #2: Use test_tfidf, not train_tfidf
                tfidf_test_shifted = self.test_tfidf - self.tfidf_min + 1e-10
                p = self.models['nb_tfidf'].predict_proba(tfidf_test_shifted)
                probas.append(p)
                weights.append(model_weights['nb_tfidf'])
            except Exception as e:
                print(f"  Warning: nb_tfidf prediction failed: {e}")

        if not probas:
            print("  Error: No valid predictions collected.")
            n = len(self.test_emb)
            dummy_proba = np.full((n, self.num_classes), 1/self.num_classes)
            return np.zeros(n, dtype=int), dummy_proba

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted average of probabilities
        avg_proba = np.zeros_like(probas[0], dtype=float)
        for p, w in zip(probas, weights):
            if p.shape == avg_proba.shape:
                avg_proba += w * p
            else:
                print(f"  Warning: Skipping proba with shape {p.shape} (expected {avg_proba.shape})")

        predictions = np.argmax(avg_proba, axis=1)

        if return_proba:
            return predictions, avg_proba
        return predictions

    def evaluate(self, y_test):
        """Evaluate ensemble on test set."""
        y_pred, y_proba = self.predict(return_proba=True)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results = {
            'H3_Ensemble': {
                'accuracy': acc,
                'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1': f1
            }
        }

        print(f"\n  Ensemble: Acc={acc*100:.2f}% | F1={f1:.4f}")
        return results, y_pred, y_proba

# ============================================================================
# SECTION 10: DETAILED PREDICTION ANALYSIS (Integrated from File 3)
# ============================================================================

def analyze_predictions(df_test, y_true, y_pred, y_proba, model_name,
                        num_classes=3, num_samples=5):
    """Detailed analysis of predictions with confidence scores."""

    print("\n" + "=" * 80)
    print(f"DETAILED PREDICTION ANALYSIS - {model_name}")
    print("=" * 80)

    label_names = {0: 'Real', 1: 'Fake', 2: 'Satire'} if num_classes == 3 \
        else {0: 'Real', 1: 'Fake'}

    results_df = df_test.copy().reset_index(drop=True)
    results_df['true_label'] = y_true
    results_df['predicted_label'] = y_pred
    results_df['confidence'] = np.max(y_proba, axis=1)
    results_df['correct'] = (
        np.array(y_true) == np.array(y_pred)
    )

    # Add probability columns
    if y_proba.shape[1] >= 1:
        results_df['prob_real'] = y_proba[:, 0]
    if y_proba.shape[1] >= 2:
        results_df['prob_fake'] = y_proba[:, 1]
    if y_proba.shape[1] >= 3:
        results_df['prob_satire'] = y_proba[:, 2]

    # Summary statistics
    total = len(results_df)
    correct = results_df['correct'].sum()
    print(f"\n📊 Summary Statistics:")
    print(f"   Total Samples:       {total}")
    print(f"   Correct:             {correct} ({correct/total*100:.2f}%)")
    print(f"   Incorrect:           {total-correct} ({(total-correct)/total*100:.2f}%)")
    print(f"   Avg Confidence:      {results_df['confidence'].mean():.4f}")
    print(f"   Min Confidence:      {results_df['confidence'].min():.4f}")
    print(f"   Max Confidence:      {results_df['confidence'].max():.4f}")

    # Confidence distribution
    high = (results_df['confidence'] >= 0.8).sum()
    med = ((results_df['confidence'] >= 0.6) & (results_df['confidence'] < 0.8)).sum()
    low = (results_df['confidence'] < 0.6).sum()
    print(f"\n🎯 Confidence Distribution:")
    print(f"   High (≥0.80): {high} ({high/total*100:.1f}%)")
    print(f"   Med  (0.6-0.8): {med} ({med/total*100:.1f}%)")
    print(f"   Low  (<0.60): {low} ({low/total*100:.1f}%)")

    # Sample correct high-confidence predictions
    correct_high = results_df[
        results_df['correct'] & (results_df['confidence'] >= 0.8)
    ].head(num_samples)

    print(f"\n✅ HIGH CONFIDENCE CORRECT PREDICTIONS (sample of {len(correct_high)}):")
    for _, row in correct_high.iterrows():
        text_preview = str(row.get('text', ''))[:100]
        true_name = label_names.get(int(row['true_label']), str(row['true_label']))
        print(f"  True: {true_name} | Conf: {row['confidence']:.4f} | Text: {text_preview}...")

    # Sample incorrect predictions
    incorrect = results_df[~results_df['correct']].head(num_samples)
    print(f"\n❌ INCORRECT PREDICTIONS (sample of {len(incorrect)}):")
    for _, row in incorrect.iterrows():
        text_preview = str(row.get('text', ''))[:100]
        true_name = label_names.get(int(row['true_label']), str(row['true_label']))
        pred_name = label_names.get(int(row['predicted_label']), str(row['predicted_label']))
        print(f"  True: {true_name} → Predicted: {pred_name} | Conf: {row['confidence']:.4f}")
        print(f"  Text: {text_preview}...")

    return results_df


def error_analysis_by_class(results_df, num_classes=3):
    """Analyze error patterns per class."""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS BY CLASS")
    print("=" * 80)

    label_names = {0: 'Real', 1: 'Fake', 2: 'Satire'} if num_classes == 3 \
        else {0: 'Real', 1: 'Fake'}

    for true_label in sorted(results_df['true_label'].unique()):
        class_data = results_df[results_df['true_label'] == true_label]
        correct = class_data['correct'].sum()
        total = len(class_data)
        acc = correct / total if total > 0 else 0

        name = label_names.get(int(true_label), str(true_label))
        print(f"\n{name} Class:")
        print(f"   Samples:     {total}")
        print(f"   Correct:     {correct} ({acc*100:.2f}%)")
        print(f"   Incorrect:   {total-correct} ({(1-acc)*100:.2f}%)")
        print(f"   Avg Conf:    {class_data['confidence'].mean():.4f}")

        if total > correct:
            misclassified = class_data[~class_data['correct']]
            print(f"   Misclassified as:")
            for pred_label in misclassified['predicted_label'].unique():
                count = (misclassified['predicted_label'] == pred_label).sum()
                pred_name = label_names.get(int(pred_label), str(pred_label))
                print(f"      → {pred_name}: {count} times")


def export_predictions(results_df, model_name, output_path=None):
    """Export predictions to CSV."""
    if output_path is None:
        safe_name = model_name.replace(' ', '_').lower()
        output_path = f'predictions_{safe_name}.csv'

    export_cols = ['text', 'true_label', 'predicted_label',
                   'confidence', 'correct']
    for col in ['prob_real', 'prob_fake', 'prob_satire']:
        if col in results_df.columns:
            export_cols.append(col)

    export_df = results_df[export_cols].copy()
    export_df['model'] = model_name
    export_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"💾 Exported {len(export_df)} predictions → {output_path}")
    return output_path

# ============================================================================
# SECTION 11: RESEARCH PAPER VISUALIZATIONS
# ============================================================================

def create_all_visualizations(all_results, y_test, all_predictions, num_classes=3):
    """
    Generate all research paper figures.
    Figure numbering is consistent throughout (FIX for Tea Leaf paper issue).
    """
    label_names = ['Real', 'Fake', 'Satire'] if num_classes == 3 else ['Real', 'Fake']

    models = list(all_results.keys())
    accuracies = [all_results[m]['accuracy'] for m in models]
    f1_scores_list = [all_results[m]['f1'] for m in models]
    precisions = [all_results[m]['precision'] for m in models]
    recalls = [all_results[m]['recall'] for m in models]

    best_model_name = max(all_results, key=lambda x: all_results[x]['accuracy'])
    best_preds = all_predictions[best_model_name]

    # ── Figure 1: Performance Comparison ──────────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Figure 1: Model Performance Comparison\nEnhanced Hybrid Fake News Detection',
                  fontsize=14, fontweight='bold')

    # Accuracy bar chart
    colors = ['#2ecc71' if a >= 0.90 else '#f39c12' if a >= 0.80 else '#e74c3c'
              for a in accuracies]
    bars = axes[0, 0].bar(range(len(models)), accuracies, color=colors)
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].axhline(y=0.90, color='green', linestyle='--', linewidth=2,
                       label='90% Target')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylim([0.5, 1.0])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')

    # F1 bar chart
    bars2 = axes[0, 1].bar(range(len(models)), f1_scores_list, color='#3498db')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('F1-Score (Macro)')
    axes[0, 1].set_title('F1-Score Comparison')
    axes[0, 1].set_ylim([0.5, 1.0])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, f1_scores_list):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f'{val:.3f}', ha='center', fontsize=8)

    # Precision vs Recall
    x = np.arange(len(models))
    w = 0.35
    axes[0, 2].bar(x - w/2, precisions, w, label='Precision', color='#e74c3c', alpha=0.8)
    axes[0, 2].bar(x + w/2, recalls, w, label='Recall', color='#3498db', alpha=0.8)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    axes[0, 2].set_title('Precision vs Recall')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].set_ylim([0.5, 1.0])
    axes[0, 2].grid(axis='y', alpha=0.3)

    # All metrics grouped
    x2 = np.arange(len(models))
    w2 = 0.2
    axes[1, 0].bar(x2 - 1.5*w2, accuracies, w2, label='Acc', color='#2ecc71')
    axes[1, 0].bar(x2 - 0.5*w2, precisions, w2, label='Prec', color='#e74c3c')
    axes[1, 0].bar(x2 + 0.5*w2, recalls, w2, label='Rec', color='#3498db')
    axes[1, 0].bar(x2 + 1.5*w2, f1_scores_list, w2, label='F1', color='#f39c12')
    axes[1, 0].set_xticks(x2)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=7)
    axes[1, 0].set_title('All Metrics Overview')
    axes[1, 0].legend(fontsize=7, loc='lower right')
    axes[1, 0].set_ylim([0.5, 1.05])
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Best model confusion matrix
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=label_names[:num_classes],
                yticklabels=label_names[:num_classes])
    best_acc = all_results[best_model_name]['accuracy']
    axes[1, 1].set_title(f'Confusion Matrix\n{best_model_name} ({best_acc*100:.2f}%)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    # Model rankings
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    rank_names = [m[0] for m in sorted_models]
    rank_f1s = [m[1]['f1'] for m in sorted_models]
    rank_colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(rank_names)))
    rank_bars = axes[1, 2].barh(rank_names, rank_f1s, color=rank_colors)
    axes[1, 2].set_xlabel('F1-Score')
    axes[1, 2].set_title('Model Rankings (F1-Score)')
    axes[1, 2].set_xlim([0.5, 1.0])
    axes[1, 2].grid(axis='x', alpha=0.3)
    for bar, val in zip(rank_bars, rank_f1s):
        axes[1, 2].text(val - 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', ha='right', va='center',
                        fontsize=8, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig('figure1_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figure1_performance_comparison.png")

    # ── Figure 2: Confusion Matrix Analysis ───────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Figure 2: Confusion Matrix and Per-Class Analysis',
                  fontsize=14, fontweight='bold')

    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[0],
                xticklabels=label_names[:num_classes],
                yticklabels=label_names[:num_classes])
    axes2[0].set_title(f'Confusion Matrix (Counts)\nBest: {best_model_name}')
    axes2[0].set_ylabel('True Label')
    axes2[0].set_xlabel('Predicted Label')

    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes2[1],
                xticklabels=label_names[:num_classes],
                yticklabels=label_names[:num_classes],
                vmin=0, vmax=1)
    axes2[1].set_title('Normalized Confusion Matrix (%)')
    axes2[1].set_ylabel('True Label')
    axes2[1].set_xlabel('Predicted Label')

    # Per-class F1 comparison (all models)
    report = classification_report(y_test, best_preds, output_dict=True, zero_division=0)
    classes_str = [str(i) for i in range(num_classes)]
    per_class_f1 = [report[c]['f1-score'] for c in classes_str if c in report]
    per_class_prec = [report[c]['precision'] for c in classes_str if c in report]
    per_class_rec = [report[c]['recall'] for c in classes_str if c in report]

    x3 = np.arange(num_classes)
    w3 = 0.25
    axes2[2].bar(x3 - w3, per_class_prec, w3, label='Precision', color='#e74c3c', alpha=0.8)
    axes2[2].bar(x3, per_class_rec, w3, label='Recall', color='#3498db', alpha=0.8)
    axes2[2].bar(x3 + w3, per_class_f1, w3, label='F1-Score', color='#f39c12', alpha=0.8)
    axes2[2].set_xticks(x3)
    axes2[2].set_xticklabels(label_names[:num_classes])
    axes2[2].set_title(f'Per-Class Metrics\n{best_model_name}')
    axes2[2].legend()
    axes2[2].set_ylim([0, 1.1])
    axes2[2].grid(axis='y', alpha=0.3)
    for i, (p, r, f) in enumerate(zip(per_class_prec, per_class_rec, per_class_f1)):
        axes2[2].text(i - w3, p + 0.02, f'{p:.2f}', ha='center', fontsize=7)
        axes2[2].text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=7)
        axes2[2].text(i + w3, f + 0.02, f'{f:.2f}', ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig('figure2_confusion_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figure2_confusion_matrix_analysis.png")

    # ── Figure 3: Statistical Summary ─────────────────────────────────────
    fig3 = plt.figure(figsize=(16, 8))
    fig3.suptitle('Figure 3: Statistical Summary and Best Model Analysis',
                  fontsize=14, fontweight='bold')

    gs = fig3.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    ax_table = fig3.add_subplot(gs[0, :])
    ax_table.axis('off')

    table_data = []
    for rank, (model, m) in enumerate(
        sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True), 1
    ):
        status = "✅" if m['accuracy'] >= 0.90 else "⚠️" if m['accuracy'] >= 0.80 else "❌"
        table_data.append([
            f"#{rank}", model,
            f"{m['accuracy']*100:.2f}%",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            status
        ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Status'],
        cellLoc='center', loc='center',
        colWidths=[0.08, 0.28, 0.12, 0.12, 0.12, 0.12, 0.08]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    for i in range(7):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, len(table_data) + 1):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
        if i == 1:
            for j in range(7):
                table[(i, j)].set_facecolor('#2ecc71')
                table[(i, j)].set_text_props(weight='bold')

    ax_table.set_title('Complete Model Performance Summary',
                       fontsize=12, fontweight='bold', pad=15)

    # Stats box
    ax_stats = fig3.add_subplot(gs[1, 0])
    ax_stats.axis('off')
    stats_text = (
        f"STATISTICAL ANALYSIS\n"
        f"{'═'*32}\n\n"
        f"Models Evaluated: {len(models)}\n"
        f"Test Samples: {len(y_test)}\n"
        f"Classes: {num_classes}\n\n"
        f"Accuracy:\n"
        f"  Mean: {np.mean(accuracies)*100:.2f}%\n"
        f"  Max:  {np.max(accuracies)*100:.2f}%\n"
        f"  Min:  {np.min(accuracies)*100:.2f}%\n"
        f"  Std:  {np.std(accuracies)*100:.2f}%\n\n"
        f"F1 (Macro):\n"
        f"  Mean: {np.mean(f1_scores_list):.4f}\n"
        f"  Max:  {np.max(f1_scores_list):.4f}\n\n"
        f"Models ≥90%: {sum(1 for a in accuracies if a >= 0.90)}/{len(accuracies)}\n"
        f"Models ≥80%: {sum(1 for a in accuracies if a >= 0.80)}/{len(accuracies)}"
    )
    ax_stats.text(0.05, 0.5, stats_text, fontsize=9,
                  verticalalignment='center', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    # Best model box
    ax_best = fig3.add_subplot(gs[1, 1])
    ax_best.axis('off')
    best_m = all_results[best_model_name]
    target_met = "✅ ACHIEVED" if best_m['accuracy'] >= 0.90 else "⚠️ CLOSE" \
        if best_m['accuracy'] >= 0.80 else "❌ NOT MET"
    best_text = (
        f"BEST MODEL\n"
        f"{'═'*32}\n\n"
        f"Model: {best_model_name}\n\n"
        f"Accuracy:  {best_m['accuracy']*100:.2f}%\n"
        f"Precision: {best_m['precision']:.4f}\n"
        f"Recall:    {best_m['recall']:.4f}\n"
        f"F1-Score:  {best_m['f1']:.4f}\n\n"
        f"Target: {target_met}"
    )
    color = '#2ecc71' if best_m['accuracy'] >= 0.90 else '#f39c12'
    ax_best.text(0.05, 0.5, best_text, fontsize=9,
                 verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Top 3 models
    ax_top3 = fig3.add_subplot(gs[1, 2])
    ax_top3.axis('off')
    top3 = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
    top3_text = "TOP 3 MODELS\n" + "═" * 32 + "\n\n"
    for i, (name, m) in enumerate(top3, 1):
        top3_text += f"#{i}. {name}\n"
        top3_text += f"    Acc: {m['accuracy']*100:.2f}%\n"
        top3_text += f"    F1:  {m['f1']:.4f}\n\n"
    ax_top3.text(0.05, 0.5, top3_text, fontsize=9,
                 verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.2))

    plt.savefig('figure3_statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: figure3_statistical_summary.png")

    print("\n" + "=" * 80)
    print("📊 ALL FIGURES GENERATED:")
    print("  1. figure1_performance_comparison.png")
    print("  2. figure2_confusion_matrix_analysis.png")
    print("  3. figure3_statistical_summary.png")
    print("=" * 80)

# ============================================================================
# SECTION 12: ARTICLE SIMULATION
# ============================================================================

def simulate_article(text, vectorizer, model, label_names=None):
    """Classify a single new article and show confidence scores."""
    if label_names is None:
        label_names = {0: 'Real', 1: 'Fake', 2: 'Satire'}

    print(f"\n--- Article Simulation ---")
    print(f"Input: '{text}'")

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    proba_map = {label_names.get(i, str(i)): f"{p*100:.2f}%"
                 for i, p in enumerate(proba)}
    proba_sorted = dict(sorted(proba_map.items(),
                               key=lambda x: float(x[1].strip('%')),
                               reverse=True))

    print(f"Predicted: {label_names.get(pred, str(pred))}")
    print(f"Confidence: {proba_sorted}")
    print("-" * 30)
    return pred

# ============================================================================
# SECTION 13: MAIN PIPELINE
# ============================================================================

def main_pipeline(file_path, text_column='text', label_column='label'):
    """
    Complete enhanced hybrid fake news detection pipeline.

    All fixes applied:
    1. Dynamic file path
    2. Data leakage fix in Hybrid3
    3. SMOTE for class imbalance
    4. Reduced TF-IDF features (memory safe)
    5. Proper error handling
    """

    print("\n" + "=" * 80)
    print("🚀 ENHANCED HYBRID FAKE NEWS DETECTION PIPELINE")
    print("   Target: 85-90% Accuracy | 3-Class Urdu Classification")
    print("=" * 80)

    # ── Load and prepare data ──────────────────────────────────────────────
    df, num_classes = load_and_prepare_dataset(file_path, text_column, label_column)

    # ── Preprocess ────────────────────────────────────────────────────────
    preprocessor = EnhancedUrduPreprocessor()
    print("\n🔄 Preprocessing text...")
    df['processed'] = df['text'].apply(preprocessor.preprocess)
    df['minimal'] = df['text'].apply(preprocessor.preprocess_minimal)
    print("✓ Preprocessing complete")

    # ── Split ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df = create_splits(df)

    # ── Baseline: Naive Bayes ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BASELINE: Naive Bayes + TF-IDF")
    print("=" * 80)

    tfidf_baseline = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    X_train_base = tfidf_baseline.fit_transform(train_df['processed'])
    X_test_base = tfidf_baseline.transform(test_df['processed'])

    nb_model = MultinomialNB()
    nb_model.fit(X_train_base, train_df['label'].values)
    nb_pred = nb_model.predict(X_test_base)

    nb_acc = accuracy_score(test_df['label'].values, nb_pred)
    nb_f1 = f1_score(test_df['label'].values, nb_pred, average='macro', zero_division=0)
    print(f"\nBaseline NB: Acc={nb_acc*100:.2f}% | F1={nb_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_df['label'].values, nb_pred, zero_division=0))

    # ── Hybrid Model 1 ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HYBRID MODEL 1: Transformer Embeddings + Classifiers")
    print("=" * 80)

    h1 = HybridModel1(num_classes)
    h1.extract_features(train_df['minimal'], val_df['minimal'], test_df['minimal'])
    h1.train(train_df['label'].values, apply_smote=True)
    results_h1, preds_h1, probas_h1 = h1.evaluate(test_df['label'].values)

    # Save embeddings for H2 and H3
    train_emb = h1.train_emb.copy()
    val_emb = h1.val_emb.copy()
    test_emb = h1.test_emb.copy()
    clear_memory()

    # ── Hybrid Model 2 ────────────────────────────────────────────────────
    h2 = HybridModel2(num_classes)
    h2.prepare_features(
        train_df['processed'], val_df['processed'], test_df['processed'],
        train_emb, val_emb, test_emb
    )
    h2.train(train_df['label'].values, apply_smote=True)
    results_h2, pred_h2, proba_h2 = h2.evaluate(test_df['label'].values)
    clear_memory()

    # ── Hybrid Model 3 ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HYBRID MODEL 3: Weighted Stacking Ensemble")
    print("=" * 80)

    # Prepare TF-IDF for ensemble
    tfidf_ens = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 3),
        min_df=2, max_df=0.85, sublinear_tf=True
    )
    train_tfidf_raw = tfidf_ens.fit_transform(train_df['processed']).toarray()
    test_tfidf_raw = tfidf_ens.transform(test_df['processed']).toarray()

    scaler_tfidf = StandardScaler()
    train_tfidf = scaler_tfidf.fit_transform(train_tfidf_raw)
    test_tfidf = scaler_tfidf.transform(test_tfidf_raw)

    h3 = HybridModel3(num_classes)
    # FIX #2: Pass both train AND test tfidf correctly
    h3.prepare_data(train_emb, test_emb, train_tfidf, test_tfidf)
    h3.train(train_df['label'].values, apply_smote=True)
    results_h3, pred_h3, proba_h3 = h3.evaluate(test_df['label'].values)

    # ── Compile all results ───────────────────────────────────────────────
    all_results = {**results_h1, **results_h2, **results_h3}
    all_predictions = {**preds_h1, 'H2_Fusion': pred_h2, 'H3_Ensemble': pred_h3}
    all_probabilities = {**probas_h1, 'H2_Fusion': proba_h2, 'H3_Ensemble': proba_h3}

    # ── Final results table ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS — ALL MODELS")
    print("=" * 80)

    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True
    )

    for model_name, m in sorted_results:
        status = "✅" if m['accuracy'] >= 0.90 else "⚠️" if m['accuracy'] >= 0.80 else "❌"
        print(f"\n{status} {model_name}:")
        print(f"   Accuracy:  {m['accuracy']*100:.2f}%")
        print(f"   Precision: {m['precision']:.4f}")
        print(f"   Recall:    {m['recall']:.4f}")
        print(f"   F1-Score:  {m['f1']:.4f}")

    best_name = max(all_results, key=lambda x: all_results[x]['accuracy'])
    best = all_results[best_name]

    print("\n" + "=" * 80)
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"   Accuracy: {best['accuracy']*100:.2f}%")
    print(f"   F1-Score: {best['f1']:.4f}")
    if best['accuracy'] >= 0.90:
        print("✅ TARGET ACHIEVED: ≥90% Accuracy!")
    elif best['accuracy'] >= 0.80:
        print(f"⚠️  Close: {best['accuracy']*100:.2f}% (Target: 90%)")
    else:
        print(f"❌ Target not met. Achieved: {best['accuracy']*100:.2f}%")
    print("=" * 80)

    # ── Detailed prediction analysis ──────────────────────────────────────
    print("\n📊 Running detailed prediction analysis...")

    best_pred = all_predictions[best_name]
    best_proba = all_probabilities[best_name]

    results_df = analyze_predictions(
        test_df, test_df['label'].values,
        best_pred, best_proba,
        best_name, num_classes=num_classes, num_samples=5
    )
    error_analysis_by_class(results_df, num_classes=num_classes)
    export_predictions(results_df, best_name, f'predictions_best_{best_name}.csv')

    # Export all model predictions
    for model_name in all_predictions:
        pred = all_predictions[model_name]
        proba = all_probabilities[model_name]
        df_pred = analyze_predictions(
            test_df, test_df['label'].values, pred, proba,
            model_name, num_classes=num_classes, num_samples=3
        )
        safe = model_name.replace(' ', '_').lower()
        export_predictions(df_pred, model_name, f'predictions_{safe}.csv')

    # ── Article simulation ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🎯 ARTICLE SIMULATION (using Baseline NB model)")
    print("=" * 80)

    articles = [
        "پاکستان کے وزیر خارجہ نے اقوام متحدہ میں کشمیر پر اہم قرارداد پیش کر دی ہے۔",
        "کرکٹ ٹیم کے کپتان نے اعلان کیا ہے کہ وہ اب صرف سونا اور ہیرے کھائیں گے۔",
        "حکومت نے نئی تعلیمی پالیسی کا اعلان کیا جس میں طلباء کو مفت کتابیں ملیں گی۔"
    ]
    for article in articles:
        simulate_article(article, tfidf_baseline, nb_model)

    # ── Visualizations ────────────────────────────────────────────────────
    print("\n📊 Generating research paper visualizations...")
    create_all_visualizations(
        all_results, test_df['label'].values,
        all_predictions, num_classes
    )

    # ── Results DataFrame ─────────────────────────────────────────────────
    results_df_final = pd.DataFrame(all_results).T.round(4)
    results_df_final.index.name = 'Model'
    results_df_final['rank'] = results_df_final['f1'].rank(
        ascending=False
    ).astype(int)
    results_df_final = results_df_final.sort_values('rank')

    print("\n" + "=" * 80)
    print("📈 FINAL SUMMARY TABLE")
    print("=" * 80)
    print(results_df_final.to_string())

    return {
        'results': all_results,
        'best_model': (best_name, best),
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'results_df': results_df_final
    }

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':

    # ── FILE PATH CONFIGURATION ────────────────────────────────────────────
    # The script tries these in order:
    # 1. KAGGLE_PATH  — set this if running on Kaggle
    # 2. COLAB_PATH   — set this if running on Google Colab
    # 3. LOCAL_PATH   — set this if running locally
    # Leave as None any paths that don't apply to you.

    KAGGLE_PATH = '/kaggle/input/datasets/uzairmoazzam203/research-paper/ml_ready_2column.csv'
    COLAB_PATH  = None   # e.g. '/content/ml_ready_2column.csv'
    LOCAL_PATH  = None   # e.g. '/home/user/data/ml_ready_2column.csv'

    TEXT_COLUMN  = 'text'
    LABEL_COLUMN = 'label'

    # Auto-select the first valid path
    FILE_PATH = None
    for candidate in [KAGGLE_PATH, COLAB_PATH, LOCAL_PATH]:
        if candidate and os.path.exists(candidate):
            FILE_PATH = candidate
            print(f"✓ Using path: {FILE_PATH}")
            break

    # Fallback: search common locations automatically
    if FILE_PATH is None:
        FILE_PATH = find_dataset('ml_ready_2column.csv')

    if FILE_PATH is None:
        print("\n" + "=" * 60)
        print("⚠️  MANUAL PATH REQUIRED")
        print("=" * 60)
        print("Edit one of these lines at the top of the execution section:")
        print("  KAGGLE_PATH = '/kaggle/input/.../ml_ready_2column.csv'")
        print("  COLAB_PATH  = '/content/ml_ready_2column.csv'")
        print("  LOCAL_PATH  = '/your/local/path/ml_ready_2column.csv'")
        print("\nFor Colab, upload first:")
        print("  from google.colab import files")
        print("  files.upload()")
        print("  COLAB_PATH = 'ml_ready_2column.csv'")
    else:
        results = main_pipeline(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)

        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 80)
        print("\n📁 Generated Files:")
        print("  Predictions: predictions_*.csv")
        print("  Figures:     figure1_*.png, figure2_*.png, figure3_*.png")
        print("=" * 80)