# Cinema.io: Personalized Movie Recommendation via Multi-Emotion NLP

**Project Name:** Cinema.io  
**Academic Context:** NLP Lab Final Project | LA01 | Group 23 | 2025 - 2026 4th Semester
**Live Demo:** https://finpronlp-cinemaio.streamlit.app/

---

## 📋 Project Overview

Cinema.io is an end-to-end machine learning pipeline that recommends movies based on your emotional state. The system classifies user emotional input using advanced NLP models and matches them with semantically aligned films from the IMDb top 1000 dataset.

**Core Flow:**

1. User inputs emotional text (e.g., "I'm feeling sad and nostalgic")
2. DistilBERT (fine-tuned) classifies the emotion (Joy, Love, Surprise, Anger, Fear, or Sadness)
3. System finds movies with matching emotion tags
4. TF-IDF + Cosine Similarity ranks movies by semantic alignment with user input
5. Top 5 recommendations displayed with metadata

---

## 🚀 Quick Start

### Option 1: Online (Recommended for Quick Testing)

Visit the deployed app at: https://finpronlp-cinemaio.streamlit.app/

No installation required—just open the link and start using the recommendation system!

### Option 2: Run Locally

#### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

#### Installation Steps

1. **Clone/Download the repository:**

   ```bash
   cd /path/to/FINPRO2
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

5. **Access the app:**
   - The app will automatically open in your browser
   - If not, navigate to: `http://localhost:8501`

#### What Gets Downloaded

- Transformer model (DistilBERT) from HuggingFace (~300MB on first run)
- Pre-trained embeddings and tokenizers
- IMDb movie dataset (~500KB)

---

## 📊 Datasets

### 1. GoEmotions Training Corpus

**Purpose:** Train emotion classification models

**Source:** Reddit community posts  
**Files:** `dataset_raw/goemotions_1.csv`, `goemotions_2.csv`, `goemotions_3.csv`  
**Processing Output:** `data/goemotions_kasar_pure.csv`

**Data Statistics:**

- Raw entries: ~43,000+
- After filtering to 6 core emotions: ~11,000
- Final training corpus: **~8,800 entries**

**Class Distribution:**
| Emotion | Count | Percentage |
|---------|-------|-----------|
| Joy | ~3,100 | 35% |
| Sadness | ~2,400 | 27% |
| Anger | ~1,900 | 22% |
| Fear | ~800 | 9% |
| Surprise | ~400 | 4% |
| Love | ~200 | 2% |

**Processing Pipeline:**

- Multi-label → Single-label conversion (priority-based hierarchy)
- Removal of neutral/non-core emotions
- Class imbalance management through strategic filtering
- Train/Test split: 80/20 (stratified)

### 2. IMDb Top 1000 Movie Dataset

**Purpose:** Movie metadata and semantic content for recommendations

**Source:** `dataset_raw/updated_imdb_top_1000.csv`  
**Output:** `data/imdb_movies_with_emotions.csv` (with self-labeled emotion tags)

**Dataset Features:**

- ~1,000 high-quality movies
- Columns: Title, Genre, Overview
- All movies have complete synopses
- Diverse genre representation

**Processing:**

- Feature selection (Title, Genre, Overview)
- Removal of entries with missing overviews
- Manual emotion annotation per movie
- Used for semantic similarity matching

---

## 🛠️ Tech Stack

### Core Framework

| Component            | Library                  | Version |
| -------------------- | ------------------------ | ------- |
| **Web App**          | Streamlit                | ≥1.35.0 |
| **ML Models**        | PyTorch                  | ≥2.0.0  |
| **NLP/Transformers** | HuggingFace Transformers | ≥4.38.0 |
| **ML/Utilities**     | Scikit-learn             | ≥1.3.0  |

### Data Processing

| Component               | Library       | Version |
| ----------------------- | ------------- | ------- |
| **Data Manipulation**   | Pandas        | ≥2.0.0  |
| **Numerical Computing** | NumPy         | ≥1.24.0 |
| **Tokenizers**          | SentencePiece | ≥0.1.99 |

### Pre-Trained Models

- **DistilBERT:** Efficient transformer for emotion classification
- **DistilBERT-Optuna:** Hyperparameter-tuned variant
- **BERT-Base:** Full-size transformer (for comparison)
- **BiLSTM:** Recurrent neural network baseline
- **SVM:** Support Vector Machine with TF-IDF features

---

## 🔄 Complete ML Pipeline

### Phase 1: Data Preprocessing

**Notebooks:** `preprocessing/`

- `filter_goemotion.ipynb` → Extract 6 emotions, handle multi-labels
- `filter_imdb.ipynb` → Extract movie metadata
- `preprocessing_data.ipynb` → Text normalization (minimal to preserve structure)

**Key Decision:** Minimal preprocessing (no stopword removal, lemmatization, or stemming) to preserve syntactic structure for bidirectional transformer attention.

### Phase 2: Data Splitting

**Notebook:** `splitter/01_data_splitter.ipynb`

- Stratified train/test split (80:20)
- Outputs: `data_train_master.csv`, `data_test_master.csv`
- Maintains emotion distribution across splits

### Phase 3: Model Training

**Training Notebooks:**

1. `training/02_train_svm.ipynb` → SVM with TF-IDF
2. `training/03_train_bilstm.ipynb` → BiLSTM neural network
3. `training/04_train_distilbert.ipynb` → DistilBERT baseline
4. `training/05_train_bertbase.ipynb` → BERT-Base
5. `tuning_model/07_tuning_distil.ipynb` → DistilBERT + Optuna hyperparameter tuning
6. `tuning_model/08_validation_performance.ipynb` → Validation and performance metrics

**Saved Models:** `preprocessing/models/`

- `distilbert/` → Best performing model
- `bert-base/`
- `bilstm/`
- `svm/`

### Phase 4: Model Evaluation

**Notebook:** `evaluation/06_evaluation.ipynb`

- Metrics: Accuracy, Precision, Recall, F1-Score
- Emotion-specific performance analysis
- Confusion matrix and classification reports

### Phase 5: Movie Self-Labeling

**Notebook:** `self_labeling/09_movie_self_labeling.ipynb`

- Uses trained emotion classifier
- Assigns emotion tags to IMDb movie overviews
- Outputs: `data/imdb_movies_with_emotions.csv`

### Phase 6: Recommendation System Development

**Notebook:** `recommendation_system/10_final_recommendation_system.ipynb`

- TF-IDF vectorization (15K features)
- Cosine similarity ranking
- Top-5 movie filtering by emotion match

---

## 🤖 Models Comparison

### Model Performance Summary

| Model                 | Architecture                        | Training Data | Key Features                                    |
| --------------------- | ----------------------------------- | ------------- | ----------------------------------------------- |
| **SVM**               | TF-IDF + LinearSVC                  | GoEmotions    | Fast inference, simple baseline                 |
| **BiLSTM**            | Embedding→BiLSTM→Dense              | GoEmotions    | Captures sequential patterns                    |
| **DistilBERT**        | Transformer (Distilled)             | GoEmotions    | Efficient, bidirectional, 40% smaller than BERT |
| **BERT-Base**         | Full Transformer                    | GoEmotions    | Highest accuracy potential, slower              |
| **DistilBERT-Optuna** | Transformer + Hyperparameter Tuning | GoEmotions    | Optimized DistilBERT variant                    |

### Selected Model: DistilBERT Baseline ✅

- **Why:** Best balance of accuracy, speed, and resource efficiency
- **Advantages:** 40% smaller than BERT, 60% faster inference, excellent emotion classification
- **Deployment:** Loaded via HuggingFace pipeline for real-time inference

---

## 🎬 Recommendation Algorithm

### Step 1: Emotion Classification

```
User Input (Text) → DistilBERT → Predicted Emotion (6 classes)
```

### Step 2: Movie Filtering

```
All 1000 IMDb Movies → Filter by Predicted Emotion → Subset of 50-200 movies
```

### Step 3: Semantic Matching

```
TF-IDF Vectorization (user input + movie overview) → Cosine Similarity
```

### Step 4: Ranking & Display

```
Top-5 movies ranked by similarity score → Display with metadata
```

## 📁 Project Structure

```
FINPRO2/
├── app.py                          # Streamlit deployment script
├── requirements.txt                 # Python dependencies
├── README.md / readme2.md          # Documentation
│
├── data/                            # Processed datasets
│   ├── data_train_master.csv       # Training data (80%)
│   ├── data_test_master.csv        # Test data (20%)
│   ├── goemotions_kasar_pure.csv   # Processed emotions
│   └── imdb_movies_with_emotions.csv # Movies with emotion tags
│
├── dataset_raw/                     # Raw source data
│   ├── goemotions_*.csv            # Raw emotion data
│   └── updated_imdb_top_1000.csv   # Raw movie data
│
├── preprocessing/
│   ├── filter_goemotion.ipynb      # Process GoEmotions
│   ├── filter_imdb.ipynb           # Process IMDb data
│   ├── preprocessing_data.ipynb    # Text normalization
│   └── models/                      # Pre-trained models
│       ├── distilbert/
│       ├── bert-base/
│       ├── bilstm/
│       └── svm/
│
├── splitter/
│   └── 01_data_splitter.ipynb      # Train/test split (80/20)
│
├── training/
│   ├── 02_train_svm.ipynb
│   ├── 03_train_bilstm.ipynb
│   ├── 04_train_distilbert.ipynb
│   └── 05_train_bertbase.ipynb
│
├── tuning_model/
│   ├── 07_tuning_distil.ipynb      # Hyperparameter tuning
│   └── 08_validation_performance.ipynb
│
├── evaluation/
│   └── 06_evaluation.ipynb         # Model performance metrics
│
├── self_labeling/
│   └── 09_movie_self_labeling.ipynb # Emotion labeling of movies
│
└── recommendation_system/
    └── 10_final_recommendation_system.ipynb # Recommendation logic
```

---

## 🔧 System Requirements

### Minimum Requirements

- **CPU:** Dual-core processor (4 cores recommended)
- **RAM:** 8GB (16GB recommended)
- **Storage:** 2GB (for models + datasets)
- **Internet:** Required for first-time model download

### Optional: GPU Support

- **NVIDIA CUDA 11.8+** for faster inference
- PyTorch will automatically use GPU if available

---


## 🌐 Deployment

### Online Version

**URL:** https://finpronlp-cinemaio.streamlit.app/

**Hosting:** Streamlit Cloud

- Automatic updates from repository
- Free tier with 1GB RAM
- No local setup required


### Web Interface

1. Enter your emotional state in the text box
2. Click "Get Movie Recommendations"
3. View top 5 recommended movies

## 📝 Preprocessing Details

### Why Minimal Preprocessing?

- **Transformers benefit from syntactic structure** (bidirectional attention)
- **Negation matters:** "not good" ≠ "good"
- **Fair model comparison:** consistent input across SVM, RNN, and Transformers
- **Better semantic preservation:** word variations capture nuanced emotions

---
---

**Last Updated:** 2026-06-16  
**Status:** Production Ready ✅
