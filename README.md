
# Kubeflow Spam Detection Pipeline

This repository showcases a **Kubeflow Pipeline** for training and evaluating a spam-detection model using **TF-IDF features** and a **RandomForest classifier**.  
It was built as part of my exploration into Kubeflow, focusing on how to compose pipelines from **custom containerised components**.  
This project demonstrates how the entire machine-learning workflow can be orchestrated end-to-end using Kubeflow.

---

## 🚀 Quick Overview

- **Goal**: Build an end-to-end ML pipeline on Kubeflow for SMS spam detection  
- **Pipeline Steps**:
  1. Data Ingestion  
  2. Preprocessing (cleaning, encoding, stemming)  
  3. Feature Engineering (TF-IDF)  
  4. Model Training (RandomForest)  
  5. Evaluation (Accuracy, Precision, Recall, AUC)  
- **Tech Used**: Kubeflow, Docker, Python, Scikit-learn, Pandas  
- **Why It Matters**: Demonstrates containerised ML workflows, parameterisation, and reproducibility

👉👉 At a glance, this repo shows how ML pipelines can be structured on Kubeflow, with containerised components and reproducible metric tracking.
---

## 📂 Repository Structure

| Path | Description |
|------|-------------|
| `pipeline.py` | Kubeflow DSL pipeline definition, compiles to `spam_detection_pipeline.yaml`. |
| `spam_detection_pipeline.yaml` | Compiled pipeline spec, ready for Kubeflow. |
| `params.yaml` | Hyperparameters (train/test split, TF-IDF, RandomForest settings). |
| `components/` | Contains code + Dockerfiles for each pipeline stage. |

Key Components:
- `data-ingestion/ingest.py` – dataset loading & splitting  
- `data-preprocessing/preprocess.py` – text cleaning & label encoding  
- `feature-engineering/feature_engineering.py` – TF-IDF feature extraction  
- `train-model/model_training.py` – RandomForest training  
- `evaluate-model/model_evaluation.py` – metrics computation  

---

## ▶️ Running the Pipeline

1. **Clone Repo**
   ```bash
   git clone https://github.com/PrakashD2003/Kubeflow-Study.git
   cd Kubeflow-Study


2. **Build & Push Images**

   ```bash
   docker build -f components/data-ingestion/Dockerfile.ingestion -t <registry>/kubeflow-ingest:latest .
   docker push <registry>/kubeflow-ingest:latest
   ```

   *(repeat for preprocess, feature-engineering, train-model, evaluate-model)*

3. **Configure Parameters**
   Edit `params.yaml` → adjust `test_size`, `max_features`, `n_estimators`.

4. **Compile Pipeline**

   ```bash
   python pipeline.py
   ```

   → generates `spam_detection_pipeline.yaml`

5. **Deploy to Kubeflow**

   * Upload via Kubeflow UI
   * Or use SDK:

     ```python
     import kfp
     client = kfp.Client()
     client.create_run_from_pipeline_package('spam_detection_pipeline.yaml')
     ```

---

## 📊 Key Learnings

* Writing container components with Kubeflow DSL
* Orchestrating end-to-end ML workflows
* NLP preprocessing (cleaning, stemming, TF-IDF)
* Training & evaluating RandomForest models
* YAML-driven hyperparameter tuning

---

## 🔬 Detailed Pipeline Breakdown

### 1. Data Ingestion

* Loads SMS dataset → drops unused cols → renames to `target` & `text`
* Splits into train/test (`test_size` from `params.yaml`)
* Saves CSVs as artifacts

### 2. Preprocessing

* Label encoding of target (`spam` / `ham`)
* Removes duplicates
* Cleans text (lowercase, stopword removal, punctuation stripping, stemming)

### 3. Feature Engineering

* Applies `TfidfVectorizer` with `max_features` from `params.yaml`
* Outputs numerical feature vectors + labels

### 4. Model Training

* Trains `RandomForestClassifier`
* Hyperparams: `n_estimators`, `random_state` from config
* Saves `model.pkl`

### 5. Model Evaluation

* Loads trained model + test set
* Computes **Accuracy, Precision, Recall, AUC**
* Saves metrics as JSON artifact

---

## 🔮 Future Improvements

* Try alternative models (SVM, neural networks) + hyper-parameter tuning
* Add **data versioning & model registry** for governance
* Extend to cross-validation & model comparison
* Package as reusable Kubeflow component for other text classification tasks

---

💡 *If you have suggestions, feel free to open an issue or connect with me.*
🙏 Thanks for reviewing my Kubeflow pipeline project!





