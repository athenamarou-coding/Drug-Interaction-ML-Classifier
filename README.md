# Drug-Drug Interaction (DDI) Prediction Model 💊

### 🎓 MSc in Bioinformatics Project
This project was developed as part of the **"Programming in Python"** course during my **Master’s in Bioinformatics**. It implements a computational pipeline to predict potential interactions between drug pairs using molecular and physicochemical data.

### 🔍 Project Overview
The model identifies the most likely interaction type between two drugs by comparing their profiles against a known training dataset derived from **DrugBank**.

- **Feature Extraction:** Utilizes Morgan fingerprints (structural signatures) and RDKit2D features (physicochemical properties).
- **Similarity Metrics:** Employs **Tanimoto Distance** for structural similarity and **Cosine Distance** for property-based similarity.
- **Algorithm:** Implements a **K-Nearest Neighbors (KNN)** approach for classification.
- **Modes of Operation:**
    - **Evaluation Mode:** Calculates model accuracy on benchmark test datasets (S0, S1, S2).
    - **Inference Mode:** Predicts the interaction between two specific DrugBank IDs and retrieves the full interaction description from a JSON mapping.

### 🛠 Tech Stack
- **Language:** Python 3.x
- **Libraries:** NumPy, SciPy (Distance metrics), Scikit-learn (Evaluation), JSON, Pickle.

### 🚀 Usage Instructions

#### 1. Evaluation (Train/Test)
To evaluate the model's accuracy (e.g., with a cutoff of 20 samples):
```bash
python PROJECT_1_DDI-PREDICTION-ML.py --mode train --molecular_feats [path_to_pkl] --train [path_to_train_txt] --test [path_to_test_txt] --cutoff 20
