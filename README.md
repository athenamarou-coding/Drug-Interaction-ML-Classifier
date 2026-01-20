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

#### 1. Evaluation (Train Mode)
To evaluate the model's accuracy (e.g., with a cutoff of 20 samples):
```bash

python PROJECT_1_DDI-PREDICTION-ML.py \
  --molecular_feats [path_to_pkl] \
  --relation2id [path_to_json] \
  --train [path_to_train_txt] \
  --test [path_to_test_txt] \
  --cutoff 20 \
  --mode train
```


#### 2. Prediction (Inference Mode)
```bash
python PROJECT_1_DDI-PREDICTION-ML.py \
  --molecular_feats [path_to_pkl] \
  --relation2id [path_to_json] \
  --train [path_to_train_txt] \
  --mode inference \
  --drugbank_1 DB13231 \
  --drugbank_2 DB00244
```


### 📊 Sample Output (Evaluation Mode)
When running the model in `train` mode with a cutoff of 20 samples, the expected output is as follows:

```text
Sample 1/20: Predicted 48, Real 48
Sample 2/20: Predicted 72, Real 72
Sample 3/20: Predicted 66, Real 66
Sample 4/20: Predicted 48, Real 48
Sample 5/20: Predicted 48, Real 48
Sample 6/20: Predicted 48, Real 1
Sample 7/20: Predicted 48, Real 48
Sample 8/20: Predicted 66, Real 66
Sample 9/20: Predicted 1, Real 1
Sample 10/20: Predicted 46, Real 48
Sample 11/20: Predicted 1, Real 1
Sample 12/20: Predicted 66, Real 66
Sample 13/20: Predicted 66, Real 66
Sample 14/20: Predicted 48, Real 48
Sample 15/20: Predicted 66, Real 66
Sample 16/20: Predicted 48, Real 48
Sample 17/20: Predicted 72, Real 72
Sample 18/20: Predicted 74, Real 72
Sample 19/20: Predicted 48, Real 48
Sample 20/20: Predicted 1, Real 1

FINAL ACCURACY FOR 20 samples, correct: 17, accuracy: 0.85
