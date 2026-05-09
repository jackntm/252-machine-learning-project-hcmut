# Adult Census Income Prediction - Machine Learning Project

## Course Information
* **Course Name:** Machine Learning
* **Course Code:** CO3117
* **Semester:** Semester 252
* **Academic Year:** 2025 - 2026
* **Institution:** Ho Chi Minh City University of Technology (HCMUT)
* **Instructor:** Truong Vinh Lan

## Group Members
| Full Name | Student ID | 
| :--- | :--- | 
| Phuong Xuong Duc | 2352270 | 
| Nguyen Tien Minh | 2352755 | 
| Tan Khanh Phong | 2352911 | 
| Nguyen Bui Huynh Khuong | 2352637 |
| Ho Hong Phuc Nguyen | 2352824 | 

## Project Objectives
The primary goal of this project is to build a robust machine learning system to predict individual income levels (<=50K or >50K) using the Adult Census Income dataset. The objectives include:
1.  **Exploratory Data Analysis (EDA):** Identifying demographic patterns, missing value distributions, and feature correlations.
2.  **Traditional ML Pipeline (Branch A):** Implementing a complete pipeline featuring Preprocessing, Dimensionality Reduction via PCA (retaining 95% variance), and evaluating Logistic Regression, SVM, and Random Forest across 4 experimental scenarios.
3.  **Deep Learning (Branch B):** Training a Multi-layer Perceptron (MLP) directly on the full preprocessed feature space to capture complex non-linear relationships.
4.  **Performance Optimization:** Analyzing accuracy, ROC-AUC, and F1-Score to address the class imbalance inherent in census data.

## Data Loading 
The dataset used is the [Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) from the UCI Machine Learning Repository. The script automatically handles the download and initial cleaning of the raw `adult.csv` file.

## Instructions for Running the Notebook

### 1. Required Libraries
Ensure your environment supports the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
```

## Project Structure 
```bash
├── notebooks/
│   └── notebook_assignment_machine_learning.ipynb      # Main execution notebook
├── reports/
│   └── ML_Assignment_Report.pdf # Comprehensive project report
├── features/                    # Extracted feature files (.npy)
│   ├── X_train_preprocessed.npy
│   ├── X_train_pca.npy
│   ├── ...
│   └── models/                  # Serialized models and transformers
│       ├── preprocessor.pkl
│       ├── pca_transformer.pkl
│       ├── random_forest_best.pkl
│       └── mlp_model.keras
└── README.md                    # Project documentation                 # Project documentation
```

## Links
* [Open In Colab](https://colab.research.google.com/drive/1BtPxZyhX83796O-LV8zSvcojC6js6_AJ?usp=sharing)
* [PDF Report](reports/Group6_CC01_HCMUT.pdf)

## Configuration & Hyperparameters

### 1. General Configuration
* **`test_size`**: 20% of data used for testing.
* **`random_state`**: Fixed at 42 for reproducibility across all splits and model initializations.

### 2. Preprocessing & Feature Engineering
* **Feature Selection**: Dropped `fnlwgt` and `education` (redundant with `education-num`).
* **Scaling**: Applied `StandardScaler` to normalize numerical features.
* **Encoding**: Utilized `OneHotEncoder` for categorical variables, resulting in an 88-dimensional space.
* **PCA**: Retained 95% variance, reducing the feature space from 88 to 24 components for Branch A models.

### 3. Model Optimizers & Training
#### Traditional ML (Branch A)
Models were tested across 4 scenarios: Baseline, High Capacity, Strong Regularization, and Cost-Sensitive.
* **Logistic Regression**: C=1.0, `max_iter=1000`.
* **SVM**: RBF kernel, C=1.0, `gamma='scale'`.
* **Random Forest**: 150 estimators, `max_depth=16`.
* **Class Weighting**: Scenario 4 utilized `class_weight='balanced'` to improve F1-Score on the minority (>50K) class.

#### Deep Learning (Branch B - MLP)
* **Architecture**: 3 hidden layers [128, 64, 32] with Batch Normalization and Dropout [0.4, 0.3, 0.2].
* **Optimizer**: **Adam** with an initial learning rate of `1e-3`.
* **Callbacks**:
    * **EarlyStopping**: Patience of 10 epochs monitoring `val_loss`.
    * **ReduceLROnPlateau**: Factor of 0.5 with patience of 5 epochs to fine-tune convergence.

## Results Summary
The Multi-layer Perceptron (MLP) demonstrated the best overall performance:
* **Accuracy**: ~86.16%
* **ROC-AUC**: ~0.9184
* **F1-Score (>50K)**: ~0.6984

PCA proved highly effective, reducing training time by 79% for traditional models with a negligible accuracy loss of less than 0.5%.
