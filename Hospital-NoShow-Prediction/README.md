# 🏥 Hospital Appointment No-Show Prediction
> *Kaggle Datathon Project: Modeling Human Behavior with Data-Centric AI & AutoML*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![AutoGluon](https://img.shields.io/badge/AutoGluon-OOM__Safe-orange?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-Optimized-yellow?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensembled-red?style=for-the-badge)
![Metric](https://img.shields.io/badge/Metric-PR--AUC-success?style=for-the-badge)
![Score](https://img.shields.io/badge/Final_Score-0.5149-brightgreen?style=for-the-badge)

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline developed for a Datathon competition to predict the probability of patients missing their hospital appointments (`label_noshow`).

Predicting human behavior is inherently noisy. Instead of blindly stacking complex algorithms, this project adopts a **Data-Centric AI** philosophy. We tackled extreme class imbalance, prevented Kaggle out-of-memory (OOM) crashes, and utilized advanced AutoML frameworks combined with smart rank ensembling techniques.

---

## 🎯 Competition Context & Evaluation
| Criteria | Description |
| :--- | :--- |
| **Target Variable** | `label_noshow` (Binary Classification: 1 = No-show, 0 = Show) |
| **Class Imbalance** | High (~20-25% No-show rate) |
| **Evaluation Metric** | **PR-AUC (Precision-Recall Area Under Curve)**. Chosen specifically over ROC-AUC to maximize precision on the minority class due to the imbalanced nature of the dataset. |
| **Final Score** | `0.5149` (Achieved via AutoGluon & Rank Ensembling) |

---

## 🧠 Key Strategies & Insights

During the competition, we discovered that deep decision trees and over-engineered demographic features pushed the models into severe overfitting. Our final, successful strategy was built on four core pillars:

### 1. Bayesian Historical Fusion (Leakage-Free)
The most powerful predictor of future behavior is past behavior. When calculating historical no-show rates, we strictly prevented **Data Leakage** by sorting patients chronologically. Furthermore, to solve the "Cold Start" problem for new patients, we implemented **Bayesian Laplace Smoothing**:
`bayesian_noshow_rate = (total_prior_noshows + 1) / (total_prior_appts + 2)`
This mathematically prevented new patients from defaulting to a naive 0.0 risk score.

### 2. Signal Hunting: Logistics over Demographics
Feature importance analysis revealed that logistical factors heavily outweighed demographics. We narrowed the model's focus to the core signals:
* `lead_time_hours`: The forgetfulness factor.
* `wait_mins_est`: The frustration factor.
* `distance_km`: Logistical difficulty.

### 3. OOM-Safe AutoML (AutoGluon) Architecture


To find the perfect ensemble without manual tuning, we deployed **AutoGluon**. However, standard AutoML often crashes Kaggle kernels (30GB RAM limit). We engineered an **"OOM Shield"**:
* **Memory Reduction Script:** Downcasted `float64` to `float32` and `int64` to `int8/int16`, reducing dataframe memory usage by ~70%.
* **Model Restriction:** Explicitly disabled RAM-heavy algorithms like Random Forest (RF), Extra Trees (XT), and KNN.
* **Boosting-Only Ensembles:** Forced AutoGluon to build meta-ensembles exclusively using memory-efficient modern algorithms (CatBoost, LightGBM, XGBoost).

### 4. Smart Blending (Rank Ensembling) & Heuristic Override


Instead of directly averaging the probability outputs of our top pipelines, which can distort distributions, we blended their risk rankings **(Rank Ensembling)** with a 60/40 ratio. Furthermore, extreme cases that the machine hesitated on (e.g., chronic no-shows with 3+ missed appointments and a 100% fail rate) were manually overridden to a 0.99 probability, maximizing the PR-AUC score.

---

## ⚙️ How to Run Locally

To replicate the results or explore the pipeline on your local machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Hospital-NoShow-Prediction.git](https://github.com/YOUR_USERNAME/Hospital-NoShow-Prediction.git)
   cd Hospital-NoShow-Prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   # Ensure AutoGluon is installed: pip install autogluon
   ```

3. Place the downloaded Kaggle datasets into the `data/` folder.

4. Execute the OOM-Safe AutoGluon pipeline:
   ```bash
   python src/train_autogluon.py
   ```

5. Generate the highest-scoring ensemble submission:
   ```bash
   python src/rank_ensemble.py
   ```

---

## 👨‍💻 Author

**Emirhan** | *Data Science & Analytics*
**Taha Yasin** | *Data Researcher & Analytics*
**Hasan** | *Data Science & Analytics*

*If you found this project insightful, consider leaving a ⭐ star!*
