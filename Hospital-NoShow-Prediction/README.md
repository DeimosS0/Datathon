<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=250&section=header&text=Hospital%20No-Show%20Prediction&fontSize=50&animation=fadeIn&fontAlignY=38&desc=Predicting%20Human%20Behavior%20with%20Data-Centric%20AI&descAlignY=55&descAlign=50" alt="Header Banner" />
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/AutoGluon-OOM__Safe-FF6F00?style=for-the-badge&logo=jupyter&logoColor=white" alt="AutoGluon"></a>
  <a href="#"><img src="https://img.shields.io/badge/CatBoost-Optimized-yellow?style=for-the-badge&logo=cat&logoColor=black" alt="CatBoost"></a>
  <a href="#"><img src="https://img.shields.io/badge/XGBoost-Ensembled-red?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"></a>
  <a href="#"><img src="https://img.shields.io/badge/Metric-PR--AUC-success?style=for-the-badge&logo=chartdotnet&logoColor=white" alt="Metric"></a>
</p>

<br>

> **Warning** > *Predicting human behavior is inherently noisy. This project completely abandons the brute-force "stacking everything" approach. Instead, it relies on strict Leakage-Free Feature Engineering, OOM-Safe AutoML, and Rank Ensembling.*

---

## 🏆 The Grandmaster Scoreboard
| Metric | Final Score | Architecture Highlights |
| :---: | :---: | :--- |
| **PR-AUC** | <kbd>0.5149</kbd> | *Bayesian Fusion, CatBoost + XGBoost, Rank Blend, Heuristic Override* |

*Note: In an extremely imbalanced dataset (~20% No-show), PR-AUC was strictly optimized over ROC-AUC to maximize minority-class precision.*

---

## 🧠 Deep Dive: The "Secret Sauce" Architecture

<details>
<summary><b>1️⃣ Bayesian Historical Fusion (Leakage-Free)</b> <i>[Click to expand]</i></summary>
<br>
The most powerful predictor of future behavior is past behavior. However, traditional historical averages cause massive <b>Data Leakage</b> and fail on new patients (Cold Start). 

We engineered a chronologically strictly-shifted cumulative history and applied **Bayesian Laplace Smoothing**:
<p align="center">
<code>bayesian_noshow_rate = (total_prior_noshows + 1) / (total_prior_appts + 2)</code>
</p>
This mathematically assigns a "reasonable doubt" to new patients rather than a naive 0.0 risk score.
</details>

<details>
<summary><b>2️⃣ Signal Hunting: Logistics over Demographics</b> <i>[Click to expand]</i></summary>
<br>
Through iterative Feature Importance analysis, we discovered that tree-based models choked on demographic noise (age, socio-economic status). We pivoted to a <b>Sniper Approach</b>, isolating the top behavioral signals:
<ul>
  <li>⏱️ <code>lead_time_hours</code>: The forgetfulness factor.</li>
  <li>⏳ <code>wait_mins_est</code>: The clinical frustration factor.</li>
  <li>📍 <code>distance_km</code>: Logistical friction.</li>
  <li>📱 <code>sms_lead_hours</code>: The exact timing/impact of reminders.</li>
</ul>
</details>

<details>
<summary><b>3️⃣ OOM-Safe AutoML (AutoGluon) Shield</b> <i>[Click to expand]</i></summary>
<br>
Standard AutoML often crashes 30GB Kaggle kernels. We engineered an <b>"OOM Shield"</b> for AutoGluon:
<ul>
  <li>📉 <b>Memory Compression:</b> Downcasted <code>float64</code> to <code>float32</code> and <code>int64</code> to <code>int8/int16</code> (70% memory reduction).</li>
  <li>🚫 <b>Algorithm Restriction:</b> Explicitly disabled RAM-heavy architectures (Random Forest, Extra Trees, KNN, FastAI).</li>
  <li>⚡ <b>Boosting-Only:</b> Forced meta-ensembles to exclusively use memory-efficient modern algorithms (CatBoost, LightGBM, XGBoost).</li>
</ul>
</details>

<details>
<summary><b>4️⃣ Smart Blending (Rank Ensembling) & Heuristic Overrides</b> <i>[Click to expand]</i></summary>
<br>
Averaging probabilities distorts underlying distributions. Instead, we blended the <b>risk rankings</b> of our top pipelines with a <code>60/40</code> ratio. 

Finally, extreme cases that the machine hesitated on (e.g., chronic no-shows with 3+ missed appointments and a 100% fail rate) were manually overridden to a <b>0.99 probability</b>, maximizing the precision peak on the PR-AUC curve.
</details>

---

## 🚀 Quick Start (How to Run Locally)

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/Hospital-NoShow-Prediction.git](https://github.com/YOUR_USERNAME/Hospital-NoShow-Prediction.git)
cd Hospital-NoShow-Prediction
```

**2. Install requirements**
```bash
pip install -r requirements.txt
# Requires AutoGluon: pip install autogluon
```

**3. Run the OOM-Safe Training Pipeline**
```bash
python src/train_autogluon.py
```

**4. Execute Rank Ensemble & Generate Submission**
```bash
python src/rank_ensemble.py
```

---

## 📂 Repository Structure

```text
📦 Hospital-NoShow-Prediction
 ┣ 📂 data                  # Raw Kaggle CSV files (Ignored in Git)
 ┣ 📂 notebooks             # EDA, Feature Importance & R&D
 ┣ 📂 src                   
 ┃ ┣ 📜 feature_engineering.py  # Bayesian logic & data compression
 ┃ ┣ 📜 train_autogluon.py      # OOM-Safe AutoML pipeline
 ┃ ┗ 📜 rank_ensemble.py        # Ranking, Blending & Override logic
 ┣ 📂 submissions           # Final generated CSVs
 ┣ 📜 requirements.txt      # Dependency list
 ┗ 📜 README.md             # You are reading this!
```

---
<br>

<h3 align="center">👨‍💻 The Team</h3>

<p align="center">
  <i>Architected with 💡 by:</i>
</p>

<p align="center">
  <b>Emirhan</b> | <i>Data Science & Analytics</i><br>
  <a href="https://github.com/YOUR_USERNAME"><img src="https://img.shields.io/badge/GitHub-Emirhan-100000?style=for-the-badge&logo=github&logoColor=white" alt="Emirhan GitHub"></a>
</p>

<p align="center">
  <b>Taha Yasin</b> | <i>Data Science & Machine Learning</i><br>
  <a href="https://github.com/Taha-Yasin-Erturk"><img src="https://img.shields.io/badge/GitHub-Taha_Yasin-100000?style=for-the-badge&logo=github&logoColor=white" alt="Taha Yasin GitHub"></a>
</p>

<p align="center">
  <b>Hasan</b> | <i>Data Science & Machine Learning</i><br>
  <a href="https://github.com/hasandemir34"><img src="https://img.shields.io/badge/GitHub-Hasan-100000?style=for-the-badge&logo=github&logoColor=white" alt="Hasan GitHub"></a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" alt="Footer" />
</p>
