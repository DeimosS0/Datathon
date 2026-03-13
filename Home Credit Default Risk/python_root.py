# Generated from: jupyter_notebook_root.ipynb
# Converted at: 2026-03-13T15:27:08.429Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
train_path=r"data\application_train.csv"
train_df=pd.read_csv(train_path)

test_path=r"data\application_test.csv"
test_df=pd.read_csv(test_path)

# Sayısal ve kategorik sütunları ayır
numeric_cols = train_df.select_dtypes(include='number').columns.tolist()
categorical_cols = train_df.select_dtypes(include='object').columns.tolist()

# Sayısallara ortalama
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())

# Kategoriklere Unknown
train_df[categorical_cols] = train_df[categorical_cols].fillna('Unknown')

import mne
import os
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

y=train_df["TARGET"].to_numpy()
X=train_df.drop("TARGET",axis=1)

numeric_cols = X.select_dtypes(include='number').columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

#Numerical sütunlar için standardizsyon
num_pipeline=Pipeline(steps=[
    
    ("eksik_deger_int",SimpleImputer(strategy="mean")),
    ("standardize",StandardScaler())
])

#Categorical sütunlar için one_hot_encoder
binary_encoding= Pipeline(steps=[

    ("eksik_deger_str",SimpleImputer(strategy="most_frequent")),
("encoding_binary",OneHotEncoder(handle_unknown="ignore",sparse_output=False,drop="if_binary"))
])

#Bu iki pipeline ı birleştiriyoruz
filtrelerim=ColumnTransformer(transformers=[
    
    ("num_pipeline",num_pipeline,numeric_cols),
    ("binary_encoding",binary_encoding,categorical_cols)],
    remainder="drop",
    n_jobs=-1)


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# --- 1. ADIM: MODELLERİ HAZIRLAYALIM ---
# Not: Paralel işlem yapacağımız için, modellerin kendi içindeki 'n_jobs' özelliğini 1 yapıyoruz.
# Böylece işlemci çakışması (bottleneck) olmaz.
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=1),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_jobs=1), 
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_jobs=1)
}

# --- 2. ADIM: TEK BİR EĞİTİM İŞLEMİNİ YAPAN FONKSİYON ---
def model_egit_ve_test_et(name, model, X_train, y_train, X_test, y_test):
    
    current_pipe = Pipeline(steps=[
        ("filtrelerim", filtrelerim),
        ("model", model)
    ])
    # Eğit
    current_pipe.fit(X_train, y_train)
    
    # Tahmin
    y_pred = current_pipe.predict(X_test)
    
    # Sonuçları hesapla
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Trained_Model": current_pipe
    }

# --- 3. ADIM: PARALEL ÇALIŞTIRMA ---
print("Modeller paralel olarak eğitiliyor... Lütfen bekleyin.")

# n_jobs=-1 bilgisayarındaki tüm gücü kullan demektir.

yapilacak_isler_listesi = []

for name, model in classifiers.items():
    
    # delayed fonksiyonu, işlemi dondurulmuş bir paket haline getirir.
    # Dikkat: delayed(fonksiyon_adi)(parametreler) şeklinde yazılır.
    gorev_paketi = delayed(model_egit_ve_test_et)(
        name, 
        model, 
        X_train, 
        y_train, 
        X_test, 
        y_test
    )
    # Bu paketi listeye ekle
    yapilacak_isler_listesi.append(gorev_paketi)

# Şu an 'yapilacak_isler_listesi' içinde 6 tane bekleyen emir var.

""" yukarıdaki for döngüsü aşağıdaki gibi kısaca yazılabilir ama ona IQ yetmedi
sonuclar = Parallel(n_jobs=-1)(
    delayed(model_egit_ve_test_et)(
        name, model, filtrelerim, X_train, y_train, X_test, y_test
    ) for name, model in classifiers.items()
)
"""

motor = Parallel(n_jobs=-1, verbose=10)

# 3. ADIM: LİSTEYİ MOTORA VER VE ÇALIŞTIR
print("Modeller eğitiliyor...")
sonuclar = motor(yapilacak_isler_listesi)

tum_modeller = {}
skor_listesi = []

for sonuc in sonuclar:
    model_adi = sonuc["Model"]
    tum_modeller[model_adi] = sonuc["Trained_Model"]
    
    # Model objesini DataFrame'e koymamak için çıkarıyoruz
    skor_listesi.append({
        k: v for k, v in sonuc.items() if k != "Trained_Model"
    })

# Bitti! Sonuçlar 'sonuclar' değişkeninde.
# --- 4. ADIM: SONUÇLARI GÖSTERME ---
df_sonuc = pd.DataFrame(sonuclar)
# F1 Score'a göre en iyiden en kötüye sırala
df_sonuc = df_sonuc.sort_values(by="F1 Score", ascending=False)

df_sonuc

# Tahmin yap
secilen_model = tum_modeller["Random Forest"]
tahminler = secilen_model.predict_proba(test_df)[:, 1]

submission = pd.DataFrame({
    'SK_ID_CURR': test_df['SK_ID_CURR'],
    'TARGET': tahminler
})
#Tahminleri csv ye çevirme
submission.to_csv('submission.csv', index=False)
print(submission.shape)  # (48744, 2) olmalı