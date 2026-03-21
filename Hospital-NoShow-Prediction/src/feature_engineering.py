import os
import gc
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data'

def reduce_mem_usage(df):
    """Veri boyutunu %70 küçülten RAM Kalkanı"""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
    return df

def build_features():
    print("🚀 Feature Engineering Başlıyor...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'appointments_train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'appointments_test.csv'))
    patients = pd.read_csv(os.path.join(DATA_DIR, 'patients.csv'))
    clinics = pd.read_csv(os.path.join(DATA_DIR, 'clinics.csv')).drop(columns=['specialty'], errors='ignore')

    train['is_test'] = 0
    test['is_test'] = 1
    df = pd.concat([train, test], axis=0, ignore_index=True)
    
    df['appointment_datetime'] = pd.to_datetime(df['appointment_datetime'])
    df['is_known'] = (df['is_test'] == 0).astype(int)
    df['known_target'] = df['label_noshow'].fillna(0)

    # 1. BAYESIAN HISTORICAL FUSION (Sızıntısız Geçmiş)
    print("📊 Altın Sinyaller Üretiliyor...")
    df = df.sort_values(by=['patient_id', 'appointment_datetime']).reset_index(drop=True)
    df = df.merge(patients, on='patient_id', how='left')
    
    df['cum_known_appts'] = df.groupby('patient_id')['is_known'].shift(1).fillna(0).cumsum()
    df['cum_known_noshows'] = df.groupby('patient_id')['known_target'].shift(1).fillna(0).cumsum()
    
    # Laplace Smoothing ile Bayesian Oranı
    df['bayesian_noshow_rate'] = (df['cum_known_noshows'] + 1) / (df['cum_known_appts'] + 2)

    # 2. LOJİSTİK SİNYALLER
    df = df.merge(clinics, on='clinic_id', how='left')
    df['lead_time_days'] = df['lead_time_hours'] / 24.0
    df['appt_dayofweek'] = df['appointment_datetime'].dt.dayofweek
    
    # 3. GEREKSİZ KOLONLARI SİL
    drop_cols = ['appointment_datetime', 'booking_datetime', 'is_test', 'is_known', 'known_target', 'clinic_id']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype('category')
        
    df = reduce_mem_usage(df)
    
    train_data = df[df['label_noshow'].notnull()].reset_index(drop=True)
    test_data = df[df['label_noshow'].isnull()].drop(columns=['label_noshow']).reset_index(drop=True)
    
    train_data.to_csv(os.path.join(DATA_DIR, 'processed_train.csv'), index=False)
    test_data.to_csv(os.path.join(DATA_DIR, 'processed_test.csv'), index=False)
    print("✅ İşlenmiş veriler kaydedildi!")

if __name__ == "__main__":
    build_features()
