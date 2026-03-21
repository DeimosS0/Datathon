import os
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

DATA_DIR = '../data'
SUB_DIR = '../submissions'
TARGET = 'label_noshow'

def train_models():
    print("AutoGluon OOM-Safe Eğitimi Başlıyor...")
    
    train_df = TabularDataset(os.path.join(DATA_DIR, 'processed_train.csv'))
    test_df = TabularDataset(os.path.join(DATA_DIR, 'processed_test.csv'))
    
    # Sadece modern Boosting algoritmalarına izin veriyoruz (RF ve KNN yasak!)
    predictor = TabularPredictor(
        label=TARGET,
        eval_metric='average_precision',
        groups='patient_id',
        path='AutogluonModels/Champion_Model'
    ).fit(
        train_data=train_df.drop(columns=['appointment_id']),
        time_limit=3600 * 3,
        num_bag_folds=3,
        hyperparameters={'GBM': {}, 'CAT': {}, 'XGB': {}},
        excluded_model_types=['NN_TORCH', 'FASTAI', 'RF', 'XT', 'KNN'],
        ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'},
        num_gpus=1,
        verbosity=2
    )
    
    print("✅ Eğitim bitti, tahminler alınıyor...")
    preds = predictor.predict_proba(test_df.drop(columns=['appointment_id']))[predictor.positive_class]
    
    os.makedirs(SUB_DIR, exist_ok=True)
    sub = pd.DataFrame({
        'appointment_id': test_df['appointment_id'].astype(int),
        TARGET: preds.values
    })
    
    out_path = os.path.join(SUB_DIR, 'autogluon_champion.csv')
    sub.to_csv(out_path, index=False)
    print(f" Model tahminleri kaydedildi: {out_path}")

if __name__ == "__main__":
    train_models()
