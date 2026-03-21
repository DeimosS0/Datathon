import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

SUB_DIR = '../submissions'

def create_rank_blend():
    print("Rank Ensembling Yapılıyor...")
    
    # Kendi ürettiğiniz en iyi iki CSV
    path_1 = os.path.join(SUB_DIR, 'submission_final.csv')
    path_2 = os.path.join(SUB_DIR, 'submission_4.csv')
    
    if not os.path.exists(path_1) or not os.path.exists(path_2):
        print(f"Hata: {SUB_DIR} klasöründe birleştirilecek dosyalar bulunamadı!")
        return
        
    sub_final = pd.read_csv(path_1)
    sub_4 = pd.read_csv(path_2)

    sub_final = sub_final.sort_values('appointment_id').reset_index(drop=True)
    sub_4 = sub_4.sort_values('appointment_id').reset_index(drop=True)

    # %60 - %40 Sıra Harmanlaması (Rank Blend)
    rank_final = rankdata(sub_final['label_noshow']) / len(sub_final)
    rank_4 = rankdata(sub_4['label_noshow']) / len(sub_4)

    blended_rank = (rank_final * 0.60) + (rank_4 * 0.40)

    # Orijinal olasılık dağılımına geri oturtma
    best_blend = pd.DataFrame({
        'appointment_id': sub_final['appointment_id'],
        'rank_score': blended_rank
    })
    
    best_blend = best_blend.sort_values('rank_score').reset_index(drop=True)
    best_blend['label_noshow'] = np.sort(sub_final['label_noshow'].values)
    best_blend = best_blend.sort_values('appointment_id').reset_index(drop=True)

    final_path = os.path.join(SUB_DIR, 'ULTIMATE_GOLD_BLEND.csv')
    best_blend[['appointment_id', 'label_noshow']].to_csv(final_path, index=False)
    
    print(f"🎯 İŞLEM TAMAM! 0.5149 Getiren Final Dosyası Kaydedildi: {final_path}")

if __name__ == "__main__":
    create_rank_blend()
