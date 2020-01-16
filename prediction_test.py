# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:34:20 2019

@author: ar1
"""
import os
import pickle
import pandas as pd
import numpy as np

os.chdir("C:/Personal/Kaggle/ASHRAE/python_scripts")
from preprocessing import read_data, parse_timestamp, parallelize_dataframe
from model_training import create_dummies, score, rmsle

## Final Submission file
submission = pd.DataFrame()

def final_model_predict(x, model):
    preds = model.predict(x)
    x['meter_reading_pred'] = preds
    return x

meter_0 = read_data("C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output/test_final_3.csv")
meter_0 = parse_timestamp(meter_0, 'timestamp')
meter_0.columns
meter_0['site_id'] = meter_0['site_id'].astype(int).astype(str)

meter_0 = create_dummies(meter_0, ['site_id', 'primary_use', 'square_feet_profile'])

drop_col_list = ['wind_direction', 'square_feet', 'year_built', 'floor_count', 'month',
                 'day', 'hour', 'year', 'primary_use', 'meter', 'site_id', 'square_feet_profile',
                 'building_id',
                 'timestamp']
meter_0 = meter_0.drop(drop_col_list, axis=1)

meter_0.shape


meter_0_model = pickle.load(open('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output/models/model3.pkl', 'rb'))
X = meter_0.drop('row_id', axis=1)
X.shape

feat = pickle.load(open("C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output/models/model_feat3.pkl", "rb"))
print(len(feat))

diff_col = set(feat) - set(X.columns)
diff_col_df = pd.DataFrame(0, index=np.arange(len(X)), columns=diff_col)

X = pd.concat([X, diff_col_df], axis=1)

#df_score_f = score(X, meter_0_model)
#df_score_f = parallelize_dataframe(X, ['building_id'], score, models=meter_0_model)

df = final_model_predict(X.drop('site_id_11', axis=1), meter_0_model)


meter_0_pred = pd.concat([meter_0['row_id'], df['meter_reading_pred']], axis=1)
meter_0_pred.to_csv("C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output/submission/meter_3_pred.csv", index=False)

submission_0 = meter_0_pred[['row_id', 'meter_reading_pred']]
submission_0.rename(columns={'meter_reading_pred':'meter_reading'}, inplace=True)
submission_0.head()


submission = submission.append(submission_0, ignore_index=True)
submission = submission.sort_values('row_id', ascending=True).reset_index(drop=True)

submission.to_csv("C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output/submission/sub_1_LGBM_base.csv", index=False)