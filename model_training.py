# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import gc
import math
import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from scipy.stats.mstats import mquantiles

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
## Function to read the data sets
def read_data(path):
    """
    Reads the data from the path and 
    prints the shape and sample rows of the dataframe
    """
    df = pd.read_csv(path)
    print(df.shape)
    print(df.head())
    return df

def parse_timestamp(df, col):
    """
    Converts the timestamp column from object to datetime type
    """
    df[col] = pd.to_datetime(df[col])
    return df

def data_split(df, date=None):
    if date == None:
        df['dataset'] = np.where(df['timestamp'] <= '2016-03-31 00:00:00', 'train', 'test')
    else:
        df['dataset'] = np.where(df['timestamp'] <= date, 'train', 'test')
    df[['site_id', 'building_id', 'meter', 'year']] = df[['site_id', 'building_id', 'meter', 'year']].astype(int).astype(str)
    return df

def create_dummies(df, col_list):
    df[col_list] = df[col_list].astype(str)
    df_dummy = pd.get_dummies(df[col_list])
    df = pd.concat([df, df_dummy], axis=1)
    return df

def train_models(x, y, params, folds=5):
    models = []

    tssp = TimeSeriesSplit(n_splits=folds, max_train_size=None)

    for train_idx, test_idx in tssp.split(x):
        x_tr, x_te = x.iloc[train_idx], x.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = LGBMRegressor(colsample_bytree = params["colsample_bytree"],
                              learning_rate = params["learning_rate"],
                              max_depth = params["max_depth"],
                              min_child_samples = params["min_child_samples"],
                              min_sum_hessian_in_leaf = params["min_sum_hessian_in_leaf"],
                              n_estimators = params["n_estimators"],
                              num_leaves = params["num_leaves"],
                              reg_alpha = params["reg_alpha"],
                              reg_lambda = params["reg_lambda"],
                              subsample = params["subsample"],
                              tree_learner = params["tree_learner"],
                              boosting_type = params["boosting_type"],
                              objective = params["objective"],
                              )
        model.fit(x_tr, y_tr)
        preds = model.predict(x_te)
        print(rmsle(y_true=y_te, y_pred=preds))
        models.append(model)
        del x_tr, x_te, y_tr, y_te, model
        print(gc.collect())
        
    return models

def final_model_train(x, y, params):
    model = LGBMRegressor(colsample_bytree = params["colsample_bytree"],
                      learning_rate = params["learning_rate"],
                      max_depth = params["max_depth"],
                      min_child_samples = params["min_child_samples"],
                      min_sum_hessian_in_leaf = params["min_sum_hessian_in_leaf"],
                      n_estimators = params["n_estimators"],
                      num_leaves = params["num_leaves"],
                      reg_alpha = params["reg_alpha"],
                      reg_lambda = params["reg_lambda"],
                      subsample = params["subsample"],
                      tree_learner = params["tree_learner"],
                      boosting_type = params["boosting_type"],
                      objective = params["objective"],
                      )
    model.fit(x, y)
    return model

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    terms_to_sum = []
    for i, true in enumerate(y_true):
        try:
            t = (math.log(y_pred[i] + 1) - math.log(true + 1)) ** 2.0
        except:
            print(y_pred[i])
            print(true)
        terms_to_sum.append(t)
    #terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

def score(x, models):
    preds_matrix = np.zeros([x.shape[0], len(models)])
    for i, model in enumerate(models):
        preds = model.predict(x)
        preds_matrix[:, i] = preds
    pred_df = pd.DataFrame(np.array(mquantiles(preds_matrix, prob=[0.05, 0.5, 0.95], axis=1)), columns = ['preds_min','preds_median','preds_max'])
    pred_df['preds_mean'] =  np.mean(preds_matrix, axis=1)
    return pred_df

def main():
    prep_df = read_data('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output/train_feat_final.csv')
    prep_df = parse_timestamp(prep_df, 'timestamp')
    
    ## Removing columns
    all_cols = prep_df.columns
    drop_col_list = ['wind_direction', 'square_feet', 'year_built', 'floor_count', 'month', 'day', 'hour']
    
    prep_df = prep_df[list((set(all_cols) - set(drop_col_list)))]
    print(prep_df.shape)
    
    ## Splitting data by date
    prep_df = data_split(prep_df, '2016-10-31 00:00:00')
    print(prep_df.groupby(['dataset'])['timestamp'].agg({'min','max'}))
    
    for meter in prep_df.meter.unique():
        print ("Meter " + str(meter))
        train_df = prep_df[prep_df['meter'] == meter].reset_index(drop=True)
        print(train_df.shape)
        print(train_df['meter'].unique())
        
        train_df = create_dummies(train_df, ['site_id', 'primary_use', 'square_feet_profile'])
        print(train_df.shape)
        
        train_df = train_df.drop(['year', 'primary_use', 'meter', 'site_id', 'square_feet_profile', 'building_id', 'timestamp'], axis=1)
        
        train_df_train = train_df[train_df['dataset'] == 'train']
        train_df_test = train_df[train_df['dataset'] == 'test']
        
#        train_df_train = train_df_train.drop(['year', 'primary_use', 'meter', 'site_id', 'square_feet_profile', 'dataset', 'building_id', 'timestamp'], axis=1)
#        train_df_test = train_df_test.drop(['year', 'primary_use', 'meter', 'site_id', 'square_feet_profile', 'dataset', 'building_id', 'timestamp'], axis=1)

        print("Train Test split")
        X_train = train_df_train.drop(['meter_reading', 'dataset'], axis=1)
        y_train = train_df_train['meter_reading']

        X_test = train_df_test.drop(['meter_reading', 'dataset'], axis=1)
        y_test = train_df_test['meter_reading']
        
        del train_df_train
        print(gc.collect())
        
        print("Setting parameters")
        params = {"colsample_bytree": 0.8,
          "learning_rate": 0.2,
          "max_depth": 15,
          "min_child_samples": 300,
          "min_sum_hessian_in_leaf": 0.129,
          "n_estimators": 2000,
          "num_leaves": 40,
          "reg_alpha": 0.5,
          "reg_lambda": 10.5,
          "subsample": 0.75,
          "tree_learner": "data",
          "boosting_type": "gbdt",
          "objective": "poisson"}
        
        print("Model Training")
        models = train_models(X_train, y_train, params)
               
        print("Model Scoring on test data")
        df_score_f=score(X_test, models)
        print(df_score_f.head())
        
        print("RMSLE")
        rmsle_train = rmsle(y_test, df_score_f['preds_median'])
        print(rmsle_train)
        
        print("Saving Test Predictions")
        pred_df = pd.concat([train_df_test, df_score_f], axis=1)
        pred_df.to_csv('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output/predictions/pred'+str(meter)+'.csv', index=False)
        
        del X_train, X_test, y_train, y_test, train_df_test
        print(gc.collect())
        
        print("Final Model")
        X_train = train_df.drop(['meter_reading', 'dataset'], axis=1)
        y_train = train_df['meter_reading']
        
        final_model = final_model_train(X_train, y_train, params)
        
        print("Features")
        feat = X_train.columns.tolist()
        print(feat)

        print("Saving final model file")
        filename = 'C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output/models/model'+str(meter)+'.pkl'
        pickle.dump(final_model, open(filename, 'wb'))
        
        print("Saving final features")
        filename = 'C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/output/models/model_feat'+str(meter)+'.pkl'
        pickle.dump(feat, open(filename, 'wb'))
        
    return "Successful!!"
        

if __name__ == "__main__":
    msg = main()
    print(msg)