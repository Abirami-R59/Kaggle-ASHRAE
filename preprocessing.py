# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import time
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from multiprocessing import Pool
from functools import partial


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/input'):
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

def merge_datasets(train, weather_train, building_metadata):
    """
    Merges the 3 dataframes train, weather_train and building_metadata
    to a single dataframe and removes the duplicate rows if any
    """
    print("Merging weather data with building metadata")
    weather_build_meta = pd.merge(weather_train, building_metadata, on='site_id', how='left')
    print(weather_build_meta.shape)
    
    print("Dropping duplicates, if any")
    weather_build_meta = weather_build_meta.drop_duplicates()
    print(weather_build_meta.shape)
    
    print("Merging with train data")
    final_data = pd.merge(weather_build_meta, train, on = ['building_id', 'timestamp'], how='outer')
    print(final_data.shape)
    
    print("Dropping duplicates, if any")
    final_data = final_data.drop_duplicates()
    print(final_data.shape)
    
    return final_data

def replicate_null_rows(df, meter_grp):
    """
    Replicates the rows which have meter as null, number of meter times and 
    fills the meter and null meter values
    """
    building = df['building_id'].unique()[0]
    print(building)
    
    rep_num = int(meter_grp.loc[meter_grp['building_id'] == building]['nunique'])
    temp_df = pd.DataFrame()
    temp_df = temp_df.append([df] * int(rep_num), ignore_index=True).sort_values('timestamp')
    i=0
    while i < rep_num:
        if i == rep_num-1:
            temp_df['meter'] = temp_df['meter'].fillna(meter_grp.loc[meter_grp['building_id'] == building]['unique'].values[0][i])
        else:
            temp_df['duplicated'] = temp_df.duplicated(['timestamp', 'building_id', 'meter'], keep='first')
            temp_df[['timestamp', 'building_id', 'meter', 'duplicated']].sort_values('timestamp').head()
            temp_df['meter'] = temp_df.apply(
                lambda row: meter_grp.loc[meter_grp['building_id'] == building]['unique'].values[0][i] if row['duplicated'] == True else row['meter'],
                axis=1)
            temp_df.drop('duplicated', axis=1, inplace=True)
        i += 1
        
    return temp_df

def parallelize_dataframe(df, group_list, func, **kwargs):
    """
    Parallelize the dataframe based on groups in a grouped dataframe, 
    runs the specified function on each of the groups in parallel and
    combines it to a dataframe
    """
    df_split = df.groupby(group_list)
    pool = Pool(os.cpu_count())
#    if args is not None:
#        func_x=partial(func, **kwargs)
#    else:
    func_x = partial(func, **kwargs)
    ret_list = pool.map(func_x, [group for name, group in df_split])
    df = pd.DataFrame()
    print("for loop")
    for dat in ret_list:
        df = df.append(dat, ignore_index=True)
    pool.close()
    pool.join()
    return df

def fill_attributes(df, attr_list):
    print(df['building_id'].unique()[0])
    for attr in attr_list:
        print(attr)
        attr_value_list = df[attr].dropna().unique().tolist()
        if len(attr_value_list) == 1:
            df[attr] = df[attr].fillna(attr_value_list[0])
        else:
            print("No unique values to fill the attributes")
    return df

def fill_numerical_data(df):
    print("Forward Fill")
    df[['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = df.groupby(['building_id','meter'])['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed'].fillna(method='ffill')
    
    print("Backward Fill")
    df[['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = df.groupby(['building_id','meter'])['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed'].fillna(method='bfill')
    
    return df

def create_features(df, limit_sqft):
    print(df['building_id'].unique()[0])
    ## Features from timestamp
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    
    df['age'] = df['year'] - df['year_built']
    
    df['square_feet_profile'] = np.where((df['square_feet'] >= limit_sqft['min']) & (df['square_feet'] < limit_sqft['25%']), 'small',
                                         np.where((df['square_feet'] >= limit_sqft['25%']) & (df['square_feet'] < limit_sqft['50%']), 'medium',
                                         np.where((df['square_feet'] >= limit_sqft['50%']) & (df['square_feet'] < limit_sqft['75%']), 'big', 
                                         np.where((df['square_feet'] >= limit_sqft['75%']) & (df['square_feet'] < limit_sqft['max']), 'huge', np.nan))))
    
    return df


def sine_transform(input_df, normalize_var):
    """Transform a input DF to Sine transform"""
    col_names = input_df.columns
    if len(col_names) == 1:
        transformed_array = np.sin(2 * np.pi * input_df[col_names[0]] / normalize_var)
        transformed_df = pd.DataFrame({f'sine_{col_names[0]}': transformed_array})
    else:
        transformed_df = None
    return transformed_df

def cosine_transform(input_df, normalize_var):
    """Transform a input DF to Cosine transform"""
    col_names = input_df.columns
    if len(col_names) == 1:
        transformed_array = np.cos(2 * np.pi * input_df[col_names[0]] / normalize_var)
        transformed_df = pd.DataFrame({f'cosine_{col_names[0]}': transformed_array})
    else:
        transformed_df = None
    return transformed_df

def create_sin_cos_features(df):
    print(df['building_id'].unique()[0])
    df['month_sin'] = sine_transform(df[['month']], 12)
    df['month_cos'] = cosine_transform(df[['month']], 12)

    df['day_sin'] = sine_transform(df[['day']], 31)
    df['day_cos'] = cosine_transform(df[['day']], 31)

    df['hour_sin'] = sine_transform(df[['hour']], 23)
    df['hour_cos'] = cosine_transform(df[['hour']], 23)

    df['wind_dir_sin'] = sine_transform(df[['wind_direction']], 360)
    df['wind_dir_cos'] = cosine_transform(df[['wind_direction']], 360)
    
    return df
    

def main():
    start = time.time()
    #global meter_grp
    ## Reading Training Data
    train = read_data('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/input/train.csv')
    weather_train = read_data('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/input/weather_train.csv')
    building_metadata = read_data('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction/input/building_metadata.csv')
    
    ## Parsing Timestamps
    train = parse_timestamp(train, 'timestamp')
    weather_train = parse_timestamp(weather_train, 'timestamp')
    
    train_final = merge_datasets(train, weather_train, building_metadata)
    print(train_final.head())
    print(train_final.columns)
    
    ## Finding missing values in the merged dataframe
    print(train_final.isnull().sum())
    
    ## Filling Meter and Meter Reading by replicating rows
    print("Getting rows were meter is null")
    meter_null_df = train_final[train_final['meter'].isnull()]
    print(meter_null_df.shape)
    
    print("Getting rows were meter is not null")
    meter_non_null_df = train_final[~train_final['meter'].isnull()]
    print(meter_non_null_df.shape)
    
    print("Getting the number of meters and the meters from the train data")
     
    meter_grp = train.groupby('building_id').meter.agg(['nunique', 'unique']).reset_index()
    print(meter_grp.head())
    
    repl_null_df = pd.DataFrame()
    #repl_null_df = repl_null_df.append(meter_null_df.groupby('building_id').apply(replicate_null_rows))
    repl_null_df = parallelize_dataframe(meter_null_df, ['building_id'], replicate_null_rows, meter_grp=meter_grp)
    print(repl_null_df.shape)
    
    train_final_null_df = meter_non_null_df.append(repl_null_df, ignore_index=True)
    print(train_final_null_df.shape)
    
    
    print("Deleting variables and garbage collection!!")
    del building_metadata, weather_train, meter_non_null_df, meter_null_df, repl_null_df
    print(gc.collect())

    print(train_final_null_df.dtypes)
    train_final_null_df['meter_reading'] = train_final_null_df['meter_reading'].astype(float)
    print(train_final_null_df.dtypes)
    
    train_final_null_df['meter_reading'] = train_final_null_df.groupby(['building_id', 'meter'])['meter_reading'].fillna(method='ffill')
    print(train_final_null_df.isnull().sum())
    
    ## Filling attribute data
    print("Filling attribute data")
    train_final_null_df[['building_id', 'site_id', 'primary_use', 'square_feet', 'year_built']] = parallelize_dataframe(train_final_null_df[['building_id', 'site_id', 'primary_use', 'square_feet', 'year_built']], ['building_id'], fill_attributes, attr_list=['site_id', 'primary_use', 'square_feet', 'year_built'])
    print(train_final_null_df.isnull().sum())
    
    del train, train_final
    print(gc.collect())
    
    ## Filling Numerical data
    print("Filling numerical data")
    train_final_null_df[['building_id', 'meter', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = parallelize_dataframe(train_final_null_df[['building_id', 'meter', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']], ['building_id','meter'], fill_numerical_data)
    print(train_final_null_df.isnull().sum())
#    print("Forward Fill")
#    train_final_null_df[['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
#       'wind_direction', 'wind_speed']] = train_final_null_df.groupby(['building_id','meter'])['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
#       'wind_direction', 'wind_speed'].fillna(method='ffill')
#    print(train_final_null_df.isnull().sum())
#    
#    print("Backward Fill")
#    train_final_null_df[['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
#       'wind_direction', 'wind_speed']] = train_final_null_df.groupby(['building_id','meter'])['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
#       'wind_direction', 'wind_speed'].fillna(method='bfill')
#    print(train_final_null_df.isnull().sum())
    
    
    ## Creating Features
    print("Creating features")
    #global limit_sqft
    limit_sqft = train_final_null_df['square_feet'].describe()
    train_final_null_df = parallelize_dataframe(train_final_null_df, ['building_id'], create_features, limit_sqft=limit_sqft)
    print(train_final_null_df.head())
    
    print("Creating cyclical features")
    train_final_null_df = parallelize_dataframe(train_final_null_df, ['building_id'], create_sin_cos_features)
    print(train_final_null_df.head())
    
    
    ## Saving the dataframe 
    train_final_null_df.to_csv('C:/Personal/Kaggle/ASHRAE/ashrae-energy-prediction\output/train_feat_final.csv', index=False)
    
    end = time.time()
    print(end-start)
    
    return("Successful!!")


    
if __name__ == "__main__":
    msg = main()
    print(msg)